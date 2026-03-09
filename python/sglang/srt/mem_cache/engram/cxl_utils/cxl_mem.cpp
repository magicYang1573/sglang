// Minimal helpers to map a CXL-backed DAX device and copy data between
// DRAM/VRAM and the mapped region. Functions mirror the patterns used in
// cxl_shm_base_test (mmap + CLFLUSH + cudaMemcpy).

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <immintrin.h>
#include <omp.h>
#include <sys/mman.h>
#include <unistd.h>

#include "cxl_mem.hpp"

constexpr std::size_t kCacheLine = 64;

struct CxlRegion {
  void *base = nullptr;
  std::size_t length = 0;
  bool cuda_registered = false;
  void *device_ptr = nullptr;
};

namespace {

// Streamed store copy to avoid polluting CPU caches when writing to CXL.
inline void nt_store_copy(void *dst, const void *src, std::size_t len) {
  auto *d8 = static_cast<std::uint8_t *>(dst);
  auto *s8 = static_cast<const std::uint8_t *>(src);

  const std::size_t qwords = len / sizeof(std::uint64_t);
  auto *d64 = reinterpret_cast<std::uint64_t *>(d8);
  auto *s64 = reinterpret_cast<const std::uint64_t *>(s8);
  for (std::size_t i = 0; i < qwords; ++i) {
    _mm_stream_si64(reinterpret_cast<long long *>(d64 + i),
                    static_cast<long long>(s64[i]));
  }

  const std::size_t consumed = qwords * sizeof(std::uint64_t);
  d8 += consumed;
  s8 += consumed;
  std::size_t rem = len - consumed;

  if (rem >= sizeof(std::uint32_t)) {
    _mm_stream_si32(reinterpret_cast<int *>(d8),
                    *reinterpret_cast<const int *>(s8));
    d8 += sizeof(std::uint32_t);
    s8 += sizeof(std::uint32_t);
    rem -= sizeof(std::uint32_t);
  }

  if (rem > 0) {
    std::memcpy(d8, s8, rem);
  }

  // _mm_sfence();
  _mm_sfence();
}

} // namespace

void clflush_range(const void *addr, std::size_t len) {
  auto p = reinterpret_cast<std::uintptr_t>(addr);
  const std::uintptr_t end = p + len;
  for (; p < end; p += kCacheLine) {
    _mm_clflushopt(reinterpret_cast<void *>(p));
  }
  _mm_sfence();
}

CxlRegion g_cxl;

bool cxl_register_cuda_region(void *addr, std::size_t length, int gpu_id,
                              CxlRegion *region);
bool cxl_unregister_cuda_region(CxlRegion *region);

// Map the CXL DAX device into a static region; optionally register with CUDA
// for faster copies.
bool cxl_init(const std::string &dev_path, std::size_t map_bytes, off_t offset,
              bool register_cuda, int gpu_id) {
  const int fd = ::open(dev_path.c_str(), O_RDWR);

  if (fd < 0) {
    std::cerr << "open " << dev_path << " failed: " << std::strerror(errno)
              << "\n";
    return false;
  }

  size_t page_size = 4096;
  size_t aligned_size = (map_bytes + page_size - 1) & ~(page_size - 1);
  void *addr = ::mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                      fd, offset);
  ::close(fd);
  if (addr == MAP_FAILED) {
    std::cerr << "mmap failed: " << std::strerror(errno)
              << " (offset=" << offset << ")\n";
    return false;
  }

  g_cxl.base = addr;
  g_cxl.length = aligned_size;
  g_cxl.cuda_registered = false;
  g_cxl.device_ptr = nullptr;

  if (register_cuda) {
    if (!cxl_register_cuda_region(addr, aligned_size, gpu_id, &g_cxl)) {
      ::munmap(addr, aligned_size);
      g_cxl = {};
      return false;
    }
  }

  return true;
}

bool dram2cxl(const void *dram_src, std::size_t bytes, std::size_t offset) {
  if (g_cxl.base == nullptr || offset + bytes > g_cxl.length) {
    std::cerr << "dram2cxl: invalid region or range\n";
    return false;
  }
  void *dst = static_cast<std::uint8_t *>(g_cxl.base) + offset;
  nt_store_copy(dst, dram_src, bytes);
  return true;
}

bool cxl2dram(void *dram_dst, std::size_t bytes, std::size_t offset) {
  if (g_cxl.base == nullptr || offset + bytes > g_cxl.length) {
    std::cerr << "cxl2dram: invalid region or range\n";
    return false;
  }
  const void *src = static_cast<const std::uint8_t *>(g_cxl.base) + offset;
  clflush_range(src, bytes);
  std::memcpy(dram_dst, src, bytes);
  return true;
}

bool cxl2dram_noflush(void *dram_dst, std::size_t bytes, std::size_t offset) {
  if (g_cxl.base == nullptr || offset + bytes > g_cxl.length) {
    std::cerr << "cxl2dram_noflush: invalid region or range\n";
    return false;
  }
  const void *src = static_cast<const std::uint8_t *>(g_cxl.base) + offset;
  std::memcpy(dram_dst, src, bytes);
  return true;
}

bool cxl2dram_noflush_parallel(void **dram_dst_list,
                               const std::size_t *bytes_list,
                               const std::size_t *offset_list,
                               std::size_t count, int num_threads) {

#pragma omp parallel for schedule(static) num_threads(64)
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(count); ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const void *src =
        static_cast<const std::uint8_t *>(g_cxl.base) + offset_list[idx];
    std::memcpy(dram_dst_list[idx], src, bytes_list[idx]);
  }
  return true;
}

bool cxl2dram_noflush_parallel_contiguous_dst(void *dram_dst,
                                              std::size_t bytes_per_segment,
                                              const std::size_t *offset_list,
                                              std::size_t count) {
  int num_threads = count < 72 ? count : 72;
#pragma omp parallel for schedule(static) num_threads(num_threads)
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(count); ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const void *src =
        static_cast<const std::uint8_t *>(g_cxl.base) + offset_list[idx];
    void *dst = static_cast<std::uint8_t *>(dram_dst) + idx * bytes_per_segment;
    std::memcpy(dst, src, bytes_per_segment);
  }
  return true;
}

bool cxl_close() {
  if (g_cxl.base == nullptr) {
    return true;
  }

  if (g_cxl.cuda_registered) {
    if (!cxl_unregister_cuda_region(&g_cxl)) {
      return false;
    }
  }

  if (::munmap(g_cxl.base, g_cxl.length) != 0) {
    std::cerr << "munmap failed: " << std::strerror(errno) << "\n";
    return false;
  }

  g_cxl = {};
  return true;
}

void cxl_barrier_tp(int32_t token, int64_t control_offset, int rank,
                    int num_ranks) {
  if (!g_cxl.base) {
    throw std::runtime_error("CXL not initialized. Call cxl_init first.");
  }

  uint8_t *base_ptr = static_cast<uint8_t *>(g_cxl.base) + control_offset;

  int32_t *my_token_ptr =
      reinterpret_cast<int32_t *>(base_ptr + rank * kCacheLine);

  // sfence is necessary, first store all the data, then set the token
  _mm_sfence();

  nt_store_copy((void *)my_token_ptr, (void *)&token, sizeof(int32_t));

  while (true) {
    bool all_ready = true;
    std::vector<int32_t> tokens;
    clflush_range((void *)(base_ptr + control_offset), num_ranks * kCacheLine);
    for (int i = 0; i < num_ranks; i++) {

      volatile int32_t *other_token_ptr =
          reinterpret_cast<int32_t *>(base_ptr + i * kCacheLine);

      int32_t val = *other_token_ptr;
      tokens.push_back(val);

      if (val < token) {
        all_ready = false;
        break;
      }
    }

    if (all_ready) {

      // lfence is necessary here, first all the token is ready, then read the
      // following data
      _mm_lfence();
      break;
    }
    _mm_pause();
  }
}
