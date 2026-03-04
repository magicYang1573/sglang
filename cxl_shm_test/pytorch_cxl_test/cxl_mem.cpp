// Minimal helpers to map a CXL-backed DAX device and copy data between
// DRAM/VRAM and the mapped region. Functions mirror the patterns used in
// cxl_shm_base_test (mmap + CLFLUSH + cudaMemcpy).

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <unistd.h>

#include "cxl_mem.hpp"

namespace {

constexpr std::size_t kCacheLine = 64;

// ---------------------------------------------------------------------------
// Write path: streaming NT stores to CXL (bypasses CPU cache on write).
// Uses AVX-512 (64 B/iter) -> AVX2 (32 B/iter) -> scalar (8 B/iter) fallback.
// All offsets in this module are 256-byte aligned, satisfying the 32/64-byte
// alignment requirement of _mm256/512_stream_si256/512.
// ---------------------------------------------------------------------------
inline void nt_store_copy(void *dst, const void *src, std::size_t len) {
	auto *d8 = static_cast<uint8_t *>(dst);
	auto *s8 = static_cast<const uint8_t *>(src);

#ifdef __AVX512F__
	const std::size_t chunks512 = len / 64;
	for (std::size_t i = 0; i < chunks512; ++i) {
		__m512i vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(s8) + i);
		_mm512_stream_si512(reinterpret_cast<__m512i *>(d8) + i, vec);
	}
	const std::size_t used512 = chunks512 * 64;
	d8 += used512;
	s8 += used512;
	len -= used512;
#endif

#ifdef __AVX2__
	const std::size_t chunks256 = len / 32;
	for (std::size_t i = 0; i < chunks256; ++i) {
		__m256i vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s8) + i);
		_mm256_stream_si256(reinterpret_cast<__m256i *>(d8) + i, vec);
	}
	const std::size_t used256 = chunks256 * 32;
	d8 += used256;
	s8 += used256;
	len -= used256;
#endif

	// Scalar fallback for remainder (< 32 bytes)
	const std::size_t qwords = len / sizeof(std::uint64_t);
	for (std::size_t i = 0; i < qwords; ++i) {
		_mm_stream_si64(reinterpret_cast<long long *>(d8) + i,
		                reinterpret_cast<const long long *>(s8)[i]);
	}
	const std::size_t used_q = qwords * 8;
	d8 += used_q;
	s8 += used_q;
	len -= used_q;
	if (len > 0) std::memcpy(d8, s8, len);

	_mm_sfence();
}

// ---------------------------------------------------------------------------
// Read path: invalidate CPU cache lines so subsequent loads hit CXL memory.
// NTA software prefetch hides some of the CXL latency while issuing clflushopt.
// ---------------------------------------------------------------------------
inline void clflush_range(const void *addr, std::size_t len) {
	auto p = reinterpret_cast<std::uintptr_t>(addr);
	const std::uintptr_t end = p + len;
	// Issue NTA prefetch a few cache lines ahead to warm up the CXL prefetch
	// buffer while the CPU processes clflushopt for earlier lines.
	constexpr std::uintptr_t kPrefetchAhead = 8 * kCacheLine;
	for (; p < end; p += kCacheLine) {
		if (p + kPrefetchAhead < end) {
			_mm_prefetch(reinterpret_cast<const char *>(p + kPrefetchAhead), _MM_HINT_NTA);
		}
		_mm_clflushopt(reinterpret_cast<void *>(p));
	}
	_mm_sfence();
}

inline bool cuda_check(cudaError_t err, const char *what) {
	if (err == cudaSuccess) {
		return true;
	}
	std::cerr << what << " failed: " << cudaGetErrorString(err) << "\n";
	return false;
}

// ---------------------------------------------------------------------------
// Pinned (page-locked) staging buffer for fast H2D DMA transfers.
// Lazily allocated; resized with free+realloc if a larger size is needed.
// cudaFreeHost must be called before CUDA context destruction; in practice
// the process exits after the inference server, so this is acceptable.
// ---------------------------------------------------------------------------
struct PinnedStagingBuf {
	void *ptr = nullptr;
	std::size_t capacity = 0;

	bool ensure(std::size_t needed) {
		if (needed <= capacity) return true;
		if (ptr) {
			cudaFreeHost(ptr);
			ptr = nullptr;
			capacity = 0;
		}
		if (cudaHostAlloc(&ptr, needed, cudaHostAllocPortable) != cudaSuccess) {
			std::cerr << "cudaHostAlloc staging buffer (" << needed << " B) failed\n";
			return false;
		}
		capacity = needed;
		return true;
	}

	~PinnedStagingBuf() {
		if (ptr) cudaFreeHost(ptr);
	}
};

} // namespace

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
struct CxlRegion {
	void *base = nullptr;
	std::size_t length = 0;
	bool cuda_registered = false;
};

static CxlRegion g_cxl;
static PinnedStagingBuf g_staging;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool cxl_init(const std::string &dev_path,
              std::size_t map_bytes,
              off_t offset,
              bool register_cuda,
              int gpu_id) {
	std::cout << "cxl_init " << g_cxl.base << std::endl;
	if (g_cxl.base != nullptr) {
		std::cout << "cxl_init: region already mapped; skipping reinit\n";
		return true;
	}

	const int fd = ::open(dev_path.c_str(), O_RDWR);
	if (fd < 0) {
		std::cerr << "open " << dev_path << " failed: " << std::strerror(errno) << "\n";
		return false;
	}

	void *addr = ::mmap(nullptr, map_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
	::close(fd);
	if (addr == MAP_FAILED) {
		std::cerr << "mmap failed: " << std::strerror(errno) << " (offset=" << offset << ")\n";
		return false;
	}

	// Advise the kernel to use transparent huge pages to reduce TLB pressure
	// when accessing large CXL windows.
	::madvise(addr, map_bytes, MADV_HUGEPAGE);

	g_cxl.base = addr;
	g_cxl.length = map_bytes;
	g_cxl.cuda_registered = false;

	if (register_cuda) {
		if (!cuda_check(cudaSetDevice(gpu_id), "cudaSetDevice")) {
			::munmap(addr, map_bytes);
			g_cxl = {};
			return false;
		}
		if (cuda_check(cudaHostRegister(addr, map_bytes, cudaHostRegisterPortable),
		               "cudaHostRegister")) {
			g_cxl.cuda_registered = true;
		} else {
			::munmap(addr, map_bytes);
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

bool vram2cxl(const void *device_src, std::size_t bytes, std::size_t offset) {
	if (g_cxl.base == nullptr || offset + bytes > g_cxl.length) {
		std::cerr << "vram2cxl: invalid region or range\n";
		return false;
	}
	// Sync the current CUDA stream so the tensor data is ready, then DMA to CXL.
	// When cudaHostRegister is active, cudaMemcpy uses GPU DMA directly to CXL
	// memory (bypassing the CPU cache on the write side); clflush is NOT needed
	// after a D2H DMA write.
	at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
	torch_stream.synchronize();

	void *dst = static_cast<std::uint8_t *>(g_cxl.base) + offset;
	if (!cuda_check(cudaMemcpy(dst, device_src, bytes, cudaMemcpyDeviceToHost),
	                "cudaMemcpy D2H")) {
		return false;
	}
	// GPU DMA bypasses CPU cache; sfence ensures the write is globally visible
	// before we post the barrier token.
	_mm_sfence();
	return true;
}

bool vram2cxl_async(const void *device_src, std::size_t bytes, std::size_t offset) {
	if (g_cxl.base == nullptr || offset + bytes > g_cxl.length) return false;

	at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
	void *dst = static_cast<std::uint8_t *>(g_cxl.base) + offset;

	if (!cuda_check(cudaMemcpyAsync(dst, device_src, bytes, cudaMemcpyDeviceToHost, stream),
	                "D2H async")) {
		return false;
	}
	stream.synchronize();
	_mm_sfence();
	return true;
}

bool cxl2vram(void *device_dst, std::size_t bytes, std::size_t offset) {
	if (g_cxl.base == nullptr || offset + bytes > g_cxl.length) {
		std::cerr << "cxl2vram: invalid region or range\n";
		return false;
	}
	const void *src = static_cast<const std::uint8_t *>(g_cxl.base) + offset;
	clflush_range(src, bytes);
	if (!cuda_check(cudaMemcpy(device_dst, src, bytes, cudaMemcpyHostToDevice),
	                "cudaMemcpy H2D")) {
		return false;
	}
	return true;
}

// ---------------------------------------------------------------------------
// Parallel shard gather: read num_shards non-contiguous CXL regions into a
// contiguous GPU buffer using one worker thread per shard.
//
// Layout on GPU:  [shard_0 | shard_1 | ... | shard_{N-1}]   (contiguous)
// Layout on CXL:  each shard at cxl_offsets[i], size shard_bytes
//
// Algorithm:
//   1. Spawn one thread per shard that does clflush + memcpy to pinned staging.
//   2. After all threads join, issue a single H2D cudaMemcpy from staging to GPU.
//
// This overlaps the CXL→DRAM copy across multiple CPU cores, then uses a
// single large DMA for the DRAM→VRAM step (lower PCIe setup overhead).
// ---------------------------------------------------------------------------
bool cxl2vram_shards(void *device_dst,
                     std::size_t num_shards,
                     const std::size_t *cxl_offsets,
                     std::size_t shard_bytes) {
	if (g_cxl.base == nullptr || num_shards == 0) return false;

	const std::size_t total_bytes = num_shards * shard_bytes;

	if (!g_staging.ensure(total_bytes)) {
		// Fall back to serial direct H2D copies if staging allocation fails.
		for (std::size_t i = 0; i < num_shards; ++i) {
			const void *src =
			    static_cast<const uint8_t *>(g_cxl.base) + cxl_offsets[i];
			void *dst = static_cast<uint8_t *>(device_dst) + i * shard_bytes;
			clflush_range(src, shard_bytes);
			if (!cuda_check(cudaMemcpy(dst, src, shard_bytes, cudaMemcpyHostToDevice),
			                "cxl2vram_shards fallback H2D")) {
				return false;
			}
		}
		return true;
	}

	// Phase 1: parallel clflush + memcpy from CXL to pinned staging buffer.
	// Each thread operates on a disjoint region of the staging buffer.
	std::vector<std::thread> workers;
	workers.reserve(num_shards);
	for (std::size_t i = 0; i < num_shards; ++i) {
		workers.emplace_back([i, cxl_offsets, shard_bytes]() {
			const void *src =
			    static_cast<const uint8_t *>(g_cxl.base) + cxl_offsets[i];
			void *dst =
			    static_cast<uint8_t *>(g_staging.ptr) + i * shard_bytes;
			clflush_range(src, shard_bytes);
			std::memcpy(dst, src, shard_bytes);
		});
	}
	for (auto &t : workers) t.join();

	// Phase 2: single H2D DMA from pinned staging to GPU.
	return cuda_check(
	    cudaMemcpy(device_dst, g_staging.ptr, total_bytes, cudaMemcpyHostToDevice),
	    "cxl2vram_shards H2D");
}

bool cxl_close() {
	if (g_cxl.base == nullptr) {
		return true;
	}

	if (g_cxl.cuda_registered) {
		if (!cuda_check(cudaHostUnregister(g_cxl.base), "cudaHostUnregister")) {
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

// ---------------------------------------------------------------------------
// Barrier implementation.
//
// Each rank posts its token to its dedicated cache line (rank * kCacheLine)
// inside the control block, then spins until all ranks have posted a token
// >= the current value.
//
// Optimizations vs. original:
//  1. Fixed double-offset bug: base_ptr already includes control_offset; the
//     clflush must start at base_ptr, not base_ptr + control_offset.
//  2. Per-rank confirmed[] bitset: once a rank's token is observed we stop
//     re-flushing its cache line, reducing CXL traffic each iteration.
//  3. Removed std::vector allocation inside the hot polling loop.
//  4. Proper sfence before posting token (data visible) and lfence after all
//     tokens confirmed (subsequent reads see peer data).
// ---------------------------------------------------------------------------
void cxl_barrier_tp(int32_t token, int64_t control_offset, int rank, int num_ranks) {
	if (!g_cxl.base) {
		throw std::runtime_error("CXL not initialized. Call cxl_init first.");
	}

	uint8_t *base_ptr = static_cast<uint8_t *>(g_cxl.base) + control_offset;
	int32_t *my_token_ptr = reinterpret_cast<int32_t *>(base_ptr + rank * kCacheLine);

	// Ensure all data writes complete before posting the barrier token.
	_mm_sfence();
	nt_store_copy(static_cast<void *>(my_token_ptr), &token, sizeof(int32_t));

	// Track which peers we have already confirmed to avoid redundant flushes.
	// Stack array; supports up to 128 ranks without heap allocation.
	bool confirmed[128] = {};
	confirmed[rank] = true;
	int remaining = num_ranks - 1;

	while (remaining > 0) {
		// Invalidate cache lines only for ranks not yet confirmed.
		for (int i = 0; i < num_ranks; ++i) {
			if (!confirmed[i]) {
				_mm_clflushopt(base_ptr + i * kCacheLine);
			}
		}
		// sfence: ensure clflushopt instructions are ordered before the loads.
		_mm_sfence();

		for (int i = 0; i < num_ranks; ++i) {
			if (confirmed[i]) continue;
			const volatile int32_t *ptr =
			    reinterpret_cast<const int32_t *>(base_ptr + i * kCacheLine);
			if (*ptr >= token) {
				confirmed[i] = true;
				--remaining;
			}
		}
		if (remaining > 0) _mm_pause();
	}

	// lfence: ensure subsequent data reads see the values written by all peers.
	_mm_lfence();
}
