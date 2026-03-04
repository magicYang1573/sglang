// Minimal helpers to map a CXL-backed DAX device and copy data between
// DRAM/VRAM and the mapped region.

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <unistd.h>

#include "cxl_mem.hpp"

namespace {

constexpr std::size_t kCacheLine = 64;

// Invalidate CPU cache lines covering [addr, addr+len) and issue sfence.
// Required before any read from non-cache-coherent CXL memory to ensure
// stale cache lines are evicted and subsequent loads see the remote write.
inline void clflush_range(const void *addr, std::size_t len) {
	auto p = reinterpret_cast<std::uintptr_t>(addr);
	const std::uintptr_t end = p + len;
	for (; p < end; p += kCacheLine) {
		_mm_clflushopt(reinterpret_cast<void *>(p));
	}
	_mm_sfence();
}

// Non-temporal (streaming) store copy: write to CXL without polluting the
// CPU cache.  sfence at the end ensures all stores are visible before the
// barrier token is published.
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
	_mm_sfence();
}

// SSE streaming load copy: bypass CPU cache when reading from CXL.
// Handles unaligned source pointers.
inline void sse_load_copy(void *dst, const void *src, std::size_t len) {
	auto *d8 = static_cast<uint8_t *>(dst);
	auto *s8 = static_cast<const uint8_t *>(src);

	auto s_addr = reinterpret_cast<std::uintptr_t>(s8);
	std::size_t align_offset = (16 - (s_addr & 15)) & 15;
	if (align_offset > len) align_offset = len;
	if (align_offset > 0) {
		std::memcpy(d8, s8, align_offset);
		d8 += align_offset;
		s8 += align_offset;
		len -= align_offset;
	}

	std::size_t chunks = len / sizeof(__m128i);
	auto *d128 = reinterpret_cast<__m128i *>(d8);
	auto *s128 = reinterpret_cast<const __m128i *>(s8);
	for (std::size_t i = 0; i < chunks; ++i) {
		_mm_storeu_si128(d128 + i,
		                 _mm_stream_load_si128(const_cast<__m128i *>(s128 + i)));
	}
	std::size_t consumed = chunks * sizeof(__m128i);
	if (consumed < len) {
		std::memcpy(d8 + consumed, s8 + consumed, len - consumed);
	}
}

inline bool cuda_check(cudaError_t err, const char *what) {
	if (err == cudaSuccess) return true;
	std::cerr << what << " failed: " << cudaGetErrorString(err) << "\n";
	return false;
}

} // namespace

// ---------------------------------------------------------------------------

struct CxlRegion {
	void *base = nullptr;
	std::size_t length = 0;
	bool cuda_registered = false;
};

static CxlRegion g_cxl;

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
		std::cerr << "mmap failed: " << std::strerror(errno)
		          << " (offset=" << offset << ")\n";
		return false;
	}

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
	nt_store_copy(static_cast<std::uint8_t *>(g_cxl.base) + offset, dram_src, bytes);
	return true;
}

bool cxl2dram(void *dram_dst, std::size_t bytes, std::size_t offset) {
	if (g_cxl.base == nullptr || offset + bytes > g_cxl.length) {
		std::cerr << "cxl2dram: invalid region or range\n";
		return false;
	}
	const void *src = static_cast<const std::uint8_t *>(g_cxl.base) + offset;
	clflush_range(src, bytes);
	sse_load_copy(dram_dst, src, bytes);
	return true;
}

bool vram2cxl(const void *device_src, std::size_t bytes, std::size_t offset) {
	if (g_cxl.base == nullptr || offset + bytes > g_cxl.length) {
		std::cerr << "vram2cxl: invalid region or range\n";
		return false;
	}
	at::cuda::getCurrentCUDAStream().synchronize();
	void *dst = static_cast<std::uint8_t *>(g_cxl.base) + offset;
	if (!cuda_check(cudaMemcpy(dst, device_src, bytes, cudaMemcpyDeviceToHost),
	                "cudaMemcpy D2H"))
		return false;
	clflush_range(dst, bytes);
	return true;
}

bool vram2cxl_async(const void *device_src, std::size_t bytes, std::size_t offset) {
	if (g_cxl.base == nullptr || offset + bytes > g_cxl.length) return false;
	at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
	void *dst = static_cast<std::uint8_t *>(g_cxl.base) + offset;
	if (!cuda_check(cudaMemcpyAsync(dst, device_src, bytes,
	                                cudaMemcpyDeviceToHost, stream), "D2H async"))
		return false;
	stream.synchronize();
	clflush_range(dst, bytes);
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
	                "cudaMemcpy H2D"))
		return false;
	return true;
}

bool cxl_close() {
	if (g_cxl.base == nullptr) return true;
	if (g_cxl.cuda_registered) {
		if (!cuda_check(cudaHostUnregister(g_cxl.base), "cudaHostUnregister"))
			return false;
	}
	if (::munmap(g_cxl.base, g_cxl.length) != 0) {
		std::cerr << "munmap failed: " << std::strerror(errno) << "\n";
		return false;
	}
	g_cxl = {};
	return true;
}

void cxl_barrier_tp(int32_t token, int64_t control_offset, int rank, int num_ranks) {
	if (!g_cxl.base)
		throw std::runtime_error("CXL not initialized. Call cxl_init first.");

	uint8_t *base_ptr = static_cast<uint8_t *>(g_cxl.base) + control_offset;
	int32_t *my_token_ptr = reinterpret_cast<int32_t *>(base_ptr + rank * kCacheLine);

	// Ensure all preceding data stores are visible before we publish the token.
	_mm_sfence();
	nt_store_copy(my_token_ptr, &token, sizeof(int32_t));

	while (true) {
		bool all_ready = true;
		// Flush the entire control block in one pass to amortize sfence cost.
		clflush_range(base_ptr, num_ranks * kCacheLine);
		for (int i = 0; i < num_ranks; i++) {
			volatile int32_t *p =
			    reinterpret_cast<int32_t *>(base_ptr + i * kCacheLine);
			if (*p < token) {
				all_ready = false;
				break;
			}
		}
		if (all_ready) {
			// Prevent subsequent loads from being reordered before the barrier.
			_mm_lfence();
			break;
		}
		_mm_pause();
	}
}

void *cxl_base_ptr() { return g_cxl.base; }
std::size_t cxl_map_length() { return g_cxl.length; }

// ---------------------------------------------------------------------------
// GPU-reduce transfer primitives
// ---------------------------------------------------------------------------

// 1-stage all-reduce — transfer phase only (no CPU reduce).
//
// Writes own slot to CXL, runs the barrier, then bulk-copies all world_size
// slots back to a contiguous GPU staging buffer.  The caller sums on GPU.
//
// CXL slot layout (contiguous):
//   data_offset + 0*slot_bytes  … rank 0's full tensor
//   data_offset + 1*slot_bytes  … rank 1's full tensor
//   …
void cxl_write_barrier_gather_1stage(
    void *device_src,
    void *device_gather_dst,
    std::size_t slot_bytes,
    std::size_t data_offset,
    std::size_t control_offset,
    int rank,
    int num_ranks,
    int32_t token)
{
	if (!g_cxl.base) throw std::runtime_error("CXL not initialized");

	// --- Step 1: GPU → CXL (own slot) ---
	void *cxl_dst =
	    static_cast<uint8_t *>(g_cxl.base) + data_offset + rank * slot_bytes;

	// Stream sync: ensure GPU kernel output is committed before D2H copy.
	at::cuda::getCurrentCUDAStream().synchronize();
	if (!cuda_check(cudaMemcpy(cxl_dst, device_src, slot_bytes,
	                            cudaMemcpyDeviceToHost), "D2H 1stage"))
		throw std::runtime_error("D2H failed in 1-stage write");

	// Flush own slot so other nodes can read it.
	clflush_range(cxl_dst, slot_bytes);

	// --- Step 2: Barrier ---
	cxl_barrier_tp(token, control_offset, rank, num_ranks);

	// --- Step 3: CXL → GPU (all slots, single contiguous read) ---
	// The barrier's _mm_lfence ensures all peer writes are visible.
	// Flush the whole block to evict any stale CPU cache lines before H2D DMA.
	const void *all_slots =
	    static_cast<const uint8_t *>(g_cxl.base) + data_offset;
	const std::size_t total_bytes = num_ranks * slot_bytes;
	clflush_range(all_slots, total_bytes);

	if (!cuda_check(cudaMemcpy(device_gather_dst, all_slots, total_bytes,
	                            cudaMemcpyHostToDevice), "H2D gather 1stage"))
		throw std::runtime_error("H2D gather failed in 1-stage");
}

// 2-stage all-reduce — scatter phase (no CPU reduce).
//
// Writes the full tensor to CXL, runs barrier 1, then uses a single
// cudaMemcpy2D call to gather "my shard" from every peer into a contiguous
// GPU staging buffer.  The caller sums on GPU.
//
// CXL shard layout for rank R (strided):
//   peer 0: data_offset + 0*slot_bytes + R*shard_bytes
//   peer 1: data_offset + 1*slot_bytes + R*shard_bytes
//   …  (stride between rows = slot_bytes)
void cxl_write_barrier_scatter_2stage(
    void *device_src,
    void *device_shard_dst,
    std::size_t slot_bytes,
    std::size_t shard_bytes,
    std::size_t data_offset,
    std::size_t control_offset,
    int rank,
    int num_ranks,
    int32_t token)
{
	if (!g_cxl.base) throw std::runtime_error("CXL not initialized");

	// --- Stage 1a: GPU → CXL (own full slot) ---
	void *cxl_dst =
	    static_cast<uint8_t *>(g_cxl.base) + data_offset + rank * slot_bytes;

	at::cuda::getCurrentCUDAStream().synchronize();
	if (!cuda_check(cudaMemcpy(cxl_dst, device_src, slot_bytes,
	                            cudaMemcpyDeviceToHost), "D2H 2stage scatter"))
		throw std::runtime_error("D2H failed in 2-stage scatter");
	clflush_range(cxl_dst, slot_bytes);

	// --- Stage 1b: Barrier ---
	cxl_barrier_tp(token, control_offset, rank, num_ranks);

	// --- Stage 1c: Strided gather → GPU staging buffer via cudaMemcpy2D ---
	// Flush each shard individually (they are strided, not contiguous).
	const std::size_t my_shard_off = rank * shard_bytes;
	for (int i = 0; i < num_ranks; i++) {
		const void *shard_ptr = static_cast<const uint8_t *>(g_cxl.base) +
		                        data_offset + i * slot_bytes + my_shard_off;
		clflush_range(shard_ptr, shard_bytes);
	}

	// cudaMemcpy2D: copy a 2-D rectangle from strided CXL to contiguous GPU.
	//   width  = shard_bytes  (bytes to copy per row)
	//   height = num_ranks    (one row per peer)
	//   srcPitch = slot_bytes (stride between rows in CXL)
	//   dstPitch = shard_bytes (tightly packed on GPU)
	const void *src_base = static_cast<const uint8_t *>(g_cxl.base) +
	                        data_offset + my_shard_off;
	if (!cuda_check(cudaMemcpy2D(
	        device_shard_dst, shard_bytes,
	        src_base,          slot_bytes,
	        shard_bytes,       num_ranks,
	        cudaMemcpyHostToDevice), "cudaMemcpy2D scatter"))
		throw std::runtime_error("cudaMemcpy2D failed in 2-stage scatter");
}

// 2-stage all-reduce — gather phase.
//
// Writes the GPU-reduced shard to CXL, runs barrier 2, then bulk-copies all
// reduced shards (contiguous in CXL) back to the GPU result buffer.
void cxl_write_barrier_allgather_2stage(
    void *device_reduced_src,
    void *device_result_dst,
    std::size_t shard_bytes,
    std::size_t total_bytes,
    std::size_t reduced_base,
    std::size_t control_offset,
    int rank,
    int num_ranks,
    int32_t token)
{
	if (!g_cxl.base) throw std::runtime_error("CXL not initialized");

	// --- Stage 2a: GPU → CXL (own reduced shard) ---
	void *red_dst =
	    static_cast<uint8_t *>(g_cxl.base) + reduced_base + rank * shard_bytes;

	// Sync before D2H: the GPU sum from the caller must be complete.
	at::cuda::getCurrentCUDAStream().synchronize();
	if (!cuda_check(cudaMemcpy(red_dst, device_reduced_src, shard_bytes,
	                            cudaMemcpyDeviceToHost), "D2H 2stage gather"))
		throw std::runtime_error("D2H failed in 2-stage gather");
	clflush_range(red_dst, shard_bytes);

	// --- Stage 2b: Barrier ---
	cxl_barrier_tp(token, control_offset, rank, num_ranks);

	// --- Stage 2c: CXL → GPU (all reduced shards, contiguous) ---
	const void *red_src =
	    static_cast<const uint8_t *>(g_cxl.base) + reduced_base;
	clflush_range(red_src, total_bytes);

	if (!cuda_check(cudaMemcpy(device_result_dst, red_src, total_bytes,
	                            cudaMemcpyHostToDevice), "H2D 2stage allgather"))
		throw std::runtime_error("H2D failed in 2-stage allgather");
}
