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

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <unistd.h>

#include "cxl_mem.hpp"

namespace {

constexpr std::size_t kCacheLine = 64;

inline void clflush_range(const void *addr, std::size_t len) {
	auto p = reinterpret_cast<std::uintptr_t>(addr);
	const std::uintptr_t end = p + len;
	for (; p < end; p += kCacheLine) {
		_mm_clflushopt(reinterpret_cast<void *>(p));
	}
	_mm_sfence();
}

inline void nt_store_copy(void *dst, const void *src, std::size_t len) {
	auto *d8 = static_cast<std::uint8_t *>(dst);
	auto *s8 = static_cast<const std::uint8_t *>(src);

	const std::size_t qwords = len / sizeof(std::uint64_t);
	auto *d64 = reinterpret_cast<std::uint64_t *>(d8);
	auto *s64 = reinterpret_cast<const std::uint64_t *>(s8);
	for (std::size_t i = 0; i < qwords; ++i) {
		_mm_stream_si64(reinterpret_cast<long long *>(d64 + i), static_cast<long long>(s64[i]));
	}

	const std::size_t consumed = qwords * sizeof(std::uint64_t);
	d8 += consumed;
	s8 += consumed;
	std::size_t rem = len - consumed;

	if (rem >= sizeof(std::uint32_t)) {
		_mm_stream_si32(reinterpret_cast<int *>(d8), *reinterpret_cast<const int *>(s8));
		d8 += sizeof(std::uint32_t);
		s8 += sizeof(std::uint32_t);
		rem -= sizeof(std::uint32_t);
	}

	if (rem > 0) {
		std::memcpy(d8, s8, rem);
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

// SSE-based load from non-cache-coherent CXL memory (128-bit streaming loads).
// Uses MOVNTDQA which bypasses cache hierarchy, reading directly from memory.
// Handles unaligned source addresses by processing the leading unaligned
// bytes with a regular copy, then switching to streaming loads for the
// aligned bulk, and finally copying any tail bytes.
inline void sse_load_copy(void *dst, const void *src, std::size_t len) {
	auto *d8 = static_cast<uint8_t *>(dst);
	auto *s8 = static_cast<const uint8_t *>(src);

	// Handle leading unaligned bytes
	auto s_addr = reinterpret_cast<std::uintptr_t>(s8);
	std::size_t align_offset = (16 - (s_addr & 15)) & 15;
	if (align_offset > len) align_offset = len;
	if (align_offset > 0) {
		std::memcpy(d8, s8, align_offset);
		d8 += align_offset;
		s8 += align_offset;
		len -= align_offset;
	}

	// Aligned bulk: streaming loads
	std::size_t chunks = len / sizeof(__m128i);
	auto *d128 = reinterpret_cast<__m128i *>(d8);
	auto *s128 = reinterpret_cast<const __m128i *>(s8);
	for (std::size_t i = 0; i < chunks; ++i) {
		_mm_storeu_si128(d128 + i, _mm_stream_load_si128(const_cast<__m128i *>(s128 + i)));
	}
	std::size_t consumed = chunks * sizeof(__m128i);
	if (consumed < len) {
		std::memcpy(d8 + consumed, s8 + consumed, len - consumed);
	}
}

// Vectorized reduce: read N shards from CXL and sum into a DRAM destination.
// Each shard is at a different CXL offset. Uses SSE streaming loads to bypass
// CPU caches (critical on non-coherent CXL).
// elem_type: 0 = float16, 1 = bfloat16, 2 = float32
static void reduce_shards_to_dram(
	void *dram_dst,
	std::size_t shard_bytes,
	const std::size_t *cxl_offsets,
	int num_srcs,
	int elem_type,
	void *cxl_base)
{
	if (num_srcs <= 0) return;

	const std::size_t float_count = (elem_type == 2) ? shard_bytes / 4 : shard_bytes / 2;

	// Temporary staging buffer for CXL reads and fp32 accumulation
	thread_local std::vector<float> accum;
	thread_local std::vector<uint8_t> staging;
	accum.resize(float_count);
	staging.resize(shard_bytes);

	auto convert_to_float = [&](const void *raw, float *out, std::size_t count, int etype) {
		if (etype == 2) {
			std::memcpy(out, raw, count * sizeof(float));
		} else if (etype == 0) {
			// float16
			const uint16_t *h = static_cast<const uint16_t *>(raw);
			for (std::size_t i = 0; i < count; i += 4) {
				__m128i packed = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(h + i));
				__m128 floats = _mm_cvtph_ps(packed);
				_mm_storeu_ps(out + i, floats);
			}
			for (std::size_t i = (count / 4) * 4; i < count; ++i) {
				__m128i packed = _mm_set1_epi16(static_cast<short>(h[i]));
				__m128 f = _mm_cvtph_ps(packed);
				out[i] = _mm_cvtss_f32(f);
			}
		} else {
			// bfloat16: shift left 16 bits to get float
			const uint16_t *b = static_cast<const uint16_t *>(raw);
			for (std::size_t i = 0; i < count; ++i) {
				uint32_t bits = static_cast<uint32_t>(b[i]) << 16;
				std::memcpy(out + i, &bits, sizeof(float));
			}
		}
	};

	auto convert_from_float = [&](const float *in, void *raw, std::size_t count, int etype) {
		if (etype == 2) {
			std::memcpy(raw, in, count * sizeof(float));
		} else if (etype == 0) {
			uint16_t *h = static_cast<uint16_t *>(raw);
			for (std::size_t i = 0; i < count; i += 4) {
				__m128 floats = _mm_loadu_ps(in + i);
				__m128i packed = _mm_cvtps_ph(floats, _MM_FROUND_TO_NEAREST_INT);
				_mm_storel_epi64(reinterpret_cast<__m128i *>(h + i), packed);
			}
			for (std::size_t i = (count / 4) * 4; i < count; ++i) {
				__m128 f = _mm_set_ss(in[i]);
				__m128i packed = _mm_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT);
				h[i] = static_cast<uint16_t>(_mm_extract_epi16(packed, 0));
			}
		} else {
			uint16_t *b = static_cast<uint16_t *>(raw);
			for (std::size_t i = 0; i < count; ++i) {
				uint32_t bits;
				std::memcpy(&bits, in + i, sizeof(float));
				b[i] = static_cast<uint16_t>(bits >> 16);
			}
		}
	};

	// First shard: streaming load + convert to fp32
	{
		const void *src = static_cast<const uint8_t *>(cxl_base) + cxl_offsets[0];
		clflush_range(src, shard_bytes);
		sse_load_copy(staging.data(), src, shard_bytes);
		convert_to_float(staging.data(), accum.data(), float_count, elem_type);
	}

	// Remaining shards: streaming load + accumulate in fp32
	thread_local std::vector<float> tmp_float;
	tmp_float.resize(float_count);
	for (int s = 1; s < num_srcs; ++s) {
		const void *src = static_cast<const uint8_t *>(cxl_base) + cxl_offsets[s];
		clflush_range(src, shard_bytes);
		sse_load_copy(staging.data(), src, shard_bytes);
		convert_to_float(staging.data(), tmp_float.data(), float_count, elem_type);

		// Vectorized accumulate
		std::size_t i = 0;
		for (; i + 4 <= float_count; i += 4) {
			__m128 a = _mm_loadu_ps(accum.data() + i);
			__m128 b = _mm_loadu_ps(tmp_float.data() + i);
			_mm_storeu_ps(accum.data() + i, _mm_add_ps(a, b));
		}
		for (; i < float_count; ++i) {
			accum[i] += tmp_float[i];
		}
	}

	// Convert back to original dtype
	convert_from_float(accum.data(), dram_dst, float_count, elem_type);
}

} // namespace

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
    std::cout<<"cxl_init "<<g_cxl.base<<std::endl;
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

	g_cxl.base = addr;
	g_cxl.length = map_bytes;
	g_cxl.cuda_registered = false;

	if (register_cuda) {
		if (!cuda_check(cudaSetDevice(gpu_id), "cudaSetDevice")) {
			::munmap(addr, map_bytes);
			g_cxl = {};
			return false;
		}
		if (cuda_check(cudaHostRegister(addr, map_bytes, cudaHostRegisterPortable), "cudaHostRegister")) {
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
	sse_load_copy(dram_dst, src, bytes);
	return true;
}

bool vram2cxl(const void *device_src, std::size_t bytes, std::size_t offset) {
	if (g_cxl.base == nullptr || offset + bytes > g_cxl.length) {
		std::cerr << "vram2cxl: invalid region or range\n";
		return false;
	}
	at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
	torch_stream.synchronize();
	
	void *dst = static_cast<std::uint8_t *>(g_cxl.base) + offset;
	if (!cuda_check(cudaMemcpy(dst, device_src, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H")) {
		return false;
	}
	clflush_range(dst, bytes);
	return true;
}


bool vram2cxl_async(const void *device_src, std::size_t bytes, std::size_t offset) {
    if (g_cxl.base == nullptr || offset + bytes > g_cxl.length) return false;

	at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    void *dst = static_cast<std::uint8_t *>(g_cxl.base) + offset;

    if (!cuda_check(cudaMemcpyAsync(dst, device_src, bytes, cudaMemcpyDeviceToHost, stream), "D2H")) {
        return false;
    }

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
	if (!cuda_check(cudaMemcpy(device_dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D")) {
		return false;
	}
	return true;
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


void cxl_barrier_tp(int32_t token, int64_t control_offset, int rank, int num_ranks) {
    if (!g_cxl.base) {
        throw std::runtime_error("CXL not initialized. Call cxl_init first.");
    }

    uint8_t* base_ptr = static_cast<uint8_t*>(g_cxl.base) + control_offset;

    int32_t* my_token_ptr = reinterpret_cast<int32_t*>(base_ptr + rank * kCacheLine);

	_mm_sfence();

	nt_store_copy((void*)my_token_ptr, (void*)&token, sizeof(int32_t));

    while (true) {
        bool all_ready = true;
		clflush_range(base_ptr, num_ranks * kCacheLine);
        for (int i = 0; i < num_ranks; i++) {
            volatile int32_t* other_token_ptr = reinterpret_cast<int32_t*>(base_ptr + i * kCacheLine);
			int32_t val = *other_token_ptr;
			if (val < token) {
                all_ready = false;
                break;
            }
		}

        if (all_ready) {
			_mm_lfence();
			break;
		}
		_mm_pause();   
    }
}

void *cxl_base_ptr() {
	return g_cxl.base;
}

std::size_t cxl_map_length() {
	return g_cxl.length;
}

// Fused: read N shards from CXL, sum them in CPU (fp32 accumulation), write result to VRAM.
// This avoids N separate CXL→VRAM transfers and N GPU kernels for summing.
void cxl_read_reduce_shards_to_vram(
	void *device_dst,
	std::size_t shard_bytes,
	const std::size_t *src_offsets,
	int num_srcs,
	int elem_type)
{
	if (!g_cxl.base) {
		throw std::runtime_error("CXL not initialized");
	}

	thread_local std::vector<uint8_t> result_buf;
	result_buf.resize(shard_bytes);

	reduce_shards_to_dram(result_buf.data(), shard_bytes, src_offsets, num_srcs, elem_type, g_cxl.base);

	if (!cuda_check(cudaMemcpy(device_dst, result_buf.data(), shard_bytes, cudaMemcpyHostToDevice),
					"cudaMemcpy H2D (reduce result)")) {
		throw std::runtime_error("Failed to copy reduced result to VRAM");
	}
}

// Fused 1-stage all-reduce:
// 1. GPU→CXL write (own slot)
// 2. Barrier
// 3. Read all slots from CXL, reduce on CPU, write result to GPU
void cxl_allreduce_1stage(
	void *device_inout,
	std::size_t slot_bytes,
	std::size_t data_offset,
	std::size_t control_offset,
	int rank,
	int num_ranks,
	int32_t token,
	int elem_type)
{
	if (!g_cxl.base) {
		throw std::runtime_error("CXL not initialized");
	}

	// Step 1: GPU → CXL (own slot)
	void *cxl_dst = static_cast<uint8_t *>(g_cxl.base) + data_offset + rank * slot_bytes;

	at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
	stream.synchronize();

	if (!cuda_check(cudaMemcpy(cxl_dst, device_inout, slot_bytes, cudaMemcpyDeviceToHost), "D2H 1stage")) {
		throw std::runtime_error("vram2cxl failed in 1-stage allreduce");
	}
	clflush_range(cxl_dst, slot_bytes);

	// Step 2: Barrier
	cxl_barrier_tp(token, control_offset, rank, num_ranks);

	// Step 3: Read all slots, reduce on CPU, copy to GPU
	std::vector<std::size_t> offsets(num_ranks);
	for (int i = 0; i < num_ranks; ++i) {
		offsets[i] = data_offset + i * slot_bytes;
	}

	thread_local std::vector<uint8_t> result_buf;
	result_buf.resize(slot_bytes);

	reduce_shards_to_dram(result_buf.data(), slot_bytes, offsets.data(), num_ranks, elem_type, g_cxl.base);

	if (!cuda_check(cudaMemcpy(device_inout, result_buf.data(), slot_bytes, cudaMemcpyHostToDevice),
					"H2D 1stage result")) {
		throw std::runtime_error("Failed to copy result to VRAM");
	}
}

// Fused 2-stage all-reduce (reduce-scatter + all-gather):
// Stage 1: GPU→CXL, barrier, each rank reduces its own shard on CPU, writes reduced shard to CXL
// Stage 2: Barrier, read all reduced shards from CXL → GPU
void cxl_allreduce_2stage(
	void *device_inout,
	std::size_t total_bytes,
	std::size_t data_offset,
	std::size_t reduced_base,
	std::size_t control_offset,
	int rank,
	int num_ranks,
	int32_t token_start,
	int elem_type)
{
	if (!g_cxl.base) {
		throw std::runtime_error("CXL not initialized");
	}

	std::size_t slot_bytes = total_bytes;
	std::size_t shard_bytes = total_bytes / num_ranks;

	// Stage 1a: GPU → CXL (own full tensor)
	void *cxl_dst = static_cast<uint8_t *>(g_cxl.base) + data_offset + rank * slot_bytes;

	at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
	stream.synchronize();

	if (!cuda_check(cudaMemcpy(cxl_dst, device_inout, slot_bytes, cudaMemcpyDeviceToHost), "D2H 2stage")) {
		throw std::runtime_error("vram2cxl failed in 2-stage allreduce");
	}
	clflush_range(cxl_dst, slot_bytes);

	// Stage 1b: Barrier (all ranks have written)
	int32_t token1 = token_start;
	cxl_barrier_tp(token1, control_offset, rank, num_ranks);

	// Stage 1c: Read own shard from all peers, reduce on CPU, write to reduced area
	std::vector<std::size_t> shard_offsets(num_ranks);
	std::size_t my_shard_byte_offset = rank * shard_bytes;
	for (int i = 0; i < num_ranks; ++i) {
		shard_offsets[i] = data_offset + i * slot_bytes + my_shard_byte_offset;
	}

	thread_local std::vector<uint8_t> reduced_buf;
	reduced_buf.resize(shard_bytes);

	reduce_shards_to_dram(reduced_buf.data(), shard_bytes, shard_offsets.data(), num_ranks, elem_type, g_cxl.base);

	// Write reduced shard to CXL
	void *red_dst = static_cast<uint8_t *>(g_cxl.base) + reduced_base + rank * shard_bytes;
	nt_store_copy(red_dst, reduced_buf.data(), shard_bytes);

	// Stage 2a: Barrier (all reduced shards written)
	int32_t token2 = token_start + 1;
	cxl_barrier_tp(token2, control_offset, rank, num_ranks);

	// Stage 2b: Read all reduced shards from CXL → GPU
	const void *red_src = static_cast<const uint8_t *>(g_cxl.base) + reduced_base;
	clflush_range(red_src, total_bytes);

	// Use SSE streaming load into a staging buffer then H2D to GPU
	thread_local std::vector<uint8_t> gather_buf;
	gather_buf.resize(total_bytes);
	sse_load_copy(gather_buf.data(), red_src, total_bytes);

	if (!cuda_check(cudaMemcpy(device_inout, gather_buf.data(), total_bytes, cudaMemcpyHostToDevice),
					"H2D 2stage gather")) {
		throw std::runtime_error("Failed to copy gathered result to VRAM");
	}
}
