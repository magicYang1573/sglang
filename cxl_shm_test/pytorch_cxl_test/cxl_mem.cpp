// Minimal helpers to map a CXL-backed DAX device and copy data between
// DRAM/VRAM and the mapped region. Functions mirror the patterns used in
// cxl_shm_base_test (mmap + CLFLUSH + cudaMemcpy).

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

inline void clflush_range(const void *addr, std::size_t len) {
	auto p = reinterpret_cast<std::uintptr_t>(addr);
	const std::uintptr_t end = p + len;
	for (; p < end; p += kCacheLine) {
		_mm_clflushopt(reinterpret_cast<void *>(p));
	}
	_mm_sfence();
}

// Streamed store copy to avoid polluting CPU caches when writing to CXL.
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

	// _mm_sfence();
	_mm_sfence();

}

inline bool cuda_check(cudaError_t err, const char *what) {
	if (err == cudaSuccess) {
		return true;
	}
	std::cerr << what << " failed: " << cudaGetErrorString(err) << "\n";
	return false;
}

} // namespace

struct CxlRegion {
	void *base = nullptr;
	std::size_t length = 0;
	bool cuda_registered = false;
};


static CxlRegion g_cxl;

// Map the CXL DAX device into a static region; optionally register with CUDA for faster copies.
bool cxl_init(const std::string &dev_path,
			   std::size_t map_bytes,
			   off_t offset,
			   bool register_cuda,
			   int gpu_id) {
	// If re-initializing, caller is responsible for ensuring no outstanding users.
    std::cout<<"cxl_init "<<g_cxl.base<<std::endl;
	if (g_cxl.base != nullptr) {
		std::cout << "cxl_init: region already mapped; skipping reinit\n";
		return true;
	}

	// const int fd = ::open(dev_path.c_str(), O_RDWR | O_SYNC);
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
	std::memcpy(dram_dst, src, bytes);
	return true;
}

bool vram2cxl(const void *device_src, std::size_t bytes, std::size_t offset) {
	if (g_cxl.base == nullptr || offset + bytes > g_cxl.length) {
		std::cerr << "vram2cxl: invalid region or range\n";
		return false;
	}
	// necessary here, as the data in tensor addr may not valid, cuda stream is async 
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

    // 1. 下发异步拷贝
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


	// sfence is necessary, first store all the data, then set the token 
	_mm_sfence();

	nt_store_copy((void*)my_token_ptr, (void*)&token, sizeof(int32_t));

    while (true) {
        bool all_ready = true;
		std::vector<int32_t> tokens;
		clflush_range((void*)(base_ptr+control_offset), num_ranks*kCacheLine);
        for (int i = 0; i < num_ranks; i++) {

            volatile int32_t* other_token_ptr = reinterpret_cast<int32_t*>(base_ptr + i * kCacheLine);

			int32_t val = *other_token_ptr;
			tokens.push_back(val);

			if (val < token) {
                all_ready = false;
                break;
            }
		}

        if (all_ready) {

			// lfence is necessary here, first all the token is ready, then read the following data
			_mm_lfence();
			break;
		}
		_mm_pause();   
		
    }
}