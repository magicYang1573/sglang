// Measure write latency from GPU VRAM into a CXL shared memory pool.
// The consistency scheme issues CLFLUSH on the destination range after each write.

#include <chrono>
#include <cinttypes>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <immintrin.h>
#include <cpuid.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

constexpr std::size_t kCacheLine = 64;
constexpr std::uint8_t kPattern = 0x5A; // known value to verify writes

struct Options {
	std::string dev_path = "/dev/dax0.0";
	std::size_t map_bytes = 64ULL * 1024 * 1024; // 64 MiB
	off_t mmap_offset = 0;
	std::size_t write_bytes = 4096;              // 4 KiB block
	std::size_t iterations = 10000;
	int gpu_id = 0;
};

bool ParseArgs(int argc, char **argv, Options &opt) {
	for (int i = 1; i < argc; ++i) {
		std::string arg(argv[i]);
		auto pos = arg.find('=');
		if (pos == std::string::npos || pos == 2) {
			std::cerr << "Unexpected arg format (use --key=value): " << arg << "\n";
			return false;
		}
		auto key = arg.substr(0, pos);
		auto val = arg.substr(pos + 1);
		if (key == "--dev") {
			opt.dev_path = val;
		} else if (key == "--map-bytes") {
			opt.map_bytes = std::stoull(val);
		} else if (key == "--offset") {
			opt.mmap_offset = static_cast<off_t>(std::stoll(val));
		} else if (key == "--write-bytes") {
			opt.write_bytes = std::stoull(val);
		} else if (key == "--iterations") {
			opt.iterations = std::stoull(val);
		} else if (key == "--gpu-id") {
			opt.gpu_id = std::stoi(val);
		} else {
			std::cerr << "Unknown option: " << key << "\n";
			return false;
		}
	}
	if (opt.write_bytes == 0 || opt.write_bytes > opt.map_bytes) {
		std::cerr << "write-bytes must be > 0 and <= map-bytes\n";
		return false;
	}
	return true;
}

void *MapCxl(const Options &opt) {
	int fd = open(opt.dev_path.c_str(), O_RDWR | O_SYNC);
	if (fd < 0) {
		std::cerr << "open " << opt.dev_path << " failed: " << strerror(errno) << "\n";
		return nullptr;
	}

	void *addr = mmap(nullptr, opt.map_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, opt.mmap_offset);
	if (addr == MAP_FAILED) {
		std::cerr << "mmap failed: " << strerror(errno) << " (offset=" << opt.mmap_offset << ")\n";
		close(fd);
		return nullptr;
	}
	close(fd);
	return addr;
}

inline void ClflushRange(const void *addr, std::size_t len) {
    auto p = reinterpret_cast<std::uintptr_t>(addr);
    std::uintptr_t end = p + len;

    // // 1. 对齐到缓存行边界
    // p &= ~(static_cast<std::uintptr_t>(kCacheLine) - 1);

    // 2. 循环展开：将 reinterpret_cast 修改为 (void *)
    // while (p + 4 * kCacheLine <= end) {
    //     _mm_clflushopt(reinterpret_cast<void *>(p));
    //     _mm_clflushopt(reinterpret_cast<void *>(p + kCacheLine));
    //     _mm_clflushopt(reinterpret_cast<void *>(p + 2 * kCacheLine));
    //     _mm_clflushopt(reinterpret_cast<void *>(p + 3 * kCacheLine));
    //     p += 4 * kCacheLine;
    // }

    // 3. 处理剩余部分
    for (; p < end; p += kCacheLine) {
        _mm_clflushopt(reinterpret_cast<void *>(p));
    }

    // 4. 必须使用 SFENCE 确保之前的异步 flush 操作完成
    _mm_sfence();
}

bool CudaCheck(cudaError_t err, const char *what) {
	if (err == cudaSuccess) {
		return true;
	}
	std::cerr << what << " failed: " << cudaGetErrorString(err) << "\n";
	return false;
}

struct Stats {
	double avg_ns = 0.0;
	double min_ns = 0.0;
	double max_ns = 0.0;
	std::size_t bad_writes = 0;
};

Stats MeasureWriteLatencyGpu(void *cxl_ptr, const Options &opt) {
	std::size_t src_bytes = opt.write_bytes * opt.iterations;

	if (!CudaCheck(cudaSetDevice(opt.gpu_id), "cudaSetDevice")) {
		return {};
	}

	void *d_src = nullptr;
	if (!CudaCheck(cudaMalloc(&d_src, src_bytes), "cudaMalloc")) {
		return {};
	}

	// Initialize the device buffer with a known pattern.
	if (!CudaCheck(cudaMemset(d_src, kPattern, src_bytes), "cudaMemset")) {
		cudaFree(d_src);
		return {};
	}

	// Attempt to register the CXL mapping so the GPU can access it directly.
	bool registered = false;
	if (CudaCheck(cudaHostRegister(cxl_ptr, opt.map_bytes, cudaHostRegisterPortable), "cudaHostRegister")) {
		registered = true;
	}

	double total = 0.0;
	double min_v = std::numeric_limits<double>::max();
	double max_v = 0.0;
	std::size_t bad_writes = 0;

	auto *base = static_cast<std::uint8_t*>(cxl_ptr);
	std::size_t span = opt.map_bytes > opt.write_bytes ? opt.map_bytes - opt.write_bytes : 0;

	// Host-side pattern for correctness check after the writes.
	std::vector<std::uint8_t> golden(opt.write_bytes, kPattern);

	for (std::size_t i = 0; i < opt.iterations; ++i) {
		std::size_t offset = span ? (i * opt.write_bytes) % span : 0;
		void *dst = base + offset;
		void *src_ptr = static_cast<std::uint8_t*>(d_src) + offset;

		auto host_start = std::chrono::high_resolution_clock::now();

		auto copy_status = cudaMemcpy(dst, src_ptr, opt.write_bytes, cudaMemcpyDeviceToHost);
		if (copy_status != cudaSuccess) {
			std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(copy_status) << "\n";
			continue;
		}

		// Consistency: flush CPU caches after the GPU write.
		ClflushRange(dst, opt.write_bytes);

		auto host_end = std::chrono::high_resolution_clock::now();
		double ns = std::chrono::duration<double, std::nano>(host_end - host_start).count();
        // std::cout<<"src: " << src_ptr << " "<< "dst: " << dst << " ";
        // std::cout << "Latency: " << ns << " ns" << std::endl;
		total += ns;
		if (ns < min_v) min_v = ns;
		if (ns > max_v) max_v = ns;
	}

	if (registered) {
		cudaHostUnregister(cxl_ptr);
	}

	// Verify contents match the expected pattern.
	for (std::size_t i = 0; i < opt.iterations; ++i) {
		std::size_t offset = span ? (i * opt.write_bytes) % span : 0;
		void *dst = base + offset;
		if (std::memcmp(dst, golden.data(), opt.write_bytes) != 0) {
			++bad_writes;
		}
	}

	cudaFree(d_src);

	Stats s;
	s.avg_ns = total / static_cast<double>(opt.iterations);
	s.min_ns = min_v;
	s.max_ns = max_v;
	s.bad_writes = bad_writes;
	return s;
}

void PrintUsage(const char *prog) {
	std::cerr << "Usage: " << prog << " [--dev=/dev/dax0.0] [--map-bytes=67108864] [--offset=0]"
			  << " [--write-bytes=4096] [--iterations=10000] [--gpu-id=0]\n";
}

} // namespace

int main(int argc, char **argv) {
	Options opt;
	if (!ParseArgs(argc, argv, opt)) {
		PrintUsage(argv[0]);
		return 1;
	}

	void *base = MapCxl(opt);
	if (base == nullptr) {
		return 1;
	}

	// Touch once to fault in the pages and establish a baseline.
	std::memset(base, 0xCD, opt.write_bytes);

	Stats stats = MeasureWriteLatencyGpu(base, opt);

	std::cout << "Device: " << opt.dev_path << "\n";
	std::cout << "GPU ID: " << opt.gpu_id << "\n";
	std::cout << "Map bytes: " << opt.map_bytes << " (offset " << opt.mmap_offset << ")\n";
	std::cout << "Write bytes: " << opt.write_bytes << "\n";
	std::cout << "Iterations: " << opt.iterations << "\n";
	std::cout << "GPU->CXL write latency (ns): avg=" << stats.avg_ns << ", min=" << stats.min_ns << ", max=" << stats.max_ns << "\n";
	std::cout << "Bad writes: " << stats.bad_writes << "\n";

	munmap(base, opt.map_bytes);
	return 0;
}
