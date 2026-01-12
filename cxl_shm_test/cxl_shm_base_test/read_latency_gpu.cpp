// Measure read latency from a CXL shared memory pool into GPU VRAM.
// Consistency scheme issues CLFLUSH on the source range before each read.

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
constexpr std::uint8_t kPattern = 0xA5; // known value to verify reads

struct Options {
	std::string dev_path = "/dev/dax0.0";
	std::size_t map_bytes = 64ULL * 1024 * 1024; // 64 MiB
	off_t mmap_offset = 0;
	std::size_t read_bytes = 4096;              // 4 KiB block
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
		} else if (key == "--read-bytes") {
			opt.read_bytes = std::stoull(val);
		} else if (key == "--iterations") {
			opt.iterations = std::stoull(val);
		} else if (key == "--gpu-id") {
			opt.gpu_id = std::stoi(val);
		} else {
			std::cerr << "Unknown option: " << key << "\n";
			return false;
		}
	}
	if (opt.read_bytes == 0 || opt.read_bytes > opt.map_bytes) {
		std::cerr << "read-bytes must be > 0 and <= map-bytes\n";
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

	for (; p < end; p += kCacheLine) {
		_mm_clflushopt(reinterpret_cast<void *>(p));
	}

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
	std::size_t bad_reads = 0;
};

Stats MeasureReadLatencyGpu(void *cxl_ptr, const Options &opt) {
	if (!CudaCheck(cudaSetDevice(opt.gpu_id), "cudaSetDevice")) {
		return {};
	}

	void *d_dst = nullptr;
	std::size_t dst_bytes = opt.read_bytes * opt.iterations;
	if (!CudaCheck(cudaMalloc(&d_dst, dst_bytes), "cudaMalloc")) {
		return {};
	}

	// Pre-fill GPU buffer with a different value so later verification proves this run's copies.
	if (!CudaCheck(cudaMemset(d_dst, 0x11, dst_bytes), "cudaMemset dst")) {
		cudaFree(d_dst);
		return {};
	}

	bool registered = false;
	if (CudaCheck(cudaHostRegister(cxl_ptr, opt.map_bytes, cudaHostRegisterPortable), "cudaHostRegister")) {
		registered = true;
	}

	double total = 0.0;
	double min_v = std::numeric_limits<double>::max();
	double max_v = 0.0;
	std::size_t bad_reads = 0;

	auto *base = static_cast<std::uint8_t*>(cxl_ptr);
	std::size_t span = opt.map_bytes > opt.read_bytes ? opt.map_bytes - opt.read_bytes : 0;
	std::vector<std::uint8_t> golden(opt.read_bytes, kPattern);
	std::vector<std::uint8_t> verify_buf(opt.read_bytes, 0);

	for (std::size_t i = 0; i < opt.iterations; ++i) {
		std::size_t offset = span ? (i * opt.read_bytes) % span : 0;
		void *src = base + offset;
        void *dst = static_cast<std::uint8_t*>(d_dst) + offset;

        auto host_start = std::chrono::high_resolution_clock::now();
		// Consistency: flush CPU caches before the GPU read.
		ClflushRange(src, opt.read_bytes);

		auto copy_status = cudaMemcpy(dst, src, opt.read_bytes, cudaMemcpyHostToDevice);
		if (copy_status != cudaSuccess) {
			std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(copy_status) << "\n";
			continue;
		}
		auto host_end = std::chrono::high_resolution_clock::now();

		double ns = std::chrono::duration<double, std::nano>(host_end - host_start).count();
		// std::cout << "src: " << src << " dst: " << dst << " Latency: " << ns << " ns" << std::endl;
		total += ns;
		if (ns < min_v) min_v = ns;
		if (ns > max_v) max_v = ns;

		// Verify contents match expected pattern without affecting measured latency window.
		auto verify_status = cudaMemcpy(verify_buf.data(), dst, opt.read_bytes, cudaMemcpyDeviceToHost);
		if (verify_status != cudaSuccess || std::memcmp(verify_buf.data(), golden.data(), opt.read_bytes) != 0) {
			++bad_reads;
		}
	}

	if (registered) {
		cudaHostUnregister(cxl_ptr);
	}

	cudaFree(d_dst);

	Stats s;
	s.avg_ns = total / static_cast<double>(opt.iterations);
	s.min_ns = min_v;
	s.max_ns = max_v;
	s.bad_reads = bad_reads;
	return s;
}

void PrintUsage(const char *prog) {
	std::cerr << "Usage: " << prog << " [--dev=/dev/dax0.0] [--map-bytes=67108864] [--offset=0]"
			<< " [--read-bytes=4096] [--iterations=10000] [--gpu-id=0]\n";
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

	// Initialize the mapped region with a known pattern once.
	std::memset(base, kPattern, opt.read_bytes*opt.iterations);

	Stats stats = MeasureReadLatencyGpu(base, opt);

	std::cout << "Device: " << opt.dev_path << "\n";
	std::cout << "GPU ID: " << opt.gpu_id << "\n";
	std::cout << "Map bytes: " << opt.map_bytes << " (offset " << opt.mmap_offset << ")\n";
	std::cout << "Read bytes: " << opt.read_bytes << "\n";
	std::cout << "Iterations: " << opt.iterations << "\n";
	std::cout << "CXL->GPU read latency (ns): avg=" << stats.avg_ns << ", min=" << stats.min_ns << ", max=" << stats.max_ns << "\n";
	std::cout << "Bad reads: " << stats.bad_reads << "\n";

	munmap(base, opt.map_bytes);
	return 0;
}
