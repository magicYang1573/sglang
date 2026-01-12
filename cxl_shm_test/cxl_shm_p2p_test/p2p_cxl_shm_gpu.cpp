// GPU P2P test using the same control/handshake logic as the CPU peer.
// Active role: GPU writes patterns into CXL memory and waits for passive ack.
// Passive role: GPU reads from CXL memory and verifies payloads.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <immintrin.h>

#include <cuda_runtime.h>

namespace {

constexpr std::size_t kCacheLine = 64;
constexpr int interval_ns = 100;

struct Options {
	std::string dev_path = "/dev/dax0.0";
	uint64_t map_bytes = 256ULL * 1024 * 1024 * 1024;
	uint64_t data_bytes = 64ULL * 1024 * 1024;
	uint64_t iterations = 100;
	uint64_t offset = 0;
	std::string role = "active";   // active | passive
	bool oneshot = false;
	int wait_timeout_ms = 30000;
	int gpu_id = 0;
};

struct alignas(64) ControlBlock {
	std::atomic<uint64_t> magic{0};
	std::atomic<uint64_t> data_bytes{0};
	std::atomic<uint64_t> iterations{0};
	std::atomic<uint64_t> ready_active{0};
	std::atomic<uint64_t> ready_passive{0};
	std::atomic<uint64_t> seq{0};
	std::atomic<uint64_t> ack{0};
	std::atomic<uint64_t> stop{0};
};

constexpr uint64_t kMagic = 0x33334c5032503275ULL;  // CXL P2P u
constexpr size_t kControlBytes = 4 * 4096;  // data region offset remains
constexpr size_t kCtrlFlushBytes = ((sizeof(ControlBlock) + 63) / 64) * 64;  // flush/invalidate actual control block size

void DumpState(const ControlBlock* ctrl) {
	std::cerr << "[ctrl] magic=" << std::hex << ctrl->magic.load(std::memory_order_acquire)
			  << std::dec
			  << " ready_active=" << ctrl->ready_active.load(std::memory_order_acquire)
			  << " ready_passive=" << ctrl->ready_passive.load(std::memory_order_acquire)
			  << " stop=" << ctrl->stop.load(std::memory_order_acquire)
			  << std::endl;
}

std::atomic<bool> g_stop{false};

void SigHandler(int) { g_stop.store(true); }

void Usage(const char* prog) {
	std::cout << "Usage: " << prog
			  << " [--role=active|passive]"
			  << " [--dev=/dev/dax0.0]"
			  << " [--map-bytes=256G]"
			  << " [--data-bytes=64M]"
			  << " [--iterations=100]"
			  << " [--offset=0]"
			  << " [--oneshot]"
			  << " [--wait-ms=30000]"
			  << " [--gpu-id=0]"
			  << std::endl;
}

bool ParseSize(const std::string& text, uint64_t& out) {
	if (text.empty()) return false;
	char unit = text.back();
	uint64_t mul = 1;
	std::string num = text;
	if (unit == 'K' || unit == 'k') { mul = 1024ULL; num.pop_back(); }
	else if (unit == 'M' || unit == 'm') { mul = 1024ULL * 1024ULL; num.pop_back(); }
	else if (unit == 'G' || unit == 'g') { mul = 1024ULL * 1024ULL * 1024ULL; num.pop_back(); }
	else if (unit == 'T' || unit == 't') { mul = 1024ULL * 1024ULL * 1024ULL * 1024ULL; num.pop_back(); }
	char* end = nullptr;
	uint64_t base = std::strtoull(num.c_str(), &end, 10);
	if (end == nullptr || *end != '\0') return false;
	out = base * mul;
	return true;
}

bool ParseUint(const std::string& text, uint64_t& out) {
	char* end = nullptr;
	uint64_t v = std::strtoull(text.c_str(), &end, 10);
	if (end == nullptr || *end != '\0') return false;
	out = v;
	return true;
}

inline void FlushCtrl(ControlBlock* ctrl) {
#if defined(__x86_64__) || defined(_M_X64)
	auto* p = reinterpret_cast<char*>(ctrl);
	// _mm_sfence();
	for (size_t off = 0; off < kCtrlFlushBytes; off += 64) {
		_mm_clflushopt(p + off);
	}
	_mm_sfence();
#else
	std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

inline void ClflushRange(const void *addr, std::size_t len) {
	auto p = reinterpret_cast<std::uintptr_t>(addr);
	std::uintptr_t end = p + len;

	for (; p < end; p += kCacheLine) {
		_mm_clflushopt(reinterpret_cast<void *>(p));
	}

	_mm_sfence();
}

bool CudaCheck(cudaError_t err, const char* what) {
	if (err == cudaSuccess) return true;
	std::cerr << what << " failed: " << cudaGetErrorString(err) << std::endl;
	return false;
}

bool WaitEqual(ControlBlock* ctrl, const std::atomic<uint64_t>& target, uint64_t expected, int timeout_ms) {
	auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
	while (std::chrono::steady_clock::now() < deadline && !g_stop.load()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(interval_ns));
		FlushCtrl(ctrl);
		if (target.load(std::memory_order_acquire) == expected) return true;
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	FlushCtrl(ctrl);
	return target.load(std::memory_order_acquire) == expected;
}

bool ActiveLatency(ControlBlock* ctrl, uint8_t* data, const Options& opt) {
	if (!CudaCheck(cudaSetDevice(opt.gpu_id), "cudaSetDevice")) return false;

	void* d_src = nullptr;
	if (!CudaCheck(cudaMalloc(&d_src, opt.data_bytes), "cudaMalloc")) return false;

	bool registered = false;
	if (CudaCheck(cudaHostRegister(data, opt.data_bytes*opt.iterations, cudaHostRegisterPortable), "cudaHostRegister")) {
		registered = true;
	}

	auto start = std::chrono::steady_clock::now();
	for (uint64_t i = 0; i < opt.iterations && !g_stop.load(); ++i) {
		uint64_t val = i + 1;
		uint8_t pattern = static_cast<uint8_t>(val & 0xFF);

		if (!CudaCheck(cudaMemset(d_src, pattern, opt.data_bytes), "cudaMemset")) break;

        uint8_t* data_ptr = data + i * opt.data_bytes;
        
		auto copy_status = cudaMemcpy(data_ptr, d_src, opt.data_bytes, cudaMemcpyDeviceToHost);
		if (copy_status != cudaSuccess) {
			std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(copy_status) << std::endl;
			break;
		}

		// Flush to ensure host caches do not hide the freshly written payload.
		ClflushRange(data_ptr, opt.data_bytes);

		ctrl->seq.store(val, std::memory_order_release);
		while (!g_stop.load()) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(interval_ns));
			FlushCtrl(ctrl);
			if (ctrl->ack.load(std::memory_order_acquire) == val) break;
		}

		std::cout << "[active] iter=" << (i + 1) << " seq=" << val << std::endl;
	}
	auto end = std::chrono::steady_clock::now();

	ctrl->stop.store(1, std::memory_order_release);
	FlushCtrl(ctrl);

	double ns_total = std::chrono::duration<double, std::nano>(end - start).count();
	double avg_ns = ns_total / static_cast<double>(opt.iterations);
	std::cout << "[latency] iterations=" << opt.iterations
			  << " avg=" << avg_ns << " ns" << std::endl;

	if (registered) {
		cudaHostUnregister(data);
	}
	cudaFree(d_src);
	return true;
}

void PassiveLatency(ControlBlock* ctrl, uint8_t* data, const Options& opt) {
	uint64_t last = 0;
	uint64_t done = 0;
	uint64_t errors = 0;
	const uint64_t expect_iters = ctrl->iterations.load(std::memory_order_acquire);
	const uint64_t data_len = ctrl->data_bytes.load(std::memory_order_acquire);

	if (!CudaCheck(cudaSetDevice(opt.gpu_id), "cudaSetDevice")) return;

	void* d_dst = nullptr;
	if (!CudaCheck(cudaMalloc(&d_dst, data_len), "cudaMalloc")) return;

	bool registered = false;
	if (CudaCheck(cudaHostRegister(data, data_len*opt.iterations, cudaHostRegisterPortable), "cudaHostRegister")) {
		registered = true;
	}

	std::vector<uint8_t> verify(data_len, 0);

	while (!g_stop.load()) {
		FlushCtrl(ctrl);    // take about 25 ns
        std::this_thread::sleep_for(std::chrono::nanoseconds(interval_ns));

		uint64_t seq = ctrl->seq.load(std::memory_order_acquire);
		if (seq != 0 && seq != last) {
			uint8_t expected = static_cast<uint8_t>(seq & 0xFF);
            uint8_t* data_ptr = data + done * data_len;
			// Ensure CPU caches are clean before GPU reads from the region.
			ClflushRange(data_ptr, data_len);
			auto copy_status = cudaMemcpy(d_dst, data_ptr, data_len, cudaMemcpyHostToDevice);
			if (copy_status != cudaSuccess) {
				std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(copy_status) << std::endl;
				break;
			}
            
            ctrl->ack.store(seq, std::memory_order_release);
			FlushCtrl(ctrl);
            
			// auto verify_status = cudaMemcpy(verify.data(), d_dst, data_len, cudaMemcpyDeviceToHost);
			// if (verify_status != cudaSuccess) {
			// 	std::cerr << "cudaMemcpy D2H verify failed: " << cudaGetErrorString(verify_status) << std::endl;
			// 	break;
			// }

			// bool ok = true;
			// for (uint64_t i = 0; i < data_len; ++i) {
			// 	if (verify[i] != expected) {
			// 		ok = false;
			// 		break;
			// 	}
			// }
			// if (!ok) ++errors;


			// std::cout << "[passiveeeee] iter=" << (done + 1) << " seq=" << seq << (ok ? " ok" : " mismatch") << std::endl;

			last = seq;
			++done;
			if (expect_iters != 0 && done >= expect_iters) break;
		}
		if (ctrl->stop.load(std::memory_order_acquire) != 0) break;
	}

	if (registered) {
		cudaHostUnregister(data);
	}
	cudaFree(d_dst);

	if (errors != 0) {
		std::cerr << "[passive] data mismatch count=" << errors << std::endl;
	}
}

void PassiveLoop(ControlBlock* ctrl, uint8_t* data, const Options& opt) {
	ctrl->ready_passive.store(1, std::memory_order_release);
	FlushCtrl(ctrl);

	PassiveLatency(ctrl, data, opt);
}

bool StartActive(ControlBlock* ctrl, uint8_t* data, const Options& opt) {
	ctrl->data_bytes.store(opt.data_bytes, std::memory_order_release);
	ctrl->iterations.store(opt.iterations, std::memory_order_release);
	ctrl->stop.store(0, std::memory_order_release);
	ctrl->seq.store(0, std::memory_order_release);
	ctrl->ack.store(0, std::memory_order_release);
	ctrl->ready_active.store(1, std::memory_order_release);
	ctrl->ready_passive.store(0, std::memory_order_release);

	FlushCtrl(ctrl);

	std::cout << "start active" << std::endl;
	DumpState(ctrl);

	if (!WaitEqual(ctrl, ctrl->ready_passive, 1, opt.wait_timeout_ms)) {
		std::cerr << "passive not ready" << std::endl;
		DumpState(ctrl);
		return false;
	}

	std::cout << "passive ready" << std::endl;

	return ActiveLatency(ctrl, data, opt);
}

bool ParseArgs(int argc, char** argv, Options& opt) {
	for (int i = 1; i < argc; ++i) {
		std::string arg(argv[i]);
		auto eat = [&](const std::string& prefix, std::string& out) {
			if (arg.rfind(prefix, 0) == 0) { out = arg.substr(prefix.size()); return true; }
			return false;
		};
		if (eat("--role=", opt.role)) continue;
		if (eat("--dev=", opt.dev_path)) continue;

		std::string val;
		if (eat("--offset=", val)) { if (!ParseUint(val, opt.offset)) return false; continue; }
		if (eat("--map-bytes=", val)) { if (!ParseSize(val, opt.map_bytes)) return false; continue; }
		if (eat("--data-bytes=", val)) { if (!ParseSize(val, opt.data_bytes)) return false; continue; }
		if (eat("--iterations=", val)) { if (!ParseUint(val, opt.iterations)) return false; continue; }
		if (eat("--wait-ms=", val)) { opt.wait_timeout_ms = std::atoi(val.c_str()); continue; }
		if (eat("--gpu-id=", val)) { opt.gpu_id = std::atoi(val.c_str()); continue; }
		if (arg == "--oneshot") { opt.oneshot = true; continue; }
		Usage(argv[0]);
		return false;
	}
	return true;
}

}  // namespace

int main(int argc, char** argv) {
	std::signal(SIGINT, SigHandler);
	std::signal(SIGTERM, SigHandler);

	Options opt;
	if (!ParseArgs(argc, argv, opt)) return 1;

	if (opt.map_bytes < kControlBytes + opt.data_bytes) {
		std::cerr << "map-bytes too small for requested data region" << std::endl;
		return 1;
	}

	int fd = ::open(opt.dev_path.c_str(), O_RDWR | O_SYNC);
	if (fd < 0) {
		std::perror("open dev dax");
		return 1;
	}

	void* addr = ::mmap(nullptr, opt.map_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, opt.offset);
	if (addr == MAP_FAILED) {
		std::perror("mmap");
		::close(fd);
		return 1;
	}
	std::cout << "mmap addr=" << addr << " fd=" << fd << " size=" << opt.map_bytes << std::endl;

	auto* ctrl = reinterpret_cast<ControlBlock*>(addr);
	auto* data = reinterpret_cast<uint8_t*>(addr) + kControlBytes;

	bool ok = true;
	if (opt.role == "active") {
		ctrl->magic.store(kMagic, std::memory_order_release);
		ctrl->ready_active.store(0, std::memory_order_release);
		ctrl->ready_passive.store(0, std::memory_order_release);
		ctrl->stop.store(0, std::memory_order_release);
		ctrl->seq.store(0, std::memory_order_release);
		ctrl->ack.store(0, std::memory_order_release);
		FlushCtrl(ctrl);

		ok = StartActive(ctrl, data, opt);
	} else if (opt.role == "passive") {
		std::cout << "start passive" << std::endl;
		FlushCtrl(ctrl);
		uint64_t magic_seen = ctrl->magic.load(std::memory_order_acquire);
		if (magic_seen != kMagic) {
			std::cerr << "magic mismatch; likely not the same control block" << std::endl;
			ok = false;
		} else {
			PassiveLoop(ctrl, data, opt);
		}
	} else {
		std::cerr << "invalid role" << std::endl;
		ok = false;
	}

	::munmap(addr, opt.map_bytes);
	::close(fd);
	return ok ? 0 : 1;
}
