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

namespace {

constexpr std::size_t kCacheLine = 64;
// constexpr int interval_ns = 3000;
constexpr int interval_ns = 0;

struct Options {
    std::string dev_path = "/dev/dax0.0";
    uint64_t map_bytes = 256ULL * 1024 * 1024 * 1024;
    uint64_t data_bytes = 64ULL * 1024 * 1024;
    uint64_t iterations = 100;
    uint64_t offset = 0;
    std::string role = "active";   // active | passive
    bool oneshot = false;
    int wait_timeout_ms = 30000;
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
    _mm_sfence();
    for (size_t off = 0; off < kCtrlFlushBytes; off += 64) {
        _mm_clflushopt(p + off);
    }
    _mm_sfence();
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

inline void NtStoreCopy(void *dst, const void *src, std::size_t len) {
    auto *d8 = static_cast<std::uint8_t*>(dst);
    auto *s8 = static_cast<const std::uint8_t*>(src);

    // Stream 8-byte chunks with _mm_stream_si64; avoid memcpy for the tail.
    std::size_t whole_qwords = len / sizeof(std::uint64_t);
    auto *d64 = reinterpret_cast<std::uint64_t*>(d8);
    auto *s64 = reinterpret_cast<const std::uint64_t*>(s8);
    for (std::size_t i = 0; i < whole_qwords; ++i) {
        _mm_stream_si64(reinterpret_cast<long long*>(d64 + i), static_cast<long long>(s64[i]));
    }

    std::size_t consumed = whole_qwords * sizeof(std::uint64_t);
    std::size_t rem = len - consumed;
    d8 += consumed;
    s8 += consumed;

    // Stream a remaining dword if present.
    if (rem >= sizeof(std::uint32_t)) {
        _mm_stream_si32(reinterpret_cast<int*>(d8), *reinterpret_cast<const int*>(s8));
        d8 += sizeof(std::uint32_t);
        s8 += sizeof(std::uint32_t);
        rem -= sizeof(std::uint32_t);
    }

    // Ensure NT stores reach persistence domain before returning.
    _mm_sfence();
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

void PassiveLatency(ControlBlock* ctrl) {
    uint64_t last = 0;
    uint64_t done = 0;
    uint64_t errors = 0;
    const uint64_t expect_iters = ctrl->iterations.load(std::memory_order_acquire);
    const uint64_t data_len = ctrl->data_bytes.load(std::memory_order_acquire);
    std::vector<uint8_t> local(data_len);

    while (!g_stop.load()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(interval_ns));
        FlushCtrl(ctrl);
        uint64_t seq = ctrl->seq.load(std::memory_order_acquire);
        if (seq != 0 && seq != last) {
            uint8_t expected = static_cast<uint8_t>(seq & 0xFF);
            uint8_t* data = reinterpret_cast<uint8_t*>(ctrl) + kControlBytes;
            
            ClflushRange(data, data_len);  
            std::memcpy(local.data(), data, data_len);

            ctrl->ack.store(seq, std::memory_order_release);
            FlushCtrl(ctrl);

            bool ok = true;
            for (uint64_t i = 0; i < data_len; ++i) {
                // std::cout<<"local["<<i<<"]="<<static_cast<uint32_t>(local[i])<<", expected="<<static_cast<uint32_t>(expected)<<std::endl;
                if (local[i] != expected) {
                    ok = false;
                    break;
                }
            }
            if (!ok) ++errors;

            std::cout << "[passive] iter=" << (done + 1) << " seq=" << seq << (ok ? " ok" : " mismatch") << std::endl;
            
            last = seq;
            ++done;
            if (expect_iters != 0 && done >= expect_iters) break;
        }
        if (ctrl->stop.load(std::memory_order_acquire) != 0) break;
    }

    if (errors != 0) {
        std::cerr << "[passive] data mismatch count=" << errors << std::endl;
    }
}

void PassiveLoop(ControlBlock* ctrl, uint8_t* /*data*/) {
    ctrl->ready_passive.store(1, std::memory_order_release);
    FlushCtrl(ctrl);

    PassiveLatency(ctrl);
}

void ActiveLatency(ControlBlock* ctrl, uint8_t* data, const Options& opt) {
    std::vector<uint8_t> payload(opt.data_bytes);

    auto start = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < opt.iterations && !g_stop.load(); ++i) {
        uint64_t val = i + 1;
        uint8_t pattern = static_cast<uint8_t>(val & 0xFF);
        std::fill(payload.begin(), payload.end(), pattern);

        // std::memcpy(data, payload.data(), payload.size());
        // std::atomic_thread_fence(std::memory_order_seq_cst);
        NtStoreCopy(data, payload.data(), payload.size());

        ctrl->seq.store(val, std::memory_order_release);
        while (!g_stop.load()) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(interval_ns));
            FlushCtrl(ctrl);
            if (ctrl->ack.load(std::memory_order_acquire) == val) break;
            // std::cout<<"waiting ack for seq="<<val<<std::endl;
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
}

bool StartActive(ControlBlock* ctrl, uint8_t* data, const Options& opt) {
    // Reset shared fields for a fresh run
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

    std::cout<<"passive ready"<<std::endl;

    ActiveLatency(ctrl, data, opt);
    return true;
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
        // Active sets up the control block signature and clears fields.
        ctrl->magic.store(kMagic, std::memory_order_release);
        ctrl->ready_active.store(0, std::memory_order_release);
        ctrl->ready_passive.store(0, std::memory_order_release);
        ctrl->stop.store(0, std::memory_order_release);
        ctrl->seq.store(0, std::memory_order_release);
        ctrl->ack.store(0, std::memory_order_release);
        FlushCtrl(ctrl);

        ok = StartActive(ctrl, data, opt);
    } else if (opt.role == "passive") {
        // Passive validates the signature to ensure both map the same region.
        std::cout<<"start passive"<<std::endl;
        FlushCtrl(ctrl);
        uint64_t magic_seen = ctrl->magic.load(std::memory_order_acquire);
        if (magic_seen != kMagic) {
            std::cerr << "magic mismatch; likely not the same control block" << std::endl;
            ok = false;
        } else {
            PassiveLoop(ctrl, data);
        }
    } else {
        std::cerr << "invalid role" << std::endl;
        ok = false;
    }

    ::munmap(addr, opt.map_bytes);
    ::close(fd);
    return ok ? 0 : 1;
}
