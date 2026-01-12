#include <chrono>
#include <cinttypes>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <immintrin.h>

namespace {

constexpr std::size_t kCacheLine = 64;
constexpr std::uint8_t kPattern = 0x5A; // known value to verify writes

struct Options {
    std::string dev_path = "/dev/dax0.0";
    std::size_t map_bytes = 64ULL * 1024 * 1024; // 64 MiB
    off_t mmap_offset = 0;
    std::size_t write_bytes = 4096;              // 4 KiB block
    std::size_t iterations = 10000;
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
    p &= ~(static_cast<std::uintptr_t>(kCacheLine) - 1);
    for (; p < end; p += kCacheLine) {
        _mm_clflush(reinterpret_cast<const void *>(p));
    }
    _mm_sfence();
}

// Non-temporal copy from DRAM to CXL using streaming stores
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

struct Stats {
    double avg_ns = 0.0;
    double min_ns = 0.0;
    double max_ns = 0.0;
    std::size_t bad_writes = 0;
};

Stats MeasureWriteLatency(void *cxl_ptr, const Options &opt) {
    void *src = nullptr;
    if (posix_memalign(&src, kCacheLine, opt.write_bytes*opt.iterations) != 0) {
        std::cerr << "posix_memalign failed\n";
        return {};
    }
    std::memset(src, kPattern, opt.write_bytes*opt.iterations);
    double total = 0.0;
    double min_v = std::numeric_limits<double>::max();
    double max_v = 0.0;
    std::size_t bad_writes = 0;

    auto *base = static_cast<std::uint8_t*>(cxl_ptr);
    std::size_t span = opt.map_bytes > opt.write_bytes ? opt.map_bytes - opt.write_bytes : 0;

    for (std::size_t i = 0; i < opt.iterations; ++i) {
        std::size_t offset = span ? (i * opt.write_bytes) % span : 0;
        void *dst = base + offset;
        void *src_ptr = static_cast<std::uint8_t*>(src) + offset;

        // ClflushRange(dst, opt.write_bytes);
        auto start = std::chrono::high_resolution_clock::now();
        NtStoreCopy(dst, src_ptr, opt.write_bytes);
        auto end = std::chrono::high_resolution_clock::now();

        double ns = std::chrono::duration<double, std::nano>(end - start).count();
        total += ns;
        if (ns < min_v) min_v = ns;
        if (ns > max_v) max_v = ns;
    }

    // Post-run correctness check: verify each written block matches the source pattern.
    for (std::size_t i = 0; i < opt.iterations; ++i) {
        std::size_t offset = span ? (i * opt.write_bytes) % span : 0;
        void *dst = base + offset;
        void *src_ptr = static_cast<std::uint8_t*>(src) + offset;
        if (std::memcmp(dst, src_ptr, opt.write_bytes) != 0) {
            ++bad_writes;
        }
    }

    free(src);

    Stats s;
    s.avg_ns = total / static_cast<double>(opt.iterations);
    s.min_ns = min_v;
    s.max_ns = max_v;
    s.bad_writes = bad_writes;
    return s;
}

void PrintUsage(const char *prog) {
    std::cerr << "Usage: " << prog << " [--dev=/dev/dax0.0] [--map-bytes=67108864] [--offset=0]"
              << " [--write-bytes=4096] [--iterations=10000]\n";
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

    Stats stats = MeasureWriteLatency(base, opt);

    std::cout << "Device: " << opt.dev_path << "\n";
    std::cout << "Map bytes: " << opt.map_bytes << " (offset " << opt.mmap_offset << ")\n";
    std::cout << "Write bytes: " << opt.write_bytes << "\n";
    std::cout << "Iterations: " << opt.iterations << "\n";
    std::cout << "DRAM->CXL write latency (ns): avg=" << stats.avg_ns << ", min=" << stats.min_ns << ", max=" << stats.max_ns << "\n";
    std::cout << "Bad writes: " << stats.bad_writes << "\n";

    munmap(base, opt.map_bytes);
    return 0;
}
