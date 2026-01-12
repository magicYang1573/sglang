#include <chrono>
#include <cinttypes>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <immintrin.h>

namespace {

constexpr std::size_t kCacheLine = 64;
constexpr std::uint8_t kPattern = 0x5A; // known value for correctness checks

struct Options {
    std::string dev_path = "/dev/dax0.0";
    std::size_t map_bytes = 64ULL * 1024 * 1024; // 64 MiB
    off_t mmap_offset = 0;
    std::size_t read_bytes = 4096;               // 4 KiB block
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
        } else if (key == "--read-bytes") {
            opt.read_bytes = std::stoull(val);
        } else if (key == "--iterations") {
            opt.iterations = std::stoull(val);
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

// slow, 20+ vs for 16KB read
// inline void ClflushRange(const void *addr, std::size_t len) {
//     auto p = reinterpret_cast<std::uintptr_t>(addr);
//     std::uintptr_t end = p + len;
//     p &= ~(static_cast<std::uintptr_t>(kCacheLine) - 1);
//     for (; p < end; p += kCacheLine) {
//         _mm_clflush(reinterpret_cast<const void *>(p));
//     }
//     _mm_sfence();
// }

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

struct Stats {
    double avg_ns = 0.0;
    double min_ns = 0.0;
    double max_ns = 0.0;
    std::size_t bad_reads = 0;
};

Stats MeasureReadLatency(void *cxl_ptr, const Options &opt) {
    std::vector<std::uint8_t> local(opt.read_bytes*opt.iterations, 0);
    auto *base = static_cast<std::uint8_t *>(cxl_ptr);
    const std::size_t max_offset = (opt.map_bytes > opt.read_bytes) ? (opt.map_bytes - opt.read_bytes) : 0;
    std::size_t bad_reads = 0;

    double total = 0.0;
    double min_v = std::numeric_limits<double>::max();
    double max_v = 0.0;

    for (std::size_t i = 0; i < opt.iterations; ++i) {
        // Walk through the mapped region in read_bytes-sized steps to avoid always hitting the same cache lines.
        const std::size_t offset = (max_offset == 0) ? 0 : ((i * opt.read_bytes) % (max_offset + 1));
        void *read_ptr = base + offset;
        void *local_ptr = local.data() + i * opt.read_bytes;

        auto start = std::chrono::high_resolution_clock::now();
        ClflushRange(read_ptr, opt.read_bytes);
        memcpy(local_ptr, read_ptr, opt.read_bytes);
        // auto d = static_cast<__m256i*>((void*)local_ptr);
        // auto s = static_cast<const __m256i*>(read_ptr);
        // for (size_t i = 0; i < opt.read_bytes / 32; ++i) {
        //     // 使用不经过缓存的存储（Non-temporal store）如果不想污染本地缓存
        //     // 或者使用普通 _mm256_load_si256
        //     _mm256_storeu_si256(d + i, _mm256_loadu_si256(s + i));
        // }
        auto end = std::chrono::high_resolution_clock::now();

        // Validate that the read region still holds the expected pattern.
        for (std::size_t b = 0; b < opt.read_bytes; ++b) {
            if (local[b] != kPattern) {
                ++bad_reads;
                break; // stop checking this block on first mismatch to keep overhead low
            }
        }

        double ns = std::chrono::duration<double, std::nano>(end - start).count();
        total += ns;
        if (ns < min_v) min_v = ns;
        if (ns > max_v) max_v = ns;
    }

    Stats s;
    s.avg_ns = total / static_cast<double>(opt.iterations);
    s.min_ns = min_v;
    s.max_ns = max_v;
    s.bad_reads = bad_reads;
    return s;
}

void PrintUsage(const char *prog) {
    std::cerr << "Usage: " << prog << " [--dev=/dev/dax0.0] [--map-bytes=67108864] [--offset=0]"
              << " [--read-bytes=4096] [--iterations=10000]\n";
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

    // Pre-fill the mapped region with a known pattern so correctness can be verified.
    std::memset(base, kPattern, opt.map_bytes);

    Stats stats = MeasureReadLatency(base, opt);

    std::cout << "Device: " << opt.dev_path << "\n";
    std::cout << "Map bytes: " << opt.map_bytes << " (offset " << opt.mmap_offset << ")\n";
    std::cout << "Read bytes: " << opt.read_bytes << "\n";
    std::cout << "Iterations: " << opt.iterations << "\n";
    std::cout << "CXL->DRAM Latency (ns): avg=" << stats.avg_ns << ", min=" << stats.min_ns << ", max=" << stats.max_ns << "\n";
    std::cout << "Bad reads: " << stats.bad_reads << "\n";

    munmap(base, opt.map_bytes);
    return 0;
}
