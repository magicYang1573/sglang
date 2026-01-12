#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <emmintrin.h>
#include <immintrin.h>

namespace {
using Value = std::uint16_t; // 2-byte value for kv cache
const int num_threads = 64;

struct Options {
    std::string dax_path = "/dev/dax0.0";
    size_t tokens = 1;            // number of tokens to read
    size_t token_total = 1;       // total tokens available in shared memory layout
    size_t layers = 1;            // number of layers to copy into DRAM layout
    size_t heads = 1;             // head count per layer
    size_t head_dim = 64;         // head dimension
    size_t token_offset = 0;      // token offset in shared memory
    size_t layer_offset = 0;      // layer offset in shared memory
    size_t layers_total = 1;      // total layers in shared memory layout stride
    size_t iterations = 10;       // measured iterations
    size_t warmup = 3;            // warmup iterations (not recorded)
    bool verbose = false;
};

size_t align_up(size_t value, size_t alignment) {
    const size_t mask = alignment - 1;
    return (value + mask) & ~mask;
}

Options parse_args(int argc, char** argv) {
    Options opt;
    bool token_total_set = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return argv[++i];
        };

        if (arg == "--dax" || arg == "--dev") {
            opt.dax_path = need_value(arg);
        } else if (arg == "--tokens") {
            opt.tokens = std::stoull(need_value(arg));
        } else if (arg == "--token-total") {
            opt.token_total = std::stoull(need_value(arg));
            token_total_set = true;
        } else if (arg == "--layers") {
            opt.layers = std::stoull(need_value(arg));
        } else if (arg == "--layers-total") {
            opt.layers_total = std::stoull(need_value(arg));
        } else if (arg == "--heads") {
            opt.heads = std::stoull(need_value(arg));
        } else if (arg == "--head-dim") {
            opt.head_dim = std::stoull(need_value(arg));
        } else if (arg == "--token-offset") {
            opt.token_offset = std::stoull(need_value(arg));
        } else if (arg == "--layer-offset") {
            opt.layer_offset = std::stoull(need_value(arg));
        } else if (arg == "--iterations") {
            opt.iterations = std::stoull(need_value(arg));
        } else if (arg == "--warmup") {
            opt.warmup = std::stoull(need_value(arg));
        } else if (arg == "-v" || arg == "--verbose") {
            opt.verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: ./kvcache_latency [options]\n"
                      << "  --dax <path>          DAX device path (default /dev/dax0.0)\n"
                      << "  --tokens <n>          Tokens to read\n"
                      << "  --token-total <n>     Total tokens present in shared layout\n"
                      << "  --layers <n>          Layers to copy into DRAM layout\n"
                      << "  --layers-total <n>    Total layers in shared layout stride (required)\n"
                      << "  --heads <n>           Heads per layer\n"
                      << "  --head-dim <n>        Head dimension\n"
                      << "  --token-offset <n>    Token offset inside shared layout\n"
                      << "  --layer-offset <n>    Layer offset inside shared layout\n"
                      << "  --iterations <n>      Timed iterations\n"
                      << "  --warmup <n>          Warmup iterations (not recorded)\n"
                      << "  -v, --verbose         Print extra info\n"
                      << "  -h, --help            Show this message\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (!token_total_set) {
        opt.token_total = opt.token_offset + opt.tokens;
    }
    if (opt.layers_total == 0) throw std::runtime_error("--layers-total must be > 0");
    if (opt.layers_total < opt.layer_offset + opt.layers) {
        throw std::runtime_error("layers_total is smaller than layer_offset + layers");
    }
    if (opt.token_total == 0) throw std::runtime_error("--token-total must be > 0");
    if (opt.token_offset + opt.tokens > opt.token_total) {
        throw std::runtime_error("token_total is smaller than token_offset + tokens");
    }
    return opt;
}

struct Mapping {
    int fd = -1;
    size_t length = 0;
    Value* ptr = nullptr;
};

Mapping map_dax(const Options& opt, size_t required_bytes) {
    Mapping m;
    m.fd = ::open(opt.dax_path.c_str(), O_RDONLY);
    if (m.fd < 0) throw std::runtime_error("Failed to open DAX device: " + opt.dax_path);

    struct stat st {};
    if (fstat(m.fd, &st) != 0) {
        ::close(m.fd);
        throw std::runtime_error("Failed to fstat DAX device");
    }

    size_t page = static_cast<size_t>(::getpagesize());
    m.length = align_up(required_bytes, page);

    void* addr = ::mmap(nullptr, m.length, PROT_READ, MAP_SHARED, m.fd, 0);
    if (addr == MAP_FAILED) {
        ::close(m.fd);
        throw std::runtime_error("mmap failed");
    }
    m.ptr = static_cast<Value*>(addr);
    return m;
}

void unmap(Mapping& m) {
    if (m.ptr) ::munmap(m.ptr, m.length);
    if (m.fd >= 0) ::close(m.fd);
    m.ptr = nullptr;
    m.fd = -1;
    m.length = 0;
}

size_t required_bytes(const Options& opt) {
    const size_t token_stride = opt.layers_total * opt.heads * opt.head_dim;
    const size_t span_tokens = opt.token_total;
    return span_tokens * token_stride * sizeof(Value);
}
__attribute__((target("clflushopt")))
inline void clflush_range(const void* ptr, size_t bytes) {
    const char* p = static_cast<const char*>(ptr);
    // Flush each cache line; step by 64 bytes (typical x86 line size).
    for (size_t offset = 0; offset < bytes; offset += 64) {
        _mm_clflushopt(const_cast<char*>(p + offset));
    }
    _mm_sfence();
}

void copy_to_dram_contiguous(const Options& opt, const Value* src, Value* dst) {
    const size_t token_stride = opt.layers_total * opt.heads * opt.head_dim;
    const size_t layer_stride = opt.heads * opt.head_dim;
    const size_t head_stride = opt.head_dim;
    
    // clflush_range(src, opt.tokens * token_stride * sizeof(Value));

    #pragma omp parallel for collapse(2) schedule(static) num_threads(num_threads)
    for (size_t l = 0; l < opt.layers; ++l) {
        for (size_t t = 0; t < opt.tokens; ++t) {
            const size_t src_token = opt.token_offset + t;
            const size_t src_layer = opt.layer_offset + l;
            const Value* src_ptr = src + (src_token * token_stride) + (src_layer * layer_stride);
            Value* dst_ptr = dst + ((l * opt.tokens + t) * layer_stride);
            clflush_range(src_ptr, layer_stride * sizeof(Value));
            std::memcpy(dst_ptr, src_ptr, layer_stride * sizeof(Value));
            // auto start = std::chrono::high_resolution_clock::now();
            // auto end = std::chrono::high_resolution_clock::now();
            // double ns = std::chrono::duration<double, std::nano>(end - start).count();
            // std::cout<<"copy: " << ns << " ns" << std::endl;
            // for (size_t h = 0; h < opt.heads; ++h) {
            //     const Value* s = src_ptr + h * head_stride;
            //     Value* d = dst_ptr + h * head_stride;
            //     std::memcpy(d, s, head_stride * sizeof(Value));
            // }
        }
    }
}

void copy_to_dram_random(const Options& opt, const Value* src, Value* dst, const std::vector<size_t>& token_indices_per_layer) {
    const size_t token_stride = opt.layers_total * opt.heads * opt.head_dim;
    const size_t layer_stride = opt.heads * opt.head_dim;
    const size_t head_stride = opt.head_dim;

    if (token_indices_per_layer.size() != opt.layers * opt.tokens) {
        throw std::runtime_error("token_indices_per_layer size mismatch");
    }

    #pragma omp parallel for collapse(2) schedule(static) num_threads(num_threads)
    for (size_t l = 0; l < opt.layers; ++l) {
        for (size_t t = 0; t < opt.tokens; ++t) {
            const size_t src_token = token_indices_per_layer[l * opt.tokens + t];
            if (src_token >= opt.token_total) {
                throw std::runtime_error("token index out of range for token_total");
            }
            const size_t src_layer = opt.layer_offset + l;
            const Value* src_ptr = src + (src_token * token_stride) + (src_layer * layer_stride);
            Value* dst_ptr = dst + ((l * opt.tokens + t) * layer_stride);
            clflush_range(src_ptr, layer_stride * sizeof(Value));
            std::memcpy(dst_ptr, src_ptr, layer_stride * sizeof(Value));
            // for (size_t h = 0; h < opt.heads; ++h) {
            //     const Value* s = src_ptr + h * head_stride;
            //     Value* d = dst_ptr + h * head_stride;
            //     clflush_range(s, head_stride * sizeof(Value));
            //     std::memcpy(d, s, head_stride * sizeof(Value));
            // }
        }
    }
}

struct Stats {
    double min_ns = 0.0;
    double max_ns = 0.0;
    double avg_ns = 0.0;
};

Stats benchmark_contiguous(const Options& opt, const Value* src, Value* dst) {
    std::vector<double> samples;
    samples.reserve(opt.iterations);
    volatile Value sink = 0; // prevents optimization of dst

    for (size_t i = 0; i < opt.warmup + opt.iterations; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        copy_to_dram_contiguous(opt, src, dst);
        auto t1 = std::chrono::steady_clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        sink += dst[(i % (opt.layers * opt.tokens * opt.heads)) * opt.head_dim];
        if (i >= opt.warmup) samples.push_back(ns);
    }

    Stats st{};
    if (!samples.empty()) {
        st.min_ns = *std::min_element(samples.begin(), samples.end());
        st.max_ns = *std::max_element(samples.begin(), samples.end());
        double sum = 0.0;
        for (double v : samples) sum += v;
        st.avg_ns = sum / samples.size();
    }
    (void)sink;
    return st;
}

Stats benchmark_random(const Options& opt, const Value* src, Value* dst) {
    std::vector<double> samples;
    samples.reserve(opt.iterations);
    volatile Value sink = 0; // prevents optimization of dst

    // Pre-generate per-layer token indices once to keep timing focused on copy.
    std::vector<size_t> token_indices(opt.layers * opt.tokens);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<size_t> dist(0, opt.token_total - 1);
    for (size_t l = 0; l < opt.layers; ++l) {
        for (size_t t = 0; t < opt.tokens; ++t) {
            token_indices[l * opt.tokens + t] = dist(rng);
        }
    }

    for (size_t i = 0; i < opt.warmup + opt.iterations; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        copy_to_dram_random(opt, src, dst, token_indices);
        auto t1 = std::chrono::steady_clock::now();
        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        sink += dst[(i % (opt.layers * opt.tokens * opt.heads)) * opt.head_dim];
        if (i >= opt.warmup) samples.push_back(ns);
    }

    Stats st{};
    if (!samples.empty()) {
        st.min_ns = *std::min_element(samples.begin(), samples.end());
        st.max_ns = *std::max_element(samples.begin(), samples.end());
        double sum = 0.0;
        for (double v : samples) sum += v;
        st.avg_ns = sum / samples.size();
    }
    (void)sink;
    return st;
}

} // namespace

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);
        const size_t bytes_needed = required_bytes(opt);

        if (opt.verbose) {
            std::cout << "Config:\n";
            std::cout << "  dax path       : " << opt.dax_path << "\n";
            std::cout << "  tokens         : " << opt.tokens << " (offset " << opt.token_offset << ")\n";
            std::cout << "  token total    : " << opt.token_total << "\n";
            std::cout << "  layers         : " << opt.layers << " (offset " << opt.layer_offset << ")\n";
            std::cout << "  layers total   : " << opt.layers_total << "\n";
            std::cout << "  heads          : " << opt.heads << "\n";
            std::cout << "  head dim       : " << opt.head_dim << "\n";
            std::cout << "  warmup/iters   : " << opt.warmup << "/" << opt.iterations << "\n";
            std::cout << "  required bytes : " << bytes_needed << "\n";
        }

        Mapping map = map_dax(opt, bytes_needed);
        std::vector<Value> dram(opt.layers * opt.tokens * opt.heads * opt.head_dim, 0);

        // contiguous tokens: [token_offset, token_offset + tokens)
        Stats st_contig = benchmark_contiguous(opt, map.ptr, dram.data());

        // random tokens: each layer has its own token selection generated inside benchmark
        Stats st_random = benchmark_random(opt, map.ptr, dram.data());

        auto print_stats = [&](const char* label, const Stats& st) {
            double bytes_copied = static_cast<double>(opt.layers) * opt.tokens * opt.heads * opt.head_dim * sizeof(Value);
            double gb_per_s = (bytes_copied / st.avg_ns) * 1e9 / (1024.0 * 1024.0 * 1024.0);
            std::cout << label << " Latency (ns): min=" << st.min_ns
                      << ", avg=" << st.avg_ns
                      << ", max=" << st.max_ns << "\n";
            std::cout << label << " Throughput: " << gb_per_s << " GiB/s (avg)\n";
        };

        print_stats("Contiguous", st_contig);
        print_stats("Random", st_random);

        unmap(map);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
