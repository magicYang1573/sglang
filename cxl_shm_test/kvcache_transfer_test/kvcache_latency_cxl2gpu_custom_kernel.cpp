// Custom CUDA kernel path for kvcache copy: copy multiple fixed-size blocks
// from host (CXL/DAX) memory to GPU memory in a single kernel launch.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <cuda_runtime.h>

namespace {

using Value = std::uint16_t; // 2-byte value for kv cache

struct Options {
	std::string dax_path = "/dev/dax0.0";
	size_t tokens = 1;            // number of tokens to read per layer
	size_t token_total = 1;       // total tokens available in shared memory layout
	size_t layers = 1;            // number of layers to copy into GPU layout
	size_t heads = 1;             // head count per layer
	size_t head_dim = 64;         // head dimension
	size_t token_offset = 0;      // token offset in shared memory
	size_t layer_offset = 0;      // layer offset in shared memory
	size_t layers_total = 1;      // total layers in shared memory layout stride
	size_t iterations = 10;       // measured iterations
	size_t warmup = 3;            // warmup iterations (not recorded)
	bool verbose = false;
};

struct Mapping {
	int fd = -1;
	size_t length = 0;
	Value* ptr = nullptr;
};

struct Stats {
	double min_ns = 0.0;
	double max_ns = 0.0;
	double avg_ns = 0.0;
};

struct VerifyResult {
	bool ok = true;
	size_t first_error_block = static_cast<size_t>(-1);
	size_t first_error_offset = static_cast<size_t>(-1);
};

struct BlockLists {
	std::vector<const uint8_t*> src_device_ptrs; // device-space pointers to host-mapped memory
	std::vector<uint8_t*> dst_device_ptrs;        // device-space pointers to GPU memory
	std::vector<const uint8_t*> src_host_ptrs;    // host pointers for cache flushes
	size_t block_bytes = 0;
};

struct DevicePointerLists {
	const uint8_t** d_src = nullptr;
	uint8_t** d_dst = nullptr;
	size_t count = 0;
};

// Basic CUDA error helper that throws on failure.
void check_cuda(cudaError_t err, const char* what) {
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
	}
}

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
			std::cout << "Usage: ./kvcache_latency_cxl2gpu_custom_kernel [options]\n"
				  << "  --dax <path>          DAX device path (default /dev/dax0.0)\n"
				  << "  --tokens <n>          Tokens to read\n"
				  << "  --token-total <n>     Total tokens present in shared layout\n"
				  << "  --layers <n>          Layers to copy into GPU layout\n"
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

Mapping map_dax(const Options& opt, size_t required_bytes) {
	Mapping m;
	m.fd = ::open(opt.dax_path.c_str(), O_RDWR | O_SYNC);
	if (m.fd < 0) throw std::runtime_error("Failed to open DAX device: " + opt.dax_path);

	struct stat st {};
	if (fstat(m.fd, &st) != 0) {
		::close(m.fd);
		throw std::runtime_error("Failed to fstat DAX device");
	}

	size_t page = static_cast<size_t>(::getpagesize());
	m.length = align_up(required_bytes, page);

	void* addr = ::mmap(nullptr, m.length, PROT_READ | PROT_WRITE, MAP_SHARED, m.fd, 0);
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
	for (size_t offset = 0; offset < bytes; offset += 64) {
		_mm_clflushopt(const_cast<char*>(p + offset));
	}
	_mm_sfence();
}

// Kernel: one block per memcpy chunk, threads sweep the chunk.
__global__ void copy_blocks_kernel(const uint8_t* const* src_list, uint8_t* const* dst_list, size_t block_size) {
	const size_t block_idx = static_cast<size_t>(blockIdx.x);
	const uint8_t* src = src_list[block_idx];
	uint8_t* dst = dst_list[block_idx];

	for (size_t i = threadIdx.x; i < block_size; i += blockDim.x) {
		dst[i] = src[i];
	}
}

DevicePointerLists upload_pointer_lists(const BlockLists& lists, cudaStream_t stream) {
	DevicePointerLists dev{};
	dev.count = lists.src_device_ptrs.size();
	if (dev.count == 0) return dev;

	const size_t bytes = dev.count * sizeof(uint8_t*);
	check_cuda(cudaMalloc(&dev.d_src, bytes), "cudaMalloc src list");
	check_cuda(cudaMalloc(&dev.d_dst, bytes), "cudaMalloc dst list");
	check_cuda(cudaMemcpyAsync(dev.d_src, lists.src_device_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream), "memcpy src list");
	check_cuda(cudaMemcpyAsync(dev.d_dst, lists.dst_device_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream), "memcpy dst list");
	check_cuda(cudaStreamSynchronize(stream), "sync pointer upload");
	return dev;
}

void free_pointer_lists(DevicePointerLists& dev) {
	if (dev.d_src) cudaFree(dev.d_src);
	if (dev.d_dst) cudaFree(dev.d_dst);
	dev.d_src = nullptr;
	dev.d_dst = nullptr;
	dev.count = 0;
}

void launch_copy_blocks(const BlockLists& lists, const DevicePointerLists& dev, cudaStream_t stream) {
	if (dev.count == 0) return;
	const int threads = 256;
	dim3 grid(static_cast<unsigned int>(dev.count));
	copy_blocks_kernel<<<grid, threads, 0, stream>>>(dev.d_src, dev.d_dst, lists.block_bytes);
	check_cuda(cudaGetLastError(), "copy_blocks_kernel launch");
}

BlockLists build_block_lists_contiguous(const Options& opt, const Value* host_base, const Value* device_view_hostmem, Value* device_dst) {
	BlockLists lists;
	const size_t token_stride = opt.layers_total * opt.heads * opt.head_dim;
	const size_t layer_stride = opt.heads * opt.head_dim;
	const size_t block_bytes = layer_stride * sizeof(Value);

	lists.block_bytes = block_bytes;
	const size_t total_blocks = opt.layers * opt.tokens;
	lists.src_device_ptrs.reserve(total_blocks);
	lists.dst_device_ptrs.reserve(total_blocks);
	lists.src_host_ptrs.reserve(total_blocks);

	for (size_t l = 0; l < opt.layers; ++l) {
		for (size_t t = 0; t < opt.tokens; ++t) {
			const size_t src_token = opt.token_offset + t;
			const size_t src_layer = opt.layer_offset + l;
			const size_t src_index = (src_token * token_stride) + (src_layer * layer_stride);

			const Value* host_src = host_base + src_index;
			const Value* device_src = device_view_hostmem + src_index; // device pointer alias to host memory
			Value* dst_ptr = device_dst + ((l * opt.tokens + t) * layer_stride);

			lists.src_host_ptrs.push_back(reinterpret_cast<const uint8_t*>(host_src));
			lists.src_device_ptrs.push_back(reinterpret_cast<const uint8_t*>(device_src));
			lists.dst_device_ptrs.push_back(reinterpret_cast<uint8_t*>(dst_ptr));
		}
	}

	return lists;
}

BlockLists build_block_lists_random(const Options& opt, const Value* host_base, const Value* device_view_hostmem, Value* device_dst, const std::vector<size_t>& token_indices_per_layer) {
	if (token_indices_per_layer.size() != opt.layers * opt.tokens) {
		throw std::runtime_error("token_indices_per_layer size mismatch");
	}

	BlockLists lists;
	const size_t token_stride = opt.layers_total * opt.heads * opt.head_dim;
	const size_t layer_stride = opt.heads * opt.head_dim;
	const size_t block_bytes = layer_stride * sizeof(Value);

	lists.block_bytes = block_bytes;
	const size_t total_blocks = opt.layers * opt.tokens;
	lists.src_device_ptrs.reserve(total_blocks);
	lists.dst_device_ptrs.reserve(total_blocks);
	lists.src_host_ptrs.reserve(total_blocks);

	for (size_t l = 0; l < opt.layers; ++l) {
		for (size_t t = 0; t < opt.tokens; ++t) {
			const size_t src_token = token_indices_per_layer[l * opt.tokens + t];
			if (src_token >= opt.token_total) {
				throw std::runtime_error("token index out of range for token_total");
			}
			const size_t src_layer = opt.layer_offset + l;
			const size_t src_index = (src_token * token_stride) + (src_layer * layer_stride);

			const Value* host_src = host_base + src_index;
			const Value* device_src = device_view_hostmem + src_index;
			Value* dst_ptr = device_dst + ((l * opt.tokens + t) * layer_stride);

			lists.src_host_ptrs.push_back(reinterpret_cast<const uint8_t*>(host_src));
			lists.src_device_ptrs.push_back(reinterpret_cast<const uint8_t*>(device_src));
			lists.dst_device_ptrs.push_back(reinterpret_cast<uint8_t*>(dst_ptr));
		}
	}

	return lists;
}

void flush_blocks(const BlockLists& lists) {
	for (const uint8_t* src : lists.src_host_ptrs) {
		clflush_range(src, lists.block_bytes);
	}
}

// Copy destination blocks back to host and compare with source blocks for correctness.
VerifyResult verify_blocks(const BlockLists& lists, cudaStream_t stream) {
	VerifyResult vr{};
	const size_t total_blocks = lists.src_host_ptrs.size();
	if (total_blocks == 0) return vr;

	std::vector<uint8_t> host_dst(total_blocks * lists.block_bytes);

	for (size_t i = 0; i < total_blocks; ++i) {
		uint8_t* dst_host = host_dst.data() + i * lists.block_bytes;
		check_cuda(cudaMemcpyAsync(dst_host, lists.dst_device_ptrs[i], lists.block_bytes, cudaMemcpyDeviceToHost, stream), "verify D2H");
	}
	check_cuda(cudaStreamSynchronize(stream), "verify sync");

	for (size_t i = 0; i < total_blocks; ++i) {
		const uint8_t* expect = lists.src_host_ptrs[i];
		const uint8_t* got = host_dst.data() + i * lists.block_bytes;
		if (std::memcmp(expect, got, lists.block_bytes) != 0) {
			vr.ok = false;
			vr.first_error_block = i;
			// find first differing offset to help debug
			for (size_t off = 0; off < lists.block_bytes; ++off) {
				if (expect[off] != got[off]) {
					vr.first_error_offset = off;
					break;
				}
			}
			break;
		}
	}
	return vr;
}

Stats benchmark_with_kernel(const BlockLists& lists, cudaStream_t stream, size_t iterations, size_t warmup) {
	Stats st{};
	if (lists.src_device_ptrs.empty()) return st;

	DevicePointerLists dev_lists = upload_pointer_lists(lists, stream);
	std::vector<double> samples;
	samples.reserve(iterations);

	for (size_t i = 0; i < warmup + iterations; ++i) {
		flush_blocks(lists);
		auto t0 = std::chrono::steady_clock::now();
		launch_copy_blocks(lists, dev_lists, stream);
		check_cuda(cudaStreamSynchronize(stream), "stream sync kernel");
		auto t1 = std::chrono::steady_clock::now();
		double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
		if (i >= warmup) samples.push_back(ns);
	}

	free_pointer_lists(dev_lists);

	if (!samples.empty()) {
		st.min_ns = *std::min_element(samples.begin(), samples.end());
		st.max_ns = *std::max_element(samples.begin(), samples.end());
		double sum = 0.0;
		for (double v : samples) sum += v;
		st.avg_ns = sum / samples.size();
	}
	return st;
}

Stats benchmark_contiguous(const Options& opt, const Mapping& map, const Value* device_view_hostmem, Value* device_dst, cudaStream_t stream, BlockLists& out_lists) {
	out_lists = build_block_lists_contiguous(opt, map.ptr, device_view_hostmem, device_dst);
	return benchmark_with_kernel(out_lists, stream, opt.iterations, opt.warmup);
}

Stats benchmark_random(const Options& opt, const Mapping& map, const Value* device_view_hostmem, Value* device_dst, cudaStream_t stream, BlockLists& out_lists, std::vector<size_t>& token_indices) {
	token_indices.resize(opt.layers * opt.tokens);
	std::mt19937_64 rng(42);
	std::uniform_int_distribution<size_t> dist(0, opt.token_total - 1);
	for (size_t l = 0; l < opt.layers; ++l) {
		for (size_t t = 0; t < opt.tokens; ++t) {
			token_indices[l * opt.tokens + t] = dist(rng);
		}
	}

	out_lists = build_block_lists_random(opt, map.ptr, device_view_hostmem, device_dst, token_indices);
	return benchmark_with_kernel(out_lists, stream, opt.iterations, opt.warmup);
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

		Value* map_device_ptr = nullptr;
		check_cuda(cudaHostRegister(reinterpret_cast<void*>(map.ptr), bytes_needed, cudaHostRegisterPortable | cudaHostRegisterMapped), "cudaHostRegister");
		check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&map_device_ptr), reinterpret_cast<void*>(map.ptr), 0), "cudaHostGetDevicePointer");

		const size_t dst_elems = opt.layers * opt.tokens * opt.heads * opt.head_dim;
		Value* device_dst = nullptr;
		check_cuda(cudaMalloc(&device_dst, dst_elems * sizeof(Value)), "cudaMalloc dst");

		cudaStream_t stream{};
		check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

		BlockLists contig_lists{};
		Stats st_contig = benchmark_contiguous(opt, map, map_device_ptr, device_dst, stream, contig_lists);
		VerifyResult verify_contig = verify_blocks(contig_lists, stream);

		BlockLists random_lists{};
		std::vector<size_t> random_indices;
		Stats st_random = benchmark_random(opt, map, map_device_ptr, device_dst, stream, random_lists, random_indices);
		VerifyResult verify_random = verify_blocks(random_lists, stream);

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

		if (!verify_contig.ok) {
			std::cerr << "Contiguous verification FAILED at block " << verify_contig.first_error_block
				  << " offset " << verify_contig.first_error_offset << "\n";
		} else {
			std::cout << "Contiguous verification PASSED\n";
		}

		if (!verify_random.ok) {
			std::cerr << "Random verification FAILED at block " << verify_random.first_error_block
				  << " offset " << verify_random.first_error_offset << "\n";
		} else {
			std::cout << "Random verification PASSED\n";
		}

		check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
		check_cuda(cudaFree(device_dst), "cudaFree dst");
		cudaHostUnregister(reinterpret_cast<void*>(map.ptr));
		unmap(map);
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "Error: " << ex.what() << "\n";
		return 1;
	}
}

