#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

#include <omp.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "cxl_mem.hpp"

namespace {

// Ensure we always work on a contiguous backing buffer to make size/ptr simple.
at::Tensor ensure_contiguous(at::Tensor t) {
    return t.is_contiguous() ? t : t.contiguous();
}

void throw_if_false(bool ok, const std::string &what) {
    if (!ok) {
        throw std::runtime_error(what + " failed");
    }
}

void tensor_to_cxl(torch::Tensor src, std::size_t offset) {
    auto t = ensure_contiguous(src);
    const std::size_t bytes = static_cast<std::size_t>(t.nbytes());

    if (t.is_cuda()) {
        throw_if_false(vram2cxl(t.data_ptr(), bytes, offset), "vram2cxl");
        // throw_if_false(vram2cxl_async(t.data_ptr(), bytes, offset), "vram2cxl_async");
    } else {
        throw_if_false(dram2cxl(t.data_ptr(), bytes, offset), "dram2cxl");
    }
}

// Copy into an existing tensor (CPU or CUDA) from the mapped CXL window.
torch::Tensor cxl_to_tensor(torch::Tensor dst, std::size_t offset) {
    auto t = ensure_contiguous(dst);
    const std::size_t bytes = static_cast<std::size_t>(t.nbytes());

    if (t.is_cuda()) {
        throw_if_false(cxl2vram(t.data_ptr(), bytes, offset), "cxl2vram");
    } else {
        throw_if_false(cxl2dram(t.data_ptr(), bytes, offset), "cxl2dram");
    }

    // If we materialized a contiguous view, copy back into the original tensor.
    if (!dst.is_contiguous()) {
        dst.copy_(t);
    }
    return dst;
}

torch::Tensor cxl_to_tensor_noflush(torch::Tensor dst, std::size_t offset) {
    auto t = ensure_contiguous(dst);
    const std::size_t bytes = static_cast<std::size_t>(t.nbytes());

    if (t.is_cuda()) {
        throw_if_false(cxl2vram_noflush(t.data_ptr(), bytes, offset), "cxl2vram_noflush");
    } else {
        throw_if_false(cxl2dram_noflush(t.data_ptr(), bytes, offset), "cxl2dram_noflush");
    }

    if (!dst.is_contiguous()) {
        dst.copy_(t);
    }
    return dst;
}

std::vector<torch::Tensor> cxl_to_tensor_noflush_parallel(std::vector<torch::Tensor> dst_list,
                                                          at::Tensor offset_list) {

    auto offsets = ensure_contiguous(offset_list);
    if (offsets.is_cuda()) {
        offsets = offsets.to(at::kCPU);
    }
    if (offsets.scalar_type() != at::kLong) {
        throw std::runtime_error("cxl_to_tensor_noflush_parallel: offsets must be int64 CPU tensor");
    }
    const std::size_t count = static_cast<std::size_t>(offsets.numel());
    if (count != dst_list.size()) {
        throw std::runtime_error("cxl_to_tensor_noflush_parallel: offsets size must match tensors size");
    }

    std::vector<std::size_t> offsets_size_t(count);
    const auto *offsets_ptr = offsets.data_ptr<int64_t>();
    for (std::size_t i = 0; i < count; ++i) {
        offsets_size_t[i] = static_cast<std::size_t>(offsets_ptr[i]);
    }

    std::vector<torch::Tensor> work_list;
    std::vector<void *> dst_ptrs;
    std::vector<std::size_t> bytes_list;
    work_list.reserve(dst_list.size());
    dst_ptrs.reserve(dst_list.size());
    bytes_list.reserve(dst_list.size());

    bool any_cuda = false;
    bool any_cpu = false;
    for (const auto &dst : dst_list) {
        auto t = ensure_contiguous(dst);
        work_list.push_back(t);
        dst_ptrs.push_back(t.data_ptr());
        bytes_list.push_back(static_cast<std::size_t>(t.nbytes()));
        any_cuda = any_cuda || t.is_cuda();
        any_cpu = any_cpu || !t.is_cuda();
    }

    if (any_cuda && any_cpu) {
        throw std::runtime_error("cxl_to_tensor_noflush_parallel: mixed CPU/CUDA tensors are not supported");
    }

    if (any_cuda) {
        for (std::size_t i = 0; i < count; ++i) {
            throw_if_false(cxl2vram_noflush(dst_ptrs[i],
                                            bytes_list[i],
                                            offsets_size_t[i]),
                           "cxl2vram_noflush");
        }
    } else {
        auto start_time = std::chrono::high_resolution_clock::now();
        throw_if_false(cxl2dram_noflush_parallel(dst_ptrs.data(),
                                                 bytes_list.data(),
                                                 offsets_size_t.data(),
                                                 dst_ptrs.size(),
                                                 64),
                       "cxl2dram_noflush_parallel");
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        std::cout << "CXL2DRAM Copy Time: " << duration << " us" << std::endl;
    }

    for (std::size_t i = 0; i < dst_list.size(); ++i) {
        if (!dst_list[i].is_contiguous()) {
            dst_list[i].copy_(work_list[i]);
        }
    }
    return dst_list;
}

torch::Tensor cxl_to_tensor_noflush_parallel_contiguous_dst(at::Tensor dst,
                                                            std::size_t bytes_per_segment,
                                                            at::Tensor offset_list) {
    auto t = ensure_contiguous(dst);
    auto offsets = ensure_contiguous(offset_list);
    if (offsets.is_cuda()) {
        offsets = offsets.to(at::kCPU);
    }
    if (offsets.scalar_type() != at::kLong) {
        throw std::runtime_error("cxl_to_tensor_noflush_parallel_contiguous_dst: offsets must be int64 CPU tensor");
    }
    const std::size_t count = static_cast<std::size_t>(offsets.numel());
    const std::size_t total_bytes = bytes_per_segment * count;
    if (total_bytes > static_cast<std::size_t>(t.nbytes())) {
        throw std::runtime_error("cxl_to_tensor_noflush_parallel_contiguous_dst: dst too small");
    }

    std::vector<std::size_t> offsets_size_t(count);
    const auto *offsets_ptr = offsets.data_ptr<int64_t>();
    for (std::size_t i = 0; i < count; ++i) {
        offsets_size_t[i] = static_cast<std::size_t>(offsets_ptr[i]);
    }

    if (t.is_cuda()) {
        auto *base = static_cast<std::uint8_t *>(t.data_ptr());
        for (std::size_t i = 0; i < count; ++i) {
            throw_if_false(cxl2vram_noflush(base + i * bytes_per_segment,
                                            bytes_per_segment,
                                            offsets_size_t[i]),
                           "cxl2vram_noflush");
        }
    } else {
        throw_if_false(cxl2dram_noflush_parallel_contiguous_dst(t.data_ptr(),
                                                                bytes_per_segment,
                                                                offsets_size_t.data(),
                                                                count),
                       "cxl2dram_noflush_parallel_contiguous_dst");
    }

    if (!dst.is_contiguous()) {
        dst.copy_(t);
    }

    return dst;
}
PYBIND11_MODULE(cxl_mem_ext, m) {
    m.doc() = "PyTorch bindings for CXL shared memory helpers";

    m.def("cxl_init",
          &cxl_init,
          pybind11::arg("dev_path") = std::string("/dev/dax0.0"),
          pybind11::arg("map_bytes") = 64ULL * 1024 * 1024,
          pybind11::arg("offset") = 0,
          pybind11::arg("register_cuda") = false,
          pybind11::arg("gpu_id") = 0,
          "Map the CXL DAX device; optionally register with CUDA.");

    m.def("cxl_close", &cxl_close, "Unmap and unregister the mapped CXL window.");

    m.def("tensor_to_cxl",
          &tensor_to_cxl,
          pybind11::arg("tensor"),
          pybind11::arg("offset") = 0,
          "Copy a tensor (CPU or CUDA) into the mapped CXL window at the given offset.");

    m.def("cxl_to_tensor",
          &cxl_to_tensor,
          pybind11::arg("tensor"),
          pybind11::arg("offset") = 0,
          "Fill a tensor (CPU or CUDA) from the mapped CXL window at the given offset.");

        m.def("cxl_to_tensor_noflush",
            &cxl_to_tensor_noflush,
            pybind11::arg("tensor"),
            pybind11::arg("offset") = 0,
            "Fill a tensor (CPU or CUDA) from the mapped CXL window at the given offset without flushing.");

        m.def("cxl_to_tensor_noflush_parallel",
            &cxl_to_tensor_noflush_parallel,
            pybind11::arg("tensors"),
            pybind11::arg("offsets"),
            "Fill tensors (CPU or CUDA) from the mapped CXL window at given offsets without flushing."
            " CPU path uses OpenMP with 64 threads; CUDA path is serial.");

        m.def("cxl_to_tensor_noflush_parallel_contiguous_dst",
            &cxl_to_tensor_noflush_parallel_contiguous_dst,
            pybind11::arg("tensor"),
            pybind11::arg("bytes_per_segment"),
            pybind11::arg("offsets"),
            "Fill a contiguous tensor from the mapped CXL window at given offsets with fixed segment size.");

    m.def("cxl_barrier_tp",
          &cxl_barrier_tp,
          pybind11::arg("token"),
          pybind11::arg("control_offset"),
          pybind11::arg("rank"),
          pybind11::arg("num_ranks"),
          "CXL barrier for tensor parallelism.");
}

}  // namespace
