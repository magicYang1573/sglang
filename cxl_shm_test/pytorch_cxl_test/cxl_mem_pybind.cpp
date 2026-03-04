#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "cxl_mem.hpp"

namespace {

torch::Tensor ensure_contiguous(torch::Tensor t) {
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

    if (!dst.is_contiguous()) {
        dst.copy_(t);
    }
    return dst;
}

// Read num_shards non-contiguous CXL regions into a contiguous CUDA tensor.
//
// dst must be a contiguous CUDA tensor of shape (num_shards, shard_elems).
// cxl_offsets[i] is the byte offset of shard i within the CXL window.
// shard_bytes must equal dst.element_size() * shard_elems.
//
// Internally uses one CPU thread per shard for parallel clflush+memcpy to a
// pinned staging buffer, followed by a single H2D cudaMemcpy.
torch::Tensor cxl_to_tensor_shards(torch::Tensor dst,
                                    const std::vector<std::size_t> &cxl_offsets,
                                    std::size_t shard_bytes) {
    if (!dst.is_cuda()) {
        throw std::runtime_error("cxl_to_tensor_shards: dst must be a CUDA tensor");
    }
    auto t = ensure_contiguous(dst);
    const std::size_t num_shards = cxl_offsets.size();
    if (num_shards == 0) return dst;

    throw_if_false(
        cxl2vram_shards(t.data_ptr(), num_shards, cxl_offsets.data(), shard_bytes),
        "cxl2vram_shards");

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

    m.def("cxl_to_tensor_shards",
          &cxl_to_tensor_shards,
          pybind11::arg("tensor"),
          pybind11::arg("cxl_offsets"),
          pybind11::arg("shard_bytes"),
          "Fill a contiguous CUDA tensor from multiple non-contiguous CXL regions "
          "using parallel CPU threads. cxl_offsets[i] is the byte offset of shard i. "
          "shard_bytes is the size of each shard in bytes.");

    m.def("cxl_barrier_tp",
          &cxl_barrier_tp,
          pybind11::arg("token"),
          pybind11::arg("control_offset"),
          pybind11::arg("rank"),
          pybind11::arg("num_ranks"),
          "CXL barrier for tensor parallelism.");
}

}  // namespace
