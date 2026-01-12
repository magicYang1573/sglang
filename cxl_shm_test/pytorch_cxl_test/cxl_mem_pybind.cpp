#include <stdexcept>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "cxl_mem.hpp"

namespace {

// Ensure we always work on a contiguous backing buffer to make size/ptr simple.
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
        // throw_if_false(vram2cxl(t.data_ptr(), bytes, offset), "vram2cxl");
        throw_if_false(vram2cxl_async(t.data_ptr(), bytes, offset), "vram2cxl_async");
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
}

}  // namespace
