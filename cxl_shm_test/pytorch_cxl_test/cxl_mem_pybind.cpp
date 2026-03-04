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

// elem_type encoding: 0=float16, 1=bfloat16, 2=float32
int dtype_to_elem_type(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat16: return 0;
        case torch::kBFloat16: return 1;
        case torch::kFloat32: return 2;
        default:
            throw std::runtime_error("Unsupported dtype for CXL reduce; need float16/bfloat16/float32");
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

// Fused: read multiple shards from CXL, reduce on CPU, write to VRAM tensor.
torch::Tensor py_cxl_read_reduce_shards(
    torch::Tensor dst,
    std::vector<int64_t> offsets)
{
    auto t = ensure_contiguous(dst);
    if (!t.is_cuda()) {
        throw std::runtime_error("cxl_read_reduce_shards requires a CUDA tensor");
    }
    int elem_type = dtype_to_elem_type(t.scalar_type());
    std::size_t shard_bytes = static_cast<std::size_t>(t.nbytes());

    std::vector<std::size_t> src_offsets(offsets.begin(), offsets.end());

    cxl_read_reduce_shards_to_vram(
        t.data_ptr(), shard_bytes,
        src_offsets.data(), static_cast<int>(src_offsets.size()),
        elem_type);

    if (!dst.is_contiguous()) {
        dst.copy_(t);
    }
    return dst;
}

// Fused 1-stage all-reduce: GPU tensor is both input and output.
torch::Tensor py_cxl_allreduce_1stage(
    torch::Tensor inout,
    std::size_t data_offset,
    std::size_t control_offset,
    int rank,
    int num_ranks,
    int32_t token)
{
    auto t = ensure_contiguous(inout);
    if (!t.is_cuda()) {
        throw std::runtime_error("cxl_allreduce_1stage requires a CUDA tensor");
    }
    int elem_type = dtype_to_elem_type(t.scalar_type());
    std::size_t slot_bytes = static_cast<std::size_t>(t.nbytes());

    cxl_allreduce_1stage(
        t.data_ptr(), slot_bytes,
        data_offset, control_offset,
        rank, num_ranks, token, elem_type);

    if (!inout.is_contiguous()) {
        inout.copy_(t);
    }
    return inout;
}

// Fused 2-stage all-reduce (reduce-scatter + all-gather).
torch::Tensor py_cxl_allreduce_2stage(
    torch::Tensor inout,
    std::size_t data_offset,
    std::size_t reduced_base,
    std::size_t control_offset,
    int rank,
    int num_ranks,
    int32_t token_start)
{
    auto t = ensure_contiguous(inout);
    if (!t.is_cuda()) {
        throw std::runtime_error("cxl_allreduce_2stage requires a CUDA tensor");
    }
    int elem_type = dtype_to_elem_type(t.scalar_type());
    std::size_t total_bytes = static_cast<std::size_t>(t.nbytes());

    cxl_allreduce_2stage(
        t.data_ptr(), total_bytes,
        data_offset, reduced_base, control_offset,
        rank, num_ranks, token_start, elem_type);

    if (!inout.is_contiguous()) {
        inout.copy_(t);
    }
    return inout;
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

    m.def("cxl_barrier_tp",
          &cxl_barrier_tp,
          pybind11::arg("token"),
          pybind11::arg("control_offset"),
          pybind11::arg("rank"),
          pybind11::arg("num_ranks"),
          "CXL barrier for tensor parallelism.");

    m.def("cxl_read_reduce_shards",
          &py_cxl_read_reduce_shards,
          pybind11::arg("dst"),
          pybind11::arg("offsets"),
          "Read multiple shards from CXL, reduce (sum) on CPU, write result to dst CUDA tensor.");

    m.def("cxl_allreduce_1stage",
          &py_cxl_allreduce_1stage,
          pybind11::arg("inout"),
          pybind11::arg("data_offset"),
          pybind11::arg("control_offset"),
          pybind11::arg("rank"),
          pybind11::arg("num_ranks"),
          pybind11::arg("token"),
          "Fused 1-stage all-reduce: write to CXL, barrier, read+reduce all slots, write back to GPU.");

    m.def("cxl_allreduce_2stage",
          &py_cxl_allreduce_2stage,
          pybind11::arg("inout"),
          pybind11::arg("data_offset"),
          pybind11::arg("reduced_base"),
          pybind11::arg("control_offset"),
          pybind11::arg("rank"),
          pybind11::arg("num_ranks"),
          pybind11::arg("token_start"),
          "Fused 2-stage all-reduce (reduce-scatter + all-gather).");
}

}  // namespace
