#include <stdexcept>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "cxl_mem.hpp"

namespace {

torch::Tensor ensure_contiguous(torch::Tensor t) {
    return t.is_contiguous() ? t : t.contiguous();
}

void throw_if_false(bool ok, const std::string &what) {
    if (!ok) throw std::runtime_error(what + " failed");
}

// -----------------------------------------------------------------------
// Basic copy helpers (kept for non-fused / legacy use)
// -----------------------------------------------------------------------

void tensor_to_cxl(torch::Tensor src, std::size_t offset) {
    auto t = ensure_contiguous(src);
    const std::size_t bytes = static_cast<std::size_t>(t.nbytes());
    if (t.is_cuda())
        throw_if_false(vram2cxl(t.data_ptr(), bytes, offset), "vram2cxl");
    else
        throw_if_false(dram2cxl(t.data_ptr(), bytes, offset), "dram2cxl");
}

torch::Tensor cxl_to_tensor(torch::Tensor dst, std::size_t offset) {
    auto t = ensure_contiguous(dst);
    const std::size_t bytes = static_cast<std::size_t>(t.nbytes());
    if (t.is_cuda())
        throw_if_false(cxl2vram(t.data_ptr(), bytes, offset), "cxl2vram");
    else
        throw_if_false(cxl2dram(t.data_ptr(), bytes, offset), "cxl2dram");
    if (!dst.is_contiguous()) dst.copy_(t);
    return dst;
}

// -----------------------------------------------------------------------
// GPU-reduce transfer primitives
// -----------------------------------------------------------------------

// 1-stage: write + barrier + bulk gather → GPU staging buffer.
// Python calls this, then does: staging.view(world_size, -1).sum(dim=0)
torch::Tensor py_cxl_write_barrier_gather_1stage(
    torch::Tensor src,
    torch::Tensor gather_dst,
    std::size_t slot_bytes,
    std::size_t data_offset,
    std::size_t control_offset,
    int rank,
    int num_ranks,
    int32_t token)
{
    auto s = ensure_contiguous(src);
    auto d = ensure_contiguous(gather_dst);
    if (!s.is_cuda() || !d.is_cuda())
        throw std::runtime_error("cxl_write_barrier_gather_1stage requires CUDA tensors");

    cxl_write_barrier_gather_1stage(
        s.data_ptr(), d.data_ptr(),
        slot_bytes, data_offset, control_offset,
        rank, num_ranks, token);
    return gather_dst;
}

// 2-stage scatter: write + barrier + cudaMemcpy2D strided gather → GPU staging.
// Python calls this, then does: shard_staging.view(world_size, -1).sum(dim=0)
torch::Tensor py_cxl_write_barrier_scatter_2stage(
    torch::Tensor src,
    torch::Tensor shard_dst,
    std::size_t slot_bytes,
    std::size_t shard_bytes,
    std::size_t data_offset,
    std::size_t control_offset,
    int rank,
    int num_ranks,
    int32_t token)
{
    auto s = ensure_contiguous(src);
    auto d = ensure_contiguous(shard_dst);
    if (!s.is_cuda() || !d.is_cuda())
        throw std::runtime_error("cxl_write_barrier_scatter_2stage requires CUDA tensors");

    cxl_write_barrier_scatter_2stage(
        s.data_ptr(), d.data_ptr(),
        slot_bytes, shard_bytes,
        data_offset, control_offset,
        rank, num_ranks, token);
    return shard_dst;
}

// 2-stage gather: write reduced shard + barrier + bulk gather → GPU result.
torch::Tensor py_cxl_write_barrier_allgather_2stage(
    torch::Tensor reduced_src,
    torch::Tensor result_dst,
    std::size_t shard_bytes,
    std::size_t total_bytes,
    std::size_t reduced_base,
    std::size_t control_offset,
    int rank,
    int num_ranks,
    int32_t token)
{
    auto s = ensure_contiguous(reduced_src);
    auto d = ensure_contiguous(result_dst);
    if (!s.is_cuda() || !d.is_cuda())
        throw std::runtime_error("cxl_write_barrier_allgather_2stage requires CUDA tensors");

    cxl_write_barrier_allgather_2stage(
        s.data_ptr(), d.data_ptr(),
        shard_bytes, total_bytes,
        reduced_base, control_offset,
        rank, num_ranks, token);
    return result_dst;
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

    m.def("cxl_close", &cxl_close,
          "Unmap and unregister the mapped CXL window.");

    m.def("tensor_to_cxl",
          &tensor_to_cxl,
          pybind11::arg("tensor"),
          pybind11::arg("offset") = 0,
          "Copy a tensor (CPU or CUDA) into the mapped CXL window.");

    m.def("cxl_to_tensor",
          &cxl_to_tensor,
          pybind11::arg("tensor"),
          pybind11::arg("offset") = 0,
          "Fill a tensor (CPU or CUDA) from the mapped CXL window.");

    m.def("cxl_barrier_tp",
          &cxl_barrier_tp,
          pybind11::arg("token"),
          pybind11::arg("control_offset"),
          pybind11::arg("rank"),
          pybind11::arg("num_ranks"),
          "CXL spin barrier for tensor parallelism.");

    m.def("cxl_write_barrier_gather_1stage",
          &py_cxl_write_barrier_gather_1stage,
          pybind11::arg("src"),
          pybind11::arg("gather_dst"),
          pybind11::arg("slot_bytes"),
          pybind11::arg("data_offset"),
          pybind11::arg("control_offset"),
          pybind11::arg("rank"),
          pybind11::arg("num_ranks"),
          pybind11::arg("token"),
          "1-stage: GPU->CXL write + barrier + bulk CXL->GPU gather (no reduce).");

    m.def("cxl_write_barrier_scatter_2stage",
          &py_cxl_write_barrier_scatter_2stage,
          pybind11::arg("src"),
          pybind11::arg("shard_dst"),
          pybind11::arg("slot_bytes"),
          pybind11::arg("shard_bytes"),
          pybind11::arg("data_offset"),
          pybind11::arg("control_offset"),
          pybind11::arg("rank"),
          pybind11::arg("num_ranks"),
          pybind11::arg("token"),
          "2-stage scatter: GPU->CXL write + barrier + cudaMemcpy2D strided gather (no reduce).");

    m.def("cxl_write_barrier_allgather_2stage",
          &py_cxl_write_barrier_allgather_2stage,
          pybind11::arg("reduced_src"),
          pybind11::arg("result_dst"),
          pybind11::arg("shard_bytes"),
          pybind11::arg("total_bytes"),
          pybind11::arg("reduced_base"),
          pybind11::arg("control_offset"),
          pybind11::arg("rank"),
          pybind11::arg("num_ranks"),
          pybind11::arg("token"),
          "2-stage gather: GPU->CXL write reduced shard + barrier + bulk CXL->GPU allgather.");
}

}  // namespace
