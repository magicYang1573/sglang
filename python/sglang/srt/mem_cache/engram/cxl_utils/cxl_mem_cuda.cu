#include <cstddef>
#include <cstdint>
#include <iostream>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "cxl_mem.hpp"
struct CxlRegion {
  void *base = nullptr;
  std::size_t length = 0;
  bool cuda_registered = false;
  void *device_ptr = nullptr;
};

extern CxlRegion g_cxl;
void clflush_range(const void *addr, std::size_t len);

namespace {

inline bool cuda_check(cudaError_t err, const char *what) {
  if (err == cudaSuccess) {
    return true;
  }
  std::cerr << what << " failed: " << cudaGetErrorString(err) << "\n";
  return false;
}

std::size_t *g_offsets_dev = nullptr;
std::size_t g_offsets_capacity = 0;

bool ensure_offsets_capacity(std::size_t count) {
  if (count <= g_offsets_capacity && g_offsets_dev != nullptr) {
    return true;
  }
  if (g_offsets_dev != nullptr) {
    cudaFree(g_offsets_dev);
    g_offsets_dev = nullptr;
    g_offsets_capacity = 0;
  }
  const std::size_t bytes = count * sizeof(std::size_t);
  if (!cuda_check(cudaMalloc(&g_offsets_dev, bytes), "cudaMalloc offsets")) {
    return false;
  }
  g_offsets_capacity = count;
  return true;
}

__global__ void cxl2vram_copy_kernel(const std::uint8_t *base,
                                     std::uint8_t *dst,
                                     const std::size_t *offsets,
                                     std::size_t bytes_per_segment) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x);
  const std::uint8_t *src = base + offsets[idx];
  std::uint8_t *out = dst + idx * bytes_per_segment;

  for (std::size_t i = static_cast<std::size_t>(threadIdx.x);
       i < bytes_per_segment; i += static_cast<std::size_t>(blockDim.x)) {
    out[i] = src[i];
  }
}

} // namespace

bool cxl_register_cuda_region(void *addr, std::size_t length, int gpu_id,
                              CxlRegion *region) {
  if (region == nullptr) {
    return false;
  }
  if (!cuda_check(cudaSetDevice(gpu_id), "cudaSetDevice")) {
    return false;
  }
  if (!cuda_check(
          cudaHostRegister(addr, length,
                           cudaHostRegisterPortable | cudaHostRegisterMapped),
          "cudaHostRegister")) {
    return false;
  }
  region->cuda_registered = true;
  if (!cuda_check(cudaHostGetDevicePointer(&region->device_ptr, addr, 0),
                  "cudaHostGetDevicePointer")) {
    cudaHostUnregister(addr);
    region->cuda_registered = false;
    region->device_ptr = nullptr;
    return false;
  }
  return true;
}

bool cxl_unregister_cuda_region(CxlRegion *region) {
  if (region == nullptr || !region->cuda_registered ||
      region->base == nullptr) {
    return true;
  }
  if (!cuda_check(cudaHostUnregister(region->base), "cudaHostUnregister")) {
    return false;
  }
  region->cuda_registered = false;
  region->device_ptr = nullptr;
  return true;
}

bool vram2cxl(const void *device_src, std::size_t bytes, std::size_t offset) {
  if (g_cxl.base == nullptr || offset + bytes > g_cxl.length) {
    std::cerr << "vram2cxl: invalid region or range\n";
    return false;
  }
  at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
  torch_stream.synchronize();

  void *dst = static_cast<std::uint8_t *>(g_cxl.base) + offset;
  if (!cuda_check(cudaMemcpy(dst, device_src, bytes, cudaMemcpyDeviceToHost),
                  "cudaMemcpy D2H")) {
    return false;
  }
  clflush_range(dst, bytes);
  return true;
}

bool vram2cxl_async(const void *device_src, std::size_t bytes,
                    std::size_t offset) {
  if (g_cxl.base == nullptr || offset + bytes > g_cxl.length) {
    return false;
  }

  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  void *dst = static_cast<std::uint8_t *>(g_cxl.base) + offset;

  if (!cuda_check(cudaMemcpyAsync(dst, device_src, bytes,
                                  cudaMemcpyDeviceToHost, stream.stream()),
                  "D2H")) {
    return false;
  }

  stream.synchronize();
  clflush_range(dst, bytes);
  return true;
}

bool cxl2vram(void *device_dst, std::size_t bytes, std::size_t offset) {
  if (g_cxl.base == nullptr || offset + bytes > g_cxl.length) {
    std::cerr << "cxl2vram: invalid region or range\n";
    return false;
  }
  const void *src = static_cast<const std::uint8_t *>(g_cxl.base) + offset;
  clflush_range(src, bytes);
  if (!cuda_check(cudaMemcpy(device_dst, src, bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy H2D")) {
    return false;
  }
  return true;
}

bool cxl2vram_noflush(void *device_dst, std::size_t bytes, std::size_t offset) {
  if (g_cxl.base == nullptr || offset + bytes > g_cxl.length) {
    std::cerr << "cxl2vram_noflush: invalid region or range\n";
    return false;
  }
  const void *src = static_cast<const std::uint8_t *>(g_cxl.base) + offset;
  if (!cuda_check(cudaMemcpy(device_dst, src, bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy H2D")) {
    return false;
  }
  return true;
}

bool cxl2vram_noflush_parallel_contiguous_dst(void *device_dst,
                                              std::size_t bytes_per_segment,
                                              const std::size_t *offset_list,
                                              std::size_t count) {
  if (g_cxl.base == nullptr) {
    std::cerr << "cxl2vram_noflush_parallel_contiguous_dst: invalid region\n";
    return false;
  }
  if (!g_cxl.cuda_registered || g_cxl.device_ptr == nullptr) {
    std::cerr << "cxl2vram_noflush_parallel_contiguous_dst: CXL region not "
                 "CUDA-mapped\n";
    return false;
  }
  if (device_dst == nullptr || offset_list == nullptr) {
    std::cerr << "cxl2vram_noflush_parallel_contiguous_dst: null pointer\n";
    return false;
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (offset_list[i] + bytes_per_segment > g_cxl.length) {
      std::cerr << "cxl2vram_noflush_parallel_contiguous_dst: invalid range\n";
      return false;
    }
  }

  const auto stream = at::cuda::getCurrentCUDAStream();
  if (!ensure_offsets_capacity(count)) {
    return false;
  }
  const std::size_t offsets_bytes = count * sizeof(std::size_t);
  if (!cuda_check(cudaMemcpyAsync(g_offsets_dev, offset_list, offsets_bytes,
                                  cudaMemcpyHostToDevice, stream.stream()),
                  "cudaMemcpyAsync offsets")) {
    return false;
  }

  const int threads = 256;
  const dim3 grid(static_cast<unsigned int>(count));
  cxl2vram_copy_kernel<<<grid, threads, 0, stream.stream()>>>(
      static_cast<const std::uint8_t *>(g_cxl.device_ptr),
      static_cast<std::uint8_t *>(device_dst), g_offsets_dev,
      bytes_per_segment);
  if (!cuda_check(cudaGetLastError(), "cxl2vram_copy_kernel launch")) {
    return false;
  }
  if (!cuda_check(cudaStreamSynchronize(stream.stream()),
                  "cudaStreamSynchronize")) {
    return false;
  }
  return true;
}
