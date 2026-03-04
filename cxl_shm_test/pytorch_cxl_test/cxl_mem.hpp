#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <string>
#include <sys/types.h>

bool cxl_init(const std::string &dev_path = "/dev/dax0.0",
              std::size_t map_bytes = 64ULL * 1024 * 1024,
              off_t offset = 0,
              bool register_cuda = false,
              int gpu_id = 0);

bool dram2cxl(const void *dram_src, std::size_t bytes, std::size_t offset = 0);
bool cxl2dram(void *dram_dst, std::size_t bytes, std::size_t offset = 0);

bool vram2cxl(const void *device_src,
              std::size_t bytes,
              std::size_t offset = 0);

bool vram2cxl_async(const void *device_src, std::size_t bytes, std::size_t offset);

bool cxl2vram(void *device_dst,
              std::size_t bytes,
              std::size_t offset = 0);

bool cxl_close();

void cxl_barrier_tp(int32_t token, int64_t control_offset, int rank, int num_ranks);

void *cxl_base_ptr();
std::size_t cxl_map_length();

// -----------------------------------------------------------------------
// GPU-reduce transfer primitives
// C++ handles all CXL clflush + cudaMemcpy bookkeeping.
// The caller (Python) performs the actual reduction on the GPU.
// -----------------------------------------------------------------------

// 1-stage: write own slot to CXL, run barrier, bulk-copy all slots to GPU.
//   device_gather_dst must hold (num_ranks * slot_bytes) bytes on the device.
void cxl_write_barrier_gather_1stage(
    void *device_src,
    void *device_gather_dst,
    std::size_t slot_bytes,
    std::size_t data_offset,
    std::size_t control_offset,
    int rank,
    int num_ranks,
    int32_t token);

// 2-stage scatter phase:
//   - Write full tensor to CXL, run barrier 1.
//   - Strided-gather "my shard" from every peer into a contiguous GPU buffer
//     using a single cudaMemcpy2D call.
//   device_shard_dst must hold (num_ranks * shard_bytes) bytes on the device.
void cxl_write_barrier_scatter_2stage(
    void *device_src,
    void *device_shard_dst,
    std::size_t slot_bytes,
    std::size_t shard_bytes,
    std::size_t data_offset,
    std::size_t control_offset,
    int rank,
    int num_ranks,
    int32_t token);

// 2-stage gather phase:
//   - Write the GPU-reduced shard to CXL, run barrier 2.
//   - Bulk-copy all reduced shards (contiguous in CXL) to GPU.
//   device_result_dst must hold (total_bytes) bytes on the device.
void cxl_write_barrier_allgather_2stage(
    void *device_reduced_src,
    void *device_result_dst,
    std::size_t shard_bytes,
    std::size_t total_bytes,
    std::size_t reduced_base,
    std::size_t control_offset,
    int rank,
    int num_ranks,
    int32_t token);
