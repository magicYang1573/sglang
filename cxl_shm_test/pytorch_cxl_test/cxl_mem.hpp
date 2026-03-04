#pragma once

#include <cstddef>
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

// Read num_shards non-contiguous CXL regions into a contiguous GPU buffer.
// cxl_offsets[i] is the byte offset within the CXL window for shard i.
// All shards must have the same size (shard_bytes).
// Uses one CPU thread per shard for parallel clflush+memcpy, then a single
// H2D cudaMemcpy from a pinned staging buffer to device_dst.
bool cxl2vram_shards(void *device_dst,
                     std::size_t num_shards,
                     const std::size_t *cxl_offsets,
                     std::size_t shard_bytes);

bool cxl_close();

void cxl_barrier_tp(int32_t token, int64_t control_offset, int rank, int num_ranks);