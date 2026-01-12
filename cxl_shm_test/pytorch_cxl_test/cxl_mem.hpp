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

bool cxl_close();
