/*
 * GPU scatter read kernels for sparse KV cache micro-benchmark.
 *
 * 1. scatter_read_kernel:
 *    Mirrors hisparse.cuh's transfer_item_warp: each block reads one token's
 *    KV entry from a single host pointer (pinned DRAM / CXL / staging) into a
 *    contiguous GPU output buffer using ld.global.nc.b64.
 *
 * 2. scatter_read_interleave_kernel:
 *    Multi-source variant for CXL interleave benchmarks. Each token is
 *    assigned to one of N source base pointers via a device_map array,
 *    simulating N CXL devices accessed concurrently.
 *
 * Build: JIT-compiled via torch.utils.cpp_extension.load (see sparse_kv_bench.py).
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {

__global__ void scatter_read_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const int64_t* __restrict__ indices,
    int64_t item_size_bytes)
{
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const uint64_t* src64 =
        reinterpret_cast<const uint64_t*>(src + indices[token_idx] * item_size_bytes);
    uint64_t* dst64 =
        reinterpret_cast<uint64_t*>(dst + static_cast<int64_t>(token_idx) * item_size_bytes);
    const int total_u64 = static_cast<int>(item_size_bytes / sizeof(uint64_t));

    for (int j = tid; j < total_u64; j += blockDim.x) {
        uint64_t tmp;
        asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src64 + j) : "memory");
        asm volatile("st.global.cg.b64 [%0],%1;" :: "l"(dst64 + j), "l"(tmp) : "memory");
    }

    const int remainder = static_cast<int>(item_size_bytes % sizeof(uint64_t));
    if (remainder > 0 && tid == 0) {
        const uint8_t* tail_src = src + indices[token_idx] * item_size_bytes +
                                  total_u64 * sizeof(uint64_t);
        uint8_t* tail_dst = dst + static_cast<int64_t>(token_idx) * item_size_bytes +
                            total_u64 * sizeof(uint64_t);
        for (int j = 0; j < remainder; ++j) {
            tail_dst[j] = tail_src[j];
        }
    }
}

/*
 * Multi-source scatter read for CXL interleave simulation.
 *
 * src_ptrs:   array of N base pointers (one per CXL device), stored on GPU
 * device_map: per-token device index, shape (num_tokens,), values in [0, N)
 * indices:    per-token offset within its device's pool
 */
__global__ void scatter_read_interleave_kernel(
    const uint64_t* __restrict__ src_ptrs,
    uint8_t* __restrict__ dst,
    const int64_t* __restrict__ indices,
    const int32_t* __restrict__ device_map,
    int64_t item_size_bytes)
{
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const uint8_t* base = reinterpret_cast<const uint8_t*>(src_ptrs[device_map[token_idx]]);
    const uint64_t* src64 =
        reinterpret_cast<const uint64_t*>(base + indices[token_idx] * item_size_bytes);
    uint64_t* dst64 =
        reinterpret_cast<uint64_t*>(dst + static_cast<int64_t>(token_idx) * item_size_bytes);
    const int total_u64 = static_cast<int>(item_size_bytes / sizeof(uint64_t));

    for (int j = tid; j < total_u64; j += blockDim.x) {
        uint64_t tmp;
        asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src64 + j) : "memory");
        asm volatile("st.global.cg.b64 [%0],%1;" :: "l"(dst64 + j), "l"(tmp) : "memory");
    }

    const int remainder = static_cast<int>(item_size_bytes % sizeof(uint64_t));
    if (remainder > 0 && tid == 0) {
        const uint8_t* tail_src = base + indices[token_idx] * item_size_bytes +
                                  total_u64 * sizeof(uint64_t);
        uint8_t* tail_dst = dst + static_cast<int64_t>(token_idx) * item_size_bytes +
                            total_u64 * sizeof(uint64_t);
        for (int j = 0; j < remainder; ++j) {
            tail_dst[j] = tail_src[j];
        }
    }
}

}  // extern "C"
