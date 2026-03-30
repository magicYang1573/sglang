/*
 * GPU scatter read kernel for sparse KV cache micro-benchmark.
 *
 * Mirrors hisparse.cuh's transfer_item_warp: each block reads one token's
 * KV entry from a host pointer (pinned DRAM / CXL / staging) into a
 * contiguous GPU output buffer using ld.global.nc.b64.
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

    /* Handle trailing bytes (item_size_bytes % 8 != 0, e.g. 656 % 8 == 0 for
       DeepSeek-V3.2 FP8 packed but keep the kernel general). */
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

}  // extern "C"
