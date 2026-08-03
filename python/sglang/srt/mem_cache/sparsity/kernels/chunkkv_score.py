"""Fused observation-query reduction and ChunkKV page scoring."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _chunkkv_query_summary_kernel(
    query_ptr,
    query_stride_t,
    query_stride_h,
    query_stride_d,
    extend_start_ptr,
    extend_start_stride_b,
    extend_len_ptr,
    extend_len_stride_b,
    active_ptr,
    active_stride_b,
    output_ptr,
    output_stride_b,
    output_stride_h,
    output_stride_d,
    HEAD_DIM: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    OBS_WINDOW: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    dim_offsets = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_valid = dim_offsets < HEAD_DIM
    active = tl.load(active_ptr + batch_idx * active_stride_b).to(tl.int1)
    extend_len = tl.load(extend_len_ptr + batch_idx * extend_len_stride_b).to(tl.int64)
    extend_start = tl.load(extend_start_ptr + batch_idx * extend_start_stride_b).to(
        tl.int64
    )
    obs_len = tl.minimum(extend_len, OBS_WINDOW)
    obs_start = extend_start + extend_len - obs_len
    group_size: tl.constexpr = QUERY_HEADS // KV_HEADS

    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for obs_offset in range(OBS_WINDOW):
        obs_valid = active & (obs_offset < obs_len)
        token_idx = obs_start + obs_offset
        for group_offset in range(group_size):
            query_head = kv_head_idx * group_size + group_offset
            values = tl.load(
                query_ptr
                + token_idx * query_stride_t
                + query_head * query_stride_h
                + dim_offsets * query_stride_d,
                mask=obs_valid & dim_valid,
                other=0.0,
            ).to(tl.float32)
            accumulator += values

    denominator = tl.maximum(obs_len * group_size, 1).to(tl.float32)
    summary = accumulator / denominator
    tl.store(
        output_ptr
        + batch_idx * output_stride_b
        + kv_head_idx * output_stride_h
        + dim_offsets * output_stride_d,
        summary,
        mask=dim_valid,
    )


@triton.jit
def _chunkkv_accumulate_page_scores_kernel(
    summary_ptr,
    summary_stride_b,
    summary_stride_h,
    summary_stride_d,
    req_pool_indices_ptr,
    req_pool_indices_stride_b,
    seq_lens_ptr,
    seq_lens_stride_b,
    active_ptr,
    active_stride_b,
    req_to_token_ptr,
    req_to_token_stride_b,
    req_to_token_stride_t,
    key_buffer_ptr,
    key_buffer_stride_t,
    key_buffer_stride_h,
    key_buffer_stride_d,
    score_pool_ptr,
    score_pool_stride_b,
    score_pool_stride_p,
    max_pages,
    scale,
    PAGE_SIZE: tl.constexpr,
    RECENT_PAGES: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_FEATURES: tl.constexpr,
    RESET: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    page_idx = tl.program_id(1)
    feature_offsets = tl.arange(0, BLOCK_FEATURES)
    feature_valid = feature_offsets < KV_HEADS * HEAD_DIM
    kv_head_idx = feature_offsets // HEAD_DIM
    dim_idx = feature_offsets % HEAD_DIM

    active = tl.load(active_ptr + batch_idx * active_stride_b).to(tl.int1)
    seq_len = tl.load(seq_lens_ptr + batch_idx * seq_lens_stride_b).to(tl.int64)
    total_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    history_end = tl.maximum(total_pages - RECENT_PAGES, 0)
    candidate = active & (page_idx < history_end) & (page_idx < max_pages)
    req_idx = tl.load(req_pool_indices_ptr + batch_idx * req_pool_indices_stride_b).to(
        tl.int64
    )

    query_summary = tl.load(
        summary_ptr
        + batch_idx * summary_stride_b
        + kv_head_idx * summary_stride_h
        + dim_idx * summary_stride_d,
        mask=feature_valid & candidate,
        other=0.0,
    ).to(tl.float32)
    key_sum = tl.zeros((BLOCK_FEATURES,), dtype=tl.float32)
    for page_offset in range(PAGE_SIZE):
        logical_token = page_idx * PAGE_SIZE + page_offset
        token_valid = candidate & (logical_token < seq_len)
        physical_token = tl.load(
            req_to_token_ptr
            + req_idx * req_to_token_stride_b
            + logical_token * req_to_token_stride_t,
            mask=token_valid,
            other=0,
        ).to(tl.int64)
        keys = tl.load(
            key_buffer_ptr
            + physical_token * key_buffer_stride_t
            + kv_head_idx * key_buffer_stride_h
            + dim_idx * key_buffer_stride_d,
            mask=token_valid & feature_valid,
            other=0.0,
        ).to(tl.float32)
        key_sum += keys

    page_score = tl.sum(query_summary * key_sum, axis=0) * scale
    score_ptr = (
        score_pool_ptr + req_idx * score_pool_stride_b + page_idx * score_pool_stride_p
    )
    if RESET:
        tl.store(score_ptr, tl.where(candidate, page_score, -float("inf")), mask=active)
    else:
        previous = tl.load(score_ptr, mask=candidate, other=0.0).to(tl.float32)
        tl.store(score_ptr, previous + page_score, mask=candidate)


def chunkkv_accumulate_page_scores_(
    *,
    query: torch.Tensor,
    extend_start: torch.Tensor,
    extend_lens: torch.Tensor,
    active_mask: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    key_buffer: torch.Tensor,
    score_pool: torch.Tensor,
    page_size: int,
    recent_pages: int,
    obs_window: int,
    reset: bool,
    query_summary: torch.Tensor | None = None,
) -> torch.Tensor:
    """Accumulate exact historical ChunkKV scores without materializing QK."""
    if not query.is_cuda or torch.version.hip is not None:
        raise ValueError("ChunkKV fused scoring requires NVIDIA CUDA tensors")
    if query.ndim != 3 or key_buffer.ndim != 3:
        raise ValueError("query and key_buffer must have shape [tokens, heads, dim]")
    batch_size = req_pool_indices.shape[0]
    if any(
        tensor.shape != (batch_size,)
        for tensor in (extend_start, extend_lens, active_mask, seq_lens)
    ):
        raise ValueError("ChunkKV batch metadata must have shape [batch]")
    if active_mask.dtype != torch.bool:
        raise TypeError("active_mask must use torch.bool")
    if req_to_token.ndim != 2 or score_pool.ndim != 2:
        raise ValueError("request mapping and score pool must be two-dimensional")
    query_heads, head_dim = query.shape[1:]
    kv_heads = key_buffer.shape[1]
    if key_buffer.shape[2] != head_dim or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads with matching dim")
    if not (1 <= page_size <= 128 and 1 <= obs_window <= 128):
        raise ValueError("fused ChunkKV requires page_size and obs_window <= 128")
    num_features = kv_heads * head_dim
    if num_features > 4096:
        raise ValueError("fused ChunkKV supports at most 4096 KV-head features")
    tensors = (
        extend_start,
        extend_lens,
        active_mask,
        req_pool_indices,
        seq_lens,
        req_to_token,
        key_buffer,
        score_pool,
    )
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("ChunkKV scoring tensors must share one device")

    if query_summary is None:
        query_summary = torch.empty(
            (batch_size, kv_heads, head_dim),
            dtype=torch.float32,
            device=query.device,
        )
    elif query_summary.shape != (batch_size, kv_heads, head_dim):
        raise ValueError("query_summary has an incompatible shape")

    block_d = min(128, triton.next_power_of_2(head_dim))
    _chunkkv_query_summary_kernel[
        (batch_size, kv_heads, triton.cdiv(head_dim, block_d))
    ](
        query,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        extend_start,
        extend_start.stride(0),
        extend_lens,
        extend_lens.stride(0),
        active_mask,
        active_mask.stride(0),
        query_summary,
        query_summary.stride(0),
        query_summary.stride(1),
        query_summary.stride(2),
        HEAD_DIM=head_dim,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        OBS_WINDOW=obs_window,
        BLOCK_D=block_d,
    )

    max_pages = score_pool.shape[1]
    block_features = triton.next_power_of_2(num_features)
    _chunkkv_accumulate_page_scores_kernel[(batch_size, max_pages)](
        query_summary,
        query_summary.stride(0),
        query_summary.stride(1),
        query_summary.stride(2),
        req_pool_indices,
        req_pool_indices.stride(0),
        seq_lens,
        seq_lens.stride(0),
        active_mask,
        active_mask.stride(0),
        req_to_token,
        req_to_token.stride(0),
        req_to_token.stride(1),
        key_buffer,
        key_buffer.stride(0),
        key_buffer.stride(1),
        key_buffer.stride(2),
        score_pool,
        score_pool.stride(0),
        score_pool.stride(1),
        max_pages,
        1.0 / math.sqrt(head_dim),
        PAGE_SIZE=page_size,
        RECENT_PAGES=recent_pages,
        KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_FEATURES=block_features,
        RESET=reset,
        num_warps=8 if block_features >= 512 else 4,
    )
    return query_summary
