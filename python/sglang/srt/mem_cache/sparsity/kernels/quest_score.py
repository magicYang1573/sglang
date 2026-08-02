"""Fused Quest page scoring.

Adapted from sgl-project/sglang#31790.  The reduction intentionally preserves
the unified KSE policy contract: a shared FA3 page table is ranked by the
strongest query head bound rather than by materializing a [B, P, H, G, D]
PyTorch tensor.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quest_page_score_serial_kernel(
    queries_ptr,
    page_k_min_ptr,
    page_k_max_ptr,
    page_valid_ptr,
    physical_pages_ptr,
    active_mask_ptr,
    history_page_counts_ptr,
    output_ptr,
    num_pages,
    num_pool_pages,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    page_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    physical_page_raw = tl.load(physical_pages_ptr + batch_idx * num_pages + page_idx)
    page_in_bounds = (physical_page_raw >= 0) & (physical_page_raw < num_pool_pages)
    request_is_active = tl.load(active_mask_ptr + batch_idx).to(tl.int1)
    history_page_count = tl.load(history_page_counts_ptr + batch_idx)
    selected = page_in_bounds & request_is_active & (page_idx < history_page_count)
    physical_page = tl.where(page_in_bounds, physical_page_raw, 0)
    page_is_valid = tl.load(page_valid_ptr + physical_page, mask=selected, other=0).to(
        tl.int1
    )
    selected &= page_is_valid

    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = (dim_offsets < HEAD_DIM) & selected
    safe_dim_offsets = tl.where(dim_mask, dim_offsets, 0)
    strongest = -float("inf")

    for kv_head in range(KV_HEADS):
        key_offsets = (
            physical_page * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + safe_dim_offsets
        )
        key_min = tl.load(page_k_min_ptr + key_offsets, mask=dim_mask, other=0.0).to(
            tl.float32
        )
        key_max = tl.load(page_k_max_ptr + key_offsets, mask=dim_mask, other=0.0).to(
            tl.float32
        )

        for group_idx in range(GROUP_SIZE):
            query_head = kv_head * GROUP_SIZE + group_idx
            query_offsets = (
                batch_idx * Q_HEADS * HEAD_DIM
                + query_head * HEAD_DIM
                + safe_dim_offsets
            )
            query = tl.load(queries_ptr + query_offsets, mask=dim_mask, other=0.0).to(
                tl.float32
            )
            bound_keys = tl.where(query >= 0, key_max, key_min)
            head_bound = tl.sum(query * bound_keys, axis=0)
            strongest = tl.maximum(strongest, head_bound)

    score = tl.where(selected, strongest, -float("inf"))
    tl.store(output_ptr + batch_idx * num_pages + page_idx, score)


@triton.jit
def _quest_page_score_vectorized_kernel(
    queries_ptr,
    page_k_min_ptr,
    page_k_max_ptr,
    page_valid_ptr,
    physical_pages_ptr,
    active_mask_ptr,
    history_page_counts_ptr,
    output_ptr,
    num_pages,
    num_pool_pages,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Score one page with KV heads and dimensions evaluated in parallel."""
    page_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    physical_page_raw = tl.load(physical_pages_ptr + batch_idx * num_pages + page_idx)
    page_in_bounds = (physical_page_raw >= 0) & (physical_page_raw < num_pool_pages)
    request_is_active = tl.load(active_mask_ptr + batch_idx).to(tl.int1)
    history_page_count = tl.load(history_page_counts_ptr + batch_idx)
    selected = page_in_bounds & request_is_active & (page_idx < history_page_count)
    physical_page = tl.where(page_in_bounds, physical_page_raw, 0)
    page_is_valid = tl.load(page_valid_ptr + physical_page, mask=selected, other=0).to(
        tl.int1
    )
    selected &= page_is_valid

    kv_head_offsets = tl.arange(0, BLOCK_KV)
    dim_offsets = tl.arange(0, BLOCK_D)
    kv_head_mask = kv_head_offsets < KV_HEADS
    dim_mask = dim_offsets < HEAD_DIM
    element_mask = selected & kv_head_mask[:, None] & dim_mask[None, :]
    safe_kv_heads = tl.where(kv_head_mask, kv_head_offsets, 0)
    safe_dims = tl.where(dim_mask, dim_offsets, 0)

    key_offsets = (
        physical_page * KV_HEADS * HEAD_DIM
        + safe_kv_heads[:, None] * HEAD_DIM
        + safe_dims[None, :]
    )
    key_min = tl.load(page_k_min_ptr + key_offsets, mask=element_mask, other=0.0).to(
        tl.float32
    )
    key_max = tl.load(page_k_max_ptr + key_offsets, mask=element_mask, other=0.0).to(
        tl.float32
    )
    strongest_per_kv = tl.full((BLOCK_KV,), -float("inf"), tl.float32)
    for group_idx in range(GROUP_SIZE):
        query_heads = safe_kv_heads * GROUP_SIZE + group_idx
        query_offsets = (
            batch_idx * Q_HEADS * HEAD_DIM
            + query_heads[:, None] * HEAD_DIM
            + safe_dims[None, :]
        )
        query = tl.load(queries_ptr + query_offsets, mask=element_mask, other=0.0).to(
            tl.float32
        )
        bounds = tl.sum(query * tl.where(query >= 0, key_max, key_min), axis=1)
        strongest_per_kv = tl.maximum(strongest_per_kv, bounds)
    strongest = tl.max(
        tl.where(selected & kv_head_mask, strongest_per_kv, -float("inf")), axis=0
    )
    tl.store(output_ptr + batch_idx * num_pages + page_idx, strongest)


def quest_page_scores(
    queries: torch.Tensor,
    page_k_min: torch.Tensor,
    page_k_max: torch.Tensor,
    page_valid: torch.Tensor,
    physical_pages: torch.Tensor,
    *,
    active_mask: torch.Tensor,
    history_page_counts: torch.Tensor,
) -> torch.Tensor:
    """Compute all request/page upper bounds without large intermediates."""
    if not queries.is_cuda or torch.version.hip is not None:
        raise ValueError("Quest page scoring requires NVIDIA CUDA tensors")
    if page_k_min.ndim != 3 or page_k_max.shape != page_k_min.shape:
        raise ValueError("Quest min/max pools must have matching 3D shapes")
    if page_k_min.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Quest min/max pools require fp16, bf16, or fp32")
    if page_k_max.dtype != page_k_min.dtype:
        raise ValueError("Quest min/max pools must use the same dtype")
    if not page_k_min.is_contiguous() or not page_k_max.is_contiguous():
        raise ValueError("Quest min/max pools must be contiguous")
    if (
        page_valid.shape != (page_k_min.shape[0],)
        or page_valid.dtype != torch.bool
        or not page_valid.is_contiguous()
    ):
        raise ValueError("Quest page validity must be a contiguous bool vector")
    if physical_pages.ndim != 2 or physical_pages.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("physical_pages must be an integer [batch, pages] tensor")
    batch_size, num_pages = physical_pages.shape
    if active_mask.shape != (batch_size,) or active_mask.dtype != torch.bool:
        raise ValueError("active_mask must be bool with shape [batch]")
    if history_page_counts.shape != (batch_size,) or history_page_counts.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("history_page_counts must be integer with shape [batch]")
    tensors = (
        page_k_min,
        page_k_max,
        page_valid,
        physical_pages,
        active_mask,
        history_page_counts,
    )
    if any(tensor.device != queries.device for tensor in tensors):
        raise ValueError("Quest score tensors must share one CUDA device")

    num_pool_pages, kv_heads, head_dim = page_k_min.shape
    if queries.ndim == 2:
        if queries.shape[1] % head_dim:
            raise ValueError("Quest query hidden size is not divisible by head_dim")
        queries = queries.reshape(batch_size, -1, head_dim)
    if queries.ndim != 3 or queries.shape[0] != batch_size:
        raise ValueError("Quest queries must have shape [batch, heads, dim]")
    query_heads = queries.shape[1]
    if queries.shape[2] != head_dim or query_heads % kv_heads:
        raise ValueError("Quest query heads must be divisible by KV heads")
    if head_dim > 256:
        raise ValueError("Quest score kernel supports head_dim <= 256")
    if not queries.is_contiguous():
        queries = queries.contiguous()
    if not physical_pages.is_contiguous():
        physical_pages = physical_pages.contiguous()
    if not active_mask.is_contiguous():
        active_mask = active_mask.contiguous()
    if not history_page_counts.is_contiguous():
        history_page_counts = history_page_counts.contiguous()

    output = torch.empty(
        (batch_size, num_pages), dtype=torch.float32, device=queries.device
    )
    if batch_size == 0 or num_pages == 0:
        return output
    block_kv = triton.next_power_of_2(kv_heads)
    block_d = triton.next_power_of_2(head_dim)
    common_args = (
        queries,
        page_k_min,
        page_k_max,
        page_valid,
        physical_pages,
        active_mask,
        history_page_counts,
        output,
        num_pages,
        num_pool_pages,
    )
    # GQA benefits from evaluating KV-head groups in parallel. MHA and MQA do
    # not expose that middle dimension, so retain the lower-register serial
    # kernel for those geometries.
    use_vectorized_gqa = 1 < kv_heads < query_heads and block_kv * block_d <= 65536
    if use_vectorized_gqa:
        _quest_page_score_vectorized_kernel[(num_pages, batch_size)](
            *common_args,
            Q_HEADS=query_heads,
            KV_HEADS=kv_heads,
            GROUP_SIZE=query_heads // kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_KV=block_kv,
            BLOCK_D=block_d,
            num_warps=8 if block_kv * block_d >= 2048 else 4,
        )
    else:
        _quest_page_score_serial_kernel[(num_pages, batch_size)](
            *common_args,
            Q_HEADS=query_heads,
            KV_HEADS=kv_heads,
            GROUP_SIZE=query_heads // kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            num_warps=4,
        )
    return output
