"""Incremental Quest representation updates for completed KV pages."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quest_update_complete_pages_kernel(
    req_pool_indices_ptr,
    seq_lens_ptr,
    req_to_token_ptr,
    k_buffer_ptr,
    last_completed_page_ptr,
    page_k_min_ptr,
    page_k_max_ptr,
    page_valid_ptr,
    req_pool_indices_stride,
    seq_lens_stride,
    req_to_token_stride_r,
    req_to_token_stride_t,
    k_buffer_stride_t,
    k_buffer_stride_h,
    k_buffer_stride_d,
    num_k_tokens,
    num_pool_pages,
    num_req_slots,
    req_to_token_width,
    PAGE_SIZE: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    REBUILD_ALL: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head = tl.program_id(1)
    req_idx = tl.load(req_pool_indices_ptr + batch_idx * req_pool_indices_stride).to(
        tl.int64
    )
    req_valid = (req_idx >= 0) & (req_idx < num_req_slots)
    safe_req_idx = tl.where(req_valid, req_idx, 0)
    seq_len = tl.load(seq_lens_ptr + batch_idx * seq_lens_stride).to(tl.int64)
    bounded_seq_len = tl.minimum(tl.maximum(seq_len, 0), req_to_token_width)
    if REBUILD_ALL:
        logical_page = 0
    else:
        logical_page = tl.load(
            last_completed_page_ptr + safe_req_idx, mask=req_valid, other=0
        ).to(tl.int64)
    logical_page = tl.maximum(logical_page, 0)
    end_page = bounded_seq_len // PAGE_SIZE

    token_offsets = tl.arange(0, BLOCK_T)
    dim_offsets = tl.arange(0, BLOCK_D)
    token_mask = token_offsets < PAGE_SIZE
    dim_mask = dim_offsets < HEAD_DIM
    safe_dim_offsets = tl.where(dim_mask, dim_offsets, 0)

    while req_valid & (logical_page < end_page):
        logical_tokens = logical_page * PAGE_SIZE + token_offsets
        physical_tokens = tl.load(
            req_to_token_ptr
            + safe_req_idx * req_to_token_stride_r
            + logical_tokens * req_to_token_stride_t,
            mask=token_mask,
            other=0,
        ).to(tl.int64)
        safe_physical_tokens = tl.minimum(
            tl.maximum(physical_tokens, 0), num_k_tokens - 1
        )
        key_offsets = (
            safe_physical_tokens[:, None] * k_buffer_stride_t
            + kv_head * k_buffer_stride_h
            + safe_dim_offsets[None, :] * k_buffer_stride_d
        )
        load_mask = token_mask[:, None] & dim_mask[None, :]
        keys = tl.load(k_buffer_ptr + key_offsets, mask=load_mask, other=0.0).to(
            tl.float32
        )
        page_min = tl.min(tl.where(load_mask, keys, float("inf")), axis=0)
        page_max = tl.max(tl.where(load_mask, keys, -float("inf")), axis=0)

        first_physical_token = tl.load(
            req_to_token_ptr
            + safe_req_idx * req_to_token_stride_r
            + logical_page * PAGE_SIZE * req_to_token_stride_t
        ).to(tl.int64)
        physical_page = tl.minimum(
            tl.maximum(first_physical_token // PAGE_SIZE, 0), num_pool_pages - 1
        )
        output_offsets = (
            physical_page * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + safe_dim_offsets
        )
        tl.store(page_k_min_ptr + output_offsets, page_min, mask=dim_mask)
        tl.store(page_k_max_ptr + output_offsets, page_max, mask=dim_mask)
        tl.store(page_valid_ptr + physical_page, 1, mask=kv_head == 0)
        logical_page += 1


@triton.jit
def _quest_advance_complete_page_tracker_kernel(
    req_pool_indices_ptr,
    seq_lens_ptr,
    last_completed_page_ptr,
    req_pool_indices_stride,
    seq_lens_stride,
    num_req_slots,
    req_to_token_width,
    PAGE_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_idx = tl.load(req_pool_indices_ptr + batch_idx * req_pool_indices_stride).to(
        tl.int64
    )
    req_valid = (req_idx >= 0) & (req_idx < num_req_slots)
    safe_req_idx = tl.where(req_valid, req_idx, 0)
    seq_len = tl.load(seq_lens_ptr + batch_idx * seq_lens_stride).to(tl.int64)
    bounded_seq_len = tl.minimum(tl.maximum(seq_len, 0), req_to_token_width)
    tl.store(
        last_completed_page_ptr + safe_req_idx,
        bounded_seq_len // PAGE_SIZE,
        mask=req_valid,
    )


def quest_update_complete_pages_(
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    k_buffer: torch.Tensor,
    last_completed_page: torch.Tensor,
    page_k_min: torch.Tensor,
    page_k_max: torch.Tensor,
    page_valid: torch.Tensor,
    page_size: int,
    *,
    advance_trackers: bool,
    rebuild_all: bool = False,
) -> None:
    """Update completed pages without device-to-host synchronization.

    ``rebuild_all`` is used by extend/prefill forwards. It deliberately ignores
    a request slot's old tracker so slot reuse and chunked prefill rebuild the
    currently mapped complete pages from logical page zero.
    """
    if not req_pool_indices.is_cuda or torch.version.hip is not None:
        raise ValueError("Quest page update requires NVIDIA CUDA tensors")
    if req_pool_indices.ndim != 1 or seq_lens.shape != req_pool_indices.shape:
        raise ValueError("request indices and sequence lengths must be 1D")
    if req_to_token.ndim != 2 or req_to_token.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("req_to_token must be a 2D integer tensor")
    if k_buffer.ndim != 3 or k_buffer.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("key buffer must be a 3D floating-point tensor")
    if page_k_min.ndim != 3 or page_k_max.shape != page_k_min.shape:
        raise ValueError("Quest min/max pools must have matching 3D shapes")
    if page_k_min.dtype != k_buffer.dtype or page_k_max.dtype != k_buffer.dtype:
        raise ValueError("Quest representations must match the key-buffer dtype")
    num_pool_pages, kv_heads, head_dim = page_k_min.shape
    if k_buffer.shape[1:] != (kv_heads, head_dim):
        raise ValueError("key buffer and representation dimensions must match")
    if page_valid.shape != (num_pool_pages,) or page_valid.dtype != torch.bool:
        raise ValueError("page validity must be a bool vector")
    if last_completed_page.shape != (req_to_token.shape[0],):
        raise ValueError("page tracker must match the request-pool size")
    if last_completed_page.dtype not in (torch.int32, torch.int64):
        raise ValueError("page tracker must use an integer dtype")
    if page_size <= 0 or page_size > 128 or head_dim > 256:
        raise ValueError("Quest page update supports page_size <= 128, dim <= 256")
    tensors = (
        seq_lens,
        req_to_token,
        k_buffer,
        last_completed_page,
        page_k_min,
        page_k_max,
        page_valid,
    )
    if any(tensor.device != req_pool_indices.device for tensor in tensors):
        raise ValueError("Quest page-update tensors must share one CUDA device")
    batch_size = req_pool_indices.numel()
    if batch_size == 0:
        return

    _quest_update_complete_pages_kernel[(batch_size, kv_heads)](
        req_pool_indices,
        seq_lens,
        req_to_token,
        k_buffer,
        last_completed_page,
        page_k_min,
        page_k_max,
        page_valid,
        req_pool_indices.stride(0),
        seq_lens.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_buffer.stride(2),
        k_buffer.shape[0],
        num_pool_pages,
        req_to_token.shape[0],
        req_to_token.shape[1],
        PAGE_SIZE=page_size,
        KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        REBUILD_ALL=rebuild_all,
        BLOCK_T=triton.next_power_of_2(page_size),
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    if advance_trackers:
        _quest_advance_complete_page_tracker_kernel[(batch_size,)](
            req_pool_indices,
            seq_lens,
            last_completed_page,
            req_pool_indices.stride(0),
            seq_lens.stride(0),
            req_to_token.shape[0],
            req_to_token.shape[1],
            PAGE_SIZE=page_size,
            num_warps=1,
        )
