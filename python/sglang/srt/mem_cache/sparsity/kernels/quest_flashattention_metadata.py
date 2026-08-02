"""Write logical Quest selections directly into fixed-address FA3 metadata."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quest_update_fa3_metadata_kernel(
    selected_indices_ptr,
    selected_stride_b,
    selected_stride_p,
    valid_lengths_ptr,
    valid_lengths_stride_b,
    sparse_mask_ptr,
    sparse_mask_stride_b,
    seq_lens_ptr,
    seq_lens_stride_b,
    req_pool_indices_ptr,
    req_pool_indices_stride_b,
    req_to_token_ptr,
    req_to_token_stride_b,
    req_to_token_stride_t,
    num_req_slots,
    req_to_token_width,
    page_table_ptr,
    page_table_stride_b,
    page_table_stride_p,
    cache_seqlens_ptr,
    cache_seqlens_stride_b,
    cu_seqlens_ptr,
    cu_seqlens_stride_b,
    max_selected,
    BATCH_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    UPDATE_LENGTHS: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    page_offsets = tl.program_id(1) * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    valid_length = tl.load(valid_lengths_ptr + batch_idx * valid_lengths_stride_b).to(
        tl.int32
    )
    valid_length = tl.minimum(tl.maximum(valid_length, 0), max_selected)
    use_sparse = tl.load(sparse_mask_ptr + batch_idx * sparse_mask_stride_b).to(tl.int1)
    active = use_sparse & (valid_length > 0)

    page_lane_mask = page_offsets < max_selected
    write_mask = active & page_lane_mask & (page_offsets < valid_length)
    safe_page_offsets = tl.where(page_lane_mask, page_offsets, 0)
    selected_pages = tl.load(
        selected_indices_ptr
        + batch_idx * selected_stride_b
        + safe_page_offsets * selected_stride_p,
        mask=write_mask,
        other=-1,
    ).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + batch_idx * seq_lens_stride_b).to(tl.int64)
    bounded_seq_len = tl.minimum(tl.maximum(seq_len, 0), req_to_token_width)
    max_logical_pages = (bounded_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    selected_in_bounds = (
        write_mask & (selected_pages >= 0) & (selected_pages < max_logical_pages)
    )
    safe_selected_pages = tl.where(selected_in_bounds, selected_pages, 0)
    req_idx = tl.load(
        req_pool_indices_ptr + batch_idx * req_pool_indices_stride_b,
        mask=active,
        other=0,
    ).to(tl.int64)
    req_valid = (req_idx >= 0) & (req_idx < num_req_slots)
    safe_req_idx = tl.where(req_valid, req_idx, 0)
    metadata_write = selected_in_bounds & req_valid
    first_tokens = tl.load(
        req_to_token_ptr
        + safe_req_idx * req_to_token_stride_b
        + safe_selected_pages * PAGE_SIZE * req_to_token_stride_t,
        mask=metadata_write,
        other=0,
    ).to(tl.int64)
    tl.store(
        page_table_ptr
        + batch_idx * page_table_stride_b
        + safe_page_offsets * page_table_stride_p,
        (first_tokens // PAGE_SIZE).to(tl.int32),
        mask=metadata_write,
    )

    # One program computes the small batch prefix sum.  The following FA3
    # launch supplies stream ordering for all page-table writes.
    if UPDATE_LENGTHS:
        if (batch_idx == 0) & (tl.program_id(1) == 0):
            cumulative_length = 0
            for idx in range(BATCH_SIZE):
                row_seq_len = tl.load(seq_lens_ptr + idx * seq_lens_stride_b).to(
                    tl.int64
                )
                row_valid_length = tl.load(
                    valid_lengths_ptr + idx * valid_lengths_stride_b
                ).to(tl.int64)
                row_valid_length = tl.minimum(
                    tl.maximum(row_valid_length, 0), max_selected
                )
                row_sparse = tl.load(sparse_mask_ptr + idx * sparse_mask_stride_b).to(
                    tl.int1
                )
                row_req_idx = tl.load(
                    req_pool_indices_ptr + idx * req_pool_indices_stride_b
                ).to(tl.int64)
                row_req_valid = (row_req_idx >= 0) & (row_req_idx < num_req_slots)
                row_active = row_sparse & row_req_valid & (row_valid_length > 0)
                last_page_length = tl.where(
                    row_seq_len > 0, (row_seq_len - 1) % PAGE_SIZE + 1, 0
                )
                sparse_seq_len = (row_valid_length - 1) * PAGE_SIZE + last_page_length
                cache_seq_len = tl.where(row_active, sparse_seq_len, row_seq_len).to(
                    tl.int32
                )
                tl.store(
                    cache_seqlens_ptr + idx * cache_seqlens_stride_b,
                    cache_seq_len,
                )
                tl.store(
                    cu_seqlens_ptr + idx * cu_seqlens_stride_b,
                    cumulative_length,
                )
                cumulative_length += cache_seq_len
            tl.store(
                cu_seqlens_ptr + BATCH_SIZE * cu_seqlens_stride_b,
                cumulative_length,
            )


def quest_update_flashattention_metadata_(
    selected_indices: torch.Tensor,
    valid_lengths: torch.Tensor,
    sparse_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_size: int,
    *,
    update_lengths: bool,
) -> None:
    """Map selected logical pages to physical FA3 metadata in place."""
    if not selected_indices.is_cuda or torch.version.hip is not None:
        raise ValueError("Quest FA3 metadata update requires NVIDIA CUDA tensors")
    if selected_indices.ndim != 2:
        raise ValueError("selected_indices must have shape [batch, pages]")
    batch_size, max_selected = selected_indices.shape
    if valid_lengths.shape != (batch_size,):
        raise ValueError("valid_lengths must have shape [batch]")
    if sparse_mask.shape != (batch_size,) or sparse_mask.dtype != torch.bool:
        raise ValueError("sparse_mask must be bool with shape [batch]")
    if seq_lens.shape != (batch_size,) or req_pool_indices.shape != (batch_size,):
        raise ValueError("sequence lengths and request indices must match batch")
    if req_to_token.ndim != 2 or not all(req_to_token.shape):
        raise ValueError("req_to_token must be a non-empty 2D tensor")
    if page_table.ndim != 2 or page_table.shape[0] != batch_size:
        raise ValueError("page_table must have shape [batch, pages]")
    if page_table.shape[1] < max_selected:
        raise ValueError("FA3 page table is narrower than the Quest selection")
    if cache_seqlens_int32.shape != (batch_size,):
        raise ValueError("cache_seqlens_int32 must have shape [batch]")
    if cu_seqlens_k.shape != (batch_size + 1,):
        raise ValueError("cu_seqlens_k must have shape [batch + 1]")
    if selected_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("selected_indices must use an integer dtype")
    if valid_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("valid_lengths must use an integer dtype")
    if seq_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("seq_lens must use an integer dtype")
    if req_pool_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("req_pool_indices must use an integer dtype")
    if req_to_token.dtype not in (torch.int32, torch.int64):
        raise ValueError("req_to_token must use an integer dtype")
    if page_table.dtype != torch.int32:
        raise ValueError("FA3 page_table must use int32")
    if cache_seqlens_int32.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise ValueError("FA3 sequence metadata must use int32")
    if page_size <= 0 or max_selected <= 0:
        raise ValueError("page_size and Quest selection width must be positive")
    tensors = (
        valid_lengths,
        sparse_mask,
        seq_lens,
        req_pool_indices,
        req_to_token,
        page_table,
        cache_seqlens_int32,
        cu_seqlens_k,
    )
    if any(tensor.device != selected_indices.device for tensor in tensors):
        raise ValueError("Quest metadata tensors must share one CUDA device")
    if batch_size == 0:
        return

    block_pages = 128
    page_blocks = max(triton.cdiv(max_selected, block_pages), 1)
    _quest_update_fa3_metadata_kernel[(batch_size, page_blocks)](
        selected_indices,
        selected_indices.stride(0),
        selected_indices.stride(1),
        valid_lengths,
        valid_lengths.stride(0),
        sparse_mask,
        sparse_mask.stride(0),
        seq_lens,
        seq_lens.stride(0),
        req_pool_indices,
        req_pool_indices.stride(0),
        req_to_token,
        req_to_token.stride(0),
        req_to_token.stride(1),
        req_to_token.shape[0],
        req_to_token.shape[1],
        page_table,
        page_table.stride(0),
        page_table.stride(1),
        cache_seqlens_int32,
        cache_seqlens_int32.stride(0),
        cu_seqlens_k,
        cu_seqlens_k.stride(0),
        max_selected,
        BATCH_SIZE=batch_size,
        PAGE_SIZE=page_size,
        UPDATE_LENGTHS=update_lengths,
        BLOCK_PAGES=block_pages,
        num_warps=4,
    )
