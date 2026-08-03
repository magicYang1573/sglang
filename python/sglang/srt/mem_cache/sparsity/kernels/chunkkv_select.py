"""Fixed-width decode selection for cached ChunkKV base pages."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _chunkkv_select_kernel(
    seq_lens_ptr,
    seq_lens_stride_b,
    req_pool_indices_ptr,
    req_pool_indices_stride_b,
    base_pages_ptr,
    base_pages_stride_b,
    base_pages_stride_p,
    state_valid_ptr,
    state_valid_stride_b,
    selected_ptr,
    selected_stride_b,
    selected_stride_p,
    valid_lengths_ptr,
    valid_lengths_stride_b,
    visible_lens_ptr,
    visible_lens_stride_b,
    sparse_mask_ptr,
    sparse_mask_stride_b,
    min_sparse_tokens,
    PAGE_SIZE: tl.constexpr,
    PAGE_BUDGET: tl.constexpr,
    RECENT_PAGES: tl.constexpr,
    HISTORY_BUDGET: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    page_offsets = tl.arange(0, BLOCK_PAGES)
    output_valid = page_offsets < PAGE_BUDGET
    seq_len = tl.load(seq_lens_ptr + batch_idx * seq_lens_stride_b).to(tl.int64)
    req_idx = tl.load(req_pool_indices_ptr + batch_idx * req_pool_indices_stride_b).to(
        tl.int64
    )
    total_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    has_state = tl.load(state_valid_ptr + req_idx * state_valid_stride_b).to(tl.int1)
    use_sparse = (
        has_state & (seq_len >= min_sparse_tokens) & (total_pages > PAGE_BUDGET)
    )

    base_page = tl.load(
        base_pages_ptr
        + req_idx * base_pages_stride_b
        + page_offsets * base_pages_stride_p,
        mask=output_valid & (page_offsets < HISTORY_BUDGET),
        other=0,
    )
    recent_offset = page_offsets - HISTORY_BUDGET
    recent_page = total_pages - RECENT_PAGES + recent_offset
    logical_page = tl.where(page_offsets < HISTORY_BUDGET, base_page, recent_page)
    tl.store(
        selected_ptr + batch_idx * selected_stride_b + page_offsets * selected_stride_p,
        tl.where(use_sparse, logical_page, -1),
        mask=output_valid,
    )

    final_page_padding = total_pages * PAGE_SIZE - seq_len
    visible_len = PAGE_BUDGET * PAGE_SIZE - final_page_padding
    tl.store(
        valid_lengths_ptr + batch_idx * valid_lengths_stride_b,
        tl.where(use_sparse, PAGE_BUDGET, 0),
    )
    tl.store(
        visible_lens_ptr + batch_idx * visible_lens_stride_b,
        tl.where(use_sparse, visible_len, 0),
    )
    tl.store(
        sparse_mask_ptr + batch_idx * sparse_mask_stride_b,
        use_sparse,
    )


def chunkkv_select(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    base_pages: torch.Tensor,
    state_valid: torch.Tensor,
    *,
    page_size: int,
    page_budget: int,
    recent_pages: int,
    min_sparse_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build sorted semantic-base + current-recent page selections."""
    if not seq_lens.is_cuda or torch.version.hip is not None:
        raise ValueError("ChunkKV fused selection requires NVIDIA CUDA tensors")
    if seq_lens.ndim != 1 or req_pool_indices.shape != seq_lens.shape:
        raise ValueError("sequence lengths and request indices must match")
    if base_pages.ndim != 2 or state_valid.ndim != 1:
        raise ValueError("ChunkKV state tensors have incompatible ranks")
    history_budget = page_budget - recent_pages
    if history_budget <= 0 or base_pages.shape[1] != history_budget:
        raise ValueError("ChunkKV base-page capacity does not match the budget")
    if page_budget > 1024:
        raise ValueError("fused ChunkKV selection supports page_budget <= 1024")
    if any(
        tensor.device != seq_lens.device
        for tensor in (req_pool_indices, base_pages, state_valid)
    ):
        raise ValueError("ChunkKV selection tensors must share one device")

    batch_size = seq_lens.shape[0]
    selected = torch.empty(
        (batch_size, page_budget), dtype=torch.int32, device=seq_lens.device
    )
    valid_lengths = torch.empty(batch_size, dtype=torch.int32, device=seq_lens.device)
    visible_lens = torch.empty_like(valid_lengths)
    sparse_mask = torch.empty(batch_size, dtype=torch.bool, device=seq_lens.device)
    if batch_size == 0:
        return selected, valid_lengths, visible_lens, sparse_mask

    block_pages = triton.next_power_of_2(page_budget)
    _chunkkv_select_kernel[(batch_size,)](
        seq_lens,
        seq_lens.stride(0),
        req_pool_indices,
        req_pool_indices.stride(0),
        base_pages,
        base_pages.stride(0),
        base_pages.stride(1),
        state_valid,
        state_valid.stride(0),
        selected,
        selected.stride(0),
        selected.stride(1),
        valid_lengths,
        valid_lengths.stride(0),
        visible_lens,
        visible_lens.stride(0),
        sparse_mask,
        sparse_mask.stride(0),
        min_sparse_tokens,
        PAGE_SIZE=page_size,
        PAGE_BUDGET=page_budget,
        RECENT_PAGES=recent_pages,
        HISTORY_BUDGET=history_budget,
        BLOCK_PAGES=block_pages,
    )
    return selected, valid_lengths, visible_lens, sparse_mask
