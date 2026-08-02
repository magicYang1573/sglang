"""Fused StreamingLLM page selection for NVIDIA CUDA."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

STREAMING_LLM_MAX_SELECTION_WIDTH = 8192


@triton.jit
def _streaming_llm_select_kernel(
    seq_lens_ptr,
    logical_indices_ptr,
    valid_lengths_ptr,
    visible_kv_lens_ptr,
    sparse_mask_ptr,
    PAGE_SIZE: tl.constexpr,
    MIN_SPARSE_TOKENS: tl.constexpr,
    SINK_PAGES: tl.constexpr,
    RECENT_PAGES: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    seq_len = tl.load(seq_lens_ptr + batch_idx).to(tl.int64)
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    use_sparse = (seq_len >= MIN_SPARSE_TOKENS) & (num_pages > CAPACITY)

    recent_offsets = offsets - SINK_PAGES
    logical_pages = tl.where(
        offsets < SINK_PAGES,
        offsets,
        num_pages - RECENT_PAGES + recent_offsets,
    )
    output_mask = offsets < CAPACITY
    tl.store(
        logical_indices_ptr + batch_idx * CAPACITY + offsets,
        tl.where(use_sparse, logical_pages, -1).to(tl.int32),
        mask=output_mask,
    )

    first_lane = offsets == 0
    valid_length = tl.where(use_sparse, CAPACITY, 0).to(tl.int32)
    final_page_padding = num_pages * PAGE_SIZE - seq_len
    visible_kv_len = tl.where(
        use_sparse, CAPACITY * PAGE_SIZE - final_page_padding, 0
    ).to(tl.int32)
    scalar_offsets = batch_idx + offsets
    tl.store(valid_lengths_ptr + scalar_offsets, valid_length, mask=first_lane)
    tl.store(visible_kv_lens_ptr + scalar_offsets, visible_kv_len, mask=first_lane)
    tl.store(sparse_mask_ptr + scalar_offsets, use_sparse, mask=first_lane)


def streaming_llm_select(
    seq_lens: torch.Tensor,
    *,
    page_size: int,
    min_sparse_tokens: int,
    sink_pages: int,
    recent_pages: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the standard SelectionResult tensors in one kernel launch."""
    if not seq_lens.is_cuda or torch.version.hip is not None:
        raise ValueError("fused StreamingLLM selection requires NVIDIA CUDA")
    if seq_lens.ndim != 1 or seq_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("seq_lens must be a 1D integer tensor")
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    if page_size <= 0 or min_sparse_tokens < 0 or sink_pages < 0 or recent_pages <= 0:
        raise ValueError("invalid StreamingLLM selection configuration")
    capacity = sink_pages + recent_pages
    if capacity <= 0 or capacity > STREAMING_LLM_MAX_SELECTION_WIDTH:
        raise ValueError(
            "fused StreamingLLM selection width must be in "
            f"[1, {STREAMING_LLM_MAX_SELECTION_WIDTH}]"
        )

    batch_size = seq_lens.shape[0]
    logical_indices = torch.empty(
        (batch_size, capacity), dtype=torch.int32, device=seq_lens.device
    )
    valid_lengths = torch.empty(batch_size, dtype=torch.int32, device=seq_lens.device)
    visible_kv_lens = torch.empty_like(valid_lengths)
    sparse_mask = torch.empty(batch_size, dtype=torch.bool, device=seq_lens.device)
    if batch_size:
        _streaming_llm_select_kernel[(batch_size,)](
            seq_lens,
            logical_indices,
            valid_lengths,
            visible_kv_lens,
            sparse_mask,
            PAGE_SIZE=page_size,
            MIN_SPARSE_TOKENS=min_sparse_tokens,
            SINK_PAGES=sink_pages,
            RECENT_PAGES=recent_pages,
            CAPACITY=capacity,
            BLOCK_SIZE=triton.next_power_of_2(capacity),
            num_warps=8 if capacity >= 2048 else 4,
        )
    return logical_indices, valid_lengths, visible_kv_lens, sparse_mask
