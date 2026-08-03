"""Build compact Triton decode metadata from logical page selections."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fill_triton_visibility_indices_kernel(
    selected_ptr,
    selected_stride_b,
    selected_stride_p,
    sparse_mask_ptr,
    sparse_mask_stride_b,
    seq_lens_ptr,
    seq_lens_stride_b,
    req_pool_indices_ptr,
    req_pool_indices_stride_b,
    req_to_token_ptr,
    req_to_token_stride_b,
    req_to_token_stride_t,
    sparse_indptr_ptr,
    sparse_indptr_stride_b,
    output_ptr,
    output_stride_t,
    num_req_slots,
    req_to_token_width,
    max_row_tokens,
    PAGE_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    token_offsets = tl.program_id(1) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    row_end = tl.load(sparse_indptr_ptr + (batch_idx + 1) * sparse_indptr_stride_b)
    row_start = tl.load(sparse_indptr_ptr + batch_idx * sparse_indptr_stride_b)
    row_length = row_end - row_start
    active = (token_offsets < row_length) & (token_offsets < max_row_tokens)

    use_sparse = tl.load(sparse_mask_ptr + batch_idx * sparse_mask_stride_b).to(tl.int1)
    page_offsets = token_offsets // PAGE_SIZE
    page_inner = token_offsets % PAGE_SIZE
    selected_page = tl.load(
        selected_ptr + batch_idx * selected_stride_b + page_offsets * selected_stride_p,
        mask=active & use_sparse,
        other=0,
    ).to(tl.int64)
    logical_tokens = tl.where(
        use_sparse, selected_page * PAGE_SIZE + page_inner, token_offsets
    )

    seq_len = tl.load(seq_lens_ptr + batch_idx * seq_lens_stride_b).to(tl.int64)
    req_idx = tl.load(req_pool_indices_ptr + batch_idx * req_pool_indices_stride_b).to(
        tl.int64
    )
    valid = (
        active
        & (req_idx >= 0)
        & (req_idx < num_req_slots)
        & (logical_tokens >= 0)
        & (logical_tokens < seq_len)
        & (logical_tokens < req_to_token_width)
    )
    safe_req_idx = tl.where(valid, req_idx, 0)
    safe_logical_tokens = tl.where(valid, logical_tokens, 0)
    physical_tokens = tl.load(
        req_to_token_ptr
        + safe_req_idx * req_to_token_stride_b
        + safe_logical_tokens * req_to_token_stride_t,
        mask=valid,
        other=0,
    )
    tl.store(
        output_ptr + (row_start + token_offsets) * output_stride_t,
        physical_tokens,
        mask=valid,
    )


def fill_triton_visibility_indices_(
    *,
    selected_indices: torch.Tensor,
    sparse_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    sparse_indptr: torch.Tensor,
    output: torch.Tensor,
    page_size: int,
    max_row_tokens: int,
) -> None:
    """Fill flattened physical token indices without host synchronization."""
    if not selected_indices.is_cuda or torch.version.hip is not None:
        raise ValueError("Triton visibility metadata requires NVIDIA CUDA tensors")
    if selected_indices.ndim != 2:
        raise ValueError("selected_indices must have shape [batch, pages]")
    batch_size = selected_indices.shape[0]
    if sparse_mask.shape != (batch_size,) or sparse_mask.dtype != torch.bool:
        raise ValueError("sparse_mask must be bool with shape [batch]")
    if seq_lens.shape != (batch_size,) or req_pool_indices.shape != (batch_size,):
        raise ValueError("sequence lengths and request indices must match batch")
    if sparse_indptr.shape != (batch_size + 1,) or sparse_indptr.dtype != torch.int32:
        raise ValueError("sparse_indptr must be int32 with shape [batch + 1]")
    if req_to_token.ndim != 2 or not all(req_to_token.shape):
        raise ValueError("req_to_token must be a non-empty 2D tensor")
    if output.ndim != 1 or output.dtype != torch.int64:
        raise ValueError("output must be a flat int64 tensor")
    if page_size <= 0 or max_row_tokens <= 0:
        raise ValueError("page_size and max_row_tokens must be positive")
    if any(
        tensor.device != selected_indices.device
        for tensor in (
            sparse_mask,
            seq_lens,
            req_pool_indices,
            req_to_token,
            sparse_indptr,
            output,
        )
    ):
        raise ValueError("Triton visibility metadata tensors must share one device")
    if batch_size == 0:
        return

    block_tokens = 256
    token_blocks = max(triton.cdiv(max_row_tokens, block_tokens), 1)
    _fill_triton_visibility_indices_kernel[(batch_size, token_blocks)](
        selected_indices,
        selected_indices.stride(0),
        selected_indices.stride(1),
        sparse_mask,
        sparse_mask.stride(0),
        seq_lens,
        seq_lens.stride(0),
        req_pool_indices,
        req_pool_indices.stride(0),
        req_to_token,
        req_to_token.stride(0),
        req_to_token.stride(1),
        sparse_indptr,
        sparse_indptr.stride(0),
        output,
        output.stride(0),
        req_to_token.shape[0],
        req_to_token.shape[1],
        max_row_tokens,
        PAGE_SIZE=page_size,
        BLOCK_TOKENS=block_tokens,
    )
