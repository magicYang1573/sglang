"""Write logical Quest selections directly into fixed-address FA3 metadata."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


QUEST_DIRECT_METADATA_MAX_WIDTH = 1024


@triton.jit
def _quest_count_valid_pages(
    topk_scores_ptr,
    topk_indices_ptr,
    k_per_req_ptr,
    recent_indices_ptr,
    recent_valid_ptr,
    batch_idx,
    max_logical_pages,
    TOPK_WIDTH: tl.constexpr,
    RECENT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    topk_mask = offsets < TOPK_WIDTH
    safe_topk = tl.where(topk_mask, offsets, 0)
    topk_scores = tl.load(
        topk_scores_ptr + batch_idx * TOPK_WIDTH + safe_topk,
        mask=topk_mask,
        other=-float("inf"),
    ).to(tl.float32)
    topk_indices = tl.load(
        topk_indices_ptr + batch_idx * TOPK_WIDTH + safe_topk,
        mask=topk_mask,
        other=-1,
    ).to(tl.int64)
    k_per_req = tl.load(k_per_req_ptr + batch_idx).to(tl.int32)
    topk_valid = (
        topk_mask
        & (offsets < k_per_req)
        & (topk_scores > -float("inf"))
        & (topk_scores < float("inf"))
        & (topk_indices >= 0)
        & (topk_indices < max_logical_pages)
    )

    recent_offsets = offsets - TOPK_WIDTH
    recent_mask = (recent_offsets >= 0) & (recent_offsets < RECENT_WIDTH)
    safe_recent = tl.where(recent_mask, recent_offsets, 0)
    recent_indices = tl.load(
        recent_indices_ptr + batch_idx * RECENT_WIDTH + safe_recent,
        mask=recent_mask,
        other=-1,
    ).to(tl.int64)
    recent_is_valid = tl.load(
        recent_valid_ptr + batch_idx * RECENT_WIDTH + safe_recent,
        mask=recent_mask,
        other=0,
    ).to(tl.int1)
    recent_is_valid &= (
        recent_mask
        & (recent_indices >= 0)
        & (recent_indices < max_logical_pages)
    )
    return tl.sum((topk_valid | recent_is_valid).to(tl.int32), axis=0)


@triton.jit
def _quest_finalize_to_fa3_metadata_small_batch_kernel(
    topk_scores_ptr,
    topk_indices_ptr,
    k_per_req_ptr,
    recent_indices_ptr,
    recent_valid_ptr,
    selected_output_ptr,
    selected_output_stride_b,
    selected_output_stride_p,
    valid_lengths_ptr,
    visible_kv_lens_ptr,
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
    TOPK_WIDTH: tl.constexpr,
    RECENT_WIDTH: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    UPDATE_LENGTHS: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One-program path: sort/map all B<=4 rows and build cu_seqlens once."""
    batch_offsets = tl.arange(0, BLOCK_BATCH)[:, None]
    offsets = tl.arange(0, BLOCK_SIZE)[None, :]
    batch_mask = batch_offsets < BATCH_SIZE
    seq_lens = tl.load(
        seq_lens_ptr + batch_offsets * seq_lens_stride_b,
        mask=batch_mask,
        other=0,
    ).to(tl.int64)
    bounded_seq_lens = tl.minimum(tl.maximum(seq_lens, 0), req_to_token_width)
    max_logical_pages = (bounded_seq_lens + PAGE_SIZE - 1) // PAGE_SIZE

    topk_mask = batch_mask & (offsets < TOPK_WIDTH)
    safe_topk = tl.where(offsets < TOPK_WIDTH, offsets, 0)
    topk_scores = tl.load(
        topk_scores_ptr + batch_offsets * TOPK_WIDTH + safe_topk,
        mask=topk_mask,
        other=-float("inf"),
    ).to(tl.float32)
    topk_indices = tl.load(
        topk_indices_ptr + batch_offsets * TOPK_WIDTH + safe_topk,
        mask=topk_mask,
        other=-1,
    ).to(tl.int64)
    k_per_req = tl.load(
        k_per_req_ptr + batch_offsets, mask=batch_mask, other=0
    ).to(tl.int32)
    topk_valid = (
        topk_mask
        & (offsets < k_per_req)
        & (topk_scores > -float("inf"))
        & (topk_scores < float("inf"))
        & (topk_indices >= 0)
        & (topk_indices < max_logical_pages)
    )

    recent_offsets = offsets - TOPK_WIDTH
    recent_mask = (
        batch_mask
        & (recent_offsets >= 0)
        & (recent_offsets < RECENT_WIDTH)
    )
    safe_recent = tl.where(recent_mask, recent_offsets, 0)
    recent_indices = tl.load(
        recent_indices_ptr + batch_offsets * RECENT_WIDTH + safe_recent,
        mask=recent_mask,
        other=-1,
    ).to(tl.int64)
    recent_is_valid = tl.load(
        recent_valid_ptr + batch_offsets * RECENT_WIDTH + safe_recent,
        mask=recent_mask,
        other=0,
    ).to(tl.int1)
    recent_is_valid &= (
        recent_mask
        & (recent_indices >= 0)
        & (recent_indices < max_logical_pages)
    )

    sentinel = 0x7FFFFFFF
    selected = tl.where(
        topk_valid,
        topk_indices.to(tl.int32),
        tl.where(recent_mask & recent_is_valid, recent_indices, sentinel),
    )
    sorted_indices = tl.sort(selected, dim=1, descending=False)
    valid = sorted_indices != sentinel
    valid_lengths = tl.sum(valid.to(tl.int32), axis=1)[:, None]

    sparse_mask = tl.load(
        sparse_mask_ptr + batch_offsets * sparse_mask_stride_b,
        mask=batch_mask,
        other=0,
    ).to(tl.int1)
    req_indices = tl.load(
        req_pool_indices_ptr + batch_offsets * req_pool_indices_stride_b,
        mask=batch_mask,
        other=0,
    ).to(tl.int64)
    req_valid = (req_indices >= 0) & (req_indices < num_req_slots)
    active = batch_mask & sparse_mask & req_valid & (valid_lengths > 0)
    active_valid_lengths = tl.where(active, valid_lengths, 0).to(tl.int32)
    last_page_lengths = tl.where(
        seq_lens > 0, (seq_lens - 1) % PAGE_SIZE + 1, 0
    )
    visible_lens = tl.where(
        active,
        (valid_lengths - 1) * PAGE_SIZE + last_page_lengths,
        0,
    ).to(tl.int32)
    tl.store(
        valid_lengths_ptr + batch_offsets,
        active_valid_lengths,
        mask=batch_mask,
    )
    tl.store(
        visible_kv_lens_ptr + batch_offsets,
        visible_lens,
        mask=batch_mask,
    )

    output_mask = batch_mask & (offsets < OUTPUT_WIDTH)
    output_valid = active & output_mask & valid
    safe_output = tl.where(offsets < OUTPUT_WIDTH, offsets, 0)
    tl.store(
        selected_output_ptr
        + batch_offsets * selected_output_stride_b
        + safe_output * selected_output_stride_p,
        tl.where(output_valid, sorted_indices, -1).to(tl.int32),
        mask=output_mask,
    )
    safe_req_indices = tl.where(req_valid, req_indices, 0)
    safe_sorted = tl.where(valid, sorted_indices, 0).to(tl.int64)
    first_tokens = tl.load(
        req_to_token_ptr
        + safe_req_indices * req_to_token_stride_b
        + safe_sorted * PAGE_SIZE * req_to_token_stride_t,
        mask=output_valid,
        other=0,
    ).to(tl.int64)
    tl.store(
        page_table_ptr
        + batch_offsets * page_table_stride_b
        + safe_output * page_table_stride_p,
        (first_tokens // PAGE_SIZE).to(tl.int32),
        mask=output_valid,
    )

    if UPDATE_LENGTHS:
        cache_seq_lens = tl.where(active, visible_lens, seq_lens).to(tl.int32)
        tl.store(
            cache_seqlens_ptr + batch_offsets * cache_seqlens_stride_b,
            cache_seq_lens,
            mask=batch_mask,
        )
        flat_cache_lens = tl.reshape(cache_seq_lens, (BLOCK_BATCH,))
        cumulative_lens = tl.cumsum(flat_cache_lens, axis=0)
        flat_batch = tl.arange(0, BLOCK_BATCH)
        tl.store(cu_seqlens_ptr, 0)
        tl.store(
            cu_seqlens_ptr + (flat_batch + 1) * cu_seqlens_stride_b,
            cumulative_lens,
            mask=flat_batch < BATCH_SIZE,
        )


@triton.jit
def _quest_finalize_to_fa3_metadata_kernel(
    topk_scores_ptr,
    topk_indices_ptr,
    k_per_req_ptr,
    recent_indices_ptr,
    recent_valid_ptr,
    selected_output_ptr,
    selected_output_stride_b,
    selected_output_stride_p,
    valid_lengths_ptr,
    visible_kv_lens_ptr,
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
    TOPK_WIDTH: tl.constexpr,
    RECENT_WIDTH: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    UPDATE_LENGTHS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    if batch_idx < BATCH_SIZE:
        seq_len = tl.load(seq_lens_ptr + batch_idx * seq_lens_stride_b).to(tl.int64)
        bounded_seq_len = tl.minimum(tl.maximum(seq_len, 0), req_to_token_width)
        max_logical_pages = (bounded_seq_len + PAGE_SIZE - 1) // PAGE_SIZE

        topk_mask = offsets < TOPK_WIDTH
        safe_topk = tl.where(topk_mask, offsets, 0)
        topk_scores = tl.load(
            topk_scores_ptr + batch_idx * TOPK_WIDTH + safe_topk,
            mask=topk_mask,
            other=-float("inf"),
        ).to(tl.float32)
        topk_indices = tl.load(
            topk_indices_ptr + batch_idx * TOPK_WIDTH + safe_topk,
            mask=topk_mask,
            other=-1,
        ).to(tl.int64)
        k_per_req = tl.load(k_per_req_ptr + batch_idx).to(tl.int32)
        topk_valid = (
            topk_mask
            & (offsets < k_per_req)
            & (topk_scores > -float("inf"))
            & (topk_scores < float("inf"))
            & (topk_indices >= 0)
            & (topk_indices < max_logical_pages)
        )

        recent_offsets = offsets - TOPK_WIDTH
        recent_mask = (recent_offsets >= 0) & (recent_offsets < RECENT_WIDTH)
        safe_recent = tl.where(recent_mask, recent_offsets, 0)
        recent_indices = tl.load(
            recent_indices_ptr + batch_idx * RECENT_WIDTH + safe_recent,
            mask=recent_mask,
            other=-1,
        ).to(tl.int64)
        recent_is_valid = tl.load(
            recent_valid_ptr + batch_idx * RECENT_WIDTH + safe_recent,
            mask=recent_mask,
            other=0,
        ).to(tl.int1)
        recent_is_valid &= (
            recent_mask
            & (recent_indices >= 0)
            & (recent_indices < max_logical_pages)
        )

        sentinel = 0x7FFFFFFF
        selected = tl.where(
            topk_valid,
            topk_indices.to(tl.int32),
            tl.where(recent_mask & recent_is_valid, recent_indices, sentinel),
        )
        sorted_indices = tl.sort(selected, descending=False)
        valid = sorted_indices != sentinel
        safe_sorted = tl.where(valid, sorted_indices, 0)
        valid_length = tl.sum(valid.to(tl.int32), axis=0)

        use_sparse = tl.load(sparse_mask_ptr + batch_idx * sparse_mask_stride_b).to(
            tl.int1
        )
        req_idx = tl.load(
            req_pool_indices_ptr + batch_idx * req_pool_indices_stride_b
        ).to(tl.int64)
        req_valid = (req_idx >= 0) & (req_idx < num_req_slots)
        active = use_sparse & req_valid & (valid_length > 0)
        active_valid_length = tl.where(active, valid_length, 0).to(tl.int32)
        last_page_length = tl.where(seq_len > 0, (seq_len - 1) % PAGE_SIZE + 1, 0)
        visible_length = tl.where(
            active,
            (valid_length - 1) * PAGE_SIZE + last_page_length,
            0,
        ).to(tl.int32)
        tl.store(valid_lengths_ptr + batch_idx, active_valid_length)
        tl.store(visible_kv_lens_ptr + batch_idx, visible_length)

        output_mask = offsets < OUTPUT_WIDTH
        output_valid = active & output_mask & valid
        safe_output = tl.where(output_mask, offsets, 0)
        tl.store(
            selected_output_ptr
            + batch_idx * selected_output_stride_b
            + safe_output * selected_output_stride_p,
            tl.where(output_valid, sorted_indices, -1).to(tl.int32),
            mask=output_mask,
        )

        safe_req_idx = tl.where(req_valid, req_idx, 0)
        first_tokens = tl.load(
            req_to_token_ptr
            + safe_req_idx * req_to_token_stride_b
            + safe_sorted.to(tl.int64) * PAGE_SIZE * req_to_token_stride_t,
            mask=output_valid,
            other=0,
        ).to(tl.int64)
        tl.store(
            page_table_ptr
            + batch_idx * page_table_stride_b
            + safe_output * page_table_stride_p,
            (first_tokens // PAGE_SIZE).to(tl.int32),
            mask=output_valid,
        )

    # A single extra program computes the small-batch prefix sum.  Recounting
    # valid pages avoids a second launch and needs no inter-program barrier.
    if UPDATE_LENGTHS and batch_idx == BATCH_SIZE:
        cumulative_length = 0
        for idx in range(BATCH_SIZE):
            seq_len = tl.load(seq_lens_ptr + idx * seq_lens_stride_b).to(tl.int64)
            bounded_seq_len = tl.minimum(tl.maximum(seq_len, 0), req_to_token_width)
            max_logical_pages = (bounded_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
            row_valid_length = _quest_count_valid_pages(
                topk_scores_ptr,
                topk_indices_ptr,
                k_per_req_ptr,
                recent_indices_ptr,
                recent_valid_ptr,
                idx,
                max_logical_pages,
                TOPK_WIDTH,
                RECENT_WIDTH,
                BLOCK_SIZE,
            ).to(tl.int64)
            row_sparse = tl.load(sparse_mask_ptr + idx * sparse_mask_stride_b).to(
                tl.int1
            )
            row_req_idx = tl.load(
                req_pool_indices_ptr + idx * req_pool_indices_stride_b
            ).to(tl.int64)
            row_req_valid = (row_req_idx >= 0) & (row_req_idx < num_req_slots)
            row_active = row_sparse & row_req_valid & (row_valid_length > 0)
            last_page_length = tl.where(seq_len > 0, (seq_len - 1) % PAGE_SIZE + 1, 0)
            sparse_seq_len = (row_valid_length - 1) * PAGE_SIZE + last_page_length
            cache_seq_len = tl.where(row_active, sparse_seq_len, seq_len).to(tl.int32)
            tl.store(cache_seqlens_ptr + idx * cache_seqlens_stride_b, cache_seq_len)
            tl.store(cu_seqlens_ptr + idx * cu_seqlens_stride_b, cumulative_length)
            cumulative_length += cache_seq_len
        tl.store(
            cu_seqlens_ptr + BATCH_SIZE * cu_seqlens_stride_b, cumulative_length
        )


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


def quest_finalize_to_flashattention_metadata_(
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    k_per_req: torch.Tensor,
    recent_indices: torch.Tensor,
    recent_valid: torch.Tensor,
    selected_output: torch.Tensor,
    valid_lengths: torch.Tensor,
    visible_kv_lens: torch.Tensor,
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
    """Finalize exact Quest top-k directly into SelectionResult and FA3 metadata."""
    if not topk_scores.is_cuda or torch.version.hip is not None:
        raise ValueError("Quest direct FA3 metadata requires NVIDIA CUDA tensors")
    if topk_scores.ndim != 2 or topk_indices.shape != topk_scores.shape:
        raise ValueError("top-k scores and indices must have matching 2D shapes")
    batch_size, topk_width = topk_scores.shape
    if recent_indices.ndim != 2 or recent_indices.shape[0] != batch_size:
        raise ValueError("recent_indices must have shape [batch, recent]")
    if recent_valid.shape != recent_indices.shape or recent_valid.dtype != torch.bool:
        raise ValueError("recent_valid must be a matching bool tensor")
    output_width = topk_width + recent_indices.shape[1]
    if output_width <= 0 or output_width > QUEST_DIRECT_METADATA_MAX_WIDTH:
        raise ValueError(
            f"Quest direct metadata width must be in [1, "
            f"{QUEST_DIRECT_METADATA_MAX_WIDTH}]"
        )
    if selected_output.shape != (batch_size, output_width):
        raise ValueError("selected_output must match the combined selection width")
    for name, tensor in (
        ("k_per_req", k_per_req),
        ("valid_lengths", valid_lengths),
        ("visible_kv_lens", visible_kv_lens),
        ("sparse_mask", sparse_mask),
        ("seq_lens", seq_lens),
        ("req_pool_indices", req_pool_indices),
    ):
        if tensor.shape != (batch_size,):
            raise ValueError(f"{name} must have shape [batch]")
    if req_to_token.ndim != 2 or not all(req_to_token.shape):
        raise ValueError("req_to_token must be a non-empty 2D tensor")
    if page_table.ndim != 2 or page_table.shape[0] != batch_size:
        raise ValueError("page_table must have shape [batch, pages]")
    if page_table.shape[1] < output_width:
        raise ValueError("FA3 page table is narrower than the Quest selection")
    if cache_seqlens_int32.shape != (batch_size,):
        raise ValueError("cache_seqlens_int32 must have shape [batch]")
    if cu_seqlens_k.shape != (batch_size + 1,):
        raise ValueError("cu_seqlens_k must have shape [batch + 1]")
    if topk_scores.dtype != torch.float32:
        raise ValueError("top-k scores must use float32")
    integer_tensors = (
        topk_indices,
        k_per_req,
        recent_indices,
        selected_output,
        valid_lengths,
        visible_kv_lens,
        seq_lens,
        req_pool_indices,
        req_to_token,
    )
    if any(t.dtype not in (torch.int32, torch.int64) for t in integer_tensors):
        raise ValueError("Quest direct metadata index/length tensors must be integers")
    if selected_output.dtype != torch.int32:
        raise ValueError("selected_output must use int32")
    if valid_lengths.dtype != torch.int32 or visible_kv_lens.dtype != torch.int32:
        raise ValueError("Quest output lengths must use int32")
    if sparse_mask.dtype != torch.bool:
        raise ValueError("sparse_mask must use bool")
    if page_table.dtype != torch.int32:
        raise ValueError("FA3 page_table must use int32")
    if cache_seqlens_int32.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise ValueError("FA3 sequence metadata must use int32")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    tensors = (
        topk_indices,
        k_per_req,
        recent_indices,
        recent_valid,
        selected_output,
        valid_lengths,
        visible_kv_lens,
        sparse_mask,
        seq_lens,
        req_pool_indices,
        req_to_token,
        page_table,
        cache_seqlens_int32,
        cu_seqlens_k,
    )
    if any(t.device != topk_scores.device for t in tensors):
        raise ValueError("Quest direct metadata tensors must share one device")
    if not all(
        t.is_contiguous()
        for t in (
            topk_scores,
            topk_indices,
            k_per_req,
            recent_indices,
            recent_valid,
            selected_output,
            valid_lengths,
            visible_kv_lens,
        )
    ):
        raise ValueError("Quest direct selection tensors must be contiguous")
    if batch_size == 0:
        return

    block_size = triton.next_power_of_2(output_width)
    if batch_size <= 4 and block_size <= 256:
        block_batch = triton.next_power_of_2(batch_size)
        _quest_finalize_to_fa3_metadata_small_batch_kernel[(1,)](
            topk_scores,
            topk_indices,
            k_per_req,
            recent_indices,
            recent_valid,
            selected_output,
            selected_output.stride(0),
            selected_output.stride(1),
            valid_lengths,
            visible_kv_lens,
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
            TOPK_WIDTH=topk_width,
            RECENT_WIDTH=recent_indices.shape[1],
            OUTPUT_WIDTH=output_width,
            BATCH_SIZE=batch_size,
            PAGE_SIZE=page_size,
            UPDATE_LENGTHS=update_lengths,
            BLOCK_BATCH=block_batch,
            BLOCK_SIZE=block_size,
            num_warps=8 if block_batch * block_size >= 256 else 4,
        )
        return

    _quest_finalize_to_fa3_metadata_kernel[
        (batch_size + (1 if update_lengths else 0),)
    ](
        topk_scores,
        topk_indices,
        k_per_req,
        recent_indices,
        recent_valid,
        selected_output,
        selected_output.stride(0),
        selected_output.stride(1),
        valid_lengths,
        visible_kv_lens,
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
        TOPK_WIDTH=topk_width,
        RECENT_WIDTH=recent_indices.shape[1],
        OUTPUT_WIDTH=output_width,
        BATCH_SIZE=batch_size,
        PAGE_SIZE=page_size,
        UPDATE_LENGTHS=update_lengths,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
