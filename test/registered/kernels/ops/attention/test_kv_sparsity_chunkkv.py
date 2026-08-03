import math

import pytest
import torch

from sglang.srt.mem_cache.sparsity.kernels.chunkkv_score import (
    chunkkv_accumulate_page_scores_,
)
from sglang.srt.mem_cache.sparsity.kernels.chunkkv_select import chunkkv_select
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="ChunkKV fused kernels require NVIDIA CUDA",
)


def test_chunkkv_fused_gqa_scores_match_explicit_qk_chunk_reference():
    torch.manual_seed(17)
    device = torch.device("cuda")
    page_size, obs_window = 4, 3
    batch_size, seq_len = 2, 13
    query_heads, kv_heads, head_dim = 4, 2, 8
    extend_lens = torch.tensor([5, 4], dtype=torch.int32, device=device)
    extend_start = torch.tensor([0, 5], dtype=torch.int32, device=device)
    query = torch.randn((9, query_heads, head_dim), dtype=torch.bfloat16, device=device)
    key_buffer = torch.randn(
        (32, kv_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    req_to_token = torch.zeros((3, 16), dtype=torch.int32, device=device)
    req_to_token[1] = torch.arange(2, 18, dtype=torch.int32, device=device)
    req_to_token[2] = torch.arange(16, 32, dtype=torch.int32, device=device)
    req_indices = torch.tensor([1, 2], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len, seq_len], dtype=torch.int32, device=device)
    active = torch.tensor([True, True], device=device)
    score_pool = torch.empty((3, 4), dtype=torch.float32, device=device)

    chunkkv_accumulate_page_scores_(
        query=query,
        extend_start=extend_start,
        extend_lens=extend_lens,
        active_mask=active,
        req_pool_indices=req_indices,
        seq_lens=seq_lens,
        req_to_token=req_to_token,
        key_buffer=key_buffer,
        score_pool=score_pool,
        page_size=page_size,
        recent_pages=1,
        obs_window=obs_window,
        reset=True,
    )

    reference = torch.full((batch_size, 4), -float("inf"), device=device)
    group_size = query_heads // kv_heads
    for batch_idx in range(batch_size):
        obs_end = int(extend_start[batch_idx] + extend_lens[batch_idx])
        q_obs = query[obs_end - obs_window : obs_end].float()
        q_grouped = q_obs.view(obs_window, kv_heads, group_size, head_dim).mean(2)
        physical = req_to_token[req_indices[batch_idx], :seq_len].long()
        keys = key_buffer[physical].float()
        token_scores = torch.einsum("ohd,shd->os", q_grouped, keys).mean(0) / math.sqrt(
            head_dim
        )
        for page_idx in range(3):
            reference[batch_idx, page_idx] = token_scores[
                page_idx * page_size : (page_idx + 1) * page_size
            ].sum()

    torch.testing.assert_close(
        score_pool[req_indices.long()], reference, atol=2e-4, rtol=2e-4
    )


def test_chunkkv_fused_selection_handles_mixed_rows_and_partial_page():
    device = torch.device("cuda")
    base_pages = torch.tensor(
        [[0, 2], [1, 3], [0, 1]], dtype=torch.int32, device=device
    )
    state_valid = torch.tensor([False, True, True], device=device)
    seq_lens = torch.tensor([11, 7], dtype=torch.int32, device=device)
    req_indices = torch.tensor([1, 2], dtype=torch.int32, device=device)

    selected, lengths, visible, sparse = chunkkv_select(
        seq_lens,
        req_indices,
        base_pages,
        state_valid,
        page_size=2,
        page_budget=3,
        recent_pages=1,
        min_sparse_tokens=8,
    )

    torch.testing.assert_close(
        selected,
        torch.tensor([[1, 3, 5], [-1, -1, -1]], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(
        lengths, torch.tensor([3, 0], dtype=torch.int32, device=device)
    )
    torch.testing.assert_close(
        visible, torch.tensor([5, 0], dtype=torch.int32, device=device)
    )
    torch.testing.assert_close(sparse, torch.tensor([True, False], device=device))
