import math
import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported
from sglang.srt.mem_cache.sparsity.backend.visibility_adaptor import (
    FlashAttentionVisibilityAdaptor,
    HBMResidentPlacement,
)
from sglang.srt.mem_cache.sparsity.config import KVSparsityConfig
from sglang.srt.mem_cache.sparsity.contracts import (
    Granularity,
    SelectionContext,
    SelectionResult,
)
from sglang.srt.mem_cache.sparsity.kernels.quest_score import quest_page_scores
from sglang.srt.mem_cache.sparsity.policies.quest import QuestPolicy
from sglang.srt.mem_cache.sparsity.policies.streaming_llm import (
    StreamingLLMPolicy,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(
    torch.cuda.is_available() and _is_fa3_supported(), "FA3 requires a supported GPU"
)
class TestKVSparsityFA3Visibility(unittest.TestCase):
    def test_streaming_llm_fused_selection_matches_page_semantics(self):
        device = torch.device("cuda")
        policy = StreamingLLMPolicy(
            KVSparsityConfig(
                policy="streaming_llm",
                page_size=16,
                min_sparse_tokens=64,
                policy_config={"sink_pages": 1, "recent_pages": 2},
            ),
            device,
        )
        result = policy.select(
            SelectionContext(
                query=None,
                layer_id=0,
                req_pool_indices=torch.arange(4, device=device),
                seq_lens=torch.tensor([32, 48, 64, 70], device=device),
            )
        )

        torch.testing.assert_close(
            result.logical_indices,
            torch.tensor(
                [[-1, -1, -1], [-1, -1, -1], [0, 2, 3], [0, 3, 4]],
                dtype=torch.int32,
                device=device,
            ),
        )
        torch.testing.assert_close(
            result.valid_lengths,
            torch.tensor([0, 0, 3, 3], dtype=torch.int32, device=device),
        )
        torch.testing.assert_close(
            result.visible_kv_lens,
            torch.tensor([0, 0, 48, 38], dtype=torch.int32, device=device),
        )
        torch.testing.assert_close(
            result.sparse_mask,
            torch.tensor([False, False, True, True], device=device),
        )

    def test_quest_vectorized_gqa_score_matches_reference(self):
        torch.manual_seed(7)
        device = torch.device("cuda")
        batch_size, num_pages = 2, 5
        query_heads, kv_heads, head_dim = 4, 2, 8
        queries = torch.randn(
            (batch_size, query_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        page_k_min = torch.randn(
            (7, kv_heads, head_dim), dtype=torch.bfloat16, device=device
        )
        page_k_max = page_k_min + torch.rand_like(page_k_min)
        page_valid = torch.tensor(
            [True, True, False, True, True, True, True], device=device
        )
        physical_pages = torch.tensor(
            [[0, 1, 2, 3, 4], [4, 5, 6, -1, -1]],
            dtype=torch.int32,
            device=device,
        )
        active_mask = torch.tensor([True, True], device=device)
        history_page_counts = torch.tensor([4, 3], dtype=torch.int32, device=device)

        scores = quest_page_scores(
            queries,
            page_k_min,
            page_k_max,
            page_valid,
            physical_pages,
            active_mask=active_mask,
            history_page_counts=history_page_counts,
        )
        reference = torch.full_like(scores, -float("inf"))
        group_size = query_heads // kv_heads
        for batch_idx in range(batch_size):
            for page_idx in range(num_pages):
                physical_page = int(physical_pages[batch_idx, page_idx].item())
                if (
                    page_idx >= int(history_page_counts[batch_idx].item())
                    or physical_page < 0
                    or not bool(page_valid[physical_page].item())
                ):
                    continue
                head_bounds = []
                for query_head in range(query_heads):
                    kv_head = query_head // group_size
                    query = queries[batch_idx, query_head].float()
                    bound_keys = torch.where(
                        query >= 0,
                        page_k_max[physical_page, kv_head].float(),
                        page_k_min[physical_page, kv_head].float(),
                    )
                    head_bounds.append((query * bound_keys).sum())
                reference[batch_idx, page_idx] = torch.stack(head_bounds).max()

        torch.testing.assert_close(scores, reference, atol=1e-4, rtol=1e-4)

    def test_noncontiguous_sparse_page_table_matches_reference_attention(self):
        torch.manual_seed(1)
        device = torch.device("cuda")
        dtype = torch.bfloat16

        query = torch.randn((1, 1, 64), device=device, dtype=dtype)
        key_cache = torch.randn((8, 1, 1, 64), device=device, dtype=dtype)
        value_cache = torch.randn((8, 1, 1, 64), device=device, dtype=dtype)
        req_to_token = torch.zeros((2, 8), device=device, dtype=torch.int32)
        req_to_token[1] = torch.arange(8, device=device, dtype=torch.int32)
        metadata = SimpleNamespace(
            page_table=req_to_token[1:2].clone(),
            cache_seqlens_int32=torch.tensor([8], device=device, dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 8], device=device, dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1], device=device, dtype=torch.int32),
            max_seq_len_k=8,
            scheduler_metadata=None,
        )
        dense_page_table = metadata.page_table.clone()
        adaptor = FlashAttentionVisibilityAdaptor(
            HBMResidentPlacement(req_to_token=req_to_token, page_size=1)
        )
        result = SelectionResult(
            granularity=Granularity.PAGE,
            logical_indices=torch.tensor(
                [[0, 1, 6, 7]], device=device, dtype=torch.int32
            ),
            valid_lengths=torch.tensor([4], device=device, dtype=torch.int32),
            visible_kv_lens=torch.tensor([4], device=device, dtype=torch.int32),
            sparse_mask=torch.tensor([True], device=device),
        )

        adaptor.capture_dense_metadata(metadata)
        adaptor.apply(
            result,
            metadata,
            SimpleNamespace(req_pool_indices=torch.tensor([1], device=device)),
        )
        output = flash_attn_with_kvcache(
            q=query,
            k_cache=key_cache,
            v_cache=value_cache,
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens_int32,
            cu_seqlens_q=metadata.cu_seqlens_q,
            max_seqlen_q=1,
            causal=False,
            softmax_scale=1 / math.sqrt(64),
            ver=3,
        )

        selected_pages = metadata.page_table[0, :4].long()
        selected_keys = key_cache[selected_pages, 0, 0].float()
        selected_values = value_cache[selected_pages, 0, 0].float()
        weights = torch.softmax(
            query[0, 0].float() @ selected_keys.T / math.sqrt(64), dim=-1
        )
        reference = weights @ selected_values

        torch.testing.assert_close(
            output[0, 0].float(), reference, atol=0.02, rtol=0.02
        )
        adaptor.restore_dense_metadata(metadata)
        torch.testing.assert_close(metadata.page_table, dense_page_table)

    def test_quest_page_selection_matches_selected_page_reference(self):
        torch.manual_seed(2)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        page_size = 2

        query = torch.randn((1, 1, 64), device=device, dtype=dtype)
        flat_keys = torch.randn((10, 1, 64), device=device, dtype=dtype)
        flat_values = torch.randn((10, 1, 64), device=device, dtype=dtype)
        key_cache = flat_keys.view(5, page_size, 1, 64)
        value_cache = flat_values.view(5, page_size, 1, 64)
        req_to_token = torch.zeros((2, 8), device=device, dtype=torch.int32)
        req_to_token[1] = torch.arange(2, 10, device=device, dtype=torch.int32)
        req_pool = SimpleNamespace(req_to_token=req_to_token)
        policy = QuestPolicy(
            KVSparsityConfig(
                policy="quest",
                page_size=page_size,
                min_sparse_tokens=0,
                policy_config={"page_budget": 2, "recent_pages": 1},
            ),
            device,
            token_to_kv_pool=SimpleNamespace(get_key_buffer=lambda layer_id: flat_keys),
            req_to_token_pool=req_pool,
        )
        metadata = SimpleNamespace(
            page_table=torch.tensor([[1, 2, 3, 4]], device=device, dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([8], device=device, dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 8], device=device, dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1], device=device, dtype=torch.int32),
            max_seq_len_k=8,
            scheduler_metadata=None,
        )
        prefill_context = SelectionContext(
            query=None,
            layer_id=0,
            req_pool_indices=torch.tensor([1], device=device),
            seq_lens=torch.tensor([8], device=device),
            forward_batch=SimpleNamespace(
                forward_mode=SimpleNamespace(is_decode=lambda: False)
            ),
            metadata=metadata,
        )
        policy.on_attention_complete(prefill_context)
        result = policy.select(
            SelectionContext(
                query=query,
                layer_id=0,
                req_pool_indices=prefill_context.req_pool_indices,
                seq_lens=prefill_context.seq_lens,
                forward_batch=SimpleNamespace(
                    forward_mode=SimpleNamespace(is_decode=lambda: True)
                ),
                metadata=metadata,
            )
        )
        adaptor = FlashAttentionVisibilityAdaptor(
            HBMResidentPlacement(req_to_token=req_to_token, page_size=page_size)
        )
        adaptor.capture_dense_metadata(metadata)
        adaptor.apply(
            result,
            metadata,
            SimpleNamespace(
                req_pool_indices=prefill_context.req_pool_indices,
                seq_lens=prefill_context.seq_lens,
            ),
        )

        output = flash_attn_with_kvcache(
            q=query,
            k_cache=key_cache,
            v_cache=value_cache,
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens_int32,
            cu_seqlens_q=metadata.cu_seqlens_q,
            max_seqlen_q=1,
            causal=False,
            softmax_scale=1 / math.sqrt(64),
            ver=3,
        )
        self.assertEqual(result.valid_lengths.item(), 2)
        selected_pages = metadata.page_table[0, :2].long()
        selected_keys = key_cache[selected_pages, :, 0].reshape(-1, 64).float()
        selected_values = value_cache[selected_pages, :, 0].reshape(-1, 64).float()
        weights = torch.softmax(
            query[0, 0].float() @ selected_keys.T / math.sqrt(64), dim=-1
        )
        reference = weights @ selected_values

        torch.testing.assert_close(
            output[0, 0].float(), reference, atol=0.02, rtol=0.02
        )

    def test_quest_decode_updates_only_newly_completed_pages(self):
        device = torch.device("cuda")
        page_size = 2
        keys = torch.zeros((10, 1, 4), device=device, dtype=torch.bfloat16)
        keys[2:5] = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0]], [[2.0, 1.0, 5.0, 3.0]], [[7.0, 0.0, -1.0, 2.0]]],
            device=device,
            dtype=torch.bfloat16,
        )
        req_to_token = torch.zeros((2, 8), device=device, dtype=torch.int32)
        req_to_token[1] = torch.arange(2, 10, device=device, dtype=torch.int32)
        policy = QuestPolicy(
            KVSparsityConfig(
                policy="quest",
                page_size=page_size,
                min_sparse_tokens=0,
                policy_config={"page_budget": 2, "recent_pages": 1},
            ),
            device,
            token_to_kv_pool=SimpleNamespace(get_key_buffer=lambda layer_id: keys),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        )
        req_indices = torch.tensor([1], device=device)
        policy.on_attention_complete(
            SelectionContext(
                query=None,
                layer_id=0,
                req_pool_indices=req_indices,
                seq_lens=torch.tensor([3], device=device),
                forward_batch=SimpleNamespace(
                    forward_mode=SimpleNamespace(is_decode=lambda: False)
                ),
                metadata=SimpleNamespace(
                    page_table=torch.zeros((1, 2), device=device, dtype=torch.int32)
                ),
            )
        )
        keys[5] = torch.tensor(
            [[[-3.0, 8.0, 4.0, -5.0]]], device=device, dtype=torch.bfloat16
        )
        policy.on_attention_complete(
            SelectionContext(
                query=torch.ones((1, 1, 4), device=device, dtype=torch.bfloat16),
                layer_id=0,
                req_pool_indices=req_indices,
                seq_lens=torch.tensor([4], device=device),
                forward_batch=SimpleNamespace(
                    forward_mode=SimpleNamespace(is_decode=lambda: True)
                ),
            )
        )

        torch.testing.assert_close(
            policy._page_k_min[0][2, 0],
            torch.tensor([-3.0, 0.0, -1.0, -5.0], device=device, dtype=torch.bfloat16),
        )
        torch.testing.assert_close(
            policy._page_k_max[0][2, 0],
            torch.tensor([7.0, 8.0, 4.0, 2.0], device=device, dtype=torch.bfloat16),
        )
        self.assertEqual(policy._last_completed_page[1].item(), 2)


if __name__ == "__main__":
    unittest.main()
