"""Unit tests for Quest + HiSparse integration (non-native sparse MHA path).

These tests focus on the pieces that do *not* require a real attention
backend or a live model: batched page selection and the page→token
expansion helper that glues Quest output to the HiSparse swap-in kernel.

The end-to-end integration test (Qwen + Quest + HiSparse) lives under
``test/registered/8-gpu-models/`` and requires GPUs.
"""

import unittest

import torch


class TestExpandSelectedPagesToTokens(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_basic_expansion(self):
        from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
            _expand_selected_pages_to_tokens,
        )

        selected_pages = torch.tensor(
            [[0, 2, -1], [1, 3, 4]], dtype=torch.int32, device=self.device
        )
        seq_lens = torch.tensor([10, 20], dtype=torch.int32, device=self.device)
        out = _expand_selected_pages_to_tokens(
            selected_pages=selected_pages,
            seq_lens=seq_lens,
            page_size=2,
            num_top_k=8,
        )
        self.assertEqual(out.shape, (2, 8))
        # Row 0: pages {0,2} -> tokens {0,1, 4,5}, rest padded with -1.
        self.assertEqual(
            sorted(x for x in out[0].tolist() if x >= 0),
            [0, 1, 4, 5],
        )
        # Row 1: pages {1,3,4} -> tokens {2,3, 6,7, 8,9}.
        self.assertEqual(
            sorted(x for x in out[1].tolist() if x >= 0),
            [2, 3, 6, 7, 8, 9],
        )

    def test_truncates_at_seq_len(self):
        from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
            _expand_selected_pages_to_tokens,
        )

        selected_pages = torch.tensor(
            [[2]], dtype=torch.int32, device=self.device
        )
        seq_lens = torch.tensor([5], dtype=torch.int32, device=self.device)
        out = _expand_selected_pages_to_tokens(
            selected_pages=selected_pages,
            seq_lens=seq_lens,
            page_size=2,
            num_top_k=4,
        )
        # Page 2 = tokens {4,5}, but seq_len=5 drops token 5.
        non_neg = [x for x in out[0].tolist() if x >= 0]
        self.assertEqual(non_neg, [4])


@unittest.skipUnless(torch.cuda.is_available(), "Quest test needs CUDA")
class TestQuestBatchedRetrieve(unittest.TestCase):
    def _make_quest(self, page_size=4, num_pages=8, num_layers=2):
        from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
            SparseConfig,
        )
        from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import (
            QuestAlgorithm,
        )

        device = torch.device("cuda")

        class _Layer:
            def __init__(self, k_buffer):
                self._k = k_buffer

        class _KVPool:
            def __init__(self, k_buffer):
                self._k = k_buffer

            def get_key_buffer(self, layer_id):
                return self._k

        # Fake K buffer big enough for num_pages * page_size tokens with 2 heads.
        kv_heads, head_dim = 2, 4
        total_tokens = num_pages * page_size
        k_buffer = torch.randn(
            total_tokens, kv_heads, head_dim, dtype=torch.float32, device=device
        )
        kv_pool = _KVPool(k_buffer)

        class _ReqToTokenPool:
            pass

        rtok = _ReqToTokenPool()
        rtok.req_to_token = torch.arange(
            total_tokens, device=device, dtype=torch.int64
        ).view(1, total_tokens)
        rtok.max_context_len = total_tokens

        class _States:
            pass

        states = _States()
        states.prompt_lens = torch.tensor([total_tokens], device=device)
        states.repr_constructed = torch.tensor([True], device=device)
        states.last_constructed_page = torch.tensor([num_pages], device=device)

        config = SparseConfig(
            top_k=page_size * 2,  # 2 pages worth of tokens
            device_buffer_size=page_size * 4,
            host_to_device_ratio=2,
            algorithm="quest",
            backend="fa3",
            page_size=page_size,
            min_sparse_prompt_len=0,
            sparse_extra_config={},
        )

        algo = QuestAlgorithm(config, device)
        algo.initialize_representation_pool(
            start_layer=0,
            end_layer=num_layers,
            token_to_kv_pool=kv_pool,
            req_to_token_pool=rtok,
            states=states,
        )
        # Populate min/max by calling the batched construct path.
        algo._compute_page_representations(
            layer_id=0,
            reqs=torch.tensor([0], device=device),
            seq_lens=torch.tensor([total_tokens], device=device),
            start_page=0,
            end_page=torch.tensor([num_pages], device=device),
            k_buffer=k_buffer,
        )
        return algo, rtok, total_tokens

    def test_retrieve_topk_shapes(self):
        algo, rtok, total_tokens = self._make_quest()
        device = torch.device("cuda")

        class _FB:
            pass

        fb = _FB()
        fb.seq_lens = torch.tensor([total_tokens], device=device, dtype=torch.int64)

        q = torch.randn(1, 2, 4, device=device)  # [bs, kv_heads, head_dim]
        sparse_mask = torch.tensor([True], device=device)
        pages, lens = algo.retrieve_topk(
            queries=q,
            layer_id=0,
            req_pool_indices=torch.tensor([0], device=device, dtype=torch.int64),
            sparse_mask=sparse_mask,
            forward_batch=fb,
        )
        self.assertEqual(pages.dim(), 2)
        self.assertEqual(pages.shape[0], 1)
        # Output width must equal ceil(top_k / page_size) — here 2*page / page = 2.
        self.assertEqual(pages.shape[1], 2)
        self.assertGreaterEqual(int(lens[0]), 1)
        # No padding slots should point beyond the page range.
        valid = pages[0][pages[0] >= 0]
        self.assertTrue(torch.all(valid < (total_tokens // 4)))


if __name__ == "__main__":
    unittest.main()
