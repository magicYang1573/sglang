"""Unit tests for the non-native sparse decode path.

These tests focus on Python-level components that do NOT depend on CUDA
kernels (FA3 / hisparse.cuh), so they can run on any host with PyTorch.

Covered components:
  * :class:`SparseConfig` parsing from JSON (factory helpers).
  * :class:`QuestAlgorithm.retrieve_topk` token-level contract.
  * :class:`Fa3SparseDecodeAdapter.adapt_for_attn_metadata` metadata rewrite.
  * :class:`SparseAlgorithmController` lifecycle orchestration with mocks.

Run with::

    cd python && python -m pytest \
      ../test/registered/unit/mem_cache/test_sparse_decode_unit.py -q
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


# ---------------------------------------------------------------------------
# SparseConfig parsing
# ---------------------------------------------------------------------------


class TestSparseConfigFactory(CustomTestCase):
    def test_parse_sparse_decode_config_defaults(self):
        from sglang.srt.mem_cache.sparsity import parse_sparse_decode_config

        args = SimpleNamespace(sparse_decode_config=None)
        cfg = parse_sparse_decode_config(args)

        self.assertEqual(cfg.algorithm, "quest")
        self.assertEqual(cfg.backend, "fa3")
        self.assertEqual(cfg.page_size, 1)
        self.assertEqual(cfg.min_sparse_prompt_len, 4096)
        self.assertEqual(cfg.top_k, 2048)
        self.assertEqual(cfg.device_buffer_size, 4096)
        self.assertEqual(cfg.host_to_device_ratio, 2)

    def test_parse_sparse_decode_config_custom_json(self):
        from sglang.srt.mem_cache.sparsity import parse_sparse_decode_config

        raw = (
            '{"top_k": 128, "device_buffer_size": 256, "host_to_device_ratio": 3, '
            '"min_sparse_prompt_len": 1024, '
            '"sparse_extra_config": {"sparsity_ratio": 0.3, "num_recent_pages": 2, '
            '"quest_page_size": 4}}'
        )
        args = SimpleNamespace(sparse_decode_config=raw)
        cfg = parse_sparse_decode_config(args)

        self.assertEqual(cfg.top_k, 128)
        self.assertEqual(cfg.device_buffer_size, 256)
        self.assertEqual(cfg.host_to_device_ratio, 3)
        self.assertEqual(cfg.min_sparse_prompt_len, 1024)
        self.assertEqual(
            cfg.sparse_extra_config,
            {"sparsity_ratio": 0.3, "num_recent_pages": 2, "quest_page_size": 4},
        )

    def test_parse_hisparse_config_quest_uses_sparse_decode_defaults(self):
        from sglang.srt.mem_cache.sparsity import parse_hisparse_config

        raw = '{"algorithm": "quest", "top_k": 128, "device_buffer_size": 256}'
        args = SimpleNamespace(hisparse_config=raw)
        cfg = parse_hisparse_config(args)

        self.assertEqual(cfg.algorithm, "quest")
        self.assertEqual(cfg.backend, "fa3")
        self.assertEqual(cfg.page_size, 1)
        self.assertEqual(cfg.min_sparse_prompt_len, 4096)
        self.assertEqual(cfg.top_k, 128)
        self.assertEqual(cfg.device_buffer_size, 256)

    def test_device_buffer_size_must_exceed_top_k(self):
        from sglang.srt.mem_cache.sparsity import parse_sparse_decode_config

        args = SimpleNamespace(
            sparse_decode_config='{"top_k": 1024, "device_buffer_size": 512}'
        )
        with self.assertRaises(ValueError):
            parse_sparse_decode_config(args)


# ---------------------------------------------------------------------------
# QuestAlgorithm.retrieve_topk contract
# ---------------------------------------------------------------------------


class _FakeReqToTokenPool:
    def __init__(self, req_to_token: torch.Tensor, max_context_len: int):
        self.req_to_token = req_to_token
        self.max_context_len = max_context_len


class _FakeKVPool:
    def __init__(self, k_buffer):
        self._k_buffer = k_buffer

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self._k_buffer[layer_id]

    def register_mapping(self, mapping):
        self.mapping = mapping


class _FakeStates:
    def __init__(self, max_pool_size: int, device: torch.device):
        self.prompt_lens = torch.zeros(max_pool_size, dtype=torch.int64, device=device)
        self.repr_constructed = torch.zeros(
            max_pool_size, dtype=torch.bool, device=device
        )
        self.last_constructed_page = torch.zeros(
            max_pool_size, dtype=torch.int64, device=device
        )

    def register(self, idx, prompt_len):
        self.prompt_lens[idx] = prompt_len
        self.repr_constructed[idx] = False
        self.last_constructed_page[idx] = 0

    def clear(self, idx):
        self.prompt_lens[idx] = 0
        self.repr_constructed[idx] = False
        self.last_constructed_page[idx] = 0


class TestQuestRetrieveTopK(CustomTestCase):
    def setUp(self):
        from sglang.srt.mem_cache.sparsity import QuestAlgorithm, SparseConfig

        self.device = torch.device("cpu")
        self.top_k_budget = 8
        self.head_num = 2
        self.head_dim = 4
        self.num_layers = 1
        self.max_num_reqs = 4
        self.total_tokens = 64

        self.config = SparseConfig(
            top_k=self.top_k_budget,
            device_buffer_size=16,
            host_to_device_ratio=2,
            algorithm="quest",
            backend="fa3",
            page_size=1,
            min_sparse_prompt_len=0,
            sparse_extra_config={
                "sparsity_ratio": 0.5,
                "num_recent_pages": 2,
                "quest_page_size": 4,
            },
        )
        self.algorithm = QuestAlgorithm(self.config, self.device)

        # Minimal context pool: identity mapping (token position = logical row).
        req_to_token = (
            torch.arange(self.max_num_reqs * self.total_tokens)
            .view(self.max_num_reqs, self.total_tokens)
            .to(torch.int64)
        )
        self.req_to_token_pool = _FakeReqToTokenPool(
            req_to_token, max_context_len=self.total_tokens
        )

        k_buffer_tensor = torch.randn(
            self.max_num_reqs * self.total_tokens,
            self.head_num,
            self.head_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self.kv_pool = _FakeKVPool([k_buffer_tensor])
        self.states = _FakeStates(self.max_num_reqs, self.device)

        self.algorithm.initialize_representation_pool(
            start_layer=0,
            end_layer=self.num_layers,
            token_to_kv_pool=self.kv_pool,
            req_to_token_pool=self.req_to_token_pool,
            states=self.states,
        )

    def _run_construct(self, req_idx, seq_len):
        reqs = torch.tensor([req_idx], dtype=torch.int64, device=self.device)
        seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=self.device)
        num_pages = torch.tensor(
            [(seq_len + self.algorithm.page_size - 1) // self.algorithm.page_size],
            dtype=torch.int64,
            device=self.device,
        )
        self.algorithm._compute_page_representations(
            layer_id=0,
            reqs=reqs,
            seq_lens=seq_lens,
            start_page=0,
            end_page=num_pages,
            k_buffer=self.kv_pool.get_key_buffer(0),
        )

    def test_quest_page_size_is_independent_from_kv_page_size(self):
        self.assertEqual(self.config.page_size, 1)
        self.assertEqual(self.algorithm.kv_page_size, 1)
        self.assertEqual(self.algorithm.quest_page_size, 4)
        self.assertEqual(self.algorithm.page_size, 4)

    def test_output_shape_and_dtype(self):
        seq_len = 32
        req_idx = 1
        self.states.register(req_idx, seq_len)
        self._run_construct(req_idx, seq_len)

        queries = torch.randn(
            1, self.head_num * self.head_dim, dtype=torch.float32, device=self.device
        )
        sparse_mask = torch.ones(1, dtype=torch.bool, device=self.device)
        req_pool_indices = torch.tensor([req_idx], dtype=torch.int64, device=self.device)

        fwd = SimpleNamespace(
            seq_lens=torch.tensor([seq_len], dtype=torch.int64, device=self.device)
        )
        top_k_tokens, valid_lengths = self.algorithm.retrieve_topk(
            queries=queries,
            layer_id=0,
            req_pool_indices=req_pool_indices,
            sparse_mask=sparse_mask,
            forward_batch=fwd,
        )

        self.assertEqual(top_k_tokens.dtype, torch.int32)
        self.assertEqual(top_k_tokens.shape, (1, self.top_k_budget))
        self.assertEqual(valid_lengths.shape, (1,))
        length = int(valid_lengths[0].item())
        self.assertGreater(length, 0)
        self.assertLessEqual(length, self.top_k_budget)
        # Values in [0, seq_len)
        valid_slice = top_k_tokens[0, :length]
        self.assertTrue(torch.all(valid_slice >= 0))
        self.assertTrue(torch.all(valid_slice < seq_len))
        # Padding entries are -1
        if length < self.top_k_budget:
            self.assertTrue(torch.all(top_k_tokens[0, length:] == -1))

    def test_partial_quest_page_expands_to_token_positions(self):
        seq_len = 18
        req_idx = 2
        self.states.register(req_idx, seq_len)
        self._run_construct(req_idx, seq_len)

        queries = torch.randn(
            1, self.head_num * self.head_dim, dtype=torch.float32, device=self.device
        )
        fwd = SimpleNamespace(
            seq_lens=torch.tensor([seq_len], dtype=torch.int64, device=self.device)
        )
        top_k_tokens, valid_lengths = self.algorithm.retrieve_topk(
            queries=queries,
            layer_id=0,
            req_pool_indices=torch.tensor(
                [req_idx], dtype=torch.int64, device=self.device
            ),
            sparse_mask=torch.ones(1, dtype=torch.bool, device=self.device),
            forward_batch=fwd,
        )

        length = int(valid_lengths[0].item())
        self.assertGreater(length, 0)
        valid_tokens = top_k_tokens[0, :length]
        self.assertTrue(torch.all(valid_tokens >= 0))
        self.assertTrue(torch.all(valid_tokens < seq_len))
        self.assertIn(16, valid_tokens.tolist())
        self.assertIn(17, valid_tokens.tolist())

    def test_batched_variable_lengths_and_sparse_mask(self):
        seq_lens = [18, 9, 0]
        req_indices = torch.tensor([0, 1, 3], dtype=torch.int64, device=self.device)
        for req_idx, seq_len in zip(req_indices.tolist(), seq_lens):
            self.states.register(req_idx, seq_len)
            if seq_len > 0:
                self._run_construct(req_idx, seq_len)

        queries = torch.randn(
            len(seq_lens),
            self.head_num * self.head_dim,
            dtype=torch.float32,
            device=self.device,
        )
        fwd = SimpleNamespace(
            seq_lens=torch.tensor(seq_lens, dtype=torch.int64, device=self.device)
        )
        sparse_mask = torch.tensor([True, False, True], dtype=torch.bool)
        top_k_tokens, valid_lengths = self.algorithm.retrieve_topk(
            queries=queries,
            layer_id=0,
            req_pool_indices=req_indices,
            sparse_mask=sparse_mask,
            forward_batch=fwd,
        )

        self.assertEqual(top_k_tokens.shape, (3, self.top_k_budget))
        self.assertGreater(int(valid_lengths[0].item()), 0)
        self.assertEqual(int(valid_lengths[1].item()), 0)
        self.assertEqual(int(valid_lengths[2].item()), 0)
        self.assertTrue(torch.all(top_k_tokens[1] == -1))
        self.assertTrue(torch.all(top_k_tokens[2] == -1))

    def test_sparse_mask_false_returns_minus_one_row(self):
        seq_len = 16
        req_idx = 0
        self.states.register(req_idx, seq_len)
        self._run_construct(req_idx, seq_len)
        queries = torch.randn(
            1, self.head_num * self.head_dim, dtype=torch.float32, device=self.device
        )
        fwd = SimpleNamespace(
            seq_lens=torch.tensor([seq_len], dtype=torch.int64, device=self.device)
        )
        top_k_tokens, valid_lengths = self.algorithm.retrieve_topk(
            queries=queries,
            layer_id=0,
            req_pool_indices=torch.tensor([req_idx], dtype=torch.int64),
            sparse_mask=torch.zeros(1, dtype=torch.bool, device=self.device),
            forward_batch=fwd,
        )
        self.assertEqual(int(valid_lengths[0].item()), 0)
        self.assertTrue(torch.all(top_k_tokens[0] == -1))


class TestHiSparseAllocatorSemantics(CustomTestCase):
    def test_sparse_decode_available_size_matches_hisparse_external_contract(self):
        from sglang.srt.mem_cache.hisparse_memory_pool import (
            HiSparseTokenToKVPoolAllocator,
        )

        allocator = HiSparseTokenToKVPoolAllocator(
            size=8,
            page_size=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
            kvcache=_FakeKVPool([torch.empty(1)]),
            need_sort=False,
            host_to_device_ratio=2,
            sparse_decode_mode=True,
        )

        self.assertEqual(allocator.logical_attn_allocator.available_size(), 16)
        self.assertEqual(allocator.hisparse_attn_allocator.available_size(), 8)
        self.assertEqual(allocator.available_size(), 8)


# ---------------------------------------------------------------------------
# Fa3SparseDecodeAdapter
# ---------------------------------------------------------------------------


class _FakeFlashAttentionMetadata:
    """Minimal stand-in matching the fields Fa3SparseDecodeAdapter touches."""

    def __init__(self, page_table, cache_seqlens_int32):
        self.page_table = page_table
        self.cache_seqlens_int32 = cache_seqlens_int32
        self.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
        )
        self.max_seq_len_k = int(cache_seqlens_int32.max().item())
        self.scheduler_metadata = torch.ones(1, dtype=torch.int32)


class TestFa3SparseDecodeAdapter(CustomTestCase):
    def setUp(self):
        from sglang.srt.mem_cache.sparsity.backend.fa3_sparse_decode_adapter import (
            Fa3SparseDecodeAdapter,
        )

        self.device = torch.device("cpu")
        self.adapter = Fa3SparseDecodeAdapter(self.device)
        self.bs = 3
        self.top_k = 4
        # Dense metadata: 3 requests with seqlen 10, 20, 30.
        self.page_table_dense = torch.arange(
            self.bs * 32, dtype=torch.int32, device=self.device
        ).view(self.bs, 32)
        self.cache_seqlens = torch.tensor(
            [10, 20, 30], dtype=torch.int32, device=self.device
        )
        self.meta = _FakeFlashAttentionMetadata(
            self.page_table_dense.clone(), self.cache_seqlens.clone()
        )

    def test_dense_rows_unchanged_when_sparse_mask_false(self):
        self.adapter.save_original_metadata(self.meta)
        top_k_device_locs = torch.full(
            (self.bs, self.top_k), 777, dtype=torch.int32, device=self.device
        )
        valid_lengths = torch.zeros(self.bs, dtype=torch.int32, device=self.device)
        sparse_mask = torch.zeros(self.bs, dtype=torch.bool, device=self.device)

        out = self.adapter.adapt_for_attn_metadata(
            selected_indices=top_k_device_locs,
            valid_lengths=valid_lengths,
            sparse_mask=sparse_mask,
            current_metadata=self.meta,
            forward_batch=None,
            req_to_token=None,
            page_size=1,
            layer_id=0,
        )
        torch.testing.assert_close(out.page_table, self.page_table_dense)
        torch.testing.assert_close(out.cache_seqlens_int32, self.cache_seqlens)
        self.assertIsNotNone(out.scheduler_metadata)

    def test_sparse_rows_overwritten_with_top_k(self):
        self.adapter.save_original_metadata(self.meta)
        top_k_device_locs = torch.tensor(
            [
                [100, 101, -1, -1],
                [200, 201, 202, -1],
                [300, 301, 302, 303],
            ],
            dtype=torch.int32,
            device=self.device,
        )
        valid_lengths = torch.tensor([2, 3, 4], dtype=torch.int32, device=self.device)
        # Mark only rows 1 and 2 as sparse; row 0 keeps dense.
        sparse_mask = torch.tensor(
            [False, True, True], dtype=torch.bool, device=self.device
        )

        out = self.adapter.adapt_for_attn_metadata(
            selected_indices=top_k_device_locs,
            valid_lengths=valid_lengths,
            sparse_mask=sparse_mask,
            current_metadata=self.meta,
            forward_batch=None,
            req_to_token=None,
            page_size=1,
            layer_id=0,
        )

        # Row 0 (dense) unchanged.
        torch.testing.assert_close(out.page_table[0], self.page_table_dense[0])
        self.assertEqual(int(out.cache_seqlens_int32[0].item()), 10)

        # Rows 1 and 2 prefix replaced.
        self.assertEqual(int(out.page_table[1, 0].item()), 200)
        self.assertEqual(int(out.page_table[1, 1].item()), 201)
        self.assertEqual(int(out.page_table[1, 2].item()), 202)
        # invalid slot replaced with pad row 0
        self.assertEqual(int(out.page_table[1, 3].item()), 0)
        self.assertEqual(int(out.cache_seqlens_int32[1].item()), 3)

        self.assertEqual(int(out.page_table[2, 0].item()), 300)
        self.assertEqual(int(out.page_table[2, 3].item()), 303)
        self.assertEqual(int(out.cache_seqlens_int32[2].item()), 4)

        # cu_seqlens_k / max_seq_len_k rebuilt.
        expected_cu = torch.tensor([0, 10, 13, 17], dtype=torch.int32)
        torch.testing.assert_close(out.cu_seqlens_k, expected_cu)
        self.assertEqual(out.max_seq_len_k, 10)
        self.assertIsNone(out.scheduler_metadata)

    def test_consecutive_layers_reset_from_snapshot(self):
        """Each layer must restart from the original dense metadata."""
        self.adapter.save_original_metadata(self.meta)
        top_k_layer1 = torch.tensor(
            [[7, 7, 7, 7], [7, 7, 7, 7], [7, 7, 7, 7]],
            dtype=torch.int32,
            device=self.device,
        )
        sparse_mask = torch.ones(self.bs, dtype=torch.bool, device=self.device)
        valid_lengths = torch.tensor([4, 4, 4], dtype=torch.int32, device=self.device)

        # Layer 1 writes 7s.
        self.adapter.adapt_for_attn_metadata(
            selected_indices=top_k_layer1,
            valid_lengths=valid_lengths,
            sparse_mask=sparse_mask,
            current_metadata=self.meta,
            forward_batch=None,
            req_to_token=None,
            page_size=1,
            layer_id=0,
        )
        self.assertTrue(torch.all(self.meta.page_table[:, :4] == 7))

        # Layer 2 with sparse_mask all False should restore dense prefix.
        self.adapter.adapt_for_attn_metadata(
            selected_indices=top_k_layer1,
            valid_lengths=valid_lengths,
            sparse_mask=torch.zeros(self.bs, dtype=torch.bool, device=self.device),
            current_metadata=self.meta,
            forward_batch=None,
            req_to_token=None,
            page_size=1,
            layer_id=1,
        )
        torch.testing.assert_close(
            self.meta.page_table[:, :4], self.page_table_dense[:, :4]
        )


# ---------------------------------------------------------------------------
# SparseAlgorithmController lifecycle with mocks
# ---------------------------------------------------------------------------


class _MockAlgorithm:
    def __init__(self, top_k):
        self.top_k = top_k
        self.construct_calls = []
        self.update_calls = []
        self.init_calls = 0

    def initialize_representation_pool(
        self, start_layer, end_layer, token_to_kv_pool, req_to_token_pool, states
    ):
        self.init_calls += 1
        self.start_layer = start_layer
        self.end_layer = end_layer

    def retrieve_topk(
        self, queries, layer_id, req_pool_indices, sparse_mask, **kwargs
    ):
        bs = queries.shape[0]
        top_k_tokens = torch.full(
            (bs, self.top_k), -1, dtype=torch.int32, device=queries.device
        )
        valid_lengths = torch.zeros(bs, dtype=torch.int32, device=queries.device)
        # Default behavior: every sparse row picks tokens [0, 1, 2, ...].
        for i in range(bs):
            if bool(sparse_mask[i]):
                n = min(self.top_k, int(kwargs["forward_batch"].seq_lens[i].item()))
                top_k_tokens[i, :n] = torch.arange(n, dtype=torch.int32)
                valid_lengths[i] = n
        return top_k_tokens, valid_lengths

    def construct_representations(self, **kwargs):
        self.construct_calls.append(kwargs["layer_id"])

    def update_representations(self, **kwargs):
        self.update_calls.append(kwargs["layer_id"])


class _MockAdapter:
    def __init__(self):
        self.saved_metadata = None
        self.adapt_calls = []

    def save_original_metadata(self, metadata):
        self.saved_metadata = metadata

    def adapt_for_attn_metadata(self, **kwargs):
        self.adapt_calls.append(kwargs["layer_id"])
        return kwargs["current_metadata"]


class _MockCoord:
    """Just enough interface for SparseAlgorithmController.bind_io."""

    def __init__(self, k_buffer):
        self.mem_pool_device = _FakeKVPool([k_buffer])
        self.swap_calls = []

    def swap_in_selected_pages(
        self, req_pool_indices, seq_lens, top_k_result, layer_id
    ):
        self.swap_calls.append((layer_id, top_k_result.clone()))
        # Echo the token indices as "device locs" for the test.
        return top_k_result


class TestSparseAlgorithmController(CustomTestCase):
    def setUp(self):
        from sglang.srt.mem_cache.sparsity import SparseAlgorithmController, SparseConfig

        self.device = torch.device("cpu")
        self.top_k = 4
        self.num_layers = 3
        self.max_num_reqs = 2
        self.seq_len = 8

        self.algorithm = _MockAlgorithm(top_k=self.top_k)
        self.adapter = _MockAdapter()
        k_buffer = torch.zeros(64, 2, 4, dtype=torch.float32, device=self.device)

        req_to_token = torch.arange(self.max_num_reqs * 16).view(self.max_num_reqs, 16)
        req_to_token_pool = _FakeReqToTokenPool(
            req_to_token.to(torch.int64), max_context_len=16
        )

        self.controller = SparseAlgorithmController(
            config=SparseConfig(
                top_k=self.top_k,
                device_buffer_size=8,
                host_to_device_ratio=2,
                algorithm="mock",
                backend="mock",
                page_size=1,
                min_sparse_prompt_len=0,
                sparse_extra_config={},
            ),
            algorithm=self.algorithm,
            adapter=self.adapter,
            req_to_token_pool=req_to_token_pool,
            device=self.device,
            start_layer=0,
            end_layer=self.num_layers,
        )
        self.controller.bind_io(_MockCoord(k_buffer))

    def test_bind_io_initializes_algorithm(self):
        self.assertEqual(self.algorithm.init_calls, 1)
        self.assertEqual(self.algorithm.start_layer, 0)
        self.assertEqual(self.algorithm.end_layer, self.num_layers)

    def test_begin_layer_decode_snapshots_and_swaps(self):
        layer = SimpleNamespace(layer_id=0)
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([self.seq_len, self.seq_len], dtype=torch.int64),
        )
        metadata = object()
        q = torch.randn(2, 8, dtype=torch.float32, device=self.device)

        self.controller.begin_layer_decode(
            q=q, layer=layer, forward_batch=forward_batch, attn_metadata=metadata
        )

        # save_original_metadata should fire on layer 0 only.
        self.assertIs(self.adapter.saved_metadata, metadata)
        # swap_in_selected_pages called exactly once.
        self.assertEqual(len(self.controller.io.swap_calls), 1)
        self.assertEqual(self.controller.io.swap_calls[0][0], 0)
        # adapter.adapt_for_attn_metadata called.
        self.assertEqual(self.adapter.adapt_calls, [0])

        # Layer 1: adapter should NOT re-save (snapshot taken only on start_layer).
        self.controller.begin_layer_decode(
            q=q,
            layer=SimpleNamespace(layer_id=1),
            forward_batch=forward_batch,
            attn_metadata=metadata,
        )
        # Still the same snapshot reference.
        self.assertIs(self.adapter.saved_metadata, metadata)
        self.assertEqual(self.adapter.adapt_calls, [0, 1])


if __name__ == "__main__":
    unittest.main()
