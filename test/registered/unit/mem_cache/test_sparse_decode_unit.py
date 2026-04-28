"""Unit tests for the non-native sparse decode path.

These tests focus on Python-level components that do NOT depend on CUDA
kernels (FA3 / hisparse.cuh), so they can run on any host with PyTorch.

Covered components:
  * :class:`SparseConfig` parsing from JSON (factory helpers).
  * :class:`QuestAlgorithm.retrieve_topk` token-level contract (no
    ``sparse_mask`` input; ``valid_lengths = clamp(seq_len, top_k)``).
  * :class:`Fa3SparseDecodeAdapter.build_sparse_call_args` packaging the
    ``SparseFa3CallArgs`` parameter pack.
  * :class:`SparseAlgorithmController` lifecycle orchestration with mocks.

Run with::

    cd python && python -m pytest \
      ../test/registered/unit/mem_cache/test_sparse_decode_unit.py -q
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch


# ---------------------------------------------------------------------------
# SparseConfig parsing
# ---------------------------------------------------------------------------


class TestSparseConfigFactory(unittest.TestCase):
    def test_parse_sparse_decode_config_defaults(self):
        from sglang.srt.mem_cache.sparsity import parse_sparse_decode_config

        args = SimpleNamespace(sparse_decode_config=None)
        cfg = parse_sparse_decode_config(args)

        self.assertEqual(cfg.algorithm, "quest")
        self.assertEqual(cfg.backend, "fa3")
        self.assertEqual(cfg.page_size, 1)
        # ``min_sparse_prompt_len`` is no longer auto-populated; the runtime
        # ignores it (kept for backward-compat parsing only).
        self.assertIsNone(cfg.min_sparse_prompt_len)
        self.assertEqual(cfg.top_k, 2048)
        self.assertEqual(cfg.device_buffer_size, 4096)
        self.assertEqual(cfg.host_to_device_ratio, 2)

    def test_parse_sparse_decode_config_custom_json(self):
        from sglang.srt.mem_cache.sparsity import parse_sparse_decode_config

        raw = (
            '{"top_k": 128, "device_buffer_size": 256, "host_to_device_ratio": 3, '
            '"sparse_extra_config": {"sparsity_ratio": 0.3, "num_recent_pages": 2}}'
        )
        args = SimpleNamespace(sparse_decode_config=raw)
        cfg = parse_sparse_decode_config(args)

        self.assertEqual(cfg.top_k, 128)
        self.assertEqual(cfg.device_buffer_size, 256)
        self.assertEqual(cfg.host_to_device_ratio, 3)
        self.assertEqual(
            cfg.sparse_extra_config,
            {"sparsity_ratio": 0.3, "num_recent_pages": 2},
        )

    def test_min_sparse_prompt_len_is_parsed_but_warned(self):
        from sglang.srt.mem_cache.sparsity import parse_sparse_decode_config

        raw = '{"top_k": 64, "min_sparse_prompt_len": 1024}'
        args = SimpleNamespace(sparse_decode_config=raw)
        with self.assertLogs(
            "sglang.srt.mem_cache.sparsity.factory", level="WARNING"
        ) as cm:
            cfg = parse_sparse_decode_config(args)
        self.assertEqual(cfg.min_sparse_prompt_len, 1024)
        self.assertTrue(any("deprecated" in msg for msg in cm.output))

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


class TestQuestRetrieveTopK(unittest.TestCase):
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
            sparse_extra_config={"sparsity_ratio": 0.5, "num_recent_pages": 2},
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
            [seq_len // self.config.page_size], dtype=torch.int64, device=self.device
        )
        self.algorithm._compute_page_representations(
            layer_id=0,
            reqs=reqs,
            seq_lens=seq_lens,
            start_page=0,
            end_page=num_pages,
            k_buffer=self.kv_pool.get_key_buffer(0),
        )

    def test_output_shape_and_dtype_long_seq(self):
        seq_len = 32  # > top_k_budget (8) → genuine sparse selection
        req_idx = 1
        self.states.register(req_idx, seq_len)
        self._run_construct(req_idx, seq_len)

        queries = torch.randn(
            1, self.head_num * self.head_dim, dtype=torch.float32, device=self.device
        )
        req_pool_indices = torch.tensor([req_idx], dtype=torch.int64, device=self.device)

        fwd = SimpleNamespace(
            seq_lens=torch.tensor([seq_len], dtype=torch.int64, device=self.device)
        )
        top_k_tokens, valid_lengths = self.algorithm.retrieve_topk(
            queries=queries,
            layer_id=0,
            req_pool_indices=req_pool_indices,
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

    def test_short_seq_auto_dense_equivalent(self):
        """When seq_len <= top_k, valid_lengths should report seq_len and the
        selection should cover the full set of tokens (equivalent to dense).
        """
        seq_len = 4  # < top_k_budget (8)
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
            forward_batch=fwd,
        )

        length = int(valid_lengths[0].item())
        # All seq_len tokens should be selected (dense-equivalent).
        self.assertEqual(length, seq_len)

        # The selected tokens (first ``length`` slots) form a permutation of
        # [0..seq_len-1]: dense coverage.  Order is not guaranteed because
        # bounding-box top-k may interleave with the recent window.
        selected = top_k_tokens[0, :length].sort().values
        torch.testing.assert_close(
            selected, torch.arange(seq_len, dtype=torch.int32)
        )
        # Padding slots beyond ``length`` are -1.
        self.assertTrue(torch.all(top_k_tokens[0, length:] == -1))


# ---------------------------------------------------------------------------
# Fa3SparseDecodeAdapter.build_sparse_call_args
# ---------------------------------------------------------------------------


class _FakeFlashAttentionMetadata:
    """Minimal stand-in matching the fields the adapter reads.

    The adapter only borrows ``cu_seqlens_q`` from this object.  In
    real FA3 ``cu_seqlens_q`` is always ``arange(0, bs+1)`` for decode.
    """

    def __init__(self, cu_seqlens_q):
        self.cu_seqlens_q = cu_seqlens_q


class TestFa3SparseDecodeAdapter(unittest.TestCase):
    def setUp(self):
        from sglang.srt.mem_cache.sparsity.backend.fa3_sparse_decode_adapter import (
            Fa3SparseDecodeAdapter,
        )

        self.device = torch.device("cpu")
        self.adapter = Fa3SparseDecodeAdapter(self.device)
        self.bs = 3
        self.top_k = 4
        # Fake "FA3 dense" cu_seqlens_q (decode → arange).
        self.dense_cu_seqlens_q = torch.arange(
            self.bs + 1, dtype=torch.int32, device=self.device
        )
        self.dense_meta = _FakeFlashAttentionMetadata(self.dense_cu_seqlens_q)

    def test_build_sparse_call_args_long_seq(self):
        top_k_device_locs = torch.tensor(
            [
                [100, 101, 102, 103],
                [200, 201, 202, 203],
                [300, 301, 302, 303],
            ],
            dtype=torch.int32,
            device=self.device,
        )
        valid_lengths = torch.tensor([4, 4, 4], dtype=torch.int32, device=self.device)

        args = self.adapter.build_sparse_call_args(
            top_k_device_locs=top_k_device_locs,
            valid_lengths=valid_lengths,
            dense_metadata=self.dense_meta,
            forward_batch=None,
            layer_id=0,
        )

        # page_table forwarded as-is.
        torch.testing.assert_close(args.page_table, top_k_device_locs)
        # cache_seqlens == valid_lengths cast to int32.
        torch.testing.assert_close(args.cache_seqlens, valid_lengths.to(torch.int32))
        # cu_seqlens_q borrowed from dense.
        self.assertIs(args.cu_seqlens_q, self.dense_cu_seqlens_q)
        # cu_seqlens_k = pad(cumsum(cache_seqlens))
        expected_cu_k = torch.tensor(
            [0, 4, 8, 12], dtype=torch.int32, device=self.device
        )
        torch.testing.assert_close(args.cu_seqlens_k, expected_cu_k)
        # max_seqlen_k is the static upper bound = top_k.
        self.assertEqual(args.max_seqlen_k, self.top_k)

    def test_build_sparse_call_args_short_seq(self):
        """When valid_lengths < top_k, padding slots should still be present
        (and ignored by FA3 via cache_seqlens)."""
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

        args = self.adapter.build_sparse_call_args(
            top_k_device_locs=top_k_device_locs,
            valid_lengths=valid_lengths,
            dense_metadata=self.dense_meta,
            forward_batch=None,
            layer_id=0,
        )

        torch.testing.assert_close(args.cache_seqlens, valid_lengths.to(torch.int32))
        expected_cu_k = torch.tensor(
            [0, 2, 5, 9], dtype=torch.int32, device=self.device
        )
        torch.testing.assert_close(args.cu_seqlens_k, expected_cu_k)
        # Adapter does not modify shared dense metadata.
        self.assertIs(self.dense_meta.cu_seqlens_q, self.dense_cu_seqlens_q)


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

    def retrieve_topk(self, queries, layer_id, req_pool_indices, **kwargs):
        bs = queries.shape[0]
        seq_lens = kwargs["forward_batch"].seq_lens
        top_k_tokens = torch.full(
            (bs, self.top_k), -1, dtype=torch.int32, device=queries.device
        )
        # valid_lengths = min(seq_len, top_k) — short requests auto-dense.
        valid_lengths = seq_lens.clamp(max=self.top_k).to(torch.int32)
        for i in range(bs):
            n = int(valid_lengths[i].item())
            if n > 0:
                top_k_tokens[i, :n] = torch.arange(n, dtype=torch.int32)
        return top_k_tokens, valid_lengths

    def construct_representations(self, **kwargs):
        self.construct_calls.append(kwargs["layer_id"])

    def update_representations(self, **kwargs):
        self.update_calls.append(kwargs["layer_id"])


class _MockAdapter:
    def __init__(self):
        self.build_calls = []

    def build_sparse_call_args(self, **kwargs):
        from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
            SparseFa3CallArgs,
        )

        self.build_calls.append(kwargs["layer_id"])
        top_k_device_locs = kwargs["top_k_device_locs"]
        valid_lengths = kwargs["valid_lengths"]
        bs = top_k_device_locs.shape[0]
        return SparseFa3CallArgs(
            page_table=top_k_device_locs,
            cache_seqlens=valid_lengths.to(torch.int32),
            cu_seqlens_q=torch.arange(bs + 1, dtype=torch.int32),
            cu_seqlens_k=torch.zeros(bs + 1, dtype=torch.int32),
            max_seqlen_k=top_k_device_locs.shape[1],
        )


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


class TestSparseAlgorithmController(unittest.TestCase):
    def setUp(self):
        from sglang.srt.mem_cache.sparsity import (
            SparseAlgorithmController,
            SparseConfig,
        )

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

    def test_begin_layer_decode_runs_full_pipeline(self):
        layer = SimpleNamespace(layer_id=0)
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([self.seq_len, self.seq_len], dtype=torch.int64),
        )
        metadata = SimpleNamespace(cu_seqlens_q=torch.arange(3, dtype=torch.int32))
        q = torch.randn(2, 8, dtype=torch.float32, device=self.device)

        sparse_args = self.controller.begin_layer_decode(
            q=q, layer=layer, forward_batch=forward_batch, attn_metadata=metadata
        )

        # swap_in_selected_pages called exactly once at layer_id=0.
        self.assertEqual(len(self.controller.io.swap_calls), 1)
        self.assertEqual(self.controller.io.swap_calls[0][0], 0)
        # adapter.build_sparse_call_args called exactly once.
        self.assertEqual(self.adapter.build_calls, [0])
        # Returned object is a SparseFa3CallArgs with the right shapes.
        self.assertEqual(sparse_args.page_table.shape, (2, self.top_k))
        self.assertEqual(sparse_args.cache_seqlens.shape, (2,))

        # Layer 1: pipeline runs again — no per-step "snapshot" gating.
        sparse_args_l1 = self.controller.begin_layer_decode(
            q=q,
            layer=SimpleNamespace(layer_id=1),
            forward_batch=forward_batch,
            attn_metadata=metadata,
        )
        self.assertEqual(self.adapter.build_calls, [0, 1])
        self.assertEqual(len(self.controller.io.swap_calls), 2)
        self.assertEqual(sparse_args_l1.page_table.shape, (2, self.top_k))


if __name__ == "__main__":
    unittest.main()
