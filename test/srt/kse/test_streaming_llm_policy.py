"""Tests for the StreamingLLM sparsity/eviction policy."""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))
sys.path.insert(0, _HERE)

import torch

from sglang.srt.layers.kse.base_policy import EvictionPolicy
from sglang.srt.layers.kse.config import KSEConfig
from sglang.srt.layers.kse.policies.streaming_llm import StreamingLLMPolicy
from sglang.srt.layers.kse.types import Frequency, Granularity

from mock_utils import (
    MockKVCache,
    MockReqToTokenPool,
    build_identity_req_to_token,
    make_decode_batch,
)


def _make_policy(num_sink_tokens=4, window_size=8):
    cfg = KSEConfig(
        policy_name="streaming_llm",
        backend_name="triton",
        policy_kwargs={
            "num_sink_tokens": num_sink_tokens,
            "window_size": window_size,
        },
    )
    return StreamingLLMPolicy(cfg, torch.device("cpu"))


def _make_env(seq_lens):
    bs = len(seq_lens)
    max_ctx = max(seq_lens) + 64
    pool = MockReqToTokenPool(bs + 1, max_ctx)
    kv = MockKVCache(2, 256, 2, 64)
    for i, sl in enumerate(seq_lens):
        build_identity_req_to_token(pool, i, sl)
    batch = make_decode_batch(seq_lens, pool, kv)
    return pool, kv, batch


class TestStreamingLLMBasic(unittest.TestCase):
    def test_granularity(self):
        p = _make_policy()
        self.assertEqual(p.granularity(), Granularity.TOKEN)

    def test_frequency(self):
        p = _make_policy()
        self.assertEqual(p.frequency(), Frequency.PER_REQUEST)

    def test_is_eviction_policy(self):
        p = _make_policy()
        self.assertIsInstance(p, EvictionPolicy)


class TestComputeEviction(unittest.TestCase):
    def test_long_seq(self):
        p = _make_policy(num_sink_tokens=4, window_size=8)
        pool, kv, batch = _make_env([32])

        evict = p.compute_eviction(
            batch.req_pool_indices, batch.seq_lens, batch
        )

        # Should evict positions [4, 5, ..., 23] (32 - 8 = 24, so range [4, 24))
        valid = evict[0][evict[0] >= 0]
        expected = torch.arange(4, 24, dtype=torch.int32)
        self.assertTrue(torch.equal(valid, expected))

    def test_short_seq_no_evict(self):
        p = _make_policy(num_sink_tokens=4, window_size=8)
        pool, kv, batch = _make_env([10])  # 10 <= 4 + 8

        evict = p.compute_eviction(
            batch.req_pool_indices, batch.seq_lens, batch
        )

        # All entries should be -1
        self.assertTrue((evict == -1).all())

    def test_exact_threshold_no_evict(self):
        p = _make_policy(num_sink_tokens=4, window_size=8)
        pool, kv, batch = _make_env([12])  # 12 == 4 + 8

        evict = p.compute_eviction(
            batch.req_pool_indices, batch.seq_lens, batch
        )
        self.assertTrue((evict == -1).all())

    def test_batch_mixed(self):
        p = _make_policy(num_sink_tokens=2, window_size=4)
        pool, kv, batch = _make_env([5, 20, 6])

        evict = p.compute_eviction(
            batch.req_pool_indices, batch.seq_lens, batch
        )

        # Request 0: 5 <= 2+4=6 → no eviction
        self.assertTrue((evict[0] == -1).all())

        # Request 1: 20 > 6 → evict [2, 3, ..., 15]
        valid_1 = evict[1][evict[1] >= 0]
        expected_1 = torch.arange(2, 16, dtype=torch.int32)
        self.assertTrue(torch.equal(valid_1, expected_1))

        # Request 2: 6 <= 6 → no eviction
        self.assertTrue((evict[2] == -1).all())

    def test_eviction_contiguous(self):
        p = _make_policy(num_sink_tokens=3, window_size=5)
        pool, kv, batch = _make_env([20])

        evict = p.compute_eviction(
            batch.req_pool_indices, batch.seq_lens, batch
        )

        valid = evict[0][evict[0] >= 0]
        # Should be contiguous: [3, 4, ..., 14]
        expected = torch.arange(3, 15, dtype=torch.int32)
        self.assertTrue(torch.equal(valid, expected))


class TestStreamingLLMSelect(unittest.TestCase):
    def test_returns_full_range(self):
        p = _make_policy()
        pool, kv, batch = _make_env([16])

        result = p.select(
            query=None,
            layer_id=-1,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            forward_batch=batch,
        )

        self.assertEqual(result.granularity, Granularity.TOKEN)
        self.assertTrue(result.sparse_mask[0].item())
        self.assertEqual(result.valid_lengths[0].item(), 16)

        selected = result.selected_indices[0, :16]
        expected = torch.arange(16, dtype=torch.int32)
        self.assertTrue(torch.equal(selected, expected))

    def test_select_batch(self):
        p = _make_policy()
        pool, kv, batch = _make_env([8, 12])

        result = p.select(
            query=None,
            layer_id=-1,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            forward_batch=batch,
        )

        self.assertEqual(result.valid_lengths[0].item(), 8)
        self.assertEqual(result.valid_lengths[1].item(), 12)
        # Padding for request 0 (max_len=12, so positions 8-11 should be -1)
        self.assertTrue((result.selected_indices[0, 8:] == -1).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
