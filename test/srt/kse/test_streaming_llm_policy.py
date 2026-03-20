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


def _make_policy(num_sink_tokens=4, window_size=8, evict_interval=64):
    cfg = KSEConfig(
        policy_name="streaming_llm",
        backend_name="triton",
        policy_kwargs={
            "num_sink_tokens": num_sink_tokens,
            "window_size": window_size,
            "evict_interval": evict_interval,
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

    def test_frequency_is_per_step(self):
        p = _make_policy()
        self.assertEqual(p.frequency(), Frequency.PER_STEP)

    def test_is_eviction_policy(self):
        p = _make_policy()
        self.assertIsInstance(p, EvictionPolicy)


class TestStreamingLLMSelect(unittest.TestCase):
    """select() should return sink + window tokens, masking out the middle."""

    def test_short_seq_no_sparsity(self):
        p = _make_policy(num_sink_tokens=4, window_size=8)
        pool, kv, batch = _make_env([10])  # 10 <= 4+8 → keep all

        result = p.select(
            query=None,
            layer_id=-1,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            forward_batch=batch,
        )

        self.assertEqual(result.valid_lengths[0].item(), 10)
        self.assertFalse(result.sparse_mask[0].item())
        selected = result.selected_indices[0, :10]
        expected = torch.arange(10, dtype=torch.int32)
        self.assertTrue(torch.equal(selected, expected))

    def test_long_seq_selects_sink_and_window(self):
        p = _make_policy(num_sink_tokens=4, window_size=8)
        pool, kv, batch = _make_env([32])

        result = p.select(
            query=None,
            layer_id=-1,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            forward_batch=batch,
        )

        self.assertTrue(result.sparse_mask[0].item())
        self.assertEqual(result.valid_lengths[0].item(), 4 + 8)

        valid = result.selected_indices[0, :12]
        # Sink: [0,1,2,3], Window: [24,25,26,27,28,29,30,31]
        expected = torch.tensor(
            [0, 1, 2, 3, 24, 25, 26, 27, 28, 29, 30, 31], dtype=torch.int32
        )
        self.assertTrue(torch.equal(valid, expected))

    def test_exact_threshold_no_sparsity(self):
        p = _make_policy(num_sink_tokens=4, window_size=8)
        pool, kv, batch = _make_env([12])  # 12 == 4+8

        result = p.select(
            query=None,
            layer_id=-1,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            forward_batch=batch,
        )

        self.assertFalse(result.sparse_mask[0].item())
        self.assertEqual(result.valid_lengths[0].item(), 12)

    def test_batch_mixed(self):
        p = _make_policy(num_sink_tokens=2, window_size=4)
        pool, kv, batch = _make_env([5, 20, 6])

        result = p.select(
            query=None,
            layer_id=-1,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            forward_batch=batch,
        )

        # Request 0: 5 <= 2+4=6 → no sparsity, keep all 5
        self.assertFalse(result.sparse_mask[0].item())
        self.assertEqual(result.valid_lengths[0].item(), 5)

        # Request 1: 20 > 6 → sparse, keep sink[0,1] + window[16..19] = 6 tokens
        self.assertTrue(result.sparse_mask[1].item())
        self.assertEqual(result.valid_lengths[1].item(), 6)
        valid_1 = result.selected_indices[1, :6]
        expected_1 = torch.tensor([0, 1, 16, 17, 18, 19], dtype=torch.int32)
        self.assertTrue(torch.equal(valid_1, expected_1))

        # Request 2: 6 <= 6 → no sparsity
        self.assertFalse(result.sparse_mask[2].item())
        self.assertEqual(result.valid_lengths[2].item(), 6)


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


class TestPeriodicEviction(unittest.TestCase):
    """Test the should_evict / on_step periodic eviction mechanism."""

    def test_default_evict_interval(self):
        p = _make_policy()
        self.assertEqual(p.evict_interval, 64)

    def test_custom_evict_interval(self):
        p = _make_policy(evict_interval=16)
        self.assertEqual(p.evict_interval, 16)

    def test_should_evict_timing(self):
        p = _make_policy(evict_interval=4)
        # Step counter starts at 0, should_evict checks counter % interval == 0
        # After on_step, counter becomes 1
        p.on_step()
        self.assertFalse(p.should_evict())  # step 1

        p.on_step()
        self.assertFalse(p.should_evict())  # step 2

        p.on_step()
        self.assertFalse(p.should_evict())  # step 3

        p.on_step()
        self.assertTrue(p.should_evict())  # step 4 → 4 % 4 == 0

        p.on_step()
        self.assertFalse(p.should_evict())  # step 5

    def test_should_evict_at_step_zero(self):
        """Step 0 (initial state) should trigger eviction (for after_prefill)."""
        p = _make_policy(evict_interval=4)
        self.assertTrue(p.should_evict())  # counter=0, 0%4==0


if __name__ == "__main__":
    unittest.main(verbosity=2)
