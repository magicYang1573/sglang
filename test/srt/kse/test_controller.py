"""Tests for KSEController coordination logic."""

import os
import sys
import unittest
from unittest.mock import MagicMock

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))
sys.path.insert(0, _HERE)

import torch

from sglang.srt.layers.kse.base_adapter import MetadataAdapter
from sglang.srt.layers.kse.base_policy import EvictionPolicy, SparsityPolicy
from sglang.srt.layers.kse.config import KSEConfig
from sglang.srt.layers.kse.controller import KSEController
from sglang.srt.layers.kse.types import Frequency, Granularity, SelectionResult

from mock_utils import (
    MockForwardMode,
    MockKVCache,
    MockReqToTokenPool,
    MockTokenToKVPoolAllocator,
    MockTritonMetadata,
    build_identity_req_to_token,
    make_decode_batch,
    make_extend_batch,
)


# ---------------------------------------------------------------------------
# Concrete stub policies for testing
# ---------------------------------------------------------------------------

class _StubTokenPolicy(SparsityPolicy):
    """TOKEN granularity, PER_REQUEST frequency."""

    def __init__(self):
        self.select_call_count = 0

    def granularity(self):
        return Granularity.TOKEN

    def frequency(self):
        return Frequency.PER_REQUEST

    def select(self, query, layer_id, req_pool_indices, seq_lens,
               forward_batch, **kw):
        self.select_call_count += 1
        bs = seq_lens.shape[0]
        return SelectionResult(
            granularity=Granularity.TOKEN,
            selected_indices=torch.zeros(bs, 1, dtype=torch.int32),
            valid_lengths=torch.ones(bs, dtype=torch.int32),
            sparse_mask=torch.ones(bs, dtype=torch.bool),
        )


class _StubPagePolicy(SparsityPolicy):
    """PAGE granularity, PER_LAYER frequency."""

    def __init__(self):
        self.select_call_count = 0

    def granularity(self):
        return Granularity.PAGE

    def frequency(self):
        return Frequency.PER_LAYER

    def select(self, query, layer_id, req_pool_indices, seq_lens,
               forward_batch, **kw):
        self.select_call_count += 1
        bs = seq_lens.shape[0]
        return SelectionResult(
            granularity=Granularity.PAGE,
            selected_indices=torch.zeros(bs, 1, dtype=torch.int32),
            valid_lengths=torch.ones(bs, dtype=torch.int32),
            sparse_mask=torch.ones(bs, dtype=torch.bool),
        )


class _StubPerStepPolicy(SparsityPolicy):
    """TOKEN granularity, PER_STEP frequency."""

    def __init__(self):
        self.select_call_count = 0

    def granularity(self):
        return Granularity.TOKEN

    def frequency(self):
        return Frequency.PER_STEP

    def select(self, query, layer_id, req_pool_indices, seq_lens,
               forward_batch, **kw):
        self.select_call_count += 1
        bs = seq_lens.shape[0]
        return SelectionResult(
            granularity=Granularity.TOKEN,
            selected_indices=torch.zeros(bs, 1, dtype=torch.int32),
            valid_lengths=torch.ones(bs, dtype=torch.int32),
            sparse_mask=torch.ones(bs, dtype=torch.bool),
        )


class _StubEvictionPolicy(SparsityPolicy, EvictionPolicy):
    """TOKEN granularity, PER_REQUEST, with eviction."""

    def granularity(self):
        return Granularity.TOKEN

    def frequency(self):
        return Frequency.PER_REQUEST

    def select(self, query, layer_id, req_pool_indices, seq_lens,
               forward_batch, **kw):
        bs = seq_lens.shape[0]
        max_len = int(seq_lens.max().item())
        indices = torch.full((bs, max_len), -1, dtype=torch.int32)
        for i in range(bs):
            n = seq_lens[i].item()
            indices[i, :n] = torch.arange(n, dtype=torch.int32)
        return SelectionResult(
            granularity=Granularity.TOKEN,
            selected_indices=indices,
            valid_lengths=seq_lens.clone(),
            sparse_mask=torch.ones(bs, dtype=torch.bool),
        )

    def compute_eviction(self, req_pool_indices, seq_lens, forward_batch):
        # Evict positions [2, 3] for any seq with len > 5
        bs = seq_lens.shape[0]
        result = torch.full((bs, 2), -1, dtype=torch.int32)
        for i in range(bs):
            if seq_lens[i].item() > 5:
                result[i] = torch.tensor([2, 3], dtype=torch.int32)
        return result


class _NoopAdapter(MetadataAdapter):
    def __init__(self):
        self.save_count = 0
        self.apply_count = 0

    def save_dense_metadata(self, forward_metadata):
        self.save_count += 1

    def apply(self, result, forward_metadata, forward_batch, layer_id):
        self.apply_count += 1
        return forward_metadata

    def restore_dense_metadata(self, forward_metadata):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _make_controller(policy, adapter=None, eviction=None, backend="triton",
                     page_size=64, end_layer=-1, allocator=None):
    if adapter is None:
        adapter = _NoopAdapter()
    if eviction is None and isinstance(policy, EvictionPolicy):
        eviction = policy
    pool = MockReqToTokenPool(4, 512)
    kv = MockKVCache(4, 512, 2, 64)
    cfg = KSEConfig(
        policy_name="test",
        backend_name=backend,
        page_size=page_size,
        end_layer=end_layer,
    )
    return KSEController(
        policy=policy,
        adapter=adapter,
        eviction=eviction,
        req_to_token_pool=pool,
        token_to_kv_pool=kv,
        token_to_kv_pool_allocator=allocator,
        config=cfg,
    )


class TestGranularityCompatibility(unittest.TestCase):
    def test_token_policy_on_page_backend_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _make_controller(_StubTokenPolicy(), backend="flashattention",
                             page_size=256)
        self.assertIn("incompatible", str(ctx.exception))

    def test_page_policy_on_token_backend_ok(self):
        ctrl = _make_controller(_StubPagePolicy(), backend="triton")
        self.assertIsNotNone(ctrl)

    def test_page_policy_small_page_size_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _make_controller(_StubPagePolicy(), backend="flashattention",
                             page_size=128)
        self.assertIn("page_size", str(ctx.exception))

    def test_page_policy_non_multiple_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _make_controller(_StubPagePolicy(), backend="flashattention",
                             page_size=300)
        self.assertIn("integer multiple", str(ctx.exception))

    def test_page_policy_valid_multiple_ok(self):
        ctrl = _make_controller(_StubPagePolicy(), backend="flashattention",
                                page_size=256)
        self.assertIsNotNone(ctrl)


class TestShouldApply(unittest.TestCase):
    def setUp(self):
        self.ctrl = _make_controller(_StubPagePolicy(), end_layer=3)
        self.pool = self.ctrl.req_to_token_pool
        self.kv = self.ctrl.token_to_kv_pool

    def test_decode_in_range(self):
        batch = make_decode_batch([128], self.pool, self.kv)
        self.assertTrue(self.ctrl._should_apply(0, batch))
        self.assertTrue(self.ctrl._should_apply(2, batch))

    def test_extend_mode_false(self):
        batch = make_extend_batch([128], self.pool, self.kv)
        self.assertFalse(self.ctrl._should_apply(0, batch))

    def test_layer_out_of_range(self):
        batch = make_decode_batch([128], self.pool, self.kv)
        self.assertFalse(self.ctrl._should_apply(3, batch))
        self.assertFalse(self.ctrl._should_apply(10, batch))

    def test_end_layer_minus_one_uses_layer_num(self):
        ctrl = _make_controller(_StubPagePolicy(), end_layer=-1)
        batch = make_decode_batch([128], ctrl.req_to_token_pool,
                                  ctrl.token_to_kv_pool)
        # layer_num = 4, so layer 3 should be valid
        self.assertTrue(ctrl._should_apply(3, batch))
        self.assertFalse(ctrl._should_apply(4, batch))


class TestFrequencyCaching(unittest.TestCase):
    def test_per_request_caches_after_prefill(self):
        policy = _StubTokenPolicy()
        adapter = _NoopAdapter()
        ctrl = _make_controller(policy, adapter)
        pool = ctrl.req_to_token_pool
        kv = ctrl.token_to_kv_pool
        build_identity_req_to_token(pool, 0, 64)
        batch = make_decode_batch([64], pool, kv)

        # Simulate after_prefill
        ctrl.after_prefill(batch)
        self.assertIsNotNone(ctrl._cached_result)
        self.assertEqual(policy.select_call_count, 1)

        # _get_selection should return cached result without calling select again
        r1 = ctrl._get_selection(None, 0, batch)
        r2 = ctrl._get_selection(None, 1, batch)
        self.assertIs(r1, r2)
        self.assertEqual(policy.select_call_count, 1)

    def test_per_step_recomputes_each_step(self):
        policy = _StubPerStepPolicy()
        adapter = _NoopAdapter()
        ctrl = _make_controller(policy, adapter)
        pool = ctrl.req_to_token_pool
        kv = ctrl.token_to_kv_pool
        batch = make_decode_batch([64], pool, kv)

        # First step
        ctrl.before_forward(batch)
        r1 = ctrl._get_selection(torch.randn(1, 2, 64), 0, batch)
        self.assertEqual(policy.select_call_count, 1)
        # Same step, different layer — should reuse
        r2 = ctrl._get_selection(torch.randn(1, 2, 64), 1, batch)
        self.assertIs(r1, r2)
        self.assertEqual(policy.select_call_count, 1)

        # New step — should recompute
        ctrl.before_forward(batch)
        r3 = ctrl._get_selection(torch.randn(1, 2, 64), 0, batch)
        self.assertEqual(policy.select_call_count, 2)
        self.assertIsNot(r1, r3)

    def test_per_layer_always_recomputes(self):
        policy = _StubPagePolicy()
        adapter = _NoopAdapter()
        ctrl = _make_controller(policy, adapter)
        pool = ctrl.req_to_token_pool
        kv = ctrl.token_to_kv_pool
        batch = make_decode_batch([64], pool, kv)

        ctrl._get_selection(torch.randn(1, 2, 64), 0, batch)
        ctrl._get_selection(torch.randn(1, 2, 64), 1, batch)
        ctrl._get_selection(torch.randn(1, 2, 64), 2, batch)
        self.assertEqual(policy.select_call_count, 3)


class TestEviction(unittest.TestCase):
    def test_eviction_compacts_req_to_token(self):
        policy = _StubEvictionPolicy()
        ctrl = _make_controller(policy)
        pool = ctrl.req_to_token_pool
        kv = ctrl.token_to_kv_pool

        # Set up: req 0 has tokens [100, 101, 102, 103, 104, 105, 106, 107]
        pool.req_to_token[0, :8] = torch.tensor(
            [100, 101, 102, 103, 104, 105, 106, 107], dtype=torch.int32
        )
        batch = make_decode_batch([8], pool, kv)

        ctrl._do_eviction(batch)

        # Positions 2,3 evicted → remaining: [100, 101, 104, 105, 106, 107]
        self.assertEqual(batch.seq_lens[0].item(), 6)
        expected = torch.tensor([100, 101, 104, 105, 106, 107], dtype=torch.int32)
        actual = pool.req_to_token[0, :6]
        self.assertTrue(torch.equal(actual, expected))

    def test_eviction_updates_seq_lens(self):
        policy = _StubEvictionPolicy()
        ctrl = _make_controller(policy)
        pool = ctrl.req_to_token_pool
        kv = ctrl.token_to_kv_pool
        build_identity_req_to_token(pool, 0, 10)
        batch = make_decode_batch([10], pool, kv)

        ctrl._do_eviction(batch)
        self.assertEqual(batch.seq_lens[0].item(), 8)

    def test_no_eviction_when_short(self):
        policy = _StubEvictionPolicy()
        ctrl = _make_controller(policy)
        pool = ctrl.req_to_token_pool
        kv = ctrl.token_to_kv_pool
        build_identity_req_to_token(pool, 0, 4)
        batch = make_decode_batch([4], pool, kv)

        ctrl._do_eviction(batch)
        # seq_len <= 5, so no eviction
        self.assertEqual(batch.seq_lens[0].item(), 4)

    def test_eviction_frees_physical_slots(self):
        policy = _StubEvictionPolicy()
        allocator = MockTokenToKVPoolAllocator()
        ctrl = _make_controller(policy, allocator=allocator)
        pool = ctrl.req_to_token_pool
        kv = ctrl.token_to_kv_pool

        pool.req_to_token[0, :8] = torch.tensor(
            [100, 101, 102, 103, 104, 105, 106, 107], dtype=torch.int32
        )
        batch = make_decode_batch([8], pool, kv)

        ctrl._do_eviction(batch)

        # Allocator should have been called with physical indices [102, 103]
        self.assertEqual(len(allocator.freed_indices), 1)
        freed = allocator.freed_indices[0]
        expected_freed = torch.tensor([102, 103], dtype=torch.int32)
        self.assertTrue(torch.equal(freed, expected_freed))

    def test_eviction_without_allocator_no_crash(self):
        policy = _StubEvictionPolicy()
        ctrl = _make_controller(policy, allocator=None)
        pool = ctrl.req_to_token_pool
        kv = ctrl.token_to_kv_pool
        build_identity_req_to_token(pool, 0, 10)
        batch = make_decode_batch([10], pool, kv)

        # Should not crash even without allocator
        ctrl._do_eviction(batch)
        self.assertEqual(batch.seq_lens[0].item(), 8)


class TestBeforeAttention(unittest.TestCase):
    def test_saves_metadata_once(self):
        policy = _StubPagePolicy()
        adapter = _NoopAdapter()
        ctrl = _make_controller(policy, adapter, end_layer=4)
        pool = ctrl.req_to_token_pool
        kv = ctrl.token_to_kv_pool
        batch = make_decode_batch([64], pool, kv)
        metadata = MagicMock()

        ctrl.before_forward(batch)
        ctrl.before_attention(torch.randn(1, 2, 64), 0, batch, metadata)
        ctrl.before_attention(torch.randn(1, 2, 64), 1, batch, metadata)
        ctrl.before_attention(torch.randn(1, 2, 64), 2, batch, metadata)

        self.assertEqual(adapter.save_count, 1)
        self.assertEqual(adapter.apply_count, 3)

    def test_skips_non_decode(self):
        policy = _StubPagePolicy()
        adapter = _NoopAdapter()
        ctrl = _make_controller(policy, adapter, end_layer=4)
        pool = ctrl.req_to_token_pool
        kv = ctrl.token_to_kv_pool
        batch = make_extend_batch([64], pool, kv)
        metadata = MagicMock()

        result = ctrl.before_attention(torch.randn(1, 2, 64), 0, batch, metadata)
        self.assertIs(result, metadata)
        self.assertEqual(adapter.apply_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
