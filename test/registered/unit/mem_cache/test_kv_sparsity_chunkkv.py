import math
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.sparsity.config import KVSparsityConfig
from sglang.srt.mem_cache.sparsity.contracts import DecisionScope, SelectionContext
from sglang.srt.mem_cache.sparsity.factory import create_kv_sparsity_controller
from sglang.srt.mem_cache.sparsity.policies.chunkkv import ChunkKVPolicy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ForwardMode:
    def __init__(self, decode: bool):
        self._decode = decode

    def is_decode(self):
        return self._decode


def _make_policy(*, page_budget=2, recent_pages=1, obs_window=2):
    req_to_token = torch.zeros((2, 12), dtype=torch.int32)
    req_to_token[1] = torch.arange(2, 14, dtype=torch.int32)
    keys = {
        0: torch.zeros((14, 1, 2), dtype=torch.float32),
        1: torch.zeros((14, 1, 2), dtype=torch.float32),
    }
    keys[0][2:4, 0, 0] = 5
    keys[0][4:6, 0, 0] = 1
    keys[1][4:6, 0, 0] = 3
    req_pool = SimpleNamespace(req_to_token=req_to_token)
    token_pool = SimpleNamespace(get_key_buffer=lambda layer_id: keys[layer_id])
    policy = ChunkKVPolicy(
        KVSparsityConfig(
            policy="chunkkv",
            page_size=2,
            min_sparse_tokens=0,
            policy_config={
                "page_budget": page_budget,
                "recent_pages": recent_pages,
                "obs_window": obs_window,
            },
        ),
        torch.device("cpu"),
        token_to_kv_pool=token_pool,
        req_to_token_pool=req_pool,
        start_layer=0,
        end_layer=2,
    )
    return policy, req_pool, token_pool


def _prefill_context(policy, *, layer_id, seq_len=10, orig_seq_len=10):
    query = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    forward_batch = SimpleNamespace(
        forward_mode=_ForwardMode(False),
        extend_start_loc=torch.tensor([0], dtype=torch.int32),
        extend_seq_lens=torch.tensor([2], dtype=torch.int32),
        orig_seq_lens=torch.tensor([orig_seq_len], dtype=torch.int32),
    )
    return SelectionContext(
        query=query,
        layer_id=layer_id,
        req_pool_indices=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        forward_batch=forward_batch,
    )


def _decode_context(seq_len):
    return SelectionContext(
        query=torch.ones((1, 1, 2)),
        layer_id=0,
        req_pool_indices=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        forward_batch=SimpleNamespace(forward_mode=_ForwardMode(True)),
    )


def test_chunkkv_accumulates_exact_layer_scores_and_reuses_base_selection():
    policy, _, _ = _make_policy()
    policy.on_attention_complete(_prefill_context(policy, layer_id=0))
    policy.on_attention_complete(_prefill_context(policy, layer_id=1))

    expected_page_zero = 10 / math.sqrt(2)
    expected_page_one = 8 / math.sqrt(2)
    torch.testing.assert_close(
        policy._score_pool[1, 0], torch.tensor(expected_page_zero)
    )
    torch.testing.assert_close(
        policy._score_pool[1, 1], torch.tensor(expected_page_one)
    )
    torch.testing.assert_close(
        policy._base_pages[1], torch.tensor([0], dtype=torch.int32)
    )

    result = policy.select(_decode_context(seq_len=11))
    assert policy.capabilities.scope == DecisionScope.STEP
    torch.testing.assert_close(
        result.logical_indices, torch.tensor([[0, 5]], dtype=torch.int32)
    )
    torch.testing.assert_close(
        result.valid_lengths, torch.tensor([2], dtype=torch.int32)
    )
    torch.testing.assert_close(
        result.visible_kv_lens, torch.tensor([3], dtype=torch.int32)
    )
    torch.testing.assert_close(result.sparse_mask, torch.tensor([True]))


def test_chunkkv_waits_for_final_prefill_chunk_and_invalidates_reused_slot():
    policy, _, _ = _make_policy()
    policy.on_attention_complete(
        _prefill_context(policy, layer_id=0, seq_len=8, orig_seq_len=10)
    )
    policy.on_attention_complete(
        _prefill_context(policy, layer_id=1, seq_len=8, orig_seq_len=10)
    )
    assert not bool(policy._state_valid[1])

    policy.on_attention_complete(_prefill_context(policy, layer_id=0))
    policy.on_attention_complete(_prefill_context(policy, layer_id=1))
    assert bool(policy._state_valid[1])

    # The next generation's first non-final extend clears the old slot state.
    policy.on_attention_complete(
        _prefill_context(policy, layer_id=0, seq_len=8, orig_seq_len=10)
    )
    assert not bool(policy._state_valid[1])
    assert not bool(policy.select(_decode_context(seq_len=8)).sparse_mask[0])


def test_chunkkv_rejects_invalid_and_unknown_configuration():
    with pytest.raises(ValueError, match="smaller than page_budget"):
        _make_policy(page_budget=2, recent_pages=2)
    with pytest.raises(ValueError, match="page_size >= 2"):
        req_to_token = torch.arange(16, dtype=torch.int32).repeat(2, 1)
        ChunkKVPolicy(
            KVSparsityConfig(
                policy="chunkkv",
                page_size=1,
                policy_config={"page_budget": 4, "recent_pages": 1},
            ),
            torch.device("cpu"),
            token_to_kv_pool=SimpleNamespace(
                get_key_buffer=lambda layer_id: torch.zeros((16, 1, 2))
            ),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        )
    with pytest.raises(ValueError, match="Unknown ChunkKV policy field"):
        policy, req_pool, token_pool = _make_policy()
        ChunkKVPolicy(
            KVSparsityConfig(
                policy="chunkkv",
                page_size=2,
                policy_config={"page_budget": 2, "recent_pages": 1, "typo": 1},
            ),
            torch.device("cpu"),
            token_to_kv_pool=token_pool,
            req_to_token_pool=req_pool,
        )


def test_factory_creates_chunkkv_with_runtime_pools():
    keys = torch.zeros((32, 1, 2))
    req_to_token = torch.arange(24, dtype=torch.int32).repeat(2, 1)
    controller = create_kv_sparsity_controller(
        device=torch.device("cpu"),
        req_to_token_pool=SimpleNamespace(
            req_to_token=req_to_token,
            req_generation=torch.zeros(2, dtype=torch.int64),
        ),
        token_to_kv_pool=SimpleNamespace(get_key_buffer=lambda layer_id: keys),
        start_layer=0,
        end_layer=1,
        server_args=SimpleNamespace(
            page_size=2,
            kv_cache_sparsity_config=(
                '{"policy":"chunkkv","page_size":2,"policy_config":'
                '{"page_budget":4,"recent_pages":1,"obs_window":2}}'
            ),
        ),
    )
    assert isinstance(controller.policy, ChunkKVPolicy)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
