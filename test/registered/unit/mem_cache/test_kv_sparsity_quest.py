from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.sparsity.config import KVSparsityConfig
from sglang.srt.mem_cache.sparsity.contracts import (
    DecisionScope,
    SelectionContext,
)
from sglang.srt.mem_cache.sparsity.factory import create_kv_sparsity_controller
from sglang.srt.mem_cache.sparsity.policies.quest import QuestPolicy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ForwardMode:
    def __init__(self, decode: bool):
        self.decode = decode

    def is_decode(self):
        return self.decode


def _make_policy(keys, *, page_size=2, page_budget=2, recent_pages=1):
    req_to_token = torch.zeros((2, keys.shape[0]), dtype=torch.int32)
    req_to_token[1] = torch.arange(keys.shape[0], dtype=torch.int32)
    req_pool = SimpleNamespace(req_to_token=req_to_token)
    token_pool = SimpleNamespace(get_key_buffer=lambda layer_id: keys)
    config = KVSparsityConfig(
        policy="quest",
        page_size=page_size,
        min_sparse_tokens=0,
        policy_config={
            "page_budget": page_budget,
            "recent_pages": recent_pages,
        },
    )
    policy = QuestPolicy(
        config,
        torch.device("cpu"),
        token_to_kv_pool=token_pool,
        req_to_token_pool=req_pool,
    )
    return policy, req_pool


def _context(*, query, seq_len, page_width, decode):
    forward_batch = SimpleNamespace(forward_mode=_ForwardMode(decode))
    return SelectionContext(
        query=query,
        key=None,
        value=None,
        output=None,
        layer_id=0,
        req_pool_indices=torch.tensor([1]),
        seq_lens=torch.tensor([seq_len]),
        forward_batch=forward_batch,
        attention_layer=SimpleNamespace(layer_id=0),
        metadata=SimpleNamespace(
            page_table=torch.zeros((1, page_width), dtype=torch.int32)
        ),
    )


def test_quest_requires_a_real_page_and_valid_budget():
    keys = torch.zeros((8, 1, 2))
    with pytest.raises(ValueError, match="page_size >= 2"):
        _make_policy(keys, page_size=1)
    with pytest.raises(ValueError, match="smaller than page_budget"):
        _make_policy(keys, page_budget=2, recent_pages=2)


def test_quest_rejects_unknown_policy_fields():
    config = KVSparsityConfig(
        policy="quest",
        page_size=2,
        policy_config={"page_budget": 2, "recent_pages": 1, "typo": 1},
    )
    keys = torch.zeros((8, 1, 2))
    with pytest.raises(ValueError, match="Unknown Quest policy field"):
        QuestPolicy(
            config,
            torch.device("cpu"),
            token_to_kv_pool=SimpleNamespace(get_key_buffer=lambda layer_id: keys),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(8, dtype=torch.int32).repeat(2, 1)
            ),
        )


def test_factory_creates_quest_with_runtime_pools():
    keys = torch.zeros((16, 1, 2))
    req_to_token = torch.arange(16, dtype=torch.int32).repeat(2, 1)
    req_pool = SimpleNamespace(
        req_to_token=req_to_token,
        req_generation=torch.zeros(2, dtype=torch.int64),
    )
    controller = create_kv_sparsity_controller(
        device=torch.device("cpu"),
        req_to_token_pool=req_pool,
        token_to_kv_pool=SimpleNamespace(get_key_buffer=lambda layer_id: keys),
        start_layer=0,
        end_layer=1,
        server_args=SimpleNamespace(
            page_size=2,
            kv_cache_sparsity_config=(
                '{"policy":"quest","page_size":2,"policy_config":'
                '{"page_budget":4,"recent_pages":1}}'
            ),
        ),
    )

    assert isinstance(controller.policy, QuestPolicy)


def test_quest_builds_page_boxes_and_keeps_pages_for_any_gqa_head():
    # Four pages. Page 0 is best for query head 0; page 1 is best for head 1.
    # Averaging the two GQA heads would cancel them, so this also checks that
    # common-page selection ranks by the strongest head instead.
    keys = torch.tensor(
        [
            [[10.0, 0.0]],
            [[10.0, 0.0]],
            [[-9.0, 0.0]],
            [[-9.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[0.0, 0.0]],
            [[0.0, 0.0]],
        ]
    )
    policy, _ = _make_policy(keys)
    prefill = _context(query=None, seq_len=8, page_width=4, decode=False)
    policy.on_attention_complete(prefill)

    query = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    result = policy.select(_context(query=query, seq_len=8, page_width=4, decode=True))

    assert policy.capabilities.scope == DecisionScope.LAYER
    torch.testing.assert_close(
        result.logical_indices, torch.tensor([[0, 3]], dtype=torch.int32)
    )
    torch.testing.assert_close(
        result.valid_lengths, torch.tensor([2], dtype=torch.int32)
    )
    torch.testing.assert_close(
        result.visible_kv_lens, torch.tensor([4], dtype=torch.int32)
    )
    torch.testing.assert_close(result.sparse_mask, torch.tensor([True]))


def test_quest_reports_exact_length_for_a_partial_recent_page():
    keys = torch.arange(16, dtype=torch.float32).view(8, 1, 2)
    policy, _ = _make_policy(keys)
    policy.on_attention_complete(
        _context(query=None, seq_len=7, page_width=4, decode=False)
    )

    result = policy.select(
        _context(
            query=torch.ones((1, 1, 2)),
            seq_len=7,
            page_width=4,
            decode=True,
        )
    )

    assert result.valid_lengths.item() == 2
    assert result.visible_kv_lens.item() == 3
    assert result.logical_indices[0, -1].item() == 3


def test_quest_recomputes_the_partial_page_after_decode():
    keys = torch.tensor(
        [
            [[0.0, 0.0]],
            [[0.0, 0.0]],
            [[1.0, 1.0]],
            [[1.0, 1.0]],
            [[4.0, -2.0]],
            [[0.0, 0.0]],
            [[0.0, 0.0]],
            [[0.0, 0.0]],
        ]
    )
    policy, _ = _make_policy(keys)
    policy.on_attention_complete(
        _context(query=None, seq_len=5, page_width=3, decode=False)
    )
    keys[5] = torch.tensor([[-3.0, 7.0]])
    policy.on_attention_complete(
        _context(
            query=torch.ones((1, 1, 2)),
            seq_len=6,
            page_width=3,
            decode=True,
        )
    )

    torch.testing.assert_close(policy._page_k_min[0][2, 0], torch.tensor([-3.0, -2.0]))
    torch.testing.assert_close(policy._page_k_max[0][2, 0], torch.tensor([4.0, 7.0]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
