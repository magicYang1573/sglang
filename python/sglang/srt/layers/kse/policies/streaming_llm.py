"""StreamingLLM — Fixed-pattern eviction policy (sink + sliding window).

StreamingLLM retains a fixed number of "sink" tokens (initial tokens that
absorb attention mass) plus a sliding window of the most recent tokens.
All tokens between the sink region and the window are permanently evicted
after prefill, freeing KV-cache slots.

Because eviction already removes the unwanted tokens, the ``select()``
method simply returns the full (post-eviction) range — no per-step or
per-layer selection overhead.

Reference: Xiao et al., "Efficient Streaming Language Models with
Attention Sinks", ICLR 2024.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.kse.base_policy import EvictionPolicy, SparsityPolicy
from sglang.srt.layers.kse.registry import register_policy
from sglang.srt.layers.kse.types import Frequency, Granularity, SelectionResult

if TYPE_CHECKING:
    from sglang.srt.layers.kse.config import KSEConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@register_policy("streaming_llm")
class StreamingLLMPolicy(SparsityPolicy, EvictionPolicy):
    """Sink + sliding-window eviction with PER_REQUEST frequency."""

    def __init__(self, config: KSEConfig, device: torch.device):
        self.config = config
        self.device = device
        self.num_sink_tokens = config.policy_kwargs.get("num_sink_tokens", 4)
        self.window_size = config.policy_kwargs.get("window_size", 1024)

    # -- SparsityPolicy interface -----------------------------------------

    def granularity(self) -> Granularity:
        return Granularity.TOKEN

    def frequency(self) -> Frequency:
        return Frequency.PER_REQUEST

    def select(
        self,
        query: Optional[torch.Tensor],
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> SelectionResult:
        """After eviction all remaining tokens participate — return full range."""
        bs = seq_lens.shape[0]
        max_len = int(seq_lens.max().item()) if bs > 0 else 0

        indices = torch.full(
            (bs, max_len), -1, dtype=torch.int32, device=self.device
        )
        valid_lengths = torch.zeros(bs, dtype=torch.int32, device=self.device)

        for i in range(bs):
            n = seq_lens[i].item()
            indices[i, :n] = torch.arange(n, dtype=torch.int32, device=self.device)
            valid_lengths[i] = n

        return SelectionResult(
            granularity=Granularity.TOKEN,
            selected_indices=indices,
            valid_lengths=valid_lengths,
            sparse_mask=torch.ones(bs, dtype=torch.bool, device=self.device),
        )

    # -- EvictionPolicy interface -----------------------------------------

    def compute_eviction(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Evict tokens between the sink region and the sliding window."""
        bs = seq_lens.shape[0]
        keep_count = self.num_sink_tokens + self.window_size

        max_evict = 0
        evict_list = []

        for i in range(bs):
            n = seq_lens[i].item()
            if n <= keep_count:
                evict_list.append(
                    torch.empty(0, dtype=torch.int32, device=self.device)
                )
                continue
            evict_start = self.num_sink_tokens
            evict_end = n - self.window_size
            evicted = torch.arange(
                evict_start, evict_end, dtype=torch.int32, device=self.device
            )
            evict_list.append(evicted)
            max_evict = max(max_evict, evicted.shape[0])

        if max_evict == 0:
            return torch.full(
                (bs, 1), -1, dtype=torch.int32, device=self.device
            )

        result = torch.full(
            (bs, max_evict), -1, dtype=torch.int32, device=self.device
        )
        for i, ev in enumerate(evict_list):
            if ev.numel() > 0:
                result[i, : ev.shape[0]] = ev

        return result
