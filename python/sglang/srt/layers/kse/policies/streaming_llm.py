"""StreamingLLM — Sink + sliding-window sparsity with periodic eviction.

StreamingLLM retains a fixed number of "sink" tokens (initial tokens that
absorb attention mass) plus a sliding window of the most recent tokens.

Unlike the original per-request-only design, this implementation uses
**PER_STEP** frequency:

- ``select()`` is called every decode step to produce a mask that covers
  only the sink tokens and the most recent ``window_size`` tokens.  This
  ensures the attention kernel never sees stale middle tokens, even
  before they are physically evicted.

- Physical eviction (compacting ``req_to_token`` and freeing KV-cache
  slots) is performed **periodically** — every ``evict_interval`` decode
  steps — to amortise the cost of the compaction.

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
    """Sink + sliding-window with PER_STEP selection and periodic eviction."""

    def __init__(self, config: KSEConfig, device: torch.device):
        self.config = config
        self.device = device
        self.num_sink_tokens = config.policy_kwargs.get("num_sink_tokens", 4)
        self.window_size = config.policy_kwargs.get("window_size", 1024)
        self.evict_interval = config.policy_kwargs.get("evict_interval", 64)
        self._step_counter = 0

    # -- SparsityPolicy interface -----------------------------------------

    def granularity(self) -> Granularity:
        return Granularity.TOKEN

    def frequency(self) -> Frequency:
        return Frequency.PER_STEP

    def select(
        self,
        query: Optional[torch.Tensor],
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> SelectionResult:
        """Select sink tokens + the most recent window_size tokens."""
        bs = seq_lens.shape[0]
        keep_count = self.num_sink_tokens + self.window_size

        all_indices = []
        all_lengths = []
        sparse_flags = []

        for i in range(bs):
            n = seq_lens[i].item()
            if n <= keep_count:
                # Short sequence — keep everything, no sparsity needed
                idx = torch.arange(n, dtype=torch.int32, device=self.device)
                all_indices.append(idx)
                all_lengths.append(n)
                sparse_flags.append(False)
            else:
                # Sink: [0, num_sink_tokens)
                # Window: [n - window_size, n)
                sink = torch.arange(
                    self.num_sink_tokens, dtype=torch.int32, device=self.device
                )
                window = torch.arange(
                    n - self.window_size, n, dtype=torch.int32, device=self.device
                )
                selected = torch.cat([sink, window])
                all_indices.append(selected)
                all_lengths.append(selected.shape[0])
                sparse_flags.append(True)

        max_sel = max(all_lengths) if all_lengths else 1
        indices_tensor = torch.full(
            (bs, max_sel), -1, dtype=torch.int32, device=self.device
        )
        lengths_tensor = torch.zeros(bs, dtype=torch.int32, device=self.device)

        for i, sel in enumerate(all_indices):
            n = all_lengths[i]
            indices_tensor[i, :n] = sel[:n]
            lengths_tensor[i] = n

        return SelectionResult(
            granularity=Granularity.TOKEN,
            selected_indices=indices_tensor,
            valid_lengths=lengths_tensor,
            sparse_mask=torch.tensor(sparse_flags, dtype=torch.bool, device=self.device),
        )

    # -- EvictionPolicy interface -----------------------------------------

    def should_evict(self) -> bool:
        """Whether physical eviction should happen this step."""
        return self._step_counter % self.evict_interval == 0

    def on_step(self) -> None:
        """Called by the controller at the start of each decode step."""
        self._step_counter += 1

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
