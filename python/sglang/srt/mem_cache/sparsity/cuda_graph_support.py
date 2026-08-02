"""KV-sparsity support for decode Breakable CUDA Graphs.

Every CUDA-graph-capable policy uses this provider to inject its controller
into capture. Quest additionally follows sgl-project/sglang#31951 by treating
context-length page capacity as a graph variant independent of batch size.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import torch

_DEFAULT_CONTEXT_BUCKETS = (10 * 1024, 33 * 1024)


class KVSparsityCudaGraphRuntimeProvider:
    def __init__(self, controller: Any):
        self.controller = controller
        self.policy = controller.policy
        self.page_size = controller.config.page_size
        self.uses_context_page_variants = controller.config.policy == "quest"
        if not self.uses_context_page_variants:
            self.page_buckets = (None,)
            return

        self.max_context_len = controller.req_to_token_pool.req_to_token.shape[1]
        self.max_num_pages = (
            self.max_context_len + self.page_size - 1
        ) // self.page_size
        context_buckets = controller.config.policy_config.get(
            "cuda_graph_context_buckets", _DEFAULT_CONTEXT_BUCKETS
        )
        if (
            not isinstance(context_buckets, (list, tuple))
            or not context_buckets
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in context_buckets
            )
        ):
            raise ValueError(
                "cuda_graph_context_buckets must contain positive integers"
            )

        minimum_pages = self.policy.page_budget + 1
        page_buckets = {
            min(
                max(
                    (context_len + self.page_size - 1) // self.page_size,
                    minimum_pages,
                ),
                self.max_num_pages,
            )
            for context_len in context_buckets
        }
        page_buckets.add(self.max_num_pages)
        self.page_buckets = tuple(sorted(page_buckets))

    def cuda_graph_capture_variants(self):
        return self.page_buckets

    def select_cuda_graph_variant(self, forward_batch):
        if not self.uses_context_page_variants:
            return True, None

        seq_lens = getattr(forward_batch, "seq_lens_cpu", None)
        if torch.is_tensor(seq_lens):
            if seq_lens.device.type != "cpu":
                return True, self.page_buckets[-1]
            max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() else 0
        elif seq_lens is None:
            return True, self.page_buckets[-1]
        else:
            try:
                max_seq_len = max((int(value) for value in seq_lens), default=0)
            except TypeError:
                return True, self.page_buckets[-1]
        required_pages = (max_seq_len + self.page_size - 1) // self.page_size
        variant = next(
            (capacity for capacity in self.page_buckets if capacity >= required_pages),
            None,
        )
        return variant is not None, variant

    def _publish_variant(self, forward_batch, runtime_variant) -> None:
        if not self.uses_context_page_variants:
            if runtime_variant is not None:
                raise ValueError(
                    f"Unexpected KV sparsity CUDA Graph variant: {runtime_variant!r}"
                )
            return
        if runtime_variant not in self.page_buckets:
            raise ValueError(
                f"Unknown KV sparsity CUDA Graph page capacity: {runtime_variant!r}"
            )
        forward_batch.kv_sparsity_page_capacity = runtime_variant

    def prepare_cuda_graph_capture(self, forward_batch, runtime_variant) -> None:
        self._publish_variant(forward_batch, runtime_variant)

    def prepare_cuda_graph_replay(self, forward_batch, runtime_variant) -> None:
        self._publish_variant(forward_batch, runtime_variant)

    def prepare_cuda_graph_capture_context(self, context):
        if not dataclasses.is_dataclass(context):
            return context
        field_names = {field.name for field in dataclasses.fields(context)}
        if "kv_sparsity_controller" not in field_names:
            return context
        return dataclasses.replace(context, kv_sparsity_controller=self.controller)


def create_cuda_graph_runtime_provider(
    controller: Any,
) -> Optional[KVSparsityCudaGraphRuntimeProvider]:
    if not controller.policy.capabilities.supports_cuda_graph:
        return None
    if torch.device(controller.policy.device).type != "cuda":
        return None
    if torch.version.hip is not None:
        return None
    return KVSparsityCudaGraphRuntimeProvider(controller)
