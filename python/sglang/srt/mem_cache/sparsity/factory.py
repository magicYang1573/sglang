"""Factory helpers for sparse attention configuration and controller setup."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.deepseek_nsa import DeepSeekNSAAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    BackendAdaptor,
    FlashAttentionAdaptor,
    NSABackendAdaptor,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
    SparseAlgorithmController,
    SparseConfig,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)


_ALGORITHM_REGISTRY = {
    "quest": lambda config, device, **kw: QuestAlgorithm(config, device, **kw),
    "deepseek_nsa": lambda config, device, **kw: DeepSeekNSAAlgorithm(
        config, device, **kw
    ),
}


def _create_sparse_algorithm(
    config: SparseConfig,
    device: torch.device,
    **kwargs,
) -> BaseSparseAlgorithm:
    if config.algorithm is None:
        raise ValueError("SparseConfig.algorithm must be set")
    algorithm_name = config.algorithm.lower()
    factory = _ALGORITHM_REGISTRY.get(algorithm_name)
    if factory is None:
        raise ValueError(
            f"Unknown sparse algorithm: {algorithm_name!r}. "
            f"Registered: {list(_ALGORITHM_REGISTRY.keys())}"
        )
    return factory(config, device, **kwargs)


def _create_backend_adaptor(
    backend: Optional[str],
    device: torch.device,
    sparse_algorithm: BaseSparseAlgorithm,
    req_to_token_pool: "ReqToTokenPool",
) -> BackendAdaptor:
    """Create a backend adaptor appropriate for ``(backend, algorithm)``."""
    if isinstance(sparse_algorithm, DeepSeekNSAAlgorithm):
        return NSABackendAdaptor(device, req_to_token_pool)

    backend_key = (backend or "fa3").lower()
    if backend_key in ("fa3", "flashattention", "fa3_sparse_decode"):
        # Lazy import to avoid pulling FA3 code paths on non-CUDA builds.
        from sglang.srt.mem_cache.sparsity.backend.fa3_sparse_decode_adapter import (
            Fa3SparseDecodeAdapter,
        )

        return Fa3SparseDecodeAdapter(device)

    if backend_key == "fa":
        return FlashAttentionAdaptor(device)

    raise ValueError(f"Unknown attention backend: {backend!r}")


def _parse_sparse_config(raw_config: Optional[str]) -> SparseConfig:
    if raw_config is not None:
        try:
            extra_config = json.loads(raw_config)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse sparse config JSON: {e}") from e
    else:
        extra_config = {}

    top_k = extra_config.pop("top_k", 2048)
    device_buffer_size = extra_config.pop("device_buffer_size", 2 * top_k)
    host_to_device_ratio = extra_config.pop("host_to_device_ratio", 2)

    if device_buffer_size < top_k:
        raise ValueError(
            f"device_buffer_size ({device_buffer_size}) must be >= top_k ({top_k})"
        )

    algorithm = extra_config.pop("algorithm", None)
    backend = extra_config.pop("backend", None)
    min_sparse_prompt_len = extra_config.pop("min_sparse_prompt_len", None)
    page_size = extra_config.pop("page_size", None)
    # Accept either a dedicated nested "sparse_extra_config" block or leave
    # remaining flat keys as algorithm-specific overrides.
    sparse_extra_config = extra_config.pop("sparse_extra_config", None)
    if sparse_extra_config is None:
        sparse_extra_config = extra_config

    return SparseConfig(
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        host_to_device_ratio=host_to_device_ratio,
        algorithm=algorithm,
        backend=backend,
        page_size=page_size,
        min_sparse_prompt_len=min_sparse_prompt_len,
        sparse_extra_config=sparse_extra_config,
    )


def parse_hisparse_config(server_args) -> SparseConfig:
    """Parse the legacy ``--hisparse-config`` JSON string.

    Kept for backward compatibility with the DSA/HiSparse path
    (``enable_hisparse``).
    """
    return _parse_sparse_config(getattr(server_args, "hisparse_config", None))


def parse_sparse_decode_config(server_args) -> SparseConfig:
    """Parse the new ``--sparse-decode-config`` JSON string, with defaults
    appropriate for the non-native sparse decode path.
    """
    cfg = _parse_sparse_config(getattr(server_args, "sparse_decode_config", None))
    # Safer defaults for non-native sparse decode.
    if cfg.algorithm is None:
        cfg.algorithm = "quest"
    if cfg.backend is None:
        cfg.backend = "fa3"
    if cfg.page_size is None:
        cfg.page_size = 1
    if cfg.min_sparse_prompt_len is None:
        cfg.min_sparse_prompt_len = 4096
    return cfg


def build_sparse_algorithm_controller(
    config: SparseConfig,
    device: torch.device,
    req_to_token_pool: "ReqToTokenPool",
    start_layer: int,
    end_layer: int,
) -> SparseAlgorithmController:
    """Compose an algorithm + adapter into a controller ready to be bound to
    ``HiSparseCoordinator``.
    """
    algorithm = _create_sparse_algorithm(config, device)
    adapter = _create_backend_adaptor(
        config.backend, device, algorithm, req_to_token_pool
    )
    return SparseAlgorithmController(
        config=config,
        algorithm=algorithm,
        adapter=adapter,
        req_to_token_pool=req_to_token_pool,
        device=device,
        start_layer=start_layer,
        end_layer=end_layer,
    )
