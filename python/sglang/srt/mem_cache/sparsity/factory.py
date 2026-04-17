import json
import logging
from typing import Optional

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.deepseek_nsa import DeepSeekNSAAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    FlashAttentionAdaptor,
    NSABackendAdaptor,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
    SparseConfig,
    SparseCoordinator,
)

logger = logging.getLogger(__name__)

_global_sparse_coordinator: Optional[SparseCoordinator] = None

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
    algorithm_name = config.algorithm.lower()
    factory = _ALGORITHM_REGISTRY.get(algorithm_name)

    if factory is None:
        raise ValueError(f"Unknown sparse algorithm: {algorithm_name}")

    return factory(config, device, **kwargs)


def _create_backend_adaptor(
    backend: str,
    device: torch.device,
    sparse_algorithm: BaseSparseAlgorithm,
    req_to_token_pool,
):
    """Create backend adaptor."""
    if isinstance(sparse_algorithm, DeepSeekNSAAlgorithm):
        return NSABackendAdaptor(device, req_to_token_pool)

    if backend in ["fa3", "flashattention"]:
        return FlashAttentionAdaptor(device)

    raise ValueError(f"Unknown attention backend: {backend}")


def _parse_sparse_config(server_args) -> SparseConfig:
    """Parse hierarchical sparse config from JSON string.

    Required fields with defaults: top_k (2048), device_buffer_size (2*top_k),
    host_to_device_ratio (2).
    Optional fields (default None): algorithm, backend, min_sparse_prompt_len,
    page_size. All remaining fields go to sparse_extra_config.
    """
    extra_config_str = server_args.hisparse_config
    if extra_config_str is not None:
        try:
            extra_config = json.loads(extra_config_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse hisparse_config: {e}") from e
    else:
        extra_config = {}

    top_k = extra_config.pop("top_k", 2048)
    device_buffer_size = extra_config.pop("device_buffer_size", 2 * top_k)
    host_to_device_ratio = extra_config.pop("host_to_device_ratio", 2)

    algorithm = extra_config.pop("algorithm", None)
    backend = extra_config.pop("backend", None)
    min_sparse_prompt_len = extra_config.pop("min_sparse_prompt_len", None)
    page_size = extra_config.pop("page_size", None)

    if device_buffer_size < top_k:
        raise ValueError(
            f"device_buffer_size ({device_buffer_size}) must be no smaller than top_k ({top_k})"
        )

    return SparseConfig(
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        host_to_device_ratio=host_to_device_ratio,
        algorithm=algorithm,
        backend=backend,
        page_size=page_size,
        min_sparse_prompt_len=min_sparse_prompt_len,
        sparse_extra_config=extra_config,
    )


def parse_hisparse_config(server_args) -> SparseConfig:
    """Parse hisparse config from server_args, returning defaults if no config provided."""
    return _parse_sparse_config(server_args)


def create_sparse_coordinator(
    device: torch.device,
    req_to_token_pool,
    token_to_kv_pool,
    start_layer: int,
    end_layer: int,
    server_args,
    io_subsystem=None,
    **kwargs,
) -> SparseCoordinator:
    config = _parse_sparse_config(server_args)

    # Resolve page sizes.
    #
    # ``backend_page_size`` comes from the attention backend's page size
    # (server_args.page_size == token_to_kv_pool.page_size).  When the user
    # does not override ``page_size`` in --hisparse-config we keep the 1:1
    # mapping (sparse_page_size == backend_page_size, N == 1).
    backend_page_size = getattr(token_to_kv_pool, "page_size", 1)
    config.backend_page_size = backend_page_size
    if config.page_size is None:
        config.page_size = backend_page_size
    if config.min_sparse_prompt_len is None:
        config.min_sparse_prompt_len = 0

    # For SGLang's FA3 path, ``metadata.page_table`` is always stored at
    # token granularity (req_to_token slice), so the adaptor can work with
    # any ``sparse_page_size``; we still keep ``sparse_page_size >= 1`` as
    # a lower bound.  For truly paged attention backends where page_table
    # is at page granularity (e.g. flashmla_sparse used by DSA), the N-fold
    # expansion constraint (sparse_page_size = backend_page_size * N) does
    # apply, but DSA takes a different code path and does not go through
    # this factory.
    if config.page_size < 1:
        raise ValueError(
            f"HiSparse sparse_page_size must be >= 1, got {config.page_size}."
        )

    # Helpful heuristic warning: Quest keeps per-sparse-page bbox tensors on
    # GPU; when sparse_page_size is tiny (e.g. == backend_page_size == 1) the
    # pool is huge (scales with num_tokens) and often OOMs at startup.
    _algo = (config.algorithm or "deepseek_nsa").lower()
    if _algo == "quest" and config.page_size < 8:
        try:
            _total_tokens = int(
                token_to_kv_pool.get_key_buffer(
                    getattr(token_to_kv_pool, "start_layer", 0)
                ).shape[0]
            )
            _head_num = int(getattr(token_to_kv_pool, "head_num", 0))
            _head_dim = int(getattr(token_to_kv_pool, "head_dim", 0))
            _num_layers = int(getattr(token_to_kv_pool, "layer_num", 0))
            _num_pages = (_total_tokens + config.page_size - 1) // config.page_size
            _bbox_bytes = _num_pages * _head_num * _head_dim * 4 * 2 * _num_layers
            if _bbox_bytes > 2 * (1024**3):  # > 2 GB
                logger.warning(
                    "HiSparse+Quest: sparse page_size=%d is small; the bbox "
                    "representation pool will take ~%.1f GB of GPU memory "
                    "across all layers. Consider setting "
                    '\"page_size\": 16 (or a larger multiple of '
                    "backend_page_size=%d) in --hisparse-config.",
                    config.page_size,
                    _bbox_bytes / (1024**3),
                    backend_page_size,
                )
        except Exception:
            pass

    algorithm = _create_sparse_algorithm(config, device, **kwargs)
    backend_adaptor = _create_backend_adaptor(
        config.backend, device, algorithm, req_to_token_pool
    )

    coordinator = SparseCoordinator(
        config=config,
        algorithm=algorithm,
        backend_adaptor=backend_adaptor,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        start_layer=start_layer,
        end_layer=end_layer,
        device=device,
        io_subsystem=io_subsystem,
    )
    register_sparse_coordinator(coordinator)
    return coordinator


def register_sparse_coordinator(coordinator: SparseCoordinator) -> None:
    global _global_sparse_coordinator
    _global_sparse_coordinator = coordinator


def get_sparse_coordinator() -> Optional[SparseCoordinator]:
    return _global_sparse_coordinator
