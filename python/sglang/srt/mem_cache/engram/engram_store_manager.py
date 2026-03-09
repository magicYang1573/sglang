# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Manager for Engram embedding store lifecycle within HiCache.

Engram stores hold embedding parameters (vocab tables) for Engram-enabled
transformer layers.  They are **not** KV Cache storage backends — they run
alongside whatever KV Cache backend the user configures (file, mooncake, …).

The manager serves as a **registry + factory**: model-side code
(``MultiHeadEmbedding``) asks the manager for a store for a given layer,
and the manager lazily creates it using the configured backend.  This avoids
the need for model code to know which backend is in use or how to configure
it.

Lifecycle
---------
1. Server/scheduler creates an ``EngramStoreManager`` with the desired
   backend configuration and sets it as the global instance.
2. Model init (``Engram`` / ``MultiHeadEmbedding``) calls
   ``get_global_engram_store_manager().get_or_create_store(...)`` to
   obtain an ``EngramStore`` for each layer.
3. During inference, ``MultiHeadEmbedding.forward`` calls
   ``store.get_many(...)`` as before.
4. On shutdown, ``manager.close()`` releases all stores.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from .engram_store import EngramStore, EngramStoreConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal factory
# ---------------------------------------------------------------------------


def _create_engram_store(
    backend: str,
    embedding_dim: int,
    vocab_size: int,
    layer_id: int,
    dtype: torch.dtype,
    cfg: EngramStoreConfig,
) -> EngramStore:
    """Instantiate the correct EngramStore backend for one layer."""
    backend = backend.lower()
    if backend == "local":
        from .local_engram_store import LocalEngramStore

        return LocalEngramStore(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            layer_id=layer_id,
            dtype=dtype,
            device=torch.device("cpu"),
        )
    elif backend == "mooncake":
        from .mooncake_engram_store import (
            MooncakeEngramStore,
            MooncakeEngramStoreConfig,
        )

        mc_config = None
        if cfg.extra:
            mc_config = MooncakeEngramStoreConfig(
                local_hostname=cfg.extra.get("mooncake_local_hostname", "localhost"),
                metadata_server=cfg.extra.get(
                    "mooncake_metadata_server",
                    "http://127.0.0.1:8080/metadata",
                ),
                master_server_address=cfg.extra.get(
                    "mooncake_master_server", "127.0.0.1:50051"
                ),
                server_id=int(cfg.extra.get("mooncake_server_id", 0)),
                num_servers=int(cfg.extra.get("mooncake_num_servers", 1)),
                vocab_block_size=int(cfg.extra.get("mooncake_vocab_block_size", 64)),
            )
        return MooncakeEngramStore(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            layer_id=layer_id,
            dtype=dtype,
            config=mc_config,
        )
    elif backend == "cxl_shm":
        from .cxl_engram_store import CxlEngramStore, CxlEngramStoreConfig

        cxl_config = None
        if cfg.extra:
            cxl_config = CxlEngramStoreConfig(
                dev_path=cfg.extra.get("cxl_dev_path", "/dev/dax0.0"),
                map_bytes=int(cfg.extra.get("cxl_map_bytes", 8 * 1024 * 1024 * 1024)),
                init_engram=cfg.extra.get("cxl_init_engram", True),
            )
        return CxlEngramStore(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            layer_id=layer_id,
            engram_layer_ids=cfg.layer_ids,
            dtype=dtype,
            config=cxl_config,
        )
    else:
        raise ValueError(f"Unknown engram store backend: {backend!r}")


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class EngramStoreManager:
    """Registry + factory for per-layer EngramStore instances.

    Stores are created lazily via :meth:`get_or_create_store` — the caller
    (model code) provides the ``vocab_size`` and ``embedding_dim`` that are
    only known after hash-mapping computation.
    """

    def __init__(self, config: EngramStoreConfig) -> None:
        self._config = config
        self._stores: Dict[int, EngramStore] = {}
        self._closed = False

        logger.info(
            "EngramStoreManager created: backend=%s, layer_ids=%s",
            config.store_backend,
            config.layer_ids,
        )

    # -- lazy creation (called from model init) ----------------------------

    def get_or_create_store(
        self,
        layer_id: int,
        vocab_size: int,
        embedding_dim: int,
        dtype: torch.dtype = torch.float16,
    ) -> EngramStore:
        """Return an existing store or create a new one for *layer_id*.

        This is the primary API for model-side code.  It is called once per
        layer during ``MultiHeadEmbedding.__init__``.
        """
        if layer_id in self._stores:
            return self._stores[layer_id]

        store = _create_engram_store(
            backend=self._config.store_backend,
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            layer_id=layer_id,
            dtype=dtype,
            cfg=self._config,
        )
        self._stores[layer_id] = store
        logger.info(
            "Created EngramStore for layer %d: backend=%s, "
            "vocab_size=%d, embedding_dim=%d",
            layer_id,
            self._config.store_backend,
            vocab_size,
            embedding_dim,
        )
        return store

    # -- read-only accessors -----------------------------------------------

    def get_store(self, layer_id: int) -> Optional[EngramStore]:
        """Return the store for *layer_id*, or ``None`` if not created yet."""
        return self._stores.get(layer_id)

    def has_layer(self, layer_id: int) -> bool:
        return layer_id in self._stores

    @property
    def config(self) -> EngramStoreConfig:
        return self._config

    @property
    def layer_ids(self) -> List[int]:
        return list(self._stores.keys())

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Release all stores.  Safe to call multiple times."""
        if self._closed:
            return
        for layer_id, store in self._stores.items():
            try:
                store.close()
            except Exception:
                logger.exception("Failed to close EngramStore for layer %d", layer_id)
        self._stores.clear()
        self._closed = True
        logger.info("EngramStoreManager closed.")


# ---------------------------------------------------------------------------
# Global singleton accessor
# ---------------------------------------------------------------------------

_GLOBAL_MANAGER: Optional[EngramStoreManager] = None


def set_global_engram_store_manager(manager: EngramStoreManager) -> None:
    """Set the process-wide EngramStoreManager singleton.

    Called once during server/scheduler initialisation.
    """
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is not None:
        logger.warning(
            "Overwriting existing global EngramStoreManager; "
            "closing the old one first."
        )
        _GLOBAL_MANAGER.close()
    _GLOBAL_MANAGER = manager


def get_global_engram_store_manager() -> Optional[EngramStoreManager]:
    """Return the process-wide EngramStoreManager, or ``None``."""
    return _GLOBAL_MANAGER


def close_global_engram_store_manager() -> None:
    """Close and clear the global manager."""
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is not None:
        _GLOBAL_MANAGER.close()
        _GLOBAL_MANAGER = None
