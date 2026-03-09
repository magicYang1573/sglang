# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Engram store backends for HiCache integration.

This package provides the storage-side abstraction for Engram embedding tables,
along with CXL library utilities and micro-benchmarks.

The model-computation side (hash mapping, projections, gating) remains in
``sglang.srt.models.engram``.  The two sides communicate through the
:class:`EngramStore` interface defined here.

Engram stores are **not** KV Cache storage backends.  They exist alongside
whatever KV Cache backend the user configures (file, mooncake, hf3fs, …).
:class:`EngramStoreManager` handles the lifecycle of per-layer stores and
can be held by the scheduler or ``HiRadixCache``.
"""

from .engram_store import EngramStore, EngramStoreConfig
from .engram_store_manager import (
    EngramStoreManager,
    close_global_engram_store_manager,
    get_global_engram_store_manager,
    set_global_engram_store_manager,
)

__all__ = [
    "EngramStore",
    "EngramStoreConfig",
    "EngramStoreManager",
    "get_global_engram_store_manager",
    "set_global_engram_store_manager",
    "close_global_engram_store_manager",
]
