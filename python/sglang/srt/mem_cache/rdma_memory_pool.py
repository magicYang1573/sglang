"""RDMA distributed memory pool for HiSparse KV Cache.

Simulates a distributed KV Cache pool where only a fraction of requests
have their KV data cached locally in DRAM.  The rest reside on a "remote"
node and must be fetched via RDMA before decode can start.

The ``rdma_scatter_ext`` native extension must be **pre-built** before
starting SGLang (no runtime JIT).  From the repo root::

    cd test/cxl_utils && make rdma PYTHON=$(which python3)

If the extension is missing or fails to import, initialization raises
``RuntimeError`` (there is no CPU simulation fallback).

Layout model — page-first remote, layer-first local:

  The remote buffer (``remote_buffer``) uses **page-first** layout:
    shape = [num_tokens, layer_num, 1, kv_cache_dim]
  so that all layers of one token are contiguous in memory, matching how
  real distributed KV pools (Mooncake, NIXL) organise data for bulk
  transfer.

  The local host pool (``kv_buffer``) keeps the standard **layer-first**
  layout:
    shape = [layer_num, num_tokens, 1, kv_cache_dim]
  used by HiSparse swap-in kernels.

Transfer model — bulk RDMA READ + transpose:

  Cold path (Round 1):
    ``simulate_remote_write`` transposes kv_buffer (layer-first) into
    remote_buffer (page-first) once after the first prefill.

  Warm path (Round 2, prefix cache hit):
    Prefix KV already exists in remote_buffer from Round 1.
    ``prefetch_for_decode`` does a bulk RDMA READ from remote_buffer
    into landing_staging, then transposes page-first → layer-first
    back into kv_buffer.  No simulate_remote_write needed.

Uses real RDMA NIC loopback — the "remote" data is stored in a local
DRAM region registered as an RDMA MR, and fetched through the NIC
hardware (ibv_post_send RDMA READ → ibv_poll_cq).

Two multi-NIC modes (controlled by ``rdma_pool.mode``):

  ``rank_interleave`` (default): each TP rank is assigned a single NIC
  via ``tp_rank % len(ib_devs)``.  Single request → single NIC.

  ``request_striping``: a single request's KV is split by token range
  into shards, each shard read in parallel by a different NIC.  This
  aggregates bandwidth from multiple RNICs for long-context prefetch.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Dict, List, Optional, Set

import torch

from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost

logger = logging.getLogger(__name__)

# Loaded once; RDMALoopbackContext instances share this module.
_rdma_ext_module: Optional[object] = None


def _load_rdma_ext():
    """Import the pre-built ``rdma_scatter_ext`` extension (no JIT).

    Build before starting the server::

        cd <repo>/test/cxl_utils && make rdma PYTHON=<same-python-as-sglang>

    Raises:
        RuntimeError: if the ``.so`` is missing or import fails.
    """
    global _rdma_ext_module
    if _rdma_ext_module is not None:
        return _rdma_ext_module

    import sys
    import sysconfig
    from pathlib import Path

    sglang_root = Path(__file__).resolve().parents[4]
    cxl_utils = sglang_root / "test" / "cxl_utils"

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    preferred = cxl_utils / f"rdma_scatter_ext{ext_suffix}"

    if preferred.is_file():
        so_hint = str(preferred)
    else:
        candidates = sorted(
            cxl_utils.glob("rdma_scatter_ext*.so"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(
                "Pre-built rdma_scatter_ext is missing. Compile before starting "
                "SGLang (runtime JIT is disabled):\n"
                f"  cd {cxl_utils} && make rdma PYTHON={sys.executable}\n"
                f"Expected file: rdma_scatter_ext{ext_suffix}\n"
                "Requires: libibverbs-dev, pybind11, Python dev headers."
            )
        so_hint = str(candidates[0])
        logger.warning(
            "rdma_scatter_ext%s not found; using newest %s — rebuild with "
            "``make rdma PYTHON=%s`` if import fails.",
            ext_suffix,
            candidates[0].name,
            sys.executable,
        )

    cxl_str = str(cxl_utils)
    if cxl_str not in sys.path:
        sys.path.insert(0, cxl_str)

    try:
        import rdma_scatter_ext as ext  # type: ignore[import-not-found]

        _rdma_ext_module = ext
        logger.info("Loaded pre-built rdma_scatter_ext from %s", cxl_utils)
        return ext
    except Exception as e:
        raise RuntimeError(
            "Failed to import pre-built rdma_scatter_ext. "
            "Rebuild with the same Python you use to launch SGLang:\n"
            f"  cd {cxl_utils} && make clean && make rdma PYTHON={sys.executable}\n"
            f"(looked for rdma_scatter_ext{ext_suffix} under {cxl_utils}; "
            f"last candidate: {so_hint})\n"
            f"Import error: {e}"
        ) from e


def _resolve_ib_dev(rdma_config: dict, tp_rank: int) -> str:
    """Pick the IB device for this rank from config (rank_interleave mode)."""
    ib_devs: Optional[List[str]] = rdma_config.get("ib_devs")
    if ib_devs and len(ib_devs) > 0:
        return ib_devs[tp_rank % len(ib_devs)]
    return rdma_config.get("ib_dev", "mlx5_0")


def _resolve_ib_devs_for_striping(
    rdma_config: dict, max_rnics: int
) -> List[str]:
    """Return the list of IB devices to use for request_striping mode."""
    ib_devs: Optional[List[str]] = rdma_config.get("ib_devs")
    if not ib_devs:
        return [rdma_config.get("ib_dev", "mlx5_0")]
    return ib_devs[:max_rnics]


class RDMALoopbackContext:
    """Manages RDMA resources for NIC loopback bulk-read.

    Registers remote_buffer region as the RDMA remote MR (READ source)
    and landing_staging as the RDMA local MR (READ destination).
    ``bulk_read(total_bytes)`` issues a contiguous RDMA READ.
    """

    def __init__(
        self,
        ib_dev: str,
        remote_buf_ptr: int,
        remote_buf_size: int,
        landing_buf_ptr: int,
        landing_buf_size: int,
        max_wr: int = 4096,
    ):
        self.rdma_ext = _load_rdma_ext()

        self.ib_dev = ib_dev

        if hasattr(self.rdma_ext, "RDMAContext"):
            self._handle = self.rdma_ext.RDMAContext()
            self._handle.init(
                ib_dev,
                landing_buf_ptr,     # local MR  = landing staging (READ dst)
                landing_buf_size,
                remote_buf_ptr,      # remote MR = remote_buffer   (READ src)
                remote_buf_size,
                max_wr,
            )
            self._use_handle = True
        else:
            self.rdma_ext.rdma_init(
                ib_dev,
                landing_buf_ptr,
                landing_buf_size,
                remote_buf_ptr,
                remote_buf_size,
                max_wr,
            )
            self._handle = None
            self._use_handle = False

        self._initialized = True
        logger.info(
            "RDMA loopback initialized: ib_dev=%s, remote=%.2f GB, "
            "landing=%.2f GB, multi_instance=%s",
            ib_dev,
            remote_buf_size / 1e9,
            landing_buf_size / 1e9,
            self._use_handle,
        )

    def bulk_read(self, total_bytes: int) -> float:
        """Execute a contiguous bulk RDMA READ of total_bytes.

        Reads from remote_buffer (remote MR) → landing_staging (local MR).
        Returns elapsed time in microseconds.
        """
        if not self._initialized:
            raise RuntimeError("RDMA context not initialized")
        if total_bytes == 0:
            return 0.0

        if self._use_handle:
            return self._handle.bulk_read(total_bytes)
        else:
            return self.rdma_ext.rdma_bulk_read(total_bytes)

    def bulk_read_offset(
        self, remote_offset: int, local_offset: int, length: int
    ) -> float:
        """RDMA READ `length` bytes from remote MR at `remote_offset`
        into local MR at `local_offset`.  Returns elapsed µs.

        Requires the multi-instance RDMAContext API (bulk_read_offset).
        """
        if not self._initialized:
            raise RuntimeError("RDMA context not initialized")
        if length == 0:
            return 0.0
        if not self._use_handle:
            raise RuntimeError(
                "bulk_read_offset requires multi-instance RDMAContext API"
            )
        return self._handle.bulk_read_offset(remote_offset, local_offset, length)

    def destroy(self):
        if self._initialized:
            if self._use_handle and self._handle is not None:
                self._handle.destroy()
            else:
                self.rdma_ext.rdma_destroy()
            self._initialized = False


class RDMAMLATokenToKVPoolHost(MLATokenToKVPoolHost):
    """MLA KV Cache host pool with RDMA-backed distributed pool.

    remote_buffer: page-first [num_tokens, layer_num, 1, kv_cache_dim]
    kv_buffer:     layer-first [layer_num, num_tokens, 1, kv_cache_dim]

    Cold path (Round 1):
      1. Prefill writes KV into kv_buffer (layer-first) as normal.
      2. simulate_remote_write: transpose kv_buffer → remote_buffer (page-first).
      3. prefetch_for_decode: bulk RDMA READ remote_buffer → landing_staging,
         then transpose+scatter page-first → layer-first back to kv_buffer.

    Warm path (Round 2, prefix cache hit):
      1. Prefix KV already in remote_buffer (from Round 1).
      2. Skip simulate_remote_write.
      3. prefetch_for_decode: same RDMA + transpose as cold path.

    Two multi-NIC modes:
      rank_interleave (default): single NIC per rank, tp_rank % len(ib_devs).
      request_striping: split one request's tokens into shards across NICs,
        parallel bulk RDMA READ, per-shard transpose/scatter.
    """

    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        rdma_config: dict,
        tp_rank: int = 0,
        override_kv_cache_dim: Optional[int] = None,
    ):
        self.local_ratio = rdma_config.get("local_ratio", 0.0)
        self.max_seq_len = rdma_config.get("max_seq_len", 131072)
        self.tp_rank = tp_rank
        self._remote_reqs: Set[int] = set()
        self._rdma_stats: Dict[str, float] = {
            "total_prefetch_us": 0.0,
            "total_transpose_us": 0.0,
            "prefetch_count": 0,
        }

        self.mode = rdma_config.get("mode", "rank_interleave")
        self.max_rnics_per_request = rdma_config.get("max_rnics_per_request", 1)
        self.min_tokens_per_rnic = rdma_config.get("min_tokens_per_rnic", 8192)

        if self.mode == "request_striping":
            self._striping_ib_devs = _resolve_ib_devs_for_striping(
                rdma_config, self.max_rnics_per_request
            )
            self.ib_dev = self._striping_ib_devs[0]
        else:
            self._striping_ib_devs = []
            self.ib_dev = _resolve_ib_dev(rdma_config, tp_rank)

        # Single-NIC context (rank_interleave or fallback)
        self._rdma_ctx: Optional[RDMALoopbackContext] = None
        # Multi-NIC contexts for request_striping
        self._rdma_ctxs: List[RDMALoopbackContext] = []
        self._landing_stagings: List[torch.Tensor] = []

        super().__init__(
            device_pool=device_pool,
            host_to_device_ratio=host_to_device_ratio,
            host_size=host_size,
            page_size=page_size,
            layout=layout,
            pin_memory=True,
            device="cpu",
            allocator_type="default",
            override_kv_cache_dim=override_kv_cache_dim,
        )

    def init_kv_buffer(self):
        """Allocate host pool (layer-first) + remote buffer (page-first) + landing staging(s)."""
        buffer = super().init_kv_buffer()

        # remote_buffer: page-first layout [num_tokens, layer_num, 1, kv_cache_dim]
        self.remote_buffer = torch.empty(
            (self.size, self.layer_num, 1, self.kv_cache_dim),
            dtype=self.dtype,
            pin_memory=True,
        )

        remote_ptr = self.remote_buffer.data_ptr()
        remote_size = self.remote_buffer.numel() * self.remote_buffer.element_size()

        if self.mode == "request_striping" and len(self._striping_ib_devs) > 1:
            self._init_striping_contexts(remote_ptr, remote_size)
        else:
            self._init_single_context(remote_ptr, remote_size)

        return buffer

    def _init_single_context(
        self, remote_ptr: int, remote_size: int
    ) -> None:
        """rank_interleave: one landing_staging, one RDMA context."""
        self.landing_staging = torch.empty(
            (self.max_seq_len, self.layer_num, 1, self.kv_cache_dim),
            dtype=self.dtype,
            pin_memory=True,
        )
        landing_ptr = self.landing_staging.data_ptr()
        landing_size = self.landing_staging.numel() * self.landing_staging.element_size()

        self._rdma_ctx = RDMALoopbackContext(
            ib_dev=self.ib_dev,
            remote_buf_ptr=remote_ptr,
            remote_buf_size=remote_size,
            landing_buf_ptr=landing_ptr,
            landing_buf_size=landing_size,
        )

        logger.info(
            "RDMA host pool (rank_interleave): tp_rank=%d, ib_dev=%s, "
            "local_ratio=%.2f, remote_buffer=%.2f GB, "
            "landing_staging=%.2f GB",
            self.tp_rank, self.ib_dev, self.local_ratio,
            remote_size / 1e9, landing_size / 1e9,
        )

    def _init_striping_contexts(
        self, remote_ptr: int, remote_size: int
    ) -> None:
        """request_striping: one RDMA context + one landing_staging per NIC."""
        num_nics = len(self._striping_ib_devs)

        # Each shard handles at most ceil(max_seq_len / num_nics) tokens
        shard_max_tokens = (self.max_seq_len + num_nics - 1) // num_nics

        # Allocate one contiguous landing buffer for all shards so each
        # sub-region can be registered as the local MR for one NIC context.
        # Total shape: [num_nics * shard_max_tokens, layer_num, 1, kv_cache_dim]
        total_staging = torch.empty(
            (num_nics * shard_max_tokens, self.layer_num, 1, self.kv_cache_dim),
            dtype=self.dtype,
            pin_memory=True,
        )

        bytes_per_token = self.layer_num * self.kv_cache_dim * self.dtype.itemsize
        shard_staging_size = shard_max_tokens * bytes_per_token

        # Also keep a single landing_staging for the fallback single-NIC path
        self.landing_staging = total_staging[:self.max_seq_len]

        self._rdma_ctxs = []
        self._landing_stagings = []
        self._shard_max_tokens = shard_max_tokens

        for i, ib_dev in enumerate(self._striping_ib_devs):
            shard_staging = total_staging[
                i * shard_max_tokens : (i + 1) * shard_max_tokens
            ]
            self._landing_stagings.append(shard_staging)

            landing_ptr = shard_staging.data_ptr()
            ctx = RDMALoopbackContext(
                ib_dev=ib_dev,
                remote_buf_ptr=remote_ptr,
                remote_buf_size=remote_size,
                landing_buf_ptr=landing_ptr,
                landing_buf_size=shard_staging_size,
            )
            self._rdma_ctxs.append(ctx)

        self._rdma_ctx = self._rdma_ctxs[0]

        logger.info(
            "RDMA host pool (request_striping): tp_rank=%d, ib_devs=%s, "
            "active_nics=%d, shard_max_tokens=%d, local_ratio=%.2f, "
            "remote_buffer=%.2f GB, total_staging=%.2f GB",
            self.tp_rank,
            self._striping_ib_devs,
            len(self._rdma_ctxs),
            shard_max_tokens,
            self.local_ratio,
            remote_size / 1e9,
            total_staging.numel() * total_staging.element_size() / 1e9,
        )

    def mark_request_locality(self, req_pool_idx: int) -> bool:
        """Determine if a request is local or remote based on local_ratio.

        Returns True if the request is remote (needs RDMA prefetch).
        """
        if random.random() < self.local_ratio:
            self._remote_reqs.discard(req_pool_idx)
            return False
        else:
            self._remote_reqs.add(req_pool_idx)
            return True

    def simulate_remote_write(
        self, req_pool_idx: int, host_indices: torch.Tensor
    ) -> None:
        """Transpose KV from kv_buffer (layer-first) into remote_buffer (page-first).

        Called ONLY on cold-start path (no prefix cache hit).
        For each valid token slot, copies all layers from kv_buffer[:, slot]
        into remote_buffer[slot, :] — a dimension transpose.
        """
        if req_pool_idx not in self._remote_reqs:
            return

        valid = host_indices[host_indices >= 0]
        if valid.numel() == 0:
            return

        valid_cpu = valid.cpu()
        # kv_buffer[layer, slot, 1, dim] → remote_buffer[slot, layer, 1, dim]
        # This is a transpose: for each slot, gather all layers.
        # kv_buffer[:, valid_cpu] has shape [layer_num, N, 1, kv_dim]
        # .permute(1,0,2,3) → [N, layer_num, 1, kv_dim] = page-first
        src = self.kv_buffer[:, valid_cpu].permute(1, 0, 2, 3).contiguous()
        self.remote_buffer[valid_cpu] = src

    def prefetch_for_decode(
        self, req_pool_idx: int, host_indices: torch.Tensor
    ) -> float:
        """Bulk RDMA READ from remote_buffer (page-first) → transpose+scatter to kv_buffer (layer-first).

        Called ONCE per remote request when transitioning to decode.
        Works for both cold path (after simulate_remote_write) and warm path
        (prefix KV already in remote_buffer from a previous round).
        Local requests skip entirely.  Returns RDMA transfer time in µs.

        Dispatches to striped multi-NIC path when mode=request_striping and
        the sequence is long enough to benefit.
        """
        if req_pool_idx not in self._remote_reqs:
            return 0.0

        valid = host_indices[host_indices >= 0]
        if valid.numel() == 0:
            return 0.0

        # Dispatch to striped path if applicable
        num_tokens = valid.numel()
        if (
            self.mode == "request_striping"
            and len(self._rdma_ctxs) > 1
            and num_tokens >= self.min_tokens_per_rnic
        ):
            return self._prefetch_for_decode_striped(req_pool_idx, valid)

        return self._prefetch_for_decode_single(req_pool_idx, valid)

    def _prefetch_for_decode_single(
        self, req_pool_idx: int, valid: torch.Tensor
    ) -> float:
        """Single-NIC prefetch path (rank_interleave or short sequence fallback)."""
        valid_cpu = valid.cpu()
        num_tokens = valid_cpu.numel()
        bytes_per_token = self.layer_num * self.kv_cache_dim * self.dtype.itemsize
        total_bytes = num_tokens * bytes_per_token

        self.landing_staging[:num_tokens] = self.remote_buffer[valid_cpu]

        if self._rdma_ctx is None:
            raise RuntimeError(
                "RDMA context is unset; RDMAMLATokenToKVPoolHost requires a "
                "successful RDMALoopbackContext init."
            )
        rdma_us = self._rdma_ctx.bulk_read(total_bytes)

        t_transpose_start = time.monotonic()
        src = self.landing_staging[:num_tokens].permute(1, 0, 2, 3).contiguous()
        self.kv_buffer[:, valid_cpu] = src
        t_transpose_end = time.monotonic()
        transpose_us = (t_transpose_end - t_transpose_start) * 1e6

        self._rdma_stats["total_prefetch_us"] += rdma_us
        self._rdma_stats["total_transpose_us"] += transpose_us
        self._rdma_stats["prefetch_count"] += 1

        logger.debug(
            "RDMA prefetch single-NIC: req=%d, tokens=%d, ib_dev=%s, "
            "rdma=%.1f us, transpose=%.1f us, total=%.1f us (%.2f MB)",
            req_pool_idx, num_tokens, self.ib_dev,
            rdma_us, transpose_us, rdma_us + transpose_us,
            total_bytes / 1e6,
        )
        return rdma_us

    def _prefetch_for_decode_striped(
        self, req_pool_idx: int, valid: torch.Tensor
    ) -> float:
        """Multi-NIC striped prefetch: split tokens across NICs, parallel RDMA READ.

        Steps:
          1. Determine shard count S = min(num_ctxs, ceil(num_tokens / min_tokens_per_rnic))
          2. Split valid host_indices into S contiguous token ranges
          3. Pack each shard's page-first data into remote_buffer (already done by simulate_remote_write)
          4. Copy each shard's remote_buffer region into its per-NIC landing_staging
          5. Parallel bulk RDMA READ across S NICs (threads release GIL in C)
          6. Per-shard transpose/scatter from landing_staging (page-first) → kv_buffer (layer-first)
        """
        valid_cpu = valid.cpu()
        num_tokens = valid_cpu.numel()
        num_ctxs = len(self._rdma_ctxs)
        bytes_per_token = self.layer_num * self.kv_cache_dim * self.dtype.itemsize

        # Determine actual shard count
        num_shards = min(
            num_ctxs,
            max(1, (num_tokens + self.min_tokens_per_rnic - 1) // self.min_tokens_per_rnic),
        )
        if num_shards <= 1:
            return self._prefetch_for_decode_single(req_pool_idx, valid)

        # Split token indices into shards (contiguous ranges)
        shard_size = (num_tokens + num_shards - 1) // num_shards
        shard_ranges = []  # (start, end) indices into valid_cpu
        for s in range(num_shards):
            start = s * shard_size
            end = min(start + shard_size, num_tokens)
            if start < end:
                shard_ranges.append((start, end))
        num_shards = len(shard_ranges)

        # Step 1: Pack each shard's page-first data into its landing_staging
        for s, (start, end) in enumerate(shard_ranges):
            shard_indices = valid_cpu[start:end]
            shard_len = end - start
            self._landing_stagings[s][:shard_len] = self.remote_buffer[shard_indices]

        # Step 2: Parallel RDMA bulk READ across NICs.
        # Each NIC reads from its own landing_staging (which serves as both
        # the packed source and the RDMA local MR destination via loopback).
        rdma_results: List[float] = [0.0] * num_shards
        errors: List[Optional[Exception]] = [None] * num_shards

        def _shard_rdma_read(shard_idx: int, shard_len: int) -> None:
            try:
                total = shard_len * bytes_per_token
                ctx = self._rdma_ctxs[shard_idx]
                rdma_results[shard_idx] = ctx.bulk_read(total)
            except Exception as e:
                errors[shard_idx] = e

        threads = []
        for s, (start, end) in enumerate(shard_ranges):
            t = threading.Thread(
                target=_shard_rdma_read, args=(s, end - start)
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        for s, err in enumerate(errors):
            if err is not None:
                raise RuntimeError(
                    f"RDMA striped shard {s} failed (no single-NIC fallback): {err}"
                ) from err

        rdma_us = max(rdma_results)

        # Step 3: Per-shard transpose/scatter from landing_staging → kv_buffer
        t_transpose_start = time.monotonic()
        for s, (start, end) in enumerate(shard_ranges):
            shard_len = end - start
            shard_indices = valid_cpu[start:end]
            src = (
                self._landing_stagings[s][:shard_len]
                .permute(1, 0, 2, 3)
                .contiguous()
            )
            self.kv_buffer[:, shard_indices] = src
        t_transpose_end = time.monotonic()
        transpose_us = (t_transpose_end - t_transpose_start) * 1e6

        total_bytes = num_tokens * bytes_per_token
        self._rdma_stats["total_prefetch_us"] += rdma_us
        self._rdma_stats["total_transpose_us"] += transpose_us
        self._rdma_stats["prefetch_count"] += 1

        logger.debug(
            "RDMA prefetch striped: req=%d, tokens=%d, shards=%d, "
            "ib_devs=%s, rdma_max=%.1f us [%s], transpose=%.1f us, "
            "total=%.1f us (%.2f MB)",
            req_pool_idx, num_tokens, num_shards,
            [self._striping_ib_devs[i] for i in range(num_shards)],
            rdma_us,
            ", ".join(f"{r:.1f}" for r in rdma_results[:num_shards]),
            transpose_us, rdma_us + transpose_us,
            total_bytes / 1e6,
        )
        return rdma_us

    def clear_request_locality(self, req_pool_idx: int) -> None:
        """Clean up locality tracking when a request finishes."""
        self._remote_reqs.discard(req_pool_idx)

    def get_rdma_stats(self) -> Dict[str, float]:
        """Return accumulated RDMA prefetch statistics."""
        stats = dict(self._rdma_stats)
        stats["mode"] = self.mode  # type: ignore[assignment]
        stats["num_nics"] = len(self._rdma_ctxs) if self._rdma_ctxs else (1 if self._rdma_ctx else 0)
        if stats["prefetch_count"] > 0:
            stats["avg_prefetch_us"] = (
                stats["total_prefetch_us"] / stats["prefetch_count"]
            )
            stats["avg_transpose_us"] = (
                stats["total_transpose_us"] / stats["prefetch_count"]
            )
        return stats

    def __del__(self):
        if self._rdma_ctxs:
            for ctx in self._rdma_ctxs:
                ctx.destroy()
            self._rdma_ctxs = []
            self._rdma_ctx = None
        elif self._rdma_ctx is not None:
            self._rdma_ctx.destroy()
