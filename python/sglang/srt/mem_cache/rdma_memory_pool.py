"""RDMA distributed memory pool simulation for HiSparse KV Cache.

Simulates a distributed KV Cache pool where only a fraction of requests
have their KV data cached locally in DRAM.  The rest reside on a "remote"
node and must be fetched via RDMA before decode can start.

Transfer model — gather → bulk RDMA READ → scatter:

  KV cache storage format (layer-first token-slot pool) is unchanged.
  Before RDMA transfer, scattered KV slots are gathered into a small
  contiguous staging buffer, transferred as a single bulk RDMA READ
  (1 MB chunks), then scattered back to the host pool's original slot
  positions.  This matches how production systems (e.g. Mooncake PD
  disaggregation) handle RDMA KV cache transfer.

  The gather/landing staging buffers are sized to hold ONE request's
  maximum KV data (max_seq_len × layer_num × kv_cache_dim × dtype_size),
  NOT a full copy of the host pool.

Uses real RDMA NIC loopback — the "remote" data is stored in a local
DRAM region registered as an RDMA MR, and fetched through the NIC
hardware (ibv_post_send RDMA READ → ibv_poll_cq).  This produces real
hardware latency, which is a lower bound of cross-machine RDMA latency.

Multi-NIC interleave: when ``ib_devs`` is provided in rdma_config, each
TP rank is assigned a NIC via ``tp_rank % len(ib_devs)``.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Dict, List, Optional, Set

import torch

from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost

logger = logging.getLogger(__name__)


def _try_load_rdma_ext():
    """Load the RDMA pybind11 extension, JIT-compiling if necessary."""
    import sys
    from pathlib import Path

    from torch.utils.cpp_extension import load

    sglang_root = Path(__file__).resolve().parents[4]
    cxl_utils = sglang_root / "test" / "cxl_utils"

    so_files = list(cxl_utils.glob("rdma_scatter_ext*.so"))
    if so_files:
        sys.path.insert(0, str(cxl_utils))
        try:
            import rdma_scatter_ext

            return rdma_scatter_ext
        except ImportError as e:
            logger.warning(
                "Pre-built rdma_scatter_ext found but failed to import: %s", e
            )

    src_c = cxl_utils / "rdma_scatter_read.c"
    src_cpp = cxl_utils / "rdma_scatter_read_py.cpp"
    src_h = cxl_utils / "rdma_scatter_read.h"

    if not (src_c.exists() and src_cpp.exists() and src_h.exists()):
        logger.warning(
            "RDMA sources not found in %s. RDMA backend unavailable.",
            cxl_utils,
        )
        return None

    build_dir = cxl_utils / "build" / "rdma_ext"
    build_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(
            "JIT-compiling rdma_scatter_ext (first run only, cached in %s)...",
            build_dir,
        )
        ext = load(
            name="rdma_scatter_ext",
            sources=[str(src_c), str(src_cpp)],
            extra_cflags=["-O3", "-Wall", "-fPIC"],
            extra_cxx_flags=["-O3", "-Wall", "-fPIC", "-std=c++17"],
            extra_ldflags=["-libverbs"],
            build_directory=str(build_dir),
            verbose=False,
            with_cuda=False,
        )
        logger.info("rdma_scatter_ext compiled successfully.")
        return ext
    except Exception as e:
        logger.warning(
            "Failed to JIT-compile rdma_scatter_ext: %s. "
            "Is libibverbs-dev installed?",
            e,
        )
        return None


def _resolve_ib_dev(rdma_config: dict, tp_rank: int) -> str:
    """Pick the IB device for this rank from config."""
    ib_devs: Optional[List[str]] = rdma_config.get("ib_devs")
    if ib_devs and len(ib_devs) > 0:
        return ib_devs[tp_rank % len(ib_devs)]
    return rdma_config.get("ib_dev", "mlx5_0")


class RDMALoopbackContext:
    """Manages RDMA resources for loopback bulk-read simulation.

    Registers gather_staging as the RDMA remote MR (READ source, simulates
    the remote-side contiguous buffer) and landing_staging as the RDMA
    local MR (READ destination).  ``bulk_read(total_bytes)`` issues a
    single contiguous RDMA READ from gather → landing.
    """

    def __init__(
        self,
        ib_dev: str,
        gather_buf_ptr: int,
        gather_buf_size: int,
        landing_buf_ptr: int,
        landing_buf_size: int,
        max_wr: int = 4096,
    ):
        self.rdma_ext = _try_load_rdma_ext()
        if self.rdma_ext is None:
            raise RuntimeError(
                "RDMA extension (rdma_scatter_ext) not found. "
                "Build it with 'make rdma' in test/cxl_utils/"
            )

        self.ib_dev = ib_dev

        if hasattr(self.rdma_ext, "RDMAContext"):
            self._handle = self.rdma_ext.RDMAContext()
            self._handle.init(
                ib_dev,
                landing_buf_ptr,    # local MR  = landing staging (READ dst)
                landing_buf_size,
                gather_buf_ptr,     # remote MR = gather staging  (READ src)
                gather_buf_size,
                max_wr,
            )
            self._use_handle = True
        else:
            self.rdma_ext.rdma_init(
                ib_dev,
                landing_buf_ptr,
                landing_buf_size,
                gather_buf_ptr,
                gather_buf_size,
                max_wr,
            )
            self._handle = None
            self._use_handle = False

        self._initialized = True
        logger.info(
            "RDMA loopback initialized: ib_dev=%s, gather=%.2f GB, "
            "landing=%.2f GB, multi_instance=%s",
            ib_dev,
            gather_buf_size / 1e9,
            landing_buf_size / 1e9,
            self._use_handle,
        )

    def bulk_read(self, total_bytes: int) -> float:
        """Execute a contiguous bulk RDMA READ of total_bytes.

        Reads from gather_staging (remote MR) → landing_staging (local MR).
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

    def destroy(self):
        if self._initialized:
            if self._use_handle and self._handle is not None:
                self._handle.destroy()
            else:
                self.rdma_ext.rdma_destroy()
            self._initialized = False


class RDMAMLATokenToKVPoolHost(MLATokenToKVPoolHost):
    """MLA KV Cache host pool with RDMA distributed pool simulation.

    KV data is split into two categories at the request level:
      - local requests: X% of requests, KV in pinned DRAM, directly accessible
      - remote requests: (1-X)% of requests, KV must be fetched via RDMA

    Transfer uses gather → bulk RDMA → scatter pipeline:
      1. Prefill/staging: KV written to host pool (pinned DRAM) as normal.
         If remote, KV is also copied to remote_buffer (RDMA MR).
      2. Staging→Decode transition: ``prefetch_for_decode()`` called ONCE:
         a. gather: remote_buffer scattered slots → gather_staging (contiguous)
         b. bulk RDMA READ: gather_staging → NIC loopback → landing_staging
         c. scatter: landing_staging (contiguous) → kv_buffer scattered slots
      3. All decode steps: HiSparse reads from unified kv_buffer view.
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
        self.ib_dev = _resolve_ib_dev(rdma_config, tp_rank)
        self.max_seq_len = rdma_config.get("max_seq_len", 131072)
        self.tp_rank = tp_rank
        self._rdma_ctx: Optional[RDMALoopbackContext] = None
        self._remote_reqs: Set[int] = set()
        self._rdma_stats: Dict[str, float] = {
            "total_prefetch_us": 0.0,
            "total_gather_us": 0.0,
            "total_scatter_us": 0.0,
            "prefetch_count": 0,
        }

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
        """Allocate host pool + remote buffer + gather/landing staging."""
        buffer = super().init_kv_buffer()

        # remote_buffer: same shape as kv_buffer, stores "remote node" KV
        self.remote_buffer = torch.empty_like(buffer, pin_memory=True)

        # Staging buffers: small contiguous, sized for one request's max KV
        # Layout in staging: [layer_0_all_tokens, layer_1_all_tokens, ...]
        bytes_per_token_per_layer = self.kv_cache_dim * self.dtype.itemsize
        staging_bytes = (
            self.max_seq_len * self.layer_num * bytes_per_token_per_layer
        )
        self.gather_staging = torch.empty(
            staging_bytes, dtype=torch.uint8, pin_memory=True
        )
        self.landing_staging = torch.empty(
            staging_bytes, dtype=torch.uint8, pin_memory=True
        )

        gather_ptr = self.gather_staging.data_ptr()
        landing_ptr = self.landing_staging.data_ptr()

        try:
            self._rdma_ctx = RDMALoopbackContext(
                ib_dev=self.ib_dev,
                gather_buf_ptr=gather_ptr,
                gather_buf_size=staging_bytes,
                landing_buf_ptr=landing_ptr,
                landing_buf_size=staging_bytes,
            )
        except RuntimeError as e:
            logger.warning(
                "RDMA loopback init failed (%s), falling back to simulation mode",
                e,
            )
            self._rdma_ctx = None

        logger.info(
            "RDMA host pool (bulk mode): tp_rank=%d, ib_dev=%s, "
            "local_ratio=%.2f, remote_buffer=%.2f GB, "
            "staging=%.2f GB each, rdma_available=%s",
            self.tp_rank,
            self.ib_dev,
            self.local_ratio,
            self.remote_buffer.numel() * self.remote_buffer.element_size() / 1e9,
            staging_bytes / 1e9,
            self._rdma_ctx is not None,
        )
        return buffer

    def mark_request_locality(self, req_pool_idx: int) -> bool:
        """Determine if a request is local or remote based on local_ratio.

        Called after staging completes for a request.
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
        """Copy a remote request's KV from host_pool to remote_buffer."""
        if req_pool_idx not in self._remote_reqs:
            return

        valid = host_indices[host_indices >= 0]
        if valid.numel() == 0:
            return

        valid_cpu = valid.cpu()
        for layer_id in range(self.layer_num):
            self.remote_buffer[layer_id, valid_cpu] = self.kv_buffer[
                layer_id, valid_cpu
            ]

    def _gather_to_staging(self, host_indices: torch.Tensor) -> int:
        """Gather scattered slots from remote_buffer into contiguous gather_staging.

        Packs data as [layer_0_all_tokens, layer_1_all_tokens, ...] into
        gather_staging.  Returns total bytes gathered.
        """
        valid = host_indices[host_indices >= 0].cpu()
        num_tokens = valid.numel()
        if num_tokens == 0:
            return 0

        bytes_per_token = self.kv_cache_dim * self.dtype.itemsize
        staging_view = self.gather_staging.view(torch.uint8)
        offset = 0
        for layer_id in range(self.layer_num):
            layer_data = self.remote_buffer[layer_id, valid].contiguous()
            layer_bytes = num_tokens * bytes_per_token
            staging_view[offset : offset + layer_bytes].copy_(
                layer_data.view(torch.uint8).reshape(-1)
            )
            offset += layer_bytes
        return offset

    def _scatter_from_staging(
        self, host_indices: torch.Tensor, total_bytes: int
    ) -> None:
        """Scatter contiguous landing_staging back into kv_buffer's scattered slots."""
        valid = host_indices[host_indices >= 0].cpu()
        num_tokens = valid.numel()
        if num_tokens == 0:
            return

        bytes_per_token = self.kv_cache_dim * self.dtype.itemsize
        staging_view = self.landing_staging.view(torch.uint8)
        offset = 0
        for layer_id in range(self.layer_num):
            layer_bytes = num_tokens * bytes_per_token
            src = (
                staging_view[offset : offset + layer_bytes]
                .view(self.dtype)
                .reshape(num_tokens, 1, self.kv_cache_dim)
            )
            self.kv_buffer[layer_id, valid] = src
            offset += layer_bytes

    def prefetch_for_decode(
        self, req_pool_idx: int, host_indices: torch.Tensor
    ) -> float:
        """One-time gather → bulk RDMA READ → scatter for a remote request.

        Called ONCE per request when transitioning to decode.
        Local requests skip entirely.  Returns RDMA transfer time in µs.
        """
        if req_pool_idx not in self._remote_reqs:
            return 0.0

        valid = host_indices[host_indices >= 0]
        if valid.numel() == 0:
            return 0.0

        # Step 1: gather remote_buffer scattered → gather_staging contiguous
        t_gather_start = time.monotonic()
        total_bytes = self._gather_to_staging(host_indices)
        t_gather_end = time.monotonic()
        gather_us = (t_gather_end - t_gather_start) * 1e6

        # Step 2: bulk RDMA READ (gather_staging → landing_staging)
        if self._rdma_ctx is not None:
            rdma_us = self._rdma_ctx.bulk_read(total_bytes)
        else:
            t0 = time.monotonic()
            self.landing_staging[:total_bytes].copy_(
                self.gather_staging[:total_bytes]
            )
            rdma_us = (time.monotonic() - t0) * 1e6

        # Step 3: scatter landing_staging contiguous → kv_buffer scattered
        t_scatter_start = time.monotonic()
        self._scatter_from_staging(host_indices, total_bytes)
        t_scatter_end = time.monotonic()
        scatter_us = (t_scatter_end - t_scatter_start) * 1e6

        self._rdma_stats["total_prefetch_us"] += rdma_us
        self._rdma_stats["total_gather_us"] += gather_us
        self._rdma_stats["total_scatter_us"] += scatter_us
        self._rdma_stats["prefetch_count"] += 1

        num_tokens = valid.numel()
        logger.debug(
            "RDMA prefetch (bulk): req=%d, tokens=%d, ib_dev=%s, "
            "gather=%.1f us, rdma=%.1f us, scatter=%.1f us, "
            "total=%.1f us (%.2f MB)",
            req_pool_idx,
            num_tokens,
            self.ib_dev,
            gather_us,
            rdma_us,
            scatter_us,
            gather_us + rdma_us + scatter_us,
            total_bytes / 1e6,
        )
        return rdma_us

    def clear_request_locality(self, req_pool_idx: int) -> None:
        """Clean up locality tracking when a request finishes."""
        self._remote_reqs.discard(req_pool_idx)

    def get_rdma_stats(self) -> Dict[str, float]:
        """Return accumulated RDMA prefetch statistics."""
        stats = dict(self._rdma_stats)
        if stats["prefetch_count"] > 0:
            stats["avg_prefetch_us"] = (
                stats["total_prefetch_us"] / stats["prefetch_count"]
            )
            stats["avg_gather_us"] = (
                stats["total_gather_us"] / stats["prefetch_count"]
            )
            stats["avg_scatter_us"] = (
                stats["total_scatter_us"] / stats["prefetch_count"]
            )
        return stats

    def __del__(self):
        if self._rdma_ctx is not None:
            self._rdma_ctx.destroy()
