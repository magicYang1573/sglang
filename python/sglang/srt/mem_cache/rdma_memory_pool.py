"""RDMA distributed memory pool simulation for HiSparse KV Cache.

Simulates a distributed KV Cache pool where only a fraction of requests
have their KV data cached locally in DRAM.  The rest reside on a "remote"
node and must be fetched via RDMA before decode can start.

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

    def destroy(self):
        if self._initialized:
            if self._use_handle and self._handle is not None:
                self._handle.destroy()
            else:
                self.rdma_ext.rdma_destroy()
            self._initialized = False


class RDMAMLATokenToKVPoolHost(MLATokenToKVPoolHost):
    """MLA KV Cache host pool with RDMA distributed pool simulation.

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
            "total_transpose_us": 0.0,
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
        """Allocate host pool (layer-first) + remote buffer (page-first) + landing staging."""
        buffer = super().init_kv_buffer()
        # buffer (kv_buffer) shape: [layer_num, num_tokens, 1, kv_cache_dim]

        # remote_buffer: page-first layout [num_tokens, layer_num, 1, kv_cache_dim]
        self.remote_buffer = torch.empty(
            (self.size, self.layer_num, 1, self.kv_cache_dim),
            dtype=self.dtype,
            pin_memory=True,
        )

        # landing_staging: page-first, sized for one request's max KV
        # shape: [max_seq_len, layer_num, 1, kv_cache_dim]
        bytes_per_token = self.layer_num * self.kv_cache_dim * self.dtype.itemsize
        staging_bytes = self.max_seq_len * bytes_per_token
        self.landing_staging = torch.empty(
            (self.max_seq_len, self.layer_num, 1, self.kv_cache_dim),
            dtype=self.dtype,
            pin_memory=True,
        )

        remote_ptr = self.remote_buffer.data_ptr()
        remote_size = self.remote_buffer.numel() * self.remote_buffer.element_size()
        landing_ptr = self.landing_staging.data_ptr()
        landing_size = self.landing_staging.numel() * self.landing_staging.element_size()

        try:
            self._rdma_ctx = RDMALoopbackContext(
                ib_dev=self.ib_dev,
                remote_buf_ptr=remote_ptr,
                remote_buf_size=remote_size,
                landing_buf_ptr=landing_ptr,
                landing_buf_size=landing_size,
            )
        except RuntimeError as e:
            logger.warning(
                "RDMA loopback init failed (%s), falling back to simulation mode",
                e,
            )
            self._rdma_ctx = None

        logger.info(
            "RDMA host pool (page-first remote): tp_rank=%d, ib_dev=%s, "
            "local_ratio=%.2f, remote_buffer=%.2f GB (page-first), "
            "landing_staging=%.2f GB, rdma_available=%s",
            self.tp_rank,
            self.ib_dev,
            self.local_ratio,
            remote_size / 1e9,
            landing_size / 1e9,
            self._rdma_ctx is not None,
        )
        return buffer

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
        """
        if req_pool_idx not in self._remote_reqs:
            return 0.0

        valid = host_indices[host_indices >= 0]
        if valid.numel() == 0:
            return 0.0

        valid_cpu = valid.cpu()
        num_tokens = valid_cpu.numel()
        bytes_per_token = self.layer_num * self.kv_cache_dim * self.dtype.itemsize
        total_bytes = num_tokens * bytes_per_token

        # Step 1: Copy remote_buffer[valid slots] → landing_staging[:num_tokens]
        #         (contiguous page-first block ready for RDMA)
        #         In a real system, this data is already contiguous at the remote
        #         side; here we just ensure the staging region is packed.
        self.landing_staging[:num_tokens] = self.remote_buffer[valid_cpu]

        # Step 2: Bulk RDMA READ (landing_staging as both src and dst via loopback,
        #         simulating the NIC transfer latency for total_bytes)
        if self._rdma_ctx is not None:
            rdma_us = self._rdma_ctx.bulk_read(total_bytes)
        else:
            t0 = time.monotonic()
            # Simulation: data is already in landing_staging, just measure time
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            rdma_us = (time.monotonic() - t0) * 1e6

        # Step 3: Transpose + scatter: landing_staging (page-first) → kv_buffer (layer-first)
        # landing_staging[:num_tokens] shape: [num_tokens, layer_num, 1, kv_dim]
        # → permute to [layer_num, num_tokens, 1, kv_dim] → scatter into kv_buffer slots
        t_transpose_start = time.monotonic()
        src = self.landing_staging[:num_tokens].permute(1, 0, 2, 3).contiguous()
        self.kv_buffer[:, valid_cpu] = src
        t_transpose_end = time.monotonic()
        transpose_us = (t_transpose_end - t_transpose_start) * 1e6

        self._rdma_stats["total_prefetch_us"] += rdma_us
        self._rdma_stats["total_transpose_us"] += transpose_us
        self._rdma_stats["prefetch_count"] += 1

        logger.debug(
            "RDMA prefetch (bulk+transpose): req=%d, tokens=%d, ib_dev=%s, "
            "rdma=%.1f us, transpose=%.1f us, total=%.1f us (%.2f MB)",
            req_pool_idx,
            num_tokens,
            self.ib_dev,
            rdma_us,
            transpose_us,
            rdma_us + transpose_us,
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
            stats["avg_transpose_us"] = (
                stats["total_transpose_us"] / stats["prefetch_count"]
            )
        return stats

    def __del__(self):
        if self._rdma_ctx is not None:
            self._rdma_ctx.destroy()
