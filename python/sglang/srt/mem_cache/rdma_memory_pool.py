"""RDMA distributed memory pool simulation for HiSparse KV Cache.

Simulates a distributed KV Cache pool where only a fraction of requests
have their KV data cached locally in DRAM.  The rest reside on a "remote"
node and must be fetched via RDMA before decode can start.

When a request transitions from prefill/staging to decode,
``prefetch_for_decode()`` is called ONCE to bulk-transfer all remote KV data
into local staging buffer.  After this one-time prefetch, all subsequent
decode steps read from local memory without further RDMA operations.

Uses real RDMA NIC loopback -- the "remote" data is stored in a local DRAM
region registered as an RDMA MR, and fetched through the NIC hardware
(ibv_post_send RDMA READ -> ibv_poll_cq).  This produces real hardware
latency, which is a lower bound of cross-machine RDMA latency.

Multi-NIC interleave: when ``ib_devs`` is provided in rdma_config, each
TP rank is assigned a NIC via ``tp_rank % len(ib_devs)``.  This uses the
per-instance ``RDMAContext`` class so that each rank holds an independent
RDMA context on its assigned NIC.
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
    """Load the RDMA pybind11 extension, JIT-compiling if necessary.

    Strategy (in order):
      1. If a pre-built ``rdma_scatter_ext*.so`` already exists in
         ``test/cxl_utils/``, import it directly (fast path).
      2. Otherwise JIT-compile from source using
         ``torch.utils.cpp_extension.load()``.  The compiled .so is cached
         in ``test/cxl_utils/build/rdma_ext/`` so subsequent runs skip
         recompilation.
      3. If sources are missing or compilation fails, return ``None`` and
         let the caller fall back to simulation mode.
    """
    import sys
    from pathlib import Path

    from torch.utils.cpp_extension import load

    sglang_root = Path(__file__).resolve().parents[4]
    cxl_utils = sglang_root / "test" / "cxl_utils"

    # Fast path: pre-built .so already present
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

    # JIT compile from source
    src_c = cxl_utils / "rdma_scatter_read.c"
    src_cpp = cxl_utils / "rdma_scatter_read_py.cpp"
    src_h = cxl_utils / "rdma_scatter_read.h"

    if not (src_c.exists() and src_cpp.exists() and src_h.exists()):
        logger.warning(
            "RDMA sources not found in %s. "
            "RDMA backend unavailable.",
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
    """Pick the IB device for this rank from config.

    If ``ib_devs`` (list) is present, select via ``tp_rank % len(ib_devs)``.
    Otherwise fall back to the scalar ``ib_dev`` field.
    """
    ib_devs: Optional[List[str]] = rdma_config.get("ib_devs")
    if ib_devs and len(ib_devs) > 0:
        return ib_devs[tp_rank % len(ib_devs)]
    return rdma_config.get("ib_dev", "mlx5_0")


class RDMALoopbackContext:
    """Manages RDMA resources for loopback bulk-read simulation.

    Uses the per-instance ``RDMAContext`` class from rdma_scatter_ext so
    that multiple ranks can each hold an independent context on different
    IB devices (multi-NIC interleave).
    """

    def __init__(
        self,
        ib_dev: str,
        remote_buf_ptr: int,
        remote_buf_size: int,
        staging_buf_ptr: int,
        staging_buf_size: int,
        max_wr: int = 4096,
    ):
        self.rdma_ext = _try_load_rdma_ext()
        if self.rdma_ext is None:
            raise RuntimeError(
                "RDMA extension (rdma_scatter_ext) not found. "
                "Build it with 'make rdma' in test/cxl_utils/"
            )

        self.ib_dev = ib_dev
        self.remote_buf_ptr = remote_buf_ptr
        self.remote_buf_size = remote_buf_size
        self.staging_buf_ptr = staging_buf_ptr
        self.staging_buf_size = staging_buf_size

        # Use per-instance RDMAContext (supports multi-NIC interleave)
        if hasattr(self.rdma_ext, "RDMAContext"):
            self._handle = self.rdma_ext.RDMAContext()
            self._handle.init(
                ib_dev,
                staging_buf_ptr,
                staging_buf_size,
                remote_buf_ptr,
                remote_buf_size,
                max_wr,
            )
            self._use_handle = True
        else:
            # Fallback to legacy global-singleton API
            self.rdma_ext.rdma_init(
                ib_dev,
                staging_buf_ptr,
                staging_buf_size,
                remote_buf_ptr,
                remote_buf_size,
                max_wr,
            )
            self._handle = None
            self._use_handle = False

        self._initialized = True
        logger.info(
            "RDMA loopback initialized: ib_dev=%s, remote=%.2f GB, "
            "staging=%.2f GB, multi_instance=%s",
            ib_dev,
            remote_buf_size / 1e9,
            staging_buf_size / 1e9,
            self._use_handle,
        )

    def bulk_read(self, indices: torch.Tensor, item_size: int) -> float:
        """Execute RDMA READs for the given token indices.

        Returns elapsed time in microseconds.
        """
        if not self._initialized:
            raise RuntimeError("RDMA context not initialized")

        top_k = indices.numel()
        if top_k == 0:
            return 0.0

        indices_cpu = indices.cpu().contiguous()
        if self._use_handle:
            elapsed_us = self._handle.scatter_read(
                indices_cpu.data_ptr(), top_k, item_size
            )
        else:
            elapsed_us = self.rdma_ext.rdma_scatter_read(
                indices_cpu.data_ptr(), top_k, item_size
            )
        return elapsed_us

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

    Multi-NIC interleave: each TP rank is assigned a NIC via
    ``tp_rank % len(ib_devs)`` so that different ranks use different
    NICs for their RDMA loopback, distributing PCIe traffic.

    Lifecycle:
      1. Prefill/staging: KV written to host pool (pinned DRAM) as normal.
         If the request is marked "remote", KV is copied to remote_buffer
         (RDMA MR) to simulate "KV stored on remote node".
      2. Staging->Decode transition: ``prefetch_for_decode()`` called ONCE
         per remote request.  RDMA READs fetch remote_buffer -> staging_buffer,
         then staging is copied back to host_pool.
      3. All decode steps: HiSparse reads from unified host_pool view.
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
        self.tp_rank = tp_rank
        self._rdma_ctx: Optional[RDMALoopbackContext] = None
        self._remote_reqs: Set[int] = set()
        self._rdma_stats: Dict[str, float] = {
            "total_prefetch_us": 0.0,
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
        """Allocate host pool + shadow remote buffer + staging buffer."""
        buffer = super().init_kv_buffer()

        self.remote_buffer = torch.empty_like(buffer, pin_memory=True)
        self.staging_buffer = torch.empty_like(buffer, pin_memory=True)

        remote_ptr = self.remote_buffer.data_ptr()
        remote_size = self.remote_buffer.numel() * self.remote_buffer.element_size()
        staging_ptr = self.staging_buffer.data_ptr()
        staging_size = self.staging_buffer.numel() * self.staging_buffer.element_size()

        try:
            self._rdma_ctx = RDMALoopbackContext(
                ib_dev=self.ib_dev,
                remote_buf_ptr=remote_ptr,
                remote_buf_size=remote_size,
                staging_buf_ptr=staging_ptr,
                staging_buf_size=staging_size,
            )
        except RuntimeError as e:
            logger.warning(
                "RDMA loopback init failed (%s), falling back to simulation mode",
                e,
            )
            self._rdma_ctx = None

        logger.info(
            "RDMA host pool: tp_rank=%d, ib_dev=%s, local_ratio=%.2f, "
            "remote_buffer=%.2f GB, staging_buffer=%.2f GB, rdma_available=%s",
            self.tp_rank,
            self.ib_dev,
            self.local_ratio,
            remote_size / 1e9,
            staging_size / 1e9,
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

    def prefetch_for_decode(
        self, req_pool_idx: int, host_indices: torch.Tensor
    ) -> float:
        """One-time RDMA READ: fetch a remote request's full KV into local.

        Called ONCE per request when transitioning to decode.
        Local requests skip this entirely (no RDMA overhead).

        Returns: RDMA transfer time in microseconds.
        """
        if req_pool_idx not in self._remote_reqs:
            return 0.0

        valid = host_indices[host_indices >= 0]
        if valid.numel() == 0:
            return 0.0

        valid_cpu = valid.cpu()
        num_tokens = valid_cpu.numel()
        item_size = self.token_stride_size

        if self._rdma_ctx is not None:
            rdma_us = self._rdma_ctx.bulk_read(valid_cpu, item_size)
        else:
            t0 = time.monotonic()
            for layer_id in range(self.layer_num):
                self.staging_buffer[layer_id, valid_cpu] = self.remote_buffer[
                    layer_id, valid_cpu
                ]
            rdma_us = (time.monotonic() - t0) * 1e6

        for layer_id in range(self.layer_num):
            self.kv_buffer[layer_id, valid_cpu] = self.staging_buffer[
                layer_id, valid_cpu
            ]

        self._rdma_stats["total_prefetch_us"] += rdma_us
        self._rdma_stats["prefetch_count"] += 1

        logger.debug(
            "RDMA prefetch: req_pool_idx=%d, tokens=%d, ib_dev=%s, "
            "%.1f us (%.2f MB)",
            req_pool_idx,
            num_tokens,
            self.ib_dev,
            rdma_us,
            num_tokens * item_size * self.layer_num / 1e6,
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
        return stats

    def __del__(self):
        if self._rdma_ctx is not None:
            self._rdma_ctx.destroy()
