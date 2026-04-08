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
"""

from __future__ import annotations

import logging
import random
import time
from typing import Dict, Optional, Set

import torch

from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost

logger = logging.getLogger(__name__)


def _try_load_rdma_ext():
    """Try to load the pre-built RDMA pybind11 extension."""
    import importlib.util
    import sys
    from pathlib import Path

    # Look for the .so in test/cxl_utils/
    sglang_root = Path(__file__).resolve().parents[4]
    cxl_utils = sglang_root / "test" / "cxl_utils"

    so_files = list(cxl_utils.glob("rdma_scatter_ext*.so"))
    if not so_files:
        return None

    sys.path.insert(0, str(cxl_utils))
    try:
        import rdma_scatter_ext

        return rdma_scatter_ext
    except ImportError as e:
        logger.warning("Failed to import rdma_scatter_ext: %s", e)
        return None


class RDMALoopbackContext:
    """Manages RDMA resources for loopback bulk-read simulation.

    Wraps the rdma_scatter_ext pybind11 module to perform bulk RDMA READs
    (contiguous reads split into max-message-size chunks) rather than
    per-token discrete reads.
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

        self.rdma_ext.rdma_init(
            ib_dev,
            staging_buf_ptr,
            staging_buf_size,
            remote_buf_ptr,
            remote_buf_size,
            max_wr,
        )
        self._initialized = True
        logger.info(
            "RDMA loopback initialized: ib_dev=%s, remote=%.2f GB, staging=%.2f GB",
            ib_dev,
            remote_buf_size / 1e9,
            staging_buf_size / 1e9,
        )

    def bulk_read(self, indices: torch.Tensor, item_size: int) -> float:
        """Execute RDMA READs for the given token indices.

        Uses per-token discrete RDMA READs (the existing scatter_read path)
        to faithfully simulate the real per-request RDMA overhead.

        Returns elapsed time in microseconds.
        """
        if not self._initialized:
            raise RuntimeError("RDMA context not initialized")

        top_k = indices.numel()
        if top_k == 0:
            return 0.0

        indices_cpu = indices.cpu().contiguous()
        elapsed_us = self.rdma_ext.rdma_scatter_read(
            indices_cpu.data_ptr(), top_k, item_size
        )
        return elapsed_us

    def destroy(self):
        if self._initialized:
            self.rdma_ext.rdma_destroy()
            self._initialized = False


class RDMAMLATokenToKVPoolHost(MLATokenToKVPoolHost):
    """MLA KV Cache host pool with RDMA distributed pool simulation.

    KV data is split into two categories at the request level:
      - local requests: X% of requests, KV in pinned DRAM, directly accessible
      - remote requests: (1-X)% of requests, KV must be fetched via RDMA

    Lifecycle:
      1. Prefill/staging: KV written to host pool (pinned DRAM) as normal.
         If the request is marked "remote", KV is copied to remote_buffer
         (RDMA MR) to simulate "KV stored on remote node".
      2. Staging->Decode transition: ``prefetch_for_decode()`` called ONCE
         per remote request.  RDMA READs fetch remote_buffer -> staging_buffer,
         then staging is copied back to host_pool.
      3. All decode steps: HiSparse reads from unified host_pool view.

    This models PD-disaggregation: decode instance fetches full KV from remote
    prefill instance once before decoding.
    """

    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        rdma_config: dict,
        override_kv_cache_dim: Optional[int] = None,
    ):
        self.local_ratio = rdma_config.get("local_ratio", 0.0)
        self.ib_dev = rdma_config.get("ib_dev", "mlx5_0")
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
        # Call parent to get the normal host pool buffer
        buffer = super().init_kv_buffer()

        # remote_buffer: regular DRAM, registered as RDMA MR.
        # Same shape as host_pool, stores "remote" requests' KV copies.
        self.remote_buffer = torch.empty_like(buffer, pin_memory=True)

        # staging_buffer: pinned DRAM, RDMA READ target.
        self.staging_buffer = torch.empty_like(buffer, pin_memory=True)

        # Register RDMA MRs and create loopback context
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
            "RDMA host pool: local_ratio=%.2f, remote_buffer=%.2f GB, "
            "staging_buffer=%.2f GB, rdma_available=%s",
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
            # Local request: KV already in host pool, no RDMA needed
            self._remote_reqs.discard(req_pool_idx)
            return False
        else:
            self._remote_reqs.add(req_pool_idx)
            return True

    def simulate_remote_write(
        self, req_pool_idx: int, host_indices: torch.Tensor
    ) -> None:
        """Copy a remote request's KV from host_pool to remote_buffer.

        Simulates "KV stored on remote node" -- after prefill, the decode
        node would not have this data locally.
        """
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
            # Use real RDMA READ through NIC loopback
            rdma_us = self._rdma_ctx.bulk_read(valid_cpu, item_size)
        else:
            # Simulation mode: measure memcpy time as approximation
            t0 = time.monotonic()
            for layer_id in range(self.layer_num):
                self.staging_buffer[layer_id, valid_cpu] = self.remote_buffer[
                    layer_id, valid_cpu
                ]
            rdma_us = (time.monotonic() - t0) * 1e6

        # Copy staging -> host_pool (simulates "RDMA READ completed, data in local")
        for layer_id in range(self.layer_num):
            self.kv_buffer[layer_id, valid_cpu] = self.staging_buffer[
                layer_id, valid_cpu
            ]

        self._rdma_stats["total_prefetch_us"] += rdma_us
        self._rdma_stats["prefetch_count"] += 1

        logger.debug(
            "RDMA prefetch: req_pool_idx=%d, tokens=%d, %.1f us (%.2f MB)",
            req_pool_idx,
            num_tokens,
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
