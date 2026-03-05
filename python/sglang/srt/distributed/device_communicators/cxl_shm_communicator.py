"""CXL shared-memory all-reduce communicator.

This module implements TP-style all-reduce via a CXL shared memory window that
is accessible to all participating ranks.

Two operating modes are supported:

**Flat mode** (``local_group=None``, original behaviour):
  Every rank participates directly in every CXL operation.  The CXL window
  holds one slot per rank and the barrier spans all ``world_size`` ranks.

**Hierarchical mode** (``local_group`` provided):
  The all-reduce is split into three stages:

  1. *Intra-node reduce* – every rank on the same physical node performs a
     GPU-native all-reduce via ``local_group`` (e.g. NCCL).  After this step
     every local GPU holds the intra-node partial sum.

  2. *Inter-node reduce via CXL* – only the "leader" rank on each node
     (``local_rank == 0``) participates in the CXL exchange.  The CXL window
     holds one slot per *node* (``num_nodes`` slots), and the barrier spans
     only the ``num_nodes`` leaders.  Inside this stage the same 1-stage or
     2-stage (reduce-scatter + all-gather) strategy is applied, but among
     nodes rather than individual GPUs.

  3. *Intra-node broadcast* – the leader broadcasts the globally reduced
     tensor to the other ranks on the same node via ``local_group``.

Cache coherency note
--------------------
The CXL switch does **not** provide cross-node cache coherency.  All required
cache-line discipline is already embedded in the C++ helpers:

* ``dram2cxl`` / ``vram2cxl``: non-temporal stores (``nt_store_copy``) +
  ``_mm_sfence`` → writes bypass CPU caches and are globally visible once
  the sfence completes.
* ``cxl2dram`` / ``cxl2vram``: ``clflush_range`` before the copy → reads
  always see the latest data from the CXL device rather than a stale cache
  line.
* ``cxl_barrier_tp``: ``_mm_sfence`` before writing the token, poll loop
  uses ``clflush_range`` on the control block, ``_mm_lfence`` after all
  tokens are confirmed → ensures a full memory ordering fence across the
  write-barrier-read pipeline.

Python-level ordering therefore only needs to guarantee:
  data write (tensor_to_cxl) → _barrier_nodes() → data read (cxl_to_tensor)
which is exactly what each stage enforces.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import is_cuda


logger = logging.getLogger(__name__)


def _align_up(val: int, align: int) -> int:
    return ((val + align - 1) // align) * align


def _load_cxl_extension() -> object:
    """Load or build the cxl_mem_ext C++ extension."""
    src_root = Path("/root/mry/sglang/cxl_shm_test/pytorch_cxl_test")
    if not src_root.exists():
        raise ImportError(
            "cxl_mem_ext is not installed and sources were not found at "
            f"{src_root}. Set PYTHONPATH to an existing build or install the "
            "extension manually."
        )

    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:
        raise ImportError("torch.utils.cpp_extension.load is unavailable") from exc

    build_dir = src_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    return load(
        name="cxl_mem_ext",
        sources=[str(src_root / "cxl_mem.cpp"), str(src_root / "cxl_mem_pybind.cpp")],
        extra_cflags=["-O3", "-std=c++17", "-mclflushopt", "-msse4.1"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        build_directory=str(build_dir),
        verbose=False,
        with_cuda=True,
    )


class CxlShmCommunicator:
    """All-reduce over a shared CXL window.

    Parameters
    ----------
    group:
        The global process group covering all TP ranks.
    device:
        The local CUDA (or CPU) device for this rank.
    local_group:
        Optional intra-node process group.  When provided the communicator
        uses hierarchical all-reduce (NCCL intra-node + CXL inter-node).
        When ``None`` every rank participates in the CXL phase directly
        (original flat behaviour).
    dev_path / map_bytes / map_offset / control_bytes / barrier_sleep:
        Low-level CXL device parameters.
    """

    _ALIGN = 256

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        local_group: Optional[ProcessGroup] = None,
        dev_path: Optional[str] = None,
        map_bytes: Optional[int] = None,
        map_offset: Optional[int] = None,
        control_bytes: Optional[int] = None,
        barrier_sleep: Optional[float] = None,
    ):
        self.disabled = True
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.device = torch.device(device)

        # ------------------------------------------------------------------
        # Topology: derive node-level view from local_group (if any).
        # ------------------------------------------------------------------
        self.local_group = local_group
        if local_group is not None:
            self.local_rank = dist.get_rank(local_group)
            self.local_size = dist.get_world_size(local_group)
        else:
            # No hierarchy: every rank is its own "node".
            self.local_rank = 0
            self.local_size = 1

        if self.world_size % self.local_size != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by "
                f"local_size ({self.local_size})"
            )

        self.num_nodes = self.world_size // self.local_size

        # Assumes ranks are laid out contiguously per node:
        #   node 0 → global ranks [0 .. local_size-1]
        #   node 1 → global ranks [local_size .. 2*local_size-1]
        #   ...
        self.node_rank = self.rank // self.local_size

        # Only local_rank-0 on each node participates in the CXL phase.
        self.is_cxl_leader = self.local_rank == 0

        # Global rank of the CXL leader on this node, used as broadcast src.
        self.local_leader_global_rank = self.rank - self.local_rank
        # ------------------------------------------------------------------

        if is_cuda():
            torch.cuda.set_device(self.device)

        try:
            self.ext = _load_cxl_extension()
        except Exception as exc:
            logger.warning("CXL extension unavailable: %s", exc)
            return

        self.dev_path = dev_path if dev_path is not None else "/dev/dax0.0"
        self.map_bytes = map_bytes if map_bytes is not None else 16 * 1024 * 1024 * 1024
        self.map_offset = map_offset if map_offset is not None else 0
        self.control_bytes = control_bytes if control_bytes is not None else 4096
        self.barrier_sleep = barrier_sleep if barrier_sleep is not None else 1e-8

        self.control_offset = 0
        self.data_offset = _align_up(self.control_bytes, self._ALIGN)
        # stage_token is a monotonically increasing counter used to
        # distinguish successive CXL barriers; it is only advanced by
        # _barrier_nodes() and therefore only needs to be consistent among
        # the leader ranks that call that function.
        self.stage_token = 1

        ok = self.ext.cxl_init(
            dev_path=self.dev_path,
            map_bytes=self.map_bytes,
            offset=self.map_offset,
            register_cuda=bool(is_cuda()),
            gpu_id=self.device.index,
        )
        if not ok:
            logger.warning("cxl_init failed for %s", self.dev_path)
            return

        ctrl_init = torch.zeros(self.control_bytes, dtype=torch.int8, device="cpu")
        self.ext.tensor_to_cxl(ctrl_init, offset=self.control_offset)

        self._buffer_cache: Dict[Tuple[int, int, torch.dtype], torch.Tensor] = {}
        self._reduced_cache: Dict[Tuple[int, int, torch.dtype], torch.Tensor] = {}
        self.disabled = False
        self.all_reduce_num = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            dist.all_reduce(inp, group=self.group)
            return inp

        flat_inp = inp.contiguous()
        total_bytes = flat_inp.numel() * flat_inp.element_size()

        # Single-node setup: intra-node all-reduce is sufficient; no CXL
        # traffic needed at all.
        if self.num_nodes == 1:
            if self.local_size > 1:
                dist.all_reduce(flat_inp, group=self.local_group)
            return flat_inp

        # Stage selection is based on the number of CXL participants
        # (num_nodes), not the total world_size, because in hierarchical
        # mode only one rank per node writes to / reads from the CXL window.
        cxl_n = self.num_nodes
        use_one_stage = (
            cxl_n == 2
            or (cxl_n == 4 and total_bytes < 512 * 1024)
            or (cxl_n == 8 and total_bytes < 256 * 1024)
        )

        if use_one_stage:
            return self.all_reduce_1_stage(flat_inp)
        return self.all_reduce_2_stage(flat_inp)

    # ------------------------------------------------------------------
    # 1-stage hierarchical all-reduce
    # ------------------------------------------------------------------

    def all_reduce_1_stage(self, inp: torch.Tensor) -> torch.Tensor:
        """Hierarchical 1-stage all-reduce.

        All ranks participate in Stage 1 (local) and Stage 3 (broadcast).
        Only leaders participate in Stage 2 (CXL).

        CXL memory layout (from data_offset)::

            [node-0 full tensor][node-1 full tensor]...[node-(N-1) full tensor]
             ^slot_bytes each
        """
        flat_inp = inp.contiguous()
        slot_bytes = flat_inp.numel() * flat_inp.element_size()
        required_bytes = self.data_offset + self.num_nodes * slot_bytes
        if required_bytes > self.map_bytes:
            logger.warning(
                "CXL window too small (need %d bytes, have %d); falling back",
                required_bytes,
                self.map_bytes,
            )
            dist.all_reduce(flat_inp, group=self.group)
            return flat_inp

        t0 = time.perf_counter()

        # ---- Stage 1: intra-node GPU-native all-reduce -------------------
        # After this every rank on the node holds the intra-node partial sum.
        if self.local_size > 1:
            dist.all_reduce(flat_inp, group=self.local_group)
        t_local = time.perf_counter()

        # ---- Stage 2: inter-node reduce via CXL (leaders only) -----------
        # The sfence inside nt_store_copy (called by tensor_to_cxl) together
        # with the sfence at the start of cxl_barrier_tp ensures the written
        # data is globally visible before any other node reads it.
        # The clflush_range inside cxl_to_tensor (cxl2dram/cxl2vram) and the
        # lfence at the end of cxl_barrier_tp ensure we always read fresh
        # data from the CXL device.
        if self.is_cxl_leader:
            self.ext.tensor_to_cxl(
                flat_inp, offset=self.data_offset + self.node_rank * slot_bytes
            )
            t_write = time.perf_counter()
            self._barrier_nodes()
            t_barrier = time.perf_counter()

            gather = torch.empty(
                (self.num_nodes, flat_inp.numel()),
                dtype=flat_inp.dtype,
                device=self.device,
            )
            gather = self.ext.cxl_to_tensor(
                gather, offset=self.data_offset
            ).view_as(gather)
            t_read = time.perf_counter()

            result = gather.sum(dim=0)
            t_cxl_reduce = time.perf_counter()
        else:
            result = torch.empty_like(flat_inp)
            t_write = t_barrier = t_read = t_cxl_reduce = t_local

        # ---- Stage 3: broadcast result from leader to local peers --------
        if self.local_size > 1:
            dist.broadcast(
                result,
                src=self.local_leader_global_rank,
                group=self.local_group,
            )
        t_bcast = time.perf_counter()

        logger.info(
            "[%d] Rank %d AR1-hier timing (us) %s: "
            "local_reduce=%.1f write=%.1f barrier=%.1f read=%.1f "
            "cxl_reduce=%.1f bcast=%.1f total=%.1f",
            self.all_reduce_num,
            self.rank,
            inp.shape,
            (t_local - t0) * 1e6,
            (t_write - t_local) * 1e6,
            (t_barrier - t_write) * 1e6,
            (t_read - t_barrier) * 1e6,
            (t_cxl_reduce - t_read) * 1e6,
            (t_bcast - t_cxl_reduce) * 1e6,
            (t_bcast - t0) * 1e6,
        )
        self.all_reduce_num += 1
        return result.view_as(inp)

    # ------------------------------------------------------------------
    # 2-stage hierarchical all-reduce (reduce-scatter + all-gather)
    # ------------------------------------------------------------------

    def all_reduce_2_stage(self, inp: torch.Tensor) -> torch.Tensor:
        """Hierarchical 2-stage (reduce-scatter + all-gather) all-reduce.

        All ranks participate in Stage 1 (local) and Stage 3 (broadcast).
        Only leaders participate in Stage 2 (CXL reduce-scatter + all-gather).

        CXL memory layout (from data_offset)::

            [node-0 full][node-1 full]...[node-(N-1) full]   ← input slots
            [node-0 shard][node-1 shard]...[node-(N-1) shard] ← reduced shards

        Each "shard" is 1/num_nodes of the full tensor, and node-k is
        responsible for reducing element indices [k*shard_h, (k+1)*shard_h).
        """
        flat_inp = inp.contiguous()
        total_ele_num = flat_inp.numel()

        if total_ele_num % self.num_nodes != 0:
            logger.warning(
                "Element count (%d) not divisible by num_nodes (%d); falling back",
                total_ele_num,
                self.num_nodes,
            )
            dist.all_reduce(flat_inp, group=self.group)
            return flat_inp

        shard_h = total_ele_num // self.num_nodes
        shard_bytes = shard_h * flat_inp.element_size()
        slot_bytes = flat_inp.numel() * flat_inp.element_size()
        reduced_base = self.data_offset + self.num_nodes * slot_bytes
        required_bytes = reduced_base + self.num_nodes * shard_bytes
        if required_bytes > self.map_bytes:
            logger.warning(
                "CXL window too small (need %d bytes, have %d); falling back",
                required_bytes,
                self.map_bytes,
            )
            dist.all_reduce(flat_inp, group=self.group)
            return flat_inp

        t0 = time.perf_counter()

        # ---- Stage 1: intra-node GPU-native all-reduce -------------------
        if self.local_size > 1:
            dist.all_reduce(flat_inp, group=self.local_group)
        t_local = time.perf_counter()

        # ---- Stage 2: inter-node CXL reduce-scatter + all-gather ---------
        if self.is_cxl_leader:
            # Reduce-scatter: each leader writes its full intra-node sum to
            # the CXL window, then reads the shard it is responsible for
            # from every other node's slot and accumulates them.
            self.ext.tensor_to_cxl(
                flat_inp, offset=self.data_offset + self.node_rank * slot_bytes
            )
            t_write = time.perf_counter()

            # Barrier 1: all leaders have written their full tensors.
            # The barrier's internal sfence-on-write + lfence-on-exit
            # guarantees that each leader sees all other leaders' freshly
            # written data after returning from this call.
            self._barrier_nodes()
            t_barrier1 = time.perf_counter()

            # Read the portion of this node's shard from every other node.
            shard_offset = self.node_rank * shard_bytes
            gather = torch.empty(
                (self.num_nodes, shard_h), dtype=flat_inp.dtype, device=self.device
            )
            for src in range(self.num_nodes):
                gather[src] = self.ext.cxl_to_tensor(
                    gather[src],
                    offset=self.data_offset + src * slot_bytes + shard_offset,
                )
            t_read_shard = time.perf_counter()

            reduced_shard = gather.sum(dim=0)
            t_cxl_reduce = time.perf_counter()

            # All-gather: write the reduced shard back to CXL so every
            # leader can assemble the full result.
            self.ext.tensor_to_cxl(
                reduced_shard,
                offset=reduced_base + self.node_rank * shard_bytes,
            )
            t_write_red = time.perf_counter()

            # Barrier 2: all leaders have written their reduced shards.
            self._barrier_nodes()
            t_barrier2 = time.perf_counter()

            reduce_res = torch.empty(
                (total_ele_num,), dtype=flat_inp.dtype, device=self.device
            )
            reduce_res = self.ext.cxl_to_tensor(reduce_res, offset=reduced_base)
            t_read_full = time.perf_counter()
        else:
            reduce_res = torch.empty_like(flat_inp)
            t_write = t_barrier1 = t_read_shard = t_cxl_reduce = (
                t_write_red
            ) = t_barrier2 = t_read_full = t_local

        # ---- Stage 3: broadcast result from leader to local peers --------
        if self.local_size > 1:
            dist.broadcast(
                reduce_res,
                src=self.local_leader_global_rank,
                group=self.local_group,
            )
        t_bcast = time.perf_counter()

        logger.info(
            "[%d] Rank %d AR2-hier timing (us) %s: "
            "local_reduce=%.1f write=%.1f barrier1=%.1f read_shard=%.1f "
            "cxl_reduce=%.1f write_red=%.1f barrier2=%.1f read_full=%.1f "
            "bcast=%.1f total=%.1f",
            self.all_reduce_num,
            self.rank,
            inp.shape,
            (t_local - t0) * 1e6,
            (t_write - t_local) * 1e6,
            (t_barrier1 - t_write) * 1e6,
            (t_read_shard - t_barrier1) * 1e6,
            (t_cxl_reduce - t_read_shard) * 1e6,
            (t_write_red - t_cxl_reduce) * 1e6,
            (t_barrier2 - t_write_red) * 1e6,
            (t_read_full - t_barrier2) * 1e6,
            (t_bcast - t_read_full) * 1e6,
            (t_bcast - t0) * 1e6,
        )
        self.all_reduce_num += 1
        return reduce_res.view_as(inp)

    # ------------------------------------------------------------------
    # Barrier helpers
    # ------------------------------------------------------------------

    def _barrier_nodes(self):
        """CXL barrier spanning only the ``num_nodes`` leader ranks.

        Uses ``node_rank`` (0 … num_nodes-1) as the rank identifier and
        ``num_nodes`` as the participant count.  Non-leader ranks must never
        call this function.

        The barrier occupies the first ``num_nodes * kCacheLine`` bytes of
        the control block, which is always ≤ control_bytes (default 4096 B
        supports up to 64 nodes given kCacheLine=64).
        """
        token = self.stage_token
        self.stage_token += 1
        self.ext.cxl_barrier_tp(
            token,
            control_offset=self.control_offset,
            rank=self.node_rank,
            num_ranks=self.num_nodes,
        )
