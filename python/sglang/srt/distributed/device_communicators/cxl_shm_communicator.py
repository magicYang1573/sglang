"""CXL shared-memory all-reduce communicator.

This module mirrors the CXL helper in mry/cxl_shm_test/pytorch_cxl_test and
implements a simple TP-style all-reduce:

1) Each rank writes its input shards into a shared CXL pool.
2) Every rank pulls the shard assigned to it from all peers and sums locally.
3) Reduced shards are written back to the pool.
4) Ranks gather the reduced shards to materialize the full tensor.

Synchronization across ranks is handled via a small control block placed at
the front of the mapped CXL window. The control block stores per-rank stage
tokens that are polled to form barriers between pipeline stages.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import is_cuda


logger = logging.getLogger(__name__)


def _align_up(val: int, align: int) -> int:
    return ((val + align - 1) // align) * align


def _load_cxl_extension() -> object:
    """Load or build the cxl_mem_ext extension."""

    src_root = Path("/root/mry/sglang/cxl_shm_test/pytorch_cxl_test")
    if not src_root.exists():
        raise ImportError(
            "cxl_mem_ext is not installed and sources were not found at "
            f"{src_root}. Set PYTHONPATH to an existing build or install the "
            "extension manually."
        )

    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:  # pragma: no cover
        raise ImportError("torch.utils.cpp_extension.load is unavailable") from exc

    build_dir = src_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    # -mavx2 enables 256-bit streaming stores in nt_store_copy (4x vs scalar).
    # -mavx512f adds 512-bit streaming stores on Sapphire Rapids and newer.
    avx_flags = ["-mavx2"]
    try:
        import subprocess
        result = subprocess.run(
            ["grep", "-m1", "avx512f", "/proc/cpuinfo"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and "avx512f" in result.stdout:
            avx_flags += ["-mavx512f", "-mavx512bw"]
    except Exception:
        pass

    return load(
        name="cxl_mem_ext",
        sources=[str(src_root / "cxl_mem.cpp"), str(src_root / "cxl_mem_pybind.cpp")],
        extra_cflags=["-O3", "-std=c++17", "-mclflushopt", "-msse4.1"] + avx_flags,
        extra_cuda_cflags=["-O3", "-std=c++17"],
        build_directory=str(build_dir),
        verbose=False,
        with_cuda=True,
    )


class CxlShmCommunicator:
    """All-reduce over a shared CXL window."""

    _ALIGN = 256

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        dev_path: str | None = None,
        map_bytes: int | None = None,
        map_offset: int | None = None,
        control_bytes: int | None = None,
        barrier_sleep: float | None = None,
    ):
        self.disabled = True
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.device = torch.device(device)

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

        # Pre-allocated tensor cache keyed by (shape_tuple, dtype).
        # Avoids torch.empty() on every all_reduce call.
        self._buf: Dict[Tuple, torch.Tensor] = {}
        self.disabled = False
        self.all_reduce_num = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_buf(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Return a cached GPU tensor for the given shape and dtype."""
        key = (shape, dtype)
        if key not in self._buf:
            self._buf[key] = torch.empty(shape, dtype=dtype, device=self.device)
        return self._buf[key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            dist.all_reduce(inp, group=self.group)
            return inp

        flat_inp = inp.contiguous()
        total_bytes = flat_inp.numel() * flat_inp.element_size()

        # Same heuristic as custom_all_reduce: use 1-stage (all-gather + sum)
        # for small tensors or 2-rank groups; 2-stage (reduce-scatter + all-
        # gather) for large tensors with more ranks.
        use_one_stage = False
        if self.world_size == 2:
            use_one_stage = True
        elif self.world_size == 4 and total_bytes < 512 * 1024:
            use_one_stage = True
        elif self.world_size == 8 and total_bytes < 256 * 1024:
            use_one_stage = True

        if use_one_stage:
            return self.all_reduce_1_stage(flat_inp)
        return self.all_reduce_2_stage(flat_inp)

    def all_reduce_1_stage(self, inp: torch.Tensor) -> torch.Tensor:
        flat_inp = inp.contiguous()
        slot_bytes = flat_inp.numel() * flat_inp.element_size()
        required_bytes = self.data_offset + self.world_size * slot_bytes
        if required_bytes > self.map_bytes:
            logger.warning(
                "CXL window too small (need %d bytes, have %d); falling back",
                required_bytes,
                self.map_bytes,
            )
            dist.all_reduce(flat_inp, group=self.group)
            return flat_inp

        t0 = time.perf_counter()

        # Write this rank's full tensor to its slot in the CXL window.
        self.ext.tensor_to_cxl(
            flat_inp, offset=self.data_offset + self.rank * slot_bytes
        )
        t_write = time.perf_counter()

        self._barrier()
        t_barrier1 = time.perf_counter()

        # Read all slots in one contiguous CXL read (avoids N separate calls).
        gather = self._get_buf((self.world_size, flat_inp.numel()), flat_inp.dtype)
        self.ext.cxl_to_tensor(gather, offset=self.data_offset)
        t_read = time.perf_counter()

        reduced = gather.sum(dim=0)
        t_reduce = time.perf_counter()

        total = t_reduce - t0
        other = total - (
            (t_write - t0)
            + (t_barrier1 - t_write)
            + (t_read - t_barrier1)
            + (t_reduce - t_read)
        )
        logger.info(
            "[%d] Rank %d AR1 timing (us) %s: write=%.1f barrier1=%.1f "
            "read=%.1f reduce=%.1f other=%.1f total=%.1f",
            self.all_reduce_num,
            self.rank,
            inp.shape,
            (t_write - t0) * 1e6,
            (t_barrier1 - t_write) * 1e6,
            (t_read - t_barrier1) * 1e6,
            (t_reduce - t_read) * 1e6,
            other * 1e6,
            total * 1e6,
        )

        self.all_reduce_num += 1
        return reduced.view_as(inp)

    def all_reduce_2_stage(self, inp: torch.Tensor) -> torch.Tensor:
        flat_inp = inp.contiguous()
        total_ele_num = flat_inp.numel()
        if total_ele_num % self.world_size != 0:
            logger.warning(
                "numel (%d) must be divisible by world_size (%d); falling back",
                total_ele_num,
                self.world_size,
            )
            dist.all_reduce(flat_inp, group=dist.group.WORLD)
            return flat_inp

        shard_h = total_ele_num // self.world_size
        shard_bytes = shard_h * flat_inp.element_size()
        slot_bytes = flat_inp.numel() * flat_inp.element_size()
        reduced_base = self.data_offset + self.world_size * slot_bytes
        required_bytes = reduced_base + self.world_size * shard_bytes
        if required_bytes > self.map_bytes:
            logger.warning(
                "CXL window too small (need %d bytes, have %d); falling back",
                required_bytes,
                self.map_bytes,
            )
            dist.all_reduce(flat_inp, group=dist.group.WORLD)
            return flat_inp

        t0 = time.perf_counter()

        # Stage 1: every rank writes its full tensor to its CXL slot.
        self.ext.tensor_to_cxl(
            flat_inp, offset=self.data_offset + self.rank * slot_bytes
        )
        t_write = time.perf_counter()
        self._barrier()
        t_barrier1 = time.perf_counter()

        # Stage 2: each rank reads its responsible shard from every peer.
        # Use cxl_to_tensor_shards for parallel multi-threaded CXL reads
        # (one thread per peer) followed by a single H2D DMA transfer.
        shard_offset = self.rank * shard_bytes
        offsets = [
            self.data_offset + src * slot_bytes + shard_offset
            for src in range(self.world_size)
        ]
        gather = self._get_buf((self.world_size, shard_h), flat_inp.dtype)
        self.ext.cxl_to_tensor_shards(gather, offsets, shard_bytes)
        t_read = time.perf_counter()

        reduced_shard = gather.sum(dim=0)
        t_reduce = time.perf_counter()

        # Stage 3: write the locally-reduced shard back to CXL.
        self.ext.tensor_to_cxl(
            reduced_shard, offset=reduced_base + self.rank * shard_bytes
        )
        t_write_reduce = time.perf_counter()
        self._barrier()
        t_barrier2 = time.perf_counter()

        # Stage 4: read all reduced shards in one contiguous CXL read.
        reduce_res = self._get_buf((total_ele_num,), flat_inp.dtype)
        self.ext.cxl_to_tensor(reduce_res, offset=reduced_base)
        t_read_reduce = time.perf_counter()

        total = t_read_reduce - t0
        other = total - (
            (t_write - t0)
            + (t_barrier1 - t_write)
            + (t_read - t_barrier1)
            + (t_reduce - t_read)
            + (t_write_reduce - t_reduce)
            + (t_barrier2 - t_write_reduce)
            + (t_read_reduce - t_barrier2)
        )
        logger.info(
            "Rank %d AR2 timing (us): write=%.1f barrier1=%.1f read_shard=%.1f "
            "reduce=%.1f write_red=%.1f barrier2=%.1f read_red=%.1f "
            "other=%.1f total=%.1f",
            self.rank,
            (t_write - t0) * 1e6,
            (t_barrier1 - t_write) * 1e6,
            (t_read - t_barrier1) * 1e6,
            (t_reduce - t_read) * 1e6,
            (t_write_reduce - t_reduce) * 1e6,
            (t_barrier2 - t_write_reduce) * 1e6,
            (t_read_reduce - t_barrier2) * 1e6,
            other * 1e6,
            total * 1e6,
        )

        self.all_reduce_num += 1
        return reduce_res.view_as(inp)

    def _barrier(self):
        token = self.stage_token
        self.stage_token += 1
        self.ext.cxl_barrier_tp(
            token,
            control_offset=self.control_offset,
            rank=self.rank,
            num_ranks=self.world_size,
        )
