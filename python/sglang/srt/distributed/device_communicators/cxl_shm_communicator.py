"""CXL shared-memory all-reduce communicator.

All-reduce flow (GPU-reduce path):
  1-stage:
    C++: GPU→CXL write (own slot) → barrier → bulk CXL→GPU gather (all slots)
    Python: GPU tensor.view(world_size, -1).sum(dim=0)

  2-stage (reduce-scatter + all-gather):
    C++: GPU→CXL write (full tensor) → barrier → cudaMemcpy2D strided gather
         (my shard from every peer → contiguous GPU staging buffer)
    Python: GPU staging.view(world_size, -1).sum(dim=0)
    C++: GPU→CXL write (reduced shard) → barrier → bulk CXL→GPU allgather
    Result sits in a pre-allocated GPU buffer.

The reduction is entirely on the GPU; the CPU only handles clflush and the
barrier spin-loop. Pre-allocated staging tensors avoid per-call allocations.
Synchronization is enforced by the CXL barrier (clflush + _mm_sfence/_mm_lfence)
in the C++ layer.
"""

import logging
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
            "cxl_mem_ext sources not found at "
            f"{src_root}. Install the extension or set PYTHONPATH."
        )

    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:
        raise ImportError("torch.utils.cpp_extension.load unavailable") from exc

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
    """All-reduce over a shared CXL window with GPU-side reduction.

    The C++ layer is responsible only for data movement (GPU↔CXL) and the
    cache-coherence protocol (clflush + CXL barrier).  All arithmetic
    reduction happens via GPU kernels, which avoids the dtype-conversion and
    precision loss that CPU-side fp32 accumulation would introduce for
    bfloat16/float16 workloads, and leverages GPU memory bandwidth for the
    summation.

    Staging buffers are pre-allocated on the GPU and reused across calls to
    avoid per-call torch.empty overhead.
    """

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

        self.dev_path = dev_path or "/dev/dax0.0"
        self.map_bytes = map_bytes or 16 * 1024 * 1024 * 1024
        self.map_offset = map_offset or 0
        self.control_bytes = control_bytes or 4096
        self.barrier_sleep = barrier_sleep or 1e-8

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

        self.disabled = False
        self.all_reduce_num = 0

        # key: (numel, dtype) → pre-allocated GPU staging tensor
        # 1-stage: shape (world_size * numel,)
        self._gather_cache: Dict[Tuple[int, torch.dtype], torch.Tensor] = {}
        # 2-stage scatter: shape (world_size * shard_h,)
        self._shard_cache: Dict[Tuple[int, torch.dtype], torch.Tensor] = {}
        # 2-stage result: shape (numel,)
        self._result_cache: Dict[Tuple[int, torch.dtype], torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            dist.all_reduce(inp, group=self.group)
            return inp

        flat_inp = inp.contiguous()
        total_bytes = flat_inp.numel() * flat_inp.element_size()

        use_one_stage = False
        if self.world_size == 2:
            use_one_stage = True
        elif self.world_size == 4 and total_bytes < 512 * 1024:
            use_one_stage = True
        elif self.world_size == 8 and total_bytes < 256 * 1024:
            use_one_stage = True

        if use_one_stage:
            return self._all_reduce_1_stage(flat_inp).view_as(inp)
        return self._all_reduce_2_stage(flat_inp).view_as(inp)

    # ------------------------------------------------------------------
    # 1-stage all-reduce
    # ------------------------------------------------------------------

    def _all_reduce_1_stage(self, inp: torch.Tensor) -> torch.Tensor:
        slot_bytes = inp.numel() * inp.element_size()
        required_bytes = self.data_offset + self.world_size * slot_bytes
        if required_bytes > self.map_bytes:
            logger.warning(
                "CXL window too small (need %d, have %d); falling back",
                required_bytes, self.map_bytes,
            )
            dist.all_reduce(inp, group=self.group)
            return inp

        if inp.is_cuda:
            return self._fused_1stage(inp, slot_bytes)
        return self._legacy_1stage(inp, slot_bytes)

    def _fused_1stage(self, inp: torch.Tensor, slot_bytes: int) -> torch.Tensor:
        token = self.stage_token
        self.stage_token += 1

        # Pre-allocate / reuse GPU gather staging: (world_size * numel,)
        key = (inp.numel(), inp.dtype)
        if key not in self._gather_cache:
            self._gather_cache[key] = torch.empty(
                self.world_size * inp.numel(), dtype=inp.dtype, device=self.device
            )
        gather_buf = self._gather_cache[key]

        t0 = time.perf_counter()

        # C++: GPU→CXL write → barrier → bulk CXL→GPU gather (all slots)
        self.ext.cxl_write_barrier_gather_1stage(
            inp,
            gather_buf,
            slot_bytes,
            self.data_offset,
            self.control_offset,
            self.rank,
            self.world_size,
            token,
        )
        t_transfer = time.perf_counter()

        # GPU reduce
        result = gather_buf.view(self.world_size, inp.numel()).sum(dim=0)
        t_reduce = time.perf_counter()

        logger.info(
            "[%d] Rank %d AR1 (us) %s: transfer=%.1f gpu_reduce=%.1f total=%.1f",
            self.all_reduce_num, self.rank, inp.shape,
            (t_transfer - t0) * 1e6,
            (t_reduce - t_transfer) * 1e6,
            (t_reduce - t0) * 1e6,
        )
        self.all_reduce_num += 1
        return result

    def _legacy_1stage(self, inp: torch.Tensor, slot_bytes: int) -> torch.Tensor:
        """CPU-tensor fallback (unchanged behaviour)."""
        t0 = time.perf_counter()
        self.ext.tensor_to_cxl(
            inp, offset=self.data_offset + self.rank * slot_bytes
        )
        t_write = time.perf_counter()
        self._barrier()
        t_b = time.perf_counter()

        gather = torch.empty(
            (self.world_size, inp.numel()), dtype=inp.dtype, device=self.device
        )
        gather = self.ext.cxl_to_tensor(gather, offset=self.data_offset).view_as(gather)
        t_read = time.perf_counter()

        reduced = gather.sum(dim=0)
        t_reduce = time.perf_counter()

        logger.info(
            "[%d] Rank %d AR1-legacy (us) %s: write=%.1f barrier=%.1f read=%.1f reduce=%.1f total=%.1f",
            self.all_reduce_num, self.rank, inp.shape,
            (t_write - t0) * 1e6, (t_b - t_write) * 1e6,
            (t_read - t_b) * 1e6, (t_reduce - t_read) * 1e6,
            (t_reduce - t0) * 1e6,
        )
        self.all_reduce_num += 1
        return reduced

    # ------------------------------------------------------------------
    # 2-stage all-reduce (reduce-scatter + all-gather)
    # ------------------------------------------------------------------

    def _all_reduce_2_stage(self, inp: torch.Tensor) -> torch.Tensor:
        total_ele_num = inp.numel()
        if total_ele_num % self.world_size != 0:
            logger.warning(
                "numel %d not divisible by world_size %d; falling back",
                total_ele_num, self.world_size,
            )
            dist.all_reduce(inp, group=dist.group.WORLD)
            return inp

        slot_bytes = total_ele_num * inp.element_size()
        shard_bytes = slot_bytes // self.world_size
        reduced_base = self.data_offset + self.world_size * slot_bytes
        required_bytes = reduced_base + self.world_size * shard_bytes
        if required_bytes > self.map_bytes:
            logger.warning(
                "CXL window too small (need %d, have %d); falling back",
                required_bytes, self.map_bytes,
            )
            dist.all_reduce(inp, group=dist.group.WORLD)
            return inp

        if inp.is_cuda:
            return self._fused_2stage(inp, slot_bytes, shard_bytes, reduced_base)
        return self._legacy_2stage(inp, slot_bytes, shard_bytes, reduced_base)

    def _fused_2stage(
        self,
        inp: torch.Tensor,
        slot_bytes: int,
        shard_bytes: int,
        reduced_base: int,
    ) -> torch.Tensor:
        token_start = self.stage_token
        self.stage_token += 2

        total_ele_num = inp.numel()
        shard_h = total_ele_num // self.world_size

        key = (total_ele_num, inp.dtype)
        if key not in self._shard_cache:
            self._shard_cache[key] = torch.empty(
                self.world_size * shard_h, dtype=inp.dtype, device=self.device
            )
        if key not in self._result_cache:
            self._result_cache[key] = torch.empty(
                total_ele_num, dtype=inp.dtype, device=self.device
            )
        shard_buf = self._shard_cache[key]
        result_buf = self._result_cache[key]

        t0 = time.perf_counter()

        # --- Stage 1: C++ scatter phase ---
        # GPU→CXL (full tensor) → barrier → cudaMemcpy2D (my shard from all peers)
        self.ext.cxl_write_barrier_scatter_2stage(
            inp,
            shard_buf,
            slot_bytes,
            shard_bytes,
            self.data_offset,
            self.control_offset,
            self.rank,
            self.world_size,
            token_start,
        )
        t_scatter = time.perf_counter()

        # GPU reduce: sum across world_size copies of my shard
        reduced_shard = shard_buf.view(self.world_size, shard_h).sum(dim=0)
        t_reduce = time.perf_counter()

        # --- Stage 2: C++ gather phase ---
        # GPU→CXL (reduced shard) → barrier → bulk CXL→GPU (all reduced shards)
        self.ext.cxl_write_barrier_allgather_2stage(
            reduced_shard,
            result_buf,
            shard_bytes,
            slot_bytes,            # total_bytes == slot_bytes for the result region
            reduced_base,
            self.control_offset,
            self.rank,
            self.world_size,
            token_start + 1,
        )
        t_gather = time.perf_counter()

        logger.info(
            "[%d] Rank %d AR2 (us) %s: scatter=%.1f gpu_reduce=%.1f gather=%.1f total=%.1f",
            self.all_reduce_num, self.rank, inp.shape,
            (t_scatter - t0) * 1e6,
            (t_reduce - t_scatter) * 1e6,
            (t_gather - t_reduce) * 1e6,
            (t_gather - t0) * 1e6,
        )
        self.all_reduce_num += 1
        return result_buf.view_as(inp)

    def _legacy_2stage(
        self,
        inp: torch.Tensor,
        slot_bytes: int,
        shard_bytes: int,
        reduced_base: int,
    ) -> torch.Tensor:
        """CPU-tensor fallback (unchanged behaviour)."""
        total_ele_num = inp.numel()
        shard_h = total_ele_num // self.world_size

        t0 = time.perf_counter()
        self.ext.tensor_to_cxl(inp, offset=self.data_offset + self.rank * slot_bytes)
        t_w = time.perf_counter()
        self._barrier()
        t_b1 = time.perf_counter()

        gather = torch.empty((self.world_size, shard_h), dtype=inp.dtype, device=self.device)
        for src in range(self.world_size):
            gather[src] = self.ext.cxl_to_tensor(
                gather[src],
                offset=self.data_offset + src * slot_bytes + self.rank * shard_bytes,
            )
        t_r = time.perf_counter()

        reduced_shard = gather.sum(dim=0)
        t_red = time.perf_counter()

        self.ext.tensor_to_cxl(reduced_shard, offset=reduced_base + self.rank * shard_bytes)
        t_w2 = time.perf_counter()
        self._barrier()
        t_b2 = time.perf_counter()

        result = torch.empty(total_ele_num, dtype=inp.dtype, device=self.device)
        result = self.ext.cxl_to_tensor(result, offset=reduced_base)
        t_end = time.perf_counter()

        logger.info(
            "Rank %d AR2-legacy (us): w=%.1f b1=%.1f r=%.1f red=%.1f w2=%.1f b2=%.1f rr=%.1f total=%.1f",
            self.rank,
            (t_w - t0) * 1e6, (t_b1 - t_w) * 1e6, (t_r - t_b1) * 1e6,
            (t_red - t_r) * 1e6, (t_w2 - t_red) * 1e6, (t_b2 - t_w2) * 1e6,
            (t_end - t_b2) * 1e6, (t_end - t0) * 1e6,
        )
        self.all_reduce_num += 1
        return result.view_as(inp)

    def _barrier(self):
        token = self.stage_token
        self.stage_token += 1
        self.ext.cxl_barrier_tp(
            token,
            control_offset=self.control_offset,
            rank=self.rank,
            num_ranks=self.world_size,
        )
