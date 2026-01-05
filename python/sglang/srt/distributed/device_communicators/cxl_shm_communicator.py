import ctypes
import logging
import math
import mmap
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


class CxlShmCommunicator:
    """
    A minimal shared-memory all-reduce over a CXL-backed /dev/dax mapping.

    The communicator keeps a per-rank slot inside a single mmap region:
    [control_bytes][slot0][slot1]...[slotN-1]

    Algorithm (host-side reduction):
    1) Every rank copies its tensor bytes into its slot and flushes.
    2) Barrier.
    3) Rank 0 sums all slots into slot0.
    4) Barrier.
    5) All ranks read slot0 and copy back to their tensor (CPU or staged GPU).
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        dax_path: Optional[str] = None,
        map_bytes: Optional[int] = None,
        slot_bytes: Optional[int] = None,
        ctrl_bytes: Optional[int] = None,
        offset_bytes: int = 0,
        barrier_timeout_s: Optional[float] = None,
    ):
        self.group = group
        self.device = device
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.disabled = False

        if self.world_size <= 1:
            self.disabled = True
            return

        dax_path = dax_path or os.environ.get("SGLANG_CXL_DAX_PATH", "/dev/dax0.0")
        map_bytes = map_bytes or int(
            os.environ.get("SGLANG_CXL_MAP_BYTES", str(512 * 1024 * 1024))
        )
        slot_bytes = slot_bytes or int(
            os.environ.get("SGLANG_CXL_SLOT_BYTES", str(64 * 1024 * 1024))
        )
        ctrl_bytes = ctrl_bytes or int(
            os.environ.get("SGLANG_CXL_CTRL_BYTES", str(4096 * 4))
        )
        self.slot_bytes = slot_bytes
        self.ctrl_bytes = ctrl_bytes
        self.barrier_timeout_s = barrier_timeout_s

        required = ctrl_bytes + slot_bytes * self.world_size
        if map_bytes < required:
            raise ValueError(
                f"map_bytes={map_bytes} is smaller than required {required} "
                f"(ctrl={ctrl_bytes}, slot={slot_bytes}, world={self.world_size})"
            )

        fd = os.open(dax_path, os.O_RDWR | os.O_SYNC)
        try:
            self._mm = mmap.mmap(
                fd,
                length=map_bytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset_bytes,
            )
        except Exception:
            os.close(fd)
            raise
        self._fd = fd
        self._map_bytes = map_bytes

        # libc.msync for best-effort cache flush (alignment handled internally).
        self._libc = ctypes.CDLL(None)
        self._msync = getattr(self._libc, "msync", None)
        self._pagesize = os.sysconf("SC_PAGESIZE")

        logger.info(
            "[CXL SHM] mmap path=%s size=%s slot_bytes=%s ctrl_bytes=%s world=%s rank=%s",
            dax_path,
            map_bytes,
            slot_bytes,
            ctrl_bytes,
            self.world_size,
            self.rank,
        )

    def __del__(self):
        try:
            if hasattr(self, "_mm"):
                self._mm.close()
        finally:
            if hasattr(self, "_fd"):
                os.close(self._fd)

    def _flush_range(self, start: int, length: int):
        if self._msync is None:
            return
        # Align to page boundaries for msync.
        page_start = (start // self._pagesize) * self._pagesize
        page_end = math.ceil((start + length) / self._pagesize) * self._pagesize
        self._msync(
            ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(self._mm, page_start))),
            ctypes.c_size_t(page_end - page_start),
            ctypes.c_int(0x4),  # MS_SYNC
        )

    def _get_slot_bounds(self, rank: int, num_bytes: int):
        start = self.ctrl_bytes + rank * self.slot_bytes
        end = start + num_bytes
        if end > self._map_bytes:
            raise ValueError("slot write exceeds mmap size")
        return start, end

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            return tensor
        if tensor.numel() == 0:
            return tensor

        # Move to host if needed.
        if tensor.is_cpu:
            host_tensor = tensor.contiguous()
        else:
            host_tensor = tensor.detach().to("cpu", non_blocking=False).contiguous()

        flat = host_tensor.view(-1)
        num_bytes = flat.numel() * flat.element_size()
        if num_bytes > self.slot_bytes:
            raise ValueError(
                f"Tensor bytes {num_bytes} exceed CXL slot_bytes={self.slot_bytes}"
            )

        slot_start, slot_end = self._get_slot_bounds(self.rank, num_bytes)
        mv = memoryview(self._mm)
        mv[slot_start:slot_end] = flat.view(torch.uint16).numpy().tobytes()
        self._flush_range(slot_start, num_bytes)

        dist.barrier(self.group, timeout=self.barrier_timeout_s)

        if self.rank == 0:
            agg = torch.zeros_like(flat)
            for r in range(self.world_size):
                r_start, r_end = self._get_slot_bounds(r, num_bytes)
                buf = mv[r_start:r_end]
                tmp = torch.frombuffer(buf, dtype=flat.dtype, count=flat.numel())
                agg += tmp
            # Write result into slot0 for everyone to read.
            mv[self.ctrl_bytes : self.ctrl_bytes + num_bytes] = agg.numpy().tobytes()
            self._flush_range(self.ctrl_bytes, num_bytes)

        dist.barrier(self.group, timeout=self.barrier_timeout_s)

        # Read slot0 result.
        res_buf = mv[self.ctrl_bytes : self.ctrl_bytes + num_bytes]
        res_flat = torch.frombuffer(res_buf, dtype=flat.dtype, count=flat.numel()).clone()
        res = res_flat.view(host_tensor.shape)

        if tensor.is_cpu:
            tensor.copy_(res)
            return tensor
        tensor.copy_(res.to(tensor.device))
        return tensor
