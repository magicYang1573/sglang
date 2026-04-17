"""HiSparse coordinator variant for MHA / GQA models.

Mirror of :class:`sglang.srt.managers.hisparse_coordinator.HiSparseCoordinator`
but specialized for ``HiSparseMHATokenToKVPool`` (K and V stored in two
buffers) and the matching MHA JIT kernel wrapper.

The control flow (staging, LRU hot buffer bookkeeping, eager backup) is
identical to the MLA coordinator; the differences are confined to:

* host pool type: :class:`MHATokenToKVPoolHost` instead of
  :class:`MLATokenToKVPoolHost`;
* swap-in kernel call: ``load_cache_to_device_buffer_mha`` with explicit
  ``host_cache_{k,v}`` and ``device_buffer_{k,v}`` pointers;
* backup helpers: we rely on ``MHATokenToKVPoolHost``'s
  ``backup_from_device_all_layer`` (already writes both K and V).
"""

from __future__ import annotations

import logging
from typing import List, NamedTuple

import torch

from sglang.jit_kernel.hisparse import load_cache_to_device_buffer_mha
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.hisparse_mha_memory_pool import (
    HiSparseMHATokenToKVPool,
    HiSparseMHATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost
from sglang.srt.utils import get_device_module

device_module = get_device_module()

logger = logging.getLogger(__name__)


class HiSparseMHAAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


class HiSparseMHATokenStats(NamedTuple):
    device_tokens: int
    device_token_usage: float
    host_tokens: int
    host_token_usage: float


class HiSparseMHACoordinator:
    """MHA / GQA variant of HiSparse coordinator.

    The API surface intentionally matches ``HiSparseCoordinator`` (MLA) so the
    upper layers (scheduler, forward batch) can treat both uniformly.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: HiSparseMHATokenToKVPoolAllocator,
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group: torch.distributed.ProcessGroup,
        host_to_device_ratio: int = 2,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device

        self.mem_pool_device: HiSparseMHATokenToKVPool = (
            self.token_to_kv_pool_allocator.get_kvcache()
        )
        self.mem_pool_host = MHATokenToKVPoolHost(
            device_pool=self.mem_pool_device,
            host_to_device_ratio=host_to_device_ratio,
            host_size=0,
            page_size=1,
            layout="layer_first",
        )

        max_num_reqs = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len

        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(
            (max_num_reqs, self.padded_buffer_size),
            dtype=torch.int64,
            device=device,
        )
        self.req_device_buffer_size = torch.zeros(
            max_num_reqs, dtype=torch.int64, device="cpu"
        )
        self.req_to_host_pool = torch.full(
            (max_num_reqs, max_context_len),
            -1,
            dtype=torch.int64,
            device=device,
        )

        self.write_staging_stream = device_module.Stream()
        self.decode_backup_stream = device_module.Stream()
        self.ack_staging_queue: List[HiSparseMHAAct] = []
        self.decode_producer_stream = None
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        layer_num = self.mem_pool_device.layer_num
        self.req_device_buffer_tokens = torch.full(
            (layer_num, max_num_reqs, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(
            (layer_num, max_num_reqs, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(layer_num, max_num_reqs, 1)
            .contiguous()
        )

        self.top_k_device_locs_buffer = torch.full(
            (max_num_reqs, self.top_k),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)
        self._skip_first_backup = [False] * max_num_reqs

        # bytes per token per buffer (K and V share stride for Qwen).
        self._item_size_bytes = int(self.mem_pool_device.bytes_per_token)

    # ------------------------------------------------------------------
    # Bookkeeping helpers (mirror MLA variant)
    # ------------------------------------------------------------------
    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def get_token_stats(self) -> HiSparseMHATokenStats:
        device_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        device_capacity = device_allocator.size
        device_tokens = device_capacity - device_allocator.available_size()
        host_capacity = self.mem_pool_host.size
        host_tokens = host_capacity - self.mem_pool_host.available_size()
        return HiSparseMHATokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
        )

    # ------------------------------------------------------------------
    # Admit / staging
    # ------------------------------------------------------------------
    def admit_request_into_staging(self, req: Req) -> None:
        req.hisparse_staging = True
        logical_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]
        device_indices = self.mem_pool_device._translate_loc_to_hisparse_device(
            logical_indices
        )

        prefill_len = len(device_indices)
        host_indices = self.mem_pool_host.alloc(prefill_len)
        if host_indices is None:
            raise RuntimeError(
                f"HiSparse(MHA) host mem pool alloc failed for {prefill_len} tokens"
            )
        host_indices = host_indices.to(device=self.device)
        self.req_to_host_pool[req.req_pool_idx, :prefill_len] = host_indices

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                io_backend="kernel",
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        self.ack_staging_queue.append(HiSparseMHAAct(start_event, finish_event, req))

    def admit_request_direct(self, req: Req) -> None:
        self.alloc_device_buffer(req)
        if req.kv_allocated_len <= self.device_buffer_size:
            self._preload_to_device_buffer(req)
        else:
            self.req_device_buffer_tokens[
                :, req.req_pool_idx, : self.device_buffer_size
            ] = -1
        req.staging = False
        self._skip_first_backup[req.req_pool_idx] = True

    def _preload_to_device_buffer(self, req: Req) -> None:
        n = req.kv_allocated_len
        host_indices = self.req_to_host_pool[req.req_pool_idx, :n]
        device_locs = self.req_to_device_buffer[req.req_pool_idx, :n]
        for layer_id in range(self.mem_pool_device.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_locs,
                layer_id,
                io_backend="kernel",
            )

    def alloc_device_buffer(self, req: Req) -> None:
        allocated_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.kv_allocated_len
        ]
        page_size = self.mem_pool_device.page_size
        alloc_size = min(
            ((req.kv_allocated_len + page_size - 1) // page_size) * page_size,
            self.device_buffer_size,
        )
        if alloc_size == self.device_buffer_size:
            alloc_size = self.padded_buffer_size
        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            allocated_indices,
            alloc_size,
        )
        if buffer_indices is None:
            raise RuntimeError(
                "HiSparse(MHA) alloc_device_buffer returned None"
            )

        self.req_to_device_buffer[req.req_pool_idx, :alloc_size] = buffer_indices
        self.req_device_buffer_size[req.req_pool_idx] = alloc_size

        self.req_device_buffer_tokens[
            :, req.req_pool_idx, : self.device_buffer_size
        ] = torch.arange(self.device_buffer_size, device=self.device)
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )

    def has_ongoing_staging(self) -> bool:
        return len(self.ack_staging_queue) > 0

    def collect_ready_reqs(self) -> List[Req]:
        ready_reqs: List[Req] = []
        if len(self.ack_staging_queue) == 0:
            return ready_reqs

        finish_count = 0
        for _, finish_event, _ in self.ack_staging_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, _, req = self.ack_staging_queue.pop(0)
            self.alloc_device_buffer(req)
            req.hisparse_staging = False
            self._skip_first_backup[req.req_pool_idx] = True
            finish_count -= 1
            ready_reqs.append(req)
        return ready_reqs

    # ------------------------------------------------------------------
    # Per-step bookkeeping
    # ------------------------------------------------------------------
    def _grow_device_buffers(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> torch.Tensor:
        current_caps = self.req_device_buffer_size[req_pool_indices_cpu]
        short_reqs_cpu = seq_lens_cpu <= self.device_buffer_size
        needs_grow_cpu = short_reqs_cpu & (seq_lens_cpu > current_caps)

        if torch.any(needs_grow_cpu):
            page_size = self.mem_pool_device.page_size
            grow_indices = torch.where(needs_grow_cpu)[0]

            req_idxs, old_caps, new_caps, grow_sizes = [], [], [], []
            total_grow = 0
            for i in grow_indices.tolist():
                req_idx = int(req_pool_indices_cpu[i])
                current_cap = int(current_caps[i])
                seq_len = int(seq_lens_cpu[i])
                new_cap = min(
                    ((seq_len + page_size - 1) // page_size) * page_size,
                    self.device_buffer_size,
                )
                if new_cap == self.device_buffer_size:
                    new_cap = self.padded_buffer_size
                grow_size = new_cap - current_cap
                if grow_size <= 0:
                    continue
                req_idxs.append(req_idx)
                old_caps.append(current_cap)
                new_caps.append(new_cap)
                grow_sizes.append(grow_size)
                total_grow += grow_size

            if total_grow > 0:
                all_new_indices = (
                    self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
                        total_grow
                    )
                )
                if all_new_indices is None:
                    raise RuntimeError(
                        f"HiSparse(MHA) grow alloc failed (total_grow={total_grow})"
                    )
                offset = 0
                for req_idx, current_cap, new_cap, grow_size in zip(
                    req_idxs, old_caps, new_caps, grow_sizes
                ):
                    chunk = all_new_indices[offset : offset + grow_size]
                    offset += grow_size
                    self.req_to_device_buffer[req_idx, current_cap:new_cap] = chunk
                    self.req_device_buffer_token_locs[
                        :, req_idx, current_cap:new_cap
                    ] = chunk
                    self.req_device_buffer_size[req_idx] = new_cap

        reserved_positions = (seq_lens - 1).clamp(max=self.device_buffer_size)
        return self.req_to_device_buffer[req_pool_indices, reserved_positions]

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> None:
        req_pool_indices_cpu = req_pool_indices.cpu()

        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )
        reserved_buffer_loc = self._grow_device_buffers(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        self.req_device_buffer_token_locs[
            :, req_pool_indices, self.device_buffer_size
        ] = reserved_buffer_loc.to(torch.int32)
        self.mem_pool_device.full_to_hisparse_device_index_mapping[out_cache_loc] = (
            reserved_buffer_loc
        )

    def _eager_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        backup_indices = []
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            if self._skip_first_backup[req_idx]:
                self._skip_first_backup[req_idx] = False
                continue
            backup_indices.append(i)
        if not backup_indices:
            return

        backup_indices_gpu = torch.tensor(
            backup_indices, dtype=torch.int64, device=self.device
        )
        actual_token_pos = seq_lens[backup_indices_gpu] - 2
        buffer_slot = actual_token_pos.clamp(max=self.device_buffer_size)

        backup_req_indices = req_pool_indices[backup_indices_gpu]
        device_locs = self.req_to_device_buffer[backup_req_indices, buffer_slot]

        host_locs = self.mem_pool_host.alloc(len(device_locs))
        if host_locs is None:
            raise RuntimeError(
                f"HiSparse(MHA) host mem pool alloc failed for {len(device_locs)} decode backup tokens"
            )
        host_locs = host_locs.to(device=self.device)
        self.req_to_host_pool[backup_req_indices, actual_token_pos] = host_locs

        if self._has_pending_backup:
            self._backup_done_event.wait(device_module.current_stream())
            self._has_pending_backup = False
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            self._backup_done_event.record()
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if backup_req_indices.is_cuda:
                backup_req_indices.record_stream(self.decode_backup_stream)
            if actual_token_pos.is_cuda:
                actual_token_pos.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True

    def wait_for_pending_backup(self) -> None:
        if not self._has_pending_backup:
            return
        self._backup_done_event.wait(device_module.current_stream())
        self._has_pending_backup = False

    # ------------------------------------------------------------------
    # Swap-in (called by SparseCoordinator.attention_begin on decode layers)
    # ------------------------------------------------------------------
    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into the MHA device buffer.

        ``top_k_tokens`` contains logical *token* positions per request
        (int32, shape ``[bs, top_k]``, -1 padded).  The returned tensor has
        the same shape and contains the device-buffer slots that the
        attention backend should address.
        """
        if req_pool_indices.dtype != torch.int64:
            raise ValueError(
                f"req_pool_indices dtype {req_pool_indices.dtype} != int64"
            )
        if seq_lens.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"seq_lens dtype {seq_lens.dtype} not int32/int64"
            )
        if top_k_tokens.dtype != torch.int32:
            raise ValueError(
                f"top_k_tokens dtype {top_k_tokens.dtype} != int32"
            )

        num_reqs = req_pool_indices.size(0)
        top_k_indices = self.top_k_device_locs_buffer[:num_reqs]
        top_k_indices.fill_(-1)
        block_size = 1024

        host_k = self.mem_pool_host.k_buffer[layer_id]
        host_v = self.mem_pool_host.v_buffer[layer_id]
        device_k = self.mem_pool_device.k_buffer[layer_id]
        device_v = self.mem_pool_device.v_buffer[layer_id]

        load_cache_to_device_buffer_mha(
            top_k_tokens=top_k_tokens,
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            host_cache_k=host_k,
            host_cache_v=host_v,
            device_buffer_k=device_k,
            device_buffer_v=device_v,
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self._item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=1,
            block_size=block_size,
            num_real_reqs=self.num_real_reqs,
        )
        return top_k_indices

    # ------------------------------------------------------------------
    # Teardown / abort
    # ------------------------------------------------------------------
    def abort_staging_request(self, req: Req) -> None:
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        self.write_staging_stream.synchronize()
        host_indices = self.req_to_host_pool[req.req_pool_idx, : req.kv_allocated_len]
        host_indices = host_indices[host_indices >= 0]
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self._skip_first_backup[req.req_pool_idx] = False
        req.hisparse_staging = False

    def retract_req(self, req: Req) -> None:
        if req.hisparse_staging:
            self.abort_staging_request(req)
        else:
            self.request_finished(req)

    def request_finished(self, req: Req):
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        if self._has_pending_backup:
            self._backup_done_event.wait(device_module.current_stream())
            self._has_pending_backup = False

        current_cap = int(self.req_device_buffer_size[req.req_pool_idx])
        buffer_indices = self.req_to_device_buffer[req.req_pool_idx, :current_cap]
        self.token_to_kv_pool_allocator.free_hisparse_indices(buffer_indices)

        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.kv_allocated_len
        ]
        self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping[
            allocated_locs
        ] = 0

        host_indices = self.req_to_host_pool[req.req_pool_idx, : req.kv_allocated_len]
        host_indices = host_indices[host_indices >= 0]
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)

        self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.lru_slots[:, req.req_pool_idx, :].copy_(self._lru_init)
        self._skip_first_backup[req.req_pool_idx] = False
