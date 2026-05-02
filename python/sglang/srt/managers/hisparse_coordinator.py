# Unified coordinator for both native sparse attention (DSA/NSA MLA) and the
# non-native sparse-decode path (MHA/GQA + pluggable sparse algorithms like
# Quest).  The class exposes a single external entry point; algorithm/adapter
# logic lives in the optional :class:`SparseAlgorithmController` injected at
# construction time.

import logging
import os
from typing import TYPE_CHECKING, List, Literal, NamedTuple, Optional, Type

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseMHATokenToKVPool,
    HiSparseNSATokenToKVPool,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool_host import (
    HostKVCache,
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.utils import get_device_module

device_module = get_device_module()

from sglang.jit_kernel.hisparse import (
    load_cache_to_device_buffer_mha,
    load_cache_to_device_buffer_mla,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
        SparseAlgorithmController,
    )

logger = logging.getLogger(__name__)


class HiSparseAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


class HiSparseTokenStats(NamedTuple):
    device_tokens: int
    device_token_usage: float
    host_tokens: int
    host_token_usage: float


class HiSparseCoordinator:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: HiSparseTokenToKVPoolAllocator,
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group: torch.distributed.ProcessGroup,
        host_to_device_ratio: int = 2,
        mode: Literal["dsa_native", "sparse_decode"] = "dsa_native",
        algorithm_controller: Optional["SparseAlgorithmController"] = None,
        host_pool_class: Optional[Type[HostKVCache]] = None,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device
        self.mode = mode

        self.mem_pool_device = self.token_to_kv_pool_allocator.get_kvcache()
        # Host pool: layout and class differ between the DSA (MLA) and
        # sparse-decode (MHA) paths.
        if host_pool_class is None:
            host_pool_class = (
                MHATokenToKVPoolHost
                if mode == "sparse_decode"
                else MLATokenToKVPoolHost
            )

        if mode == "sparse_decode":
            assert isinstance(self.mem_pool_device, HiSparseMHATokenToKVPool), (
                "sparse_decode mode requires HiSparseMHATokenToKVPool, got "
                f"{type(self.mem_pool_device).__name__}"
            )
            self.mem_pool_host = host_pool_class(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=1,
                layout="layer_first",
            )
        else:
            assert isinstance(self.mem_pool_device, HiSparseNSATokenToKVPool), (
                "dsa_native mode requires HiSparseNSATokenToKVPool, got "
                f"{type(self.mem_pool_device).__name__}"
            )
            self.mem_pool_host = host_pool_class(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=1,
                layout="layer_first",
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )

        max_num_reqs = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len

        # to have an extra page for new tokens
        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(
            (max_num_reqs, self.padded_buffer_size), dtype=torch.int64, device=device
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
        self.ack_staging_queue: List[HiSparseAct] = []
        self.decode_producer_stream = None
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # initialize data structures for swap-in kernel
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

        # Pre-allocated output buffer for swap_in_selected_pages (CUDA-graph safe)
        self.top_k_device_locs_buffer = torch.full(
            (max_num_reqs, self.top_k), -1, dtype=torch.int32, device=device
        )
        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        # CPU flag: True means "skip backup on the next decode step" because
        # staging already backed up all prefill tokens.  Cleared after one step.
        self._skip_first_backup = [False] * max_num_reqs
        self._host_backed_up_len = [0] * max_num_reqs

        # Sparse-decode controller (non-DSA path).  Bound here so the
        # algorithm's representation pool can be initialized against the
        # coordinator-owned device pool.
        self.algorithm_controller: Optional["SparseAlgorithmController"] = (
            algorithm_controller
        )
        if self.algorithm_controller is not None:
            if self.mode != "sparse_decode":
                raise ValueError(
                    "algorithm_controller is only valid when mode='sparse_decode'"
                )
            self.algorithm_controller.bind_io(self)

        # Per-decode-step "every request stays dense" flag.  Refreshed once
        # per step in :meth:`map_last_loc_to_buffer` and consumed in
        # :meth:`SparseAlgorithmController.begin_layer_decode` to skip the
        # retrieve / swap-in / metadata rewrite path when none of the active
        # requests is long enough to trigger sparse attention.  Defaults to
        # False so the first call behaves like the old code path.
        self._all_dense_this_step: bool = False

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def get_token_stats(self) -> HiSparseTokenStats:
        device_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        device_capacity = device_allocator.size
        device_tokens = device_capacity - device_allocator.available_size()
        host_capacity = self.mem_pool_host.size
        host_tokens = host_capacity - self.mem_pool_host.available_size()
        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
        )

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
            logger.error(
                "HiSparse: host mem pool alloc failed for %d tokens (req %s)",
                prefill_len,
                req.rid,
            )
            raise RuntimeError(
                f"HiSparse host mem pool alloc failed for {prefill_len} tokens"
            )
        host_indices = host_indices.to(device=self.device)
        self.req_to_host_pool[req.req_pool_idx, :prefill_len] = host_indices

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices, device_indices, io_backend="kernel"
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def admit_request_direct(self, req: Req) -> None:
        """Direct-to-host path: KV data already resides in host pool via RDMA.

        Skips staging DMA entirely. Only allocates a small device buffer
        (4KB) for decode-time swap-in, then marks the request as ready.
        Host indices were already written to req_to_host_pool.

        Metadata fixups after alloc_device_buffer():
        - alloc_device_buffer() sets device_buffer_tokens = [0, 1, ..., buf_size-1],
          which tells the swap-in kernel that those tokens are cached in the device
          buffer.  In the staging path this is correct (prefill filled the buffer),
          but here the buffer is empty.
        """
        self.alloc_device_buffer(req)

        if req.kv_allocated_len <= self.device_buffer_size:
            # Short sequences (seq_len <= device_buffer_size): the kernel fast path
            # returns device_buffer_locs directly without any host loading, so we
            # must preload all tokens from host pool into the device buffer
            # TODO(hzh0425): Optimize this.
            self._preload_to_device_buffer(req)
        else:
            # Long sequence: reset device_buffer_tokens to -1 so the kernel
            # sees all slots as empty → every top-k lookup is a miss → host load.
            self.req_device_buffer_tokens[
                :, req.req_pool_idx, : self.device_buffer_size
            ] = -1

        req.staging = False
        self._skip_first_backup[req.req_pool_idx] = True
        self._host_backed_up_len[req.req_pool_idx] = req.kv_allocated_len
        logger.debug("HiSparse: admitting request %s directly", req.rid)

    def _preload_to_device_buffer(self, req: Req) -> None:
        """Preload all tokens from host pool into the device buffer."""
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
        # Allocate only enough for current tokens (page-aligned).
        # When prefill already fills device_buffer_size, include the reserved page.
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
            logger.error(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(kv_allocated_len=%d, alloc_size=%d)",
                req.rid,
                req.kv_allocated_len,
                alloc_size,
            )
            raise RuntimeError("HiSparse alloc_device_buffer returned None")

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
        ready_reqs = []
        if len(self.ack_staging_queue) == 0:
            return ready_reqs

        finish_count = 0
        for _, finish_event, _ in self.ack_staging_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make sure the same update to scheduler
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, _, req = self.ack_staging_queue.pop(0)
            # prepare device buffer and update req
            self.alloc_device_buffer(req)
            req.hisparse_staging = False
            self._skip_first_backup[req.req_pool_idx] = True
            self._host_backed_up_len[req.req_pool_idx] = req.kv_allocated_len
            finish_count -= 1
            ready_reqs.append(req)
        return ready_reqs

    def _grow_device_buffers(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Grow device buffers for requests whose sequence length exceeds current capacity."""
        current_caps = self.req_device_buffer_size[req_pool_indices_cpu]
        short_reqs_cpu = seq_lens_cpu <= self.device_buffer_size
        needs_grow_cpu = short_reqs_cpu & (seq_lens_cpu > current_caps)

        if torch.any(needs_grow_cpu):
            page_size = self.mem_pool_device.page_size
            grow_indices = torch.where(needs_grow_cpu)[0]

            # Compute all grow sizes on CPU, then do a single bulk allocation
            req_idxs = []
            old_caps = []
            new_caps = []
            grow_sizes = []
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
                    logger.error(
                        "HiSparse: _grow_device_buffers bulk alloc failed "
                        "(total_grow=%d)",
                        total_grow,
                    )
                    raise RuntimeError(
                        f"HiSparse _grow_device_buffers failed (total_grow={total_grow})"
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
        # Grow device buffers if needed and resolve the latest-token slot.
        reserved_buffer_loc = self._grow_device_buffers(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        self.req_device_buffer_token_locs[
            :, req_pool_indices, self.device_buffer_size
        ] = reserved_buffer_loc.to(torch.int32)

        # todo, clear the prior mapping as well
        self.mem_pool_device.full_to_hisparse_device_index_mapping[out_cache_loc] = (
            reserved_buffer_loc
        )

        # Refresh the "every request stays dense" flag once per step.  It is
        # consumed by :meth:`SparseAlgorithmController.begin_layer_decode` to
        # skip the sparse pipeline entirely (retrieve_topk / swap_in_kernel /
        # metadata rewrite) when no request in this batch is long enough to
        # trigger sparse attention.  Staying on CPU avoids a GPU sync in the
        # decode hot path.
        if self.algorithm_controller is not None:
            self._all_dense_this_step = (
                self.algorithm_controller.compute_all_dense_flag(
                    req_pool_indices_cpu.tolist(),
                    seq_lens_cpu=seq_lens_cpu,
                    device_buffer_size=self.device_buffer_size,
                )
            )
        else:
            self._all_dense_this_step = False

    def _eager_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Back up the previous decode token to host memory.

        Every decode step, the token written in the *previous* step must be
        backed up to host so the swap-in kernel can later recover it.

        The only exception is the first decode step right after staging: all
        prefill tokens were already backed up during staging, so there is nothing new to save yet.
        """
        # Build the list of batch positions that need a host backup.
        # Skip the first decode step after staging (prefill already backed up).
        backup_indices = []
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            actual_token_pos = int(seq_lens_cpu[i]) - 2
            if actual_token_pos < 0:
                continue

            host_loc = int(self.req_to_host_pool[req_idx, actual_token_pos].item())
            if host_loc >= 0:
                self._host_backed_up_len[req_idx] = max(
                    self._host_backed_up_len[req_idx], actual_token_pos + 1
                )
                if self._skip_first_backup[req_idx]:
                    self._skip_first_backup[req_idx] = False
                continue

            if self._skip_first_backup[req_idx]:
                self._skip_first_backup[req_idx] = False
                if actual_token_pos < self._host_backed_up_len[req_idx]:
                    continue
            backup_indices.append(i)

        if not backup_indices:
            return

        backup_indices_gpu = torch.tensor(
            backup_indices, dtype=torch.int64, device=self.device
        )
        # The previous token's position and its device buffer slot:
        #  - short seq: slot = seq_len - 2  (within the regular buffer)
        #  - long seq:  slot = device_buffer_size  (the reserved slot)
        actual_token_pos = seq_lens[backup_indices_gpu] - 2
        buffer_slot = actual_token_pos.clamp(max=self.device_buffer_size)

        backup_req_indices = req_pool_indices[backup_indices_gpu]
        device_locs = self.req_to_device_buffer[backup_req_indices, buffer_slot]

        host_locs = self.mem_pool_host.alloc(len(device_locs))
        if host_locs is None:
            logger.error(
                "HiSparse: host mem pool alloc failed for %d decode backup tokens",
                len(device_locs),
            )
            raise RuntimeError(
                f"HiSparse host mem pool alloc failed for {len(device_locs)} decode backup tokens"
            )
        host_locs = host_locs.to(device=self.device)
        self.req_to_host_pool[backup_req_indices, actual_token_pos] = host_locs
        for req_idx, token_pos in zip(
            backup_req_indices.detach().cpu().tolist(),
            actual_token_pos.detach().cpu().tolist(),
        ):
            req_idx = int(req_idx)
            self._host_backed_up_len[req_idx] = max(
                self._host_backed_up_len[req_idx], int(token_pos) + 1
            )

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

    def get_front_topk_tokens(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        top_k_indices = self.req_to_device_buffer[req_pool_indices, : self.top_k].to(
            torch.int32
        )
        topk_col_indices = torch.arange(self.top_k, device=self.device).unsqueeze(0)
        # Mask out positions beyond each request's seq_len
        mask = topk_col_indices >= seq_lens.unsqueeze(1)
        top_k_indices[mask] = -1
        return top_k_indices

    def naive_load_topk(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Load top-k selected tokens into device memory and return their device indices.

        This is a naive per-request loop implementation for debugging/validation.
        Production code uses swap_in_selected_pages (JIT CUDA kernel) instead.

        Args:
            req_pool_indices: Pool indices for each request.  Shape: (num_reqs,)
            seq_lens: Sequence lengths for each request.  Shape: (num_reqs,)
            top_k_tokens: Selected token positions per request.  Shape: (num_reqs, top_k)
            layer_id: The layer to load KV cache for.

        Returns:
            Device KV cache indices for the selected tokens.  Shape: (num_reqs, top_k)
        """
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )

        for i in range(num_reqs):
            seq_len = int(seq_lens[i].item())
            top_n = min(seq_len, self.top_k)
            if top_n == 0:
                continue

            req_idx = int(req_pool_indices[i].item())
            selected_tokens = top_k_tokens[i, :top_n].to(dtype=torch.int64)

            assert torch.all(
                selected_tokens >= 0
            ), f"Req {req_idx}: selected tokens contain negative positions"
            assert torch.all(selected_tokens < seq_len), (
                f"Req {req_idx}: selected tokens {selected_tokens.tolist()} "
                f"out of range for seq_len={seq_len}"
            )

            if seq_len <= self.device_buffer_size:
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]
            else:
                device_indices = torch.empty(
                    top_n, dtype=torch.int64, device=self.device
                )

                is_latest_token = selected_tokens == (seq_len - 1)
                needs_host_load = ~is_latest_token

                device_indices[is_latest_token] = self.req_to_device_buffer[
                    req_idx, self.device_buffer_size
                ]

                num_to_load = int(needs_host_load.sum().item())
                if num_to_load > 0:
                    tokens_to_load = selected_tokens[needs_host_load]
                    host_locs = self.req_to_host_pool[req_idx, tokens_to_load]

                    invalid_mask = host_locs < 0
                    if torch.any(invalid_mask):
                        bad_positions = tokens_to_load[invalid_mask].tolist()
                        raise AssertionError(
                            f"Req {req_idx} (seq_len={seq_len}, layer={layer_id}): "
                            f"missing host backup at token positions {bad_positions}"
                        )

                    buffer_locs = self.req_to_device_buffer[req_idx, :num_to_load]
                    device_indices[needs_host_load] = buffer_locs

                    self.mem_pool_host.load_to_device_per_layer(
                        self.mem_pool_device,
                        host_locs,
                        buffer_locs,
                        layer_id,
                        io_backend="kernel",
                    )

            top_k_indices[i, :top_n] = device_indices.to(torch.int32)

        return top_k_indices

    def abort_staging_request(self, req: Req) -> None:
        """Remove a request from the staging queue and free its host resources.

        Must be called when aborting a request that has been admitted into staging
        but has not yet completed (i.e. req.hisparse_staging is True).
        """
        # Remove from staging queue
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        # Wait for any in-flight staging DMA to complete before freeing
        self.write_staging_stream.synchronize()

        # Free host memory that was allocated during admit_request_into_staging
        host_indices = self.req_to_host_pool[req.req_pool_idx, : req.kv_allocated_len]
        host_indices = host_indices[host_indices >= 0]
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self._skip_first_backup[req.req_pool_idx] = False
        self._host_backed_up_len[req.req_pool_idx] = 0
        req.hisparse_staging = False

    def retract_req(self, req: Req) -> None:
        if req.hisparse_staging:
            self.abort_staging_request(req)
        else:
            self.request_finished(req)

    def request_finished(self, req: Req):
        # release resources only after the execution of a potential overlapped batch
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        if self._has_pending_backup:
            self._backup_done_event.wait(device_module.current_stream())
            self._has_pending_backup = False

        if self.algorithm_controller is not None:
            self.algorithm_controller.on_request_end(req)

        # release memory — only free actually-allocated buffer indices
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
        # clear req info
        self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.lru_slots[:, req.req_pool_idx, :].copy_(self._lru_init)
        self._skip_first_backup[req.req_pool_idx] = False
        self._host_backed_up_len[req.req_pool_idx] = 0

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices."""
        # The CUDA kernel expects req_pool_indices as int64 and seq_lens as int32 or int64.
        if req_pool_indices.dtype != torch.int64:
            raise ValueError(
                f"req_pool_indices dtype {req_pool_indices.dtype} is not int64 as expected"
            )
        if seq_lens.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"seq_lens dtype {seq_lens.dtype} is not int32 or int64 as expected"
            )
        if top_k_result.dtype != torch.int32:
            raise ValueError(
                f"top_k_result dtype {top_k_result.dtype} is not int32 as expected"
            )

        self._debug_check_top_k_tokens_for_swap_in(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            top_k_result=top_k_result,
            layer_id=layer_id,
        )

        num_reqs = req_pool_indices.size(0)
        top_k_indices = self.top_k_device_locs_buffer[:num_reqs]
        top_k_indices.fill_(-1)
        # todo, adjustable for performance
        block_size = 1024

        if self.mode == "sparse_decode":
            load_cache_to_device_buffer_mha(
                top_k_tokens=top_k_result,
                device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
                host_cache_locs=self.req_to_host_pool,
                device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
                host_k_cache=self.mem_pool_host.k_buffer[layer_id],
                host_v_cache=self.mem_pool_host.v_buffer[layer_id],
                device_k_buffer=self.mem_pool_device.k_buffer[layer_id],
                device_v_buffer=self.mem_pool_device.v_buffer[layer_id],
                top_k_device_locs=top_k_indices,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                lru_slots=self.lru_slots[layer_id],
                item_size_bytes=self.mem_pool_host.token_stride_size,
                num_top_k=self.top_k,
                hot_buffer_size=self.device_buffer_size,
                page_size=1,
                block_size=block_size,
                num_real_reqs=self.num_real_reqs,
            )
        else:
            load_cache_to_device_buffer_mla(
                top_k_tokens=top_k_result,
                device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
                host_cache_locs=self.req_to_host_pool,
                device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
                host_cache=self.mem_pool_host.kv_buffer[layer_id],
                device_buffer=self.mem_pool_device.kv_buffer[layer_id],
                top_k_device_locs=top_k_indices,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                lru_slots=self.lru_slots[layer_id],
                item_size_bytes=self.mem_pool_host.token_stride_size,
                num_top_k=self.top_k,
                hot_buffer_size=self.device_buffer_size,
                page_size=1,
                block_size=block_size,
                num_real_reqs=self.num_real_reqs,
            )
        return top_k_indices

    def _debug_check_top_k_tokens_for_swap_in(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> None:
        """Fail early with actionable context before the swap-in kernel.

        The CUDA kernel can legally read the newest token from the reserved
        device slot, and it can read older tokens from host only if
        req_to_host_pool[req, token] has been populated.  If Quest returns an
        older token that has no host backup, the kernel would dereference host
        loc -1 and crash with an illegal memory access.  Keep this as a debug
        check while validating the FlashInfer sparse-decode path.
        """
        token_pos = top_k_result.to(torch.int64)
        seq_lens_i64 = seq_lens.to(torch.int64).view(-1, 1)
        valid_range = (token_pos >= 0) & (token_pos < seq_lens_i64)
        newest_token = seq_lens_i64 - 1
        is_newest = valid_range & (token_pos == newest_token)

        safe_token_pos = token_pos.clamp(min=0, max=self.req_to_host_pool.shape[1] - 1)
        host_locs = self.req_to_host_pool[
            req_pool_indices.to(self.req_to_host_pool.device).view(-1, 1),
            safe_token_pos.to(self.req_to_host_pool.device),
        ].to(top_k_result.device)
        has_host_backup = valid_range & (host_locs >= 0)
        bad = valid_range & (~is_newest) & (~has_host_backup)
        if torch.any(bad):
            prev_token = seq_lens_i64 - 2
            repairable = bad & (token_pos == prev_token)
            if torch.any(repairable):
                repair_rows = torch.unique(repairable.nonzero(as_tuple=True)[0])
                self._backup_previous_tokens_now(
                    req_pool_indices=req_pool_indices,
                    seq_lens=seq_lens,
                    rows=repair_rows,
                )
                host_locs = self.req_to_host_pool[
                    req_pool_indices.to(self.req_to_host_pool.device).view(-1, 1),
                    safe_token_pos.to(self.req_to_host_pool.device),
                ].to(top_k_result.device)
                has_host_backup = valid_range & (host_locs >= 0)
                bad = valid_range & (~is_newest) & (~has_host_backup)

        if not torch.any(bad):
            return

        bad_rows, bad_cols = bad.nonzero(as_tuple=True)
        max_report = min(int(bad_rows.numel()), 16)
        rows = bad_rows[:max_report]
        cols = bad_cols[:max_report]
        reqs = req_pool_indices[rows]
        bad_tokens = token_pos[rows, cols]
        bad_host_locs = host_locs[rows, cols]
        row_seq_lens = seq_lens_i64.view(-1)[rows]
        newest = newest_token.view(-1)[rows]
        logger.error(
            "HiSparse swap-in found top-k tokens without host backup. "
            "layer_id=%d, kernel will be skipped by raising. examples=%s, "
            "state=%s",
            layer_id,
            [
                {
                    "row": int(rows[i].item()),
                    "topk_col": int(cols[i].item()),
                    "req_pool_idx": int(reqs[i].item()),
                    "token_pos": int(bad_tokens[i].item()),
                    "seq_len": int(row_seq_lens[i].item()),
                    "newest_token": int(newest[i].item()),
                    "host_loc": int(bad_host_locs[i].item()),
                }
                for i in range(max_report)
            ],
            self._debug_dump_req_state(req_pool_indices[rows[:1]]),
        )
        raise RuntimeError(
            "HiSparse swap-in received top-k token(s) without host backup. "
            "See preceding log for token positions and request ids."
        )

    def _backup_previous_tokens_now(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        rows: torch.Tensor,
    ) -> None:
        if rows.numel() == 0:
            return
        rows = rows.to(device=req_pool_indices.device, dtype=torch.int64)
        backup_req_indices = req_pool_indices[rows]
        actual_token_pos = seq_lens[rows].to(torch.int64) - 2
        device_locs = self.req_to_device_buffer[
            backup_req_indices,
            actual_token_pos.clamp(max=self.device_buffer_size),
        ]

        already_backed = self.req_to_host_pool[
            backup_req_indices.to(self.req_to_host_pool.device),
            actual_token_pos.to(self.req_to_host_pool.device),
        ] >= 0
        if bool(torch.all(already_backed).item()):
            return

        backup_req_indices = backup_req_indices[~already_backed]
        actual_token_pos = actual_token_pos[~already_backed]
        device_locs = device_locs[~already_backed]
        host_locs = self.mem_pool_host.alloc(len(device_locs))
        if host_locs is None:
            raise RuntimeError(
                f"HiSparse host mem pool alloc failed for {len(device_locs)} repair backup tokens"
            )
        host_locs = host_locs.to(device=self.device)
        self.req_to_host_pool[backup_req_indices, actual_token_pos] = host_locs
        self.mem_pool_host.backup_from_device_all_layer(
            self.mem_pool_device,
            host_locs,
            device_locs,
            io_backend="kernel",
        )
        for req_idx, token_pos in zip(
            backup_req_indices.detach().cpu().tolist(),
            actual_token_pos.detach().cpu().tolist(),
        ):
            req_idx = int(req_idx)
            self._host_backed_up_len[req_idx] = max(
                self._host_backed_up_len[req_idx], int(token_pos) + 1
            )

    def _debug_dump_req_state(self, req_pool_indices: torch.Tensor):
        if req_pool_indices.numel() == 0:
            return {}
        req_idx = int(req_pool_indices[0].item())
        host_head = (
            self.req_to_host_pool[req_idx, : min(32, self.req_to_host_pool.shape[1])]
            .detach()
            .cpu()
            .tolist()
        )
        dev_cap = int(self.req_device_buffer_size[req_idx])
        dev_head = (
            self.req_to_device_buffer[req_idx, : min(32, self.req_to_device_buffer.shape[1])]
            .detach()
            .cpu()
            .tolist()
        )
        token_head = (
            self.req_device_buffer_tokens[0, req_idx, : min(32, self.req_device_buffer_tokens.shape[2])]
            .detach()
            .cpu()
            .tolist()
        )
        prompt_len = None
        if self.algorithm_controller is not None:
            prompt_len = self.algorithm_controller.states.prompt_lens_cpu.get(req_idx)
        return {
            "req_pool_idx": req_idx,
            "prompt_len_cpu": prompt_len,
            "req_device_buffer_size": dev_cap,
            "req_to_host_pool_head": host_head,
            "req_to_device_buffer_head": dev_head,
            "req_device_buffer_tokens_layer0_head": token_head,
        }

    def admit_request_from_gpu(self, req: Req) -> None:
        """Single-instance counterpart of :meth:`admit_request_into_staging`.

        In the sparse-decode path the prefill runs dense on the local GPU (no
        staging / RDMA).  After prefill finishes this method:

          1. Triggers one-shot per-layer representation construction on the
             still-GPU-resident full KV (the algorithm needs the raw K to
             compute page min/max).
          2. Copies the full logical KV rows of ``req`` to the host pool.
          3. Reserves a device hot buffer via :meth:`alloc_device_buffer` and
             migrates the tail tokens into it.
          4. Frees the logical rows so subsequent requests can reuse them.
          5. Marks the request ready for decode.
        """
        if self.mode != "sparse_decode":
            raise RuntimeError(
                "admit_request_from_gpu is only valid in sparse_decode mode; "
                "use admit_request_into_staging for the DSA path."
            )
        if self.algorithm_controller is None:
            raise RuntimeError(
                "sparse_decode mode requires an algorithm_controller to be "
                "bound to HiSparseCoordinator."
            )

        kv_len = int(req.kv_allocated_len)
        if kv_len <= 0:
            return

        logical_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_len
        ].to(device=self.device, dtype=torch.int64)

        # The request's req_pool_idx can be assigned/reused across chunked
        # prefill.  Register at admission time as well, so sparse-mask decisions
        # during decode use the final live req_pool_idx instead of falling back
        # to prompt_len=0.
        self.algorithm_controller.on_request_begin(req)

        # 1. Algorithm representation construction.  At this point the
        # request's KV still lives in the logical rows of the
        # HiSparseMHATokenToKVPool and is GPU-visible via get_key_buffer.
        seq_lens_tensor = torch.tensor([kv_len], dtype=torch.int64, device=self.device)
        req_pool_indices_tensor = torch.tensor(
            [req.req_pool_idx], dtype=torch.int64, device=self.device
        )
        self.algorithm_controller.construct_prefill_representations(
            req=req,
            prefill_seq_lens=seq_lens_tensor,
            prefill_req_pool_indices=req_pool_indices_tensor,
        )

        # 2. Allocate host rows and backup full KV.
        host_indices = self.mem_pool_host.alloc(kv_len)
        if host_indices is None:
            raise RuntimeError(
                f"HiSparse(host) alloc failed for {kv_len} tokens (req {req.rid})"
            )
        host_indices = host_indices.to(device=self.device)
        self.req_to_host_pool[req.req_pool_idx, :kv_len] = host_indices

        if os.environ.get("SGLANG_HISPARSE_FLASHINFER_DEBUG", "0") == "1":
            logger.warning(
                "HiSparse admit_request_from_gpu: req_pool_idx=%s kv_len=%d "
                "origin_len=%d host_head=%s logical_head=%s",
                req.req_pool_idx,
                kv_len,
                len(req.origin_input_ids),
                host_indices[: min(16, host_indices.numel())].detach().cpu().tolist(),
                logical_indices[: min(16, logical_indices.numel())]
                .detach()
                .cpu()
                .tolist(),
            )

        self.mem_pool_host.backup_from_device_all_layer(
            self.mem_pool_device,
            host_indices,
            logical_indices,
            io_backend="kernel",
        )

        # 3. Allocate the device hot buffer and populate it with the tail of
        # the request's tokens (so the very first decode step has valid data
        # without waiting for the swap-in kernel).  The hot buffer slots are
        # DISTINCT physical rows from the prefill-time logical rows, since
        # ``HiSparseMHATokenToKVPool`` is sized ``size*ratio`` and the two
        # sub-allocators carve non-overlapping ranges.
        self.alloc_device_buffer(req)
        if kv_len <= self.device_buffer_size:
            self._preload_to_device_buffer(req)
        else:
            # Long sequence: mark hot buffer empty so every first-step lookup
            # becomes a miss and flows through the swap-in kernel.
            self.req_device_buffer_tokens[
                :, req.req_pool_idx, : self.device_buffer_size
            ] = -1

        # NOTE: We intentionally do NOT free the logical rows here.  The
        # scheduler later calls ``tree_cache.cache_finished_req`` which
        # reads ``req_to_token`` and frees ALL allocated logical rows
        # (prefill + decode) at request end via
        # ``HiSparseTokenToKVPoolAllocator.free``.  Freeing here would
        # double-free on request completion.  The memory cost is bounded
        # because the physical tensor is sized ``size * host_to_device_ratio``
        # to accommodate both roles simultaneously.

        req.hisparse_staging = False
        self._skip_first_backup[req.req_pool_idx] = True
        self._host_backed_up_len[req.req_pool_idx] = kv_len
        self.algorithm_controller.on_prefill_finished(req)
        logger.debug("HiSparse: admitted request %s from GPU (len=%d)", req.rid, kv_len)
