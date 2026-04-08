# to be combined with the sparse coordinator class and sparse algorithm family

import logging
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseNSATokenToKVPool,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost
from sglang.srt.utils import get_device_module

device_module = get_device_module()

from sglang.jit_kernel.hisparse import load_cache_to_device_buffer_mla
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)


PrefixHostCacheKey = Tuple[Optional[str], Tuple[int, ...]]


@dataclass
class PrefixHostCacheEntry:
    """Tracks a shared prefix's host pool slots for reference-counted reuse.

    When the first request with a given prefix completes staging, its prefix
    portion's host indices are registered here.  Subsequent requests that hit
    the same prefix in the radix cache reuse these host indices directly
    (no DMA), and only stage the extend portion.

    ref_count tracks how many in-flight requests reference this entry.
    When it drops to 0 the host slots can be reclaimed.
    """
    cache_key: PrefixHostCacheKey
    host_indices: torch.Tensor  # shape (prefix_len,), on coordinator device
    ref_count: int = 0
    restored_device_block: Optional[torch.Tensor] = None
    restored_prefix_len: int = 0


@dataclass
class ReqPrefixInfo:
    """Per-request record of which prefix host cache entry it references."""
    cache_key: Optional[PrefixHostCacheKey] = None
    prefix_len: int = 0


class HiSparseAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


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
        cxl_config: Optional[Dict] = None,
        tp_rank: int = 0,
        tp_size: int = 1,
        gpu_id: int = 0,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device

        self.mem_pool_device: HiSparseNSATokenToKVPool = (
            self.token_to_kv_pool_allocator.get_kvcache()
        )

        self._cxl_region = None
        if cxl_config is not None and cxl_config.get("enabled", False):
            self.mem_pool_host = self._init_cxl_host_pool(
                cxl_config, host_to_device_ratio, tp_rank, tp_size, gpu_id
            )
        else:
            self.mem_pool_host = MLATokenToKVPoolHost(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=1,
                layout="layer_first",
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )

        max_num_reqs = req_to_token_pool.size
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
        self.ack_staging_queue: List[HiSparseAct] = []
        self.decode_producer_stream = None

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

        # --- Prefix caching support ---
        # Maps prefix token tuple -> PrefixHostCacheEntry.
        # When radix cache is enabled, prefix-hit requests reuse host indices
        # from the first cold-start request instead of re-staging them.
        self.prefix_host_cache: Dict[PrefixHostCacheKey, PrefixHostCacheEntry] = {}
        # Per-request: tracks which prefix_host_cache entry the request uses.
        self.req_prefix_info: Dict[int, ReqPrefixInfo] = {}
    def _make_prefix_cache_key(
        self, req: Req, token_ids: Tuple[int, ...]
    ) -> PrefixHostCacheKey:
        return (req.extra_key, token_ids)

    def _find_prefix_host_entry(
        self, req: Req, prefix_token_ids: Tuple[int, ...]
    ) -> Tuple[Optional[PrefixHostCacheKey], Optional[PrefixHostCacheEntry]]:
        """Find a host-cache entry whose token_ids start with the prefix."""
        exact_key = self._make_prefix_cache_key(req, prefix_token_ids)
        exact = self.prefix_host_cache.get(exact_key)
        if exact is not None and len(exact.host_indices) >= len(prefix_token_ids):
            return exact_key, exact

        best_key = None
        best_entry = None
        for key, entry in self.prefix_host_cache.items():
            extra_key, token_ids = key
            if extra_key != req.extra_key:
                continue
            if len(token_ids) < len(prefix_token_ids):
                continue
            if token_ids[: len(prefix_token_ids)] != prefix_token_ids:
                continue
            if best_key is None or len(token_ids) < len(best_key[1]):
                best_key = key
                best_entry = entry
        return best_key, best_entry

    def _ensure_restored_device_block(
        self, entry: PrefixHostCacheEntry, prefix_len: int
    ) -> torch.Tensor:
        page_size = self.mem_pool_device.page_size
        alloc_size = ((prefix_len + page_size - 1) // page_size) * page_size

        if (
            entry.restored_device_block is not None
            and entry.restored_device_block.numel() >= alloc_size
            and entry.restored_prefix_len >= prefix_len
        ):
            return entry.restored_device_block[:prefix_len]

        if entry.restored_device_block is not None:
            self.token_to_kv_pool_allocator.free_hisparse_indices(
                entry.restored_device_block
            )

        block = self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
            alloc_size
        )
        if block is None:
            raise RuntimeError(
                f"HiSparse prefix reload failed: need {alloc_size} device slots"
            )
        entry.restored_device_block = block
        entry.restored_prefix_len = prefix_len
        return block[:prefix_len]

    def _restore_prefix_kv_for_prefill(
        self,
        req: Req,
        prefix_len: int,
        entry: PrefixHostCacheEntry,
    ) -> int:
        """Restore missing prefix KV mappings/device data before prefill."""
        if prefix_len <= 0:
            return 0

        if req.req_pool_idx is not None:
            self.req_to_host_pool[req.req_pool_idx, :prefix_len] = entry.host_indices[
                :prefix_len
            ]

        logical_prefix = req.prefix_indices[:prefix_len]
        mapping = self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping[
            logical_prefix
        ]
        missing_mask = mapping <= 0
        if not torch.any(missing_mask):
            return 0

        restored_device = self._ensure_restored_device_block(entry, prefix_len).contiguous()
        old_positive = torch.unique(mapping[mapping > 0])
        if old_positive.numel() > 0:
            to_free_mask = ~torch.isin(old_positive, restored_device)
            to_free = old_positive[to_free_mask]
            if to_free.numel() > 0:
                self.token_to_kv_pool_allocator.free_hisparse_indices(to_free)

        self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping[
            logical_prefix
        ] = restored_device

        host_prefix = entry.host_indices[:prefix_len].contiguous()
        for layer_id in range(self.mem_pool_device.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_prefix,
                restored_device,
                layer_id,
                io_backend="kernel",
            )
        return int(prefix_len)

    def prepare_prefill_prefix_mappings(self, reqs: List[Req]) -> None:
        """Restore prefix logical->device mappings before alloc_for_extend.

        This must run before alloc_for_extend because HiSparse alloc_extend()
        consults the prefix tail's hisparse mapping to allocate the extend part.
        """
        for req in reqs:
            prefix_len = len(req.prefix_indices)
            if prefix_len <= 0:
                continue

            prefix_token_ids = tuple(req.fill_ids[:prefix_len])
            _, entry = self._find_prefix_host_entry(req, prefix_token_ids)
            if entry is None:
                continue

            restored = self._restore_prefix_kv_for_prefill(req, prefix_len, entry)
            if restored > 0:
                logger.info(
                    "HiSparse prefix-cache prefill-restore: req=%s prefix_len=%d restored_tokens=%d",
                    req.rid,
                    prefix_len,
                    restored,
                )

    def prepare_prefill_prefix_kv(self, reqs: List[Req]) -> None:
        """Restore prefix KV from host before a prefix-hit prefill batch runs."""
        for req in reqs:
            prefix_len = len(req.prefix_indices)
            if prefix_len <= 0 or req.req_pool_idx is None:
                continue

            prefix_token_ids = tuple(req.fill_ids[:prefix_len])
            cache_key, entry = self._find_prefix_host_entry(req, prefix_token_ids)
            if entry is None:
                continue

            pinfo = self.req_prefix_info.get(req.req_pool_idx)
            if pinfo is None or pinfo.cache_key != cache_key:
                entry.ref_count += 1
                self.req_prefix_info[req.req_pool_idx] = ReqPrefixInfo(
                    cache_key=cache_key,
                    prefix_len=prefix_len,
                )

            restored = self._restore_prefix_kv_for_prefill(req, prefix_len, entry)
            logger.info(
                "HiSparse prefix-cache active: req=%s prefix_len=%d restored_tokens=%d",
                req.rid,
                prefix_len,
                restored,
            )

    def _init_cxl_host_pool(
        self,
        cxl_config: Dict,
        host_to_device_ratio: int,
        tp_rank: int,
        tp_size: int,
        gpu_id: int,
    ) -> "CXLMLATokenToKVPoolHost":
        from sglang.srt.mem_cache.cxl_memory_pool import (
            CXLConfig,
            CXLMemoryRegion,
            CXLMLATokenToKVPoolHost,
        )

        total_map_bytes = cxl_config.get("map_bytes", 64 * 1024**3)
        region = CXLMemoryRegion(
            CXLConfig(
                dev_path=cxl_config.get("dev_path", "/dev/dax0.0"),
                map_bytes=total_map_bytes,
                gpu_id=gpu_id,
            )
        )
        region.init()

        # Partition the CXL region among TP ranks so each rank uses
        # a disjoint slice. All TP processes mmap the same DAX device
        # but only allocate from their own partition.
        region.set_partition(tp_rank, tp_size)
        self._cxl_region = region

        logger.info(
            "HiSparse tp_rank=%d/%d: CXL pool at %s, partition offset=0x%x, "
            "partition size=%.2f GB",
            tp_rank, tp_size, cxl_config.get("dev_path"),
            region._alloc_offset, region._partition_size / 1e9,
        )
        return CXLMLATokenToKVPoolHost(
            device_pool=self.mem_pool_device,
            host_to_device_ratio=host_to_device_ratio,
            host_size=0,
            page_size=1,
            layout="layer_first",
            cxl_region=region,
            override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
        )

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def admit_request_into_staging(self, req: Req) -> None:
        req.hisparse_staging = True
        total_len = len(req.fill_ids)
        prefix_len = len(req.prefix_indices)

        logical_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :total_len
        ]

        # --- Prefix-hit path ---
        # If radix cache matched a prefix, try to reuse host pool entries.
        prefix_cache_key = None
        if prefix_len > 0:
            prefix_token_ids = tuple(req.fill_ids[:prefix_len])
            prefix_cache_key, entry = self._find_prefix_host_entry(
                req, prefix_token_ids
            )
            if entry is not None and len(entry.host_indices) >= prefix_len:
                # Reuse prefix host indices: no DMA needed for prefix portion.
                pinfo = self.req_prefix_info.get(req.req_pool_idx)
                if pinfo is None or pinfo.cache_key != prefix_cache_key:
                    entry.ref_count += 1
                    self.req_prefix_info[req.req_pool_idx] = ReqPrefixInfo(
                        cache_key=prefix_cache_key,
                        prefix_len=prefix_len,
                    )
                self.req_to_host_pool[
                    req.req_pool_idx, :prefix_len
                ] = entry.host_indices[:prefix_len]

                # Only stage the extend portion (tokens after prefix).
                extend_len = total_len - prefix_len
                if extend_len > 0:
                    extend_logical = logical_indices[prefix_len:]
                    extend_device = (
                        self.mem_pool_device._translate_loc_to_hisparse_device(
                            extend_logical
                        )
                    )
                    extend_host = self.mem_pool_host.alloc(extend_len)
                    if extend_host is None:
                        logger.error(
                            "HiSparse: host alloc failed for %d extend tokens (req %s)",
                            extend_len, req.rid,
                        )
                        raise RuntimeError(
                            f"HiSparse host alloc failed for {extend_len} extend tokens"
                        )
                    extend_host = extend_host.to(device=self.device)
                    self.req_to_host_pool[
                        req.req_pool_idx, prefix_len:total_len
                    ] = extend_host

                    start_event = device_module.Event()
                    finish_event = device_module.Event()
                    start_event.record()
                    with device_module.stream(self.write_staging_stream):
                        start_event.wait(self.write_staging_stream)
                        self.mem_pool_host.backup_from_device_all_layer(
                            self.mem_pool_device, extend_host, extend_device,
                            io_backend="kernel",
                        )
                        finish_event.record()
                        if extend_host.is_cuda:
                            extend_host.record_stream(self.write_staging_stream)
                        if extend_device.is_cuda:
                            extend_device.record_stream(self.write_staging_stream)
                else:
                    # Pure prefix hit, no extend: staging completes immediately.
                    start_event = device_module.Event()
                    finish_event = device_module.Event()
                    start_event.record()
                    finish_event.record()

                logger.debug(
                    "HiSparse prefix-hit: req %s reuses %d prefix host slots, "
                    "staging %d extend tokens",
                    req.rid, prefix_len, total_len - prefix_len,
                )
                logger.info(
                    "HiSparse prefix-cache staging-reuse: req=%s prefix_len=%d extend_len=%d",
                    req.rid,
                    prefix_len,
                    total_len - prefix_len,
                )
                self.ack_staging_queue.append(
                    HiSparseAct(start_event, finish_event, req)
                )
                return

        # --- Cold-start path (no prefix hit or prefix cache not populated) ---
        device_indices = self.mem_pool_device._translate_loc_to_hisparse_device(
            logical_indices
        )
        prefill_len = len(device_indices)
        host_indices = self.mem_pool_host.alloc(prefill_len)
        if host_indices is None:
            logger.error(
                "HiSparse: host mem pool alloc failed for %d tokens (req %s)",
                prefill_len, req.rid,
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
                self.mem_pool_device, host_indices, device_indices,
                io_backend="kernel",
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        # Register host cache entry for future prefix reuse.
        if prefix_len > 0 and prefix_cache_key is not None:
            if prefix_cache_key not in self.prefix_host_cache:
                self.prefix_host_cache[prefix_cache_key] = PrefixHostCacheEntry(
                    cache_key=prefix_cache_key,
                    host_indices=host_indices[:prefix_len].clone(),
                    ref_count=1,
                )
            else:
                self.prefix_host_cache[prefix_cache_key].ref_count += 1
            self.req_prefix_info[req.req_pool_idx] = ReqPrefixInfo(
                cache_key=prefix_cache_key,
                prefix_len=prefix_len,
            )
        elif prefix_len == 0 and total_len > 0:
            # Keep the first cold-start request's full host backup alive so a later
            # prefix-hit request can recover its prefix KV back into device memory
            # before prefill.
            full_cache_key = self._make_prefix_cache_key(
                req, tuple(req.fill_ids[:total_len])
            )
            if full_cache_key not in self.prefix_host_cache:
                self.prefix_host_cache[full_cache_key] = PrefixHostCacheEntry(
                    cache_key=full_cache_key,
                    host_indices=host_indices[:total_len].clone(),
                    ref_count=1,
                )
            else:
                self.prefix_host_cache[full_cache_key].ref_count += 1
            self.req_prefix_info[req.req_pool_idx] = ReqPrefixInfo(
                cache_key=full_cache_key,
                prefix_len=total_len,
            )

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def _materialize_short_seq_device_buffer_from_host(
        self, req: Req, seq_len: int, device_indices: torch.Tensor
    ) -> None:
        """Eagerly populate short-seq device buffer slots from host.

        The HiSparse kernel has a short-sequence fast path (`seq_len <= hot buffer`)
        that assumes token position `i` already resides in `device_buffer_locs[i]`
        and does not consult `device_buffer_tokens`.

        For prefix-hit requests we intentionally initialize the device buffer as
        empty and rely on host swap-in during decode. That works for the general
        path, but not for the short-seq fast path. To preserve correctness, we
        eagerly materialize the whole short sequence into the per-request device
        buffer before decode starts.
        """
        host_indices = self.req_to_host_pool[req.req_pool_idx, :seq_len]
        invalid_mask = host_indices < 0
        if torch.any(invalid_mask):
            bad_positions = torch.where(invalid_mask)[0].tolist()
            raise AssertionError(
                f"Req {req.rid}: missing host backup for short-seq tokens at positions "
                f"{bad_positions}"
            )

        host_indices = host_indices.contiguous()
        device_indices = device_indices[:seq_len].contiguous()
        for layer_id in range(self.mem_pool_device.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
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

        # Determine if this is a prefix-hit request.
        # Prefix-hit requests have some tokens whose KV data is only in the
        # host pool (not in the device buffer), so we initialize device_buffer_tokens
        # to -1 and let the first decode step swap-in the needed top-k.
        pinfo = self.req_prefix_info.get(req.req_pool_idx)
        is_prefix_hit = (
            pinfo is not None
            and pinfo.prefix_len > 0
            and pinfo.prefix_len < req.kv_allocated_len
        )

        if is_prefix_hit:
            self.req_device_buffer_tokens[
                :, req.req_pool_idx, : self.device_buffer_size
            ] = -1
            if req.kv_allocated_len <= self.device_buffer_size:
                # Short-seq fast path in the kernel assumes token i already lives in
                # device_buffer_locs[i]. Materialize the full sequence now so the
                # fast path remains valid for prefix-hit requests.
                self._materialize_short_seq_device_buffer_from_host(
                    req, req.kv_allocated_len, buffer_indices
                )
                self.req_device_buffer_tokens[
                    :, req.req_pool_idx, : req.kv_allocated_len
                ] = torch.arange(req.kv_allocated_len, device=self.device)
        else:
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
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)

        # Build the list of batch positions that need a host backup.
        # Skip the first decode step after staging (prefill already backed up).
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

        self.mem_pool_host.backup_from_device_all_layer(
            self.mem_pool_device,
            host_locs,
            device_locs.contiguous(),
            io_backend="kernel",
        )

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

        # Clean up prefix cache reference if applicable.
        pinfo = self.req_prefix_info.pop(req.req_pool_idx, None)
        prefix_len = pinfo.prefix_len if pinfo is not None else 0
        prefix_cache_key = pinfo.cache_key if pinfo is not None else None

        # Free extend portion only (prefix portion managed by ref count).
        if prefix_len < req.kv_allocated_len:
            extend_host = self.req_to_host_pool[
                req.req_pool_idx, prefix_len : req.kv_allocated_len
            ]
            extend_host = extend_host[extend_host >= 0]
            if extend_host.numel() > 0:
                self.mem_pool_host.free(extend_host)

        if prefix_cache_key is not None and prefix_cache_key in self.prefix_host_cache:
            entry = self.prefix_host_cache[prefix_cache_key]
            entry.ref_count = max(0, entry.ref_count - 1)
        elif prefix_len == 0:
            # No prefix tracking — free all host indices (original behavior).
            host_indices = self.req_to_host_pool[
                req.req_pool_idx, : req.kv_allocated_len
            ]
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
        # release resources only after the execution of a potential overlapped batch
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)

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

        # Free host pool indices with prefix-aware reference counting.
        pinfo = self.req_prefix_info.pop(req.req_pool_idx, None)
        prefix_len = pinfo.prefix_len if pinfo is not None else 0
        prefix_cache_key = pinfo.cache_key if pinfo is not None else None

        # Free extend portion (non-prefix host indices).
        if prefix_len < req.kv_allocated_len:
            extend_host = self.req_to_host_pool[
                req.req_pool_idx, prefix_len : req.kv_allocated_len
            ]
            extend_host = extend_host[extend_host >= 0]
            if extend_host.numel() > 0:
                self.mem_pool_host.free(extend_host)

        # Decrement prefix host cache ref_count.
        # The entry is kept alive even when ref_count reaches 0 so that
        # future requests with the same prefix can still reuse the host
        # indices. The entry is only cleaned up explicitly via
        # evict_prefix_host_cache() or when the coordinator is destroyed.
        if prefix_cache_key is not None and prefix_cache_key in self.prefix_host_cache:
            entry = self.prefix_host_cache[prefix_cache_key]
            entry.ref_count = max(0, entry.ref_count - 1)
        elif prefix_len == 0:
            # No prefix tracking — free all host indices (original behavior).
            host_indices = self.req_to_host_pool[
                req.req_pool_idx, : req.kv_allocated_len
            ]
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

    def evict_prefix_host_cache(self) -> int:
        """Free all unreferenced prefix host cache entries.

        Returns the number of host pool slots freed.
        """
        freed = 0
        to_delete = []
        for key, entry in self.prefix_host_cache.items():
            if entry.ref_count <= 0:
                to_delete.append(key)
                if entry.host_indices.numel() > 0:
                    self.mem_pool_host.free(entry.host_indices)
                    freed += entry.host_indices.numel()
                if entry.restored_device_block is not None:
                    self.token_to_kv_pool_allocator.free_hisparse_indices(
                        entry.restored_device_block
                    )
        for key in to_delete:
            del self.prefix_host_cache[key]
        if freed > 0:
            logger.debug(
                "HiSparse: evicted %d prefix host cache entries, freed %d host slots",
                len(to_delete), freed,
            )
        return freed

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
            top_k_result = top_k_result.to(torch.int32)

        num_reqs = req_pool_indices.size(0)
        top_k_indices = self.top_k_device_locs_buffer[:num_reqs]
        top_k_indices.fill_(-1)
        # todo, adjustable for performance
        block_size = 1024
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
