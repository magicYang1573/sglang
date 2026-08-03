"""Metadata adaptors for HBM-resident visibility-only sparsity."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.mem_cache.sparsity.contracts import Granularity, SelectionResult

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class MetadataAdaptor(ABC):
    """Translate logical selections into backend-owned metadata."""

    def bind_attention_backend(self, attention_backend: Any) -> None:
        """Bind optional backend services after attention backend creation."""

    @abstractmethod
    def capture_dense_metadata(self, metadata: Any) -> None: ...

    @abstractmethod
    def apply(
        self,
        result: SelectionResult,
        metadata: Any,
        forward_batch: ForwardBatch,
    ) -> Any: ...

    @abstractmethod
    def restore_dense_metadata(self, metadata: Any) -> None: ...


class HBMResidentPlacement:
    """Map request-relative logical pages to their resident HBM page IDs."""

    def __init__(self, req_to_token: torch.Tensor, page_size: int):
        self.req_to_token = req_to_token
        self.page_size = page_size

    def logical_to_physical_pages(
        self,
        logical_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> torch.Tensor:
        if logical_pages.ndim != 2:
            raise ValueError("logical_pages must have shape [batch, capacity]")
        if req_pool_indices.ndim != 1:
            raise ValueError("req_pool_indices must have shape [batch]")
        if logical_pages.shape[0] != req_pool_indices.shape[0]:
            raise ValueError("selection and request batch sizes must match")

        token_positions = (logical_pages * self.page_size).clamp(min=0).long()
        rows = req_pool_indices.long().unsqueeze(1).expand_as(token_positions)
        physical_tokens = self.req_to_token[rows, token_positions]
        physical_pages = torch.div(
            physical_tokens, self.page_size, rounding_mode="floor"
        )
        return torch.where(
            logical_pages >= 0,
            physical_pages,
            torch.zeros_like(physical_pages),
        ).to(torch.int32)


class FlashAttentionVisibilityAdaptor(MetadataAdaptor):
    """Rewrite FA3 page-table metadata without moving or freeing KV data."""

    def __init__(self, placement: HBMResidentPlacement):
        self.placement = placement
        self._dense_page_table = None
        self._dense_cache_seqlens = None
        self._dense_cu_seqlens_k = None
        self._dense_max_seq_len_k = None
        self._dense_scheduler_metadata = None
        self._sparse_scheduler_metadata = None
        self._scheduler_metadata_prepared = False
        self._scheduler_metadata_builder = None
        self._capture_active = False

    def bind_attention_backend(self, attention_backend: Any) -> None:
        builder = getattr(attention_backend, "_compute_scheduler_metadata", None)
        if builder is None:
            raise TypeError(
                "FA3 visibility requires an attention backend with scheduler "
                "metadata support"
            )
        self._scheduler_metadata_builder = builder

    def capture_dense_metadata(self, metadata: Any) -> None:
        required = (
            "page_table",
            "cache_seqlens_int32",
            "cu_seqlens_k",
            "max_seq_len_k",
        )
        if metadata is None or not all(hasattr(metadata, name) for name in required):
            raise TypeError("FA3 sparse visibility requires FlashAttention metadata")
        # Selection only overwrites a short prefix. Snapshot that prefix lazily
        # in apply(), once its width is known, instead of cloning the full dense
        # page table for every decode token.
        self._dense_page_table = None
        self._dense_cache_seqlens = metadata.cache_seqlens_int32.clone()
        self._dense_cu_seqlens_k = metadata.cu_seqlens_k.clone()
        self._dense_max_seq_len_k = metadata.max_seq_len_k
        self._dense_scheduler_metadata = getattr(metadata, "scheduler_metadata", None)
        self._sparse_scheduler_metadata = None
        self._scheduler_metadata_prepared = False
        self._capture_active = True

    def apply(
        self,
        result: SelectionResult,
        metadata: Any,
        forward_batch: ForwardBatch,
    ) -> Any:
        if result.granularity != Granularity.PAGE:
            raise ValueError("FA3 visibility adaptor requires page selections")
        if not self._capture_active:
            raise RuntimeError("capture_dense_metadata must be called before apply")
        # During warmup or short decode, no row is sparse and FA3 may publish a
        # page table narrower than the policy's fixed capacity. Truncating is
        # safe: if any row were sparse, its sequence would already be wider
        # than the retained capacity and so would the batch page table.
        visible_capacity = min(result.capacity, metadata.page_table.shape[1])
        if self._dense_page_table is None:
            self._dense_page_table = metadata.page_table[:, :visible_capacity].clone()
        elif self._dense_page_table.shape[1] != visible_capacity:
            raise RuntimeError("selection capacity changed within one decode forward")

        kernel_applied = False
        seq_lens = getattr(forward_batch, "seq_lens", None)
        from sglang.srt.mem_cache.sparsity.policies.quest import (
            _QuestFA3SelectionResult,
        )

        if (
            isinstance(result, _QuestFA3SelectionResult)
            and seq_lens is not None
            and visible_capacity == result.capacity
        ):
            from sglang.srt.mem_cache.sparsity.kernels.quest_flashattention_metadata import (
                quest_finalize_to_flashattention_metadata_,
            )

            quest_finalize_to_flashattention_metadata_(
                topk_scores=result.topk_scores,
                topk_indices=result.topk_indices,
                k_per_req=result.k_per_req,
                recent_indices=result.recent_indices,
                recent_valid=result.recent_valid,
                selected_output=result.logical_indices,
                valid_lengths=result.valid_lengths,
                visible_kv_lens=result.visible_kv_lens,
                sparse_mask=result.sparse_mask,
                seq_lens=seq_lens,
                req_pool_indices=forward_batch.req_pool_indices,
                req_to_token=self.placement.req_to_token,
                page_table=metadata.page_table,
                cache_seqlens_int32=metadata.cache_seqlens_int32,
                cu_seqlens_k=metadata.cu_seqlens_k,
                page_size=self.placement.page_size,
                update_lengths=True,
            )
            kernel_applied = True
        elif (
            result.logical_indices.is_cuda
            and torch.version.hip is None
            and seq_lens is not None
        ):
            from sglang.srt.mem_cache.sparsity.kernels.quest_flashattention_metadata import (
                quest_update_flashattention_metadata_,
            )

            quest_update_flashattention_metadata_(
                selected_indices=result.logical_indices[:, :visible_capacity],
                valid_lengths=result.valid_lengths,
                sparse_mask=result.sparse_mask,
                seq_lens=seq_lens,
                req_pool_indices=forward_batch.req_pool_indices,
                req_to_token=self.placement.req_to_token,
                page_table=metadata.page_table,
                cache_seqlens_int32=metadata.cache_seqlens_int32,
                cu_seqlens_k=metadata.cu_seqlens_k,
                page_size=self.placement.page_size,
                update_lengths=True,
            )
            kernel_applied = True

        if not kernel_applied:
            physical_pages = self.placement.logical_to_physical_pages(
                result.logical_indices[:, :visible_capacity],
                forward_batch.req_pool_indices,
            )
            columns = torch.arange(
                visible_capacity, device=physical_pages.device
            ).unsqueeze(0)
            valid = columns < result.valid_lengths.unsqueeze(1)
            update = result.sparse_mask.unsqueeze(1) & valid
            dense_prefix = self._dense_page_table[:, :visible_capacity]
            metadata.page_table[:, :visible_capacity].copy_(
                torch.where(update, physical_pages, dense_prefix)
            )
            metadata.cache_seqlens_int32.copy_(
                torch.where(
                    result.sparse_mask,
                    result.visible_kv_lens.to(torch.int32),
                    self._dense_cache_seqlens,
                )
            )
            metadata.cu_seqlens_k[0].zero_()
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
            )
        sparse_max_seq_len_k = self._dense_max_seq_len_k
        if result.max_visible_kv_len is not None:
            sparse_max_seq_len_k = min(sparse_max_seq_len_k, result.max_visible_kv_len)
        metadata.max_seq_len_k = sparse_max_seq_len_k
        if hasattr(metadata, "scheduler_metadata"):
            # A schedule precomputed for dense lengths is invalid. Recompute it
            # once per decode step and reuse it across every sparse layer.
            if (
                not self._scheduler_metadata_prepared
                and self._scheduler_metadata_builder is not None
            ):
                self._sparse_scheduler_metadata = self._scheduler_metadata_builder(
                    metadata.cache_seqlens_int32.shape[0],
                    max(sparse_max_seq_len_k, 1),
                    metadata.cache_seqlens_int32,
                    metadata.cu_seqlens_q,
                )
                self._scheduler_metadata_prepared = True
            metadata.scheduler_metadata = self._sparse_scheduler_metadata
        return metadata

    def restore_dense_metadata(self, metadata: Any) -> None:
        if not self._capture_active:
            return
        if self._dense_page_table is not None:
            width = self._dense_page_table.shape[1]
            metadata.page_table[:, :width].copy_(self._dense_page_table)
        metadata.cache_seqlens_int32.copy_(self._dense_cache_seqlens)
        metadata.cu_seqlens_k.copy_(self._dense_cu_seqlens_k)
        metadata.max_seq_len_k = self._dense_max_seq_len_k
        if hasattr(metadata, "scheduler_metadata"):
            metadata.scheduler_metadata = self._dense_scheduler_metadata
        self._capture_active = False
        self._dense_page_table = None


class TritonVisibilityAdaptor(MetadataAdaptor):
    """Swap Triton decode metadata to compact selected-token buffers.

    The dense backend-owned tensors remain untouched and are restored after the
    configured sparse layer range.  Sparse buffers are reused across layers and
    decode steps, so the hot path performs no host read or exact-size allocation.
    """

    def __init__(self, placement: HBMResidentPlacement):
        self.placement = placement
        self._num_kv_splits_builder = None
        self._dense_kv_indptr = None
        self._dense_kv_indices = None
        self._dense_num_kv_splits = None
        self._sparse_kv_indptr = None
        self._sparse_kv_indices = None
        self._sparse_num_kv_splits = None
        self._capture_active = False

    def bind_attention_backend(self, attention_backend: Any) -> None:
        builder = getattr(attention_backend, "get_num_kv_splits", None)
        if builder is None:
            raise TypeError(
                "Triton visibility requires an attention backend with "
                "get_num_kv_splits support"
            )
        self._num_kv_splits_builder = builder

    def capture_dense_metadata(self, metadata: Any) -> None:
        required = ("kv_indptr", "kv_indices", "num_kv_splits")
        if metadata is None or not all(hasattr(metadata, name) for name in required):
            raise TypeError("Triton sparse visibility requires decode metadata")
        if metadata.kv_indptr is None or metadata.kv_indices is None:
            raise TypeError("Triton sparse visibility requires KV index metadata")
        self._dense_kv_indptr = metadata.kv_indptr
        self._dense_kv_indices = metadata.kv_indices
        self._dense_num_kv_splits = metadata.num_kv_splits
        if (
            self._sparse_kv_indptr is None
            or self._sparse_kv_indptr.numel() < metadata.kv_indptr.numel()
        ):
            self._sparse_kv_indptr = torch.empty_like(metadata.kv_indptr)
        if (
            self._sparse_kv_indices is None
            or self._sparse_kv_indices.numel() < metadata.kv_indices.numel()
        ):
            self._sparse_kv_indices = torch.empty_like(metadata.kv_indices)
        if metadata.num_kv_splits is not None and (
            self._sparse_num_kv_splits is None
            or self._sparse_num_kv_splits.numel() < metadata.num_kv_splits.numel()
        ):
            self._sparse_num_kv_splits = torch.empty_like(metadata.num_kv_splits)
        self._capture_active = True

    def apply(
        self,
        result: SelectionResult,
        metadata: Any,
        forward_batch: ForwardBatch,
    ) -> Any:
        if result.granularity != Granularity.PAGE:
            raise ValueError("Triton visibility adaptor requires page selections")
        if not self._capture_active:
            raise RuntimeError("capture_dense_metadata must be called before apply")
        if result.max_visible_kv_len is None:
            raise ValueError("Triton visibility requires a static max_visible_kv_len")
        seq_lens = forward_batch.seq_lens
        batch_size = seq_lens.shape[0]
        sparse_lengths = torch.where(
            result.sparse_mask,
            result.visible_kv_lens.to(seq_lens.dtype),
            seq_lens,
        ).to(torch.int32)
        sparse_indptr = self._sparse_kv_indptr[: batch_size + 1]
        sparse_indptr[0].zero_()
        sparse_indptr[1:].copy_(torch.cumsum(sparse_lengths, dim=0, dtype=torch.int32))

        if seq_lens.is_cuda and torch.version.hip is None:
            from sglang.srt.mem_cache.sparsity.kernels.triton_visibility_metadata import (
                fill_triton_visibility_indices_,
            )

            fill_triton_visibility_indices_(
                selected_indices=result.logical_indices,
                sparse_mask=result.sparse_mask,
                seq_lens=seq_lens,
                req_pool_indices=forward_batch.req_pool_indices,
                req_to_token=self.placement.req_to_token,
                sparse_indptr=sparse_indptr,
                output=self._sparse_kv_indices,
                page_size=self.placement.page_size,
                max_row_tokens=result.max_visible_kv_len,
            )
        else:
            write_offset = 0
            for row in range(batch_size):
                seq_len = int(seq_lens[row])
                req_idx = int(forward_batch.req_pool_indices[row])
                if bool(result.sparse_mask[row]):
                    valid_pages = int(result.valid_lengths[row])
                    logical_tokens = []
                    for page in result.logical_indices[row, :valid_pages].tolist():
                        start = page * self.placement.page_size
                        logical_tokens.extend(
                            range(start, min(start + self.placement.page_size, seq_len))
                        )
                    logical_tokens = logical_tokens[: int(result.visible_kv_lens[row])]
                    logical = torch.tensor(
                        logical_tokens,
                        dtype=torch.long,
                        device=self.placement.req_to_token.device,
                    )
                else:
                    logical = torch.arange(
                        seq_len,
                        dtype=torch.long,
                        device=self.placement.req_to_token.device,
                    )
                physical = self.placement.req_to_token[req_idx, logical]
                self._sparse_kv_indices[
                    write_offset : write_offset + physical.numel()
                ].copy_(physical)
                write_offset += physical.numel()

        metadata.kv_indptr = sparse_indptr
        metadata.kv_indices = self._sparse_kv_indices
        if self._dense_num_kv_splits is not None:
            sparse_num_kv_splits = self._sparse_num_kv_splits[:batch_size]
            self._num_kv_splits_builder(
                sparse_num_kv_splits, sparse_lengths.clamp_min(1)
            )
            metadata.num_kv_splits = sparse_num_kv_splits
        return metadata

    def restore_dense_metadata(self, metadata: Any) -> None:
        if not self._capture_active:
            return
        metadata.kv_indptr = self._dense_kv_indptr
        metadata.kv_indices = self._dense_kv_indices
        metadata.num_kv_splits = self._dense_num_kv_splits
        self._dense_kv_indptr = None
        self._dense_kv_indices = None
        self._dense_num_kv_splits = None
        self._capture_active = False
