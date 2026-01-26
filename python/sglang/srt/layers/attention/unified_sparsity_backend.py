from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
	from sglang.srt.layers.radix_attention import RadixAttention
	from sglang.srt.model_executor.model_runner import ModelRunner


class UnifiedSparsityAttnBackend(AttentionBackend):
	def __init__(self, model_runner: ModelRunner):
		from sglang.srt.layers.attention.triton_ops.unified_sparsity_attention import (
			# extend_attention_fwd,
			# flash_decode_attention_fwd,
			flash_decode_sparse_attention_fwd,
		)
		from sglang.srt.layers.attention.triton_ops.double_sparsity_attention import (
			extend_attention_fwd,
			flash_decode_attention_fwd,
		)
		super().__init__()

		self.decode_attention_fwd = flash_decode_attention_fwd
		self.decode_sparse_attention_fwd = flash_decode_sparse_attention_fwd
		self.extend_attention_fwd = extend_attention_fwd
		self.num_head = model_runner.model_config.num_attention_heads
		self.head_dim = model_runner.model_config.hidden_size // self.num_head
		self.heavy_token_num = model_runner.server_args.unified_sparsity_heavy_token_num
		self.sparse_decode_threshold = model_runner.server_args.unified_sparsity_decode_threshold
		

		self.mid_out: torch.Tensor = None
		self.mid_o_logexpsum: torch.Tensor = None

		self.BLOCK_SEQ = 128

		if get_global_server_args().triton_attention_reduce_in_fp32:
			self.reduce_dtype = torch.float32
		else:
			self.reduce_dtype = torch.float16

		self.forward_metadata = None

	def init_forward_metadata(self, forward_batch: ForwardBatch):
		if forward_batch.forward_mode.is_decode():
			start_loc = torch.zeros_like(forward_batch.seq_lens, dtype=torch.int32)
			start_loc[1:] = torch.cumsum(forward_batch.seq_lens[:-1], dim=0)

			total_num_tokens = torch.sum(forward_batch.seq_lens).item()
			attn_logits = torch.empty(
				(self.num_head, total_num_tokens),
				dtype=self.reduce_dtype,
				device="cuda",
			)

			max_seq_len = torch.max(forward_batch.seq_lens).item()
			min_seq_len = torch.min(forward_batch.seq_lens).item()
			max_extend_len = None
			req_to_token = forward_batch.req_to_token_pool.req_to_token[
				forward_batch.req_pool_indices
			]
		else:
			start_loc = attn_logits = max_seq_len = min_seq_len = None
			prefix_lens = forward_batch.extend_prefix_lens
			max_extend_len = torch.max(forward_batch.seq_lens - prefix_lens).item()
			req_to_token = None

		self.forward_metadata = (
			start_loc,
			attn_logits,
			max_seq_len,
			min_seq_len,
			max_extend_len,
			req_to_token,
		)

	def forward_extend(
		self,
		q,
		k,
		v,
		layer: RadixAttention,
		forward_batch: ForwardBatch,
		save_kv_cache=True,
	):
		if layer.qk_head_dim != layer.v_head_dim:
			o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
		else:
			o = torch.empty_like(q)

		if save_kv_cache:
			forward_batch.token_to_kv_pool.set_kv_buffer(
				layer, forward_batch.out_cache_loc, k, v
			)

		(
			start_loc,
			attn_logits,
			max_seq_len,
			min_seq_len,
			max_extend_len,
			req_to_token,
		) = self.forward_metadata
		self.extend_attention_fwd(
			q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
			k.contiguous(),
			v.contiguous(),
			o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
			forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
			forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
			forward_batch.req_to_token_pool.req_to_token,
			forward_batch.req_pool_indices,
			forward_batch.seq_lens,
			forward_batch.extend_seq_lens,
			forward_batch.extend_start_loc,
			max_extend_len,
			layer.scaling,
			layer.logit_cap,
		)
		return o


	def _ensure_mid_buffers(self, batch_size, head_num, head_dim, heavy_token_num):
		block_seq_num = (heavy_token_num + self.BLOCK_SEQ - 1) // self.BLOCK_SEQ
		if (
			self.mid_out is None
			or self.mid_out.shape[0] != batch_size
			or self.mid_out.shape[1] != head_num
			or self.mid_out.shape[2] != block_seq_num
			or self.mid_out.shape[3] != head_dim
		):
			self.mid_out = torch.empty(
				[batch_size, head_num, block_seq_num, head_dim],
				dtype=torch.float32,
				device="cuda",
			)
			self.mid_o_logexpsum = torch.empty(
				[batch_size, head_num, block_seq_num],
				dtype=torch.float32,
				device="cuda",
			)

	def forward_decode(
		self,
		q,
		k,
		v,
		layer: RadixAttention,
		forward_batch: ForwardBatch,
		save_kv_cache=True,
	):
		q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

		if layer.qk_head_dim != layer.v_head_dim:
			o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
		else:
			o = torch.empty_like(q)

		(
			start_loc,
			attn_logits,
			max_seq_len,
			min_seq_len,
			max_extend_len,
			req_to_token,
		) = self.forward_metadata

		if save_kv_cache:
			forward_batch.token_to_kv_pool.set_kv_buffer(
				layer, forward_batch.out_cache_loc, k, v
			)

		print(min_seq_len, max_seq_len, self.heavy_token_num, self.sparse_decode_threshold)
		
		if (
			min_seq_len < self.heavy_token_num
			or max_seq_len < self.sparse_decode_threshold
		):
			self.decode_attention_fwd(
				q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
				forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
				forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
				o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
				forward_batch.req_to_token_pool.req_to_token,
				forward_batch.req_pool_indices,
				start_loc,
				forward_batch.seq_lens,
				attn_logits,
				max_seq_len,
				layer.scaling,
				layer.logit_cap,
			)
		else:

			# tmp all token for topk_token_indices
			base_indices = torch.arange(max_seq_len, device=forward_batch.seq_lens.device).unsqueeze(0)
			mask = base_indices < forward_batch.seq_lens.unsqueeze(1)

			# topk_token_indices: [H, B, k], Req_to_tokens: [B, S]
    		# topk_token_indices = torch.arange(0, heavy_token_num, device=q.device).unsqueeze(0).unsqueeze(0).expand(q.shape[1], q.shape[0], -1)
			
			# topk_token_indices: [H, B, k], Req_to_tokens: [B, S]
			k = self.heavy_token_num
			seq_lens = forward_batch.seq_lens  # [B]

			# 选择每个 seq 的最后 k 个 token
			start = seq_lens - k  # [B], 已保证 min_seq_len >= k
			offsets = torch.arange(k, device=seq_lens.device, dtype=seq_lens.dtype)  # [k]
			topk_token_indices = start.unsqueeze(1) + offsets.unsqueeze(0)  # [B, k]

			# 扩展到 [H, B, k]
			topk_token_indices = topk_token_indices.unsqueeze(0).expand(
				layer.tp_q_head_num, -1, -1
			)

			print(topk_token_indices)

			self._ensure_mid_buffers(
				q.shape[0],
				layer.tp_q_head_num,
				layer.qk_head_dim,
				topk_token_indices.shape[-1],
			)
			self.decode_sparse_attention_fwd(
				q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
				forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
				forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
				o.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
				req_to_token,
				topk_token_indices,
				forward_batch.seq_lens,
				max_seq_len,
				layer.scaling,
				layer.logit_cap,
				self.mid_out,
				self.mid_o_logexpsum,
				self.BLOCK_SEQ,
			)

		return o
