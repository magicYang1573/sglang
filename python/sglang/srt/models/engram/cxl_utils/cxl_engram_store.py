from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time
import threading
from typing import Optional, Dict, List, Union

import torch
import numpy as np


def _env_int(name: str, default: int) -> int:
	value = os.getenv(name)
	if value is None:
		return default
	return int(value)


def _env_bool(name: str, default: bool) -> bool:
	value = os.getenv(name)
	if value is None:
		return default
	return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class CxlStoreConfig:
	dev_path: str = "/dev/dax0.0"
	map_bytes: int = 8 * 1024 * 1024 * 1024
	offset: int = 0
	register_cuda: bool = False
	gpu_id: int = 0
	build_directory: Optional[str] = None
	verbose_build: bool = False
	init_engram: bool = True

	@staticmethod
	def from_env(base: Optional["CxlStoreConfig"] = None) -> "CxlStoreConfig":
		base = base or CxlStoreConfig()
		register_cuda_default = base.register_cuda or torch.cuda.is_available()
		return CxlStoreConfig(
			dev_path=os.getenv("CXL_DEV_PATH", base.dev_path),
			map_bytes=_env_int("CXL_MAP_BYTES", base.map_bytes),
			offset=_env_int("CXL_OFFSET", base.offset),
			register_cuda=_env_bool("CXL_REGISTER_CUDA", register_cuda_default),
			gpu_id=_env_int("CXL_GPU_ID", base.gpu_id),
			build_directory=os.getenv("CXL_BUILD_DIR", base.build_directory or ""),
			verbose_build=_env_bool("CXL_VERBOSE_BUILD", base.verbose_build),
			init_engram=_env_bool("CXL_INIT_ENGRAM", base.init_engram),
		)


_GLOBAL_EXT = None
_GLOBAL_CONFIG: Optional[CxlStoreConfig] = None
_GLOBAL_INIT = False
_ALLOCATOR_LOCK = threading.Lock()
_CONTROL_REGION_BYTES = 64
_READY_FLAG_OFFSET = 0
_ALLOCATOR_OFFSET = _CONTROL_REGION_BYTES
_ALLOCATIONS: Dict[int, int] = {}


def _load_cxl_extension(config: CxlStoreConfig):
	global _GLOBAL_EXT
	if _GLOBAL_EXT is not None:
		return _GLOBAL_EXT

	try:
		import cxl_mem_ext  # type: ignore
		_GLOBAL_EXT = cxl_mem_ext
		return _GLOBAL_EXT
	except Exception:
		from torch.utils.cpp_extension import load

		root = Path(__file__).resolve().parent / "pytorch_cxl_test"
		build_dir = config.build_directory or str(root / "build")
		Path(build_dir).mkdir(parents=True, exist_ok=True)

		_GLOBAL_EXT = load(
			name="cxl_mem_ext",
			sources=[
				str(root / "cxl_mem.cpp"),
				str(root / "cxl_mem_pybind.cpp"),
			],
			extra_cflags=["-O3", "-std=c++17", "-mclflushopt", "-mclwb", "-msse4.1", "-fopenmp"],
			extra_cuda_cflags=["-O3", "-std=c++17"],
			build_directory=build_dir,
			verbose=config.verbose_build,
			with_cuda=True,
		)
		return _GLOBAL_EXT


def _init_cxl(config: CxlStoreConfig):
	global _GLOBAL_INIT, _GLOBAL_CONFIG
	if _GLOBAL_INIT:
		if _GLOBAL_CONFIG != config:
			raise RuntimeError("CXL store already initialized with a different config")
		return _GLOBAL_EXT

	ext = _load_cxl_extension(config)
	ok = ext.cxl_init(
		dev_path=config.dev_path,
		map_bytes=int(config.map_bytes),
		offset=int(config.offset),
		register_cuda=bool(config.register_cuda),
		gpu_id=int(config.gpu_id),
	)
	if not ok:
		raise RuntimeError("cxl_init failed; check dev_path permissions and size")
	_GLOBAL_INIT = True
	_GLOBAL_CONFIG = config
	return ext


def _align(value: int, alignment: int = 64) -> int:
	return (value + alignment - 1) // alignment * alignment


def _allocate_bytes(key: int, size_bytes: int, config: CxlStoreConfig) -> int:
	global _ALLOCATOR_OFFSET
	with _ALLOCATOR_LOCK:
		if key in _ALLOCATIONS:
			return _ALLOCATIONS[key]
		base = _align(_ALLOCATOR_OFFSET, 64)
		end = base + size_bytes
		if end > config.map_bytes:
			raise MemoryError(
				f"CXL map_bytes too small: need {end} bytes, have {config.map_bytes}"
			)
		_ALLOCATIONS[key] = base
		_ALLOCATOR_OFFSET = end
		return base


def close_cxl_store() -> None:
	global _GLOBAL_INIT, _GLOBAL_CONFIG, _GLOBAL_EXT, _ALLOCATOR_OFFSET, _ALLOCATIONS

	if _GLOBAL_INIT and _GLOBAL_EXT is not None:
		_GLOBAL_EXT.cxl_close()
	_GLOBAL_INIT = False
	_GLOBAL_CONFIG = None
	_GLOBAL_EXT = None
	_ALLOCATOR_OFFSET = _CONTROL_REGION_BYTES
	_ALLOCATIONS = {}


class CxlEngramStore:
	"""CXL-backed embedding store compatible with MooncakeEmbeddingStore interface."""

	def __init__(
		self,
		embedding_dim: int,
		vocab_size: int,
		layer_id: int,
		engram_layer_ids: List[int],
		dtype: torch.dtype = torch.float16,
		config: Optional[CxlStoreConfig] = None,
	) -> None:
		self.config = CxlStoreConfig.from_env(config)
		self.embedding_dim = int(embedding_dim)
		self.vocab_size = int(vocab_size)
		self.layer_id = int(layer_id)
		self.engram_layer_ids = list(engram_layer_ids)
		self.last_layer_id = self.engram_layer_ids[-1] if self.engram_layer_ids else self.layer_id
		self.dtype = dtype
		self.storage_dtype = torch.float16
		self.ext = _init_cxl(self.config)
		self._ready_checked = False

		self.row_bytes = self.embedding_dim * 2  # float16 storage
		total_bytes = self.vocab_size * self.row_bytes
		self.base_offset = _allocate_bytes(self.layer_id, total_bytes, self.config)

		self.global_buffer = torch.empty(
			(10240, self.embedding_dim), dtype=self.storage_dtype, device="cpu"
		)

	def _offset_for_index(self, index: int) -> int:
		return self.base_offset + index * self.row_bytes

	def put_sharded(self, vocab_table: torch.Tensor) -> None:
		if not self.config.init_engram:
			self._ensure_ready()
			return
		# Single-process initialization: write the entire table once.
		if isinstance(vocab_table, torch.Tensor):
			if vocab_table.ndim != 2:
				raise ValueError("vocab_table must be 2D")
			if vocab_table.shape[0] != self.vocab_size or vocab_table.shape[1] != self.embedding_dim:
				raise ValueError("vocab_table shape must match (vocab_size, embedding_dim)")
			data = vocab_table.detach()
			if data.dtype != self.storage_dtype:
				data = data.to(dtype=self.storage_dtype)
			if data.device.type != "cpu":
				data = data.to("cpu")
			data = data.contiguous()
		else:
			raise TypeError("vocab_table must be a torch.Tensor")
		self.ext.tensor_to_cxl(data, offset=int(self.base_offset))
		if self.layer_id == self.last_layer_id:
			self._set_ready_flag()

	def get_one(self, index: int, layer_id: int, device: torch.device) -> torch.Tensor:
		if index < 0 or index >= self.vocab_size:
			vec = torch.zeros((self.embedding_dim,), dtype=self.dtype, device=device)
			return vec

		out = torch.empty((self.embedding_dim,), dtype=self.storage_dtype, device=device)
		self.ext.cxl_to_tensor_noflush(out, offset=int(self._offset_for_index(int(index))))
		if self.dtype != self.storage_dtype:
			out = out.to(dtype=self.dtype)
		return out

	def get_many(self, indices: torch.Tensor, layer_id: int, device: torch.device) -> torch.Tensor:
		num_elements = indices.numel()
		out = self.global_buffer[:num_elements]
		
		offsets = indices * self.row_bytes + self.base_offset
		if num_elements >= 1:
			self.ext.cxl_to_tensor_noflush_parallel_contiguous_dst(
				out,
				int(self.row_bytes),
				offsets,
			)

		if self.dtype != self.storage_dtype:
			out = out.to(dtype=self.dtype)
			
		return out.view(*indices.shape, self.embedding_dim)

	def close(self) -> None:
		self._set_ready_flag(value=0)
		close_cxl_store()

	def _set_ready_flag(self, value=1) -> None:
		flag = torch.tensor([value], dtype=torch.int32)
		self.ext.tensor_to_cxl(flag, offset=_READY_FLAG_OFFSET)

	def _read_ready_flag(self) -> int:
		flag = torch.zeros((1,), dtype=torch.int32)
		self.ext.cxl_to_tensor(flag, offset=_READY_FLAG_OFFSET)
		return int(flag.item())

	def _ensure_ready(self) -> None:
		if self._ready_checked or self.config.init_engram:
			return
		timeout_s = _env_int("CXL_READY_TIMEOUT_S", 0)
		poll_s = float(os.getenv("CXL_READY_POLL_S", "0.1"))
		start = time.time()
		while True:
			if self._read_ready_flag() == 1:
				print("ready")
				self._ready_checked = True
				return
			if timeout_s > 0 and (time.time() - start) > timeout_s:
				raise TimeoutError("CXL Engram ready flag not set within timeout")
			time.sleep(poll_s)
