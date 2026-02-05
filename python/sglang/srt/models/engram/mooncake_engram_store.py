from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Optional, Dict, Union

import numpy as np
import torch

try:
    from mooncake.store import MooncakeDistributedStore
except Exception:  # pragma: no cover - optional dependency
    MooncakeDistributedStore = None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


@dataclass
class MooncakeStoreConfig:
    local_hostname: str = "localhost"
    metadata_server: str = "http://127.0.0.1:8080/metadata"
    global_segment_size: int = 16 * 1024 * 1024 * 1024
    local_buffer_size: int = 4 * 1024 * 1024 * 1024
    protocol: str = "tcp"
    device_name: str = ""
    master_server_address: str = "127.0.0.1:50051"
    server_id: int = 0
    num_servers: int = 1
    vocab_block_size: int = 64

    @staticmethod
    def from_env(base: Optional["MooncakeStoreConfig"] = None) -> "MooncakeStoreConfig":
        base = base or MooncakeStoreConfig()
        return MooncakeStoreConfig(
            local_hostname=os.getenv("MOONCAKE_LOCAL_HOSTNAME", base.local_hostname),
            metadata_server=os.getenv(
                "MOONCAKE_METADATA_SERVER", base.metadata_server
            ),
            global_segment_size=_env_int(
                "MOONCAKE_GLOBAL_SEGMENT_SIZE", base.global_segment_size
            ),
            local_buffer_size=_env_int(
                "MOONCAKE_LOCAL_BUFFER_SIZE", base.local_buffer_size
            ),
            protocol=os.getenv("MOONCAKE_PROTOCOL", base.protocol),
            device_name=os.getenv("MOONCAKE_DEVICE_NAME", base.device_name),
            master_server_address=os.getenv(
                "MOONCAKE_MASTER_SERVER", base.master_server_address
            ),
            server_id=_env_int("MOONCAKE_SERVER_ID", base.server_id),
            num_servers=_env_int("MOONCAKE_NUM_SERVERS", base.num_servers),
            vocab_block_size=_env_int("MOONCAKE_VOCAB_BLOCK_SIZE", base.vocab_block_size),
        )


def create_mooncake_store(config: MooncakeStoreConfig) -> "MooncakeDistributedStore":
    if MooncakeDistributedStore is None:
        raise ImportError("Mooncake is not installed. Please install 'mooncake'.")
    print("Creating Mooncake Distributed Store with config:", config)
    store = MooncakeDistributedStore()
    ret = store.setup(
        config.local_hostname,
        config.metadata_server,
        config.global_segment_size,
        config.local_buffer_size,
        config.protocol,
        config.device_name,
        config.master_server_address,
    )
    if ret != 0:
        raise RuntimeError(f"Mooncake store setup failed with code {ret}")
    return store


_GLOBAL_STORE: Optional["MooncakeDistributedStore"] = None

def get_mooncake_store(config: MooncakeStoreConfig) -> "MooncakeDistributedStore":
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = create_mooncake_store(config)
        return _GLOBAL_STORE
    return _GLOBAL_STORE


def close_mooncake_store() -> None:
    global _GLOBAL_STORE
    if _GLOBAL_STORE is not None:
        _GLOBAL_STORE.close()
    _GLOBAL_STORE = None


class MooncakeEngramStore:
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        layer_id: int,
        dtype: torch.dtype = torch.float16,
        config: Optional[MooncakeStoreConfig] = None,
    ) -> None:
        self.config = MooncakeStoreConfig.from_env(config)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.layer_id = layer_id
        self.dtype = dtype
        self.store = get_mooncake_store(self.config)
        self.server_id = int(self.config.server_id)
        self.num_servers = max(1, int(self.config.num_servers))
        self.vocab_block_size = max(1, int(self.config.vocab_block_size))

    def _block_key(self, block_id: int, layer_id: int) -> str:
        return f"{layer_id}_{block_id}"

    def _block_id(self, index: int) -> int:
        return index // self.vocab_block_size

    def _block_offset(self, index: int) -> int:
        return index % self.vocab_block_size

    def _block_len(self, block_id: int) -> int:
        start = block_id * self.vocab_block_size
        return max(0, min(self.vocab_block_size, self.vocab_size - start))

    def _wait_key_data(
        self,
        keys,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.05,
    ) -> None:
        if isinstance(keys, str):
            keys = [keys]

        pending = list(keys)
        start_times = {k: time.time() for k in pending}
        while pending:
            still_pending = []
            batch_size = 16384
            for start in range(0, len(pending), batch_size):
                chunk_keys = pending[start : start + batch_size]
                chunk_batch = self.store.get_batch(chunk_keys)
                for k, data in zip(chunk_keys, chunk_batch):
                    if data is not None and len(data) > 0:
                        continue
                    if time.time() - start_times[k] > timeout_s:
                        raise TimeoutError(f"Timeout waiting for key data: {k}")
                    still_pending.append(k)
            pending = still_pending
            if pending:
                time.sleep(poll_interval_s)


    def put_sharded(self, vocab_table: torch.Tensor) -> None:
        if isinstance(vocab_table, torch.Tensor):
            if vocab_table.ndim != 2:
                raise ValueError("vocab_table must be 2D")
            if vocab_table.shape[0] != self.vocab_size or vocab_table.shape[1] != self.embedding_dim:
                raise ValueError("vocab_table shape must match (vocab_size, embedding_dim)")
            table = vocab_table.detach()
            if table.dtype != torch.float16:
                table = table.to(dtype=torch.float16)
            if table.device.type != "cpu":
                table = table.to("cpu")
            table_np = table.contiguous().numpy()
        else:
            raise TypeError("vocab_table must be a torch.Tensor")
        keys = []
        values = []
        num_blocks = (self.vocab_size + self.vocab_block_size - 1) // self.vocab_block_size
        for block_id in range(num_blocks):
            if block_id % self.num_servers != self.server_id:
                continue
            block_len = self._block_len(block_id)
            if block_len <= 0:
                continue
            block_start = block_id * self.vocab_block_size
            block_end = block_start + block_len
            data = table_np[block_start:block_end]
            values.append(data.tobytes())
            keys.append(self._block_key(block_id, self.layer_id))
        if keys:
            batch_size = 16384
            for start in range(0, len(keys), batch_size):
                chunk_keys = keys[start : start + batch_size]
                chunk_values = values[start : start + batch_size]
                ret = self.store.put_batch(chunk_keys, chunk_values)
                if ret != 0:
                    raise RuntimeError(f"Mooncake PUT_BATCH failed with code {ret}")
        
        self._sync_all_sharded()

    def _sync_all_sharded(self) -> None:
        # ensure all the distributed data is stored into Mooncake
        # A simple control flag to sync is not enough, because 'put' is not ordered, much check all the data before inference
        num_blocks = (self.vocab_size + self.vocab_block_size - 1) // self.vocab_block_size
        all_keys = []
        for block_id in range(num_blocks):
            key = self._block_key(block_id, self.layer_id)
            block_len = self._block_len(block_id)
            all_keys.append(key)
        self._wait_key_data(
            all_keys,
            timeout_s=20.0,
        )

    def get_one(self, index: int, layer_id: int, device: torch.device) -> torch.Tensor:
        block_id = self._block_id(index)
        offset = self._block_offset(index)
        data = self.store.get(self._block_key(block_id, layer_id))
        required = (offset + 1) * self.embedding_dim * 2
        if not data or len(data) < required:
            print("Warning: get empty data from mooncake store for index ", index, " layer_id ", layer_id)
            vec = np.zeros((self.embedding_dim,), dtype=np.float16)
        else:
            base = offset * self.embedding_dim
            vec = np.frombuffer(data, dtype=np.float16, count=self.embedding_dim, offset=base * 2).copy()
        tensor = torch.from_numpy(vec).to(device=device, dtype=self.dtype)
        return tensor

    def get_many(self, indices: torch.Tensor, layer_id: int, device: torch.device) -> torch.Tensor:
        flat = indices.reshape(-1).to("cpu")
        block_ids = [self._block_id(int(idx)) for idx in flat.tolist()]
        unique_block_ids = list(dict.fromkeys(block_ids))
        keys = [self._block_key(bid, layer_id) for bid in unique_block_ids]
        batch = self.store.get_batch(keys)
        block_data = {bid: data for bid, data in zip(unique_block_ids, batch)}
        out = np.zeros((len(flat), self.embedding_dim), dtype=np.float16)
        for i, idx in enumerate(flat.tolist()):
            block_id = self._block_id(int(idx))
            offset = self._block_offset(int(idx))
            data = block_data.get(block_id)
            required = (offset + 1) * self.embedding_dim * 2
            if data and len(data) >= required:
                base = offset * self.embedding_dim
                out[i] = np.frombuffer(data, dtype=np.float16, count=self.embedding_dim, offset=base * 2)
            else:
                print("Warning: get empty data from mooncake store for index ", idx, " layer_id ", layer_id)
        return torch.from_numpy(out).to(device=device, dtype=self.dtype).view(*indices.shape, self.embedding_dim)
    

    def close(self) -> None:
        close_mooncake_store()
