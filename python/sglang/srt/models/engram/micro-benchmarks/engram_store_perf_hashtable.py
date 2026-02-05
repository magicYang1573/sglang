from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List

import torch
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from mooncake_engram_store import MooncakeEngramStore, MooncakeStoreConfig, close_mooncake_store
from cxl_utils.cxl_engram_store import CxlEngramStore, CxlStoreConfig, close_cxl_store


def build_vocab_table(vocab_size: int, embedding_dim: int) -> torch.Tensor:
	data = torch.arange(vocab_size * embedding_dim, dtype=torch.float16)
	return data.view(vocab_size, embedding_dim)


def sample_indices(vocab_size: int, num_indices: int, seed: int) -> torch.Tensor:
	g = torch.Generator(device="cpu")
	g.manual_seed(seed)
	return torch.randint(0, vocab_size, (num_indices,), generator=g, dtype=torch.long)


def get_many(store, layer_id: int, indices: torch.Tensor) -> None:
	got = store.get_many(indices, layer_id, device=torch.device("cpu")).cpu()


def count_unique_indices(indices: torch.Tensor | List[int]) -> int:
	if isinstance(indices, torch.Tensor):
		return torch.unique(indices).numel()
	return len(set(indices))


def main() -> None:
	parser = argparse.ArgumentParser(description="Engram store microbenchmark")
	parser.add_argument("--backend", type=str, default="mooncake", choices=["mooncake", "cxl_shm"])
	parser.add_argument("--vocab-size", type=int, default=1024)
	parser.add_argument("--embedding-dim", type=int, default=64)
	parser.add_argument("--num-indices", type=int, default=128)
	parser.add_argument("--seed", type=int, default=1234)
	parser.add_argument("--repeat", type=int, default=10)
	args = parser.parse_args()

	vocab_table = build_vocab_table(args.vocab_size, args.embedding_dim)

	layer_ids = [0]
	if args.backend == "mooncake":
		store = MooncakeEngramStore(
			embedding_dim=args.embedding_dim,
			vocab_size=args.vocab_size,
			layer_id=layer_ids[0],
			dtype=torch.float16,
			config=MooncakeStoreConfig.from_env(),
		)
	else:
		store = CxlEngramStore(
			embedding_dim=args.embedding_dim,
			vocab_size=args.vocab_size,
			layer_id=layer_ids[0],
			engram_layer_ids=layer_ids,
			dtype=torch.float16,
			config=CxlStoreConfig.from_env(),
		)

	store.put_sharded(vocab_table)
	latencies = []

	for i in range(args.repeat):
		indices = sample_indices(args.vocab_size, args.num_indices, args.seed + i)
		unique_get_many = count_unique_indices(indices)

		start = time.perf_counter()
		get_many(store, layer_ids[0], indices)
		end = time.perf_counter()
		duration_s = end - start
		latencies.append(duration_s)

		# print(f"get_many耗时: {duration_s:.6f} 秒, unique_hash_items: {unique_get_many}")

	avg_latency = sum(latencies) / len(latencies)
	max_latency = max(latencies)
	sorted_latencies = sorted(latencies)
	p90_index = max(0, math.ceil(0.9 * len(sorted_latencies)) - 1)
	p90_latency = sorted_latencies[p90_index]
	print(f"get_many平均耗时: {avg_latency * 1000.0:.6f} ms")
	print(f"get_many最大耗时: {max_latency * 1000.0:.6f} ms")
	print(f"get_many P90耗时: {p90_latency * 1000.0:.6f} ms")


	time.sleep(10)

	if args.backend == "mooncake":
		store.close()
	else:
		store.close()


if __name__ == "__main__":
	main()
