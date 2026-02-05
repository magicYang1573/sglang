from __future__ import annotations

import argparse
import math
import time
from typing import Dict

import torch


DTYPE_MAP: Dict[str, torch.dtype] = {
	"float16": torch.float16,
	"float32": torch.float32,
	"float64": torch.float64,
}


def build_array(vocab_size: int, embedding_dim: int, dtype: torch.dtype) -> torch.Tensor:
	data = torch.arange(vocab_size * embedding_dim, dtype=dtype, device="cpu")
	return data.view(vocab_size, embedding_dim)


def sample_indices(vocab_size: int, num_indices: int, seed: int) -> torch.Tensor:
	g = torch.Generator(device="cpu")
	g.manual_seed(seed)
	return torch.randint(0, vocab_size, (num_indices,), generator=g, dtype=torch.long)


def get_many(table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
	return table.index_select(0, indices)


def main() -> None:
	parser = argparse.ArgumentParser(description="CPU DRAM array latency microbenchmark")
	parser.add_argument("--vocab-size", type=int, default=1024)
	parser.add_argument("--embedding-dim", type=int, default=64)
	parser.add_argument("--num-indices", type=int, default=128)
	parser.add_argument("--seed", type=int, default=1234)
	parser.add_argument("--repeat", type=int, default=10)
	parser.add_argument("--warmup", type=int, default=5)
	parser.add_argument("--dtype", type=str, default="float16", choices=sorted(DTYPE_MAP.keys()))
	parser.add_argument("--num-threads", type=int, default=0, help="0 keeps default torch threads")
	args = parser.parse_args()

	if args.num_threads > 0:
		torch.set_num_threads(args.num_threads)

	dtype = DTYPE_MAP[args.dtype]
	table = build_array(args.vocab_size, args.embedding_dim, dtype)

	for i in range(args.warmup):
		indices = sample_indices(args.vocab_size, args.num_indices, args.seed + i)
		_ = get_many(table, indices)

	latencies = []
	for i in range(args.repeat):
		indices = sample_indices(args.vocab_size, args.num_indices, args.seed + args.warmup + i)
		start = time.perf_counter()
		output = get_many(table, indices)
		end = time.perf_counter()
		latencies.append(end - start)

		print(f"index_select耗时: {latencies[-1]:.6f} 秒")

	avg_latency = sum(latencies) / len(latencies)
	max_latency = max(latencies)
	sorted_latencies = sorted(latencies)
	p90_index = max(0, math.ceil(0.9 * len(sorted_latencies)) - 1)
	p90_latency = sorted_latencies[p90_index]

	print(f"index_select平均耗时: {avg_latency * 1000.0:.6f} ms")
	print(f"index_select最大耗时: {max_latency * 1000.0:.6f} ms")
	print(f"index_select P90耗时: {p90_latency * 1000.0:.6f} ms")


if __name__ == "__main__":
	main()
