from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import torch

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


def verify_get_one(store, vocab_table: torch.Tensor, layer_id: int, indices: List[int]) -> None:
	for idx in indices:
		got = store.get_one(idx, layer_id, device=torch.device("cpu")).cpu()
		expected = vocab_table[idx]
		if not torch.equal(got, expected):
			raise AssertionError(f"get_one mismatch at index {idx}")


def verify_get_many(store, vocab_table: torch.Tensor, layer_id: int, indices: torch.Tensor) -> None:
	got = store.get_many(indices, layer_id, device=torch.device("cpu")).cpu()
	expected = vocab_table[indices]
	if not torch.equal(got, expected):
		raise AssertionError("get_many mismatch")


def main() -> None:
	parser = argparse.ArgumentParser(description="Engram store microbenchmark")
	parser.add_argument("--backend", type=str, default="mooncake", choices=["mooncake", "cxl_shm"])
	parser.add_argument("--vocab-size", type=int, default=1024)
	parser.add_argument("--embedding-dim", type=int, default=64)
	parser.add_argument("--layer-id", type=int, default=0)
	parser.add_argument("--num-indices", type=int, default=128)
	parser.add_argument("--seed", type=int, default=1234)
	parser.add_argument("--repeat", type=int, default=10)
	args = parser.parse_args()

	vocab_table = build_vocab_table(args.vocab_size, args.embedding_dim)

	if args.backend == "mooncake":
		store = MooncakeEngramStore(
			embedding_dim=args.embedding_dim,
			vocab_size=args.vocab_size,
			layer_id=args.layer_id,
			dtype=torch.float16,
			config=MooncakeStoreConfig.from_env(),
		)
	else:
		store = CxlEngramStore(
			embedding_dim=args.embedding_dim,
			vocab_size=args.vocab_size,
			layer_id=args.layer_id,
			engram_layer_ids=[args.layer_id],
			dtype=torch.float16,
			config=CxlStoreConfig.from_env(),
		)

	store.put_sharded(vocab_table)

	for i in range(args.repeat):
		indices = sample_indices(args.vocab_size, args.num_indices, args.seed + i)
		verify_get_one(store, vocab_table, args.layer_id, indices[: min(8, len(indices))].tolist())
		verify_get_many(store, vocab_table, args.layer_id, indices)

	print(f"✅ put_sharded/get_one/get_many correctness verified (repeat={args.repeat})")

	time.sleep(20)

	if args.backend == "mooncake":
		store.close()
	else:
		store.close()


if __name__ == "__main__":
	main()
