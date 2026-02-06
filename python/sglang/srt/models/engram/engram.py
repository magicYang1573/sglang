"""
================================================================================
[Engram Architecture Demo Implementation]

DISCLAIMER:
1. Demo Purpose Only: 
   This code is a demonstration version intended to illustrate the core logic and 
   data flow of the Engram module.

2. Production Readiness: 
   This implementation requires further optimization for actual production use 
   (e.g., custom CUDA kernels, distributed training support).

3. Simplifications: 
   Standard components (Normalization, Attention, MoE) and complex Hyper-connection 
   mechanisms are omitted or mocked in this version to focus exclusively on the 
   Engram module implementation.
================================================================================
"""

"""
pip install torch numpy transformers sympy
"""

## built-in
from typing import List, Optional
from dataclasses import dataclass, field
import argparse
import os
import math
import time

## third-party
from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex 

from .mooncake_engram_store import (
    MooncakeEngramStore,
    MooncakeStoreConfig,
    close_mooncake_store,
)
from .cxl_utils.cxl_engram_store import (
    CxlEngramStore,
    CxlStoreConfig,
    close_cxl_store,
)
from .local_engram_store import LocalEngramStore

@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    # engram_vocab_size: List[int] = field(default_factory=lambda: [129280*5, 129280*5])
    engram_vocab_size: List[int] = field(default_factory=lambda: [1024, 1024])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
    store_backend: str = "local"  # "mooncake", "cxl_shm", or "local"
    enable_prefetch: bool = True
    
@dataclass
class BackBoneConfig:
    hidden_size: int = 1024
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 30
    
engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()

def build_engram_config_from_cli() -> EngramConfig:
    parser = argparse.ArgumentParser(description="Engram demo config")
    parser.add_argument("--store-backend", type=str, default=None, choices=["mooncake", "cxl_shm", "local"])
    parser.add_argument("--engram-vocab-size", type=int, default=None)
    parser.add_argument("--engram-emb-size", type=int, default=None)
    parser.add_argument("--engram-head", type=int, default=None)
    args = parser.parse_args()

    cfg = EngramConfig()
    if args.store_backend:
        cfg.store_backend = args.store_backend
    if args.engram_vocab_size:
        for i in range(len(cfg.engram_vocab_size)):
            cfg.engram_vocab_size[i] = args.engram_vocab_size
    if args.engram_emb_size:
        cfg.n_embed_per_ngram = args.engram_emb_size
    if args.engram_head:
        cfg.n_head_per_ngram = args.engram_head
    return cfg

class CompressedTokenizer:
    def __init__(
        self,
        tokenizer_name_or_path,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        
        self.lookup_table, self.num_new_token = self._build_lookup_table()
    
    def __len__(self):
        return self.num_new_token
    
    def _build_lookup_table(self):
        old2new = {}
        key2new = {}          
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            
            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)
    
    def _compress(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        if valid_ids.size == 0:
            return out
        vocab_size = self.lookup_table.shape[0]
        if valid_ids.max(initial=-1) >= vocab_size:
            if engram_cfg.pad_id is not None:
                valid_ids = np.where(
                    valid_ids < vocab_size, valid_ids, int(engram_cfg.pad_id)
                )
            else:
                valid_ids = np.clip(valid_ids, 0, vocab_size - 1)
        out[pos_mask] = self.lookup_table[valid_ids]
        return out
    
    def __call__(self, input_ids):
        return self._compress(input_ids)
            
class ShortConv(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        kernel_size: int = 4, 
        dilation: int = 1, 
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation
        
        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps) 
            for _ in range(hc_mult)
        ])
        
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B,L,HC_MULT,D)
        Output: (B,L,HC_MULT,D)
        """
        B, T, G, C = x.shape
        
        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))
        
        x_norm = torch.cat(normed_chunks, dim=-1)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]

        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()
        
        return y
    
def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1

class NgramHashMapping:
    def __init__(
        self, 
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        layer_ids,
        tokenizer_name_or_path,
        pad_id,
        seed,  
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )            
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        
        self.layer_multipliers = {}

        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}
        
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1
                
                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, 
                        seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
            
        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0: return x
            shifted = np.pad(x, ((0, 0), (k, 0)),
                                mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []
        
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            
            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers

class MultiHeadEmbedding(nn.Module):
    def __init__(
        self,
        list_of_N: List[int],
        layer_id: int,
        D: int,
        vocab_table: Optional[torch.Tensor] = None,
        store_config: Optional[MooncakeStoreConfig] = None,
        cxl_config: Optional[CxlStoreConfig] = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        self.layer_id = layer_id

        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)

        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        total_N = sum(list_of_N)
        backend = engram_cfg.store_backend.lower()
        if backend == "cxl_shm":
            self.store = CxlEngramStore(
                embedding_dim=D,
                vocab_size=total_N,
                layer_id=layer_id,
                engram_layer_ids=engram_cfg.layer_ids,
                dtype=dtype,
                config=cxl_config,
            )
        elif backend == "mooncake":
            self.store = MooncakeEngramStore(
                embedding_dim=D,
                vocab_size=total_N,
                layer_id=layer_id,
                dtype=dtype,
                config=store_config,
            )
        elif backend == "local":
            self.store = LocalEngramStore(
                embedding_dim=D,
                vocab_size=total_N,
                layer_id=layer_id,
                dtype=dtype,
                device=torch.device("cpu"),
            )
        else:
            raise ValueError(f"Unknown store_backend: {engram_cfg.store_backend}")
        if vocab_table is None:
            raise ValueError("vocab_table must be provided for put_sharded")
        self.store.put_sharded(vocab_table)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        offsets = self.offsets.to(device=input_ids.device)
        shifted_input_ids = input_ids + offsets
        output = self.store.get_many(shifted_input_ids, self.layer_id, device=input_ids.device)
        return output
    
class Engram(nn.Module):
    def __init__(self, layer_id, vocab_table: Optional[torch.Tensor] = None):
        super().__init__()
        self.layer_id = layer_id
        self.enable_prefetch = engram_cfg.enable_prefetch
        self._prefetch_embeddings: Optional[torch.Tensor] = None
        self._prefetch_event: Optional[torch.cuda.Event] = None
        self._prefetch_shape: Optional[tuple[int, int]] = None
        self._prefetch_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        print(
            "[Engram] init",
            {
                "layer_id": layer_id,
                "hidden_size": backbone_config.hidden_size,
                "hc_mult": backbone_config.hc_mult,
                "vocab_size": backbone_config.vocab_size,
                "num_layers": backbone_config.num_layers,
                "max_ngram_size": engram_cfg.max_ngram_size,
                "n_embed_per_ngram": engram_cfg.n_embed_per_ngram,
                "n_head_per_ngram": engram_cfg.n_head_per_ngram,
                "engram_vocab_size": engram_cfg.engram_vocab_size,
                "kernel_size": engram_cfg.kernel_size,
                "store_backend": engram_cfg.store_backend,
            },
        )
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size = engram_cfg.max_ngram_size,
            n_embed_per_ngram = engram_cfg.n_embed_per_ngram,
            n_head_per_ngram = engram_cfg.n_head_per_ngram,
            layer_ids = engram_cfg.layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id = engram_cfg.pad_id,
            seed = engram_cfg.seed,
        )

        list_of_N = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y]

        
        print("Layer:", self.layer_id, "; Vocab size:", list_of_N)

        # random/deterministic vocab table initialization
        total_N = sum(list_of_N)
        embedding_dim = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram
        # vocab_table = torch.randn(total_N, embedding_dim, dtype=torch.float16)
        vocab_table = torch.arange(total_N, dtype=torch.float16).unsqueeze(1).expand(-1, embedding_dim)
        
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N = list_of_N,
            layer_id = self.layer_id,
            D = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
            vocab_table = vocab_table,
        )
        self.short_conv = ShortConv(
            hidden_size = backbone_config.hidden_size,
            kernel_size = engram_cfg.kernel_size,
            dilation    = engram_cfg.max_ngram_size,
            hc_mult     = backbone_config.hc_mult,
        )
        engram_hidden_size = (engram_cfg.max_ngram_size-1) * engram_cfg.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size,backbone_config.hidden_size)
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size,backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)]
        )
        self.norm1 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])

    def start_prefetch(
        self,
        input_ids: np.ndarray,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not self.enable_prefetch:
            return

        hash_input_ids = torch.from_numpy(
            self.hash_mapping.hash(input_ids)[self.layer_id]
        ).to(device=device)

        if self._prefetch_stream is not None and device.type == "cuda":
            stream = self._prefetch_stream
            with torch.cuda.stream(stream):
                embeddings = self.multi_head_embedding(hash_input_ids).flatten(
                    start_dim=-2
                )
                if embeddings.dtype != dtype:
                    embeddings = embeddings.to(dtype=dtype)
            event = torch.cuda.Event()
            event.record(stream)
            self._prefetch_event = event
            self._prefetch_embeddings = embeddings
        else:
            embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
            if embeddings.dtype != dtype:
                embeddings = embeddings.to(dtype=dtype)
            self._prefetch_event = None
            self._prefetch_embeddings = embeddings

        self._prefetch_shape = tuple(input_ids.shape[:2])

    def _consume_prefetch(self, input_ids: np.ndarray) -> Optional[torch.Tensor]:
        if not self.enable_prefetch:
            return None
        if self._prefetch_embeddings is None:
            return None
        if self._prefetch_shape is not None and tuple(input_ids.shape[:2]) != self._prefetch_shape:
            self._prefetch_embeddings = None
            self._prefetch_event = None
            self._prefetch_shape = None
            return None

        if self._prefetch_event is not None:
            torch.cuda.current_stream().wait_event(self._prefetch_event)
        embeddings = self._prefetch_embeddings
        self._prefetch_embeddings = None
        self._prefetch_event = None
        self._prefetch_shape = None
        return embeddings
    
    def forward(self,hidden_states,input_ids):
        """
        hidden_states: [B, L, HC_MULT, D]
        input_ids: [B, L]
        """
        embeddings = self._consume_prefetch(input_ids)
        if embeddings is None:
            hash_input_ids = torch.from_numpy(
                self.hash_mapping.hash(input_ids)[self.layer_id]
            ).to(device=hidden_states.device)
            embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        if embeddings.dtype != hidden_states.dtype:
            embeddings = embeddings.to(dtype=hidden_states.dtype)
        gates = []
        for hc_idx in range(backbone_config.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = hidden_states[:,:,hc_idx,:]
            normed_query = self.norm2[hc_idx](query)
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(backbone_config.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
        gates = torch.stack(gates,dim=2)
        value = gates * self.value_proj(embeddings).unsqueeze(2)
        output = value + self.short_conv(value)
        return output 



            