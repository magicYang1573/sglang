"""Micro-test for FlashInfer CUDA Graph paged-kv indices mutability.

This script intentionally does not depend on SGLang server state.  It verifies
whether ``BatchDecodeWithPagedKVCacheWrapper(use_cuda_graph=True)`` reads the
current values of the user-provided ``paged_kv_indices_buffer`` during graph
replay.

Run manually on a CUDA machine with FlashInfer installed:

    python3 test/srt/test_flashinfer_cudagraph_kv_indices_micro.py

Expected behavior: replay output changes when ``kv_indices_buf[0]`` changes.
"""

from __future__ import annotations

import math

import torch


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    max_diff = (actual.float() - expected.float()).abs().max().item()
    print(f"{name}: max_diff={max_diff:.6g}")
    if not torch.allclose(actual.float(), expected.float(), atol=1e-2, rtol=1e-2):
        raise AssertionError(f"{name} mismatch: max_diff={max_diff}")


def main() -> None:
    import flashinfer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this micro-test.")

    device = "cuda"
    dtype = torch.bfloat16

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    batch_size = 1
    num_qo_heads = 1
    num_kv_heads = 1
    head_dim = 128
    page_size = 1
    num_pages = 2

    q = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=dtype, device=device)
    q[0, 0, 0] = 1

    k_cache = torch.zeros(
        (num_pages, page_size, num_kv_heads, head_dim), dtype=dtype, device=device
    )
    v_cache = torch.zeros_like(k_cache)

    # Make both pages have the same key score, but different values.
    k_cache[0, 0, 0, 0] = 1
    v_cache[0, 0, 0, 0] = 10
    k_cache[1, 0, 0, 0] = 1
    v_cache[1, 0, 0, 0] = 20

    kv_indptr_buf = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indices_buf = torch.tensor([0], dtype=torch.int32, device=device)
    kv_last_page_len_buf = torch.tensor([1], dtype=torch.int32, device=device)

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=kv_indptr_buf,
        paged_kv_indices_buffer=kv_indices_buf,
        paged_kv_last_page_len_buffer=kv_last_page_len_buf,
    )

    wrapper.plan(
        kv_indptr_buf,
        kv_indices_buf,
        kv_last_page_len_buf,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
        data_type=dtype,
    )

    out_buf = torch.empty_like(q)
    expected_page0 = v_cache[0, 0].transpose(0, 1).reshape_as(q)
    expected_page1 = v_cache[1, 0].transpose(0, 1).reshape_as(q)

    # Eager sanity.
    kv_indices_buf[0] = 0
    out_buf.copy_(wrapper.run(q, (k_cache, v_cache)))
    torch.cuda.synchronize()
    _assert_close("eager_page0", out_buf, expected_page0)

    kv_indices_buf[0] = 1
    out_buf.copy_(wrapper.run(q, (k_cache, v_cache)))
    torch.cuda.synchronize()
    _assert_close("eager_page1", out_buf, expected_page1)

    # Capture with the fixed buffers.
    kv_indices_buf[0] = 0
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        out_buf.copy_(wrapper.run(q, (k_cache, v_cache)))

    kv_indices_buf[0] = 0
    graph.replay()
    torch.cuda.synchronize()
    _assert_close("graph_page0", out_buf, expected_page0)

    kv_indices_buf[0] = 1
    graph.replay()
    torch.cuda.synchronize()
    _assert_close("graph_page1", out_buf, expected_page1)

    if not math.isclose(float(out_buf[0, 0, 0].item()), 20.0, rel_tol=0, abs_tol=0.1):
        raise AssertionError(
            "CUDA Graph replay did not observe the updated kv_indices buffer."
        )

    print("PASS: FlashInfer CUDA Graph replay reads updated kv_indices buffer.")


if __name__ == "__main__":
    main()
