import pytest
import torch

from sglang.srt.mem_cache.sparsity.kernels.triton_visibility_metadata import (
    fill_triton_visibility_indices_,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Triton visibility metadata requires NVIDIA CUDA",
)


def test_triton_visibility_kernel_matches_explicit_mixed_batch_reference():
    device = torch.device("cuda")
    req_to_token = torch.tensor(
        [
            [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120],
            [200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220],
        ],
        dtype=torch.int64,
        device=device,
    )
    selected = torch.tensor([[0, 2], [-1, -1]], dtype=torch.int32, device=device)
    sparse_mask = torch.tensor([True, False], dtype=torch.bool, device=device)
    seq_lens = torch.tensor([11, 7], dtype=torch.int32, device=device)
    req_pool_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    sparse_indptr = torch.tensor([0, 7, 14], dtype=torch.int32, device=device)
    output = torch.full((18,), -1, dtype=torch.int64, device=device)

    fill_triton_visibility_indices_(
        selected_indices=selected,
        sparse_mask=sparse_mask,
        seq_lens=seq_lens,
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        sparse_indptr=sparse_indptr,
        output=output,
        page_size=4,
        max_row_tokens=8,
    )

    expected = torch.tensor(
        [100, 102, 104, 106, 116, 118, 120, 200, 202, 204, 206, 208, 210, 212],
        dtype=torch.int64,
        device=device,
    )
    torch.testing.assert_close(output[:14], expected)
    torch.testing.assert_close(output[14:], torch.full((4,), -1, device=device))
