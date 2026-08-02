import dataclasses
from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.mem_cache.sparsity.cuda_graph_support import (
    create_cuda_graph_runtime_provider,
)
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_breakable_cuda_graph_buffers_dataclass_outputs():
    backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
    output = LogitsProcessorOutput(
        next_token_logits=torch.arange(6, dtype=torch.float32).view(2, 3),
        hidden_states=torch.arange(8, dtype=torch.float32).view(2, 4),
        customized_info={"source": ["decode"]},
    )

    assert backend._output_rows(output, cap=4) == 2
    buffer = backend._alloc_full_buffer(output, size=4)
    backend._copy_output_to_buffer(output, buffer, num_tokens=2)
    sliced = backend._slice_output(buffer, num_tokens=2)
    torch.testing.assert_close(sliced.next_token_logits, output.next_token_logits)
    torch.testing.assert_close(sliced.hidden_states, output.hidden_states)
    assert sliced.customized_info == {"source": ["decode"]}


class _RuntimeExtension:
    def cuda_graph_capture_variants(self):
        return (256, 1024)

    def select_cuda_graph_variant(self, forward_batch):
        return forward_batch.supported, forward_batch.variant


def test_decode_cuda_graph_runtime_variant_is_part_of_shape_key():
    runner = object.__new__(DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(
        cuda_graph_runtime_extension=_RuntimeExtension()
    )
    forward_batch = SimpleNamespace(supported=True, variant=1024)

    assert runner._runtime_graph_capture_variants() == (256, 1024)
    assert runner._select_captured_runtime_graph_variant(forward_batch) == (
        True,
        1024,
    )
    assert runner._make_graph_key(8, runtime_variant=256) != runner._make_graph_key(
        8, runtime_variant=1024
    )


@dataclasses.dataclass(frozen=True)
class _Context:
    attn_backend: str
    kv_sparsity_controller: object = None


def _provider():
    policy = SimpleNamespace(
        page_budget=64,
        device=torch.device("cuda"),
        capabilities=SimpleNamespace(supports_cuda_graph=True),
    )
    controller = SimpleNamespace(
        policy=policy,
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.empty((2, 40 * 1024), dtype=torch.int32)
        ),
        config=SimpleNamespace(
            policy="quest",
            page_size=16,
            policy_config={"cuda_graph_context_buckets": [8 * 1024, 32 * 1024]},
        ),
    )
    return controller, create_cuda_graph_runtime_provider(controller)


def test_quest_cuda_graph_provider_selects_context_buckets_and_context():
    controller, provider = _provider()
    assert provider.cuda_graph_capture_variants() == (512, 2048, 2560)
    assert provider.select_cuda_graph_variant(
        SimpleNamespace(seq_lens_cpu=[1, 8192])
    ) == (True, 512)
    assert provider.select_cuda_graph_variant(
        SimpleNamespace(seq_lens_cpu=torch.tensor([8193, 32768]))
    ) == (True, 2048)
    assert provider.select_cuda_graph_variant(
        SimpleNamespace(seq_lens_cpu=[40 * 1024 + 1])
    ) == (False, None)

    capture_batch = SimpleNamespace()
    provider.prepare_cuda_graph_capture(capture_batch, 512)
    assert capture_batch.kv_sparsity_page_capacity == 512
    context = provider.prepare_cuda_graph_capture_context(_Context("fa3"))
    assert context.kv_sparsity_controller is controller


def test_streaming_llm_cuda_graph_provider_uses_one_variant_and_context():
    controller = SimpleNamespace(
        policy=SimpleNamespace(
            device=torch.device("cuda"),
            capabilities=SimpleNamespace(supports_cuda_graph=True),
        ),
        config=SimpleNamespace(policy="streaming_llm", page_size=1, policy_config={}),
    )
    provider = create_cuda_graph_runtime_provider(controller)

    assert provider.cuda_graph_capture_variants() == (None,)
    assert provider.select_cuda_graph_variant(
        SimpleNamespace(seq_lens_cpu=[1, 32 * 1024])
    ) == (True, None)
    capture_batch = SimpleNamespace()
    provider.prepare_cuda_graph_capture(capture_batch, None)
    assert not hasattr(capture_batch, "kv_sparsity_page_capacity")
    context = provider.prepare_cuda_graph_capture_context(_Context("fa3"))
    assert context.kv_sparsity_controller is controller
