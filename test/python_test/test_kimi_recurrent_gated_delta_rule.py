# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A3 Kimi recurrence: FP32 state, per-key gate, indexed cache and padding."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    state: torch.Tensor,
    lengths: tuple[int, ...],
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent CPU delta-rule recurrence in [H, V, K] state layout."""
    output = torch.zeros_like(v)
    next_state = state.clone()
    offset = 0
    for length in lengths:
        if length == 0:
            continue
        slot = int(indices[offset])
        if slot < 0:
            offset += length
            continue
        current = next_state[slot].float().clone()
        for token in range(offset, offset + length):
            decay = gate[token].float().exp().unsqueeze(-2)
            current = current * decay
            delta = v[token].float() - (
                current * k[token].float().unsqueeze(-2)
            ).sum(-1)
            delta = delta * beta[token].float().unsqueeze(-1)
            current = current + delta.unsqueeze(-1) * k[token].float().unsqueeze(-2)
            output[token] = (
                current * (q[token].float() * q.shape[-1] ** -0.5).unsqueeze(-2)
            ).sum(-1)
            next_state[int(indices[token])] = current.to(state.dtype)
        offset += length
    return output, next_state


def _inputs(
    lengths: tuple[int, ...],
    state_dtype: torch.dtype,
    padding_row: int | None,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20261008)
    tokens, heads, dim = sum(lengths), 2, 128
    q = torch.randn(tokens, heads, dim).to(torch.bfloat16)
    k = torch.randn_like(q)
    q = (q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)).to(q.dtype)
    k = (k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)).to(k.dtype)
    v = torch.randn_like(q)
    beta = torch.randn(tokens, heads).sigmoid()
    raw = torch.randn(tokens, heads, dim).to(torch.bfloat16)
    a_log = torch.randn(heads) * 0.1
    dt_bias = torch.randn(heads, dim) * 0.1
    gate = -5.0 * torch.sigmoid(a_log.exp().view(1, heads, 1) * (raw.float() + dt_bias))
    state = (torch.randn(tokens + 3, heads, dim, dim) * 0.03).to(state_dtype)
    indices = torch.arange(tokens, 0, -1, dtype=torch.int32)
    if padding_row is not None:
        start = sum(lengths[:padding_row])
        indices[start : start + lengths[padding_row]] = -1
    return q, k, v, beta, gate, state, indices


def _call(
    tensors: tuple[torch.Tensor, ...],
    kernel_lengths: torch.Tensor,
) -> torch.Tensor:
    q, k, v, beta, gate, state, indices = tensors
    return custom_ops.recurrent_gated_delta_rule_npu(
        q, k, v, state, beta, q.shape[-1] ** -0.5,
        kernel_lengths, indices, None, None, gate,
    )


@pytest.mark.parametrize("lengths", [(1,), (1, 1), (1,) * 4, (1,) * 8, (1,) * 16, (1, 2, 1)])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("padding_row", [None, 0])
@torch.inference_mode()
def test_kimi_recurrence(
    lengths: tuple[int, ...],
    state_dtype: torch.dtype,
    padding_row: int | None,
) -> None:
    torch_npu.npu.set_device(0)
    inputs = _inputs(lengths, state_dtype, padding_row)
    expected, expected_state = _reference(*inputs[:6], lengths, inputs[6])
    device_inputs = tuple(tensor.to("npu:0") for tensor in inputs)
    kernel_lengths = torch.tensor((0, *lengths), dtype=torch.int32, device="npu:0")
    actual = _call(device_inputs, kernel_lengths)
    torch_npu.npu.synchronize()
    actual, actual_state = actual.cpu(), device_inputs[5].cpu()
    output_diff = (actual.float() - expected.float()).abs().max().item()
    state_diff = (actual_state.float() - expected_state.float()).abs().max().item()
    print(f"lengths={lengths} state={state_dtype} padding={padding_row} output_max_abs={output_diff:.8g} state_max_abs={state_diff:.8g}")
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.004)
    torch.testing.assert_close(actual_state, expected_state, rtol=0.01, atol=0.004)
    unused = set(range(inputs[5].shape[0])) - set(inputs[6][inputs[6] >= 0].tolist())
    for slot in unused:
        torch.testing.assert_close(actual_state[slot], inputs[5][slot], rtol=0, atol=0)
    if padding_row is not None:
        start = sum(lengths[:padding_row])
        assert torch.count_nonzero(actual[start : start + lengths[padding_row]]) == 0


@torch.inference_mode()
def test_kimi_recurrence_graph_replay_updates_state_indices() -> None:
    torch_npu.npu.set_device(0)
    lengths = (1, 1, 1, 1)
    inputs = _inputs(lengths, torch.float32, None)
    device_inputs = tuple(tensor.to("npu:0") for tensor in inputs)
    kernel_lengths = torch.tensor((0, *lengths), dtype=torch.int32, device="npu:0")
    for _ in range(3):
        _call(device_inputs, kernel_lengths)
    torch_npu.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = _call(device_inputs, kernel_lengths)
    for indices in (
        torch.tensor([4, 3, 2, 1], dtype=torch.int32),
        torch.tensor([-1, 2, -1, 4], dtype=torch.int32),
        torch.tensor([3, -1, 1, 2], dtype=torch.int32),
    ):
        device_inputs[5].copy_(inputs[5])
        device_inputs[6].copy_(indices)
        expected, expected_state = _reference(*inputs[:6], lengths, indices)
        graph.replay()
        torch_npu.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.01, atol=0.004)
        torch.testing.assert_close(device_inputs[5].cpu(), expected_state, rtol=0.01, atol=0.004)
