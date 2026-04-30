#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "converted_torch_pro_bundle" / "train_gpt_torch_loop_fullattnres_loopq_fastcuda.py"


def load_training_module():
    spec = importlib.util.spec_from_file_location("loop_fullattnres", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_case(module, dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the fused AttnRes parity check")
    if not module.TRITON_AVAILABLE:
        raise SystemExit("Triton is required for the fused AttnRes parity check")

    torch.manual_seed(1234)
    device = torch.device("cuda")
    batch, seq_len, dim, nsrc = 2, 16, 64, 7
    target = torch.randn(batch, seq_len, dim, device=device, dtype=torch.float32)
    base_values = [
        torch.randn(batch, seq_len, dim, device=device, dtype=dtype) * 0.5
        for _ in range(nsrc)
    ]

    os.environ["FUSED_ATTNRES"] = "0"
    ref = module.FullDepthMixer(dim, max_sources=32, num_loops=1, loop_specific=False, use_gate=False).to(device)
    with torch.no_grad():
        ref.attnres_query.normal_(0.0, 0.2)
        ref.attnres_bias.normal_(0.0, 0.2)
    state = {name: tensor.detach().clone() for name, tensor in ref.state_dict().items()}

    os.environ["FUSED_ATTNRES"] = "1"
    fused = module.FullDepthMixer(dim, max_sources=32, num_loops=1, loop_specific=False, use_gate=False).to(device)
    fused.load_state_dict(state)

    def eval_one(mixer: torch.nn.Module):
        values = [value.detach().clone().requires_grad_(True) for value in base_values]
        out = mixer(values)
        loss = (out.float() * target).sum()
        loss.backward()
        value_grads = [value.grad.detach().clone() for value in values]
        query_grad = mixer.attnres_query.grad.detach().clone()
        bias_grad = mixer.attnres_bias.grad.detach().clone()
        return out.detach(), value_grads, query_grad, bias_grad

    ref_out, ref_value_grads, ref_query_grad, ref_bias_grad = eval_one(ref)
    fused_out, fused_value_grads, fused_query_grad, fused_bias_grad = eval_one(fused)

    def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.float() - b.float()).abs().max().item())

    out_diff = max_abs(ref_out, fused_out)
    value_grad_diff = max(max_abs(a, b) for a, b in zip(ref_value_grads, fused_value_grads, strict=True))
    query_grad_diff = max_abs(ref_query_grad, fused_query_grad)
    bias_grad_diff = max_abs(ref_bias_grad, fused_bias_grad)
    print(
        f"dtype={dtype} out_diff={out_diff:.6g} value_grad_diff={value_grad_diff:.6g} "
        f"query_grad_diff={query_grad_diff:.6g} bias_grad_diff={bias_grad_diff:.6g}"
    )
    tol = 2e-2 if dtype == torch.bfloat16 else 2e-4
    assert out_diff < tol
    assert value_grad_diff < tol
    assert query_grad_diff < 5 * tol
    assert bias_grad_diff < 5 * tol


def main() -> None:
    module = load_training_module()
    run_case(module, torch.float32)
    run_case(module, torch.bfloat16)


if __name__ == "__main__":
    main()
