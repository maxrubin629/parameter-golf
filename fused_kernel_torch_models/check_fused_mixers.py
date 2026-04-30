#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_cuda_triton(module) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for fused mixer parity checks")
    if not module.TRITON_AVAILABLE:
        raise SystemExit("Triton is required for fused mixer parity checks")


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def compare_tensors(label: str, ref: torch.Tensor, fused: torch.Tensor, tol: float) -> None:
    diff = max_abs(ref, fused)
    print(f"{label}: {diff:.6g}")
    assert diff < tol, f"{label} diff {diff} >= {tol}"


def check_mixer(module, cls_name: str, dtype: torch.dtype, dim: int = 64, nsrc: int = 7) -> None:
    assert_cuda_triton(module)
    torch.manual_seed(1234)
    device = torch.device("cuda")
    batch, seq_len = 2, 16
    target = torch.randn(batch, seq_len, dim, device=device, dtype=torch.float32)
    base_values = [torch.randn(batch, seq_len, dim, device=device, dtype=dtype) * 0.5 for _ in range(nsrc)]
    mixer_cls = getattr(module, cls_name)

    os.environ["FUSED_ATTNRES"] = "0"
    ref = mixer_cls(dim, max_sources=32, num_loops=1, loop_specific=False, use_gate=False).to(device)
    with torch.no_grad():
        ref.attnres_query.normal_(0.0, 0.2)
        ref.attnres_bias.normal_(0.0, 0.2)
    state = {name: tensor.detach().clone() for name, tensor in ref.state_dict().items()}

    os.environ["FUSED_ATTNRES"] = "1"
    fused = mixer_cls(dim, max_sources=32, num_loops=1, loop_specific=False, use_gate=False).to(device)
    fused.load_state_dict(state)

    def run_one(mixer: torch.nn.Module):
        values = [value.detach().clone().requires_grad_(True) for value in base_values]
        out = mixer(values)
        loss = (out.float() * target).sum()
        loss.backward()
        return (
            out.detach(),
            [value.grad.detach().clone() for value in values],
            mixer.attnres_query.grad.detach().clone(),
            mixer.attnres_bias.grad.detach().clone(),
        )

    ref_out, ref_value_grads, ref_query_grad, ref_bias_grad = run_one(ref)
    fused_out, fused_value_grads, fused_query_grad, fused_bias_grad = run_one(fused)
    tol = 2e-2 if dtype == torch.bfloat16 else 2e-4
    prefix = f"{module.__name__}.{cls_name}.{dtype}"
    compare_tensors(f"{prefix}.out", ref_out, fused_out, tol)
    compare_tensors(
        f"{prefix}.value_grad",
        torch.stack([g.float() for g in ref_value_grads]),
        torch.stack([g.float() for g in fused_value_grads]),
        tol,
    )
    compare_tensors(f"{prefix}.query_grad", ref_query_grad, fused_query_grad, 5 * tol)
    compare_tensors(f"{prefix}.bias_grad", ref_bias_grad, fused_bias_grad, 5 * tol)


def check_memory_depthmix(module, dtype: torch.dtype) -> None:
    assert_cuda_triton(module)
    torch.manual_seed(1234)
    device = torch.device("cuda")
    args = module.Hyperparameters()
    model = module.MemoryMoRGPT(args).to(device)
    with torch.no_grad():
        model.depth_queries.normal_(0.0, 0.2)
        model.depth_gates.normal_(0.0, 0.2)

    batch, seq_len, dim, nsrc = 2, 16, args.model_dim, 4
    target = torch.randn(batch, seq_len, dim, device=device, dtype=torch.float32)
    base_x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype) * 0.5
    base_sources = [torch.randn(batch, seq_len, dim, device=device, dtype=dtype) * 0.5 for _ in range(nsrc)]

    def run_one(fused: bool):
        model.zero_grad(set_to_none=True)
        model.fused_depthmix = fused
        x = base_x.detach().clone().requires_grad_(True)
        sources = [source.detach().clone().requires_grad_(True) for source in base_sources]
        out = model._depth_mix(x, sources, 0)
        loss = (out.float() * target).sum()
        loss.backward()
        return (
            out.detach(),
            x.grad.detach().clone(),
            [source.grad.detach().clone() for source in sources],
            model.depth_queries.grad.detach().clone(),
            model.depth_gates.grad.detach().clone(),
        )

    ref_out, ref_x_grad, ref_source_grads, ref_query_grad, ref_gate_grad = run_one(False)
    fused_out, fused_x_grad, fused_source_grads, fused_query_grad, fused_gate_grad = run_one(True)
    tol = 2e-2 if dtype == torch.bfloat16 else 2e-4
    prefix = f"{module.__name__}.memory_depthmix.{dtype}"
    compare_tensors(f"{prefix}.out", ref_out, fused_out, tol)
    compare_tensors(f"{prefix}.x_grad", ref_x_grad, fused_x_grad, tol)
    compare_tensors(
        f"{prefix}.source_grad",
        torch.stack([g.float() for g in ref_source_grads]),
        torch.stack([g.float() for g in fused_source_grads]),
        tol,
    )
    compare_tensors(f"{prefix}.query_grad", ref_query_grad, fused_query_grad, 5 * tol)
    compare_tensors(f"{prefix}.gate_grad", ref_gate_grad, fused_gate_grad, 5 * tol)


def main() -> None:
    full = load_module("train_gpt_torch_loop_fullattnres_loopq_fused_attnres_fastcuda.py", "full_fused")
    block = load_module("train_gpt_torch_sp8192_loop_xsa_attnres_loopq_fused_attnres_fastcuda.py", "block_fused")
    memory = load_module("train_gpt_torch_memory_mor_blockattn_fused_depthmix_fastcuda.py", "memory_fused")
    for dtype in (torch.float32, torch.bfloat16):
        check_mixer(full, "FullDepthMixer", dtype)
        check_mixer(block, "BlockDepthMixer", dtype)
        check_memory_depthmix(memory, dtype)


if __name__ == "__main__":
    main()
