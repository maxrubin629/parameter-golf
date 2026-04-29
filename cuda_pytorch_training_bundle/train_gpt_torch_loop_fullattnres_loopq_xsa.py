#!/usr/bin/env python3
"""
Standalone CUDA/PyTorch conversion of the MLX loop_fullattnres_loopq architecture,
with XSA intentionally added during the port.

The training, validation, logging, optimizer split, and checkpoint/export shape
follow the repo-root train_gpt.py style. The model architecture follows
parameter_golf_looped_full_attnres_mlx_optimized_bundle (3)/
train_gpt_mlx_loop_fullattnres_loopq_fastmlx.py, except that recurrent core
attention also applies the fixed projection-removal XSA used by the closest MLX
XSA variants in this repo.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


# -----------------------------
# ARCHITECTURE VARIANT DEFAULTS
# -----------------------------

ARCH_NAME = "loop_fullattnres_loopq_xsa_fastmlx"
ARCH_LOOPED = True
ARCH_STABLE_RECURRENCE = True
ARCH_XSA_ENABLED = True
ARCH_XSA_LEARNED = False
ARCH_XSA_LAMBDA_INIT = 1.0
ARCH_ATTNRES_ENABLED = True
ARCH_ATTNRES_USE_GATE = False
ARCH_ATTNRES_PREV_LOOP_BLOCKS = False


# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 0))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20_000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", 1024)))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model shape.
    arch_name = os.environ.get("ARCH_NAME", ARCH_NAME)
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    logit_chunk_tokens = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))

    # Looped decoder architecture controls.
    looped = bool(int(os.environ.get("LOOPED", "1" if ARCH_LOOPED else "0")))
    prelude_layers = int(os.environ.get("PRELUDE_LAYERS", 2))
    core_layers = int(os.environ.get("CORE_LAYERS", 2))
    coda_layers = int(os.environ.get("CODA_LAYERS", 2))
    num_loops = int(os.environ.get("NUM_LOOPS", 3))
    stable_recurrence = bool(int(os.environ.get("STABLE_RECURRENCE", "1" if ARCH_STABLE_RECURRENCE else "0")))
    recurrence_decay_init = float(os.environ.get("RECURRENCE_DECAY_INIT", 0.75))
    recurrence_inject_scale = float(os.environ.get("RECURRENCE_INJECT_SCALE", 1.0))
    xsa_enabled = bool(int(os.environ.get("XSA_ENABLED", "1" if ARCH_XSA_ENABLED else "0")))
    xsa_learned = bool(int(os.environ.get("XSA_LEARNED", "1" if ARCH_XSA_LEARNED else "0")))
    xsa_lambda_init = float(os.environ.get("XSA_LAMBDA_INIT", ARCH_XSA_LAMBDA_INIT))
    attnres_enabled = bool(int(os.environ.get("ATTNRES_ENABLED", "1" if ARCH_ATTNRES_ENABLED else "0")))
    attnres_use_gate = bool(int(os.environ.get("ATTNRES_USE_GATE", "1" if ARCH_ATTNRES_USE_GATE else "0")))
    attnres_prev_loop_blocks = bool(
        int(os.environ.get("ATTNRES_PREV_LOOP_BLOCKS", "1" if ARCH_ATTNRES_PREV_LOOP_BLOCKS else "0"))
    )
    attnres_max_sources = int(os.environ.get("ATTNRES_MAX_SOURCES", 128))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    out_dir = os.environ.get("OUT_DIR", "logs")


# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP
# -----------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        (
            "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
            "q_gain,skip_weight,skip_weights,attnres_query,attnres_bias,"
            "attnres_gate,xsa_lambda,recur_decay,prelude_inject"
        ),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        if local_tokens < seq_len or local_tokens % seq_len != 0:
            raise ValueError(
                "TRAIN_BATCH_TOKENS must yield an integral number of sequences per rank/micro-step; "
                f"got local_tokens={local_tokens}, seq_len={seq_len}, world_size={self.world_size}, "
                f"grad_accum_steps={grad_accum_steps}"
            )
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

def rms_norm_no_weight(x: Tensor, eps: float = 1e-6) -> Tensor:
    return F.rms_norm(x, (x.size(-1),), eps=eps)


class RMSNormNoWeight(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm_no_weight(x, self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            should_keep = param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
            if should_keep and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        xsa_enabled: bool = False,
        xsa_lambda_init: float = 1.0,
        xsa_learned: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.xsa_enabled = xsa_enabled
        self.xsa_learned = xsa_learned
        if xsa_enabled and xsa_learned:
            self.xsa_lambda = nn.Parameter(torch.full((num_heads,), xsa_lambda_init, dtype=torch.float32))
        else:
            self.xsa_lambda_value = float(xsa_lambda_init if xsa_enabled else 0.0)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = rms_norm_no_weight(q)
        k = rms_norm_no_weight(k)
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )

        if self.xsa_enabled:
            # Intentional port-time addition: follow the repo's fast MLX XSA
            # semantics by subtracting each query head's projection onto the
            # current token's own GQA value vector after SDPA.
            rep = self.num_heads // self.num_kv_heads
            y_grouped = y.reshape(bsz, self.num_kv_heads, rep, seqlen, self.head_dim)
            v_self = v.to(dtype=y.dtype)[:, :, None, :, :]
            v_self = v_self * torch.rsqrt((v_self * v_self).sum(dim=-1, keepdim=True) + 1e-6)
            proj = (y_grouped * v_self).sum(dim=-1, keepdim=True) * v_self
            if self.xsa_learned:
                lam = self.xsa_lambda.to(dtype=y.dtype).reshape(self.num_kv_heads, rep)[None, :, :, None, None]
                y_grouped = y_grouped - lam * proj
            elif self.xsa_lambda_value == 1.0:
                y_grouped = y_grouped - proj
            else:
                y_grouped = y_grouped - self.xsa_lambda_value * proj
            y = y_grouped.reshape(bsz, self.num_heads, seqlen, self.head_dim)

        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class DepthMixer(nn.Module):
    # Full-AttnRes same-position mixer. PyTorch keeps value/key lists instead of
    # the MLX AttnResHistory object; cached keys and streaming value readout keep
    # the math aligned without building a [B, T, N, D] value tensor.
    def __init__(
        self,
        dim: int,
        max_sources: int,
        num_loops: int,
        loop_specific: bool,
        use_gate: bool,
    ):
        super().__init__()
        self.max_sources = max_sources
        self.loop_specific = loop_specific
        self.use_gate = use_gate
        self.num_slots = max(num_loops, 1) if loop_specific else 1
        self.attnres_query = nn.Parameter(torch.zeros(self.num_slots, dim, dtype=torch.float32))
        self.attnres_bias = nn.Parameter(torch.zeros(self.num_slots, max_sources, dtype=torch.float32))
        if use_gate:
            self.attnres_gate = nn.Parameter(torch.zeros(self.num_slots, max_sources, dtype=torch.float32))
        self.scale = dim ** -0.5

    def forward(
        self,
        values: list[Tensor],
        keys: list[Tensor] | None = None,
        loop_idx: int = 0,
    ) -> Tensor:
        nsrc = len(values)
        if nsrc <= 0:
            raise ValueError("DepthMixer requires at least one source")
        if nsrc > self.max_sources:
            raise ValueError(
                f"Full AttnRes source history has {nsrc} sources, "
                f"but ATTNRES_MAX_SOURCES={self.max_sources}. Increase ATTNRES_MAX_SOURCES."
            )
        if keys is None:
            keys = [rms_norm_no_weight(src) for src in values]
        if len(keys) != nsrc:
            raise ValueError(f"DepthMixer got {nsrc} source values but {len(keys)} source keys")
        if nsrc == 1:
            return values[0]

        slot = loop_idx if self.loop_specific else 0
        if slot >= self.num_slots:
            raise ValueError(f"loop_idx={loop_idx} exceeds available AttnRes slots={self.num_slots}")

        query = self.attnres_query[slot].to(dtype=keys[0].dtype)
        scores = torch.stack([(key * query).sum(dim=-1) * self.scale for key in keys], dim=-1).float()
        scores = scores + self.attnres_bias[slot, :nsrc][None, None, :]
        if self.use_gate:
            scores = scores + torch.tanh(self.attnres_gate[slot, :nsrc])[None, None, :]

        weights = torch.softmax(scores, dim=-1).to(dtype=values[0].dtype)
        out = values[0] * weights[:, :, 0:1]
        for i in range(1, nsrc):
            out = out + values[i] * weights[:, :, i : i + 1]
        return out


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        xsa_enabled: bool = False,
        xsa_lambda_init: float = 1.0,
        xsa_learned: bool = False,
        attnres_enabled: bool = False,
        attnres_loop_specific: bool = False,
        attnres_max_sources: int = 6,
        attnres_num_loops: int = 1,
        attnres_use_gate: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            xsa_enabled=xsa_enabled,
            xsa_lambda_init=xsa_lambda_init,
            xsa_learned=xsa_learned,
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.attnres_enabled = attnres_enabled
        if attnres_enabled:
            self.attn_mixer = DepthMixer(dim, attnres_max_sources, attnres_num_loops, attnres_loop_specific, attnres_use_gate)
            self.mlp_mixer = DepthMixer(dim, attnres_max_sources, attnres_num_loops, attnres_loop_specific, attnres_use_gate)

    def forward(
        self,
        x: Tensor,
        x0: Tensor,
        attn_sources: list[Tensor] | None = None,
        mlp_sources: list[Tensor] | None = None,
        loop_idx: int = 0,
    ) -> Tensor:
        if self.attnres_enabled and attn_sources is not None:
            residual = x
            attn_in = self.attn_mixer(attn_sources, None, loop_idx)
            attn_out = self.attn(self.attn_norm(attn_in))
            x = residual + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        else:
            mix = self.resid_mix.to(dtype=x.dtype)
            x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            attn_out = self.attn(self.attn_norm(x))
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out

        if self.attnres_enabled and mlp_sources is not None:
            mlp_in = self.mlp_mixer(mlp_sources, None, loop_idx)
        else:
            mlp_in = x
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(mlp_in))
        return x

    def full_attnres_attn(self, values: list[Tensor], keys: list[Tensor], loop_idx: int = 0) -> Tensor:
        attn_in = self.attn_mixer(values, keys, loop_idx)
        return self.attn_scale.to(dtype=attn_in.dtype)[None, None, :] * self.attn(self.attn_norm(attn_in))

    def full_attnres_mlp(self, values: list[Tensor], keys: list[Tensor], loop_idx: int = 0) -> Tensor:
        mlp_in = self.mlp_mixer(values, keys, loop_idx)
        return self.mlp_scale.to(dtype=mlp_in.dtype)[None, None, :] * self.mlp(self.mlp_norm(mlp_in))


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_chunk_tokens: int,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        args: Hyperparameters,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self.arch_name = args.arch_name
        self.looped = args.looped
        self.prelude_layers = args.prelude_layers if args.looped else num_layers // 2
        self.core_layers = args.core_layers if args.looped else 0
        self.coda_layers = args.coda_layers if args.looped else num_layers - self.prelude_layers
        self.num_loops = max(args.num_loops, 1) if args.looped else 1
        self.stable_recurrence = bool(args.stable_recurrence and args.looped)
        self.attnres_enabled = bool(args.attnres_enabled and args.looped)
        self.attnres_prev_loop_blocks = bool(args.attnres_prev_loop_blocks and self.attnres_enabled)
        self.attnres_max_sources = args.attnres_max_sources

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        total_blocks = self.prelude_layers + self.core_layers + self.coda_layers if args.looped else num_layers
        self.blocks = nn.ModuleList()
        for i in range(total_blocks):
            in_core = args.looped and self.prelude_layers <= i < self.prelude_layers + self.core_layers
            attnres_on = bool(args.attnres_enabled and args.looped)
            loop_specific_mixer = bool(attnres_on and in_core)
            self.blocks.append(
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    xsa_enabled=bool(args.xsa_enabled and in_core),
                    xsa_lambda_init=args.xsa_lambda_init,
                    xsa_learned=bool(args.xsa_learned and in_core),
                    attnres_enabled=attnres_on,
                    attnres_loop_specific=loop_specific_mixer,
                    attnres_max_sources=args.attnres_max_sources,
                    attnres_num_loops=self.num_loops if loop_specific_mixer else 1,
                    attnres_use_gate=bool(args.attnres_use_gate),
                )
            )
        self.final_norm = RMSNormNoWeight()
        if self.attnres_enabled:
            self.final_mixer = DepthMixer(model_dim, args.attnres_max_sources, 1, False, False)
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        if self.stable_recurrence:
            decay = min(max(args.recurrence_decay_init, 1e-4), 1.0 - 1e-4)
            decay_logit = math.log(decay / (1.0 - decay))
            self.recur_decay_logit = nn.Parameter(torch.full((model_dim,), decay_logit, dtype=torch.float32))
            self.prelude_inject_scale = nn.Parameter(
                torch.full((model_dim,), args.recurrence_inject_scale, dtype=torch.float32)
            )

        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def softcap(self, logits: Tensor) -> Tensor:
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def _stable_loop_input(self, h: Tensor, e: Tensor) -> Tensor:
        decay = torch.sigmoid(self.recur_decay_logit.to(dtype=h.dtype)).reshape(1, 1, -1)
        inject = self.prelude_inject_scale.to(dtype=h.dtype).reshape(1, 1, -1)
        return rms_norm_no_weight(decay * h + (1.0 - decay) * inject * e)

    def _append_full_history(
        self,
        values: list[Tensor],
        keys: list[Tensor],
        total: Tensor,
        source: Tensor,
    ) -> Tensor:
        if len(values) >= self.attnres_max_sources:
            raise ValueError(
                f"Full AttnRes source history would exceed ATTNRES_MAX_SOURCES={self.attnres_max_sources}. "
                "Increase ATTNRES_MAX_SOURCES."
            )
        values.append(source)
        keys.append(rms_norm_no_weight(source))
        return total + source

    @staticmethod
    def _history_mean(values: list[Tensor], total: Tensor) -> Tensor:
        if not values:
            raise ValueError("Full AttnRes history is empty")
        return total / float(len(values))

    @staticmethod
    def _history_mean_since(
        values: list[Tensor],
        total: Tensor,
        snapshot: tuple[int, Tensor],
    ) -> Tensor:
        start_count, start_total = snapshot
        nsrc = len(values) - start_count
        if nsrc <= 0:
            raise ValueError("mean_since requires at least one new source")
        return (total - start_total) / float(nsrc)

    def encode(self, input_ids: Tensor) -> Tensor:
        x = rms_norm_no_weight(self.tok_emb(input_ids))
        x0 = x

        if not self.looped:
            for block in self.blocks:
                x = block(x, x0)
            return self.final_norm(x)

        if not self.attnres_enabled:
            for i in range(self.prelude_layers):
                x = self.blocks[i](x, x0)
            e = rms_norm_no_weight(x)
            h = e
            core_start = self.prelude_layers
            for loop_idx in range(self.num_loops):
                if self.stable_recurrence:
                    h = self._stable_loop_input(h, e)
                for j in range(self.core_layers):
                    h = self.blocks[core_start + j](h, e, loop_idx=loop_idx)
            coda_start = self.prelude_layers + self.core_layers
            for j in range(self.coda_layers):
                h = self.blocks[coda_start + j](h, e)
            return self.final_norm(h)

        history_values: list[Tensor] = [x0]
        history_keys: list[Tensor] = [rms_norm_no_weight(x0)]
        history_total = x0

        h = x0
        for i in range(self.prelude_layers):
            block = self.blocks[i]
            attn_out = block.full_attnres_attn(history_values, history_keys, loop_idx=0)
            history_total = self._append_full_history(history_values, history_keys, history_total, attn_out)
            mlp_out = block.full_attnres_mlp(history_values, history_keys, loop_idx=0)
            history_total = self._append_full_history(history_values, history_keys, history_total, mlp_out)
            h = mlp_out

        # The MLX source seeds recurrence from the mean full-source readout.
        e = rms_norm_no_weight(self._history_mean(history_values, history_total))
        h = e

        core_start = self.prelude_layers
        for loop_idx in range(self.num_loops):
            if self.stable_recurrence:
                h = self._stable_loop_input(h, e)
            loop_snapshot = (len(history_values), history_total)
            history_total = self._append_full_history(history_values, history_keys, history_total, h)
            for j in range(self.core_layers):
                block = self.blocks[core_start + j]
                attn_out = block.full_attnres_attn(history_values, history_keys, loop_idx=loop_idx)
                history_total = self._append_full_history(history_values, history_keys, history_total, attn_out)
                mlp_out = block.full_attnres_mlp(history_values, history_keys, loop_idx=loop_idx)
                history_total = self._append_full_history(history_values, history_keys, history_total, mlp_out)
                h = mlp_out
            if self.attnres_prev_loop_blocks:
                summary = self._history_mean_since(history_values, history_total, loop_snapshot)
                history_total = self._append_full_history(history_values, history_keys, history_total, summary)

        coda_start = self.prelude_layers + self.core_layers
        for j in range(self.coda_layers):
            block = self.blocks[coda_start + j]
            attn_out = block.full_attnres_attn(history_values, history_keys, loop_idx=0)
            history_total = self._append_full_history(history_values, history_keys, history_total, attn_out)
            mlp_out = block.full_attnres_mlp(history_values, history_keys, loop_idx=0)
            history_total = self._append_full_history(history_values, history_keys, history_total, mlp_out)
            h = mlp_out

        h = self.final_mixer(history_values, history_keys, loop_idx=0)
        return self.final_norm(h)

    def _logits_loss(self, x: Tensor, target_ids: Tensor) -> Tensor:
        x = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)

        if self.logit_chunk_tokens <= 0 or x.size(0) <= self.logit_chunk_tokens:
            if self.tie_embeddings:
                logits_proj = F.linear(x, self.tok_emb.weight.to(dtype=x.dtype))
            else:
                if self.lm_head is None:
                    raise RuntimeError("lm_head is required when tie_embeddings=False")
                logits_proj = self.lm_head(x)
            logits = self.softcap(logits_proj)
            return F.cross_entropy(logits.float(), targets, reduction="mean")

        loss_sum = torch.zeros((), device=x.device, dtype=torch.float32)
        for start in range(0, x.size(0), self.logit_chunk_tokens):
            end = min(start + self.logit_chunk_tokens, x.size(0))
            chunk = x[start:end]
            if self.tie_embeddings:
                logits_proj = F.linear(chunk, self.tok_emb.weight.to(dtype=chunk.dtype))
            else:
                if self.lm_head is None:
                    raise RuntimeError("lm_head is required when tie_embeddings=False")
                logits_proj = self.lm_head(chunk)
            logits = self.softcap(logits_proj)
            loss_sum = loss_sum + F.cross_entropy(logits.float(), targets[start:end], reduction="sum")
        return loss_sum / float(x.size(0))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        return self._logits_loss(self.encode(input_ids), target_ids)


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)
        logfile = f"{args.out_dir}/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        args=args,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    compile_enabled = bool(int(os.environ.get("TORCH_COMPILE", "1")))
    compile_fullgraph = bool(int(os.environ.get("TORCH_COMPILE_FULLGRAPH", "0")))
    compiled_model = (
        torch.compile(base_model, dynamic=False, fullgraph=compile_fullgraph)
        if compile_enabled
        else base_model
    )
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    named_params = list(base_model.named_parameters())
    embedding_params = [base_model.tok_emb.weight]
    head_params = [base_model.lm_head.weight] if base_model.lm_head is not None else []
    excluded = {id(p) for p in embedding_params + head_params}
    matrix_params = [
        p
        for name, p in named_params
        if id(p) not in excluded
        and p.ndim == 2
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in named_params
        if id(p) not in excluded
        and (p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))
    ]
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": embedding_params, "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if head_params:
        optimizer_head = torch.optim.Adam(
            [{"params": head_params, "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(
        f"arch:{args.arch_name} physical_layers:{len(base_model.blocks)} "
        f"prelude:{args.prelude_layers} core:{args.core_layers} coda:{args.coda_layers} loops:{args.num_loops} "
        f"stable:{args.stable_recurrence} xsa:{args.xsa_enabled} xsa_learned:{args.xsa_learned} "
        f"xsa_lambda:{args.xsa_lambda_init} attnres:{args.attnres_enabled} scope:full_all_sublayers "
        f"attnres_core_loop_slots:{args.attnres_enabled} prev_loop_blocks:{args.attnres_prev_loop_blocks} "
        f"source_budget:{args.attnres_max_sources}"
    )
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(f"torch_compile:{compile_enabled} fullgraph:{compile_fullgraph}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
