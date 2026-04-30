#!/usr/bin/env python3
"""
CUDA/PyTorch conversion of train_gpt_mlx_memory_mor_blockattn_fastmlx.py.
The model classes preserve the MLX architecture in the source script: the looped/XSA
variants keep the recurrent prelude-core-coda structure, optional block/full
Attention Residual mixers, stable recurrence, and tied embedding head; the memory
variant keeps the global memory read, block-depth mixer, and adaptive extra passes.
The training/runtime shell follows the CUDA/DDP style of train_gpt.py and is set up
for torchrun on 8x H100s.
"""
from __future__ import annotations
import copy
import glob
import io
import json
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
try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None
COMPUTE_DTYPE = torch.bfloat16
TRITON_AVAILABLE = triton is not None
FUSED_ATTNRES_MAX_SOURCES = 32
ARCH_KIND = "memory_mor_blockattn"
ARCH_NAME_DEFAULT = "memory_mor_blockattn_fused_depthmix"
DEFAULT_DATA_PATH = "./data/datasets/fineweb10B_sp8192"
DEFAULT_TOKENIZER_PATH = "./data/tokenizers/fineweb_8192_bpe.model"
DEFAULT_RUN_ID_PREFIX = "memory_mor_blockattn_fused_depthmix"
DEFAULT_VOCAB_SIZE = 8192
DEFAULT_NUM_LAYERS = 8
DEFAULT_MODEL_DIM = 384
DEFAULT_NUM_HEADS = 6
DEFAULT_NUM_KV_HEADS = 2
DEFAULT_MLP_MULT = 0
DEFAULT_MLP_HIDDEN_DIM = 1280
DEFAULT_QK_GAIN_INIT = 5.25
DEFAULT_TRAIN_SEQ_LEN = 512
DEFAULT_TRAIN_BATCH_TOKENS = 32768
DEFAULT_VAL_BATCH_SIZE = 131072
DEFAULT_ITERATIONS = 80
DEFAULT_VAL_LOSS_EVERY = 0
DEFAULT_TRAIN_LOG_EVERY = 5
DEFAULT_WARMUP_STEPS = 2
DEFAULT_WARMDOWN_ITERS = 120
DEFAULT_MAX_WALLCLOCK_SECONDS = 180.0
DEFAULT_MUON_MOMENTUM_WARMUP_STEPS = 100
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_MEMORY_SLOTS = 512
ARCH_LOOPED = False
ARCH_STABLE_RECURRENCE = False
ARCH_XSA_ENABLED = False
ARCH_XSA_LEARNED = False
ARCH_XSA_LAMBDA_INIT = 0.0
ARCH_ATTNRES_ENABLED = False
ARCH_ATTNRES_MODE = "block"
ARCH_ATTNRES_USE_GATE = False
ARCH_ATTNRES_PREV_LOOP_BLOCKS = False
DEFAULT_ATTNRES_MAX_SOURCES = 8
class Hyperparameters:
    # Data / tokenizer.
    data_path: str = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", DEFAULT_TOKENIZER_PATH)
    run_id: str = os.environ.get(
        "RUN_ID",
        (DEFAULT_RUN_ID_PREFIX + "-" if DEFAULT_RUN_ID_PREFIX else "") + str(uuid.uuid4()),
    )
    seed: int = int(os.environ.get("SEED", 1337))
    # Training loop. GRAD_ACCUM_STEPS is the single-process MLX accumulation count.
    # In DDP we divide it across ranks, so the default 8 becomes 1 microstep on 8 GPUs.
    iterations: int = int(os.environ.get("ITERATIONS", DEFAULT_ITERATIONS))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", DEFAULT_VAL_LOSS_EVERY))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", DEFAULT_VAL_BATCH_SIZE))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", DEFAULT_TRAIN_LOG_EVERY))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", DEFAULT_TRAIN_BATCH_TOKENS))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", DEFAULT_TRAIN_SEQ_LEN)))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", DEFAULT_WARMUP_STEPS))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", DEFAULT_WARMDOWN_ITERS))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", DEFAULT_MAX_WALLCLOCK_SECONDS))
    # Model.
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", DEFAULT_VOCAB_SIZE))
    num_layers: int = int(os.environ.get("NUM_LAYERS", DEFAULT_NUM_LAYERS))
    model_dim: int = int(os.environ.get("MODEL_DIM", DEFAULT_MODEL_DIM))
    num_heads: int = int(os.environ.get("NUM_HEADS", DEFAULT_NUM_HEADS))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", DEFAULT_NUM_KV_HEADS))
    mlp_mult: int = int(os.environ.get("MLP_MULT", DEFAULT_MLP_MULT))
    mlp_hidden_dim: int = int(os.environ.get("MLP_HIDDEN_DIM", DEFAULT_MLP_HIDDEN_DIM))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", DEFAULT_QK_GAIN_INIT))
    # Looped architecture controls.
    arch_name: str = os.environ.get("ARCH_NAME", ARCH_NAME_DEFAULT)
    looped: bool = bool(int(os.environ.get("LOOPED", "1" if ARCH_LOOPED else "0")))
    prelude_layers: int = int(os.environ.get("PRELUDE_LAYERS", 2))
    core_layers: int = int(os.environ.get("CORE_LAYERS", 2))
    coda_layers: int = int(os.environ.get("CODA_LAYERS", 2))
    num_loops: int = int(os.environ.get("NUM_LOOPS", 3))
    stable_recurrence: bool = bool(int(os.environ.get("STABLE_RECURRENCE", "1" if ARCH_STABLE_RECURRENCE else "0")))
    recurrence_decay_init: float = float(os.environ.get("RECURRENCE_DECAY_INIT", 0.75))
    recurrence_inject_scale: float = float(os.environ.get("RECURRENCE_INJECT_SCALE", 1.0))
    xsa_enabled: bool = bool(int(os.environ.get("XSA_ENABLED", "1" if ARCH_XSA_ENABLED else "0")))
    xsa_learned: bool = bool(int(os.environ.get("XSA_LEARNED", "1" if ARCH_XSA_LEARNED else "0")))
    xsa_lambda_init: float = float(os.environ.get("XSA_LAMBDA_INIT", ARCH_XSA_LAMBDA_INIT))
    attnres_enabled: bool = bool(int(os.environ.get("ATTNRES_ENABLED", "1" if ARCH_ATTNRES_ENABLED else "0")))
    attnres_mode: str = os.environ.get("ATTNRES_MODE", ARCH_ATTNRES_MODE)
    attnres_use_gate: bool = bool(int(os.environ.get("ATTNRES_USE_GATE", "1" if ARCH_ATTNRES_USE_GATE else "0")))
    attnres_prev_loop_blocks: bool = bool(int(os.environ.get("ATTNRES_PREV_LOOP_BLOCKS", "1" if ARCH_ATTNRES_PREV_LOOP_BLOCKS else "0")))
    attnres_max_sources: int = int(os.environ.get("ATTNRES_MAX_SOURCES", DEFAULT_ATTNRES_MAX_SOURCES))
    # Memory-MoR controls.
    memory_slots: int = int(os.environ.get("MEMORY_SLOTS", DEFAULT_MEMORY_SLOTS))
    use_block_attnres: bool = bool(int(os.environ.get("USE_BLOCK_ATTNRES", "1")))
    attnres_source_every: int = max(1, int(os.environ.get("ATTNRES_SOURCE_EVERY", "2")))
    max_depth_sources: int = max(2, int(os.environ.get("MAX_DEPTH_SOURCES", "8")))
    extra_loop_passes: int = max(0, int(os.environ.get("EXTRA_LOOP_PASSES", "1")))
    extra_loop_schedule: str = os.environ.get("EXTRA_LOOP_SCHEDULE", "")
    # Optimizer.
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", DEFAULT_MUON_MOMENTUM_WARMUP_STEPS))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", DEFAULT_GRAD_CLIP_NORM))
    # Runtime / output.
    out_dir: str = os.environ.get("OUT_DIR", "logs")
    torch_compile: bool = bool(int(os.environ.get("TORCH_COMPILE", "1")))
    torch_compile_fullgraph: bool = bool(int(os.environ.get("TORCH_COMPILE_FULLGRAPH", "0")))
    fused_attnres: bool = bool(int(os.environ.get("FUSED_ATTNRES", "1")))
    fused_attnres_block_m: int = max(1, int(os.environ.get("FUSED_ATTNRES_BLOCK_M", "4")))
    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"
    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"
    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,depth_query,depth_queries,depth_gate,depth_gates,loop_scale,loop_scales,extra_loop_scale,extra_loop_scales,ngram_scale,memory_gate,route_bias",
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
# -----------------------------
# MUON OPTIMIZER
# -----------------------------
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
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
# TOKENIZER-AGNOSTIC EVALUATION
# -----------------------------
def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
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
def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None
    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None) if tokenizer_name else None
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files
# -----------------------------
# QUANTIZATION
# -----------------------------
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
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
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
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
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
        if local_tokens % seq_len != 0:
            raise ValueError(
                f"Per-rank microbatch tokens {local_tokens} must be divisible by TRAIN_SEQ_LEN={seq_len}. "
                "Adjust TRAIN_BATCH_TOKENS, WORLD_SIZE, or GRAD_ACCUM_STEPS."
            )
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]
# -----------------------------
# MODEL BLOCKS
# -----------------------------
def rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    scale = torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps)
    return (x * scale.to(dtype=x.dtype)).to(dtype=x.dtype)
class RMSNormNoWeight(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x)
if TRITON_AVAILABLE:
    @triton.jit
    def _select_fused_attnres_ptr(
        idx: tl.constexpr,
        P0, P1, P2, P3, P4, P5, P6, P7,
        P8, P9, P10, P11, P12, P13, P14, P15,
        P16, P17, P18, P19, P20, P21, P22, P23,
        P24, P25, P26, P27, P28, P29, P30, P31,
    ):
        if idx == 0:
            return P0
        if idx == 1:
            return P1
        if idx == 2:
            return P2
        if idx == 3:
            return P3
        if idx == 4:
            return P4
        if idx == 5:
            return P5
        if idx == 6:
            return P6
        if idx == 7:
            return P7
        if idx == 8:
            return P8
        if idx == 9:
            return P9
        if idx == 10:
            return P10
        if idx == 11:
            return P11
        if idx == 12:
            return P12
        if idx == 13:
            return P13
        if idx == 14:
            return P14
        if idx == 15:
            return P15
        if idx == 16:
            return P16
        if idx == 17:
            return P17
        if idx == 18:
            return P18
        if idx == 19:
            return P19
        if idx == 20:
            return P20
        if idx == 21:
            return P21
        if idx == 22:
            return P22
        if idx == 23:
            return P23
        if idx == 24:
            return P24
        if idx == 25:
            return P25
        if idx == 26:
            return P26
        if idx == 27:
            return P27
        if idx == 28:
            return P28
        if idx == 29:
            return P29
        if idx == 30:
            return P30
        return P31

    @triton.jit
    def _fused_full_depth_mixer_fwd_kernel(
        V0, V1, V2, V3, V4, V5, V6, V7,
        V8, V9, V10, V11, V12, V13, V14, V15,
        V16, V17, V18, V19, V20, V21, V22, V23,
        V24, V25, V26, V27, V28, V29, V30, V31,
        K0, K1, K2, K3, K4, K5, K6, K7,
        K8, K9, K10, K11, K12, K13, K14, K15,
        K16, K17, K18, K19, K20, K21, K22, K23,
        K24, K25, K26, K27, K28, K29, K30, K31,
        query_ptr,
        bias_ptr,
        out_ptr,
        weights_ptr,
        total_tokens,
        dim: tl.constexpr,
        nsrc: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_SOURCES: tl.constexpr,
        QUERY_BF16: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        src_ids = tl.arange(0, MAX_SOURCES)
        token_mask = offs_m < total_tokens
        dim_mask = offs_d < dim
        md_mask = token_mask[:, None] & dim_mask[None, :]
        q = tl.load(query_ptr + offs_d, mask=dim_mask, other=0.0)
        if QUERY_BF16:
            q = q.to(tl.bfloat16)
        q = q.to(tl.float32)
        scores = tl.full((BLOCK_M, MAX_SOURCES), -float("inf"), tl.float32)
        for s in tl.static_range(0, MAX_SOURCES):
            if s < nsrc:
                k_ptr = _select_fused_attnres_ptr(
                    s,
                    K0, K1, K2, K3, K4, K5, K6, K7,
                    K8, K9, K10, K11, K12, K13, K14, K15,
                    K16, K17, K18, K19, K20, K21, K22, K23,
                    K24, K25, K26, K27, K28, K29, K30, K31,
                )
                k = tl.load(k_ptr + offs_m[:, None] * dim + offs_d[None, :], mask=md_mask, other=0.0).to(tl.float32)
                score = tl.sum(k * q[None, :], axis=1) * scale + tl.load(bias_ptr + s).to(tl.float32)
                scores = tl.where(src_ids[None, :] == s, score[:, None], scores)
        score_max = tl.max(scores, axis=1)
        weights = tl.exp(scores - score_max[:, None])
        weights = weights / tl.sum(weights, axis=1)[:, None]
        tl.store(
            weights_ptr + offs_m[:, None] * MAX_SOURCES + src_ids[None, :],
            weights,
            mask=token_mask[:, None] & (src_ids[None, :] < nsrc),
        )
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        for s in tl.static_range(0, MAX_SOURCES):
            if s < nsrc:
                v_ptr = _select_fused_attnres_ptr(
                    s,
                    V0, V1, V2, V3, V4, V5, V6, V7,
                    V8, V9, V10, V11, V12, V13, V14, V15,
                    V16, V17, V18, V19, V20, V21, V22, V23,
                    V24, V25, V26, V27, V28, V29, V30, V31,
                )
                v = tl.load(v_ptr + offs_m[:, None] * dim + offs_d[None, :], mask=md_mask, other=0.0).to(tl.float32)
                w = tl.sum(tl.where(src_ids[None, :] == s, weights, 0.0), axis=1)
                acc += v * w[:, None]
        tl.store(out_ptr + offs_m[:, None] * dim + offs_d[None, :], acc, mask=md_mask)

    @triton.jit
    def _fused_full_depth_mixer_bwd_kernel(
        V0, V1, V2, V3, V4, V5, V6, V7,
        V8, V9, V10, V11, V12, V13, V14, V15,
        V16, V17, V18, V19, V20, V21, V22, V23,
        V24, V25, V26, V27, V28, V29, V30, V31,
        K0, K1, K2, K3, K4, K5, K6, K7,
        K8, K9, K10, K11, K12, K13, K14, K15,
        K16, K17, K18, K19, K20, K21, K22, K23,
        K24, K25, K26, K27, K28, K29, K30, K31,
        DV0, DV1, DV2, DV3, DV4, DV5, DV6, DV7,
        DV8, DV9, DV10, DV11, DV12, DV13, DV14, DV15,
        DV16, DV17, DV18, DV19, DV20, DV21, DV22, DV23,
        DV24, DV25, DV26, DV27, DV28, DV29, DV30, DV31,
        DK0, DK1, DK2, DK3, DK4, DK5, DK6, DK7,
        DK8, DK9, DK10, DK11, DK12, DK13, DK14, DK15,
        DK16, DK17, DK18, DK19, DK20, DK21, DK22, DK23,
        DK24, DK25, DK26, DK27, DK28, DK29, DK30, DK31,
        query_ptr,
        out_ptr,
        grad_out_ptr,
        weights_ptr,
        dquery_partials_ptr,
        dbias_partials_ptr,
        total_tokens,
        dim: tl.constexpr,
        nsrc: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_SOURCES: tl.constexpr,
        QUERY_BF16: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        src_ids = tl.arange(0, MAX_SOURCES)
        token_mask = offs_m < total_tokens
        dim_mask = offs_d < dim
        md_mask = token_mask[:, None] & dim_mask[None, :]
        q = tl.load(query_ptr + offs_d, mask=dim_mask, other=0.0)
        if QUERY_BF16:
            q = q.to(tl.bfloat16)
        q = q.to(tl.float32)
        out = tl.load(out_ptr + offs_m[:, None] * dim + offs_d[None, :], mask=md_mask, other=0.0).to(tl.float32)
        grad_out = tl.load(grad_out_ptr + offs_m[:, None] * dim + offs_d[None, :], mask=md_mask, other=0.0).to(tl.float32)
        weights = tl.load(
            weights_ptr + offs_m[:, None] * MAX_SOURCES + src_ids[None, :],
            mask=token_mask[:, None] & (src_ids[None, :] < nsrc),
            other=0.0,
        ).to(tl.float32)
        dq_acc = tl.zeros((BLOCK_D,), tl.float32)
        db_acc = tl.zeros((MAX_SOURCES,), tl.float32)
        for s in tl.static_range(0, MAX_SOURCES):
            if s < nsrc:
                v_ptr = _select_fused_attnres_ptr(
                    s,
                    V0, V1, V2, V3, V4, V5, V6, V7,
                    V8, V9, V10, V11, V12, V13, V14, V15,
                    V16, V17, V18, V19, V20, V21, V22, V23,
                    V24, V25, V26, V27, V28, V29, V30, V31,
                )
                k_ptr = _select_fused_attnres_ptr(
                    s,
                    K0, K1, K2, K3, K4, K5, K6, K7,
                    K8, K9, K10, K11, K12, K13, K14, K15,
                    K16, K17, K18, K19, K20, K21, K22, K23,
                    K24, K25, K26, K27, K28, K29, K30, K31,
                )
                dv_ptr = _select_fused_attnres_ptr(
                    s,
                    DV0, DV1, DV2, DV3, DV4, DV5, DV6, DV7,
                    DV8, DV9, DV10, DV11, DV12, DV13, DV14, DV15,
                    DV16, DV17, DV18, DV19, DV20, DV21, DV22, DV23,
                    DV24, DV25, DV26, DV27, DV28, DV29, DV30, DV31,
                )
                dk_ptr = _select_fused_attnres_ptr(
                    s,
                    DK0, DK1, DK2, DK3, DK4, DK5, DK6, DK7,
                    DK8, DK9, DK10, DK11, DK12, DK13, DK14, DK15,
                    DK16, DK17, DK18, DK19, DK20, DK21, DK22, DK23,
                    DK24, DK25, DK26, DK27, DK28, DK29, DK30, DK31,
                )
                v = tl.load(v_ptr + offs_m[:, None] * dim + offs_d[None, :], mask=md_mask, other=0.0).to(tl.float32)
                k = tl.load(k_ptr + offs_m[:, None] * dim + offs_d[None, :], mask=md_mask, other=0.0).to(tl.float32)
                w = tl.sum(tl.where(src_ids[None, :] == s, weights, 0.0), axis=1)
                dscore = w * tl.sum(grad_out * (v - out), axis=1)
                tl.store(dv_ptr + offs_m[:, None] * dim + offs_d[None, :], grad_out * w[:, None], mask=md_mask)
                tl.store(dk_ptr + offs_m[:, None] * dim + offs_d[None, :], dscore[:, None] * scale * q[None, :], mask=md_mask)
                dq_acc += tl.sum(k * (dscore[:, None] * scale), axis=0)
                db_acc += tl.where(src_ids == s, tl.sum(dscore, axis=0), 0.0)
        tl.store(dquery_partials_ptr + pid * BLOCK_D + offs_d, dq_acc, mask=offs_d < BLOCK_D)
        tl.store(dbias_partials_ptr + pid * MAX_SOURCES + src_ids, db_acc, mask=src_ids < MAX_SOURCES)

    class _FusedFullDepthMixerFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, query: Tensor, bias: Tensor, *args):
            values = list(args[:FUSED_ATTNRES_MAX_SOURCES])
            keys = list(args[FUSED_ATTNRES_MAX_SOURCES : 2 * FUSED_ATTNRES_MAX_SOURCES])
            scale = float(args[2 * FUSED_ATTNRES_MAX_SOURCES])
            nsrc = int(args[2 * FUSED_ATTNRES_MAX_SOURCES + 1])
            block_m = int(args[2 * FUSED_ATTNRES_MAX_SOURCES + 2])
            shape = values[0].shape
            dim = int(shape[-1])
            total_tokens = values[0].numel() // dim
            block_d = 1 << (dim - 1).bit_length()
            out = torch.empty_like(values[0])
            weights = torch.empty((total_tokens, FUSED_ATTNRES_MAX_SOURCES), device=values[0].device, dtype=torch.float32)
            grid = (triton.cdiv(total_tokens, block_m),)
            _fused_full_depth_mixer_fwd_kernel[grid](
                *values,
                *keys,
                query,
                bias,
                out,
                weights,
                total_tokens,
                dim,
                nsrc,
                scale,
                BLOCK_M=block_m,
                BLOCK_D=block_d,
                MAX_SOURCES=FUSED_ATTNRES_MAX_SOURCES,
                QUERY_BF16=values[0].dtype == torch.bfloat16,
                num_warps=8,
            )
            ctx.save_for_backward(query, out, weights, *values[:nsrc], *keys[:nsrc])
            ctx.nsrc = nsrc
            ctx.scale = scale
            ctx.block_m = block_m
            ctx.dim = dim
            ctx.block_d = block_d
            ctx.total_tokens = total_tokens
            ctx.bias_numel = bias.numel()
            ctx.query_bf16 = values[0].dtype == torch.bfloat16
            return out

        @staticmethod
        def backward(ctx, grad_out: Tensor):
            saved = ctx.saved_tensors
            query, out, weights = saved[:3]
            values = list(saved[3 : 3 + ctx.nsrc])
            keys = list(saved[3 + ctx.nsrc : 3 + 2 * ctx.nsrc])
            dvalues = [torch.empty_like(v) for v in values]
            dkeys = [torch.empty_like(k) for k in keys]
            pad_value = values[0]
            pad_key = keys[0]
            pad_dvalue = dvalues[0]
            pad_dkey = dkeys[0]
            value_ptrs = values + [pad_value] * (FUSED_ATTNRES_MAX_SOURCES - ctx.nsrc)
            key_ptrs = keys + [pad_key] * (FUSED_ATTNRES_MAX_SOURCES - ctx.nsrc)
            dvalue_ptrs = dvalues + [pad_dvalue] * (FUSED_ATTNRES_MAX_SOURCES - ctx.nsrc)
            dkey_ptrs = dkeys + [pad_dkey] * (FUSED_ATTNRES_MAX_SOURCES - ctx.nsrc)
            num_blocks = triton.cdiv(ctx.total_tokens, ctx.block_m)
            dquery_partials = torch.empty((num_blocks, ctx.block_d), device=grad_out.device, dtype=torch.float32)
            dbias_partials = torch.empty((num_blocks, FUSED_ATTNRES_MAX_SOURCES), device=grad_out.device, dtype=torch.float32)
            grad_out_flat = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
            _fused_full_depth_mixer_bwd_kernel[(num_blocks,)](
                *value_ptrs,
                *key_ptrs,
                *dvalue_ptrs,
                *dkey_ptrs,
                query,
                out,
                grad_out_flat,
                weights,
                dquery_partials,
                dbias_partials,
                ctx.total_tokens,
                ctx.dim,
                ctx.nsrc,
                ctx.scale,
                BLOCK_M=ctx.block_m,
                BLOCK_D=ctx.block_d,
                MAX_SOURCES=FUSED_ATTNRES_MAX_SOURCES,
                QUERY_BF16=ctx.query_bf16,
                num_warps=8,
            )
            dquery = dquery_partials[:, : ctx.dim].sum(dim=0)
            dbias_reduced = dbias_partials.sum(dim=0).to(dtype=query.dtype)
            dbias = torch.zeros((ctx.bias_numel,), device=grad_out.device, dtype=query.dtype)
            dbias_n = min(ctx.bias_numel, FUSED_ATTNRES_MAX_SOURCES)
            dbias[:dbias_n] = dbias_reduced[:dbias_n]
            return (
                dquery,
                dbias,
                *dvalues,
                *([None] * (FUSED_ATTNRES_MAX_SOURCES - ctx.nsrc)),
                *dkeys,
                *([None] * (FUSED_ATTNRES_MAX_SOURCES - ctx.nsrc)),
                None,
                None,
                None,
            )
else:
    _FusedFullDepthMixerFn = None

def fused_full_depth_mixer(
    values: list[Tensor],
    keys: list[Tensor],
    query: Tensor,
    bias: Tensor,
    scale: float,
    use_gate: bool,
    block_m: int,
) -> Tensor | None:
    if (
        _FusedFullDepthMixerFn is None
        or use_gate
        or not values
        or len(values) != len(keys)
        or len(values) > FUSED_ATTNRES_MAX_SOURCES
        or len(values) <= 1
        or not values[0].is_cuda
    ):
        return None
    dim = values[0].shape[-1]
    block_d = 1 << (dim - 1).bit_length()
    if block_d > 1024:
        return None
    for value, key in zip(values, keys, strict=True):
        if value.shape != values[0].shape or key.shape != values[0].shape:
            return None
        if not value.is_contiguous() or not key.is_contiguous():
            return None
    padded_values = values + [values[0]] * (FUSED_ATTNRES_MAX_SOURCES - len(values))
    padded_keys = keys + [keys[0]] * (FUSED_ATTNRES_MAX_SOURCES - len(keys))
    return _FusedFullDepthMixerFn.apply(
        query,
        bias,
        *padded_values,
        *padded_keys,
        scale,
        len(values),
        block_m,
    )
try:
    fused_full_depth_mixer = torch._dynamo.disable(fused_full_depth_mixer)
except Exception:
    pass
class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim), dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype), None)
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
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.xsa_enabled = xsa_enabled
        self.xsa_learned = xsa_learned
        if xsa_enabled and xsa_learned:
            self.xsa_lambda = nn.Parameter(torch.full((num_heads,), float(xsa_lambda_init), dtype=torch.float32))
        else:
            self.xsa_lambda_value = float(xsa_lambda_init if xsa_enabled else 0.0)
    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = rms_norm(q).to(dtype=COMPUTE_DTYPE)
        k = rms_norm(k).to(dtype=COMPUTE_DTYPE)
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
            rep = self.num_heads // self.num_kv_heads
            y_grouped = y.reshape(bsz, self.num_kv_heads, rep, seqlen, self.head_dim)
            v_self = v.to(dtype=y.dtype)[:, :, None, :, :]
            v_self = v_self * torch.rsqrt(torch.sum(v_self * v_self, dim=-1, keepdim=True) + 1e-6)
            proj = torch.sum(y_grouped * v_self, dim=-1, keepdim=True) * v_self
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
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc = CastedLinear(dim, hidden_dim)
        self.proj = CastedLinear(hidden_dim, dim)
    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x * x)
class BlockDepthMixer(nn.Module):
    def __init__(self, dim: int, max_sources: int, num_loops: int, loop_specific: bool, use_gate: bool):
        super().__init__()
        self.max_sources = max_sources
        self.loop_specific = loop_specific
        self.use_gate = use_gate
        self.num_slots = max(num_loops, 1) if loop_specific else 1
        self.attnres_query = nn.Parameter(torch.zeros((self.num_slots, dim), dtype=torch.float32))
        self.attnres_bias = nn.Parameter(torch.zeros((self.num_slots, max_sources), dtype=torch.float32))
        if use_gate:
            self.attnres_gate = nn.Parameter(torch.zeros((self.num_slots, max_sources), dtype=torch.float32))
        self.scale = dim ** -0.5
    def forward(self, sources: list[Tensor], loop_idx: int = 0) -> Tensor:
        if not sources:
            raise ValueError("DepthMixer requires at least one source")
        if len(sources) > self.max_sources:
            sources = sources[:1] + sources[-(self.max_sources - 1):]
        slot = loop_idx if self.loop_specific else 0
        if slot >= self.num_slots:
            raise ValueError(f"loop_idx={loop_idx} exceeds available AttnRes slots={self.num_slots}")
        stacked = torch.stack(sources, dim=2)  # [B, T, sources, D]
        nsrc = len(sources)
        query = self.attnres_query[slot].to(dtype=stacked.dtype)
        keys = rms_norm(stacked)
        scores = torch.sum(keys * query[None, None, None, :], dim=-1) * self.scale
        scores = scores.float() + self.attnres_bias[slot, :nsrc][None, None, :]
        if self.use_gate:
            scores = scores + torch.tanh(self.attnres_gate[slot, :nsrc])[None, None, :]
        weights = torch.softmax(scores, dim=-1).to(dtype=stacked.dtype)
        return torch.sum(weights[:, :, :, None] * stacked, dim=2)
class AttnResHistory:
    def __init__(self, first_source: Tensor, max_sources: int):
        self.max_sources = max_sources
        self.values: list[Tensor] = []
        self.keys: list[Tensor] = []
        self.total: Tensor | None = None
        self.append(first_source)
    def __len__(self) -> int:
        return len(self.values)
    def _check_capacity(self) -> None:
        if len(self.values) >= self.max_sources:
            raise ValueError(
                f"Full AttnRes source history would exceed ATTNRES_MAX_SOURCES={self.max_sources}. Increase ATTNRES_MAX_SOURCES."
            )
    def append(self, source: Tensor) -> None:
        self._check_capacity()
        self.values.append(source)
        self.keys.append(rms_norm(source))
        self.total = source if self.total is None else self.total + source
    def mean(self) -> Tensor:
        if self.total is None or not self.values:
            raise ValueError("AttnResHistory is empty")
        return self.total / float(len(self.values))
    def snapshot(self) -> tuple[int, Tensor]:
        if self.total is None:
            raise ValueError("AttnResHistory is empty")
        return len(self.values), self.total
    def mean_since(self, snapshot: tuple[int, Tensor]) -> Tensor:
        if self.total is None:
            raise ValueError("AttnResHistory is empty")
        start_count, start_total = snapshot
        n = len(self.values) - start_count
        if n <= 0:
            raise ValueError("mean_since requires at least one new source")
        return (self.total - start_total) / float(n)
class FullDepthMixer(nn.Module):
    def __init__(self, dim: int, max_sources: int, num_loops: int, loop_specific: bool, use_gate: bool):
        super().__init__()
        self.max_sources = max_sources
        self.loop_specific = loop_specific
        self.use_gate = use_gate
        self.fused = bool(int(os.environ.get("FUSED_ATTNRES", "1")))
        self.fused_block_m = max(1, int(os.environ.get("FUSED_ATTNRES_BLOCK_M", "4")))
        self.num_slots = max(num_loops, 1) if loop_specific else 1
        self.attnres_query = nn.Parameter(torch.zeros((self.num_slots, dim), dtype=torch.float32))
        self.attnres_bias = nn.Parameter(torch.zeros((self.num_slots, max_sources), dtype=torch.float32))
        if use_gate:
            self.attnres_gate = nn.Parameter(torch.zeros((self.num_slots, max_sources), dtype=torch.float32))
        self.scale = dim ** -0.5
    def forward(self, sources: AttnResHistory | list[Tensor], source_keys: list[Tensor] | None = None, loop_idx: int = 0) -> Tensor:
        if isinstance(sources, AttnResHistory):
            values = sources.values
            keys = sources.keys
            nsrc = len(sources)
            if source_keys is not None:
                raise ValueError("Pass either AttnResHistory or explicit source_keys, not both")
        else:
            values = sources
            nsrc = len(values)
            keys = [rms_norm(src) for src in values] if source_keys is None else source_keys
        if nsrc <= 0:
            raise ValueError("DepthMixer requires at least one source")
        if nsrc > self.max_sources:
            raise ValueError(f"Full AttnRes source history has {nsrc} sources, but ATTNRES_MAX_SOURCES={self.max_sources}.")
        if len(keys) != nsrc:
            raise ValueError(f"DepthMixer got {nsrc} source values but {len(keys)} source keys")
        if nsrc == 1:
            return values[0]
        slot = loop_idx if self.loop_specific else 0
        if slot >= self.num_slots:
            raise ValueError(f"loop_idx={loop_idx} exceeds available AttnRes slots={self.num_slots}")
        if self.fused:
            fused = fused_full_depth_mixer(
                values,
                keys,
                self.attnres_query[slot],
                self.attnres_bias[slot],
                self.scale,
                self.use_gate,
                self.fused_block_m,
            )
            if fused is not None:
                return fused
        query = self.attnres_query[slot].to(dtype=keys[0].dtype)
        scores = torch.stack([torch.sum(key * query, dim=-1) * self.scale for key in keys], dim=-1).float()
        scores = scores + self.attnres_bias[slot, :nsrc][None, None, :]
        if self.use_gate:
            scores = scores + torch.tanh(self.attnres_gate[slot, :nsrc])[None, None, :]
        weights = torch.softmax(scores, dim=-1).to(dtype=values[0].dtype)
        out = values[0] * weights[:, :, 0:1]
        for i in range(1, nsrc):
            out = out + values[i] * weights[:, :, i : i + 1]
        return out
class LoopedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_hidden_dim: int,
        rope_base: float,
        qk_gain_init: float,
        xsa_enabled: bool = False,
        xsa_lambda_init: float = 1.0,
        xsa_learned: bool = False,
        attnres_enabled: bool = False,
        attnres_mode: str = "block",
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
        self.mlp = MLP(dim, mlp_hidden_dim)
        self.attn_scale = nn.Parameter(torch.ones((dim,), dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones((dim,), dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.attnres_enabled = attnres_enabled
        self.attnres_mode = attnres_mode
        if attnres_enabled:
            mixer_cls = FullDepthMixer if attnres_mode == "full" else BlockDepthMixer
            self.attn_mixer = mixer_cls(dim, attnres_max_sources, attnres_num_loops, attnres_loop_specific, attnres_use_gate)
            self.mlp_mixer = mixer_cls(dim, attnres_max_sources, attnres_num_loops, attnres_loop_specific, attnres_use_gate)
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
            attn_in = self.attn_mixer(attn_sources, loop_idx=loop_idx)
            attn_out = self.attn(self.attn_norm(attn_in))
            x = residual + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        else:
            mix = self.resid_mix.to(dtype=x.dtype)
            x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            attn_out = self.attn(self.attn_norm(x))
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        if self.attnres_enabled and mlp_sources is not None:
            mlp_in = self.mlp_mixer(mlp_sources, loop_idx=loop_idx)
        else:
            mlp_in = x
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(mlp_in))
        return x
    def full_attnres_forward(self, history: AttnResHistory, x0: Tensor, loop_idx: int = 0) -> Tensor:
        attn_in = self.attn_mixer(history, loop_idx=loop_idx)
        attn_out = self.attn_scale.to(dtype=attn_in.dtype)[None, None, :] * self.attn(self.attn_norm(attn_in))
        history.append(attn_out)
        mlp_in = self.mlp_mixer(history, loop_idx=loop_idx)
        mlp_out = self.mlp_scale.to(dtype=mlp_in.dtype)[None, None, :] * self.mlp(self.mlp_norm(mlp_in))
        history.append(mlp_out)
        return mlp_out
class LoopedGPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        if args.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {args.logit_softcap}")
        if not args.tie_embeddings:
            raise NotImplementedError("Converted MLX variants only support tied embeddings")
        self.logit_chunk_tokens = args.logit_chunk_tokens
        self.logit_softcap = args.logit_softcap
        self.arch_name = args.arch_name
        self.looped = args.looped
        self.prelude_layers = args.prelude_layers if args.looped else args.num_layers // 2
        self.core_layers = args.core_layers if args.looped else 0
        self.coda_layers = args.coda_layers if args.looped else args.num_layers - self.prelude_layers
        self.num_loops = max(args.num_loops, 1) if args.looped else 1
        self.stable_recurrence = bool(args.stable_recurrence and args.looped)
        self.attnres_enabled = bool(args.attnres_enabled and args.looped)
        self.attnres_mode = args.attnres_mode
        self.attnres_prev_loop_blocks = bool(args.attnres_prev_loop_blocks and self.attnres_enabled)
        self.attnres_max_sources = args.attnres_max_sources
        dim = args.model_dim
        self.tok_emb = nn.Embedding(args.vocab_size, dim)
        total_blocks = self.prelude_layers + self.core_layers + self.coda_layers if args.looped else args.num_layers
        blocks: list[LoopedBlock] = []
        for i in range(total_blocks):
            in_core = args.looped and self.prelude_layers <= i < self.prelude_layers + self.core_layers
            attnres_on = bool(args.attnres_enabled and args.looped)
            loop_specific_mixer = bool(attnres_on and in_core)
            blocks.append(
                LoopedBlock(
                    dim,
                    args.num_heads,
                    args.num_kv_heads,
                    args.mlp_mult * dim,
                    args.rope_base,
                    args.qk_gain_init,
                    xsa_enabled=bool(args.xsa_enabled and in_core),
                    xsa_lambda_init=args.xsa_lambda_init,
                    xsa_learned=bool(args.xsa_learned and in_core),
                    attnres_enabled=attnres_on,
                    attnres_mode=args.attnres_mode,
                    attnres_loop_specific=loop_specific_mixer,
                    attnres_max_sources=args.attnres_max_sources,
                    attnres_num_loops=self.num_loops if loop_specific_mixer else 1,
                    attnres_use_gate=bool(args.attnres_use_gate),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNormNoWeight()
        if self.attnres_enabled and self.attnres_mode == "full":
            self.final_mixer = FullDepthMixer(dim, args.attnres_max_sources, 1, False, False)
        if self.stable_recurrence:
            decay = min(max(args.recurrence_decay_init, 1e-4), 1.0 - 1e-4)
            decay_logit = math.log(decay / (1.0 - decay))
            self.recur_decay_logit = nn.Parameter(torch.full((dim,), decay_logit, dtype=torch.float32))
            self.prelude_inject_scale = nn.Parameter(torch.full((dim,), args.recurrence_inject_scale, dtype=torch.float32))
        self._init_weights(args.tied_embed_init_std)
    def _init_weights(self, tied_embed_init_std: float) -> None:
        with torch.no_grad():
            for b in self.blocks:
                b.attn.proj.weight.zero_()
                b.mlp.proj.weight.zero_()
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
            self.tok_emb.weight.data = self.tok_emb.weight.data.to(dtype=COMPUTE_DTYPE)
    def softcap(self, logits: Tensor) -> Tensor:
        c = self.logit_softcap
        return c * torch.tanh(logits / c)
    def _source_window(self, sources: list[Tensor]) -> list[Tensor]:
        if len(sources) <= self.attnres_max_sources:
            return sources
        return sources[:1] + sources[-(self.attnres_max_sources - 1):]
    def _stable_loop_input(self, h: Tensor, e: Tensor) -> Tensor:
        decay = torch.sigmoid(self.recur_decay_logit.to(dtype=h.dtype)).reshape(1, 1, -1)
        inject = self.prelude_inject_scale.to(dtype=h.dtype).reshape(1, 1, -1)
        return rms_norm(decay * h + (1.0 - decay) * inject * e)
    def forward_features(self, input_ids: Tensor) -> Tensor:
        x = rms_norm(self.tok_emb(input_ids).to(dtype=COMPUTE_DTYPE))
        x0 = x
        if not self.looped:
            for block in self.blocks:
                x = block(x, x0)
            return self.final_norm(x)
        if self.attnres_enabled and self.attnres_mode == "full":
            full_history = AttnResHistory(x0, self.attnres_max_sources)
            h = x0
            for i in range(self.prelude_layers):
                h = self.blocks[i].full_attnres_forward(full_history, x0, loop_idx=0)
            e = rms_norm(full_history.mean())
            h = e
            core_start = self.prelude_layers
            for loop_idx in range(self.num_loops):
                if self.stable_recurrence:
                    h = self._stable_loop_input(h, e)
                loop_snapshot = full_history.snapshot()
                full_history.append(h)
                for j in range(self.core_layers):
                    h = self.blocks[core_start + j].full_attnres_forward(full_history, e, loop_idx=loop_idx)
                if self.attnres_prev_loop_blocks:
                    full_history.append(full_history.mean_since(loop_snapshot))
            coda_start = self.prelude_layers + self.core_layers
            for j in range(self.coda_layers):
                h = self.blocks[coda_start + j].full_attnres_forward(full_history, e, loop_idx=0)
            h = self.final_mixer(full_history, loop_idx=0)
            return self.final_norm(h)
        prelude_states: list[Tensor] = [x]
        for i in range(self.prelude_layers):
            block = self.blocks[i]
            if self.attnres_enabled and block.attnres_enabled:
                sources = self._source_window(prelude_states)
                x = block(x, x0, attn_sources=sources, mlp_sources=sources, loop_idx=0)
            else:
                x = block(x, x0)
            prelude_states.append(x)
        e = rms_norm(x)
        h = e
        prev_loop_blocks: list[Tensor] = []
        core_start = self.prelude_layers
        for loop_idx in range(self.num_loops):
            if self.stable_recurrence:
                h = self._stable_loop_input(h, e)
            loop_start = h
            current_loop_blocks: list[Tensor] = []
            core_states: list[Tensor] = [loop_start]
            base_sources: list[Tensor] = [e]
            if self.attnres_prev_loop_blocks and prev_loop_blocks:
                base_sources.extend(prev_loop_blocks)
            for j in range(self.core_layers):
                block = self.blocks[core_start + j]
                if self.attnres_enabled and block.attnres_enabled:
                    attn_sources = self._source_window(base_sources + core_states)
                    h = block(h, e, attn_sources=attn_sources, mlp_sources=attn_sources, loop_idx=loop_idx)
                    current_loop_blocks.append(h)
                    core_states.append(h)
                else:
                    h = block(h, e, loop_idx=loop_idx)
            prev_loop_blocks = current_loop_blocks
        coda_start = self.prelude_layers + self.core_layers
        coda_states: list[Tensor] = [h]
        for j in range(self.coda_layers):
            block = self.blocks[coda_start + j]
            if self.attnres_enabled and block.attnres_enabled:
                sources = self._source_window([e] + coda_states)
                h = block(h, e, attn_sources=sources, mlp_sources=sources, loop_idx=0)
            else:
                h = block(h, e)
            coda_states.append(h)
        return self.final_norm(h)
    def loss(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.forward_features(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        embed_weight = self.tok_emb.weight.to(dtype=x.dtype)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = F.linear(x, embed_weight)
            logits = self.softcap(logits_proj)
            return F.cross_entropy(logits.float(), y, reduction="mean")
        loss_sum = torch.zeros((), device=x.device, dtype=torch.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits_proj = F.linear(x[s:e], embed_weight)
            logits = self.softcap(logits_proj)
            loss_sum = loss_sum + F.cross_entropy(logits.float(), y[s:e], reduction="sum")
        return loss_sum / float(n)
    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None) -> Tensor:
        return self.forward_features(input_ids) if target_ids is None else self.loss(input_ids, target_ids)
class MemoryBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_hidden_dim: int, rope_base: float, qk_gain_init: float, use_xsa: bool):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, xsa_enabled=use_xsa, xsa_lambda_init=1.0)
        self.mlp = MLP(dim, mlp_hidden_dim)
        self.attn_scale = nn.Parameter(torch.ones((dim,), dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones((dim,), dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0].reshape(1, 1, -1) * x + mix[1].reshape(1, 1, -1) * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype).reshape(1, 1, -1) * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype).reshape(1, 1, -1) * self.mlp(self.mlp_norm(x))
        return x
class MemoryMoRGPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        if args.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {args.logit_softcap}")
        if not args.tie_embeddings:
            raise NotImplementedError("Converted MLX variants only support tied embeddings")
        self.arch_name = args.arch_name
        self.dim = args.model_dim
        self.inv_sqrt_dim = args.model_dim ** -0.5
        self.logit_chunk_tokens = args.logit_chunk_tokens
        self.logit_softcap = args.logit_softcap
        self.use_depth_mix = args.use_block_attnres
        self.attnres_source_every = args.attnres_source_every
        self.max_depth_sources = args.max_depth_sources
        self.fused_depthmix = bool(args.fused_attnres)
        self.fused_block_m = args.fused_attnres_block_m
        self.memory_slots = max(0, args.memory_slots)
        self.extra_loop_passes = args.extra_loop_passes
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.blocks = nn.ModuleList(
            [
                MemoryBlock(
                    args.model_dim,
                    args.num_heads,
                    args.num_kv_heads,
                    args.mlp_hidden_dim,
                    args.rope_base,
                    args.qk_gain_init,
                    use_xsa=(i >= max(2, args.num_layers // 2)),
                )
                for i in range(args.num_layers)
            ]
        )
        self.depth_queries = nn.Parameter(torch.zeros((args.num_layers + 4, args.model_dim), dtype=torch.float32))
        self.depth_gates = nn.Parameter(torch.zeros((args.num_layers + 4, args.model_dim), dtype=torch.float32))
        self.register_buffer("_fused_depthmix_zero_bias", torch.zeros((FUSED_ATTNRES_MAX_SOURCES,), dtype=torch.float32), persistent=False)
        self.mem_q = CastedLinear(args.model_dim, args.model_dim)
        if self.memory_slots > 0:
            mk = (torch.randn((self.memory_slots, args.model_dim), dtype=torch.float32) * 0.01).to(dtype=COMPUTE_DTYPE)
            mv = (torch.randn((self.memory_slots, args.model_dim), dtype=torch.float32) * 0.01).to(dtype=COMPUTE_DTYPE)
        else:
            mk = torch.zeros((1, args.model_dim), dtype=COMPUTE_DTYPE)
            mv = torch.zeros((1, args.model_dim), dtype=COMPUTE_DTYPE)
        self.memory_k = nn.Parameter(mk)
        self.memory_v = nn.Parameter(mv)
        self.memory_gate = nn.Parameter(torch.zeros((args.model_dim,), dtype=torch.float32))
        self.router = CastedLinear(args.model_dim, 1)
        self.route_bias = nn.Parameter(torch.tensor([-1.0], dtype=torch.float32))
        self.extra_schedule = self._build_extra_schedule(args.num_layers, args.extra_loop_schedule)
        self.extra_loop_scales = nn.Parameter(torch.ones((max(1, len(self.extra_schedule)), args.model_dim), dtype=torch.float32) * 0.50)
        self.final_norm = RMSNormNoWeight()
        self._init_weights(args.tied_embed_init_std)
    def _init_weights(self, tied_embed_init_std: float) -> None:
        with torch.no_grad():
            for b in self.blocks:
                b.attn.proj.weight.zero_()
                b.mlp.proj.weight.zero_()
            self.mem_q.weight.zero_()
            self.router.weight.zero_()
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
            self.tok_emb.weight.data = self.tok_emb.weight.data.to(dtype=COMPUTE_DTYPE)
    def _build_extra_schedule(self, num_layers: int, raw: str) -> list[int]:
        raw = raw.strip()
        if raw:
            schedule = [int(x) for x in raw.replace(";", ",").split(",") if x.strip()]
            if not schedule or max(schedule) >= num_layers or min(schedule) < 0:
                raise ValueError(f"Invalid EXTRA_LOOP_SCHEDULE={raw!r} for NUM_LAYERS={num_layers}")
            return schedule
        if num_layers >= 7:
            return [4, 5, 6]
        return list(range(max(1, num_layers // 2), max(1, num_layers - 1)))
    def _depth_mix(self, x: Tensor, sources: list[Tensor], idx: int) -> Tensor:
        if not self.use_depth_mix or not sources:
            return x
        if self.fused_depthmix:
            values = sources + [x]
            keys = [rms_norm(value) for value in values]
            mixed = fused_full_depth_mixer(
                values,
                keys,
                self.depth_queries[idx],
                self._fused_depthmix_zero_bias[: len(values)],
                self.inv_sqrt_dim,
                False,
                self.fused_block_m,
            )
            if mixed is not None:
                gate = torch.tanh(self.depth_gates[idx]).to(dtype=x.dtype).reshape(1, 1, self.dim)
                return x + 0.5 * gate * (mixed - x)
        vals = torch.stack(sources + [x], dim=0)
        keys = rms_norm(vals)
        q = self.depth_queries[idx].to(dtype=x.dtype).reshape(1, 1, 1, self.dim)
        logits = torch.sum(keys * q, dim=-1) * self.inv_sqrt_dim
        weights = torch.softmax(logits.float(), dim=0).to(dtype=x.dtype)
        mixed = torch.sum(weights[..., None] * vals, dim=0)
        gate = torch.tanh(self.depth_gates[idx]).to(dtype=x.dtype).reshape(1, 1, self.dim)
        return x + 0.5 * gate * (mixed - x)
    def _memory_read(self, x: Tensor) -> Tensor:
        if self.memory_slots <= 0:
            return x
        q = rms_norm(self.mem_q(rms_norm(x))).to(dtype=COMPUTE_DTYPE)
        k = rms_norm(self.memory_k.to(dtype=q.dtype))
        logits = (q @ k.T) * self.inv_sqrt_dim
        weights = torch.softmax(logits.float(), dim=-1).to(dtype=q.dtype)
        mem = weights @ self.memory_v.to(dtype=q.dtype)
        gate = torch.tanh(self.memory_gate).to(dtype=q.dtype).reshape(1, 1, self.dim)
        return x + 0.5 * gate * mem
    def _adaptive_extra_loop(self, x: Tensor, x0: Tensor) -> Tensor:
        if self.extra_loop_passes <= 0 or not self.extra_schedule:
            return x
        route_logits = self.router(rms_norm(x)).float() + self.route_bias.float()
        route = torch.sigmoid(route_logits).to(dtype=x.dtype)
        for _ in range(self.extra_loop_passes):
            for j, block_idx in enumerate(self.extra_schedule):
                before = x
                block_out = self.blocks[block_idx](x, x0)
                scale = self.extra_loop_scales[j].to(dtype=x.dtype).reshape(1, 1, self.dim)
                x = before + route * scale * (block_out - before)
        return x
    def softcap(self, logits: Tensor) -> Tensor:
        c = self.logit_softcap
        return c * torch.tanh(logits / c)
    def forward_features(self, input_ids: Tensor) -> Tensor:
        x = rms_norm(self.tok_emb(input_ids).to(dtype=COMPUTE_DTYPE))
        x0 = x
        sources: list[Tensor] = [x0]
        mix_idx = 0
        for i, block in enumerate(self.blocks):
            x = self._depth_mix(x, sources, mix_idx)
            mix_idx += 1
            x = block(x, x0)
            if i == len(self.blocks) // 2:
                x = self._memory_read(x)
            if self.use_depth_mix and (i + 1) % self.attnres_source_every == 0:
                sources.append(x)
                if len(sources) > self.max_depth_sources:
                    sources = [sources[0]] + sources[-(self.max_depth_sources - 1):]
        x = self._memory_read(x)
        x = self._adaptive_extra_loop(x, x0)
        return self.final_norm(x)
    def loss(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.forward_features(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        embed_weight = self.tok_emb.weight.to(dtype=x.dtype)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = F.linear(x, embed_weight)
            logits = self.softcap(logits_proj)
            return F.cross_entropy(logits.float(), y, reduction="mean")
        loss_sum = torch.zeros((), device=x.device, dtype=torch.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits_proj = F.linear(x[s:e], embed_weight)
            logits = self.softcap(logits_proj)
            loss_sum = loss_sum + F.cross_entropy(logits.float(), y[s:e], reduction="sum")
        return loss_sum / float(n)
    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None) -> Tensor:
        return self.forward_features(input_ids) if target_ids is None else self.loss(input_ids, target_ids)
def make_model(args: Hyperparameters) -> nn.Module:
    if ARCH_KIND == "memory_mor_blockattn":
        return MemoryMoRGPT(args)
    return LoopedGPT(args)
# -----------------------------
# VALIDATION + TRAINING
# -----------------------------
@torch.no_grad()
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
def ddp_grad_accum_steps(args: Hyperparameters, world_size: int) -> int:
    if args.grad_accum_steps <= 0:
        raise ValueError(f"GRAD_ACCUM_STEPS must be positive, got {args.grad_accum_steps}")
    if args.grad_accum_steps % world_size != 0:
        raise ValueError(
            f"GRAD_ACCUM_STEPS={args.grad_accum_steps} must be divisible by WORLD_SIZE={world_size} "
            "to preserve the MLX effective batch exactly."
        )
    grad_accum_steps = args.grad_accum_steps // world_size
    if grad_accum_steps <= 0:
        raise ValueError(f"WORLD_SIZE={world_size} exceeds GRAD_ACCUM_STEPS={args.grad_accum_steps}")
    local_tokens = args.train_batch_tokens // (world_size * grad_accum_steps)
    if local_tokens % args.train_seq_len != 0:
        raise ValueError(
            f"TRAIN_BATCH_TOKENS={args.train_batch_tokens} gives per-rank microbatch {local_tokens}, "
            f"not divisible by TRAIN_SEQ_LEN={args.train_seq_len}."
        )
    return grad_accum_steps
def split_optimizer_params(base_model: nn.Module):
    named = [(name, p) for name, p in base_model.named_parameters() if p.requires_grad]
    embed_params = [p for name, p in named if name == "tok_emb.weight"]
    matrix_names = {
        name
        for name, p in named
        if name != "tok_emb.weight" and p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    }
    matrix_params = [p for name, p in named if name in matrix_names]
    scalar_params = [p for name, p in named if name != "tok_emb.weight" and name not in matrix_names]
    return embed_params, matrix_params, scalar_params, matrix_names
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
    grad_accum_steps = ddp_grad_accum_steps(args, world_size)
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
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    out_dir = Path(args.out_dir)
    logfile: Path | None = None
    if master_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        logfile = out_dir / f"{args.run_id}.txt"
        print(logfile)
    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with logfile.open("a", encoding="utf-8") as f:
                print(msg, file=f)
    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}")
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(args.data_path, args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    base_model = make_model(args).to(device)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=args.torch_compile_fullgraph) if args.torch_compile else base_model
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
        if distributed
        else compiled_model
    )
    embed_params, matrix_params, scalar_params, matrix_names = split_optimizer_params(base_model)
    if not embed_params:
        raise RuntimeError("tok_emb.weight was not found for tied embedding optimizer")
    optimizer_tok = torch.optim.Adam(
        [{"params": embed_params, "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon]
    if scalar_params:
        optimizer_scalar = torch.optim.Adam(
            [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_scalar)
    else:
        optimizer_scalar = None
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"run_id:{args.run_id}")
    log0(f"arch:{ARCH_NAME_DEFAULT} kind:{ARCH_KIND}")
    log0(f"model_params:{n_params} matrix_params:{sum(p.numel() for p in matrix_params)} scalar_params:{sum(p.numel() for p in scalar_params)}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps} source_grad_accum_steps:{args.grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}" + ("" if expected_train_files is None else f"/{expected_train_files}"))
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    log0(f"tokenizer_path:{args.tokenizer_path}")
    log0(
        f"vocab_size:{args.vocab_size} layers:{args.num_layers} dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
        f"seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings}"
    )
    log0(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} val_batch_size:{args.val_batch_size} "
        f"warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    if ARCH_KIND == "memory_mor_blockattn":
        log0(
            f"memory_slots:{args.memory_slots} use_block_attnres:{args.use_block_attnres} "
            f"extra_loop_passes:{args.extra_loop_passes} extra_schedule:{base_model.extra_schedule}"
        )
    else:
        log0(
            f"looped:{args.looped} prelude:{args.prelude_layers} core:{args.core_layers} coda:{args.coda_layers} loops:{args.num_loops} "
            f"stable_recurrence:{args.stable_recurrence} xsa:{args.xsa_enabled} attnres:{args.attnres_enabled} mode:{args.attnres_mode}"
        )
    log0(
        f"optimizer:muon+adam embed_lr:{args.tied_embed_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps}"
    )
    log0(f"torch_compile:{args.torch_compile} fullgraph:{args.torch_compile_fullgraph} compute_dtype:{COMPUTE_DTYPE}")
    log0(
        f"fused_attnres:{args.fused_attnres} triton_available:{TRITON_AVAILABLE} "
        f"max_sources:{FUSED_ATTNRES_MAX_SOURCES} block_m:{args.fused_attnres_block_m}"
    )
    log0(f"seed:{args.seed}")
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
    # Warmup primes compiled forward/backward/optimizer paths, then restores exact initial state.
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
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
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
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        lr_scale = args.lr_mul(step, elapsed_ms)
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
                group["lr"] = group["base_lr"] * lr_scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
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
    raw_path = out_dir / f"{args.run_id}_model.pt"
    quant_path = out_dir / f"{args.run_id}_model.int8.ptz"
    if master_process:
        torch.save(base_model.state_dict(), raw_path)
        model_bytes = raw_path.stat().st_size
        code_bytes = len(code.encode("utf-8"))
        log0(f"saved_model:{raw_path} bytes:{model_bytes}")
        log0(f"code_size:{code_bytes} bytes total_submission_size:{model_bytes + code_bytes} bytes")
        quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_raw = quant_buf.getvalue()
        quant_blob = zlib.compress(quant_raw, level=9)
        with quant_path.open("wb") as f:
            f.write(quant_blob)
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"serialized_model_int8_zlib:{quant_path.stat().st_size} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{len(quant_raw)} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"total_submission_size_int8_zlib:{quant_path.stat().st_size + code_bytes} bytes")
    if distributed:
        dist.barrier()
    with quant_path.open("rb") as f:
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
