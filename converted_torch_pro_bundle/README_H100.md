# CUDA/PyTorch conversions for the fast MLX training scripts

This folder contains PyTorch/CUDA versions of the four MLX scripts in the uploaded archive. They are self-contained and use DDP/NCCL, bf16 autocast, FlashAttention SDPA, Muon+Adam optimizer splits, and the same int8+zlib roundtrip export path as the PyTorch baseline.

## 8x H100 launch

Run one script per experiment, for example:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt_torch_sp8192_loop_xsa_fastcuda.py
```

Typical dataset/tokenizer overrides:

```bash
DATA_PATH=/path/to/data/datasets/fineweb10B_sp8192 TOKENIZER_PATH=/path/to/data/tokenizers/fineweb_8192_bpe.model torchrun --standalone --nproc_per_node=8 train_gpt_torch_sp8192_loop_xsa_attnres_loopq_fastcuda.py
```

The scripts preserve the MLX single-process effective batch by interpreting `GRAD_ACCUM_STEPS` as the original MLX accumulation count and dividing it across ranks. With the default `GRAD_ACCUM_STEPS=8`, an 8-GPU run uses one microstep per rank while keeping `TRAIN_BATCH_TOKENS` unchanged.

`TORCH_COMPILE=1` is enabled by default with `fullgraph=False`; set `TORCH_COMPILE=0` if you need easier debugging.
