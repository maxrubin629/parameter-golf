# Fused Torch Model Variants

This folder contains copied model scripts with Triton fused depth/AttnRes mixer paths.
The originals in `converted_torch_pro_bundle/` are intentionally untouched.

## Scripts

- `train_gpt_torch_loop_fullattnres_loopq_fused_attnres_fastcuda.py`
  - Fuses the `FullDepthMixer` path for `converted_loop_fullattnres_loopq`.
- `train_gpt_torch_sp8192_loop_xsa_attnres_loopq_fused_attnres_fastcuda.py`
  - Fuses the `BlockDepthMixer` path for `converted_sp8192_loop_xsa_attnres_loopq`.
- `train_gpt_torch_memory_mor_blockattn_fused_depthmix_fastcuda.py`
  - Fuses the core weighted source mixer used by memory-MoR `_depth_mix`, then applies the existing gate update.

## Runtime Switches

Fused paths are on by default:

```bash
FUSED_ATTNRES=1
FUSED_ATTNRES_BLOCK_M=4
```

Set `FUSED_ATTNRES=0` to force the original PyTorch mixer path.

## Parity Check

On a CUDA/Triton machine:

```bash
python fused_kernel_torch_models/check_fused_mixers.py
```
