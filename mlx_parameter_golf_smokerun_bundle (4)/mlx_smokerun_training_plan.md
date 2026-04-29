# MLX smokerun training plan for the three tiny Parameter Golf architectures

This plan is for fast local checks on a Mac before spending 8×H100 time. The goal is not to prove final BPB locally; it is to catch shape bugs, memory pressure, unstable loss, bad optimizer grouping, tokenizer mismatches, and quantization/serialization failures.

## Files in this bundle

| File | Architecture | Local smokerun purpose |
|---|---|---|
| `train_gpt_mlx_xsa_parcae_ladder.py` | 10 physical 384d blocks, 15 virtual block applications, XSA, gated BlockAttnRes-lite | Safest first bet. Tests recurrence + XSA + depth mixing. |
| `train_gpt_mlx_hyperngram_512_core.py` | 5 physical 512d blocks, looped 1-2-3 core, XSA, neural n-gram hidden residual | High-upside width/local-statistics bet. Tests whether 512d + local residual learns quickly. |
| `train_gpt_mlx_memory_mor_blockattn.py` | 8 physical 384d blocks, learned memory bank, gated depth mixing, adaptive extra loop | Weird/high-upside bet. Tests memory reads and token-wise extra passes. |

Approximate default SP8192 parameter counts: `xsa_parcae_ladder` ≈ 16.94M, `hyperngram_512_core` ≈ 16.27M, and `memory_mor_blockattn` ≈ 14.72M with `MEMORY_SLOTS=512` (≈ 15.90M with `MEMORY_SLOTS=2048`).

All three scripts inherit the important baseline mechanics from `train_gpt_mlx.py`: MLX training, SentencePiece validation BPB, Muon+Adam split, tied embeddings, logit softcap, int8+zlib artifact serialization, and quantized roundtrip validation.

## Assumptions

The scripts default to SP8192:

```bash
DATA_PATH=./data/datasets/fineweb10B_sp8192
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
VOCAB_SIZE=8192
```

If your Mac checkout only has the public SP1024 cache, run the same scripts with SP1024 overrides. That is useful for smoke testing code paths, but it is not a faithful architecture comparison for the SP8192 target.

```bash
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
```

## Install / data checklist

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install mlx sentencepiece numpy
```

For public SP1024 smoke tests:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

For SP8192, point `DATA_PATH`, `TOKENIZER_PATH`, and `VOCAB_SIZE` at your local SP8192 export. The scripts intentionally fail fast if the tokenizer vocab size and `VOCAB_SIZE` disagree.

## Phase 0: syntax and parameter-count check

Run this before training:

```bash
python3 -m py_compile train_gpt_mlx_xsa_parcae_ladder.py
python3 -m py_compile train_gpt_mlx_hyperngram_512_core.py
python3 -m py_compile train_gpt_mlx_memory_mor_blockattn.py
```

Then run one tiny compile step. This is mostly to make sure MLX captures the architecture state correctly.

```bash
COMMON='ITERATIONS=1 WARMUP_STEPS=0 TRAIN_SEQ_LEN=128 TRAIN_BATCH_TOKENS=4096 GRAD_ACCUM_STEPS=4 MLX_MAX_MICROBATCH_TOKENS=1024 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 LOGIT_CHUNK_TOKENS=1024'

env $COMMON RUN_ID=smoke_xsa python3 train_gpt_mlx_xsa_parcae_ladder.py
env $COMMON RUN_ID=smoke_ngram python3 train_gpt_mlx_hyperngram_512_core.py
env $COMMON RUN_ID=smoke_mem MEMORY_SLOTS=128 EXTRA_LOOP_PASSES=1 python3 train_gpt_mlx_memory_mor_blockattn.py
```

Expected result: each run prints `model_params`, trains one step, saves raw and int8+zlib artifacts, and completes `final_int8_zlib_roundtrip`. With `VOCAB_SIZE=8192`, the default parameter counts should remain under 18M.

## Phase 1: five-minute Mac smoke

This is the first meaningful loss-decrease test. Keep validation off to avoid spending most of the run scanning the fixed val split.

```bash
export TRAIN_SEQ_LEN=256
export TRAIN_BATCH_TOKENS=16384
export GRAD_ACCUM_STEPS=8
export MLX_MAX_MICROBATCH_TOKENS=2048
export LOGIT_CHUNK_TOKENS=2048
export ITERATIONS=80
export WARMUP_STEPS=2
export VAL_LOSS_EVERY=0
export MAX_WALLCLOCK_SECONDS=300
export TRAIN_LOG_EVERY=5
export MLX_EAGER_EVAL=1

RUN_ID=mac5_xsa python3 train_gpt_mlx_xsa_parcae_ladder.py
RUN_ID=mac5_ngram python3 train_gpt_mlx_hyperngram_512_core.py
RUN_ID=mac5_mem MEMORY_SLOTS=256 EXTRA_LOOP_PASSES=1 python3 train_gpt_mlx_memory_mor_blockattn.py
```

Pass criteria:

1. No compile/capture errors.
2. No obvious NaNs or loss explosions.
3. `train_loss` decreases over the first 20–80 optimizer steps.
4. Final quantized roundtrip runs without a huge BPB/loss mismatch.

## Phase 2: local architecture ranking smoke

Run enough steps to expose instability, but still short enough that you can iterate repeatedly.

```bash
export TRAIN_SEQ_LEN=512
export TRAIN_BATCH_TOKENS=32768
export GRAD_ACCUM_STEPS=8
export MLX_MAX_MICROBATCH_TOKENS=4096
export LOGIT_CHUNK_TOKENS=4096
export ITERATIONS=300
export WARMUP_STEPS=3
export VAL_LOSS_EVERY=100
export VAL_BATCH_SIZE=65536
export MAX_WALLCLOCK_SECONDS=1200
export TRAIN_LOG_EVERY=10

RUN_ID=rank_xsa_s1337 SEED=1337 python3 train_gpt_mlx_xsa_parcae_ladder.py
RUN_ID=rank_ngram_s1337 SEED=1337 python3 train_gpt_mlx_hyperngram_512_core.py
RUN_ID=rank_mem_s1337 SEED=1337 MEMORY_SLOTS=256 EXTRA_LOOP_PASSES=1 python3 train_gpt_mlx_memory_mor_blockattn.py
```

Then repeat only the best two with `SEED=42` and `SEED=2026`. Do not overfit to one seed; recurrent/gated variants can look great or terrible depending on the first few hundred steps.

## Phase 3: knobs to try if a run fails

### Universal memory relief

Use these on 16GB Macs or if MLX graph growth is painful:

```bash
export TRAIN_SEQ_LEN=256
export TRAIN_BATCH_TOKENS=8192
export GRAD_ACCUM_STEPS=8
export MLX_MAX_MICROBATCH_TOKENS=1024
export LOGIT_CHUNK_TOKENS=1024
export MLX_EAGER_EVAL=1
```

### Universal stability relief

```bash
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export GRAD_CLIP_NORM=0.5
export QK_GAIN_INIT=3.5
```

### XSA-Parcae Ladder specific

```bash
# Less recurrent injection at the start.
export LOOP_SCALE_INIT=0.65

# Disable depth mixer to isolate XSA/recurrence.
export USE_BLOCK_ATTNRES=0

# Delay XSA if early training is jagged.
export XSA_START_LAYER=5
```

### HyperNgram specific

```bash
# Fewer virtual layers for very quick tests.
export CORE_CYCLES=2

# Lower recurrent injection.
export LOOP_SCALE_INIT=0.65
```

### Memory-MoR specific

```bash
# Shrink memory first, then re-enable after the rest is stable.
export MEMORY_SLOTS=128

# Isolate memory without adaptive extra loops.
export EXTRA_LOOP_PASSES=0

# Isolate adaptive loops without memory pressure.
export MEMORY_SLOTS=0
```

## Phase 4: what to promote to 8×H100

Promote a variant only if all of these are true locally:

1. It survives one-step compile, short training, validation, final save, and int8+zlib roundtrip.
2. It has a smoother first-300-step loss curve than the baseline or at least no worse than baseline.
3. Post-quant validation is close to pre-quant validation.
4. Parameter count is under the target when `VOCAB_SIZE=8192`.

Recommended promotion order:

1. `xsa_parcae_ladder`: safest, most likely to give a clean signal.
2. `hyperngram_512_core`: try if XSA-Parcae is stable; this has the best width/local-statistics upside.
3. `memory_mor_blockattn`: try after the others; reduce `MEMORY_SLOTS` and `EXTRA_LOOP_PASSES` if it wastes too much wall clock.

## Notes on interpreting Mac results

Mac smokeruns are primarily debugging runs. They are useful for detecting instability and implementation mistakes, but they are weak predictors of final BPB because the serious setting has much larger token throughput, longer context, different wall-clock constraints, and possibly SP8192-specific behavior. Treat local BPB as a sanity check, not as a leaderboard estimate.
