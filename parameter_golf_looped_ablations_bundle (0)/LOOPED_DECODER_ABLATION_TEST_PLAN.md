# Parameter Golf looped-decoder ablation test plan

This bundle contains a baseline MLX run plus a cleaned-up ablation ladder for the architecture idea:

> a stable looped decoder-only transformer that shares the expensive recurrent transformer weights across loop iterations, uses Block Attention Residuals across **all** transformer stages when AttnRes is enabled, gives the recurrent section its own loop-specific AttnRes mixer slots per iteration, and optionally uses XSA inside recurrent sequence attention.

The goal is **not inference-time efficiency**. The goal is lower **validation BPB per fixed training wall-clock budget** on your MacBook while keeping the eventual artifact close to the 16 MB limit. Looping is attractive because it spends extra compute while adding only small routing/mixer parameters.

## Important correction from v2

There is no longer an `ATTNRES_ALL_BLOCKS` switch.

The ablation distinction is now clean:

1. **Standard additive residuals everywhere**: `loop_base`, `loop_stable`, `loop_xsa`.
2. **Block Attention Residuals everywhere**: all `attnres` variants.

For AttnRes variants:

- prelude blocks use Block AttnRes;
- recurrent-core blocks use Block AttnRes;
- coda blocks use Block AttnRes;
- the recurrent core gets separate learned mixer slots for each loop iteration;
- prelude and coda use a single mixer slot because they run once.

This follows the Attention Residuals paper’s core setup: replace fixed additive residual accumulation with depth-wise softmax selection over RMSNormed block/source states using learned pseudo-query vectors. In these compact golf scripts, the source set is implemented as block/stage summaries plus the current partial stage state, capped by `ATTNRES_MAX_SOURCES`.

## Files in this bundle

Copy these files into the root of your Parameter Golf repo, next to `train_gpt_mlx.py`.

| File | Purpose |
|---|---|
| `train_gpt_mlx_pg_baseline.py` | Exact copy of your current MLX baseline. Use this as the reference run. |
| `train_gpt_mlx_loop_base.py` | Prelude/core/coda looped decoder, shared recurrent core, additive residuals, no stable recurrence, no XSA. |
| `train_gpt_mlx_loop_stable.py` | Adds stable recurrent input mixing to the looped decoder; still additive residuals everywhere. |
| `train_gpt_mlx_loop_xsa.py` | Stable looped decoder + XSA in recurrent-core sequence attention; still additive residuals everywhere. |
| `train_gpt_mlx_loop_attnres_loopq.py` | Stable looped decoder + Block AttnRes in prelude/core/coda; recurrent core has per-loop mixer slots; no XSA. |
| `train_gpt_mlx_loop_xsa_attnres_loopq.py` | Main target: stable looped decoder + XSA + all-stage Block AttnRes + per-loop recurrent mixer slots. |
| `train_gpt_mlx_loop_xsa_attnres_loopq_gate.py` | Main target + loop-specific source gates in the AttnRes mixers. |
| `train_gpt_mlx_loop_xsa_attnres_loopq_prevblocks.py` | Main target + previous-loop block summaries as extra recurrent-core mixer sources. |
| `train_gpt_mlx_current_record.py` | MLX architecture port of the 2026-04-09 SP8192 current record: 11 physical layers, layers 3-5 looped into a 17-application schedule, partial RoPE, XSA, parallel residuals from layer 7, gated skips, and QK gain 5.25. |

## Shared command shape

Use your requested MacBook command shape:

```bash
RUN_ID=<your_arch_name> \
TRAIN_BATCH_TOKENS=524288 \
GRAD_ACCUM_STEPS=8 \
TRAIN_SEQ_LEN=1024 \
MLX_MAX_MICROBATCH_TOKENS=8192 \
MLX_EAGER_EVAL=1 \
uv run train_gpt_mlx_<your_arch_name>.py
```

Recommended default architecture knobs for the looped runs are already set in each file:

```bash
PRELUDE_LAYERS=2
CORE_LAYERS=2
CODA_LAYERS=2
NUM_LOOPS=3
ATTNRES_MAX_SOURCES=6
```

That gives 6 stored transformer blocks and 10 virtual block applications: `2 + 2*3 + 2`. This is a reasonable first comparison against the 9-layer baseline because it tests whether parameter-efficient recurrence buys better BPB under the same wall-clock training budget.

## Current-record MLX reference

`train_gpt_mlx_current_record.py` is a separate SP8192 reference architecture, not part of the SP1024 looped-decoder ladder above.

Its defaults follow the record README:

```bash
VOCAB_SIZE=8192
TRAIN_SEQ_LEN=2048
NUM_LAYERS=11
MLP_MULT=4.0
ROPE_DIMS=16
QK_GAIN_INIT=5.25
NUM_LOOPS=2
LOOP_START=3
LOOP_END=5
ENABLE_LOOPING_AT=0.35
PARALLEL_RESIDUAL_START=7
SKIP_GATES_ENABLED=1
XSA_LAST_N=11
```

That yields the record's virtual schedule once looping activates:

- encoder: `[0,1,2,3,4,5,3,4]`;
- decoder: `[5,3,4,5,6,7,8,9,10]`.

Use explicit SP8192 dataset/tokenizer paths if your checkout does not already have them at the defaults:

```bash
RUN_ID=current_record_mlx \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
TRAIN_BATCH_TOKENS=786432 \
GRAD_ACCUM_STEPS=8 \
TRAIN_SEQ_LEN=2048 \
MLX_MAX_MICROBATCH_TOKENS=8192 \
MLX_EAGER_EVAL=1 \
uv run train_gpt_mlx_current_record.py
```

This script intentionally keeps the MLX baseline's simple int8/zlib roundtrip path. It does **not** port the record's full GPTQ SDClip exporter, Brotli/LZMA wrapper, distributed H100 training path, or legal score-first TTT evaluator.

## Phase 0: environment sanity check

Run from the repo root:

```bash
uv sync
python - <<'PY'
from pathlib import Path
required = [
    Path('data/datasets/fineweb10B_sp1024'),
    Path('data/tokenizers/fineweb_1024_bpe.model'),
]
for p in required:
    print(p, 'OK' if p.exists() else 'MISSING')
PY
```

If the dataset/tokenizer paths differ, set `DATA_PATH` and `TOKENIZER_PATH` explicitly in the commands below.

## Phase 1: smoke tests first

Purpose: catch shape bugs, memory blow-ups, graph-compile problems, and variants that are much slower than expected. These are **not** final BPB comparisons.

Use a fixed 200-step smoke and one final validation:

```bash
COMMON="TRAIN_BATCH_TOKENS=524288 GRAD_ACCUM_STEPS=8 TRAIN_SEQ_LEN=1024 MLX_MAX_MICROBATCH_TOKENS=8192 MLX_EAGER_EVAL=1 ITERATIONS=200 MAX_WALLCLOCK_SECONDS=999999 WARMUP_STEPS=20 TRAIN_LOG_EVERY=10 VAL_LOSS_EVERY=0"
```

Run the smoke ladder:

```bash
env $COMMON RUN_ID=smoke_pg_baseline uv run train_gpt_mlx_pg_baseline.py
env $COMMON RUN_ID=smoke_loop_base uv run train_gpt_mlx_loop_base.py
env $COMMON RUN_ID=smoke_loop_stable uv run train_gpt_mlx_loop_stable.py
env $COMMON RUN_ID=smoke_loop_xsa uv run train_gpt_mlx_loop_xsa.py
env $COMMON RUN_ID=smoke_loop_attnres_loopq uv run train_gpt_mlx_loop_attnres_loopq.py
env $COMMON RUN_ID=smoke_loop_xsa_attnres_loopq uv run train_gpt_mlx_loop_xsa_attnres_loopq.py
```

Then, if the main target is healthy, smoke the two richer variants:

```bash
env $COMMON RUN_ID=smoke_loop_xsa_attnres_loopq_gate uv run train_gpt_mlx_loop_xsa_attnres_loopq_gate.py
env $COMMON RUN_ID=smoke_loop_xsa_attnres_loopq_prevblocks uv run train_gpt_mlx_loop_xsa_attnres_loopq_prevblocks.py
```

Smoke the SP8192 current-record MLX reference separately:

```bash
CURRENT_RECORD_COMMON="DATA_PATH=./data/datasets/fineweb10B_sp8192 TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model VOCAB_SIZE=8192 TRAIN_BATCH_TOKENS=786432 GRAD_ACCUM_STEPS=8 TRAIN_SEQ_LEN=2048 MLX_MAX_MICROBATCH_TOKENS=8192 MLX_EAGER_EVAL=1 ITERATIONS=200 MAX_WALLCLOCK_SECONDS=999999 WARMUP_STEPS=20 TRAIN_LOG_EVERY=10 VAL_LOSS_EVERY=0"
env $CURRENT_RECORD_COMMON RUN_ID=smoke_current_record_mlx uv run train_gpt_mlx_current_record.py
```

Smoke-pass criteria:

1. script reaches final int8/zlib roundtrip eval;
2. no NaNs or exploding train loss;
3. step time is not catastrophically worse than baseline;
4. final smoke BPB is not obviously broken relative to nearby variants;
5. quantized roundtrip BPB is not dramatically worse than pre-quant validation.

After smoke tests, compare logs:

```bash
grep -E "run_id:|model_params:|step:[0-9]+/[0-9]+ train_loss|final_int8_zlib_roundtrip val_loss|serialized_model_int8_zlib" logs/smoke_*.txt
```

## Phase 2: first real 10-minute ablation ladder

Run baseline first, then the additive-residual loop ladder, then the all-stage AttnRes ladder.

### Baseline

```bash
RUN_ID=pg_baseline \
TRAIN_BATCH_TOKENS=524288 \
GRAD_ACCUM_STEPS=8 \
TRAIN_SEQ_LEN=1024 \
MLX_MAX_MICROBATCH_TOKENS=8192 \
MLX_EAGER_EVAL=1 \
uv run train_gpt_mlx_pg_baseline.py
```

### Additive-residual loop ladder

```bash
RUN_ID=loop_base \
TRAIN_BATCH_TOKENS=524288 \
GRAD_ACCUM_STEPS=8 \
TRAIN_SEQ_LEN=1024 \
MLX_MAX_MICROBATCH_TOKENS=8192 \
MLX_EAGER_EVAL=1 \
uv run train_gpt_mlx_loop_base.py
```

```bash
RUN_ID=loop_stable \
TRAIN_BATCH_TOKENS=524288 \
GRAD_ACCUM_STEPS=8 \
TRAIN_SEQ_LEN=1024 \
MLX_MAX_MICROBATCH_TOKENS=8192 \
MLX_EAGER_EVAL=1 \
uv run train_gpt_mlx_loop_stable.py
```

```bash
RUN_ID=loop_xsa \
TRAIN_BATCH_TOKENS=524288 \
GRAD_ACCUM_STEPS=8 \
TRAIN_SEQ_LEN=1024 \
MLX_MAX_MICROBATCH_TOKENS=8192 \
MLX_EAGER_EVAL=1 \
uv run train_gpt_mlx_loop_xsa.py
```

### All-stage Block AttnRes ladder

```bash
RUN_ID=loop_attnres_loopq \
TRAIN_BATCH_TOKENS=524288 \
GRAD_ACCUM_STEPS=8 \
TRAIN_SEQ_LEN=1024 \
MLX_MAX_MICROBATCH_TOKENS=8192 \
MLX_EAGER_EVAL=1 \
uv run train_gpt_mlx_loop_attnres_loopq.py
```

```bash
RUN_ID=loop_xsa_attnres_loopq \
TRAIN_BATCH_TOKENS=524288 \
GRAD_ACCUM_STEPS=8 \
TRAIN_SEQ_LEN=1024 \
MLX_MAX_MICROBATCH_TOKENS=8192 \
MLX_EAGER_EVAL=1 \
uv run train_gpt_mlx_loop_xsa_attnres_loopq.py
```

Primary comparisons:

| Comparison | Question |
|---|---|
| `loop_base` vs `pg_baseline` | Does the raw looped scaffold survive the fixed wall-clock comparison? |
| `loop_stable` vs `loop_base` | Does stable recurrence matter? |
| `loop_xsa` vs `loop_stable` | Does XSA help the looped core by itself? |
| `loop_attnres_loopq` vs `loop_stable` | Does replacing additive residuals with all-stage Block AttnRes help? |
| `loop_xsa_attnres_loopq` vs `loop_attnres_loopq` | Does XSA help once depth-wise residual routing exists? |
| `loop_xsa_attnres_loopq` vs `loop_xsa` | Does all-stage Block AttnRes help once XSA exists? |
| `loop_xsa_attnres_loopq` vs `pg_baseline` | Does the full idea beat the starter under fixed wall-clock? |

## Phase 3: richer target variants

Run these only if `loop_xsa_attnres_loopq` is healthy in smoke and competitive in Phase 2.

### Loop-specific source gates

```bash
RUN_ID=loop_xsa_attnres_loopq_gate \
TRAIN_BATCH_TOKENS=524288 \
GRAD_ACCUM_STEPS=8 \
TRAIN_SEQ_LEN=1024 \
MLX_MAX_MICROBATCH_TOKENS=8192 \
MLX_EAGER_EVAL=1 \
uv run train_gpt_mlx_loop_xsa_attnres_loopq_gate.py
```

### Previous-loop block summaries

```bash
RUN_ID=loop_xsa_attnres_loopq_prevblocks \
TRAIN_BATCH_TOKENS=524288 \
GRAD_ACCUM_STEPS=8 \
TRAIN_SEQ_LEN=1024 \
MLX_MAX_MICROBATCH_TOKENS=8192 \
MLX_EAGER_EVAL=1 \
uv run train_gpt_mlx_loop_xsa_attnres_loopq_prevblocks.py
```

Interpretation:

- `gate` tests whether the mixer benefits from an extra learned source preference beyond query+bias softmax.
- `prevblocks` tests whether the loop should access prior-loop block summaries, not just the previous loop final state and current-loop partial states.

## Phase 4: loop-count and physical-layout sweeps

Only sweep the top 1-2 variants from Phase 2/3. Start with `train_gpt_mlx_loop_xsa_attnres_loopq.py`.

### Loop count sweep

```bash
for T in 2 3 4 5; do
  RUN_ID=loopq_T${T} \
  NUM_LOOPS=$T \
  TRAIN_BATCH_TOKENS=524288 \
  GRAD_ACCUM_STEPS=8 \
  TRAIN_SEQ_LEN=1024 \
  MLX_MAX_MICROBATCH_TOKENS=8192 \
  MLX_EAGER_EVAL=1 \
  uv run train_gpt_mlx_loop_xsa_attnres_loopq.py
done
```

Expected sweet spot: `NUM_LOOPS=3` or `4`. More loops may reduce optimizer steps enough to hurt BPB even if the architecture is more expressive.

### Prelude/core/coda sweep

```bash
for SHAPE in 2,2,2 2,3,2 3,2,2 2,3,3 3,3,2; do
  IFS=, read P C D <<< "$SHAPE"
  RUN_ID=loopq_shape_${P}_${C}_${D} \
  PRELUDE_LAYERS=$P CORE_LAYERS=$C CODA_LAYERS=$D NUM_LOOPS=3 \
  TRAIN_BATCH_TOKENS=524288 \
  GRAD_ACCUM_STEPS=8 \
  TRAIN_SEQ_LEN=1024 \
  MLX_MAX_MICROBATCH_TOKENS=8192 \
  MLX_EAGER_EVAL=1 \
  uv run train_gpt_mlx_loop_xsa_attnres_loopq.py
done
```

Expected sweet spot: a small recurrent middle, usually `2/2/2` or `2/3/2`, unless the coda is too weak to calibrate logits.

## Phase 5: XSA sweep

Use the target script and vary XSA strength:

```bash
for LAMBDA in 0.25 0.5 0.75 1.0; do
  RUN_ID=loopq_xsa_${LAMBDA} \
  XSA_LAMBDA_INIT=$LAMBDA \
  XSA_LEARNED=1 \
  TRAIN_BATCH_TOKENS=524288 \
  GRAD_ACCUM_STEPS=8 \
  TRAIN_SEQ_LEN=1024 \
  MLX_MAX_MICROBATCH_TOKENS=8192 \
  MLX_EAGER_EVAL=1 \
  uv run train_gpt_mlx_loop_xsa_attnres_loopq.py
done
```

Also test fixed full XSA:

```bash
RUN_ID=loopq_xsa_full_fixed \
XSA_LAMBDA_INIT=1.0 \
XSA_LEARNED=0 \
TRAIN_BATCH_TOKENS=524288 \
GRAD_ACCUM_STEPS=8 \
TRAIN_SEQ_LEN=1024 \
MLX_MAX_MICROBATCH_TOKENS=8192 \
MLX_EAGER_EVAL=1 \
uv run train_gpt_mlx_loop_xsa_attnres_loopq.py
```

Expected safest default: learned XSA initialized at `0.5`. Full fixed XSA may be too aggressive in a tiny quantized model.

## Phase 6: Block AttnRes source-cap sweep

Run this only after the simple `loop_xsa_attnres_loopq` target wins.

```bash
for N in 4 6 8; do
  RUN_ID=loopq_src${N} \
  ATTNRES_MAX_SOURCES=$N \
  TRAIN_BATCH_TOKENS=524288 \
  GRAD_ACCUM_STEPS=8 \
  TRAIN_SEQ_LEN=1024 \
  MLX_MAX_MICROBATCH_TOKENS=8192 \
  MLX_EAGER_EVAL=1 \
  uv run train_gpt_mlx_loop_xsa_attnres_loopq.py
done
```

Then repeat for the previous-loop-summaries variant if it looked promising:

```bash
for N in 4 6 8; do
  RUN_ID=loopq_prevblocks_src${N} \
  ATTNRES_MAX_SOURCES=$N \
  TRAIN_BATCH_TOKENS=524288 \
  GRAD_ACCUM_STEPS=8 \
  TRAIN_SEQ_LEN=1024 \
  MLX_MAX_MICROBATCH_TOKENS=8192 \
  MLX_EAGER_EVAL=1 \
  uv run train_gpt_mlx_loop_xsa_attnres_loopq_prevblocks.py
done
```

## Phase 7: recurrence stability sweep

Run only on the best loop/XSA/AttnRes variant.

```bash
for DECAY in 0.5 0.75 0.9 0.97; do
  RUN_ID=loopq_decay_${DECAY} \
  RECURRENCE_DECAY_INIT=$DECAY \
  TRAIN_BATCH_TOKENS=524288 \
  GRAD_ACCUM_STEPS=8 \
  TRAIN_SEQ_LEN=1024 \
  MLX_MAX_MICROBATCH_TOKENS=8192 \
  MLX_EAGER_EVAL=1 \
  uv run train_gpt_mlx_loop_xsa_attnres_loopq.py
done
```

Suggested interpretation:

- low decay means the prelude representation is re-injected strongly each loop;
- high decay means the previous recurrent state dominates;
- if high decay causes spikes or poor quantization, the loop is accumulating too much self-similar state.

## Metrics to extract

Use this after a batch of runs:

```bash
python - <<'PY'
from pathlib import Path
import re
for p in sorted(Path('logs').glob('*.txt')):
    text = p.read_text(errors='ignore')
    rid = re.findall(r'run_id:(.*)', text)
    final = re.findall(r'final_int8_zlib_roundtrip val_loss:([0-9.]+) val_bpb:([0-9.]+)', text)
    qbytes = re.findall(r'serialized_model_int8_zlib:([0-9]+) bytes', text)
    steps = re.findall(r'step:([0-9]+)/[0-9]+ train_loss:([0-9.]+).*?step_avg:([0-9.]+)ms tok_s:([0-9.]+)', text)
    if rid and final:
        last_step = steps[-1] if steps else ('?', '?', '?', '?')
        print(f'{rid[-1]:40s} bpb={final[-1][1]} loss={final[-1][0]} qbytes={qbytes[-1] if qbytes else "?"} step={last_step[0]} step_avg_ms={last_step[2]} tok_s={last_step[3]}')
PY
```

Primary score: **final quantized BPB after fixed wall-clock training**.

Secondary scores:

- final pre-quant validation BPB, if you enable intermediate validation;
- quantization gap;
- serialized int8/zlib bytes;
- optimizer steps completed under the wall-clock cap;
- tokens/sec.

## Decision rules

Promote a variant only if it improves final quantized BPB at the same wall-clock budget.

Keep the full target `loop_xsa_attnres_loopq` if:

1. it beats `loop_xsa`;
2. it beats `loop_attnres_loopq`; and
3. it does not have a much worse quantization gap.

Drop XSA if:

1. `loop_xsa` loses clearly to `loop_stable`; and
2. `loop_xsa_attnres_loopq` loses to `loop_attnres_loopq`; and
3. the loss is not just from an overly aggressive `XSA_LAMBDA_INIT`.

Drop the previous-loop block-summary source set unless `loop_xsa_attnres_loopq_prevblocks` improves BPB enough to justify its extra runtime.

Prefer the gated mixer only if `loop_xsa_attnres_loopq_gate` improves quantized BPB, not merely train loss.

## Notes on artifact size

These files still use the repo's simple int8/zlib roundtrip, not the strongest 16 MB submission exporter. The first goal is directional BPB under the same script/export path. Once a variant wins, port it into the stronger quantization/compression pipeline.

The loop-specific mixer parameters are intentionally tiny compared with full per-loop transformer weights. The key design constraint is: **specialize phase/routing, not the expensive QKV/MLP matrices**.
