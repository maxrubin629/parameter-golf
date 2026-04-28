# Parameter Golf looped-decoder Full Attention Residuals test plan — MLX optimized

This bundle is the Full Attention Residuals version of the looped-decoder ablation ladder.

Implementation update: the Full AttnRes MLX path now uses an `AttnResHistory` cache. Source values are still preserved exactly for Full AttnRes, but each source key is RMS-normalized once when appended, and the mixer avoids building a full `[batch, seq, sources, dim]` value stack at every sublayer. See `OPTIMIZATION_NOTES.md`.

The architecture idea being tested is:

> a stable looped decoder-only transformer that shares the expensive recurrent transformer weights across loop iterations, uses **Full Attention Residuals** across every effective unrolled attention/MLP sublayer, gives each loop repetition its own AttnRes pseudo-query slot, and optionally uses XSA inside recurrent sequence attention.

The important change from the previous bundle is that AttnRes is no longer block/stage-summary routing. The Full AttnRes scripts keep a global source history across the whole unrolled computation:

1. source 0 is the token embedding;
2. every attention sublayer appends its raw output;
3. every MLP sublayer appends its raw output;
4. every later residualized sublayer attends over the complete prior source history;
5. prelude, recurrent-core, coda, and final readout all use AttnRes rather than silently falling back to additive residuals.

This matches the core Full AttnRes idea from the paper more closely: each layer has a learned pseudo-query over previous layer outputs. In this looped version, the recurrent transformer weights can be shared, but the pseudo-query slots are indexed by effective unrolled depth/loop position.

## Files in this bundle

Copy these files into the root of your Parameter Golf repo, next to `train_gpt_mlx.py`.

| File | Purpose |
|---|---|
| `train_gpt_mlx_pg_baseline.py` | Exact copy of your current MLX baseline. Use this as the reference run. |
| `train_gpt_mlx_loop_base.py` | Prelude/core/coda looped decoder, shared recurrent core, additive residuals, no stable recurrence, no XSA. |
| `train_gpt_mlx_loop_stable.py` | Adds stable recurrent input mixing to the looped decoder; still additive residuals everywhere. |
| `train_gpt_mlx_loop_xsa.py` | Stable looped decoder + XSA in recurrent-core sequence attention; still additive residuals everywhere. |
| `train_gpt_mlx_loop_fullattnres_loopq.py` | Stable looped decoder + Full AttnRes everywhere; recurrent core has per-loop pseudo-query slots; no XSA. |
| `train_gpt_mlx_loop_xsa_fullattnres_loopq.py` | Main target: stable looped decoder + XSA + all-sublayer Full AttnRes + per-loop recurrent pseudo-query slots. |
| `train_gpt_mlx_loop_xsa_fullattnres_loopq_gate.py` | Main target + loop/source gates in the AttnRes mixers. |
| `train_gpt_mlx_loop_xsa_fullattnres_loopq_prevsummary.py` | Main target + an extra previous-loop summary source. Full history is already present; this tests whether a direct summary helps. |
| `train_gpt_mlx_current_record.py` | MLX architecture port of the SP8192 current-record reference, included only as an optional comparison. |

## Design correction: every effective layer gets its own pseudo-query

For Full AttnRes runs:

- every prelude attention sublayer has its own pseudo-query;
- every prelude MLP sublayer has its own pseudo-query;
- every recurrent-core attention sublayer has its own pseudo-query **per loop iteration**;
- every recurrent-core MLP sublayer has its own pseudo-query **per loop iteration**;
- every coda attention sublayer has its own pseudo-query;
- every coda MLP sublayer has its own pseudo-query;
- the final output readout has its own pseudo-query.

The expensive QKV/MLP matrices are still shared across recurrent loop repetitions. Only the tiny AttnRes routing parameters are unshared by loop position.

With the default layout:

```bash
PRELUDE_LAYERS=2
CORE_LAYERS=2
CODA_LAYERS=2
NUM_LOOPS=3
```

The model has `2 + 2*3 + 2 = 10` effective transformer-block applications, or 20 residualized sublayers. Full AttnRes therefore sees up to 21 normal sources including the embedding, plus a few loop-boundary/summary sources used by these compact scripts. The default source budget is set high enough for local sweeps:

```bash
ATTNRES_MAX_SOURCES=128
```

Unlike the previous block-style scripts, these Full AttnRes scripts intentionally raise an error if the history exceeds `ATTNRES_MAX_SOURCES`; they do not silently truncate to recent sources. If you make the unrolled model much deeper, increase that value.

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

Recommended default knobs for the looped runs:

```bash
PRELUDE_LAYERS=2
CORE_LAYERS=2
CODA_LAYERS=2
NUM_LOOPS=3
ATTNRES_MAX_SOURCES=128
```

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

Purpose: catch shape bugs, memory blow-ups, graph-compile problems, source-budget mistakes, and variants that are much slower than expected. These are **not** final BPB comparisons.

Use a fixed 200-step smoke and one final validation:

```bash
COMMON="TRAIN_BATCH_TOKENS=524288 GRAD_ACCUM_STEPS=8 TRAIN_SEQ_LEN=1024 MLX_MAX_MICROBATCH_TOKENS=8192 MLX_EAGER_EVAL=1 ITERATIONS=200 MAX_WALLCLOCK_SECONDS=999999 WARMUP_STEPS=20 TRAIN_LOG_EVERY=10 VAL_LOSS_EVERY=0 ATTNRES_MAX_SOURCES=128"
```

Run the smoke ladder:

```bash
env $COMMON RUN_ID=smoke_pg_baseline uv run train_gpt_mlx_pg_baseline.py
env $COMMON RUN_ID=smoke_loop_base uv run train_gpt_mlx_loop_base.py
env $COMMON RUN_ID=smoke_loop_stable uv run train_gpt_mlx_loop_stable.py
env $COMMON RUN_ID=smoke_loop_xsa uv run train_gpt_mlx_loop_xsa.py
env $COMMON RUN_ID=smoke_loop_fullattnres_loopq uv run train_gpt_mlx_loop_fullattnres_loopq.py
env $COMMON RUN_ID=smoke_loop_xsa_fullattnres_loopq uv run train_gpt_mlx_loop_xsa_fullattnres_loopq.py
```

Then, if the main target is healthy, smoke the two richer variants:

```bash
env $COMMON RUN_ID=smoke_loop_xsa_fullattnres_loopq_gate uv run train_gpt_mlx_loop_xsa_fullattnres_loopq_gate.py
env $COMMON RUN_ID=smoke_loop_xsa_fullattnres_loopq_prevsummary uv run train_gpt_mlx_loop_xsa_fullattnres_loopq_prevsummary.py
```

Smoke-pass criteria:

1. script reaches final int8/zlib roundtrip eval;
2. no NaNs or exploding train loss;
3. no `Full AttnRes source history` source-budget error;
4. step time is not catastrophically worse than baseline;
5. final smoke BPB is not obviously broken relative to nearby variants;
6. quantized roundtrip BPB is not dramatically worse than pre-quant validation.

After smoke tests, compare logs:

```bash
grep -E "run_id:|model_params:|step:[0-9]+/[0-9]+ train_loss|final_int8_zlib_roundtrip val_loss|serialized_model_int8_zlib" logs/smoke_*.txt
```

## Phase 2: first real 10-minute ablation ladder

Run baseline first, then the additive-residual loop ladder, then the Full AttnRes ladder.

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

### Full AttnRes ladder

```bash
RUN_ID=loop_fullattnres_loopq \
TRAIN_BATCH_TOKENS=524288 \
GRAD_ACCUM_STEPS=8 \
TRAIN_SEQ_LEN=1024 \
MLX_MAX_MICROBATCH_TOKENS=8192 \
MLX_EAGER_EVAL=1 \
ATTNRES_MAX_SOURCES=128 \
uv run train_gpt_mlx_loop_fullattnres_loopq.py
```

```bash
RUN_ID=loop_xsa_fullattnres_loopq \
TRAIN_BATCH_TOKENS=524288 \
GRAD_ACCUM_STEPS=8 \
TRAIN_SEQ_LEN=1024 \
MLX_MAX_MICROBATCH_TOKENS=8192 \
MLX_EAGER_EVAL=1 \
ATTNRES_MAX_SOURCES=128 \
uv run train_gpt_mlx_loop_xsa_fullattnres_loopq.py
```

Primary comparisons:

| Comparison | Question |
|---|---|
| `loop_base` vs `pg_baseline` | Does the raw looped scaffold survive the fixed wall-clock comparison? |
| `loop_stable` vs `loop_base` | Does stable recurrence matter? |
| `loop_xsa` vs `loop_stable` | Does XSA help the looped core by itself? |
| `loop_fullattnres_loopq` vs `loop_stable` | Does replacing additive residuals with Full AttnRes help? |
| `loop_xsa_fullattnres_loopq` vs `loop_fullattnres_loopq` | Does XSA help once full depth-wise residual routing exists? |
| `loop_xsa_fullattnres_loopq` vs `loop_xsa` | Does Full AttnRes help once XSA exists? |
| `loop_xsa_fullattnres_loopq` vs `pg_baseline` | Does the full idea beat the starter under fixed wall-clock? |

## Phase 3: richer target variants

Run these only if `loop_xsa_fullattnres_loopq` is healthy in smoke and competitive in Phase 2.

### Loop/source gates

```bash
RUN_ID=loop_xsa_fullattnres_loopq_gate \
TRAIN_BATCH_TOKENS=524288 \
GRAD_ACCUM_STEPS=8 \
TRAIN_SEQ_LEN=1024 \
MLX_MAX_MICROBATCH_TOKENS=8192 \
MLX_EAGER_EVAL=1 \
ATTNRES_MAX_SOURCES=128 \
uv run train_gpt_mlx_loop_xsa_fullattnres_loopq_gate.py
```

### Previous-loop summary source

```bash
RUN_ID=loop_xsa_fullattnres_loopq_prevsummary \
TRAIN_BATCH_TOKENS=524288 \
GRAD_ACCUM_STEPS=8 \
TRAIN_SEQ_LEN=1024 \
MLX_MAX_MICROBATCH_TOKENS=8192 \
MLX_EAGER_EVAL=1 \
ATTNRES_MAX_SOURCES=128 \
uv run train_gpt_mlx_loop_xsa_fullattnres_loopq_prevsummary.py
```

Interpretation:

- `gate` tests whether the mixer benefits from an extra learned source preference beyond query+bias softmax.
- `prevsummary` tests whether a direct previous-loop summary is helpful even though the complete previous-loop sublayer history is already available.

## Phase 4: loop-count and physical-layout sweeps

Only sweep the top 1-2 variants from Phase 2/3. Start with `train_gpt_mlx_loop_xsa_fullattnres_loopq.py`.

### Loop count sweep

```bash
for T in 2 3 4 5; do
  RUN_ID=fullattnres_T${T} \
  NUM_LOOPS=$T \
  ATTNRES_MAX_SOURCES=128 \
  TRAIN_BATCH_TOKENS=524288 \
  GRAD_ACCUM_STEPS=8 \
  TRAIN_SEQ_LEN=1024 \
  MLX_MAX_MICROBATCH_TOKENS=8192 \
  MLX_EAGER_EVAL=1 \
  uv run train_gpt_mlx_loop_xsa_fullattnres_loopq.py
done
```

Expected sweet spot: `NUM_LOOPS=3` or `4`. More loops may reduce optimizer steps enough to hurt BPB even if the architecture is more expressive.

### Prelude/core/coda sweep

```bash
for SHAPE in 2,2,2 2,3,2 3,2,2 2,3,3 3,3,2; do
  IFS=, read P C D <<< "$SHAPE"
  RUN_ID=fullattnres_shape_${P}_${C}_${D} \
  PRELUDE_LAYERS=$P CORE_LAYERS=$C CODA_LAYERS=$D NUM_LOOPS=3 \
  ATTNRES_MAX_SOURCES=128 \
  TRAIN_BATCH_TOKENS=524288 \
  GRAD_ACCUM_STEPS=8 \
  TRAIN_SEQ_LEN=1024 \
  MLX_MAX_MICROBATCH_TOKENS=8192 \
  MLX_EAGER_EVAL=1 \
  uv run train_gpt_mlx_loop_xsa_fullattnres_loopq.py
done
```

If a sweep exceeds 128 sources, increase `ATTNRES_MAX_SOURCES` rather than letting the run truncate.

## Phase 5: XSA sweep

Use the target script and vary XSA strength:

```bash
for LAMBDA in 0.25 0.5 0.75 1.0; do
  RUN_ID=fullattnres_xsa_${LAMBDA} \
  XSA_LAMBDA_INIT=$LAMBDA \
  XSA_LEARNED=1 \
  ATTNRES_MAX_SOURCES=128 \
  TRAIN_BATCH_TOKENS=524288 \
  GRAD_ACCUM_STEPS=8 \
  TRAIN_SEQ_LEN=1024 \
  MLX_MAX_MICROBATCH_TOKENS=8192 \
  MLX_EAGER_EVAL=1 \
  uv run train_gpt_mlx_loop_xsa_fullattnres_loopq.py
done
```

Also test fixed full XSA:

```bash
RUN_ID=fullattnres_xsa_full_fixed \
XSA_LAMBDA_INIT=1.0 \
XSA_LEARNED=0 \
ATTNRES_MAX_SOURCES=128 \
TRAIN_BATCH_TOKENS=524288 \
GRAD_ACCUM_STEPS=8 \
TRAIN_SEQ_LEN=1024 \
MLX_MAX_MICROBATCH_TOKENS=8192 \
MLX_EAGER_EVAL=1 \
uv run train_gpt_mlx_loop_xsa_fullattnres_loopq.py
```

Expected safest default: learned XSA initialized at `0.5`. Full fixed XSA may be too aggressive in a tiny quantized model.

## Phase 6: source-budget sanity sweep

This is not a block-size sweep. Full AttnRes should use all available prior sources. The point is only to verify that the source budget is not masking a shape bug or adding too many bias/gate parameters.

```bash
for N in 64 128 192; do
  RUN_ID=fullattnres_srcbudget${N} \
  ATTNRES_MAX_SOURCES=$N \
  TRAIN_BATCH_TOKENS=524288 \
  GRAD_ACCUM_STEPS=8 \
  TRAIN_SEQ_LEN=1024 \
  MLX_MAX_MICROBATCH_TOKENS=8192 \
  MLX_EAGER_EVAL=1 \
  uv run train_gpt_mlx_loop_xsa_fullattnres_loopq.py
done
```

For the default architecture, these should be functionally close except for the number of bias/gate parameters. If 64 works and 128 is worse, prefer 64 for artifact-size hygiene. If 64 errors because a deeper sweep exceeded the budget, increase the budget.

## Phase 7: recurrence stability sweep

Run only on the best loop/XSA/Full-AttnRes variant.

```bash
for DECAY in 0.5 0.75 0.9 0.97; do
  RUN_ID=fullattnres_decay_${DECAY} \
  RECURRENCE_DECAY_INIT=$DECAY \
  ATTNRES_MAX_SOURCES=128 \
  TRAIN_BATCH_TOKENS=524288 \
  GRAD_ACCUM_STEPS=8 \
  TRAIN_SEQ_LEN=1024 \
  MLX_MAX_MICROBATCH_TOKENS=8192 \
  MLX_EAGER_EVAL=1 \
  uv run train_gpt_mlx_loop_xsa_fullattnres_loopq.py
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
        print(f'{rid[-1]:45s} bpb={final[-1][1]} loss={final[-1][0]} qbytes={qbytes[-1] if qbytes else "?"} step={last_step[0]} step_avg_ms={last_step[2]} tok_s={last_step[3]}')
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

Keep the full target `loop_xsa_fullattnres_loopq` if:

1. it beats `loop_xsa`;
2. it beats `loop_fullattnres_loopq`; and
3. it does not have a much worse quantization gap.

Drop XSA if:

1. `loop_xsa` loses clearly to `loop_stable`; and
2. `loop_xsa_fullattnres_loopq` loses to `loop_fullattnres_loopq`; and
3. the loss is not just from an overly aggressive `XSA_LAMBDA_INIT`.

Drop the previous-loop summary unless `loop_xsa_fullattnres_loopq_prevsummary` improves BPB enough to justify its extra source and runtime.

Prefer the gated mixer only if `loop_xsa_fullattnres_loopq_gate` improves quantized BPB, not merely train loss.

## Notes on artifact size and runtime

These files still use the repo's simple int8/zlib roundtrip, not the strongest 16 MB submission exporter. The first goal is directional BPB under the same script/export path. Once a variant wins, port it into the stronger quantization/compression pipeline.

Full AttnRes adds only tiny query/bias/gate parameters, but it does add more depth-history reads than the prior block-style bundle. On your MacBook this is probably acceptable for a sub-20M model, but the relevant comparison is fixed wall-clock final BPB, not just raw loss per optimizer step.
