# Parameter Golf looped-decoder Identity Full Attention Residuals test plan

This bundle is a semantics-fix successor to `parameter_golf_looped_full_attnres_mlx_optimized_bundle`.

The previous optimized Full AttnRes implementation reduced repeated MLX work, but the smoke runs showed two problems:

1. Full AttnRes was still materially slower than the additive/XSA looped rows.
2. BPB got worse, even though the parameter increase was tiny.

The main hypothesis here is that the previous implementation was too disruptive: it appended raw attention and MLP deltas to history, then asked later sublayers to reconstruct the residual stream from those deltas. This bundle keeps the optimized `AttnResHistory` cache, but changes the semantics so Full AttnRes is identity-preserving by default.

## Highest-value changes

| Change | Why it matters |
|---|---|
| Post-residual hidden-state sources | History now stores meaningful residual-stream states, not raw attention/MLP deltas. |
| Normal residual stream preserved | Each block still does `x = x + attention_delta` and `x = x + mlp_delta`; AttnRes only chooses the input source for each sublayer. |
| Latest-source recency prior | Mixers initialize near "use the latest residual state" instead of uniform averaging all prior sources. |
| Paper-faithful XSA default | XSA-enabled scripts use fixed projection removal with `XSA_LAMBDA_INIT=1.0` and `XSA_LEARNED=0`. |
| Smoke controls | Full AttnRes scripts support `MAX_VAL_TOKENS` and `SKIP_QUANTIZED_VALIDATION` for fast triage before full roundtrip runs. |

## Files in this bundle

| File | Purpose |
|---|---|
| `train_gpt_mlx_pg_baseline.py` | Baseline reference, copied unchanged. |
| `train_gpt_mlx_loop_base.py` | Additive looped decoder, copied unchanged. |
| `train_gpt_mlx_loop_stable.py` | Stable recurrence looped decoder, copied unchanged. |
| `train_gpt_mlx_loop_xsa.py` | Stable looped decoder with XSA, copied unchanged. |
| `train_gpt_mlx_loop_fullattnres_loopq.py` | Identity Full AttnRes without XSA. |
| `train_gpt_mlx_loop_xsa_fullattnres_loopq.py` | Main target: XSA plus identity Full AttnRes. |
| `train_gpt_mlx_loop_xsa_fullattnres_loopq_gate.py` | Main target plus learned gate. |
| `train_gpt_mlx_loop_xsa_fullattnres_loopq_prevsummary.py` | Main target plus previous-loop summary source. |
| `train_gpt_mlx_current_record.py` | Optional current-record comparison, copied unchanged. |

## New implementation details

### Full AttnRes source semantics

The previous Full AttnRes path used:

```python
attn_out = attention(norm(mixer(history)))
history.append(attn_out)
mlp_out = mlp(norm(mixer(history)))
history.append(mlp_out)
return mlp_out
```

This bundle uses:

```python
x = history.latest()
attn_delta = attention(norm(mixer(history)))
x = x + attn_delta
history.append(x)
mlp_delta = mlp(norm(mixer(history)))
x = x + mlp_delta
history.append(x)
return x
```

So the model starts from the same residual-stream topology as the stable looped decoder. Full AttnRes is now a routing choice for sublayer inputs, not a replacement for the residual stream itself.

### Recency prior

Each `DepthMixer` still has learned query and learned source bias. In addition, scores get a fixed dynamic recency prior:

```python
recency = arange(nsrc) - (nsrc - 1)
scores += ATTNRES_RECENCY_BIAS_INIT * recency
```

Default:

```bash
ATTNRES_RECENCY_BIAS_INIT=4.0
```

At initialization, the latest source dominates. The learned query/bias can still pull from older sources if useful.

### Smoke-only controls

The Full AttnRes scripts now support:

```bash
MAX_VAL_TOKENS=65536              # truncate validation tokens for fast triage; 0 means full validation
SKIP_QUANTIZED_VALIDATION=1       # write int8/zlib artifact but skip final roundtrip eval
```

Use these only for development triage. Final comparisons should run full validation and final int8/zlib roundtrip.

## Phase 1A: fast semantic smoke

Purpose: quickly catch shape/compile/runtime issues without burning a full validation pass per variant.

```bash
COMMON="TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=8 TRAIN_SEQ_LEN=1024 MLX_MAX_MICROBATCH_TOKENS=8192 MLX_EAGER_EVAL=1 ITERATIONS=200 MAX_WALLCLOCK_SECONDS=999999 WARMUP_STEPS=20 TRAIN_LOG_EVERY=10 VAL_LOSS_EVERY=0 ATTNRES_MAX_SOURCES=128 MAX_VAL_TOKENS=65536 SKIP_QUANTIZED_VALIDATION=1"
```

Run the two highest-value rows first:

```bash
env $COMMON RUN_ID=identity_p1a_sp1024_loop_xsa_fullattnres_loopq python parameter_golf_looped_full_attnres_identity_mlx_bundle/train_gpt_mlx_loop_xsa_fullattnres_loopq.py

env $COMMON RUN_ID=identity_p1a_sp8192_loop_xsa_fullattnres_loopq DATA_PATH=./data/datasets/fineweb10B_sp8192 TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model VOCAB_SIZE=8192 python parameter_golf_looped_full_attnres_identity_mlx_bundle/train_gpt_mlx_loop_xsa_fullattnres_loopq.py
```

Then run the no-XSA control if the main target is healthy:

```bash
env $COMMON RUN_ID=identity_p1a_sp8192_loop_fullattnres_loopq DATA_PATH=./data/datasets/fineweb10B_sp8192 TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model VOCAB_SIZE=8192 python parameter_golf_looped_full_attnres_identity_mlx_bundle/train_gpt_mlx_loop_fullattnres_loopq.py
```

Pass criteria:

1. no NaNs or exploding loss;
2. no source-budget error;
3. step time is closer to optimized Full AttnRes than the unoptimized path;
4. BPB is not obviously worse than the previous optimized Full AttnRes rows on the same tokenizer.

## Phase 1B: full comparable smoke

If Phase 1A passes, rerun full validation and final int8/zlib roundtrip.

```bash
COMMON="TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=8 TRAIN_SEQ_LEN=1024 MLX_MAX_MICROBATCH_TOKENS=8192 MLX_EAGER_EVAL=1 ITERATIONS=200 MAX_WALLCLOCK_SECONDS=999999 WARMUP_STEPS=20 TRAIN_LOG_EVERY=10 VAL_LOSS_EVERY=0 ATTNRES_MAX_SOURCES=128"
```

Recommended order:

```bash
env $COMMON RUN_ID=identity_p1b_sp8192_loop_xsa_fullattnres_loopq DATA_PATH=./data/datasets/fineweb10B_sp8192 TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model VOCAB_SIZE=8192 python parameter_golf_looped_full_attnres_identity_mlx_bundle/train_gpt_mlx_loop_xsa_fullattnres_loopq.py

env $COMMON RUN_ID=identity_p1b_sp8192_loop_xsa_fullattnres_loopq_gate DATA_PATH=./data/datasets/fineweb10B_sp8192 TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model VOCAB_SIZE=8192 python parameter_golf_looped_full_attnres_identity_mlx_bundle/train_gpt_mlx_loop_xsa_fullattnres_loopq_gate.py

env $COMMON RUN_ID=identity_p1b_sp8192_loop_xsa_fullattnres_loopq_prevsummary DATA_PATH=./data/datasets/fineweb10B_sp8192 TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model VOCAB_SIZE=8192 python parameter_golf_looped_full_attnres_identity_mlx_bundle/train_gpt_mlx_loop_xsa_fullattnres_loopq_prevsummary.py
```

Compare against the known SP8192 rows:

| Prior row | Roundtrip BPB | Note |
|---|---:|---|
| `sp8192_newfile_loop_xsa` | 1.9012 | current best smoke reference |
| `fa_p1_sp8192_opt_loop_fullattnres_loopq` | 2.0473 | previous optimized Full AttnRes, no XSA |

## Phase 2: ablations if identity Full AttnRes helps

Only continue if `identity_p1b_sp8192_loop_xsa_fullattnres_loopq` is competitive.

| Ablation | Env override | Question |
|---|---|---|
| Remove recency prior | `ATTNRES_RECENCY_BIAS_INIT=0.0` | Is near-identity init required? |
| Weaker recency prior | `ATTNRES_RECENCY_BIAS_INIT=2.0` | Is 4.0 too sticky? |
| Stronger recency prior | `ATTNRES_RECENCY_BIAS_INIT=6.0` | Does stronger identity improve short smoke? |
| No XSA | use `train_gpt_mlx_loop_fullattnres_loopq.py` | Is XSA necessary for the benefit? |
| Gate | use `train_gpt_mlx_loop_xsa_fullattnres_loopq_gate.py` | Does learned source gating help once identity init is fixed? |

## Expected readout

This bundle should answer whether Full AttnRes was bad because the idea is bad for this tiny model, or because the previous implementation destroyed the residual-stream inductive bias. If this bundle still loses badly, the likely conclusion is that full all-sublayer history is too expensive/noisy for this training horizon and we should go back to lighter skip/block routing.
