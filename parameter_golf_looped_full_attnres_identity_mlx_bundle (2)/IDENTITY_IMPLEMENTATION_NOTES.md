# Identity Full AttnRes implementation notes

This bundle starts from `parameter_golf_looped_full_attnres_mlx_optimized_bundle` and keeps the `AttnResHistory` MLX cache, but changes the Full AttnRes semantics.

## What changed

1. Full AttnRes sources are now post-residual hidden states.
2. The normal additive residual stream is preserved inside each attention and MLP sublayer.
3. The mixer has a fixed dynamic recency prior controlled by `ATTNRES_RECENCY_BIAS_INIT`, default `4.0`, so initialization is near "select latest source" rather than uniform averaging.
4. XSA-enabled scripts default to fixed paper-faithful projection removal with `XSA_LAMBDA_INIT=1.0` and `XSA_LEARNED=0`.
5. Full AttnRes scripts expose `MAX_VAL_TOKENS` and `SKIP_QUANTIZED_VALIDATION` for fast smoke triage.

## Why this should be higher value than more speed work

The optimized bundle already proved the cache helps speed, but BPB still got worse. The likely failure was semantic: raw-delta histories plus uniform source mixing made the model learn basic residual routing from scratch. This bundle makes Full AttnRes a low-risk routing augmentation on top of the known-good residual stream.

## Default source count

With `PRELUDE_LAYERS=2`, `CORE_LAYERS=2`, `NUM_LOOPS=3`, and `CODA_LAYERS=2`, the normal identity Full AttnRes path stores:

- one initial embedding source;
- two post-residual hidden states per effective transformer block;
- one loop-boundary source per recurrent loop;
- optional previous-loop summary sources for the prevsummary variant.

The default `ATTNRES_MAX_SOURCES=128` remains safely above this.
