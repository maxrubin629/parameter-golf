# MLX Full AttnRes optimization notes

This optimized bundle keeps the same Full Attention Residuals architecture and ablation ladder, but changes the implementation of the Full AttnRes path to reduce repeated MLX work.

Changes applied to the four `*fullattnres*.py` scripts:

1. Added `AttnResHistory`, a runtime cache containing source values, one cached `RMSNorm(source)` key per source, and a running source sum.
2. Removed repeated `rms_norm(mx.stack(history))` calls from every residualized sublayer.
3. Avoided materializing the full `[batch, seq, sources, dim]` value stack in `DepthMixer`; it now builds only the `[batch, seq, sources]` score tensor and streams the weighted value sum over the source list.
4. Replaced Python-loop source means with `AttnResHistory.mean()` and snapshot-based `mean_since()` for the optional previous-loop summary source.
5. Kept the existing source-budget behavior: exceeding `ATTNRES_MAX_SOURCES` raises an error rather than silently truncating Full AttnRes history.

Expected effect: lower peak memory and less repeated RMSNorm/stack work in the Full AttnRes variants, especially at `TRAIN_SEQ_LEN=1024` and large microbatches. The architecture, pseudo-query allocation, and ablation defaults are unchanged.

Validation performed here: Python syntax compilation for all scripts. MLX training was not run in this environment.
