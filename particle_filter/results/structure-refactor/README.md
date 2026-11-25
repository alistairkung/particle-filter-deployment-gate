# Structural refactor verification

This development rerun checks whether reorganising the approximate model and extracting the regime transitions changed the study results. The baseline is PR #14 at `5e52549` and the refactored source is `3baeb52`. The [original evidence](../development/) is retained alongside this directory.

| Comparison | Evidence checked |
|---|---|
| Development splits | Five training inventories, held-out folds and the 36 panel root assignments |
| Observation likelihood | All five teacher-forced interval streams, fold diagnostics, root-level intervals and plots |
| Sequential inference | Per-root cut estimates, filtering traces, delivery checks and seed variation at 128 and 256 particles |
| Matched forecasting | The four paired outcome counts per root, action risks and signed increments for the full model and queue-state ablation |
| Evaluation | Scores, calibration and paired confidence intervals, including results by condition |
| Command-line variants | All four configurations together and the standalone queue-state ablation, at 32 particles and two forecast pairs |

The [comparison manifest](equivalence.json) records the original and rerun file hashes and a common hash for the scientific content. JSON comparison used exact values, with keys sorted. Source fingerprints, hashes of receipts containing those fingerprints, and measured time and memory were recorded separately. Each validation report's input hashes and each producer's fold or stage hashes were checked against the corresponding files. Compressed interval logs were compared after decompression and plots were compared byte for byte.

The [validation report](report/validation.json) and [particle-count diagnostic](convergence/convergence.json) retain the original disposition. Likelihood calibration and particle stability require further scientific review. Method lock remains withheld.

To reproduce the rerun, follow the [development commands](../../README.md#reproduce-the-development-checks) and the [particle-count command](../development/README.md#particle-count-check-and-runtime) with fresh output directories. The stored receipts retain the runtime of this rerun, during which development commands ran concurrently.
