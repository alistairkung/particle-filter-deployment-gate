# Issue #15 development validation

The correction fits the positive part of the slow-publisher pending-count law on implied new commits when the starting reading is fresh. The comparison uses the same 1,200 roots, 14,400 readings, five training inventories and fold-specific tail thresholds as merged baseline `2b492da`. The corrected source is `e78c582`. All comparisons use development data. The likelihood decision is based on calibration and conditional log mass. [Issue #15](https://github.com/alistairkung/particle-filter-deployment-gate/issues/15) · [Approved likelihood](../../SCIENTIFIC_SPEC.md#slow-publisher-pending-count-correction) · [Baseline receipts](../structure-refactor/)

## Teacher-forced validation

Each row uses the fold-specific training threshold for the ending pending count. The penultimate and final labels are evaluator-only strata. Each pools 400 slow-publisher histories. The overall row pools all six slow-publisher intervals per history.

| Slow-publisher intervals | Observed tail | Predicted tail with marks integrated, before → after | Predicted tail given observed marks, before → after | Mean conditional pending log mass, before → after (nats) |
|---|---:|---:|---:|---:|
| All | 10.08% | 15.93% → 9.86% | 16.18% → 10.00% | −6.125 → −4.075 |
| Penultimate | 0.75% | 25.99% → 0.18% | 26.52% → 0.34% | −6.168 → −4.116 |
| Final | 59.75% | 51.00% → 58.97% | 51.37% → 59.69% | −6.176 → −4.055 |

The paired gain in mean conditional log mass was positive in all five folds. Lower whole-root 95% interval bounds ranged from +1.912 to +2.038 nats per interval. Each fold receipt records the paired log-mass difference and whole-root 95% intervals for the frequency errors under `pending_comparison`. For log mass, the receipt fields `observed` and `predicted` contain corrected and baseline log mass respectively, so `error_ci95` is the interval for the paired gain. The baseline calculation uses the original fitted pending law and reproduces the archived mark-integrated tail and interval in each fold. [Teacher receipt and remaining flags](teacher/summary.json)

The repeated flags fell from 17 to 15. The backlog tail and positive lag-one pending residual no longer triggered under the slow publisher. Conditional delay flags persist in all regimes. The lag-one mark residual flags in A/B and the pending residual flag in B persist. The full A/B teacher-forced diagnostics, along with C mark, empty and timing diagnostics, reproduced the baseline to within `1e-12`. All corrected interval log masses were finite. [Teacher diagnostic plot](teacher/tail-and-serial-residuals.png)

## Sequential validation

The identical 36-root panel was rerun with 128 particles and 64 matched forecast pairs per root. The pooled C backlog-tail prediction moved from 14.62% to 8.86%, against 13.89% observed. The teacher-forced calibration gain did not carry over to this inferred-state prediction on the smaller panel. The corresponding A prediction moved from 13.47% to 13.62% against 5.90% observed and B moved from 4.72% to 5.08% against 8.33% observed. The repeated flags fell from 21 to 20. The positive lag-one pending residual in C no longer flagged. There were no new repeated flags. The remaining flags include conditional delay, the B lag-one mark residual, A backlog and mark tails, and C empty frequency and mark tail. The combined validation disposition remains `investigate`. The earlier particle-doubling failure is a separate numerical question for issue #16. Method lock remains withheld. [Sequential receipt and flags](sequential/sequential.json) · [Sequential diagnostics](report/sequential.png) · [Combined validation receipt](report/validation.json)

## Reproduce

From the repository root with the pinned study environment, use a fresh output directory.

```sh
PYTHON=feasibility_pilot/.venv/bin/python
RUN=$(mktemp -d)
VECLIB_MAXIMUM_THREADS=1 "$PYTHON" -m particle_filter.validation.teacher_validation "$RUN/teacher" --workers 4
VECLIB_MAXIMUM_THREADS=1 "$PYTHON" -m particle_filter.validation.sequential_validation "$RUN/sequential" --workers 4
cp particle_filter/results/structure-refactor/phase_challenge.json "$RUN/phase_challenge.json"
"$PYTHON" -m particle_filter.validation.validation_report \
  "$RUN/teacher" "$RUN/sequential" "$RUN/phase_challenge.json" "$RUN/report"
```

The report reuses the phase challenge receipt after verifying its unchanged physical and protocol fingerprints. The producer receipts bind the source and fold files. The interval digest manifest hashes the decompressed teacher records, so compression timestamps do not affect reproduction. [Teacher payload digests](teacher/interval-payload-digests.json)
