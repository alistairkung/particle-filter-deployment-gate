# Development validation of the outbox particle filter

**Method lock is withheld.** The fitted joint likelihood gives repeatable timing and count errors on held-out development roots, including when the diagnostic supplies the correct physical regime and unfinished work. A larger particle population cannot by itself correct those conditional probabilities. No protected roots were generated or accessed. [Validation disposition and file hashes](report/validation.json)

| Check | Population | Finding |
|---|---|---|
| Observation likelihood given the true development states | All 1,200 roots, 14,400 intervals, five root-disjoint folds | Every observation had finite log mass. Seventeen frequency or serial-residual checks triggered investigation across at least two folds. For slow publishing, the predicted high-backlog frequency was 15.0–16.0% in the three flagged folds, compared with 8.3–9.0% observed. [Likelihood diagnosis](TEACHER_VALIDATION.md) |
| Observation likelihood during filtering | Twelve held-out roots per condition, 36 total | Twenty-one frequency or serial-residual checks triggered investigation. Cut-work RMSE was 47.7 ms for the PF and 58.6 ms for the diagnostic using the true regime’s training mean. This development result alone cannot establish improved action forecasts. [Sequential checks](sequential/sequential.json) |
| Revealing publisher phase | All 1,200 development roots | On 13,967 supported intervals, revealing phase and residual acknowledgement gave a paired decrease in absolute mark-count error of −0.011 marks (95% interval −0.029 to 0.006). The other 433 intervals lacked compatible phase-specific donors. [Phase challenge](phase_challenge.md) |

## What the forecasts did on the development panel

The table uses held-out predictions for the same 36 roots. For this check the PF used 128 particles and 64 matched hold/promote pairs per root. These are diagnostic settings. The protected budget remains 300 roots, and its separate approval gate remains closed.

| Predictor | Signed increment MSE | Hold Brier | Promote Brier |
|---|---:|---:|---:|
| Full PF pipeline | 0.19915 | 0.17369 | 0.20299 |
| PF with regime-conditional work | 0.19906 | 0.17394 | 0.20283 |
| Temporal outbox history | 0.19106 | 0.14272 | 0.17788 |
| Complete snapshot | 0.19571 | 0.18130 | 0.19274 |
| Oldest age | 0.19956 | 0.19236 | 0.19197 |
| Action prior | 0.21145 | 0.24324 | 0.20060 |

The paired gain on signed incremental-risk MSE was −0.00809 for temporal history minus full PF (development 95% interval −0.02729 to +0.01153). The ablated-minus-full difference was −0.00009 (−0.00220 to +0.00204). Both intervals include zero. The fold files retain per-root predictions, paired cells, time-causal delivery checks and particle traces. The aggregate file includes action calibration, scores for each condition and paired bootstrap intervals. [Development panel evidence](sequential/)

For a worked example, the first root in fold 2 had 8 both-breach pairs, 30 promote-only pairs, zero hold-only pairs and 26 neither pairs. Its predicted hold risk was 8/64, its promote risk was 38/64 and its incremental promotion risk was 30/64. All 64 pairs contribute to one root's forecast. [Worked root and trace](sequential/fold-2.json)

## Particle-count check and runtime

The first planned doubling, from 128 to 256 particles, **failed the development stability criteria**. Each setting covered the same 36 held-out roots with four seeds. In B the largest RMS change in a cut regime probability was 0.073. In C it was 0.438, and the cut mean work changed by 0.0149 s RMS. The tolerances are 0.03 for a regime probability and 0.01 s for mean work. These measurements and the separately observed likelihood errors leave both particle count and forecast-pair count unlocked.

On four local workers, the 144 filtering runs took 106.2 s at 128 particles and 146.9 s at 256. Peak measured worker memory was 176.3 MiB at 256. A single timed forecast pair took 0.10 s. These timings cover development runs under the recorded conditions. A verified runtime projection for the full protected run, independent verification and reporting remains outstanding. The panel command below reproduces the state-doubling check in the study environment. [Numerical diagnostics and runtime](convergence/convergence.json)

```sh
"$PYTHON" -m particle_filter.validation.convergence --diagnostics-only --state-check-only --workers 4 --max-hours 12 --output /tmp/pf-numerical-check
```

## Reproduction and interpretation

In each diagnostic plot, the five points for a quantity are folds 0 through 4 from left to right. A vertical bar gives its whole-root 95% interval where estimable. The zero line indicates agreement for frequency errors and zero lag-one residual product.

Follow the study guide for the run commands. The teacher, sequential and phase receipts bind their scientific inputs and producing code. The report verifies the individual fold files before combining them. The development evidence supports examining the conditional pending-excess and age-conditioned delay laws in the affected strata before another attempt at method lock. [Researcher guide](../../README.md) · [Teacher-forced residual plot](report/teacher.png) · [Sequential residual plot](report/sequential.png)
