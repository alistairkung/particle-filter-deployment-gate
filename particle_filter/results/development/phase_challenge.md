# Does revealing publisher phase improve the physical predictions?

On 1,200 development roots, revealing the publisher phase and residual acknowledgement time gave little change in five-second prediction error where both reconstructions had empirical support. Both variants received the same true starting regime, unfinished work and pending count for this diagnostic. This isolates the reconstruction assumption. A live forecast must infer the regime and unfinished work.

| Held-out outcome | Ordinary reconstruction | Phase and acknowledgement revealed | Paired decrease in absolute error (95% interval) |
|---|---:|---:|---:|
| Completed marks per five seconds, MAE | 14.943 | 14.955 | −0.011 (−0.029 to 0.006) |
| Unfinished DB work at interval end, MAE in ms | 45.87252 | 45.87247 | 0.00005 (−0.00012 to 0.00020) |

Each fold supplied 960 training roots for both donor distributions and its backlog cutoff. The ordinary reconstruction could represent every held-out physical state without its final fallback. Conditioning the donors on the exact phase left 433 of 14,400 intervals and one of 36 decision cuts without a compatible description. These histories have no paired error estimate. The available comparisons provide no evidence here for adding a latent publisher phase. They leave its value on unsupported histories unresolved.

On the 35 supported decision cuts, both reconstructions gave the following mean action probabilities across 16 matched forecast pairs per cut. The small panel and diagnostic rollout budget limit how precisely these fractions can establish calibration.

| Pending count relative to training-fold median | Roots | Hold risk | Hold breach fraction | Promote risk | Promote breach fraction |
|---|---:|---:|---:|---:|---:|
| At or below | 18 | 0.448 | 0.556 | 0.733 | 0.778 |
| Above | 17 | 0.349 | 0.294 | 0.691 | 0.706 |

Reproduce from the repository root with the study Python environment using `python -m particle_filter.phase_challenge`. The runner uses four independent paired five-second projections per interval and 16 paired action forecasts on each preselected development cut. It averages the simulated values before calculating errors. The phase-revealed draws condition the fold-trained donor distribution on the observed physical phase and retain the true residual when awaiting acknowledgement. Both variants share future primitives. Intervals use 2,000 percentile resamples of contributing whole roots within experimental condition. The fold-specific medians, coverage, regime results, publisher-phase and backlog strata and both action Brier scores are in the [development receipt](phase_challenge.json).
