# Outbox particle-filter study

At the 60-second decision cut, this study estimates the chance of a write-SLO breach if Java traffic stays at 25% or increases to 60%. The forecast reports both probabilities and their signed difference, promote minus hold. Implementation and development validation are authorised. Generating or accessing the 300 protected roots requires successful development validation, locked methods, verified runtime and separate owner approval.

**Current development result.** The issue #15 correction improved the held-out slow-publisher backlog tail. Conditional-delay and serial-residual discrepancies remain. Method lock is withheld. [Issue #15 calibration evidence](results/issue-15/README.md) · [Original results, uncertainty intervals and a worked forecast](results/development/README.md)

## Follow one observation

1. **Read the outbox.** A five-second reading carries a measurement interval and a delivery time. The controller can use it from delivery onwards. An empty outbox has oldest age zero. When no publication marks completed, median completed delay is undefined. [Frozen public reading contract](../simulator/observations.py)
2. **Propose the next hidden state.** Each particle has an operating regime and an amount of unfinished DB work. The approximate model reconstructs anonymous obligations, then uses the frozen FIFO event rules to advance a hypothetical queue. The actual simulator queue stays with the evaluator. [Regime transitions](model/regimes.py) · [Approximate physical model](model/physical.py) · [Frozen simulator](../simulator/README.md)
3. **Score the reading.** The likelihood first scores completed marks, then pending count given marks, then age and completed delay given the counts. Inference uses the particle’s proposed boundary states. [Joint observation likelihood](model/likelihood.py) · [FIFO count and timing constraints](model/outbox_support.py) · [Fitted distributions](model/emission_families.py)
4. **Update the posterior.** The bootstrap filter adjusts the particle weights in log space, checks effective sample size and resamples when necessary. Late readings are scored against the transitions that originally produced their measurement intervals. [Particle filter](filter.py)
5. **Forecast both actions.** Each weighted draw supplies one regime and queue workload. One reconstructed starting state and one set of fresh arrival and demand draws feed both action branches. The regime persists for 120 seconds while the queue and publisher keep moving. The four paired breach counts yield hold risk, promote risk and incremental promotion risk. [Matched forecasts](forecast.py)
6. **Compare with temporal history.** The primary paired score subtracts full-PF squared error from the locked temporal-history predictor’s squared error on the signed outcome. A positive gain favours the complete PF pipeline. The queue-state ablation measures the contribution of the history-specific work estimate. [Experiment](experiment.py) · [Paired scoring](evaluation.py)

## Run the development experiment

Run from the **repository root** with the existing Python 3.14 study environment. On a fresh checkout, follow the [pilot environment setup](../feasibility_pilot/README.md) with the [numerical](../feasibility_pilot/requirements.lock) and [Parquet](../requirements-storage.lock) dependency locks. Choose a fresh output directory for each run. Particle and forecast-pair counts below are development settings. The convergence study determines whether a budget can be locked.

```sh
PYTHON=feasibility_pilot/.venv/bin/python
"$PYTHON" -m particle_filter --help
"$PYTHON" -m particle_filter /tmp/pf-all --particles 128 --pairs 64
"$PYTHON" -m particle_filter /tmp/pf-work-ablation --variant regime-conditional-work --particles 128 --pairs 64
```

The default run fits on four development folds and forecasts the preselected panel roots in the fifth, repeating for all five folds. Each fold output contains the paired outcomes counted by the forecaster and the predicted risks. The combined run also scores the full and ablated forecasts against the four existing direct predictors. Each independent development root supplies one matched pair of evaluator outcomes regardless of the number of forecast continuations.

| Configuration | Pre-cut regime half-life | Decision-time work used by the shared forecaster |
|---|---:|---|
| [full](configs/full.toml) | 120 s | Joint posterior draw |
| [regime-conditional-work](configs/regime-conditional-work.toml) | 120 s | Development cut-work draw given the same inferred regime |
| [half-life-60](configs/half-life-60.toml) | 60 s | Joint posterior draw |
| [half-life-240](configs/half-life-240.toml) | 240 s | Joint posterior draw |

Use one or more `--variant` arguments to run a subset. To produce a separate directory per variant:

```sh
for variant in full regime-conditional-work half-life-60 half-life-240; do
  "$PYTHON" -m particle_filter "/tmp/pf-single-$variant" --variant "$variant" --particles 128 --pairs 64
done
```

Each file is checked against the reviewed configuration before the experiment begins. [Configuration checks](config.py)

## Reproduce the development checks

The validation package owns interval-aligned development labels, fold assignments, empirical fit diagnostics and numerical convergence. Public filtering and forecasting consume only permitted observations, calibrated models and independent randomness.

```sh
"$PYTHON" -m pytest -q particle_filter/tests
"$PYTHON" -m doctest particle_filter/SCIENTIFIC_SPEC.md
mkdir -p /tmp/pf-validation
"$PYTHON" -m particle_filter.validation.teacher_validation /tmp/pf-validation/teacher --workers 4
"$PYTHON" -m particle_filter.validation.sequential_validation /tmp/pf-validation/sequential --workers 4
"$PYTHON" -m particle_filter.validation.phase_challenge > /tmp/pf-validation/phase.json
"$PYTHON" -m particle_filter.validation.validation_report /tmp/pf-validation/teacher /tmp/pf-validation/sequential /tmp/pf-validation/phase.json /tmp/pf-validation/report
"$PYTHON" -m particle_filter.validation.convergence --help
```

The report binds the source model used at the start of each run and the hashes of its input files. Keep the report, teacher and sequential directories and the phase JSON together when moving the evidence bundle.

Teacher-forced validation supplies the **true development boundary state** to the observation likelihood. Sequential validation tests the same likelihood with **inferred states**. Comparing their errors helps distinguish emission misspecification from particle approximation error. The phase challenge asks whether revealing publisher phase and the residual acknowledgement improves held-out physical predictions. These private development inputs end at the diagnostic boundary.

The tests mirror the scientific components. Start with the [forecast boundary tests](tests/test_forecast.py) and the [posterior update tests](tests/test_inference.py). The remaining fixtures cover physical propagation, emissions, variant configuration, budgets and scoring. Generated evidence and plots live separately from the implementation. [Tests](tests/) · [Development evidence](results/development/) · [Structural refactor verification](results/structure-refactor/README.md) · [Reviewed specification](SCIENTIFIC_SPEC.md) · [Methods and acceptance checks](METHODS_PLAN.md) · [Protected protocol](EVALUATION_PLAN.md)
