# Run the outbox telemetry sufficiency study

The [scientific protocol](SCIENTIFIC_SPEC.md) and [engineering contract](IMPLEMENTATION_SPEC.md) define the frozen experiment. Use the pinned CPython 3.14.0 environment in the [pilot reproduction guide](../feasibility_pilot/README.md). Invoke the new study from the repository root.

```sh
python=feasibility_pilot/.venv/bin/python
run=telemetry_sufficiency/results/v1
"$python" -m pytest -q telemetry_sufficiency/tests
"$python" -m telemetry_sufficiency validate --run "$run"
"$python" -m telemetry_sufficiency develop --run "$run"
"$python" -m telemetry_sufficiency lock --run "$run"
"$python" -m telemetry_sufficiency evaluate --run "$run"
"$python" -m telemetry_sufficiency audit --run "$run"
"$python" -m telemetry_sufficiency.reproduce "$run/pilot-reproduction.json"
```

For a clean second evaluation use a fresh run directory and the same source and pinned runtime. Each root uses a private stable seed. Changing worker count preserves the sampling design. The default uses six workers. The protected set is consumed for this study. A rerun is a computational check and contributes no independent evidence.

The audit replays protected roots independently. An independent reviewer also inspects the controller data boundary and records the assessment in `verification-review.json` inside the run directory, with the JSON boolean `passed`, nonempty text findings in `findings`, and the SHA-256 hashes `audit` and `lock` for `independent-audit.json` and `predictor-lock.json`. A failed review prevents publication. Once the audit and pilot reproduction receipts are present, render the report and its plots.

```sh
"$python" -m telemetry_sufficiency report --run "$run"
```

The root inventory binds each opaque transport key to its private split and condition. Each split contains compressed newline-delimited JSON observation and evaluator tables. Predictions from permitted histories and evaluator-only models are retained separately. The development fit contains preprocessing values, coefficients, fold predictions and the candidate selection ledger. Stage-level hashes bind every published table. Worker completion markers under `work/` support retries with identical inputs and can be regenerated from the same named streams. Failures and previous publication attempts are preserved under `failures/`.

A failed stage writes `readiness.json` and an Experimental Readiness Report with the verdict **NOT READY**, the failing stage and its reason. The corresponding directory under `failures/` retains the incoming manifest, available receipts, prior report, and expected and observed hashes for mismatched evidence. A fresh pilot reproduction takes the run lock and archives any previous receipt and published report. Rerun the report stage after the required evidence reconciles.
