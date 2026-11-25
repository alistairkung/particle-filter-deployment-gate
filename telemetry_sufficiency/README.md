# Outbox telemetry sufficiency study

The [scientific protocol](SCIENTIFIC_SPEC.md) and [original engineering contract](IMPLEMENTATION_SPEC.md) define the frozen experiment. The audited publication at revision `624541b` retains its original [READY report](results/v1/Experimental_Readiness_Report.md), plots, scores, and signed-off receipt chain. The canonical evidence now uses Zstandard-compressed Parquet. Use CPython 3.14.0 and the pinned [numerical environment](../feasibility_pilot/requirements.lock) and [storage dependency](../requirements-storage.lock). Run these commands from the repository root.

## Reconstruct the retained publication

```sh
python=feasibility_pilot/.venv/bin/python
"$python" -m pip install -r requirements-storage.lock
run=telemetry_sufficiency/results/v1
"$python" -m telemetry_sufficiency report --run "$run" --output /tmp/sufficiency-report
"$python" -m telemetry_sufficiency audit --run "$run" --output /tmp/sufficiency-audit --workers 6
```

Both commands read the canonical tables and write to a separate output directory. The reader checks the reconstructed scientific values against the original manifest by reproducing its deterministic transport digests in memory. The report command checks the new Markdown, all three plots, and readiness decision against their original hashes. The audit command replays every protected branch and independently recalculates predictions, comparisons, and the verdict. It compares the regenerated replay records to the retained records per root. Its new receipt binds the physical Parquet replay ledger.

## Run a fresh computational reproduction

```sh
python=feasibility_pilot/.venv/bin/python
run=$(mktemp -d)
"$python" -m pytest -q telemetry_sufficiency/tests
"$python" -m telemetry_sufficiency validate --run "$run"
"$python" -m telemetry_sufficiency develop --run "$run"
"$python" -m telemetry_sufficiency lock --run "$run"
"$python" -m telemetry_sufficiency evaluate --run "$run"
"$python" -m telemetry_sufficiency audit --run "$run"
"$python" -m telemetry_sufficiency.reproduce "$run/pilot-reproduction.json"
```

Each root has a private stable seed. Changing worker count preserves the sampling design. The default uses six workers. The protected partition was consumed in the original study. Rerunning the experiment checks its computation.

An independent reviewer inspects the controller data boundary and records the assessment in `verification-review.json` inside the new run directory. The receipt requires the JSON boolean `passed`, nonempty text findings in `findings`, and the SHA-256 hashes `audit` and `lock` of the new `independent-audit.json` and `predictor-lock.json`. Once the review and pilot reproduction receipts reconcile, render the report.

```sh
"$python" -m telemetry_sufficiency report --run "$run"
```

## Read the Parquet evidence

Each [schema](parquet_evidence.py) binds its version, study, partition, table name, and configuration reference in the Arrow metadata. Transport keys are opaque strings. The [inventory](results/v1/inventory.parquet) maps them to the private condition, split, and within-cell root index. The [manifest](results/v1/manifest.json) fixes the named seed scheme, tick ordering, configuration hash, bootstrap seed and fold seed. The historical predictor lock and access seal record the order in which the protected partition was opened.

| Table in each partition | Row and units |
| --- | --- |
| `observations` | One delivered reading per root and five-second interval. Interval and delivery times are signed int64 nanoseconds relative to the decision. Pending count is integral, oldest age and completed delay are seconds, and throughput is marks per second. A null delay denotes an interval with zero completed marks. |
| `history-index` | Ordered opaque root key and number of readings, including zero for an empty history. The public reader removes the key before prediction. |
| `targets` | One private root with condition, index, realised query demand (s/s), unfinished service (s), integer work/duration components (ns), and cut, prefix and history hashes. |
| `branches`, `windows` | One private endpoint per root and branch (including surprise branches where allocated), and four ordered horizon tally rows per endpoint. Window indices 0–3 denote successive 30-second intervals. |
| `example-roots`, `example-tapes`, `example-visits`, `example-writes` | Three predesignated index-zero traces per partition. Ordinals preserve event order even at time ties. Application primitive draws are float64, query demand is integral ns, and absent lifecycle events or a pending acknowledgement due time retain their nulls. |

Development adds `models` with preprocessing, feature order, target, coefficients, intercepts, and convergence per fit. Fold −1 identifies a fit on all development roots. Coefficient vectors have one component per output class, or one for a scalar target. `oof` records the test root, fold, training inventory digest, and labels. `oof-predictions` and `oof-evaluator-predictions` retain all individual predictions in separate tables. The small `selection.json` records the solvers, selection scores, class ordering, and candidates. Protected `predictions`, `evaluator-predictions`, and `replay-audit` retain each forecast and replay receipt. The root is the sampling unit throughout.

Fresh stage manifests hash every published table. Tables and receipts are written atomically under the stage lock. A worker stores a private digest-bound root checkpoint under `work/` during collection. A retry validates and reuses it. Successful development and evaluation remove their checkpoints after the stage manifest is saved and the published files are reverified. Failures retain the checkpoint evidence. Each failed stage writes a NOT READY report and a diagnostic directory under `failures/` with the incoming manifest, receipts, and mismatched hashes. A fresh pilot reproduction archives any previous receipt and report, and binds the comparison to the frozen pilot's canonical inventory.
