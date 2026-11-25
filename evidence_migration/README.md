# Frozen evidence migration verification

Both scientific decisions survived the migration. The pilot reproduced **GO** from write-level deadlines and paired branch outcomes. The telemetry study reproduced **READY** from the locked fits, public histories and protected outcomes. The pre-removal check read the original bytes from the two frozen Git commits. The same check runs after cleanup without a legacy data directory.

| Published result directory | Original files | Original bytes | Current files | Current bytes |
| --- | ---: | ---: | ---: | ---: |
| `feasibility_pilot/results/frozen` | 2,433 | 106,897,262 | 26 | 68,707,184 |
| `telemetry_sufficiency/results/v1` | 28 | 6,862,756 | 40 | 8,116,542 |
| **Combined** | **2,461** | **113,760,018** | **66** | **76,823,726** |

These counts include the published manifests, diagnostics, plots and reports inside the named directories. They exclude Git history, local lock files and ignored per-root resume caches. The pilot consolidated 2,419 artifacts into 12 Parquet tables. The telemetry study replaced 15 artifacts with 26 typed tables and a small model-selection record. Its published footprint grew by 1,253,786 bytes. Together the result directories use 36,936,292 fewer bytes.

## Scientific checks

- **Pilot.** The verifier checks every root, arm and window field, every flattened window CSV cell, and every NPZ elapsed nanosecond, route flag and censor flag at its original ordinal. It reconciles every window’s deadline misses with its underlying writes, recalculates the four analytical queue controls from all 128 seed means and checks all 16 placebo roots. The four families contain 3,664 branches, 14,656 windows and 17,546,533 writes. The verifier rebuilds 156 diagnostic rows, including route-specific quantiles and DB work. The original independent audit takes its inputs from a temporary view built from the canonical tables. Its paired lower bounds are 0.32444 at reference load and 0.12124 at higher demand. Both clear the 0.10 margin. [Pilot receipt](receipts/pilot-equivalence.json)
- **Telemetry.** The verifier compares the full ordered records decoded from each Git gzip stream with the canonical reader, including all six example event ledgers, histories, targets, interventions, windows, fit coefficients, fold assignments, out-of-fold predictions and replay receipts. It checks the exact public sensor fields, so an extra evaluator field fails verification. The frozen scientific audit reconstructs all 64,800 forecasts and 77 paired comparisons, calibration, overlap, stratified and surprise results. Independent replay also reproduces 3,200 protected branches and 25,491,716 admissions. The locked history model clears the lower-bound threshold for both demand (0.000725034) and promotion increment (0.000325930). [Telemetry receipt](receipts/telemetry-equivalence.json)
- **Provenance.** Original blobs come from the pilot commit `bb20ebed72471f43443c873a11b2ac6e32fbbc00` and the telemetry commit `624541bae2405fe47a59a193a55482775a7e0d0a`. The inventory identifies each tracked study file, byte size, Git object and SHA-256. Each original evidence table also has a record unit, count and field schema. The verification receipt records each canonical table’s SHA-256, physical schema and row count, plus the 47 retained byte-identical artifacts, including both historical reports. [Source files](receipts/source-files.csv) · [Source schemas](receipts/source-schemas.json) · [Verification receipt](receipts/migration-verification.json)

Stored observations, keys, ordering, nulls and predictions require exact equality. Floating-point comparisons for independently calculated statistics use absolute and relative tolerances of `1e-10`. The pilot’s numeric route-share column uses binary64. For the three arms whose original JSON encoded an integer, the verifier requires an integral value before reconstructing that representation.

A separate same-seed execution produced the same 23 canonical pilot files and nine plots. Its pilot report differs only in relative link destinations. The telemetry execution matched all 26 Parquet tables, the selection record, scores, replay ledger, READY report and three plots byte for byte. New execution receipts identify the reproduction and bind its own predictor lock. An independent reviewer also regenerated all four public forecast records for each protected root with file access disabled during prediction. [Same-seed reproduction receipt](receipts/fresh-reproduction.json)

## Reproduce

From the repository root, use the pilot’s Python environment with the scientific dependencies and the storage lock installed. The verification command reads both immutable Git objects and compares against the committed receipts. It requires the superseded files to be absent and checks the current result directories against the published inventory.

```sh
feasibility_pilot/.venv/bin/python -m evidence_migration.verify --require-cleanup
feasibility_pilot/.venv/bin/python -m pytest -q evidence_migration/test_equivalence.py
```

The parallelism setting `--workers N` controls the protected physical replays. The verifier may use temporary disk space to expose canonical pilot records to the original audit. No study simulation or legacy export is needed to open the published tables. The original commits remain available from [pilot PR #1](https://github.com/alistairkung/particle-filter-deployment-gate/pull/1) and [telemetry PR #7](https://github.com/alistairkung/particle-filter-deployment-gate/pull/7).
