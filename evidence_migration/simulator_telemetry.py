"""Reproduce the telemetry publication with the current plant and frozen fits."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from telemetry_sufficiency import archive, custody, parquet_evidence, workflow
from telemetry_sufficiency.methods import predict, predict_evaluator
from telemetry_sufficiency.scoring import score

from .source import exact


def _check_development_predictions(run, roots, fitted):
    """Apply the stored held-out fold model to each regenerated development root."""
    oof = {record["key"]: record for record in fitted["oof"]}
    if set(oof) != {row["key"] for row in roots} or len(oof) != len(fitted["oof"]):
        raise ValueError("Incomplete or duplicate out-of-fold forecast inventory")
    for row in custody.labelled_rows(run, roots):
        key = row["key"]
        fold = fitted["folds"][key]
        bundle = fitted["fold_models"][str(fold)]
        forecasts = predict(bundle["models"], row["history"])
        forecasts["evaluator"] = predict_evaluator(
            bundle["evaluator_models"],
            row["condition"],
            row["demand"],
            row["unfinished"],
        )
        exact(forecasts, oof[key]["predictions"], f"{key} out-of-fold forecasts")
    return len(oof)


def _compare_regenerated_evidence(run, published):
    # The fit and its fold ledger are read-only inputs. The separate migration
    # verifier reconstructs the independent protected replay ledger.
    retained = {
        "models",
        "oof",
        "oof-predictions",
        "oof-evaluator-predictions",
        "replay-audit",
    }
    expected = {
        path.relative_to(published)
        for path in published.rglob("*.parquet")
        if path.stem not in retained
    }
    observed = {path.relative_to(run) for path in run.rglob("*.parquet")}
    if observed != expected:
        raise ValueError(
            f"Reproduced table inventory differs: {sorted(expected ^ observed)}"
        )
    files = [run / path for path in sorted(expected)] + [run / "scores.json"]
    hashes = workflow.relative_files(run, files)
    workflow.verify_files(published, hashes)
    return hashes


def verify(run, workers):
    """Write only into an empty scratch directory. Every comparison here is exact."""
    if any(run.iterdir()):
        raise ValueError("An empty scratch directory is required")
    published = archive.V1
    manifest = archive.verify_inputs(published)
    lock_hash = custody.file_hash(published / "predictor-lock.json")
    signature = custody.digest(custody.canonical(workflow.study_contract()))
    fitted = workflow.get_fit(published)
    lock = custody.read(published / "predictor-lock.json")
    exact(fitted["selected"], lock["selected"], "locked claim selections")

    parquet_evidence.write(run, "inventory", custody.inventory())
    for partition in custody.PARTITIONS:
        roots = workflow.roots(partition, published)
        workflow.collect(run, roots, signature, workers)
    development = workflow.roots("development", run)
    protected = workflow.roots("protected", run)
    oof_count = _check_development_predictions(run, development, fitted)
    workflow.write_predictions(run, protected, fitted)
    scores = score(
        workflow.score_rows(run, protected), lock["selected"], lock["bootstrap_seed"]
    )
    custody.save(run / "scores.json", scores)
    hashes = _compare_regenerated_evidence(run, published)
    report_dir = run / "report"
    archive.regenerate("report", published, report_dir, workers)
    report_files = [
        report_dir / name
        for name in manifest["completed"]["report"]
        if name == "readiness.json" or name.endswith((".md", ".png"))
    ]
    return dict(
        study_contract_sha256=signature,
        frozen_lock_sha256=lock_hash,
        development_roots=len(development),
        protected_roots=len(protected),
        branches=sum(len(custody.branch_names(row)) for row in development + protected),
        out_of_fold_forecast_records=oof_count,
        exact_reproduction_sha256=hashes,
        regenerated_publication_sha256=workflow.relative_files(
            report_dir, report_files
        ),
        claims=scores["claims"],
        verdict=custody.read(report_dir / "readiness.json")["verdict"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    with TemporaryDirectory(prefix="simulator-v1-telemetry-") as directory:
        print(
            json.dumps(verify(Path(directory), args.workers), indent=2, sort_keys=True)
        )


if __name__ == "__main__":
    main()
