"""Reconstruct the locked sufficiency results from canonical development and protected tables."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json

import pyarrow.parquet as pq

from telemetry_sufficiency.audit import audit_analysis, audit_root
from telemetry_sufficiency.parquet_evidence import legacy_records

from .source import READY, ROOT, exact


def _replay(record):
    checked = audit_root(
        record["condition"], record["partition"], record["index"], record
    )
    return {"key": record["key"], "receipt": checked}


def _physical_counts(tables, directory):
    # Independently count source records for each normalized relation, including trace children.
    counts = Counter()
    counts["inventory"] = len(tables["inventory.json"])
    for partition in ("development", "protected"):

        def count(name, n):
            counts[f"{partition}/{name}"] += n

        for row in tables[f"{partition}/observations.jsonl.gz"]:
            count("history-index", 1)
            count("observations", len(row["history"]))
        for row in tables[f"{partition}/evaluator.jsonl.gz"]:
            count("targets", 1)
            count("branches", len(row["branches"]))
            count("windows", sum(len(b["windows"]) for b in row["branches"].values()))
        for name, records in tables.items():
            if name.startswith(f"{partition}/examples/"):
                for row in records:
                    trace = row["trace"]
                    count("example-roots", 1)
                    count("example-tapes", len(trace["prefix"]) + len(trace["future"]))
                    count("example-writes", len(trace["writes"]))
                    count("example-visits", len(trace["visits"]))
    fit = tables["development/fit.jsonl.gz"][0]["fit"]
    counts["development/models"] = sum(
        len(model["fits"])
        for bundle in [fit, *fit["fold_models"].values()]
        for model in [*bundle["models"].values(), bundle["evaluator_models"]]
    )
    counts["development/oof"] = len(fit["oof"])
    counts["development/oof-predictions"] = sum(
        len(row["predictions"]) - 1 for row in fit["oof"]
    )
    counts["development/oof-evaluator-predictions"] = len(fit["oof"])
    counts["protected/replay-audit"] = len(tables["protected/replay-audit.jsonl.gz"])
    counts["protected/predictions"] = sum(
        len(row["predictions"]) for row in tables["protected/predictions.jsonl.gz"]
    )
    counts["protected/evaluator-predictions"] = len(
        tables["protected/evaluator-predictions.jsonl.gz"]
    )
    physical = {
        str(path.relative_to(directory).with_suffix("")): pq.read_metadata(
            path
        ).num_rows
        for path in directory.rglob("*.parquet")
    }
    exact(physical, counts, "telemetry physical table and row inventory")
    public = {
        "key",
        "interval_start_ns",
        "interval_end_ns",
        "available_at_ns",
        "pending_count",
        "oldest_pending_age_s",
        "publication_throughput_per_s",
        "completed_delay_median_s",
    }
    for partition in ("development", "protected"):
        if (
            set(pq.read_schema(directory / partition / "observations.parquet").names)
            != public
        ):
            raise ValueError("Unexpected public sensor field")
    return physical


def verify(tree, read, workers):
    directory = ROOT / READY
    tables, source_counts = {}, {}
    for path in tree:
        if not path.is_relative_to(READY) or not (
            path.suffix == ".gz" or path.name == "inventory.json"
        ):
            continue
        key = str(path.relative_to(READY))
        data = read(tree[path])
        source = (
            json.loads(data)
            if path.suffix == ".json"
            else [json.loads(line) for line in gzip.decompress(data).splitlines()]
        )
        actual = list(legacy_records(directory, key))
        for i, (row, original) in enumerate(zip(actual, source, strict=True)):
            exact(row, original, f"{key}[{i}]")
        tables[key] = actual
        source_counts[key] = len(source)
    physical = _physical_counts(tables, directory)
    labelled, protected_records = {}, []
    for partition in ("development", "protected"):
        public = tables[f"{partition}/observations.jsonl.gz"]
        private = tables[f"{partition}/evaluator.jsonl.gz"]
        inventory = [r for r in tables["inventory.json"] if r["partition"] == partition]
        labelled[partition] = []
        for obs, evaluator, root in zip(public, private, inventory, strict=True):
            if evaluator["key"] != root["key"] or obs["key"] != root["key"]:
                raise ValueError("Misaligned public/evaluator keys")
            record = dict(evaluator, history=obs["history"])
            for branch in evaluator["branches"].values():
                breach = int(
                    any(20 * w["misses"] > w["eligible"] for w in branch["windows"])
                )
                if branch["breach"] != breach:
                    raise ValueError(
                        "Stored horizon breach differs from deadline counts"
                    )
            if partition == "protected":
                protected_records.append(record)
            labelled[partition].append(
                dict(
                    key=root["key"],
                    condition=root["condition"],
                    partition=partition,
                    history=obs["history"],
                    demand=evaluator["truth"]["demand"],
                    unfinished=evaluator["truth"]["unfinished"],
                    **{k: v["breach"] for k, v in evaluator["branches"].items()},
                )
            )
    for row, public, private in zip(
        labelled["protected"],
        tables["protected/predictions.jsonl.gz"],
        tables["protected/evaluator-predictions.jsonl.gz"],
        strict=True,
    ):
        if not row["key"] == public["key"] == private["key"]:
            raise ValueError("Misaligned protected forecasts")
        row["predictions"] = dict(
            public["predictions"], evaluator=private["predictions"]
        )
    manifest = json.loads((directory / "manifest.json").read_bytes())
    scores = json.loads((directory / "scores.json").read_bytes())
    analysis = audit_analysis(
        labelled["development"],
        labelled["protected"],
        tables["development/fit.jsonl.gz"][0]["fit"],
        scores,
        manifest["fold_seed"],
        manifest["bootstrap_seed"],
    )
    frozen = json.loads(read(tree[READY / "independent-audit.json"]))
    exact(
        analysis,
        {k: frozen[k] for k in analysis},
        "independently recomputed telemetry score audit",
    )
    if analysis["verdict"] != "READY":
        raise ValueError("Telemetry verdict changed")
    replay = tables["protected/replay-audit.jsonl.gz"]
    admissions = branches = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for new, saved in zip(
            pool.map(_replay, protected_records, chunksize=4), replay, strict=True
        ):
            exact(new, saved, f"protected physical replay {saved['key']}")
            admissions += new["receipt"]["admissions"]
            branches += new["receipt"]["branches"]
    return dict(
        **analysis,
        locked_claims=scores["claims"],
        original_record_counts=source_counts,
        parquet_rows=physical,
        replayed_protected_branches=branches,
        replayed_admissions=admissions,
    )
