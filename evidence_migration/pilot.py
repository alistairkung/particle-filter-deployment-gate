"""Prove the frozen pilot's keyed branch, window and complete write evidence equivalent."""

import ast
from collections import defaultdict
import hashlib
import importlib.util
import io
import json
from math import ceil, isclose
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.stats import t

from .source import FROZEN, PILOT, ROOT, canonical, exact, read_csv, sha


# Declared independently of either converter. These apply to recomputed statistics only.
RTOL = ATOL = 1e-10
FAMILIES = {"C1": 16, "C23": 400, "C4": 400, "C5": 400}


def derived(actual, expected, label):
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"{label}: incorrect fields")
        for name, val in expected.items():
            derived(actual[name], val, f"{label}.{name}")
    elif isinstance(expected, list):
        for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
            derived(left, right, f"{label}[{i}]")
    elif isinstance(expected, float):
        if not isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL):
            raise ValueError(
                f"{label}: reconstructed {actual!r}, expected {expected!r}"
            )
    elif actual != expected:
        raise ValueError(f"{label}: reconstructed {actual!r}, expected {expected!r}")


def _by_key(path, family, keys):
    data = {}
    for row in pq.read_table(path, page_checksum_verification=True).to_pylist():
        if row.pop("family") != family:
            raise ValueError(f"{path}: wrong family")
        key = tuple(row.pop(k) for k in keys)
        if key in data:
            raise ValueError(f"{path}: duplicate {key}")
        data[key] = row
    return data


def _sample_groups(table, family, root, sample_info, archive):
    expected_fields = {
        "family": pa.string(),
        "root": pa.int32(),
        "arm": pa.string(),
        "sample_index": pa.int32(),
        "elapsed_ns": pa.int64(),
        "java": pa.bool_(),
        "censored": pa.bool_(),
    }
    if {f.name: f.type for f in table.schema} != expected_fields or any(
        c.null_count for c in table.columns
    ):
        raise ValueError(f"{family}/{root}: invalid sample schema")
    for column, value in (("family", family), ("root", root)):
        if not pc.all(pc.equal(table[column], value)).as_py():
            raise ValueError(
                f"{family}/{root}: sample group contains unexpected {column}"
            )
    arms = set(pc.unique(table["arm"]).to_pylist())
    names = {arm + "_" + field for arm in arms for field in ("ns", "java", "censored")}
    if arms != set(sample_info) or sorted(names) != sorted(archive.files):
        raise ValueError(f"{family}/{root}: sample arm inventory mismatch")
    for arm in sorted(arms):
        group = table.filter(pc.equal(table["arm"], arm))
        ordinals = group["sample_index"].to_numpy()
        if not np.array_equal(ordinals, np.arange(len(group))):
            raise ValueError(
                f"{family}/{root}/{arm}: missing, extra or out-of-order write ordinal"
            )
        columns = [
            group[field].to_numpy() for field in ("elapsed_ns", "java", "censored")
        ]
        checksum = hashlib.sha256()
        for arr, field in zip(columns, ("ns", "java", "censored"), strict=True):
            legacy = archive[f"{arm}_{field}"]
            if legacy.dtype != arr.dtype or not np.array_equal(legacy, arr):
                raise ValueError(
                    f"{family}/{root}/{arm}.{field}: stored write sample mismatch"
                )
            checksum.update(arr.tobytes())
        scored, censored, values_hash = sample_info[arm]
        if (len(group), int(columns[2].sum()), checksum.hexdigest()) != (
            scored,
            censored,
            values_hash,
        ):
            raise ValueError(
                f"{family}/{root}/{arm}: incorrect sample counts or digest"
            )
        yield arm, columns


def _csv_windows(data, windows):
    seen = set()
    for raw in read_csv(data):
        key = int(raw.pop("root")), raw.pop("arm"), int(raw.pop("window"))
        source = windows[key]
        flat = {k: v for k, v in source.items() if k != "db"}
        flat.update(
            {
                f"{queue}_{k}": v
                for queue, values in source["db"].items()
                for k, v in values.items()
            }
        )
        if key in seen or raw != {
            k: "" if v is None else str(v) for k, v in flat.items()
        }:
            raise ValueError(f"CSV window discrepancy at {key}")
        seen.add(key)
    if seen != windows.keys():
        raise ValueError("CSV and Parquet window inventories differ")


def _latency_summary(ns, censored):
    # Select order statistics directly in integer nanoseconds for the inverted empirical CDF.
    result = {"writes": len(ns), "censored": int(censored.sum())}
    completed = np.sort(ns[~censored])
    lower = np.sort(ns)
    for numerator in (50, 95, 99):
        value = None
        if len(ns):
            index = ceil(numerator * len(ns) / 100) - 1
            value = float(lower[index]) / 1_000_000
            if index >= len(completed) or lower[index] != completed[index]:
                value = {"lower": value}
        result[f"p{numerator}_ms"] = value
    return result


def _add_diagnostics(family, roots, arrays, c, diagnostics):
    from experiments.report import branch_diagnostics

    for arm, samples in arrays.items():
        group = [root[arm] for root in roots]
        rebuilt = branch_diagnostics(c, family, arm, group)
        diagnostics["branch-diagnostics"][family, arm] = rebuilt["branch"]
        for row in rebuilt["windows"]:
            diagnostics["window-diagnostics"][family, arm, str(row.pop("window"))] = row
        for workload, row in rebuilt["workloads"].items():
            diagnostics["db-work"][family, arm, workload] = row
        ns, java, censored = [
            np.concatenate(column) for column in zip(*samples, strict=True)
        ]
        for route, mask in (("pooled", slice(None)), ("Ruby", ~java), ("Java", java)):
            diagnostics["latency-by-route"][family, arm, route] = _latency_summary(
                ns[mask], censored[mask]
            )


def _diagnostics_match(diagnostics, load):
    for name, more_keys in (
        ("branch-diagnostics", []),
        ("window-diagnostics", ["window"]),
        ("db-work", ["workload"]),
        ("latency-by-route", ["route"]),
    ):
        observed = diagnostics[name].copy()
        for row in read_csv(load(f"summary/{name}.csv")):
            key = tuple(row.pop(k) for k in ["family", "arm", *more_keys])
            expected = {k: ast.literal_eval(v) if v else None for k, v in row.items()}
            derived(observed.pop(key), expected, f"{name}{key}")
        if observed:
            raise ValueError(f"{name}: extra diagnostic rows")


def _frozen_audit(tree, read, temp, config, verdict):
    # Run the original independent audit on a temporary legacy-shaped view generated solely
    # from canonical observations. No old branch file or scientific producer is its input.
    path = temp / "original_audit.py"
    path.write_bytes(read(tree[Path(PILOT, "experiments/audit.py")]))
    spec = importlib.util.spec_from_file_location("frozen_pilot_scientific_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit(config, temp, verdict)


def _verify_queue(cells, config):
    for cell in cells:
        measurements = cell["seed_means_ms"]
        mean = float(np.mean(measurements))
        half = float(
            t.ppf(0.975, len(measurements) - 1)
            * np.std(measurements, ddof=1)
            / np.sqrt(len(measurements))
        )
        rho = cell["rate"] / config.queue_service_rate
        target = 1000 / config.queue_service_rate * rho / (1 - rho)
        if cell["distribution"] == "M/D/1":
            target /= 2
        interval = [mean - half, mean + half]
        tolerance = [
            target * (1 - config.queue_tolerance),
            target * (1 + config.queue_tolerance),
        ]
        passed = tolerance[0] <= interval[0] and interval[1] <= tolerance[1]
        calculated = dict(
            expected_ms=target,
            mean_ms=mean,
            interval_ms=interval,
            tolerance_ms=tolerance,
            passed=passed,
            status="pass" if passed else "fail",
        )
        derived(calculated, {k: cell[k] for k in calculated}, "C0 analytic calibration")


def _validated_controls(load, config):
    validation = json.loads(load("validation/result.json"))
    placebo = json.loads(load("validation/placebo.json"))
    exact(validation["placebo"], placebo, "C1 legacy validation copies")
    for name in ("queue", "checks"):
        exact(
            validation[name],
            json.loads(load(f"validation/{name}.json")),
            f"C0 legacy {name} copies",
        )
    _verify_queue(validation["queue"], config)
    return validation


def _check_write_deadlines(values, windows, deadline_ms):
    elapsed, _, censored = values
    missed = (elapsed > deadline_ms * 1_000_000) | censored
    start = 0
    for window in windows:
        end = start + window["eligible"]
        if int(missed[start:end].sum()) != window["misses"]:
            raise ValueError("Write deadlines disagree with window")
        start = end
    if start != len(elapsed):
        raise ValueError("Cohort denominator disagrees with write observations")


def _restore_arms(family, root, original_arms, branches, windows, signature, digest):
    restored = {}
    sample_info = {}
    for arm in original_arms:
        branch = branches.pop((root, arm))
        sample_hash = branch.pop("sample_values_sha256")
        if (branch.pop("signature"), branch.pop("sample_digest")) != (
            signature,
            digest,
        ):
            raise ValueError(f"{family}/{root}/{arm}: wrong checksum or signature")
        if branch["trace"] is None:
            del branch["trace"]
        # The original JSON encoded these three binary64 shares as integers.
        if (family, arm) in (("C1", "all"), ("C1", "zero"), ("C4", "ruby")):
            if not branch["share"].is_integer():
                raise ValueError("Non-integral all/zero share")
            branch["share"] = int(branch["share"])
        branch["windows"] = [
            dict(window=i, **windows.pop((root, arm, i))) for i in range(1, 5)
        ]
        restored[arm] = branch
        sample_info[arm] = (branch["scored"], branch["censored"], sample_hash)
    return restored, sample_info


def _verify_root(
    family,
    root,
    source,
    source_npz,
    branches,
    windows,
    sample_table,
    signature,
    deadline_ms,
):
    digest = sha(source_npz)
    original_arms = source["arms"] if family == "C1" else source["branches"]
    arms, sample_info = _restore_arms(
        family, root, original_arms, branches, windows, signature, digest
    )
    if family == "C1":
        hold = arms["hold"]
        passed = all(
            v["trace"] == hold["trace"] and v["windows"] == hold["windows"]
            for v in arms.values()
        )
        record = dict(arms=arms, root_id=root, passed=passed, sample_digest=digest)
    else:
        record = dict(
            family=family,
            root=root,
            branches=arms,
            sample_digest=digest,
            signature=signature,
        )
    exact(record, source, f"{family}/{root}")
    with np.load(io.BytesIO(source_npz), allow_pickle=False) as archive:
        samples = dict(_sample_groups(sample_table, family, root, sample_info, archive))
    for arm, values in samples.items():
        _check_write_deadlines(values, arms[arm]["windows"], deadline_ms)
    return record, samples


def _verify_family(family, directory, load, validation, config, diagnostics):
    folder = directory / "evidence" / family
    branches = _by_key(folder / "branches.parquet", family, ("root", "arm"))
    windows = _by_key(folder / "windows.parquet", family, ("root", "arm", "window"))
    samples = pq.ParquetFile(
        folder / "samples.parquet", page_checksum_verification=True
    )
    count = FAMILIES[family]
    receipt = dict(
        roots=count,
        branches=len(branches),
        windows=len(windows),
        samples=samples.metadata.num_rows,
    )
    if samples.num_row_groups != count:
        raise ValueError(f"{family}: unexpected root count in sample table")
    if family != "C1":
        _csv_windows(load(f"windows/{family}.csv"), windows)
    records = []
    arrays = defaultdict(list)
    for root in range(count):
        source = (
            validation["placebo"][root]
            if family == "C1"
            else json.loads(load(f"branches/{family}/{root:04}.json"))
        )
        record, observations = _verify_root(
            family,
            root,
            source,
            load(f"samples/{family}/{root:04}.npz"),
            branches,
            windows,
            samples.read_row_group(root),
            validation["signature"],
            config.deadline_ms,
        )
        records.append(record)
        for arm, values in observations.items():
            arrays[arm].append(values)
    if branches or windows:
        raise ValueError(f"{family}: additional branch or window keys")
    key = "arms" if family == "C1" else "branches"
    _add_diagnostics(family, [r[key] for r in records], arrays, config, diagnostics)
    return records, receipt


def _reconstruct_decision(tree, read, load, config, records, validation):
    with TemporaryDirectory() as target:
        temp = Path(target)
        (temp / "validation").mkdir()
        for family, roots in records.items():
            if family != "C1":
                folder = temp / "branches" / family
                folder.mkdir(parents=True)
                for row in roots:
                    (folder / f"{row['root']:04}.json").write_bytes(canonical(row))
        (temp / "validation/result.json").write_bytes(
            canonical(dict(validation, placebo=records["C1"]))
        )
        (temp / "validation/workload-integrity.json").write_bytes(
            load("validation/workload-integrity.json")
        )
        return _frozen_audit(
            tree, read, temp, config, json.loads(load("summary/verdict.json"))
        )


def verify(tree, read):
    from experiments.config import read_config

    directory = ROOT / FROZEN
    config = read_config(directory / "pilot.toml")

    def load(path):
        return read(tree[FROZEN / path])

    validation = _validated_controls(load, config)
    diagnostics = defaultdict(dict)
    records, receipts = {}, {}
    for family in FAMILIES:
        records[family], receipts[family] = _verify_family(
            family, directory, load, validation, config, diagnostics
        )
    _diagnostics_match(diagnostics, load)
    audit = _reconstruct_decision(tree, read, load, config, records, validation)
    derived(
        audit,
        json.loads(load("summary/independent-audit.json")),
        "independent GO reconstruction",
    )
    if audit["verified"] is not True or audit["reconstructed"]["verdict"] != "GO":
        raise ValueError("Pilot decision changed")
    return dict(
        families=receipts,
        verdict="GO",
        analytical_queue_cells=len(validation["queue"]),
        analytical_seed_means=sum(
            len(cell["seed_means_ms"]) for cell in validation["queue"]
        ),
        diagnostic_rows={k: len(v) for k, v in diagnostics.items()},
        independently_rebuilt=audit["reconstructed"],
    )
