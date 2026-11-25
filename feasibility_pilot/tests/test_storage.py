from dataclasses import replace

import numpy as np
import pytest

from experiments.config import read_config
from experiments.storage import compute_root, read_root, root_paths


def test_resume_preserves_pair_and_rejects_wrong_signature_or_samples(tmp_path):
    c = replace(
        read_config("configs/pilot.toml"), seed=123, warmup=1, window=1, horizon=2
    )
    task = (c, tmp_path, "C5", 0, "test-signature")
    compute_root(task)
    branches = read_root(tmp_path, c, "C5", 0, "test-signature")
    assert set(branches) == {"nominal", "slow"}
    assert branches["nominal"]["cut"] == branches["slow"]["cut"]
    path, samples = root_paths(tmp_path, "C5", 0)
    before = path.stat().st_mtime_ns
    compute_root(task)
    assert path.stat().st_mtime_ns == before
    with np.load(samples) as arrays:
        assert len(arrays["nominal_ns"]) == branches["nominal"]["scored"]
    with pytest.raises(ValueError, match="provenance"):
        read_root(tmp_path, c, "C5", 0, "other")
    samples.write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="digest"):
        compute_root(task)


def test_parquet_rejects_reordered_or_modified_individual_samples(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from experiments.migrate import consolidate
    from experiments.parquet import iter_samples, read_records

    c = replace(
        read_config("configs/pilot.toml"),
        warmup=1,
        window=1,
        horizon=2,
        publisher_roots=1,
    )
    compute_root((c, tmp_path, "C5", 0, "dev"))
    consolidate(tmp_path, tmp_path, c, "C5", "dev")
    roots = read_records(tmp_path, "C5", {"nominal", "slow"}, 1, "dev", 2)
    path = tmp_path / "evidence/C5/samples.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0], rows[1] = rows[1], rows[0]
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="sample order"):
        list(iter_samples(tmp_path, "C5", roots))
    rows[0], rows[1] = rows[1], rows[0]
    rows[0]["elapsed_ns"] += 1
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="Sample digest"):
        list(iter_samples(tmp_path, "C5", roots))


def test_parquet_retains_zero_sample_cohort(tmp_path):
    from experiments.migrate import consolidate
    from experiments.parquet import iter_samples, read_records

    c = replace(
        read_config("configs/pilot.toml"),
        warmup=1,
        window=1,
        horizon=1,
        deadline_ms=2000,
        publisher_roots=1,
    )
    compute_root((c, tmp_path, "C5", 0, "dev"))
    consolidate(tmp_path, tmp_path, c, "C5", "dev")
    roots = read_records(tmp_path, "C5", {"nominal", "slow"}, 1, "dev", 1)
    [samples] = iter_samples(tmp_path, "C5", roots)
    assert all(len(array) == 0 for array in samples.values())
    assert all(branch["scored"] == 0 for branch in roots[0].values())


def test_pilot_rejects_receipt_and_report_destinations_in_both_publications(tmp_path):
    from experiments.publications import PILOT, require_output

    telemetry = PILOT.parent / "telemetry_sufficiency/results/v1"
    alias = tmp_path / "archive"
    alias.symlink_to(telemetry, target_is_directory=True)
    for path in (alias / "manifest.json", telemetry / "protected", PILOT / "results/frozen/evidence"):
        with pytest.raises(ValueError, match="retained publications"):
            require_output(path)
    require_output(tmp_path / "new-report.md")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("unknown_root",), "extra"),
        (("root",), False),
        (("branches", "slow", "extra_field"), "diagnostic"),
        (("branches", "slow", "breach"), 0.9),
        (("branches", "slow", "share"), True),
        (("branches", "slow", "share"), 2**53 + 1),
        (("branches", "slow", "scored"), True),
        (("branches", "slow", "java_arrivals"), 2**63),
        (("branches", "slow", "windows", 0, "queue"), None),
        (("branches", "slow", "windows", 0, "empty"), 1),
        (("branches", "slow", "windows", 0, "db", "write", "new_diagnostic"), 11),
        (("branches", "slow", "windows", 0, "db", "write", "wait_p95_ms"), "missing"),
    ],
)
def test_consolidation_rejects_lossy_source_rows(tmp_path, path, value):
    from experiments.migrate import consolidate
    from experiments.storage import load_json, save_json

    c = replace(
        read_config("configs/pilot.toml"),
        warmup=1,
        window=1,
        horizon=2,
        publisher_roots=1,
    )
    compute_root((c, tmp_path, "C5", 0, "dev"))
    marker, _ = root_paths(tmp_path, "C5", 0)
    record = load_json(marker)
    row = record
    for key in path[:-1]:
        row = row[key]
    row[path[-1]] = value
    save_json(marker, record)
    with pytest.raises(ValueError, match="fields|lossless"):
        consolidate(tmp_path, tmp_path, c, "C5", "dev")
    assert not (tmp_path / "evidence/C5").exists()


def test_consolidation_rejects_extra_worker_checkpoint(tmp_path):
    from experiments.migrate import consolidate

    c = replace(
        read_config("configs/pilot.toml"),
        warmup=1,
        window=1,
        horizon=2,
        publisher_roots=1,
    )
    compute_root((c, tmp_path, "C5", 0, "dev"))
    marker, _ = root_paths(tmp_path, "C5", 0)
    marker.with_name("0001.json").write_bytes(marker.read_bytes())
    with pytest.raises(ValueError, match="0001.json"):
        consolidate(tmp_path, tmp_path, c, "C5", "dev")
    assert not (tmp_path / "evidence/C5").exists()


def test_failed_scientific_consolidation_preserves_worker_markers_and_arrays(tmp_path, monkeypatch):
    from experiments.__main__ import execute_scientific
    from experiments.storage import save_json

    c = replace(read_config("configs/pilot.toml"), warmup=1, window=1, horizon=2,
                migration_roots=1, overload_roots=1, publisher_roots=1)
    save_json(tmp_path / "validation/result.json", {"passed": True, "signature": "dev"})

    def failed_consolidation(*args):
        raise ValueError("cannot publish family")

    monkeypatch.setattr("experiments.migrate.consolidate", failed_consolidation)
    with pytest.raises(ValueError, match="cannot publish"):
        execute_scientific(c, tmp_path, "dev", 1, {})
    [archive] = (tmp_path / "failures").glob("*-C23-workers")
    branches = read_root(archive, c, "C23", 0, "dev")
    assert len(branches) == 4
    assert not (tmp_path / "evidence/C23").exists()
