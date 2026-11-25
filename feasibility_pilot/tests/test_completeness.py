import json
from pathlib import Path

import pytest

from experiments.compare import compare
from experiments.storage import save_json


def test_two_empty_inventories_cannot_reproduce(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    record = dict(
        status="report complete", signature="same", completed={}, timings_s={}
    )
    for path in (first, second):
        save_json(path / "manifest.json", record)
    with pytest.raises((ValueError, FileNotFoundError), match="manifest|pilot.toml"):
        compare(first, second)


def test_source_signature_substitution_preserves_every_scientific_field(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from experiments.compare import _with_source_signature, _inventory_digest
    from experiments.storage import digest

    table = Path("evidence/C23/branches.parquet")
    validation = Path("validation/result.json")
    samples = Path("evidence/C23/samples.parquet")

    def example(root, signature, breach):
        (root / table).parent.mkdir(parents=True)
        pq.write_table(
            pa.table({"signature": [signature], "breach": [breach]}),
            root / table,
            compression="zstd",
            write_page_checksum=True,
        )
        save_json(root / validation, {"signature": signature, "passed": True})
        (root / samples).write_bytes(b"unchanged latency evidence")
        return {p: digest(root / p) for p in (table, validation, samples)}

    frozen = tmp_path / "frozen"
    original = example(frozen, "original", 0)
    expected = _with_source_signature(frozen, original, "extracted")
    fresh = example(tmp_path / "fresh", "extracted", 0)
    assert _inventory_digest(fresh) == expected
    assert _inventory_digest(example(tmp_path / "changed", "extracted", 1)) != expected
    assert digest(frozen / table) == original[table]


def test_identical_missing_root_file_fails_even_when_both_manifests_claim_complete(
    tmp_path,
):
    from dataclasses import asdict
    import hashlib
    import shutil

    from experiments.config import read_config
    from experiments.compare import expected_inventory
    from experiments.storage import budgets, digest

    config_path = tmp_path / "dev.toml"
    frozen = Path("configs/pilot.toml").read_text()
    for field in [
        "placebo_roots",
        "migration_roots",
        "overload_roots",
        "publisher_roots",
    ]:
        import re

        frozen = re.sub(rf"^{field} = \d+$", field + " = 1", frozen, flags=re.MULTILINE)
    config_path.write_text(frozen)
    config = read_config(config_path)
    files = {"config": digest(config_path)}
    signature = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = dict(
        config=asdict(config),
        expected=budgets(config),
        completed=budgets(config),
        input_digests=files,
        signature=signature,
        status="report complete",
    )
    first, second = tmp_path / "first", tmp_path / "second"
    for path in (first, second):
        save_json(path / "manifest.json", manifest)
        shutil.copyfile(config_path, path / "pilot.toml")
        for entry in expected_inventory(config):
            if entry != Path("evidence/C5/samples.parquet"):
                (path / entry).parent.mkdir(parents=True, exist_ok=True)
                (path / entry).write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="evidence/C5/samples.parquet"):
        compare(first, second, config_path)
    manifest["completed"]["C4"] = 0
    save_json(first / "manifest.json", manifest)
    with pytest.raises(ValueError, match="root counts"):
        compare(first, second, config_path)


def test_failed_report_archives_prior_conclusion_and_records_unaudited_status(
    tmp_path, monkeypatch
):
    import shutil
    import sys

    from experiments.__main__ import main
    from experiments.storage import load_json

    result = tmp_path / "run"
    save_json(result / "summary/verdict.json", {"verdict": "GO"})
    save_json(result / "summary/independent-audit.json", {"verified": True})
    shutil.copyfile("configs/pilot.toml", result / "pilot.toml")
    report = tmp_path / "previous_report.md"
    report.write_text("old GO finding")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experiments",
            "report",
            "--from",
            str(result),
            "--plots",
            str(tmp_path / "plots"),
            "--output",
            str(report),
        ],
    )
    with pytest.raises(FileNotFoundError):
        main()
    invalid = load_json(result / "summary/verdict.json")
    assert invalid["verdict"] == "MODIFY / inconclusive" and not invalid["audited"]
    assert load_json(result / "summary/independent-audit.json")["verified"] is False
    assert "MODIFY / inconclusive" in report.read_text()
    archived = list((result / "history").glob("*/summary/verdict.json"))
    assert len(archived) == 1 and load_json(archived[0]) == {"verdict": "GO"}
    assert (archived[0].parents[1] / "report.md").read_text() == "old GO finding"


def test_first_config_error_can_be_corrected_without_discarding_failure(tmp_path, monkeypatch):
    import sys
    from experiments.__main__ import main
    from experiments.storage import load_json

    result = tmp_path / "run"
    argv = ["experiments", "validate", "--out", str(result), "--config"]
    monkeypatch.setattr(sys, "argv", argv + [str(tmp_path / "missing.toml")])
    with pytest.raises(FileNotFoundError):
        main()
    monkeypatch.setattr("experiments.validation.validate", lambda *args: {"passed": True})
    monkeypatch.setattr(sys, "argv", argv + ["configs/pilot.toml"])
    main()
    manifest = load_json(result / "manifest.json")
    assert manifest["signature"] and manifest["status"] == "validate complete"
    assert len(list((result / "failures").glob("*.json"))) == 1


def test_malformed_manifest_preserved_and_failed_publication_invalidated(tmp_path, monkeypatch):
    import sys
    from experiments.__main__ import main
    from experiments.storage import load_json

    result = tmp_path / "run"
    save_json(result / "summary/verdict.json", {"verdict": "GO"})
    (result / "manifest.json").write_text("{damaged")
    report = result / "report.md"
    monkeypatch.setattr(sys, "argv", [
        "experiments", "report", "--from", str(result), "--plots", str(result / "plots"),
        "--output", str(report),
    ])
    with pytest.raises(json.JSONDecodeError):
        main()
    assert next((result / "failures").glob("*-manifest.json")).read_text() == "{damaged"
    assert load_json(result / "summary/verdict.json")["audited"] is False
    assert "MODIFY / inconclusive" in report.read_text()
    with pytest.raises(ValueError, match="Invalid prior manifest"):
        main()


def test_failed_report_to_new_path_also_invalidates_prior_publication(tmp_path, monkeypatch):
    import sys
    from experiments.__main__ import main
    from experiments.storage import digest

    old = tmp_path / "published.md"
    new = tmp_path / "alternate.md"
    old.write_text("old GO finding")
    save_json(tmp_path / "manifest.json", {
        "published_report": {"path": str(old), "sha256": digest(old)}
    })
    monkeypatch.setattr(sys, "argv", [
        "experiments", "report", "--from", str(tmp_path), "--plots", str(tmp_path / "plots"),
        "--output", str(new),
    ])
    with pytest.raises(FileNotFoundError):
        main()
    assert all("MODIFY / inconclusive" in p.read_text() for p in (old, new))
    assert next((tmp_path / "history").glob("*/previous-publication.md")).read_text() == "old GO finding"


def test_copy_of_frozen_run_cannot_archive_historical_report(tmp_path):
    from experiments.__main__ import FROZEN_REPORT, previous_publication
    from experiments.storage import digest

    manifest = {
        "published_report": {"path": str(FROZEN_REPORT), "sha256": digest(FROZEN_REPORT)}
    }
    assert previous_publication(manifest, tmp_path) is None
