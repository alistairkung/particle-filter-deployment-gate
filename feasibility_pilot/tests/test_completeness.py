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
            if entry != Path("samples/C5/0000.npz"):
                (path / entry).parent.mkdir(parents=True, exist_ok=True)
                (path / entry).write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="samples/C5/0000.npz"):
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
