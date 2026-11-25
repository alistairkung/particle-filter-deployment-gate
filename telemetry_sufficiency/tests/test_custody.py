import pytest
from telemetry_sufficiency import custody


def test_inventory_groups_all_children_of_an_opaque_root():
    inventory = custody.inventory()
    assert len(inventory) == len({row["key"] for row in inventory}) == 2400
    assert sum(row["partition"] == "protected" for row in inventory) == 1200
    for row in inventory:
        assert row["condition"] not in row["key"]
        expected = (
            4
            if (row["condition"], row["partition"]) == ("reference", "protected")
            else 2
        )
        assert len(custody.branch_names(row)) == expected


def test_evidence_inventory_rejects_cross_partition_and_swapped_keys(tmp_path):
    rows = custody.inventory()[:2]
    data = [{"key": r["key"], "value": n} for n, r in enumerate(rows)]
    path = tmp_path / "out.jsonl.gz"
    custody.write_table(path, data)
    assert list(custody.read_table(path, rows)) == data
    with pytest.raises(ValueError):
        list(custody.read_table(path, rows[::-1]))
    with pytest.raises(ValueError):
        list(custody.read_table(path, rows[:1]))


def test_public_envelope_excludes_evaluator_truth(tmp_path):
    row = custody.inventory()[0]
    record = dict(
        condition=row["condition"],
        partition=row["partition"],
        index=row["index"],
        history=[],
        truth={"demand": 0.2},
        evidence={"cut": "private"},
        branches={name: {"breach": 0} for name in custody.branch_names(row)},
    )
    custody.publish_partition(tmp_path, [row], [record])
    public = list(
        custody.read_table(tmp_path / "development" / "observations.jsonl.gz", [row])
    )
    assert public == [{"key": row["key"], "history": []}]
    assert list(custody.records(tmp_path, [row])) == [record]


def test_failed_stage_invalidates_downstream_and_preserves_failure(tmp_path):
    from telemetry_sufficiency.workflow import attempt

    custody.save(
        tmp_path / "manifest.json",
        {"completed": {"validate": {}, "lock": {}, "report": {}}},
    )
    with pytest.raises(ValueError), attempt(tmp_path, "lock"):
        raise ValueError("invalid lock")
    manifest = custody.read(tmp_path / "manifest.json")
    assert set(manifest["completed"]) == {"validate"}
    assert manifest["status"] == "lock failed"
    assert list((tmp_path / "failures").glob("*/failure.json"))


def test_invalidated_evidence_preserves_incoming_manifest_and_observed_digest(tmp_path):
    from telemetry_sufficiency.workflow import execute

    source = tmp_path / "inventory.json"
    custody.save(source, ["original"])
    frozen = custody.file_hash(source)
    incoming = {"completed": {"validate": {"inventory.json": frozen}, "report": {}}}
    custody.save(tmp_path / "manifest.json", incoming)
    (tmp_path / "Experimental_Readiness_Report.md").write_text("Prior decision")
    custody.save(source, ["changed"])
    with pytest.raises(ValueError, match="digest mismatch"):
        execute("validate", tmp_path, 1)

    (failure,) = (tmp_path / "failures").glob("*/failure.json")
    assert custody.read(failure)["mismatch"] == {
        "path": "inventory.json",
        "expected": frozen,
        "observed": custody.file_hash(source),
    }
    assert custody.read(failure.parent / "manifest.json") == incoming
    assert (
        failure.parent / "Experimental_Readiness_Report.md"
    ).read_text() == "Prior decision"
    assert custody.read(tmp_path / "readiness.json")["verdict"] == "NOT READY"
    assert "NOT READY" in (tmp_path / "Experimental_Readiness_Report.md").read_text()


def test_pilot_reproduction_guard_and_failed_retry(tmp_path, monkeypatch):
    import fcntl
    import subprocess
    from telemetry_sufficiency import PILOT
    from telemetry_sufficiency.reproduce import reproduce

    def fail(*args, **kwargs):
        raise RuntimeError("fixture pilot validation failed")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(ValueError, match="frozen pilot"):
        reproduce(PILOT / "pilot-reproduction.json")

    receipt = tmp_path / "pilot-reproduction.json"
    custody.save(receipt, {"passed": True})
    incoming = {"completed": {"audit": {}, "report": {}}}
    custody.save(tmp_path / "manifest.json", incoming)
    (tmp_path / "Experimental_Readiness_Report.md").write_text("prior report")
    with (tmp_path / ".active").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            reproduce(receipt)
    with pytest.raises(RuntimeError, match="fixture pilot validation"):
        reproduce(receipt)
    assert not receipt.exists()
    assert custody.read(tmp_path / "manifest.json")["completed"] == {"audit": {}}
    (archive,) = (tmp_path / "failures").iterdir()
    assert custody.read(archive / receipt.name) == {"passed": True}
    assert (archive / "Experimental_Readiness_Report.md").read_text() == "prior report"
    assert custody.read(tmp_path / "readiness.json")["stage"] == "reproduce"


def test_corrupt_manifest_publishes_failure_and_preserves_original_bytes(tmp_path):
    from telemetry_sufficiency.workflow import execute

    (tmp_path / "manifest.json").write_text('{"completed":null}')
    with pytest.raises(ValueError, match="manifest"):
        execute("report", tmp_path, 1)
    (archive,) = (tmp_path / "failures").iterdir()
    assert (archive / "manifest.json").read_text() == '{"completed":null}'
    assert custody.read(tmp_path / "readiness.json")["verdict"] == "NOT READY"
