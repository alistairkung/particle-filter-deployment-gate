import pytest
from telemetry_sufficiency import custody, parquet_evidence as evidence


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
    directory = tmp_path / "development"
    data = [{"key": r["key"], "readings": n} for n, r in enumerate(rows)]
    evidence.write(directory, "history-index", data)

    def checked(expected):
        return list(
            evidence.checked(evidence.rows(directory, "history-index"), expected)
        )

    assert checked(rows) == data
    with pytest.raises(ValueError):
        checked(rows[::-1])
    with pytest.raises(ValueError):
        checked(rows[:1])
    import shutil

    shutil.copytree(directory, tmp_path / "protected")
    with pytest.raises(ValueError, match="partition"):
        list(evidence.rows(tmp_path / "protected", "history-index"))


def test_public_envelope_excludes_evaluator_truth(tmp_path):
    row = custody.inventory()[0]
    from telemetry_sufficiency.plant import simulate_root

    record = simulate_root(row["condition"], row["partition"], row["index"])
    record["history"] = []
    custody.publish_partition(tmp_path, [row], [record])
    public = list(evidence.histories(tmp_path / "development", [row]))
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


def test_retry_uses_hash_bound_marker_then_only_parquet(tmp_path, monkeypatch):
    from telemetry_sufficiency import plant, workflow

    root = custody.inventory()[0]
    task = tmp_path, root, "locked-signature"
    original = workflow._one_root(task)

    def fail(*args):
        raise AssertionError("A completed root must not be simulated again")

    monkeypatch.setattr(plant, "simulate_root", fail)
    assert workflow._one_root(task) == original
    with pytest.raises(ValueError, match="completion marker"):
        workflow._one_root((tmp_path, root, "different"))
    with workflow.attempt(tmp_path, "develop") as manifest:
        files = custody.publish_partition(tmp_path, [root], [original])
        manifest["completed"]["develop"] = workflow.relative_files(tmp_path, files)
    assert not list((tmp_path / "work/development").glob("*.json"))
    assert list(custody.records(tmp_path, [root])) == [original]


def test_observation_reader_rejects_private_columns(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    partition = tmp_path / "development"
    root = custody.inventory()[0]
    evidence.write(partition, "history-index", [{"key": root["key"], "readings": 0}])
    path = evidence.write(partition, "observations", [])
    amended = pq.read_table(path).append_column(
        "condition", pa.array([], type=pa.string())
    )
    pq.write_table(amended, path)
    with pytest.raises(ValueError, match="schema"):
        list(evidence.histories(partition, [root]))


@pytest.mark.parametrize("study", ["pilot", "telemetry"])
def test_cli_rejects_descendants_of_both_frozen_publications(study):
    from telemetry_sufficiency import PILOT
    from telemetry_sufficiency.archive import V1, regenerate
    from telemetry_sufficiency.migrate import migrate
    from telemetry_sufficiency.reproduce import reproduce
    from telemetry_sufficiency.workflow import execute

    root = V1 if study == "telemetry" else PILOT / "results/frozen"
    child = root / "nested/attempt"
    for action in (
        lambda: execute("validate", child, 1),
        lambda: reproduce(child / "pilot-reproduction.json"),
        lambda: migrate(child),
        lambda: regenerate("report", V1, child, 1),
    ):
        with pytest.raises(ValueError, match="output path"):
            action()
    assert not child.exists()


def test_output_symlink_cannot_publish_into_archive(tmp_path):
    from telemetry_sufficiency.archive import V1

    (tmp_path / "plots").symlink_to(V1 / "plots", target_is_directory=True)
    with pytest.raises(ValueError, match="retained telemetry"):
        custody.save(tmp_path / "plots/action-calibration.png", {})


def test_stage_completion_verifies_outputs_before_publication(tmp_path):
    from telemetry_sufficiency.workflow import attempt, relative_files, EvidenceMismatch

    evidence = tmp_path / "observed"
    evidence.write_bytes(b"original")
    with pytest.raises(EvidenceMismatch), attempt(tmp_path, "develop") as manifest:
        manifest["completed"]["develop"] = relative_files(tmp_path, [evidence])
        evidence.write_bytes(b"corrupt")
    saved = custody.read(tmp_path / "manifest.json")
    assert saved["completed"] == {}
    assert saved["attempts"][-1]["status"] == "develop failed"
    assert custody.read(tmp_path / "readiness.json")["verdict"] == "NOT READY"


def test_failed_checkpoint_cleanup_revokes_ready_and_retries(tmp_path, monkeypatch):
    from telemetry_sufficiency import workflow

    root = custody.inventory()[0]
    marker = tmp_path / "work/development" / (root["key"] + ".json")
    custody.save(marker, {"completed": True})
    custody.save(
        tmp_path / "manifest.json", {"completed": {"develop": {}, "report": {}}}
    )
    custody.save(tmp_path / "readiness.json", {"verdict": "READY"})

    def fail(*args):
        raise OSError("fixture marker cleanup failed")

    with monkeypatch.context() as scoped:
        scoped.setattr(workflow, "_clear_root_markers", fail)
        with (
            pytest.raises(OSError, match="cleanup"),
            workflow.attempt(tmp_path, "develop"),
        ):
            pass
    saved = custody.read(tmp_path / "manifest.json")
    assert "report" not in saved["completed"] and "develop" not in saved["completed"]
    assert custody.read(tmp_path / "readiness.json")["verdict"] == "NOT READY"
    assert marker.exists()
    with workflow.attempt(tmp_path, "develop") as manifest:
        manifest["completed"]["develop"] = {}
        manifest["status"] = "develop complete"
    assert not marker.exists()
    assert custody.read(tmp_path / "manifest.json")["status"] == "develop complete"
