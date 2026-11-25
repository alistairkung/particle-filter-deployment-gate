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
    assert list((tmp_path / "failures").glob("*.json"))
