"""Validation protocol fixtures with known queue waits and independent seed samples."""
import pytest

from experiments.validation import isolated_wait_ms


def test_queue_mean_excludes_warmup_but_retains_its_unfinished_work():
    # Warm-up job runs from tick 1 to 11; the two scored jobs start at 11 and 13.
    assert isolated_wait_ms([1, 2, 12], [10, 2, 3], discard=1) == pytest.approx(5 / 1_000_000)


@pytest.mark.parametrize(("distribution","rate","expected"), [
    ("M/M/1", 50, 10), ("M/M/1", 90, 90),
    ("M/D/1", 50, 5), ("M/D/1", 90, 45),
])
def test_queue_validation_uses_entire_interval(distribution, rate, expected):
    from experiments.validation import queue_cell
    exact = queue_cell(distribution, rate, 100, [expected]*32, 0.1)
    assert exact["passed"] and exact["expected_ms"] == pytest.approx(expected)
    uncertain = queue_cell(distribution, rate, 100, [0, 2*expected]*16, 0.1)
    assert not uncertain["passed"] and uncertain["status"] == "insufficient precision"
    disagrees = queue_cell(distribution, rate, 100, [2*expected]*32, 0.1)
    assert not disagrees["passed"] and disagrees["status"] == "disagreement"



def test_replay_fixture_checks_inherited_ack_and_residual_db_work():
    from experiments.config import read_config
    from experiments.validation import replay_check
    evidence = replay_check(read_config("configs/pilot.toml"))
    assert evidence["passed"]
    assert evidence["cut_phase"] == "ack"
    assert evidence["inherited_ack_due_ns"] == 1_000_000
    assert evidence["active_query_residual_ns"] == 9_500_000



def test_validation_writes_failed_queue_checks_and_placebo_evidence(tmp_path):
    from dataclasses import replace
    from experiments.config import read_config
    from experiments.validation import validate
    cfg = replace(read_config("configs/pilot.toml"), seed=13, queue_warmup=0,
                  queue_roots=2, queue_scored=1, placebo_roots=1, warmup=0,
                  horizon=2, window=1, queue_rates=[50])
    result = validate(cfg, tmp_path, 1)
    assert not result["passed"]
    assert result["checks"]["replay"]["passed"]
    assert result["placebo"][0]["passed"]
    assert result["queue"][0]["status"] == "disagreement"
    assert (tmp_path / "validation/queue.json").exists()
    assert (tmp_path / "samples/C1/0000.npz").exists()

