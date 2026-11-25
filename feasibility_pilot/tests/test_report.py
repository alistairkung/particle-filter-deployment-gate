"""Acceptance examples for reports built from saved evidence."""

from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.report import latency_distribution


def test_latency_distribution_keeps_censored_writes_in_the_denominator():
    # Three writes: 1 and 3 ms complete, the remaining write is still pending at 2 ms.
    sample = latency_distribution(
        np.array([1_000_000, 2_000_000, 3_000_000]),
        np.array([False, True, False]),
    )
    assert sample["writes"] == 3
    assert sample["censored"] == 1
    assert sample["p50_ms"] == {"lower": 2.0}
    assert sample["p99_ms"] == {"lower": 3.0}


def test_control_summary_keeps_migration_result_when_publisher_is_inconclusive():
    from experiments.report import control_findings

    summary = dict(
        verdict="MODIFY / inconclusive",
        migration_finding="consequential",
        publisher={"passed": False, "health": "inconclusive", "age": {"passed": True}},
    )
    findings = control_findings(summary)
    assert "consequential" in findings["migration"]
    assert "inconclusive" in findings["publisher"]
    assert "demonstrated" in findings["outbox"]


def test_branch_diagnostics_weight_waits_by_started_visits_and_misses_by_writes():
    from experiments.runner import run_root
    from experiments.config import read_config
    from experiments.report import branch_diagnostics

    c = read_config("configs/pilot.toml")
    # Development seed family, not a scientific run.
    from dataclasses import replace

    c = replace(c, seed=12345, warmup=1)
    roots = [run_root(c, "C5", i)[0] for i in range(2)]
    write_windows = [w for root in roots for w in root["slow"]["windows"]]
    diag = branch_diagnostics(c, "C5", "slow", [r["slow"] for r in roots])
    assert diag["branch"]["eligible"] == sum(w["eligible"] for w in write_windows)
    assert diag["branch"]["misses"] == sum(w["misses"] for w in write_windows)
    expected_wait = sum(w["db"]["write"]["wait_sum_ms"] for w in write_windows) / sum(
        w["db"]["write"]["waits"] for w in write_windows
    )
    assert abs(diag["workloads"]["write"]["mean_wait_ms"] - expected_wait) < 1e-10
    assert len(diag["windows"]) == 4
    assert diag["windows"][-1]["pending"] == np.mean(
        [r["slow"]["windows"][-1]["pending"] for r in roots]
    )
    assert diag["workloads"]["query"]["mean_wait_ms"] is None


def test_report_displays_censored_ecdf_band_and_indeterminate_tail(tmp_path):
    from experiments.report import saved_latency_distributions

    branch = {"scored": 3, "censored": 1}
    samples = {
        "slow_ns": np.array([1_000_000, 2_000_000, 3_000_000]),
        "slow_java": np.array([False, True, False]),
        "slow_censored": np.array([False, True, False]),
    }
    table, curve = saved_latency_distributions("slow", [branch], [samples])
    assert table["pooled"]["censored"] == 1 and table["Java"]["p95_ms"] == {
        "lower": 2.0
    }
    assert table["Ruby"]["p50_ms"] == 1.0
    assert curve["lower"][-1] == 2 / 3 and curve["upper"][-1] == 1


def test_figures_are_written_headlessly_for_labeled_review(tmp_path):
    from experiments.figures import plot_risks

    branches = {
        "C23": {
            "ref_hold": {"risk": 0.10, "lower": 0.05, "upper": 0.19},
            "ref_promote": {"risk": 0.42, "lower": 0.31, "upper": 0.48},
        }
    }
    effects = {"ref": {"estimate": 0.32, "lower": 0.16, "upper": 0.42}}
    plot_risks(branches, effects, 0.10, tmp_path)
    assert (tmp_path / "breach-risks.png").stat().st_size > 2000


def test_rendered_report_carries_validation_and_both_independent_findings(tmp_path):
    from dataclasses import replace
    from experiments.config import read_config
    from experiments.report import generate
    from experiments.runner import run_root

    c = replace(read_config("configs/pilot.toml"), seed=456, warmup=1)
    from experiments.parquet import publish_family, read_records

    roots = {}
    for family in ("C1", "C23", "C4", "C5"):
        branches, samples = run_root(c, family, 0)
        record = dict(
            root=0,
            family=family,
            signature="test",
            sample_digest="test",
            branches=branches,
        )
        arrays = {
            arm + "_" + name: a
            for arm, values in samples.items()
            for name, a in zip(("ns", "java", "censored"), values, strict=True)
        }
        publish_family(tmp_path, family, [(record, arrays)])
        roots[family] = read_records(
            tmp_path, family, set(branches), 1, "test", c.horizon // c.window
        )
    validation = {
        "signature": "test",
        "passed": True,
        "queue": [],
        "placebo": [{"arms": roots.pop("C1")[0], "passed": True}],
    }
    pair = dict(
        n=1,
        neither=0,
        hold_only=0,
        promote_only=1,
        both=0,
        estimate=1.0,
        lower=0.2,
        upper=1.0,
    )
    pub_pair = dict(
        n=1,
        neither=1,
        nominal_only=0,
        slow_only=0,
        both=0,
        estimate=0.0,
        lower=-0.1,
        upper=0.1,
    )
    summary = dict(
        verdict="MODIFY / inconclusive",
        reasons=["C5 health remains inconclusive."],
        validation_passed=True,
        risks={
            family: {
                arm: dict(
                    n=1, breaches=b["breach"], risk=b["breach"], lower=0.0, upper=1.0
                )
                for arm, b in values[0].items()
            }
            for family, values in roots.items()
        },
        migration={"ref": pair, "high": pair},
        migration_finding="consequential",
        interaction={"estimate": 0.0, "lower": 0.0, "upper": 0.0},
        overload=pair,
        publisher=dict(
            n=1,
            paired=pub_pair,
            absolute={arm: {"lower": 0.0, "upper": 0.1} for arm in ("nominal", "slow")},
            health="inconclusive",
            age=dict(successes=1, fraction=1.0, lower=0.05, passed=False),
            passed=False,
        ),
    )
    output = tmp_path / "report.md"
    generate(c, tmp_path, tmp_path / "plots", output, validation, roots, summary)
    report = output.read_text()
    for phrase in (
        "MODIFY / inconclusive",
        "C2/C3",
        "C5",
        "inconclusive",
        "p99",
        "throughput",
        "C4",
    ):
        assert phrase.casefold() in report.casefold()
    import re

    readme_link = re.search(r"\[reproduction instructions\]\(([^)]+)\)", report).group(
        1
    )
    assert (output.parent / readme_link).resolve() == Path(__file__).resolve().parents[
        1
    ] / "README.md"
    assert not (
        tmp_path / "summary" / "verdict.json"
    ).exists()  # publication follows independent audit
    assert (tmp_path / "summary" / "window-diagnostics.csv").is_file()
    assert (tmp_path / "plots" / "trace-C5.png").is_file()


@pytest.mark.parametrize("use_symlink", [False, True])
def test_report_output_cannot_move_an_existing_directory(
    tmp_path, monkeypatch, use_symlink
):
    from experiments.__main__ import main

    destination = tmp_path / "published"
    destination.mkdir()
    sentinel = destination / "existing.txt"
    sentinel.write_text("preserve")
    output = destination
    if use_symlink:
        output = tmp_path / "alias"
        output.symlink_to(destination, target_is_directory=True)
    run = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experiments",
            "report",
            "--from",
            str(run),
            "--plots",
            str(tmp_path / "plots"),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 2
    assert sentinel.read_text() == "preserve"
    assert not run.exists()
