import json

import pytest

from telemetry_sufficiency import PILOT, custody, parquet_evidence
from telemetry_sufficiency.report import verified_verdict
from telemetry_sufficiency.scoring import score
from telemetry_sufficiency.tests.test_scoring import protected_rows


@pytest.fixture(scope="module")
def pilot_inventory():
    from experiments.compare import frozen_reproduction_receipt

    return frozen_reproduction_receipt()


@pytest.fixture
def evidence(tmp_path, pilot_inventory, monkeypatch):
    from experiments import compare

    monkeypatch.setattr(compare, "frozen_reproduction_receipt", lambda: pilot_inventory)
    # Synthetic paired outcomes with all four cells equally frequent in every stratum.
    rows = protected_rows()
    selected = {"demand": "current", "increment": "current"}
    scores = score(rows, selected, 100)
    for population in scores["populations"].values():
        population["n"] *= 100
        population["paired"] = {k: v * 100 for k, v in population["paired"].items()}
        for model in population["models"].values():
            for bins in model["calibration"].values():
                for b in bins:
                    b["count"] *= 100
            if "confusion" in model:
                model["confusion"] = [
                    [100 * v for v in row] for row in model["confusion"]
                ]
    for claim in scores["claims"].values():
        claim["n"] *= 100
        claim["counts"] = {k: v * 100 for k, v in claim["counts"].items()}
    custody.save(tmp_path / "scores.json", scores)
    custody.save(tmp_path / "predictor-lock.json", dict(selected=selected))
    ledger = parquet_evidence.write_replays(
        tmp_path / "protected",
        [
            dict(
                key="fixture",
                receipt=dict(
                    partition="protected",
                    condition="reference",
                    index=0,
                    admissions=1,
                    branches=4,
                    replay_digest="fixture",
                ),
            )
        ],
    )
    audit = dict(
        verified=True,
        protected_roots=1200,
        development_roots=1200,
        predictions_checked=64800,
        comparisons_checked=77,
        replay_ledger=custody.file_hash(ledger),
        scores=custody.file_hash(tmp_path / "scores.json"),
        lock=custody.file_hash(tmp_path / "predictor-lock.json"),
        verdict="READY",
    )
    custody.save(tmp_path / "independent-audit.json", audit)
    frozen = PILOT / "results/frozen"
    original = frozen / "summary/clean-reproduction.json"
    custody.save(
        tmp_path / "pilot-reproduction.json",
        dict(
            pilot_inventory,
            reproduction_elapsed_s=custody.read(original)["reproduction_elapsed_s"],
            verdict="GO",
            pilot_revision="bb20ebe",
            original_receipt=custody.file_hash(original),
            frozen_manifest=custody.file_hash(frozen / "manifest.json"),
            commands=[
                ["pytest", "-q", "tests"],
                [
                    "experiments",
                    "validate",
                    "--config",
                    "configs/pilot.toml",
                    "--out",
                    "/tmp/fixture",
                ],
                [
                    "experiments",
                    "run",
                    "--config",
                    "configs/pilot.toml",
                    "--out",
                    "/tmp/fixture",
                ],
                [
                    "experiments",
                    "report",
                    "--from",
                    "/tmp/fixture",
                    "--plots",
                    "/tmp/fixture/plots",
                    "--output",
                    "/tmp/fixture/FEASIBILITY_REPORT.md",
                ],
            ],
        ),
    )
    custody.save(
        tmp_path / "verification-review.json",
        dict(
            passed=True,
            findings=["Reviewed fixture"],
            audit=custody.file_hash(tmp_path / "independent-audit.json"),
            lock=audit["lock"],
        ),
    )
    return tmp_path


@pytest.mark.parametrize(
    "file,key,value",
    [
        ("verification-review", "passed", "false"),
        ("independent-audit", "verified", 1),
        ("pilot-reproduction", "passed", "true"),
        ("pilot-reproduction", "commands", []),
        ("pilot-reproduction", "files_compared", 0),
        ("pilot-reproduction", "frozen_manifest", "stale"),
        ("pilot-reproduction", "reproduction_signature", "stale"),
        ("pilot-reproduction", "reproduction_inventory_sha256", "stale"),
        ("verification-review", "findings", [False]),
        ("independent-audit", "comparisons_checked", 76),
        ("independent-audit", "scores", "stale"),
        ("independent-audit", "replay_ledger", "stale"),
    ],
)
def test_report_rejects_malformed_receipts(evidence, file, key, value):
    assert verified_verdict(evidence) == "READY"
    path = evidence / f"{file}.json"
    custody.save(path, custody.read(path) | {key: value})
    if file == "independent-audit":
        review = evidence / "verification-review.json"
        custody.save(review, custody.read(review) | {"audit": custody.file_hash(path)})
    with pytest.raises(ValueError):
        verified_verdict(evidence)


@pytest.mark.parametrize(
    "path,value",
    [
        (("claims", "demand", "passes"), "true"),
        (("claims", "increment", "counts", "reference"), 399),
        (("claims", "increment", "constant"), None),
        (("claims", "demand", "se"), float("nan")),
        (("populations", "all", "models", "age", "scores", "promote"), None),
        (("populations", "reference", "paired", "both"), 1.0),
        (("comparisons", 0, "lower"), None),
    ],
)
def test_report_requires_complete_finite_claims_and_scores(evidence, path, value):
    score_path = evidence / "scores.json"
    scores = custody.read(score_path)
    target = scores
    for k in path[:-1]:
        target = target[k]
    target[path[-1]] = value
    score_path.write_text(json.dumps(scores))
    audit = evidence / "independent-audit.json"
    custody.save(audit, custody.read(audit) | {"scores": custody.file_hash(score_path)})
    review = evidence / "verification-review.json"
    custody.save(review, custody.read(review) | {"audit": custody.file_hash(audit)})
    with pytest.raises(ValueError):
        verified_verdict(evidence)


def test_report_interprets_retained_diagnostics(evidence):
    from telemetry_sufficiency.report import render_report

    text = render_report(
        custody.read(evidence / "scores.json"), verified_verdict(evidence)
    )
    assert (
        "Within-condition demand gains were positive in reference and contention"
        in text
    )
    assert "contention–publisher" in text
    assert "100.0%" in text
    assert "surprise" in text and "frozen" in text
    assert "power" in text and "floor" in text


def test_fresh_report_rejects_original_transport_receipt(evidence):
    original = PILOT / "results/frozen/summary/clean-reproduction.json"
    path = evidence / "pilot-reproduction.json"
    old = custody.read(path)
    old.pop("format")
    custody.save(path, old | custody.read(original))
    with pytest.raises(ValueError, match="fresh study"):
        verified_verdict(evidence)
