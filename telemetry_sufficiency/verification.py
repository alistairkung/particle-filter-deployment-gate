"""Publication checks for the score lock, independent audit and pilot comparison."""

from math import isfinite
from pathlib import Path

from .reproduce import reproduction_commands

from . import CONDITIONS, PILOT
from .archive import historical, legacy_digest
from .custody import canonical, file_hash, read


def verify_report_evidence(run):
    scores = read(run / "scores.json")
    try:
        verdict = _score_verdict(scores, read(run / "predictor-lock.json")["selected"])
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"Incomplete score evidence: {exc}") from exc
    lock = file_hash(run / "predictor-lock.json")
    audit_path = run / "independent-audit.json"
    audit = read(audit_path)
    expected = dict(
        verified=True,
        verdict=verdict,
        development_roots=1200,
        protected_roots=1200,
        predictions_checked=64800,
        comparisons_checked=77,
        lock=lock,
        scores=file_hash(run / "scores.json"),
        replay_ledger=(
            legacy_digest(run, "protected/replay-audit.jsonl.gz")
            if historical(run)
            else file_hash(run / "protected/replay-audit.parquet")
        ),
    )
    if any(canonical(audit.get(k)) != canonical(v) for k, v in expected.items()):
        raise ValueError(
            "Independent scientific reconstruction missing, incomplete or stale"
        )

    verify_pilot_receipt(
        read(run / "pilot-reproduction.json"), archived=historical(run)
    )
    review = read(run / "verification-review.json")
    findings = review.get("findings")
    if isinstance(findings, str):
        findings = [findings]
    if not (
        review.get("passed") is True
        and review.get("audit") == file_hash(audit_path)
        and review.get("lock") == lock
        and isinstance(findings, list)
        and findings
        and all(isinstance(line, str) and line.strip() for line in findings)
    ):
        raise ValueError("Independent implementation review missing, invalid or stale")
    return verdict


def verify_pilot_receipt(pilot, *, archived=False):
    frozen = PILOT / "results/frozen"
    original_path = frozen / "summary/clean-reproduction.json"
    original = read(original_path)
    reference = original
    if not archived:
        if pilot.get("format") != "parquet-v1":
            raise ValueError(
                "A fresh study requires a canonical Parquet pilot reproduction"
            )
        from experiments.compare import frozen_reproduction_receipt

        reference = frozen_reproduction_receipt()
    expected = {
        k: reference[k]
        for k in (
            "files_compared",
            "inventory_sha256",
            "signature",
            "by_area",
            "completed_roots",
        )
    }
    expected.update(
        passed=True,
        verdict="GO",
        pilot_revision="bb20ebe",
        original_receipt=file_hash(original_path),
        frozen_manifest=file_hash(frozen / "manifest.json"),
    )
    if not archived:
        for key in ("reproduction_signature", "reproduction_inventory_sha256"):
            expected[key] = reference[key]
    commands = pilot.get("commands")
    try:
        output = Path(commands[1][-1])
    except (IndexError, TypeError, KeyError) as exc:
        raise ValueError("Missing frozen pilot reproduction commands") from exc
    if (
        not output.is_absolute()
        or output.is_relative_to(PILOT)
        or commands != reproduction_commands(output)
    ):
        raise ValueError("Invalid frozen pilot reproduction commands")
    elapsed = pilot.get("reproduction_elapsed_s")
    if not (
        all(canonical(pilot.get(k)) == canonical(v) for k, v in expected.items())
        and isinstance(elapsed, dict)
        and set(elapsed) == set(original["reproduction_elapsed_s"])
        and all(
            type(t) in (float, int) and isfinite(t) and t >= 0 for t in elapsed.values()
        )
    ):
        raise ValueError("Frozen pilot reproduction missing, invalid or stale")


RISK = ("hold", "promote", "increment")
MODELS = ("prior", "age", "current", "history", "evaluator")
PHYSICAL_SCORES = (
    "demand",
    "unfinished",
    "condition",
    "demand_rmse",
    "unfinished_rmse",
)


def _finite(row, fields):
    for field in fields:
        value = row[field]
        if type(value) not in (int, float) or not isfinite(value):
            raise ValueError(f"Missing or nonfinite scientific value for {field}")


def _counts(values, n, size):
    counts = list(values)
    if (
        len(counts) != size
        or any(type(v) is not int or v < 0 for v in counts)
        or sum(counts) != n
    ):
        raise ValueError(f"Invalid count inventory for {n} roots")


def _check_population(population, n, surprise):
    if type(population["n"]) is not int or population["n"] != n:
        raise ValueError(f"Invalid population size, expected {n}")
    _counts(
        (
            population["paired"][k]
            for k in ("neither", "promote_only", "hold_only", "both")
        ),
        n,
        4,
    )
    _finite(population["rates"], RISK)
    for level in MODELS:
        model = population["models"][level]
        _finite(model["scores"], RISK)
        if level != "evaluator" and not surprise:
            _finite(model["scores"], PHYSICAL_SCORES)
            matrix = model["confusion"]
            if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                raise ValueError("Incomplete condition confusion matrix")
            _counts((v for row in matrix for v in row), n, 9)
        for action in RISK[:2]:
            bins = model["calibration"][action]
            _counts((b["count"] for b in bins), n, 10)
            for b in bins:
                if b["count"]:
                    _finite(b, ("mean_prediction", "event_rate"))
                elif (b["mean_prediction"], b["event_rate"]) != (None, None):
                    raise ValueError("Empty calibration bin has a reported estimate")


def _score_verdict(scores, selected):
    claims = scores["claims"]
    if set(claims) != {"demand", "increment"} or set(selected) != set(claims):
        raise ValueError("Incomplete locked claims")
    for name, conditions in (("demand", CONDITIONS[:2]), ("increment", CONDITIONS)):
        row = claims[name]
        _finite(row, ("constant", "gain", "prior_gain", "lower", "se"))
        sizes = {c: 400 for c in conditions}
        if (
            row["candidate"] != selected[name]
            or selected[name] not in MODELS[1:4]
            or canonical(row["counts"]) != canonical(sizes)
            or type(row["n"]) is not int
            or row["n"] != sum(sizes.values())
            or row["se"] < 0
            or row["passes"] is not (row["se"] > 0 and row["lower"] > 0)
        ):
            raise ValueError(f"Invalid locked {name} claim")
    sizes = {
        "all": 1200,
        "nominal_publisher": 800,
        "surprise": 400,
        **dict.fromkeys(CONDITIONS, 400),
    }
    for name, n in sizes.items():
        _check_population(scores["populations"][name], n, name == "surprise")
    comparisons = scores["comparisons"]
    keys = {
        (r["population"], r["target"], r["baseline"], r["candidate"])
        for r in comparisons
    }
    if len(comparisons) != 77 or len(keys) != 77:
        raise ValueError("Incomplete paired comparisons")
    for row in comparisons:
        _finite(row, ("gain", "lower", "upper"))
    overlap = scores["overlap"]
    if (
        len({(r["first"], r["second"], r["metric"]) for r in overlap}) != 6
        or len(overlap) != 6
    ):
        raise ValueError("Incomplete condition overlap")
    for row in overlap:
        _finite(row, ("overlap",))
    changes = scores["surprise_changes"]
    for group in [changes["rates"], *(changes["scores"][level] for level in MODELS)]:
        for target in RISK:
            _finite(group[target], ("change", "lower", "upper"))
    return (
        "READY"
        if all(row["passes"] for row in claims.values())
        else "READY WITH LIMITATIONS"
    )
