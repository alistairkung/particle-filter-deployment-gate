"""Auditable progression from validation to a locked, independently verified report."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import fcntl
from importlib import metadata
from pathlib import Path
import platform
import subprocess
import sys
import time

from . import PILOT
from . import custody as c, parquet_evidence as evidence

HERE = Path(__file__).resolve().parent
STAGES = ("validate", "develop", "lock", "evaluate", "audit", "report")


def study_contract():
    source = sorted(HERE.glob("*.py")) + sorted(HERE.glob("tests/*.py"))
    source += [
        HERE / "SCIENTIFIC_SPEC.md",
        HERE / "IMPLEMENTATION_SPEC.md",
        PILOT / "configs/pilot.toml",
        PILOT / "requirements.lock",
        HERE.parent / "requirements-storage.lock",
    ]
    source += sorted((HERE.parent / "simulator").glob("*.py"))
    source += sorted((PILOT / "simulator").glob("*.py")) + sorted(
        (PILOT / "experiments").glob("*.py")
    )
    return {str(f.relative_to(HERE.parent)): c.file_hash(f) for f in source}


def environment():
    dependencies = [
        s.split("==")[0]
        for lockfile in (
            PILOT / "requirements.lock",
            HERE.parent / "requirements-storage.lock",
        )
        for s in lockfile.read_text().splitlines()
        if "==" in s and not s.startswith("#")
    ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {name: metadata.version(name) for name in dependencies},
    }


def relative_files(run, paths):
    return {str(p.relative_to(run)): c.file_hash(p) for p in paths}


class EvidenceMismatch(ValueError):
    def __init__(self, name, expected, observed):
        self.evidence = dict(path=name, expected=expected, observed=observed)
        super().__init__(
            f"Evidence digest mismatch: {name} (expected {expected}, observed {observed})"
        )


def verify_files(run, mapping):
    for name, expected in mapping.items():
        observed = c.file_hash(run / name) if (run / name).is_file() else None
        if observed != expected:
            raise EvidenceMismatch(name, expected, observed)


REPORT = "Experimental_Readiness_Report.md"
RECEIPTS = (
    "manifest.json",
    "readiness.json",
    REPORT,
    "validation.json",
    "independent-audit.json",
    "verification-review.json",
    "pilot-reproduction.json",
    "predictor-lock.json",
)


def receipt_snapshot(run):
    return {
        name: (run / name).read_bytes() for name in RECEIPTS if (run / name).is_file()
    }


def archive_snapshot(directory, snapshot):
    for name, content in snapshot.items():
        c.atomic(directory / name, lambda stream: stream.write(content))


@contextmanager
def attempt(run, stage, *, started=None):
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".active").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = run / "manifest.json"
        started = started or time.time_ns()
        archive = run / "failures" / str(started)
        incoming = receipt_snapshot(run)
        manifest = {"completed": {}, "attempts": []}
        try:
            if path.exists():
                previous = c.read(path)
                if not (
                    isinstance(previous, dict)
                    and isinstance(previous.get("completed"), dict)
                    and all(
                        k in STAGES and isinstance(v, dict)
                        for k, v in previous["completed"].items()
                    )
                    and isinstance(previous.get("attempts", []), list)
                ):
                    raise ValueError("Invalid incoming study manifest")
                manifest = previous
            if stage == "reproduce":
                # A fresh attempt supersedes the old receipt even if the subprocess fails.
                if any(
                    name in incoming for name in ("pilot-reproduction.json", REPORT)
                ):
                    archive_snapshot(archive, incoming)
                    (run / "pilot-reproduction.json").unlink(missing_ok=True)
                    readiness = run / "readiness.json"
                    if (
                        not readiness.exists()
                        or c.read(readiness)["verdict"] != "NOT READY"
                    ):
                        (run / REPORT).unlink(missing_ok=True)
                        readiness.unlink(missing_ok=True)
                manifest["completed"].pop("report", None)
            yield manifest
            if stage in manifest["completed"]:
                verify_files(run, manifest["completed"][stage])
                if stage in ("develop", "evaluate"):
                    # A crash during cleanup leaves verifiable tables and a resumable run.
                    c.save(path, manifest)
                    _clear_root_markers(run, stage)
        except BaseException as exc:
            archive_snapshot(archive, incoming)
            # A failing validation may have produced fresh diagnostic evidence.
            current = {
                name: content
                for name, content in receipt_snapshot(run).items()
                if content != incoming.get(name)
            }
            archive_snapshot(archive / "attempted", current)
            manifest["status"] = f"{stage} failed"
            invalid_stage = "report" if stage == "reproduce" else stage
            if isinstance(exc, EvidenceMismatch):
                invalid_stage = next(
                    (
                        s
                        for s, files in manifest["completed"].items()
                        if exc.evidence["path"] in files
                    ),
                    stage,
                )
            for invalid in STAGES[STAGES.index(invalid_stage) :]:
                manifest["completed"].pop(invalid, None)
            failure = dict(
                stage=stage,
                verdict="NOT READY",
                severity="material",
                reason=f"{type(exc).__name__}: {exc}",
                at_ns=time.time_ns(),
            )
            if isinstance(exc, EvidenceMismatch):
                failure["mismatch"] = exc.evidence
            c.save(archive / "failure.json", failure)
            c.save(run / "readiness.json", failure)
            summary = (
                "# Experimental Readiness Report\n\n**Verdict — NOT READY.**\n\n"
                f"The {stage} stage failed.\n\n{failure['reason']}\n\n"
                f"[Failure evidence](failures/{started}/failure.json)\n"
            )
            c.atomic(run / REPORT, lambda stream: stream.write(summary.encode()))
            raise
        finally:
            manifest.setdefault("attempts", []).append(
                dict(
                    stage=stage,
                    started_ns=started,
                    ended_ns=time.time_ns(),
                    command=sys.argv,
                    status=manifest.get("status"),
                )
            )
            c.save(path, manifest)


def _clear_root_markers(run, stage):
    from .archive import protect_destination

    partition = "development" if stage == "develop" else "protected"
    for root in roots(partition):
        marker = run / "work" / partition / (root["key"] + ".json")
        if marker.exists():
            protect_destination(marker)
            marker.unlink()


def _fixture_validation(run):
    command = [sys.executable, "-m", "pytest", "-q", str(HERE / "tests")]
    result = subprocess.run(
        command,
        cwd=HERE.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    path = run / "validation.json"
    c.save(
        path,
        {"passed": result.returncode == 0, "command": command, "output": result.stdout},
    )
    if result.returncode:
        raise ValueError(result.stdout[-4000:])
    return path


def validate(run, manifest, workers):
    inventory_path = evidence.write(run, "inventory", c.inventory())
    manifest.update(
        study="outbox-sufficiency-v1",
        pilot_revision="bb20ebe",
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip(),
        dirty=subprocess.check_output(
            ["git", "status", "--short"], cwd=HERE, text=True
        ).splitlines(),
        source_hashes=study_contract(),
        runtime=environment(),
        seed_scheme="SHA256 UTF-8 slash-joined 57012026/outbox-sufficiency-v1/condition/partition/index/phase/source, PCG64",
        phases=["prefix", "future"],
        stream_names=[
            "application",
            "routing",
            "business",
            "reference-arrivals",
            "reference-work",
            "extra-arrivals",
            "extra-work",
        ],
        root_index="decimal, unpadded, 0 through 399",
        fold_seed=c.FOLD_SEED,
        bootstrap_seed=c.BOOTSTRAP_SEED,
        clock="signed integer nanoseconds, admissions (application, reference, extra) before completion callbacks; sample after all events through tick",
        expected={
            "roots": 2400,
            "development": 1200,
            "protected": 1200,
            "primary_branches": 4800,
            "surprise_branches": 800,
        },
    )
    return [inventory_path, _fixture_validation(run)]


def roots(partition, run=None):
    inventory = c.inventory()
    if run is not None and list(evidence.rows(run, "inventory")) != inventory:
        raise ValueError("Persisted root inventory differs from the frozen design")
    return [row for row in inventory if row["partition"] == partition]


def _one_root(task):
    from .plant import simulate_root

    run, row, signature = task
    path = run / "work" / row["partition"] / (row["key"] + ".json")
    if path.exists():
        wrapper = c.read(path)
        if wrapper["signature"] != signature or wrapper["digest"] != c.digest(
            c.canonical(wrapper["data"])
        ):
            raise ValueError(f"Damaged or incompatible root completion marker: {path}")
        return wrapper["data"]
    data = simulate_root(row["condition"], row["partition"], row["index"])
    c.save(
        path, dict(signature=signature, digest=c.digest(c.canonical(data)), data=data)
    )
    return data


def _parallel(function, items, workers):
    if workers == 1:
        yield from map(function, items)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(function, items, chunksize=4, buffersize=workers * 3)


def collect(run, rows, signature, workers):
    part = run / rows[0]["partition"]
    jobs = ((run, row, signature) for row in rows)
    paths = c.publish_partition(run, rows, _parallel(_one_root, jobs, workers))
    from .plant import example_trace

    examples = (
        dict(
            key=row["key"],
            trace=example_trace(row["condition"], row["partition"], 0, "hold"),
        )
        for row in rows
        if row["index"] == 0
    )
    return [*paths, *evidence.write_examples(part, examples)]


def develop(run, manifest, workers):
    from .methods import fit_development

    rows = roots("development", run)
    outputs = collect(
        run, rows, c.digest(c.canonical(manifest["source_hashes"])), workers
    )
    fitted = fit_development(list(c.labelled_rows(run, rows)), c.FOLD_SEED)
    return [*outputs, *evidence.write_fit(run / "development", fitted)]


def get_fit(run):
    return evidence.read_fit(run / "development")


def lock(run, manifest, workers):
    fitted = get_fit(run)
    path = run / "predictor-lock.json"
    c.save(
        path,
        dict(
            contracts=manifest["source_hashes"],
            selected=fitted["selected"],
            development=manifest["completed"]["develop"],
            inventory=c.file_hash(run / "inventory.parquet"),
            fold_seed=c.FOLD_SEED,
            bootstrap_seed=c.BOOTSTRAP_SEED,
        ),
    )
    return [path]


def write_predictions(run, rows, fitted):
    from .methods import predict, predict_evaluator

    # Predictor receives only public measurements. Finish this table before reading the private table.
    directory = run / "protected"
    observations = evidence.histories(directory, rows)
    path = evidence.write_predictions(
        directory,
        "predictions",
        (
            dict(key=v["key"], predictions=predict(fitted["models"], v["history"]))
            for v in observations
        ),
    )

    def evaluator_rows():
        for record in evidence.private(directory, rows):
            truth = record["truth"]
            pred = predict_evaluator(
                fitted["evaluator_models"],
                record["condition"],
                truth["demand"],
                truth["unfinished"],
            )
            yield dict(key=record["key"], predictions=pred)

    private_path = evidence.write_predictions(
        directory, "evaluator-predictions", evaluator_rows()
    )
    return [path, private_path]


def score_rows(run, rows):
    public = evidence.predictions(run / "protected", "predictions", rows)
    evaluator = evidence.predictions(run / "protected", "evaluator-predictions", rows)
    return [
        dict(
            row, predictions=dict(pub["predictions"], evaluator=private["predictions"])
        )
        for row, pub, private in zip(
            c.labelled_rows(run, rows), public, evaluator, strict=True
        )
    ]


def evaluate(run, manifest, workers):
    from .scoring import score

    frozen_lock = c.file_hash(run / "predictor-lock.json")
    seal = run / "protected-access.json"
    if seal.exists() and c.read(seal)["lock"] != frozen_lock:
        raise ValueError("Protected roots already allocated to another lock")
    if not seal.exists():
        c.save(
            seal,
            dict(
                lock=frozen_lock,
                opened_ns=time.time_ns(),
                status="consumed for readiness study; reserve fresh roots for future evaluation",
            ),
        )
    rows = roots("protected", run)
    outputs = collect(run, rows, frozen_lock, workers)
    fitted = get_fit(run)
    predictions = write_predictions(run, rows, fitted)
    scores = score(score_rows(run, rows), fitted["selected"], c.BOOTSTRAP_SEED)
    path = run / "scores.json"
    c.save(path, scores)
    return [seal, *outputs, *predictions, path]


def _audit_job(task):
    from .audit import audit_root

    row, record = task
    return dict(
        key=row["key"],
        receipt=audit_root(row["condition"], row["partition"], row["index"], record),
    )


def audit(run, manifest, workers):
    return compute_audit(run, run, workers)


def compute_audit(run, output, workers):
    from .audit import audit_analysis

    rows = roots("protected", run)
    jobs = zip(rows, c.records(run, rows), strict=True)
    ledger = evidence.write_replays(
        output / "protected", _parallel(_audit_job, jobs, workers)
    )
    dev = list(c.labelled_rows(run, roots("development", run)))
    result = audit_analysis(
        dev,
        score_rows(run, rows),
        get_fit(run),
        c.read(run / "scores.json"),
        c.FOLD_SEED,
        c.BOOTSTRAP_SEED,
    )
    path = output / "independent-audit.json"
    c.save(
        path,
        dict(
            result,
            replay_ledger=c.file_hash(ledger),
            verified=True,
            scores=c.file_hash(run / "scores.json"),
            lock=c.file_hash(run / "predictor-lock.json"),
        ),
    )
    return [ledger, path]


def execute(stage, run, workers):
    from .report import report

    functions = dict(
        validate=validate,
        develop=develop,
        lock=lock,
        evaluate=evaluate,
        audit=audit,
        report=report,
    )
    run = Path(run).resolve()
    from .archive import historical, protect_destination

    protect_destination(run)
    if workers < 1:
        raise ValueError("Workers must be positive")
    if historical(run):
        raise ValueError(
            "Retained publication: use --output to regenerate audit or report in a separate directory"
        )
    with attempt(run, stage) as manifest:
        if (
            manifest.get("source_hashes")
            and manifest["source_hashes"] != study_contract()
        ):
            raise ValueError(
                "Study inputs changed. Preserve run and use its locked source."
            )
        if manifest.get("runtime") and manifest["runtime"] != environment():
            raise ValueError("Locked study runtime changed")
        for earlier in STAGES[: STAGES.index(stage)]:
            if earlier not in manifest["completed"]:
                raise ValueError(f"Required stage {earlier} has no completion receipt")
            verify_files(run, manifest["completed"][earlier])
        if stage in manifest["completed"]:
            verify_files(run, manifest["completed"][stage])
            return
        outputs = functions[stage](run, manifest, workers)
        manifest["completed"][stage] = relative_files(run, outputs)
        manifest["status"] = f"{stage} complete"
