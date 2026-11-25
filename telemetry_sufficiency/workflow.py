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
from . import custody as c

HERE = Path(__file__).resolve().parent
STAGES = ("validate", "develop", "lock", "evaluate", "audit", "report")


def study_contract():
    source = sorted(HERE.glob("*.py")) + sorted(HERE.glob("tests/*.py"))
    source += [
        HERE / "SCIENTIFIC_SPEC.md",
        HERE / "IMPLEMENTATION_SPEC.md",
        PILOT / "configs/pilot.toml",
        PILOT / "requirements.lock",
    ]
    source += sorted((PILOT / "simulator").glob("*.py")) + sorted(
        (PILOT / "experiments").glob("*.py")
    )
    return {str(f.relative_to(HERE.parent)): c.file_hash(f) for f in source}


def environment():
    dependencies = [
        s.split("==")[0]
        for s in (PILOT / "requirements.lock").read_text().splitlines()
        if s
    ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {name: metadata.version(name) for name in dependencies},
    }


def relative_files(run, paths):
    return {str(p.relative_to(run)): c.file_hash(p) for p in paths}


def verify_files(run, mapping):
    for name, expected in mapping.items():
        if c.file_hash(run / name) != expected:
            raise ValueError(f"Evidence digest mismatch: {name}")


@contextmanager
def attempt(run, stage):
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".active").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = run / "manifest.json"
        manifest = c.read(path) if path.exists() else {"completed": {}, "attempts": []}
        started = time.time_ns()
        try:
            yield manifest
        except BaseException as exc:
            manifest["status"] = f"{stage} failed"
            for invalid in STAGES[STAGES.index(stage) :]:
                manifest["completed"].pop(invalid, None)
            failure = dict(
                stage=stage, error=f"{type(exc).__name__}: {exc}", at_ns=time.time_ns()
            )
            c.save(run / "failures" / f"{started}.json", failure)
            # A failed attempt cannot leave a published verdict at the normal entry point.
            report = run / "Experimental_Readiness_Report.md"
            if report.exists():
                report.rename(run / "failures" / f"{started}-prior-report.md")
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
    inventory_path = run / "inventory.json"
    c.save(inventory_path, c.inventory())
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


def roots(partition):
    return [row for row in c.inventory() if row["partition"] == partition]


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
    c.publish_partition(run, rows, _parallel(_one_root, jobs, workers))
    paths = [part / "observations.jsonl.gz", part / "evaluator.jsonl.gz"]
    from .plant import example_trace

    for row in rows:
        if row["index"] == 0:
            path = part / "examples" / (row["key"] + ".jsonl.gz")
            trace = example_trace(row["condition"], row["partition"], 0, "hold")
            c.write_table(path, [dict(key=row["key"], trace=trace)])
            paths.append(path)
    return paths


def develop(run, manifest, workers):
    from .methods import fit_development

    rows = roots("development")
    outputs = collect(
        run, rows, c.digest(c.canonical(manifest["source_hashes"])), workers
    )
    fitted = fit_development(list(c.labelled_rows(run, rows)), c.FOLD_SEED)
    path = run / "development" / "fit.jsonl.gz"
    c.write_table(path, [dict(key="development-fit", fit=fitted)])
    return [*outputs, path]


def get_fit(run):
    return next(
        c.read_table(run / "development" / "fit.jsonl.gz", [{"key": "development-fit"}])
    )["fit"]


def lock(run, manifest, workers):
    fitted = get_fit(run)
    path = run / "predictor-lock.json"
    c.save(
        path,
        dict(
            contracts=manifest["source_hashes"],
            selected=fitted["selected"],
            development=manifest["completed"]["develop"],
            inventory=c.file_hash(run / "inventory.json"),
            fold_seed=c.FOLD_SEED,
            bootstrap_seed=c.BOOTSTRAP_SEED,
        ),
    )
    return [path]


def write_predictions(run, rows, fitted):
    from .methods import predict, predict_evaluator

    # Predictor receives only public measurements. Finish this table before reading the private table.
    directory = run / "protected"
    path = directory / "predictions.jsonl.gz"
    observations = c.read_table(directory / "observations.jsonl.gz", rows)
    c.write_table(
        path,
        (
            dict(key=v["key"], predictions=predict(fitted["models"], v["history"]))
            for v in observations
        ),
    )
    private_path = directory / "evaluator-predictions.jsonl.gz"

    def evaluator_rows():
        for record in c.read_table(directory / "evaluator.jsonl.gz", rows):
            truth = record["truth"]
            pred = predict_evaluator(
                fitted["evaluator_models"],
                record["condition"],
                truth["demand"],
                truth["unfinished"],
            )
            yield dict(key=record["key"], predictions=pred)

    c.write_table(private_path, evaluator_rows())
    return [path, private_path]


def score_rows(run, rows):
    public = c.read_table(run / "protected" / "predictions.jsonl.gz", rows)
    evaluator = c.read_table(run / "protected" / "evaluator-predictions.jsonl.gz", rows)
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
    rows = roots("protected")
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
    from .audit import audit_analysis

    rows = roots("protected")
    jobs = zip(rows, c.records(run, rows), strict=True)
    ledger = run / "protected" / "replay-audit.jsonl.gz"
    c.write_table(ledger, _parallel(_audit_job, jobs, workers))
    dev = list(c.labelled_rows(run, roots("development")))
    result = audit_analysis(
        dev,
        score_rows(run, rows),
        get_fit(run),
        c.read(run / "scores.json"),
        c.FOLD_SEED,
        c.BOOTSTRAP_SEED,
    )
    path = run / "independent-audit.json"
    c.save(path, dict(result, replay_ledger=c.file_hash(ledger), verified=True))
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
    if run.is_relative_to(PILOT):
        raise ValueError("Study output path inside frozen pilot")
    if workers < 1:
        raise ValueError("Workers must be positive")
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
            verify_files(run, manifest["completed"][earlier])
        if stage in manifest["completed"]:
            verify_files(run, manifest["completed"][stage])
            return
        outputs = functions[stage](run, manifest, workers)
        manifest["completed"][stage] = relative_files(run, outputs)
        manifest["status"] = f"{stage} complete"
