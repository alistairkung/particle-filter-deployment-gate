"""Commands to validate, execute and report the fixed experiment."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from experiments.config import read_config
from experiments.storage import (
    budgets,
    compute_root,
    digest,
    load_json,
    read_root,
    save_json,
)

BASE = Path(__file__).resolve().parents[1]


def identity(config_path):
    physics = [
        "SPEC.md",
        "requirements.lock",
        "experiments/config.py",
        "experiments/storage.py",
        "experiments/observations.py",
        "experiments/runner.py",
        "experiments/validation.py",
    ]
    physics += sorted(
        str(p.relative_to(BASE)) for p in (BASE / "simulator").glob("*.py")
    )
    files = {f: digest(BASE / f) for f in physics}
    files["config"] = digest(config_path)
    signature = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return signature, files


def source_files():
    sources = [
        p
        for directory in ("experiments", "simulator", "tests")
        for p in (BASE / directory).glob("*.py")
    ]
    return {str(p.relative_to(BASE)): digest(p) for p in sorted(sources)}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=BASE, text=True).strip()


def new_manifest(config_path, c, signature, files):
    return dict(
        signature=signature,
        input_digests=files,
        source_files=source_files(),
        config=asdict(c),
        config_filename=config_path.name,
        python=sys.version,
        platform=platform.platform(),
        git_revision=git("rev-parse", "HEAD"),
        git_status=git("status", "--short", "--untracked-files=normal"),
        dependencies={
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        time_representation="integer nanoseconds; nearest nanosecond, positive Poisson gaps",
        ordering="external admissions, then DB/sink completions; stable insertion order at equal priority",
        seeds=dict(
            master=c.seed,
            generator="PCG64",
            entropy="SHA256 of slash-joined master/family/root/phase/source, big endian",
            scientific_root_ids={f: [0, n - 1] for f, n in budgets(c).items()},
            application_sources=["application", "routing", "business"],
            background_sources=[
                "reference-arrivals",
                "reference-work",
                "extra-arrivals",
                "extra-work",
            ],
        ),
        expected=budgets(c),
        completed={},
        timings_s={},
        status="new",
    )


@contextmanager
def exclusive_output(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def incomplete_report(path, out, reason):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Feasibility pilot\n\n**MODIFY / inconclusive.** "
        "The frozen evidence is incomplete or invalid.\n\n"
        + reason
        + "\n\nEvidence directory: "
        + str(out)
        + "\n"
    )


def execute_scientific(c, out, signature, workers, manifest):
    from experiments.storage import root_paths

    validation = load_json(out / "validation/result.json")
    if not validation["passed"] or validation["signature"] != signature:
        raise ValueError("Scientific run requires successful matching validation")
    tasks = [
        (c, out, f, i, signature)
        for f, count in budgets(c).items()
        for i in range(count)
    ]
    counts = {f: 0 for f in budgets(c)}
    try:
        with ProcessPoolExecutor(workers) as pool:
            for family, root in pool.map(compute_root, tasks, chunksize=2):
                counts[family] += 1
                if counts[family] % 100 == 0:
                    print(f"{family}: {counts[family]} complete", flush=True)
    finally:
        manifest["completed"] = {
            f: sum(root_paths(out, f, i)[0].exists() for i in range(n))
            for f, n in budgets(c).items()
        }


def write_window_tables(out, roots):
    import csv

    directory = out / "windows"
    directory.mkdir(exist_ok=True)
    for family, records in roots.items():
        with (directory / (family + ".csv")).open("w", newline="") as stream:
            writer = None
            for i, branches in enumerate(records):
                for arm, branch in branches.items():
                    for window in branch["windows"]:
                        row = dict(root=i, arm=arm)
                        row.update({k: v for k, v in window.items() if k != "db"})
                        for k, data in window["db"].items():
                            row.update(
                                {k + "_" + key: value for key, value in data.items()}
                            )
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)


def render(c, out, signature, args, manifest):
    from experiments.audit import audit
    from experiments.integrity import verify_scientific_controls
    from experiments.metrics import summarize
    from experiments.report import generate

    validation = load_json(out / "validation/result.json")
    if validation["signature"] != signature or not validation["passed"]:
        incomplete_report(
            args.output, out, "The analytical or physical prerequisite did not pass."
        )
        return False
    roots = {
        f: [read_root(out, c, f, i, signature) for i in range(n)]
        for f, n in budgets(c).items()
    }
    write_window_tables(out, roots)
    integrity = verify_scientific_controls(c, out, roots)
    save_json(out / "validation/workload-integrity.json", integrity)
    controls_passed = validation["passed"] and integrity["passed"]
    summary = summarize(c, roots, controls_passed)
    save_json(out / "summary/verdict.json", summary)
    reconstruction = audit(c, out, summary, controls_passed)
    save_json(out / "summary/independent-audit.json", reconstruction)
    generate(c, out, args.plots, args.output, validation, roots, summary)
    manifest["analysis_source_files"] = source_files()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--out", type=Path, required=True)
        sub.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    report = commands.add_parser("report")
    report.add_argument("--from", dest="out", type=Path, required=True)
    report.add_argument("--plots", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if not __debug__:
        parser.error("Validation requires enabled invariant checks")
    config_path = out / "pilot.toml" if args.command == "report" else args.config
    c = read_config(config_path)
    signature, files = identity(config_path)
    with exclusive_output(out):
        path = out / "manifest.json"
        manifest = (
            load_json(path)
            if path.exists()
            else new_manifest(config_path, c, signature, files)
        )
        if manifest["signature"] != signature:
            raise ValueError(
                "Source or config changed; preserve this attempt and use a fresh output directory"
            )
        if args.command != "report":
            shutil.copyfile(config_path, out / "pilot.toml")
        if args.command == "run" and not (out / "validation/result.json").exists():
            raise ValueError("Run the validation command first")
        start = time.monotonic()
        manifest["status"] = args.command + " in progress"
        save_json(path, manifest)
        try:
            if args.command == "validate":
                from experiments.validation import validate

                validation = validate(c, out, args.workers)
                validation["signature"] = signature
                save_json(out / "validation/result.json", validation)
                ok = validation["passed"]
            elif args.command == "run":
                execute_scientific(c, out, signature, args.workers, manifest)
                ok = True
            else:
                ok = render(c, out, signature, args, manifest)
            manifest["status"] = args.command + (" complete" if ok else " failed")
        except Exception as exc:
            manifest["status"] = args.command + " failed"
            message = f"{type(exc).__name__}: {exc}"
            save_json(
                out / "failures" / f"{time.time_ns()}.json",
                dict(stage=args.command, error=message),
            )
            if args.command == "report":
                incomplete_report(args.output, out, message)
            raise
        finally:
            manifest["timings_s"][args.command] = time.monotonic() - start
            save_json(path, manifest)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
