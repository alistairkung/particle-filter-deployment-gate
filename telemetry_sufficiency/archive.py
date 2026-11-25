"""Read-only audit and report regeneration for the immutable 624541b publication."""

import gzip
import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory

from . import PILOT, custody as c, parquet_evidence as p

V1 = Path(__file__).resolve().parent / "results/v1"
MANIFEST = "95bcad7fcf1fd103389d9ae22e6b5316c074861058002cd44a9af75c27f41b8d"


def protect_destination(path):
    path = Path(path).resolve()
    if path.is_relative_to(PILOT):
        raise ValueError("Study output path inside frozen pilot")
    if path.is_relative_to(V1):
        raise ValueError("Study output path inside retained telemetry publication")


def historical(run):
    path = run / "manifest.json"
    return path.is_file() and c.file_hash(path) == MANIFEST


def legacy_digest(run, name):
    """Hash the original canonical transport bytes reconstructed from typed values."""
    if name == "inventory.json":
        return c.digest(c.canonical(list(p.legacy_records(run, name))) + b"\n")
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buffer, mode="wb", mtime=0) as stream:
        for record in p.legacy_records(run, name):
            stream.write(c.canonical(record) + b"\n")
    return hashlib.sha256(buffer.getbuffer()).hexdigest()


def verify_inputs(run):
    if c.file_hash(run / "manifest.json") != MANIFEST:
        raise ValueError("Archived publication manifest digest differs from 624541b")
    manifest = c.read(run / "manifest.json")
    for stage, files in manifest["completed"].items():
        for name, expected in files.items():
            if name.endswith(".jsonl.gz") or name == "inventory.json":
                observed = legacy_digest(run, name)
            elif stage == "report" and name not in (
                "verification-review.json",
                "pilot-reproduction.json",
            ):
                continue
            else:
                observed = c.file_hash(run / name)
            if observed != expected:
                raise ValueError(f"Archived publication mismatch: {name}")
    return manifest


def regenerate(stage, run, output, workers):
    output = output.resolve()
    protect_destination(output)
    if output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError("Choose a separate output directory outside retained evidence")
    if workers < 1:
        raise ValueError("Workers must be positive")
    if stage not in ("audit", "report"):
        raise ValueError("Read-only regeneration supports audit and report")
    manifest = verify_inputs(run)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix="sufficiency-replay-", dir=output.parent
    ) as temporary:
        stage_dir = Path(temporary)
        _regenerate(stage, run, stage_dir, workers, manifest)
        artifacts = [p for p in stage_dir.rglob("*") if p.is_file()]
        # A preexisting plot or subdirectory may alias a frozen publication.
        for artifact in artifacts:
            protect_destination(output / artifact.relative_to(stage_dir))
        for artifact in artifacts:
            destination = output / artifact.relative_to(stage_dir)
            c.atomic(destination, lambda stream: stream.write(artifact.read_bytes()))


def _regenerate(stage, run, output, workers, manifest):
    from .workflow import compute_audit
    from .report import report

    if stage == "audit":
        ledger, summary = compute_audit(run, output, workers)
        reference = run / "protected"
        actual = p.rows(ledger.parent, "replay-audit")
        expected = p.rows(reference, "replay-audit")
        if any(a != b for a, b in zip(actual, expected, strict=True)):
            raise ValueError("Replayed protected ledger differs from retained values")
        previous = c.read(run / "independent-audit.json")
        previous["replay_ledger"] = c.file_hash(ledger)
        if c.read(summary) != previous:
            raise ValueError(
                "Independent reconstruction changed the scientific receipt"
            )
    else:
        paths = report(run, manifest, workers, output=output)
        for path in paths:
            name = str(path.relative_to(output))
            if c.file_hash(path) != manifest["completed"]["report"][name]:
                raise ValueError(f"Regenerated publication differs: {name}")
        # Keep the report's relative links readable in the separate output directory.
        for name in (
            "scores.json",
            "independent-audit.json",
            "pilot-reproduction.json",
            "predictor-lock.json",
            "manifest.json",
        ):
            c.atomic(
                output / name, lambda stream: stream.write((run / name).read_bytes())
            )
