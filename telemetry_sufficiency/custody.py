"""Inventory and evidence custody. Transport keys never enter the predictors."""

import os
import tempfile
import hashlib
import json
from pathlib import Path

from . import CONDITIONS

PARTITIONS = ("development", "protected")
FOLD_SEED = "outbox-sufficiency-v1/folds"
BOOTSTRAP_SEED = 5701202601


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory():
    return [
        dict(
            condition=c,
            partition=p,
            index=i,
            key=digest(f"envelope-v1/{c}/{p}/{i}".encode())[:32],
        )
        for p in PARTITIONS
        for c in CONDITIONS
        for i in range(400)
    ]


def branch_names(row):
    if row["condition"] == "reference" and row["partition"] == "protected":
        return {"hold", "promote", "surprise_hold", "surprise_promote"}
    return {"hold", "promote"}


def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(path)


def save(path, data):
    atomic(path, lambda stream: stream.write(canonical(data) + b"\n"))


def read(path):
    return json.loads(Path(path).read_bytes())


def _check_record(row, record):
    if any(row[name] != record[name] for name in ("condition", "partition", "index")):
        raise ValueError(f"Root identity mismatch: {row['key']}")
    if set(record["branches"]) != branch_names(row):
        raise ValueError(f"Branch inventory mismatch: {row['key']}")


def publish_partition(run, rows, raw):
    from .parquet_evidence import write_partition

    return write_partition(run / rows[0]["partition"], rows, raw)


def records(run, rows):
    from . import parquet_evidence as p

    part = run / rows[0]["partition"]
    public = p.histories(part, rows)
    private = p.private(part, rows)
    for row, observations, truth in zip(rows, public, private, strict=True):
        truth.pop("key")
        record = dict(truth, history=observations["history"])
        _check_record(row, record)
        yield record


def labelled_rows(run, rows):
    for row, record in zip(rows, records(run, rows), strict=True):
        truth = record["truth"]
        outcomes = {a: b["breach"] for a, b in record["branches"].items()}
        yield dict(
            key=row["key"],
            partition=row["partition"],
            condition=row["condition"],
            history=record["history"],
            demand=truth["demand"],
            unfinished=truth["unfinished"],
            **outcomes,
        )
