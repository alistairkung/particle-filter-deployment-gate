"""Inventory and evidence custody. Transport keys never enter the predictors."""

import gzip
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


def write_table(path, rows):
    def write(stream):
        with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as output:
            for row in rows:
                output.write(canonical(row) + b"\n")

    atomic(path, write)


def read_table(path, expected):
    with gzip.open(path, "rb") as stream:
        for line, root in zip(stream, expected, strict=True):
            record = json.loads(line)
            if record["key"] != root["key"]:
                raise ValueError(f"Wrong root in {path}: expected {root['key']}")
            yield record


def _check_record(row, record):
    if any(row[name] != record[name] for name in ("condition", "partition", "index")):
        raise ValueError(f"Root identity mismatch: {row['key']}")
    if set(record["branches"]) != branch_names(row):
        raise ValueError(f"Branch inventory mismatch: {row['key']}")


def publish_partition(run, rows, raw):
    part = run / rows[0]["partition"]
    histories, private = [], []
    for row, record in zip(rows, raw, strict=True):
        _check_record(row, record)
        histories.append(dict(key=row["key"], history=record["history"]))
        private.append(
            dict(key=row["key"], **{k: v for k, v in record.items() if k != "history"})
        )
    write_table(part / "observations.jsonl.gz", histories)
    write_table(part / "evaluator.jsonl.gz", private)


def records(run, rows):
    part = run / rows[0]["partition"]
    public = read_table(part / "observations.jsonl.gz", rows)
    private = read_table(part / "evaluator.jsonl.gz", rows)
    for row, observations, truth in zip(rows, public, private, strict=True):
        if set(observations) != {"key", "history"}:
            raise ValueError("Unexpected observation transport field")
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
