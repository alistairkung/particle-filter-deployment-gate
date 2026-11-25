"""Inventory the two immutable study trees and the schemas of their legacy evidence."""

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager

from numpy.lib import format as npformat


ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "evidence_migration" / "receipts"
PILOT = "feasibility_pilot"
TELEMETRY = "telemetry_sufficiency"
REVISIONS = {
    PILOT: "bb20ebed72471f43443c873a11b2ac6e32fbbc00",
    TELEMETRY: "624541bae2405fe47a59a193a55482775a7e0d0a",
}
FROZEN = Path(PILOT, "results/frozen")
READY = Path(TELEMETRY, "results/v1")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    # JSON numbers retain their integer vs floating-point representation, including -0.0.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def exact(observed, expected, label):
    if canonical(observed) != canonical(expected):
        raise ValueError(f"{label}: stored fields, types, order or values differ")


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def tree(study):
    raw = subprocess.check_output(
        ["git", "ls-tree", "-rzl", REVISIONS[study], "--", study], cwd=ROOT
    )
    result = {}
    for line in raw.rstrip(b"\0").split(b"\0"):
        metadata, filename = line.split(b"\t", 1)
        mode, kind, blob, size = metadata.decode().split()
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(f"Unexpected Git object at {filename!r}")
        result[Path(filename.decode())] = (blob, int(size))
    return result


@contextmanager
def git_objects():
    """Read blob content by object ID, independently of the working tree and converter."""
    with subprocess.Popen(
        ["git", "cat-file", "--batch"], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    ) as proc:
        def read(entry):
            blob, size = entry
            proc.stdin.write((blob + "\n").encode())
            proc.stdin.flush()
            header = proc.stdout.readline().decode().split()
            if header != [blob, "blob", str(size)]:
                raise ValueError(f"Git object header mismatch for {blob}")
            data = proc.stdout.read(size)
            if len(data) != size or proc.stdout.read(1) != b"\n":
                raise ValueError(f"Truncated Git object {blob}")
            return data

        try:
            yield read
        finally:
            proc.stdin.close()


def _shape(val, fields, path="$"):
    kind = type(val).__name__
    fields[path].add(kind)
    if isinstance(val, dict):
        for k, child in val.items():
            key = "<root_key>" if re.fullmatch(r"[0-9a-f]{32}", k) else k
            _shape(child, fields, f"{path}.{key}")
    elif isinstance(val, list):
        for item in val:
            _shape(item, fields, path + "[]")


def read_csv(data):
    return csv.DictReader(io.StringIO(data.decode(), newline=""))


def _npz_shape(data, fields):
    sizes = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                if npformat.read_magic(stream) != (1, 0):
                    raise ValueError(f"Unexpected NPY header in {name}")
                shape, _, dtype = npformat.read_array_header_1_0(stream)
            arm, component = name.removesuffix(".npy").rsplit("_", 1)
            if len(shape) != 1 or name in sizes.setdefault(arm, {}):
                raise ValueError(f"Unexpected NPZ array {name}")
            sizes[arm][component] = shape[0]
            fields[name].add(dtype.str)
    for group in sizes.values():
        if set(group) != {"ns", "java", "censored"} or len(set(group.values())) != 1:
            raise ValueError("Invalid latency, route and censor inventory")
    return sum(v["ns"] for v in sizes.values())


def _describe(path, data, schemas):
    if "results" not in path.parts:
        return "", "", ""
    schema = re.sub(r"/[0-9]{4}(?=\.)", "/ROOT", str(path))
    schema = re.sub(r"/[0-9a-f]{32}(?=\.)", "/ROOT_KEY", schema)
    if path.suffix not in {".json", ".gz", ".csv", ".npz"}:
        return "", "", ""
    fields = schemas[schema]
    if path.suffix == ".npz":
        return schema, _npz_shape(data, fields), "branch writes"
    if path.suffix == ".csv":
        rows = read_csv(data)
        for name in rows.fieldnames:
            fields[name].add("csv text")
        return schema, sum(1 for _ in rows), "rows"
    if path.suffix == ".gz":
        rows = [json.loads(line) for line in gzip.decompress(data).splitlines()]
        unit = "JSON lines"
    else:
        value = json.loads(data)
        rows = value if isinstance(value, list) else [value]
        unit = "array items" if isinstance(value, list) else "JSON object"
    for row in rows:
        _shape(row, fields)
    return schema, len(rows), unit


def inventory(directory):
    directory.mkdir(parents=True, exist_ok=True)
    schemas = defaultdict(lambda: defaultdict(set))
    totals = {}
    with git_objects() as read, (directory / "source-files.csv").open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["study", "commit", "path", "bytes", "git_blob", "sha256",
                         "source_schema", "records", "record_unit"])
        for study, rev in REVISIONS.items():
            counts = Counter()
            for path, entry in tree(study).items():
                data = read(entry)
                schema, records, unit = _describe(path, data, schemas)
                writer.writerow([study, rev, path, len(data), entry[0], sha(data), schema, records, unit])
                counts["study_files"] += 1
                counts["study_bytes"] += len(data)
                if path.is_relative_to(FROZEN) or path.is_relative_to(READY):
                    counts["frozen_files"] += 1
                    counts["frozen_bytes"] += len(data)
            totals[study] = dict(counts)
    dump(directory / "source-schemas.json", {
        schema: {field: sorted(types) for field, types in fields.items()}
        for schema, fields in schemas.items()
    })
    dump(directory / "source-inventory.json", {
        "commits": REVISIONS, "counts": totals, "files_sha256": sha((directory / "source-files.csv").read_bytes()),
        "schemas_sha256": sha((directory / "source-schemas.json").read_bytes()),
    })


if __name__ == "__main__":
    inventory(RECEIPTS)
