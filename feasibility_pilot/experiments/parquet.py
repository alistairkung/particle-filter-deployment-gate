"""Typed pilot observations. One sample row group holds one independent root."""

import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _fields(names, dtype, nullable=False):
    return [pa.field(name, dtype, nullable=nullable) for name in names.split()]


KEY = [
    *_fields("family", pa.string()),
    *_fields("root", pa.int32()),
    *_fields("arm", pa.string()),
]
BRANCH = pa.schema(
    [
        *KEY,
        *_fields("signature sample_digest sample_values_sha256 cut", pa.string()),
        *_fields("trace", pa.string(), nullable=True),
        *_fields("breach censored scored java_arrivals", pa.int64()),
        *_fields("expected_load share query_rate", pa.float64()),
    ]
)
VISIT = pa.struct(
    [
        *_fields("visits completed waits still_waiting", pa.int64()),
        *_fields("busy_s offered_s wait_sum_ms", pa.float64()),
        *_fields("wait_p95_ms", pa.float64(), nullable=True),
    ]
)
WINDOW = pa.schema(
    [
        *KEY,
        *_fields(
            "window eligible misses commits marks pending pending_start queue_start queue",
            pa.int64(),
        ),
        *_fields("breach empty", pa.bool_()),
        *_fields(
            "age_s age_start_s throughput work_start_s work_s realised_cohort_work_s "
            "relay_owed_s deferred_start_s deferred_s utilisation",
            pa.float64(),
        ),
        *_fields("delay_p50_s delay_p95_s", pa.float64(), nullable=True),
        pa.field(
            "db", pa.struct(_fields("claim mark query write", VISIT)), nullable=False
        ),
    ]
)
SAMPLE = pa.schema(
    [
        *KEY,
        *_fields("sample_index", pa.int32()),
        *_fields("elapsed_ns", pa.int64()),
        *_fields("java censored", pa.bool_()),
    ]
)


def _sample_hash(ns, java, censored):
    checksum = hashlib.sha256()
    for array in (ns.astype("<i8", copy=False), java, censored):
        checksum.update(array.tobytes())
    return checksum.hexdigest()


def _sample_table(key, arrays):
    ns, java, censored = arrays
    n = len(ns)
    if (
        ns.dtype != np.int64
        or java.dtype != bool
        or censored.dtype != bool
        or any(a.shape != (n,) for a in arrays)
    ):
        raise ValueError(f"Invalid sample dtypes or length for {key}")
    return pa.Table.from_arrays(
        [
            *(pa.repeat(pa.scalar(key[f.name], type=f.type), n) for f in KEY),
            np.arange(n, dtype=np.int32),
            ns,
            java,
            censored,
        ],
        schema=SAMPLE,
    )


def _encode_root(family, root_id, record, samples):
    if (record["family"], record["root"]) != (family, root_id):
        raise ValueError(f"Root identity mismatch for {family}/{root_id}")
    branches = record["branches"]
    expected = {
        arm + suffix for arm in branches for suffix in ("_ns", "_java", "_censored")
    }
    if set(samples) != expected:
        raise ValueError(f"Incorrect sample vector for {family}/{root_id}")
    root_samples, branch_rows, window_rows = [], [], []
    for arm, branch in sorted(branches.items()):
        key = dict(family=family, root=root_id, arm=arm)
        arrays = [samples[arm + suffix] for suffix in ("_ns", "_java", "_censored")]
        if len(arrays[0]) != branch["scored"]:
            raise ValueError(f"Scored cohort mismatch for {key}")
        root_samples.append(_sample_table(key, arrays))
        row = {k: v for k, v in branch.items() if k != "windows"}
        branch_rows.append(
            dict(
                **key,
                **row,
                signature=record["signature"],
                sample_digest=record["sample_digest"],
                sample_values_sha256=_sample_hash(*arrays),
            )
        )
        window_rows.extend({**key, **w} for w in branch["windows"])
    return branch_rows, window_rows, pa.concat_tables(root_samples)


def publish_family(out, family, records):
    """Install a complete partition atomically. Each input is a root record and its arrays."""
    parent = Path(out) / "evidence"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out, prefix=".parquet-stage-") as name:
        stage = Path(name) / family
        stage.mkdir()
        branch_rows, window_rows = [], []
        with pq.ParquetWriter(
            stage / "samples.parquet",
            SAMPLE,
            compression="zstd",
            use_dictionary=["family", "arm"],
            column_encoding={
                "sample_index": "DELTA_BINARY_PACKED",
                "elapsed_ns": "DELTA_BINARY_PACKED",
            },
            write_page_checksum=True,
        ) as writer:
            for root_id, (record, samples) in enumerate(records):
                branches, windows, table = _encode_root(
                    family, root_id, record, samples
                )
                branch_rows.extend(branches)
                window_rows.extend(windows)
                writer.write_table(table, row_group_size=max(len(table), 1))
        for filename, rows, schema in (
            ("branches", branch_rows, BRANCH),
            ("windows", window_rows, WINDOW),
        ):
            pq.write_table(
                pa.Table.from_pylist(rows, schema=schema),
                stage / f"{filename}.parquet",
                compression="zstd",
                write_page_checksum=True,
            )
        partition = parent / family
        if partition.exists():
            # A retried validation must produce the same physical observations.
            for filename in ("branches.parquet", "windows.parquet", "samples.parquet"):
                with (
                    (stage / filename).open("rb") as new,
                    (partition / filename).open("rb") as saved,
                ):
                    if (
                        hashlib.file_digest(new, "sha256").digest()
                        != hashlib.file_digest(saved, "sha256").digest()
                    ):
                        raise ValueError(
                            f"Published partition differs at {partition / filename}"
                        )
        else:
            stage.rename(partition)


def _open(out, family, name, schema):
    path = Path(out) / "evidence" / family / f"{name}.parquet"
    file = pq.ParquetFile(path, page_checksum_verification=True)
    if file.schema_arrow != schema:
        raise ValueError(f"Invalid {name} schema at {path}")
    return file


def read_records(out, family, arm_names, count, signature, window_count):
    """Join branch and window tables by key, checking the full expected root/arm/window grid."""
    branches = [{} for _ in range(count)]
    for row in _open(out, family, "branches", BRANCH).read().to_pylist():
        root, arm = row.pop("root"), row.pop("arm")
        if (
            row.pop("family") != family
            or not 0 <= root < count
            or arm not in arm_names
            or arm in branches[root]
            or row.pop("signature") != signature
        ):
            raise ValueError(
                f"Invalid branch identity or provenance for {family}/{root}/{arm}"
            )
        if row["trace"] is None:
            del row["trace"]
        branches[root][arm] = dict(**row, windows=[])
    if any(set(group) != set(arm_names) for group in branches):
        raise ValueError(f"Incomplete branch vector for {family}")
    for row in _open(out, family, "windows", WINDOW).read().to_pylist():
        root, arm = row.pop("root"), row.pop("arm")
        if row.pop("family") != family or not 0 <= root < count or arm not in arm_names:
            raise ValueError(f"Unexpected window in {family}/{root}/{arm}")
        windows = branches[root][arm]["windows"]
        if row["window"] != 1 + len(windows):
            raise ValueError(
                f"Missing, duplicated or unordered window in {family}/{root}/{arm}"
            )
        windows.append(row)
    if any(
        len(b["windows"]) != window_count for group in branches for b in group.values()
    ):
        raise ValueError(f"Incomplete window grid for {family}")
    return branches


def iter_samples(out, family, roots):
    """Yield the exact scored sequences in root order, including roots with empty cohorts."""
    file = _open(out, family, "samples", SAMPLE)
    if file.num_row_groups != len(roots):
        raise ValueError(f"Incorrect sample root count for {family}")
    for root, branches in enumerate(roots):
        table = file.read_row_group(root)
        samples, offset = {}, 0
        for arm, branch in sorted(branches.items()):
            count = branch["scored"]
            group = table.slice(offset, count)
            if len(group) != count or any(
                count and not pc.all(pc.equal(group[key], value)).as_py()
                for key, value in (("family", family), ("root", root), ("arm", arm))
            ):
                raise ValueError(f"Invalid sample key for {family}/{root}/{arm}")
            if not np.array_equal(group["sample_index"].to_numpy(), np.arange(count)):
                raise ValueError(f"Invalid sample order for {family}/{root}/{arm}")
            arrays = [
                group[field].to_numpy() for field in ("elapsed_ns", "java", "censored")
            ]
            if _sample_hash(*arrays) != branch["sample_values_sha256"]:
                raise ValueError(f"Sample digest mismatch for {family}/{root}/{arm}")
            samples.update(
                {
                    arm + suffix: a
                    for suffix, a in zip(
                        ("_ns", "_java", "_censored"), arrays, strict=True
                    )
                }
            )
            offset += count
        if offset != len(table):
            raise ValueError(f"Unexpected sample rows for {family}/{root}")
        yield samples
