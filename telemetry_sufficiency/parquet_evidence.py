"""Versioned, typed evidence tables. The only public sensor table is observations."""

from itertools import groupby
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import custody as c
from .observation import parse_history

I, F, S, B = pa.int64(), pa.float64(), pa.string(), pa.bool_()
VF = pa.list_(F)
READINGS = dict(
    interval_start_ns=I,
    interval_end_ns=I,
    available_at_ns=I,
    pending_count=I,
    oldest_pending_age_s=F,
    publication_throughput_per_s=F,
    completed_delay_median_s=F,
)
IDENTITY = dict(key=S, partition=S, condition=S, index=I)
TRUTH = dict(
    demand=F,
    unfinished=F,
    query_work_ns=I,
    duration_ns=I,
    queued_ns=I,
    active_residual_ns=I,
)
DIGESTS = dict(physical_digest=S, prefix_digest=S, history_digest=S, state_digest=S)
ACTIONS = dict(hold=F, promote=F, increment=F)
PUBLIC_PRED = dict(demand=F, unfinished=F, condition=VF, **ACTIONS)
LEVELS = ("prior", "age", "current", "history")
EXAMPLE_META = dict(branch=S, phase=S, ack_due=I, cut_digest=S, horizon_ns=I)
WRITE_EVENTS = dict(
    arrival=I, demand=I, eligible=B, java=B, commit=I, claim=I, ack=I, mark=I
)
VISIT = dict(kind=I, identity=I, queued_ns=I, start_ns=I, end_ns=I)
TABLES = {
    "inventory": IDENTITY,
    "history-index": dict(key=S, readings=I),
    "observations": dict(key=S, **READINGS),
    "targets": IDENTITY | TRUTH | DIGESTS,
    "branches": dict(
        key=S, branch=S, breach=I, admissions_digest=S, cut_digest=S, future_digest=S
    ),
    "windows": dict(key=S, branch=S, window_index=I, eligible=I, misses=I),
    "predictions": dict(key=S, level=S, **PUBLIC_PRED),
    "evaluator-predictions": dict(key=S, **ACTIONS),
    "replay-audit": IDENTITY | dict(branches=I, admissions=I, replay_digest=S),
    "oof": dict(
        key=S,
        fold=I,
        training_hash=S,
        condition=S,
        demand=F,
        unfinished=F,
        hold=I,
        promote=I,
    ),
    "oof-predictions": dict(key=S, level=S, **PUBLIC_PRED),
    "oof-evaluator-predictions": dict(key=S, **ACTIONS),
    "models": dict(
        fold=I,
        level=S,
        target=S,
        features=pa.list_(S),
        medians=VF,
        means=VF,
        scales=VF,
        kind=S,
        converged=B,
        iterations=I,
        intercept=VF,
        coef=pa.list_(VF),
    ),
    "example-roots": IDENTITY | EXAMPLE_META,
    "example-tapes": dict(
        key=S,
        phase=S,
        ordinal=I,
        tick_ns=I,
        source=S,
        identity=I,
        routing_uniform=F,
        business_draw=F,
        query_work_ns=I,
    ),
    "example-visits": dict(key=S, ordinal=I, **VISIT),
    "example-writes": dict(key=S, ordinal=I, identity=I, **WRITE_EVENTS),
}
OPTIONAL = {
    "observations": {"completed_delay_median_s"},
    "example-roots": {"ack_due"},
    "example-tapes": {"routing_uniform", "business_draw", "query_work_ns"},
    "example-writes": {"commit", "claim", "ack", "mark"},
}
PRIVATE_TABLES = ("targets", "branches", "windows")
PUBLIC_TABLES = ("history-index", "observations")
EXAMPLES = ("example-roots", "example-tapes", "example-visits", "example-writes")


def schema(table, partition):
    fields = [
        pa.field(k, t, nullable=k in OPTIONAL.get(table, ()))
        for k, t in TABLES[table].items()
    ]
    return pa.schema(
        fields,
        metadata={
            b"study": b"outbox-sufficiency-v1",
            b"schema_version": b"1",
            b"table": table.encode(),
            b"partition": partition.encode(),
            b"configuration": b"feasibility_pilot/configs/pilot.toml",
        },
    )


def _schema(directory, table):
    partition = "all" if table == "inventory" else directory.name
    if partition not in (*c.PARTITIONS, "all"):
        raise ValueError(f"Invalid partition directory: {directory}")
    return schema(table, partition)


def write(directory, table, rows):
    path = directory / (table + ".parquet")
    spec = _schema(directory, table)
    names = set(spec.names)
    required = names - OPTIONAL.get(table, set())
    rows = list(rows)
    for row in rows:
        if row.keys() != names or any(row[k] is None for k in required):
            raise ValueError(f"Incomplete or unknown columns for {table}: {row.keys()}")
        for name, kind in TABLES[table].items():
            if kind == I and row[name] is not None and type(row[name]) is not int:
                raise ValueError(f"Expected exact integer {table}.{name}")
    batch = pa.Table.from_pylist(rows, schema=spec)
    c.atomic(
        path,
        lambda stream: pq.write_table(
            batch, stream, compression="zstd", compression_level=9, row_group_size=16384
        ),
    )
    return path


def rows(directory, table):
    source = pq.ParquetFile(directory / (table + ".parquet"))
    spec = _schema(directory, table)
    if (
        not source.schema_arrow.equals(spec)
        or source.schema_arrow.metadata != spec.metadata
    ):
        raise ValueError(f"Wrong table schema or partition: {directory}/{table}")
    for batch in source.iter_batches(batch_size=16384):
        for field, column in zip(spec, batch.columns, strict=True):
            if not field.nullable and column.null_count:
                raise ValueError(f"Missing required {table}.{field.name}")
        yield from batch.to_pylist()


def checked(actual, expected):
    for record, root in zip(actual, expected, strict=True):
        if record["key"] != root["key"]:
            raise ValueError(f"Wrong root: expected {root['key']}, got {record['key']}")
        yield record


def _fields(record, names):
    return {k: record[k] for k in names}


def _finish(directory, tables):
    return [write(directory, name, entries) for name, entries in tables.items()]


def write_partition(directory, inventory, records):
    tables = {name: [] for name in PUBLIC_TABLES + PRIVATE_TABLES}
    for root, record in zip(inventory, records, strict=True):
        if root["partition"] != directory.name:
            raise ValueError("Cross-partition root")
        c._check_record(root, record)
        key = root["key"]
        history = record["history"]
        parse_history(history)
        tables["history-index"].append(dict(key=key, readings=len(history)))
        tables["observations"].extend(dict(key=key, **r) for r in history)
        tables["targets"].append(dict(root, **record["truth"], **record["evidence"]))
        for branch, endpoint in sorted(record["branches"].items()):
            values = dict(endpoint)
            windows = values.pop("windows")
            tables["branches"].append(dict(key=key, branch=branch, **values))
            if len(windows) != 4:
                raise ValueError("Expected four frozen windows")
            tables["windows"].extend(
                dict(key=key, branch=branch, window_index=i, **v)
                for i, v in enumerate(windows)
            )
    return _finish(directory, tables)


def histories(directory, inventory):
    indices = checked(rows(directory, "history-index"), inventory)
    stream = iter(rows(directory, "observations"))
    for index in indices:
        history = []
        if index["readings"] < 0:
            raise ValueError("Negative reading count")
        for _ in range(index["readings"]):
            reading = next(stream, None)
            if reading is None or reading.pop("key") != index["key"]:
                raise ValueError("Missing reading or wrong observation root")
            history.append(reading)
        parse_history(history)
        yield dict(key=index["key"], history=history)
    if next(stream, None) is not None:
        raise ValueError("Orphan observation")


def private(directory, inventory):
    endpoints = iter(rows(directory, "branches"))
    windows = iter(rows(directory, "windows"))
    for target, root in zip(
        checked(rows(directory, "targets"), inventory), inventory, strict=True
    ):
        record = _fields(target, IDENTITY)
        record.update(
            truth=_fields(target, TRUTH), evidence=_fields(target, DIGESTS), branches={}
        )
        for branch in sorted(c.branch_names(root)):
            endpoint = next(endpoints, None)
            if endpoint is None or (endpoint.pop("key"), endpoint.pop("branch")) != (
                root["key"],
                branch,
            ):
                raise ValueError("Branch inventory mismatch")
            endpoint["windows"] = []
            for index in range(4):
                row = next(windows, None)
                if row is None or (
                    row.pop("key"),
                    row.pop("branch"),
                    row.pop("window_index"),
                ) != (root["key"], branch, index):
                    raise ValueError("Window inventory mismatch")
                endpoint["windows"].append(row)
            record["branches"][branch] = endpoint
        c._check_record(root, record)
        yield record
    if next(endpoints, None) is not None or next(windows, None) is not None:
        raise ValueError("Orphan endpoint or window")


def write_predictions(directory, table, records):
    def unpack():
        for record in records:
            key, values = record["key"], record["predictions"]
            if "evaluator" in table:
                yield dict(key=key, **values)
            else:
                if values.keys() != set(LEVELS):
                    raise ValueError("Incomplete public model inventory")
                for level in LEVELS:
                    yield dict(key=key, level=level, **values[level])

    return write(directory, table, unpack())


def predictions(directory, table, inventory):
    def unpack():
        for key, values in groupby(rows(directory, table), key=lambda v: v.pop("key")):
            if "evaluator" in table:
                (pred,) = values
            else:
                pred = {}
                for level, row in zip(LEVELS, values, strict=True):
                    if row.pop("level") != level:
                        raise ValueError("Wrong prediction level")
                    pred[level] = row
            yield dict(key=key, predictions=pred)

    return checked(unpack(), inventory)


def write_fit(directory, fit):
    selection = directory / "selection.json"
    c.save(selection, _fields(fit, ("selected", "selection_scores", "solver")))

    def models():
        bundles = [
            (-1, fit),
            *((int(f), v) for f, v in sorted(fit["fold_models"].items())),
        ]
        for fold, bundle in bundles:
            for level, model in [
                *sorted(bundle["models"].items()),
                ("evaluator", bundle["evaluator_models"]),
            ]:
                for target, fit_values in sorted(model["fits"].items()):
                    values = dict(fit_values)
                    if target != "condition":
                        values["coef"] = [[v] for v in values["coef"]]
                        values["intercept"] = [values["intercept"]]
                    yield dict(
                        fold=fold,
                        level=level,
                        target=target,
                        **values,
                        **_fields(model, ("features", "medians", "means", "scales")),
                    )

    oof = fit["oof"]
    if {r["key"]: r["fold"] for r in oof} != fit["folds"] or len(oof) != len(
        fit["folds"]
    ):
        raise ValueError("OOF root and fold inventory differ")
    output = [
        selection,
        write(directory, "models", models()),
        write(
            directory,
            "oof",
            (
                dict(_fields(r, ("key", "fold", "training_hash")), **r["targets"])
                for r in oof
            ),
        ),
    ]
    for private_fit, table in [
        (False, "oof-predictions"),
        (True, "oof-evaluator-predictions"),
    ]:
        output.append(
            write_predictions(
                directory,
                table,
                (
                    dict(
                        key=r["key"],
                        predictions=r["predictions"]["evaluator"]
                        if private_fit
                        else _fields(r["predictions"], LEVELS),
                    )
                    for r in oof
                ),
            )
        )
    return output


def read_fit(directory):
    fit = c.read(directory / "selection.json")
    fit.update(fold_models={}, folds={}, oof=[])
    for row in rows(directory, "models"):
        fold, level, target = row.pop("fold"), row.pop("level"), row.pop("target")
        bundle = fit if fold == -1 else fit["fold_models"].setdefault(str(fold), {})
        meta = {k: row.pop(k) for k in ("features", "medians", "means", "scales")}
        if level == "evaluator":
            model = bundle.setdefault("evaluator_models", dict(meta, fits={}))
        else:
            model = bundle.setdefault("models", {}).setdefault(
                level, dict(meta, fits={})
            )
        if _fields(model, meta) != meta or target in model["fits"]:
            raise ValueError("Inconsistent model metadata")
        if target != "condition":
            (row["intercept"],) = row["intercept"]
            row["coef"] = [v for (v,) in row["coef"]]
        model["fits"][target] = row
    oof = list(rows(directory, "oof"))
    public = predictions(directory, "oof-predictions", oof)
    evaluator = predictions(directory, "oof-evaluator-predictions", oof)
    for row, pred, ev in zip(oof, public, evaluator, strict=True):
        key = row["key"]
        if key in fit["folds"]:
            raise ValueError("Duplicate OOF root")
        fit["folds"][key] = row["fold"]
        fit["oof"].append(
            dict(
                _fields(row, ("key", "fold", "training_hash")),
                targets=_fields(
                    row, ("condition", "demand", "unfinished", "hold", "promote")
                ),
                predictions=dict(pred["predictions"], evaluator=ev["predictions"]),
            )
        )
    return fit


def write_examples(directory, records):
    tables = {name: [] for name in EXAMPLES}
    for record in records:
        key, trace = record["key"], dict(record["trace"])
        for phase in ("prefix", "future"):
            for ordinal, (tick, source, identity, *draws) in enumerate(
                trace.pop(phase)
            ):
                uniform, business, work = None, None, None
                if source == "write":
                    uniform, business = draws
                elif source in ("reference", "extra"):
                    (work,) = draws
                else:
                    raise ValueError("Unknown input source")
                tables["example-tapes"].append(
                    dict(
                        key=key,
                        phase=phase,
                        ordinal=ordinal,
                        tick_ns=tick,
                        source=source,
                        identity=identity,
                        routing_uniform=uniform,
                        business_draw=business,
                        query_work_ns=work,
                    )
                )
        for ordinal, visit in enumerate(trace.pop("visits")):
            tables["example-visits"].append(
                dict(key=key, ordinal=ordinal, **dict(zip(VISIT, visit, strict=True)))
            )
        for ordinal, (identity, write_events) in enumerate(trace.pop("writes")):
            tables["example-writes"].append(
                dict(key=key, ordinal=ordinal, identity=identity, **write_events)
            )
        tables["example-roots"].append(dict(key=key, **trace))
    return _finish(directory, tables)


def examples(directory):
    children = {
        name: groupby(rows(directory, name), key=lambda row: row.pop("key"))
        for name in EXAMPLES[1:]
    }
    for root in rows(directory, "example-roots"):
        key = root.pop("key")
        trace = dict(root, prefix=[], future=[], visits=[], writes=[])
        for name, groups in children.items():
            group = next(groups, None)
            if group is None or group[0] != key:
                raise ValueError("Example root has missing or swapped children")
            for row in group[1]:
                if name == "example-tapes":
                    source = row["source"]
                    if source == "write" and row["query_work_ns"] is None:
                        draw = [row["routing_uniform"], row["business_draw"]]
                    elif (
                        source in ("reference", "extra")
                        and row["routing_uniform"] is None
                        and row["business_draw"] is None
                    ):
                        draw = [row["query_work_ns"]]
                    else:
                        raise ValueError("Invalid tagged tape row")
                    if None in draw:
                        raise ValueError("Missing tape draw")
                    output = trace[row["phase"]]
                    value = [row["tick_ns"], source, row["identity"], *draw]
                elif name == "example-visits":
                    output = trace["visits"]
                    value = [row[k] for k in VISIT]
                else:
                    output = trace["writes"]
                    value = [row["identity"], _fields(row, WRITE_EVENTS)]
                if row["ordinal"] != len(output):
                    raise ValueError("Wrong example event order")
                output.append(value)
        yield dict(key=key, trace=trace)
    if any(next(groups, None) is not None for groups in children.values()):
        raise ValueError("Orphan example trace")


def write_replays(directory, records):
    return write(
        directory, "replay-audit", (dict(key=v["key"], **v["receipt"]) for v in records)
    )


def legacy_records(run, path):
    """Reassemble one historical logical stream for independent before/after verification."""
    path = Path(path)
    if path == Path("inventory.json"):
        yield from rows(run, "inventory")
        return
    partition = path.parts[0]
    directory = run / partition
    inventory = [r for r in rows(run, "inventory") if r["partition"] == partition]
    table = path.name.removesuffix(".jsonl.gz")
    if path.parent.name == "examples":
        selected = [r for r in examples(directory) if r["key"] == table]
        (record,) = selected
        yield record
    elif table == "fit":
        yield dict(key="development-fit", fit=read_fit(directory))
    elif table == "observations":
        yield from histories(directory, inventory)
    elif table == "evaluator":
        yield from private(directory, inventory)
    elif table in ("predictions", "evaluator-predictions"):
        yield from predictions(directory, table, inventory)
    elif table == "replay-audit":
        for row in checked(rows(directory, table), inventory):
            key = row.pop("key")
            yield dict(key=key, receipt=row)
    else:
        raise ValueError(f"Unexpected historical artifact {path}")
