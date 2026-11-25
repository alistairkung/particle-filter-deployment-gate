"""Generate the report and figures from saved scientific evidence."""

from pathlib import Path
import csv
import json
import os

import numpy as np

from experiments.config import MS
from experiments.observations import latency_quantiles


def latency_distribution(elapsed_ns, censored):
    """Summarise the complete scored cohort, including unresolved writes."""
    quantiles = latency_quantiles(elapsed_ns / MS, censored)
    return dict(
        writes=len(elapsed_ns),
        censored=int(censored.sum()),
        **dict(zip(("p50_ms", "p95_ms", "p99_ms"), quantiles)),
    )


def control_findings(summary):
    publisher = summary["publisher"]
    interpretation = {
        "consequential": "A consequential migration effect was demonstrated in at least one predeclared setting.",
        "excluded": "Both upper bounds rule out a consequential effect for this intervention and endpoint.",
        "inconclusive": "The migration bounds leave the practical effect unresolved.",
    }
    return {
        "migration": interpretation[summary["migration_finding"]],
        "publisher": f"Write health was {publisher['health']}.",
        "outbox": "Meaningful outbox deterioration was demonstrated."
        if publisher["age"]["passed"]
        else "The paired oldest-age criterion did not clear its threshold.",
    }


def _mean(values):
    present = [x for x in values if x is not None]
    return float(np.mean(present)) if present else None


def _pooled_counts(windows):
    eligible = sum(w["eligible"] for w in windows)
    misses = sum(w["misses"] for w in windows)
    return {
        "eligible": eligible,
        "misses": misses,
        "miss_fraction": misses / eligible if eligible else None,
    }


def _db_summary(windows, roots, expected, horizon):
    """Visit-weighted observed waits and comparable wall-clock loads."""
    workloads = {}
    for name in ("write", "query", "claim", "mark"):
        visits = [w["db"][name] for w in windows]
        started = sum(d["waits"] for d in visits)
        wait_sum = sum(d["wait_sum_ms"] for d in visits)
        workloads[name] = dict(
            expected_load=expected[name],
            admitted_mean=sum(d["visits"] for d in visits) / roots,
            completed_mean=sum(d["completed"] for d in visits) / roots,
            admitted_load=sum(d["offered_s"] for d in visits) / (roots * horizon),
            busy_s=sum(d["busy_s"] for d in visits) / roots,
            started=started,
            mean_wait_ms=wait_sum / started if started else None,
        )
    return workloads


def branch_diagnostics(c, family, arm, branches):
    """Aggregate descriptive evidence without making each window a replicate."""
    window_groups = list(zip(*(b["windows"] for b in branches), strict=True))
    windows = [w for group in window_groups for w in group]
    n = len(branches)
    share = branches[0]["share"]
    mean_business = (
        c.ruby_ms if family == "C1" else c.ruby_ms + share * (c.java_ms - c.ruby_ms)
    )
    expected = {
        "write": c.writes * (mean_business + c.insert_ms) / 1000,
        "query": branches[0]["query_rate"] * c.query_ms / 1000,
        "claim": c.writes * c.claim_ms / 1000,
        "mark": c.writes * c.mark_ms / 1000,
    }
    db = _db_summary(windows, n, expected, c.horizon)
    initial, final = window_groups[0], window_groups[-1]
    for name, measures in db.items():
        measures["still_waiting_mean"] = _mean(
            w["db"][name]["still_waiting"] for w in final
        )

    window_rows = []
    for index, group in enumerate(window_groups, start=1):
        row = dict(window=index, **_pooled_counts(group))
        for key in (
            "pending",
            "age_s",
            "throughput",
            "queue",
            "work_s",
            "utilisation",
            "delay_p50_s",
            "delay_p95_s",
            "empty",
        ):
            row[key] = _mean(w[key] for w in group)
        row["no_marks"] = sum(w["marks"] == 0 for w in group)
        window_rows.append(row)

    stats = dict(
        roots=n,
        expected_load=branches[0]["expected_load"],
        realised_load=sum(w["realised_cohort_work_s"] for w in windows)
        / (n * c.horizon),
        utilisation=_mean(w["utilisation"] for w in windows),
        **_pooled_counts(windows),
        queue_start=_mean(w["queue_start"] for w in initial),
        queue_end=_mean(w["queue"] for w in final),
        work_start_s=_mean(w["work_start_s"] for w in initial),
        work_end_s=_mean(w["work_s"] for w in final),
        pending_start=_mean(w["pending_start"] for w in initial),
        pending_end=_mean(w["pending"] for w in final),
        age_end_s=_mean(w["age_s"] for w in final),
        marks_per_s=sum(w["marks"] for w in windows) / (n * c.horizon),
        typical_delay_p50_s=_mean(w["delay_p50_s"] for w in windows),
        typical_delay_p95_s=_mean(w["delay_p95_s"] for w in windows),
        no_mark_windows=sum(w["marks"] == 0 for w in windows),
    )
    return dict(branch=stats, workloads=db, windows=window_rows)


def saved_latency_distributions(out_dir, family, arm, branches):
    """Check retained samples against their root records before pooling writes."""
    elapsed, routes, unresolved = [], [], []
    for root, branch in enumerate(branches):
        path = out_dir / "samples" / family / f"{root:04d}.npz"
        with np.load(path, allow_pickle=False) as data:
            ns = data[f"{arm}_ns"]
            java = data[f"{arm}_java"].astype(bool)
            censored = data[f"{arm}_censored"].astype(bool)
        if not (ns.shape == java.shape == censored.shape == (branch["scored"],)):
            raise ValueError(f"Scored cohort mismatch in {path}, arm {arm}")
        if censored.sum() != branch["censored"]:
            raise ValueError(f"Censor count mismatch in {path}, arm {arm}")
        elapsed.append(ns)
        routes.append(java)
        unresolved.append(censored)

    ns, java, censored = map(np.concatenate, (elapsed, routes, unresolved))
    table = {
        "pooled": latency_distribution(ns, censored),
        "Ruby": latency_distribution(ns[~java], censored[~java]),
        "Java": latency_distribution(ns[java], censored[java]),
    }
    # Rasterise only at plotting resolution. Quantile tables use all observations.
    if not len(ns):
        return table, {"ms": [], "lower": [], "upper": []}
    grid = np.geomspace(max(ns.min(), 1) * 0.99, max(ns.max(), 1) * 1.01, 700)
    complete = np.searchsorted(np.sort(ns[~censored]), grid, side="right")
    ambiguous = np.searchsorted(np.sort(ns[censored]), grid, side="left")
    curve = dict(
        ms=grid / MS, lower=complete / len(ns), upper=(complete + ambiguous) / len(ns)
    )
    return table, curve


def _table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def _number(value, digits=2):
    if value is None:
        return "—"
    if isinstance(value, dict):
        return f"≥{value['lower']:,.{digits}f} (censored)"
    return f"{value:,.{digits}f}"


def _pct(x):
    return f"{x * 100:.2f}%"


def _pp(x):
    return f"{x * 100:+.2f}"


def _write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, obj):
    contents = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(contents)


def _label(family, arm):
    from experiments.figures import ARMS

    return ARMS[family][arm] if family == "C23" else family + " " + ARMS[family][arm]


def _relative(path, parent):
    return Path(os.path.relpath(path, parent)).as_posix()


def _collect(c, out_dir, plots_dir, roots):
    from experiments.figures import ARMS, plot_latency, plot_trace

    arms, windows, db, latencies = [], [], [], []
    for family, root_set in roots.items():
        curves = {}
        for arm in ARMS[family]:
            branches = [root[arm] for root in root_set]
            label = {"family": family, "arm": arm}
            diagnostics = branch_diagnostics(c, family, arm, branches)
            arms.append({**label, **diagnostics["branch"]})
            windows.extend({**label, **row} for row in diagnostics["windows"])
            db.extend(
                {**label, "workload": name, **row}
                for name, row in diagnostics["workloads"].items()
            )
            table, curves[arm] = saved_latency_distributions(
                out_dir, family, arm, branches
            )
            latencies.extend(
                {**label, "route": route, **row} for route, row in table.items()
            )
        plot_latency(family, curves, c.deadline_ms, plots_dir)
        plot_trace(family, root_set[0], c, plots_dir)
    return arms, windows, db, latencies


def _findings_section(c, summary, validation):
    findings = control_findings(summary)
    pub = summary["publisher"]
    rows = [
        (
            "C1 equal work",
            f"{sum(r['passed'] for r in validation['placebo'])} of "
            f"{len(validation['placebo'])} root histories passed the four-share trace comparison.",
        ),
        (
            "C2 reference demand",
            f"Promotion changed breach risk by {_pp(summary['migration']['ref']['estimate'])} pp.",
        ),
        (
            "C3 higher demand",
            f"Promotion changed breach risk by {_pp(summary['migration']['high']['estimate'])} pp. "
            f"The high-minus-reference interaction was {_pp(summary['interaction']['estimate'])} pp.",
        ),
        (
            "C4 background overload",
            f"All-Ruby absolute risk was {_pct(summary['risks']['C4']['ruby']['risk'])}. "
            f"The paired promotion effect was {_pp(summary['overload']['estimate'])} pp.",
        ),
        (
            "C5 slow publisher",
            "Sensor ambiguity demonstrated."
            if pub["passed"]
            else "Publisher-specific ambiguity was not established.",
        ),
    ]
    return [
        "# AIMS5701 feasibility pilot",
        f"**Overall verdict — {summary['verdict']}.** " + " ".join(summary["reasons"]),
        f"**C2/C3 migration mechanism.** {findings['migration']} The comparison holds at "
        f"{c.hold:.0%} Java or promotes to {c.promote:.0%} under the same simulated inputs.",
        f"**C5 publisher-specific ambiguity.** {findings['publisher']} {findings['outbox']} "
        "This control requires evidence for both conditions.",
        f"Each breach is one {c.horizon} s history with more than {c.miss_fraction:.0%} late writes in any "
        f"{c.window} s window. Deadlines are {c.deadline_ms} ms from arrival. Absolute risks use "
        "independent histories as denominators. The historical four-nines availability target motivated the study. "
        "These results concern a synthetic latency endpoint.",
        f"One non-preemptive FIFO DB serves both routes, background queries and the common outbox publisher. "
        f"At {c.writes:g} writes/s the migration condition assigns a mean of {c.ruby_ms+c.insert_ms} ms of "
        f"synchronous DB work to a Ruby write and {c.java_ms+c.insert_ms} ms to a Java write, including the "
        f"same atomic insert. Promotion therefore adds "
        f"{100*(c.load_fraction(c.promote,0)-c.load_fraction(c.hold,0)):.1f} percentage points of offered DB "
        f"load at either background setting. These service costs represent a synthetic implementation difference.",
        _table(("Control", "Finding"), rows),
    ]


def _validation_section(validation):
    rows = []
    for cell in validation["queue"]:
        lower, upper = cell["interval_ms"]
        tolerance = cell["tolerance_ms"]
        rows.append(
            (
                cell["distribution"],
                cell["rate"],
                len(cell["seed_means_ms"]),
                _number(cell["expected_ms"]),
                _number(cell["mean_ms"]),
                f"{lower:.2f} to {upper:.2f}",
                f"{tolerance[0]:.2f} to {tolerance[1]:.2f}",
                cell["status"],
            )
        )
    return [
        "## C0 queue validation and C1 placebo",
        "The isolated queue uses the simulator's FIFO service path. Each cell compares the "
        "95% Student-t interval across independent seed means with its analytical equivalence band.",
        _table(
            (
                "Queue",
                "Arrivals/s",
                "Seeds",
                "Expected wait (ms)",
                "Observed (ms)",
                "95% interval (ms)",
                "±10% band (ms)",
                "Status",
            ),
            rows,
        ),
        "C1 matches the per-request service draw and publisher lifecycle across 0%, 25%, 60% and 100% Java. "
        "Its exact path check excludes the routing label and includes physical times and states.",
    ]


def _effect_section(summary):
    rows = []
    for key, row in summary["migration"].items():
        name = "C2 reference" if key == "ref" else "C3 higher"
        rows.append(
            (
                name,
                row["n"],
                row["neither"],
                row["hold_only"],
                row["promote_only"],
                row["both"],
                _pp(row["estimate"]),
                _pp(row["lower"]),
                _pp(row["upper"]),
            )
        )
    hold_risks = summary["risks"]["C23"]
    room_reference = 100 * (1 - hold_risks["ref_hold"]["risk"])
    room_higher = 100 * (1 - hold_risks["high_hold"]["risk"])
    interaction = summary["interaction"]
    if interaction["lower"] > 0:
        interpretation = (
            "Higher background demand amplified the measured promotion effect."
        )
    elif interaction["upper"] < 0:
        interpretation = "Higher background demand reduced the measured promotion effect on this endpoint."
    else:
        interpretation = "The descriptive interaction interval includes zero."
    return [
        "## C2/C3 routing effect",
        _table(
            (
                "Setting",
                "Paired roots",
                "Neither",
                "Hold only",
                "Promote only",
                "Both",
                "Δ risk (pp)",
                "Lower (pp)",
                "Upper (pp)",
            ),
            rows,
        ),
        "The paired estimate subtracts hold-only from promote-only counts and divides by all roots. "
        "Each column of directional bounds has simultaneous one-sided 95% coverage across both settings. "
        "A lower bound reaching +10 pp qualifies a setting. Exclusion requires both upper bounds strictly below +10 pp.",
        f"At the observed hold risks, the largest possible *sample* increase was "
        f"{room_reference:.2f} pp under reference demand and {room_higher:.2f} pp under higher demand. "
        "This ceiling helps interpret saturation. The frozen bounds determine the migration finding.",
        f"**Background interaction.** High minus reference was {_pp(interaction['estimate'])} pp "
        f"(descriptive 95% whole-root bootstrap interval {_pp(interaction['lower'])} to {_pp(interaction['upper'])} pp). "
        + interpretation
        + " All four branches shared one 60 s prefix. The extra queries began at the cut, "
        "so pre-decision telemetry carried no warning of their arrival.",
    ]


def _risk_rows(summary):
    return [
        (
            _label(family, arm),
            r["breaches"],
            r["n"],
            _pct(r["risk"]),
            f"{_pct(r['lower'])} to {_pct(r['upper'])}",
        )
        for family, arms in summary["risks"].items()
        for arm, r in arms.items()
    ]


def _publisher_section(c, publisher):
    pair = publisher["paired"]
    absolute = publisher["absolute"]
    age = publisher["age"]
    rows = [
        (name, _pct(absolute[name]["upper"]), f"< {_pct(c.health_limit)}")
        for name in ("nominal", "slow")
    ]
    rows.append(
        ("Slow minus nominal", _pp(pair["upper"]) + " pp", f"< {_pp(c.harm_limit)} pp")
    )
    return [
        "## C5 publisher-only intervention",
        "The two branches share 25% Java, 40 writes/s, an initial nominal publisher and zero background arrivals. "
        "New acknowledgement waits last 2 or 50 ms. DB work remains 1 ms per claim and per mark.",
        _table(
            ("Write-health quantity", "One-sided 95% upper bound", "Required"), rows
        ),
        f"The paired risk difference was {_pp(pair['estimate'])} pp. Of {pair['n']} pairs, "
        f"{pair['slow_only']} breached only under slow acknowledgement, {pair['nominal_only']} only under nominal, "
        f"{pair['both']} under both and {pair['neither']} under neither. "
        f"The corresponding one-sided lower bound on harm was {_pp(pair['lower'])} pp. "
        f"Write health was {publisher['health']}.",
        f"**Oldest pending age.** In {age['successes']} of {publisher['n']} paired histories "
        f"({_pct(age['fraction'])}), the slow branch's oldest pending row was at least {c.age_margin} s older "
        f"at the horizon. The exact one-sided 95% lower bound was {_pct(age['lower'])}. "
        f"The required bound is strictly above {_pct(c.success_fraction)}. Empty outboxes contribute zero seconds. "
        "Claimed rows remain pending until their marks complete.",
    ]


def _diagnostic_sections(c, summary, arms, db, latencies):
    overload = summary["overload"]
    c4 = summary["risks"]["C4"]
    c4_text = ""
    if (
        c4["hold"]["breaches"] == c4["hold"]["n"]
        and c4["promote"]["breaches"] == c4["promote"]["n"]
    ):
        c4_text = (" Both branches breached in every sampled horizon. The binary endpoint saturated. "
                   "All paired differences were zero, so every empirical bootstrap resample returned zero. "
                   "This degenerate interval leaves the magnitude of additional waiting to the latency diagnostics.")
    elif min(c4["hold"]["risk"], c4["promote"]["risk"]) > 0.95:
        c4_text = " Both branch risks exceeded 95% in this sample, leaving little room for this endpoint to increase."
    totals = []
    queues = []
    relay = []
    waits = []
    for row in arms:
        label = _label(row["family"], row["arm"])
        totals.append(
            (
                label,
                _pct(row["expected_load"]),
                _pct(row["realised_load"]),
                _pct(row["utilisation"]),
                row["misses"],
                row["eligible"],
                _pct(row["miss_fraction"]) if row["eligible"] else "—",
            )
        )
        queues.append(
            (
                label,
                _number(row["queue_start"]),
                _number(row["queue_end"]),
                _number(row["queue_end"] - row["queue_start"]),
                _number(row["work_start_s"], 3),
                _number(row["work_end_s"], 3),
                _number(row["work_end_s"] - row["work_start_s"], 3),
            )
        )
        relay.append(
            (
                label,
                _number(row["pending_start"]),
                _number(row["pending_end"]),
                _number(row["age_end_s"], 3),
                _number(row["marks_per_s"]),
                _number(row["typical_delay_p50_s"], 3),
                _number(row["typical_delay_p95_s"], 3),
                row["no_mark_windows"],
            )
        )
    for row in db:
        waits.append(
            (
                _label(row["family"], row["arm"]),
                row["workload"],
                _pct(row["expected_load"]),
                _pct(row["admitted_load"]),
                _number(row["completed_mean"]),
                _number(row["busy_s"], 3),
                _number(row["mean_wait_ms"]),
                _number(row["still_waiting_mean"]),
            )
        )

    tails = [
        (
            _label(row["family"], row["arm"]),
            row["route"],
            row["writes"],
            row["censored"],
            *[_number(row[k]) for k in ("p50_ms", "p95_ms", "p99_ms")],
        )
        for row in latencies
    ]

    return [
        "## Absolute breach risks",
        _table(
            ("Branch", "Breached", "Histories", "Risk", "95% exact interval"),
            _risk_rows(summary),
        ),
        "## C4 background overload",
        f"Under overload the all-Ruby risk was {_pct(c4['ruby']['risk'])}, compared with "
        f"{_pct(c4['hold']['risk'])} at 25% and {_pct(c4['promote']['risk'])} at 60%. The paired "
        f"promote-minus-hold increment was {_pp(overload['estimate'])} pp (descriptive 95% bootstrap interval "
        f"{_pp(overload['lower'])} to {_pp(overload['upper'])} pp). The all-Ruby baseline warmed up at 0% Java. "
        f"The other two branches shared a 25% warm-up.{c4_text}",
        "## Application writes and database demand",
        "The expected and realised demand columns count total work owed by the arrival cohort relative to capacity. "
        "DB utilisation counts in-window occupied time, including partial visits. Miss fractions pool eligible writes. "
        "A busy server can accumulate additional work while utilisation stays close to 100%.",
        _table(
            (
                "Branch",
                "Expected demand",
                "Realised demand",
                "DB utilisation",
                "Misses",
                "Eligible writes",
                "Miss fraction",
            ),
            totals,
        ),
        "### Write latency",
        "Empirical percentiles use the same deadline-based cohort as the breach score. The pooled row weights writes "
        "equally across both routes. Outstanding writes can complete during the closed-arrival diagnostic drain. "
        "A lower-bound symbol identifies an indeterminate percentile due to right censoring. Dashes denote empty cohorts.",
        _table(
            ("Branch", "Route", "Writes", "Censored", "p50 ms", "p95 ms", "p99 ms"),
            tails,
        ),
        "### DB queue growth",
        "Values are per-root means at the decision and horizon. Unfinished work includes the active visit's residual "
        "and queued visits. Future publisher visits owed by pending rows appear in cohort demand and are tracked in the outbox.",
        _table(
            (
                "Branch",
                "Queued at cut",
                "Queued at 120 s",
                "Change",
                "Work at cut (s)",
                "Work at 120 s (s)",
                "Change (s)",
            ),
            queues,
        ),
        "### Outbox and publisher",
        "Counts and oldest age are mean snapshots. Throughput is completed marks per horizon second. Delay columns "
        "average per-window percentiles across windows with marks. The no-mark count preserves windows with no observed delay.",
        _table(
            (
                "Branch",
                "Pending at cut",
                "Pending at 120 s",
                "Oldest at 120 s (s)",
                "Throughput (marks/s)",
                "Mean window p50 delay (s)",
                "Mean window p95 delay (s)",
                "No-mark windows",
            ),
            relay,
        ),
        "### DB visits and waiting by workload",
        f"Completed visits and busy seconds are means per {c.horizon} s horizon. Mean wait pools visits that entered service. "
        "The final column counts visits still queued at the horizon, averaged over roots. Expected demand charges "
        "eventual publication to each write. Admitted demand charges actual claim and mark visits when they reach the DB.",
        _table(
            (
                "Branch",
                "Workload",
                "Expected demand",
                "Admitted demand",
                "Completed visits",
                "Busy (s)",
                "Observed mean wait (ms)",
                "Still queued at 120 s",
            ),
            waits,
        ),
    ]


def generate(c, out_dir, plots_dir, report_path, validation, roots, summary):
    """Write deterministic tables, plots and the scoped scientific verdict."""
    from experiments.figures import plot_risks

    out_dir, plots_dir, report_path = map(Path, (out_dir, plots_dir, report_path))
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    all_roots = {"C1": [r["arms"] for r in validation["placebo"]], **roots}
    arms, windows, db, latencies = _collect(c, out_dir, plots_dir, all_roots)
    plot_risks(summary["risks"], summary["migration"], c.margin, plots_dir)
    for name, rows in (
        ("branch-diagnostics", arms),
        ("window-diagnostics", windows),
        ("db-work", db),
        ("latency-by-route", latencies),
    ):
        _write_csv(summary_dir / (name + ".csv"), rows)
    _write_json(summary_dir / "verdict.json", summary)
    plots = _relative(plots_dir, report_path.parent)
    results = _relative(out_dir, report_path.parent)
    sections = [
        *_findings_section(c, summary, validation),
        *_validation_section(validation),
        *_effect_section(summary),
        f"![Absolute branch risks and paired directional risk bounds]({plots}/breach-risks.png)",
        *_publisher_section(c, summary["publisher"]),
        *_diagnostic_sections(c, summary, arms, db, latencies),
        "## Paired trajectories and latency distributions",
        "Traces use root ID 0, selected before execution. Points show the fixed measurement boundaries. "
        "Window quantities cover the preceding 30 seconds. Each latency curve pools the entire scored cohort across roots. "
        "Shading, where present, spans the possible cumulative fractions under right censoring.",
    ]
    names = {
        "C1": "Equal-work placebo",
        "C23": "Reference and higher background demand",
        "C4": "Background overload",
        "C5": "Publisher slowdown",
    }
    for family in all_roots:
        sections.extend(
            [
                f"### {names[family]}",
                f"![{family} paired root-zero trajectories]({plots}/trace-{family}.png)",
                f"![{family} empirical latency distribution]({plots}/latency-{family}.png)",
            ]
        )
    sections.extend(
        [
            "## Scope and reproduction",
            "The model represents one non-preemptive FIFO resource shared by application writes, publisher DB visits and "
            "independent queries. The 60 s initial history and 120 s horizon define a transient experiment. "
            "Its results apply to these fixed workloads, the 25%→60% routing step and the binary latency endpoint. "
            "Aurora scheduling, production availability, sensor lead time and particle-filter performance require separate evidence.",
            f"Master seed {c.seed}. The independent sampling unit is a root history. The exact config, source and environment "
            f"are recorded in the [run manifest]({results}/manifest.json). The [machine-readable verdict]({results}/summary/verdict.json) "
            f"retains full precision. The [window diagnostics]({results}/summary/window-diagnostics.csv) retain pooled denominators "
            f"and deadline misses by branch and window. See the [reproduction instructions](README.md) and "
            f"[independent reconstruction]({results}/summary/independent-audit.json).",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n\n".join(sections) + "\n")
