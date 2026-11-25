"""Generate the readiness decision only from reconciled and reviewed evidence."""

from . import PILOT
from .custody import file_hash, read

LEVELS = ("prior", "age", "current", "history", "evaluator")


def verified_verdict(run):
    scores = read(run / "scores.json")
    verdict = (
        "READY"
        if all(scores["claims"][name]["passes"] for name in ("demand", "increment"))
        else "READY WITH LIMITATIONS"
    )
    audit = read(run / "independent-audit.json")
    if (
        not audit["verified"]
        or audit["protected_roots"] != 1200
        or audit["verdict"] != verdict
    ):
        raise ValueError("Independent scientific reconstruction disagrees")
    pilot = read(run / "pilot-reproduction.json")
    original = PILOT / "results/frozen/summary/clean-reproduction.json"
    if not (
        pilot["passed"]
        and pilot["verdict"] == "GO"
        and pilot["original_receipt"] == file_hash(original)
        and pilot["inventory_sha256"] == read(original)["inventory_sha256"]
    ):
        raise ValueError("Missing or invalid frozen reproduction")
    review = read(run / "verification-review.json")
    if not (
        review["passed"]
        and review["audit"] == file_hash(run / "independent-audit.json")
        and review["lock"] == file_hash(run / "predictor-lock.json")
        and review["findings"]
    ):
        raise ValueError("Independent implementation review missing or stale")
    return verdict


def _plots(run, scores):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .custody import inventory, labelled_rows

    directory = run / "plots"
    directory.mkdir(exist_ok=True)
    rows = list(
        labelled_rows(run, [r for r in inventory() if r["partition"] == "protected"])
    )
    conditions = ("reference", "contention", "publisher")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), layout="constrained")
    for ax, field, title in zip(
        axes,
        ("demand", "unfinished"),
        ("Realised query demand (s/s)", "Unfinished DB work (s)"),
    ):
        ax.boxplot(
            [[r[field] for r in rows if r["condition"] == cond] for cond in conditions],
            tick_labels=conditions,
            showfliers=False,
        )
        ax.set_ylabel(title)
    fig.savefig(directory / "physical-targets.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), layout="constrained")
    for ax, action in zip(axes, ("hold", "promote")):
        ax.plot([0, 1], [0, 1], color="0.6", linestyle="--")
        for level in ("age", "current", "history"):
            bins = scores["populations"]["all"]["models"][level]["calibration"][action]
            used = [b for b in bins if b["count"]]
            ax.plot(
                [b["mean_prediction"] for b in used],
                [b["event_rate"] for b in used],
                marker="o",
                label=level,
            )
        ax.set(
            title=action.capitalize(),
            xlabel="Mean predicted risk",
            ylabel="Observed breach rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )
    axes[-1].legend()
    fig.savefig(directory / "action-calibration.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.6), layout="constrained")
    for ax, claim in zip(axes, ("demand", "increment")):
        row = scores["claims"][claim]
        ax.axvline(0, color="0.6")
        ax.plot([row["lower"], row["gain"]], [0, 0], lw=2)
        ax.plot(row["gain"], 0, "o")
        ax.set(
            yticks=[], title=claim.capitalize(), xlabel="Mean squared error reduction"
        )
        ax.margins(x=0.18)
    fig.suptitle("Score gain and simultaneous one-sided lower bound")
    fig.savefig(directory / "information-claims.png", dpi=140)
    plt.close(fig)
    return sorted(directory.glob("*.png"))


def report(run, manifest, workers):
    verdict = verified_verdict(run)
    scores = read(run / "scores.json")
    methods = scores["populations"]["all"]["models"]
    claims = scores["claims"]
    comparisons = scores["comparisons"]

    def comparison(target, baseline, candidate):
        return next(
            r
            for r in comparisons
            if (r["population"], r["target"], r["baseline"], r["candidate"])
            == ("all", target, baseline, candidate)
        )

    def estimate(r):
        return f"{r['gain']:+.5f} [{r['lower']:+.5f}, {r['upper']:+.5f}]"

    figures = _plots(run, scores)
    text = [
        "# AIMS5701 · Experimental Readiness Report",
        "",
        f"**Verdict — {verdict}.**",
        "",
        "## 1. Can the simulator supply legitimate pre-decision telemetry?",
        "",
        "Yes, under the passive lifecycle-feed assumption. Each prefix produced twelve five-second aggregates. Rows appeared on commit and remained pending through acknowledgement until their mark completed. Delivery at interval end was idealised for this study. Boundary tests exercised delayed and missing delivery, no completed marks and empty outboxes. The audit independently reconstructed the histories on all 1,200 protected roots.",
        "",
        "## 2. Can it compare both actions fairly?",
        "",
        "Yes. Hold and promote inherited the same 25% prefix and the same future primitive draws. The independent replay checked applied admissions, cut state, endpoint counts and 800 additional surprise continuations. There was one primary paired future per root. Each pair samples a possible outcome conditional on its prefix.",
        "",
        "| Protected population | Neither | Promote only | Hold only | Both | Hold risk | Promote risk | Increment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("all", "reference", "contention", "publisher", "surprise"):
        r = scores["populations"][name]
        cells = r["paired"]
        rates = r["rates"]
        text.append(
            f"| {name} | {cells['neither']} | {cells['promote_only']} | {cells['hold_only']} | {cells['both']} | {rates['hold']:.1%} | {rates['promote']:.1%} | {rates['increment']:+.1%} |"
        )
    text += [
        "",
        "## 3. Do outbox observations predict contention and future promotion harm?",
        "",
        "| Confirmatory claim | Development-selected level | Roots | Gain over best constant | Simultaneous lower bound | Supported |",
        "|---|---|---:|---:|---:|---|",
    ]
    for name, r in claims.items():
        text.append(
            f"| {name} | {r['candidate']} | {r['n']} | {r['gain']:+.6f} | {r['lower']:+.6f} | {'yes' if r['passes'] else 'no'} |"
        )
    text += [
        "",
        "The physical claim compares realised query demand among nominal publishers under reference and elevated query rates. The incremental-risk claim uses all three conditions. Each gain is a reduction in squared error, with one-sided Bonferroni bounds giving approximate 95% familywise coverage conditional on the development lock. The signed outcome includes hold-only breaches with weight −1.",
        "",
        "| Diagnostic | Hold Brier | Promote Brier | Increment MSE | Demand RMSE (s/s) | Unfinished work RMSE (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for level in LEVELS:
        s = methods[level]["scores"]
        d = f"{s['demand_rmse']:.4f}" if "demand_rmse" in s else "—"
        u = f"{s['unfinished_rmse']:.4f}" if "unfinished_rmse" in s else "—"
        text.append(
            f"| {level} | {s['hold']:.4f} | {s['promote']:.4f} | {s['increment']:.4f} | {d} | {u} |"
        )
    # Pooled gains can arise solely from identifying the assigned condition.
    text += [
        "",
        "| Within-condition gain over its own mean | Demand | Signed increment |",
        "|---|---:|---:|",
    ]
    for population in ("reference", "contention", "publisher"):
        selected_gains = []
        for target in ("demand", "increment"):
            entry = next(
                row
                for row in comparisons
                if (row["population"], row["target"], row["baseline"], row["candidate"])
                == (population, target, "stratum_mean", claims[target]["candidate"])
            )
            selected_gains.append(estimate(entry))
        text.append(f"| {population} | {selected_gains[0]} | {selected_gains[1]} |")
    matrix = methods["history"]["confusion"]
    text += [
        "",
        "These descriptive intervals compare each selected forecast to a constant that already knows the condition. Between-condition differences can therefore support a pooled claim even if prediction within each individual condition remains weak.",
        "",
        f"The temporal classifier identified {matrix[2][2]}/400 publisher-slowdown roots and {matrix[1][1]}/400 established-contention roots. It labelled {matrix[1][2]} contention roots as publisher slowdown and {matrix[2][1]} publisher roots as contention. Classification answers a separate question from estimating realised query work.",
    ]
    p = comparison("promote", "prior", "current")
    text += [
        "",
        f"Current outbox versus the action prior improved promotion Brier score by {estimate(p)} (paired descriptive 95% interval; positive favours current). The evaluator-only model had access to the assigned condition and two physical targets. Its scores quantify the discrimination extracted by that diagnostic and cannot establish an upper bound on predictability.",
        "",
        "![Physical targets by condition](plots/physical-targets.png)",
        "",
        "![Action risk calibration](plots/action-calibration.png)",
        "",
        "## 4. Does observation history improve on the current complete outbox?",
        "",
        "| Target | History gain over current [descriptive 95% interval] |",
        "|---|---:|",
    ]
    for target in ("hold", "promote", "increment", "demand", "unfinished", "condition"):
        text.append(
            f"| {target} | {estimate(comparison(target, 'current', 'history'))} |"
        )
    text += [
        "",
        "Intervals resample whole roots within condition with fixed fits. An interval crossing zero leaves improvement unresolved. The comparison with a complete current reading controls for the number of available sensor types.",
        "",
        "## 5. Is simulator v1 ready to freeze?",
        "",
    ]
    temporal = comparison("increment", "current", "history")
    if temporal["lower"] > 0:
        temporal_interpretation = "The descriptive interval favours history for predicting the signed increment."
    elif temporal["upper"] < 0:
        temporal_interpretation = "The descriptive interval favours current measurements for predicting the signed increment."
    else:
        temporal_interpretation = "The incremental-risk comparison leaves the added value of history unresolved."
    text.insert(-2, temporal_interpretation)
    if verdict == "READY":
        text.append(
            "Both protected information claims passed and the validity checks reconciled. The simulator is suitable for a subsequent synthetic forecasting experiment under this workload law. A later particle filter must earn any advantage over the complete current-value and simple temporal baselines on fresh protected histories."
        )
    else:
        unsupported = ", ".join(name for name, r in claims.items() if not r["passes"])
        text.append(
            f"The validity checks reconciled. The following information claim remains unsupported: {unsupported}. A defensible follow-up asks whether a particle filter improves held-out action-specific and incremental forecasts over these locked baselines under the same bounded sensor law. The present evidence limits the expected gain."
        )
    text += [
        "",
        "![Score gains and uncertainty](plots/information-claims.png)",
        "",
        "The three conditions have equal design weights and fixed 30-second exposures. An unexpected future query increase received identical pre-decision forecasts to its ordinary reference sibling. Its outcomes appear separately above. The scores do not estimate deployment cost, production prevalence or a production gate threshold.",
        "",
        "The [machine-readable results](scores.json) contain per-condition comparisons, calibration counts, classifier confusions, overlap and surprise comparisons. The [independent audit](independent-audit.json), [pilot reproduction](pilot-reproduction.json), [predictor lock](predictor-lock.json) and [run manifest](manifest.json) identify the retained evidence.",
    ]
    path = run / "Experimental_Readiness_Report.md"
    from .custody import atomic

    atomic(path, lambda stream: stream.write(("\n".join(text) + "\n").encode()))
    return [
        path,
        *figures,
        run / "verification-review.json",
        run / "pilot-reproduction.json",
    ]
