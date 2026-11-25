"""Headless figures from root-level evidence and pooled scored writes."""

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mpl-cache")
)
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


ARMS = {
    "C1": {
        "zero": "0% Java",
        "hold": "25% Java",
        "promote": "60% Java",
        "all": "100% Java",
    },
    "C23": {
        "ref_hold": "C2 ref 25%",
        "ref_promote": "C2 ref 60%",
        "high_hold": "C3 high 25%",
        "high_promote": "C3 high 60%",
    },
    "C4": {"ruby": "All Ruby", "hold": "25% Java", "promote": "60% Java"},
    "C5": {"nominal": "Nominal 2 ms", "slow": "Slow 50 ms"},
}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#8C62AA")
STYLES = ("-", "--", "-.", ":")


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_risks(risks, effects, margin, plots):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    rows = [
        (
            f"{family} {ARMS[family][arm]}" if family != "C23" else ARMS[family][arm],
            branches[arm],
        )
        for family, branches in risks.items()
        for arm in ARMS[family]
        if arm in branches
    ]
    for y, (_, row) in enumerate(rows):
        left.hlines(
            y, row["lower"] * 100, row["upper"] * 100, color=COLORS[0], linewidth=2
        )
        left.plot(row["risk"] * 100, y, "o", color=COLORS[0])
    left.set(
        yticks=range(len(rows)),
        yticklabels=[name for name, _ in rows],
        xlim=(-2, 102),
        xlabel="Breach probability (%)",
        title="Absolute risk · exact two-sided 95% intervals",
    )
    left.invert_yaxis()
    labels = {"ref": "C2 reference", "high": "C3 higher"}
    for y, (_, effect) in enumerate(effects.items()):
        left_bound, right_bound = effect["lower"] * 100, effect["upper"] * 100
        right.scatter(
            [left_bound],
            [y],
            marker=">",
            color=COLORS[0],
            label="95% lower" if y == 0 else None,
        )
        right.scatter(
            [right_bound],
            [y],
            marker="<",
            color=COLORS[1],
            label="95% upper" if y == 0 else None,
        )
        right.plot(
            effect["estimate"] * 100,
            y,
            "o",
            color="black",
            label="Estimate" if y == 0 else None,
        )
    right.axvline(margin * 100, linestyle="--", color="0.35", label="Practical margin")
    right.set(
        yticks=range(len(effects)),
        yticklabels=[labels[name] for name in effects],
        xlabel="Promote minus hold (percentage points)",
        title="Paired effect · simultaneous bounds in each direction",
    )
    right.set_ylim(len(effects) - 0.5, -0.5)
    right.legend(loc="lower center", fontsize=8, ncol=2)
    for axis in (left, right):
        axis.grid(axis="x", alpha=0.20)
    _save(fig, plots / "breach-risks.png")


def plot_latency(family, curves, deadline_ms, plots):
    fig, axis = plt.subplots(figsize=(7.3, 4.1), layout="constrained")
    for index, (arm, label) in enumerate(ARMS[family].items()):
        curve = curves[arm]
        axis.plot(
            curve["ms"],
            curve["lower"],
            color=COLORS[index],
            linestyle=STYLES[index],
            label=label,
        )
        if np.any(np.asarray(curve["upper"]) > curve["lower"]):
            axis.fill_between(
                curve["ms"],
                curve["lower"],
                curve["upper"],
                alpha=0.15,
                color=COLORS[index],
            )
    axis.axvline(
        deadline_ms, linestyle=":", color="0.20", label=f"{deadline_ms} ms deadline"
    )
    axis.set(
        xscale="log",
        xlabel="Arrival-to-commit latency (ms, log scale)",
        ylabel="Fraction of scored writes",
        ylim=(-0.01, 1.02),
        title=f"{family} · pooled write latency",
    )
    axis.legend(loc="lower right", fontsize=8)
    axis.grid(axis="x", alpha=0.20)
    _save(fig, plots / f"latency-{family}.png")


def plot_trace(family, branches, c, plots):
    series = (
        ("miss_fraction", "Deadline misses (%)", None),
        ("queue", "Queued DB operations", "queue_start"),
        ("work_s", "Unfinished DB work (s)", "work_start_s"),
        ("pending", "Pending outbox rows", "pending_start"),
        ("age_s", "Oldest pending age (s)", "age_start_s"),
        ("throughput", "Publisher throughput (marks/s)", None),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, layout="constrained")
    for axis, (metric, label, start_key) in zip(axes.flat, series):
        for index, (arm, name) in enumerate(ARMS[family].items()):
            windows = branches[arm]["windows"]
            time = [w["window"] * c.window for w in windows]
            if metric == "miss_fraction":
                values = [
                    100 * w["misses"] / w["eligible"] if w["eligible"] else np.nan
                    for w in windows
                ]
            else:
                values = [w[metric] for w in windows]
            if start_key:
                time.insert(0, 0)
                values.insert(0, windows[0][start_key])
            axis.plot(
                time,
                values,
                color=COLORS[index],
                linestyle=STYLES[index],
                marker=".",
                markersize=5,
                label=name,
            )
        if metric == "miss_fraction":
            axis.axhline(
                c.miss_fraction * 100, color="0.35", linewidth=1, linestyle=":"
            )
        axis.set(ylabel=label, ylim=(0, None), xticks=range(0, c.horizon + 1, c.window))
        axis.grid(alpha=0.18)
    for axis in axes[-1]:
        axis.set_xlabel("Time after decision (s)")
    fig.suptitle(f"{family} · paired continuations from root 0")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncols=len(branches),
        frameon=False,
    )
    _save(fig, plots / f"trace-{family}.png")
