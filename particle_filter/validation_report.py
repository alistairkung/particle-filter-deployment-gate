"""Summarise held-out development evidence before the method lock."""

import argparse
import json
import os
from pathlib import Path

from telemetry_sufficiency.custody import file_hash, save

from .development import study_source_digest


def _diagnostic_plot(folds, dest):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout="constrained")
    metrics = (
        "mark_tail",
        "backlog_tail",
        "empty",
        "mark_pit_lag1",
        "pending_pit_lag1",
    )
    for j, regime in enumerate("ABC"):
        for i, metric in enumerate(metrics):
            ax = axes[int(i >= 3), j]
            for fold, report in enumerate(folds):
                check = report["diagnostics"][regime][metric]
                centre = check["observed"] - check["predicted"]
                x = i % 3 + (fold - 2) * 0.095
                ax.plot(x, centre, ".", color=f"C{fold}")
                if bounds := check["error_ci95"]:
                    ax.vlines(x, *bounds, color=f"C{fold}", lw=1)
        for row, labels in enumerate(
            (("Marks tail", "Backlog tail", "Empty"), ("Marks", "Pending"))
        ):
            ax = axes[row, j]
            ax.set_xticks(range(len(labels)), labels, fontsize=9)
            ax.axhline(0, color="0.4", lw=0.7)
            if row == 0:
                ax.set_title(f"Regime {regime}")
        axes[0, 0].set_ylabel("Observed − predicted fraction")
        axes[1, 0].set_ylabel("Lag-one centred residual product")
    fig.suptitle("Held-out development diagnostics by fold (whole-root 95% intervals)")
    fig.savefig(dest, dpi=160)
    plt.close(fig)


def summarise(teacher, sequential, phase, output):
    # Keep the report and its sibling evidence directories together when relocating a run.
    bundle = output.resolve().parent
    if any(
        not p.resolve().is_relative_to(bundle) for p in (teacher, sequential, phase)
    ):
        raise ValueError("Validation inputs must be inside the output evidence bundle")
    teacher_summary = json.loads((teacher / "summary.json").read_text())
    sequence = json.loads((sequential / "sequential.json").read_text())
    phase_report = json.loads(phase.read_text())
    if teacher_summary["roots"] != 1200 or teacher_summary["intervals"] != 14400:
        raise ValueError("incomplete teacher-forced evidence")
    if phase_report["development_roots"] != 1200 or sequence["roots"] != 36:
        raise ValueError("incomplete phase or sequential evidence")
    output.mkdir(parents=True, exist_ok=False)
    paths = dict(
        teacher=teacher / "summary.json",
        sequential=sequential / "sequential.json",
        phase=phase,
    )
    for fold in range(5):
        a, b = teacher / f"fold-{fold}.json", sequential / f"fold-{fold}.json"
        if (
            json.loads(a.read_text())["training_digest"]
            != json.loads(b.read_text())["training_digest"]
        ):
            raise ValueError(
                "held-out teacher and sequential fits have different training roots"
            )
        paths[f"teacher_fold_{fold}"], paths[f"sequential_fold_{fold}"] = a, b
    for name, directory in (("teacher", teacher), ("sequential", sequential)):
        folds = [
            json.loads((directory / f"fold-{i}.json").read_text()) for i in range(5)
        ]
        _diagnostic_plot(folds, output / f"{name}.png")
    flags = teacher_summary["repeatable_discrepancies"]
    status = "requires-review"
    if flags or sequence["flags"]:
        status = "investigate"
    result = dict(
        status=status,
        source_digest=study_source_digest(),
        evidence_digests={
            os.path.relpath(p, output): file_hash(p) for p in paths.values()
        },
        teacher_flags=flags,
        sequential_flags=sequence["flags"],
        development_roots=teacher_summary["roots"],
        panel_roots=sequence["roots"],
    )
    save(output / "validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher", "sequential", "phase", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            summarise(args.teacher, args.sequential, args.phase, args.output), indent=2
        )
    )
