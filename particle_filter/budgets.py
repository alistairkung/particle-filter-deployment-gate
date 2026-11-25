"""Apply the predeclared numerical tolerances to the held-out development panel."""

import numpy as np


def _panel(values, width):
    a = np.asarray(values, dtype=float)
    if a.shape != (3, 12, 4, width) or not np.isfinite(a).all():
        raise ValueError("expected finite A/B/C × 12 roots × 4 seeds panel")
    return a


def particle_change(smaller, larger):
    rms = np.sqrt(np.mean((_panel(smaller, 4) - _panel(larger, 4)) ** 2, axis=(1, 2)))
    report = {z: dict(regime_rms=row[:3].tolist(), work_rms_s=float(row[3]))
              for z, row in zip("ABC", rms, strict=True)}
    report["passes"] = bool((rms[:, :3] <= .03).all() and (rms[:, 3] <= .01).all())
    return report


def rollout_variation(risks):
    """Finite-K noise conditional on one posterior per root across independent batches."""
    variance = np.var(_panel(risks, 3), axis=2, ddof=1).mean(axis=1)
    pooled = variance.mean(axis=0)
    return dict(per_regime_rms={z: np.sqrt(row).tolist() for z, row in zip("ABC", variance, strict=True)},
                pooled_rms=np.sqrt(pooled).tolist(), increment_variance=float(pooled[2]),
                passes=bool((np.sqrt(pooled) <= .03).all()))
