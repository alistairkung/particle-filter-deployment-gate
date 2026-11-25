from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.config import read_config
from experiments.storage import compute_root, read_root, root_paths


def test_resume_preserves_pair_and_rejects_wrong_signature_or_samples(tmp_path):
    c = replace(
        read_config("configs/pilot.toml"), seed=123, warmup=1, window=1, horizon=2
    )
    task = (c, tmp_path, "C5", 0, "test-signature")
    compute_root(task)
    branches = read_root(tmp_path, c, "C5", 0, "test-signature")
    assert set(branches) == {"nominal", "slow"}
    assert branches["nominal"]["cut"] == branches["slow"]["cut"]
    path, samples = root_paths(tmp_path, "C5", 0)
    before = path.stat().st_mtime_ns
    compute_root(task)
    assert path.stat().st_mtime_ns == before
    with np.load(samples) as arrays:
        assert len(arrays["nominal_ns"]) == branches["nominal"]["scored"]
    with pytest.raises(ValueError, match="provenance"):
        read_root(tmp_path, c, "C5", 0, "other")
    samples.write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="digest"):
        compute_root(task)
