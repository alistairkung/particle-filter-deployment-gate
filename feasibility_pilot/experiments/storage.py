"""Root is the atomic unit for resume, checksums and statistical pairing."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from experiments.runner import arms, run_root


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        tmp = stream.name
    os.replace(tmp, path)


def load_json(path):
    return json.loads(Path(path).read_text())


def budgets(c):
    return {"C23": c.migration_roots, "C4": c.overload_roots, "C5": c.publisher_roots}


def root_paths(out, family, root):
    return (
        out / "branches" / family / f"{root:04d}.json",
        out / "samples" / family / f"{root:04d}.npz",
    )


def read_root(out, c, family, root, signature):
    branch_path, sample_path = root_paths(out, family, root)
    record = load_json(branch_path)
    if (record["family"], record["root"], record["signature"]) != (
        family,
        root,
        signature,
    ):
        raise ValueError(f"Root provenance mismatch at {branch_path}")
    if set(record["branches"]) != {a.name for a in arms(c, family)}:
        raise ValueError(f"Incomplete branch vector at {branch_path}")
    if digest(sample_path) != record["sample_digest"]:
        raise ValueError(f"Sample digest mismatch at {sample_path}")
    return record["branches"]


def compute_root(task):
    c, out, family, root, signature = task
    branch_path, sample_path = root_paths(out, family, root)
    if branch_path.exists():
        read_root(out, c, family, root, signature)
        return family, root
    branches, samples = run_root(c, family, root)
    arrays = {}
    for arm, (elapsed, java, censored) in samples.items():
        arrays.update(
            {arm + "_ns": elapsed, arm + "_java": java, arm + "_censored": censored}
        )
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=sample_path.parent, delete=False) as stream:
        np.savez_compressed(stream, **arrays)
        tmp = stream.name
    os.replace(tmp, sample_path)
    save_json(
        branch_path,
        dict(
            family=family,
            root=root,
            signature=signature,
            sample_digest=digest(sample_path),
            branches=branches,
        ),
    )
    return family, root
