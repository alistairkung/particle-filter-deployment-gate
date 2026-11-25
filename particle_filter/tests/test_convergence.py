import hashlib
import json

import pytest


def test_numerical_panel_requires_the_exact_validated_report(tmp_path):
    from particle_filter.convergence import validate_evidence

    report = tmp_path / "validation.json"
    report.write_text(json.dumps({"status": "requires-review"}))
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="validated"):
        validate_evidence(report, digest)
    report.write_text(json.dumps({"status": "validated"}))
    with pytest.raises(ValueError, match="digest"):
        validate_evidence(report, digest)
    current = hashlib.sha256(report.read_bytes()).hexdigest()
    assert validate_evidence(report, current)["status"] == "validated"


def test_particle_doubling_loss_uses_signed_paired_outcomes():
    import numpy as np
    from particle_filter.convergence import paired_loss_change

    # Independently of action base rates, the signed target is -1 on each root.
    target = -np.ones((3, 12))
    small = np.zeros((3, 12, 4))
    large = np.full(small.shape, -0.5)
    report = paired_loss_change(small, large, target)
    assert report["larger_minus_smaller_mse"] == pytest.approx(-0.75)
    assert report["seed_sd"] == pytest.approx(0)


def test_panel_cli_preserves_validation_failure_without_opening_any_roots(tmp_path):
    from particle_filter.convergence import main

    evidence = tmp_path / "validation.json"
    evidence.write_text('{"status": "failed"}')
    output = tmp_path / "panel"
    with pytest.raises(ValueError, match="validated"):
        main(
            [
                "--validation",
                str(evidence),
                "--validation-sha256",
                hashlib.sha256(evidence.read_bytes()).hexdigest(),
                "--output",
                str(output),
                "--max-hours",
                "1",
                "--workers",
                "1",
            ]
        )
    report = json.loads((output / "convergence.json").read_text())
    assert report["status"] == "failed"
    assert "validated" in report["reason"]


def test_validation_receipt_verifies_sibling_evidence_after_relocation(tmp_path):
    from particle_filter.convergence import validate_evidence

    bundle = tmp_path / "relocated"
    teacher = bundle / "teacher" / "summary.json"
    teacher.parent.mkdir(parents=True)
    teacher.write_text('{"intervals":14400}')
    report = bundle / "report" / "validation.json"
    report.parent.mkdir()
    metadata = {
        "status": "validated",
        "evidence_digests": {
            "../teacher/summary.json": hashlib.sha256(teacher.read_bytes()).hexdigest()
        },
    }
    report.write_text(json.dumps(metadata))
    seal = hashlib.sha256(report.read_bytes()).hexdigest()
    assert validate_evidence(report, seal)["status"] == "validated"
    teacher.write_text("{}")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_evidence(report, seal)
    metadata["evidence_digests"] = {"../../outside.json": "0" * 64}
    report.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="bundle"):
        validate_evidence(report, hashlib.sha256(report.read_bytes()).hexdigest())
