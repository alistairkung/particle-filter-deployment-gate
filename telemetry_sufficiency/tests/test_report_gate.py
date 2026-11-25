import pytest
from telemetry_sufficiency import custody
from telemetry_sufficiency.report import verified_verdict


def test_readiness_requires_two_valid_claims_and_external_receipts(tmp_path):
    scores = {"claims": {"demand": {"passes": True}, "increment": {"passes": False}}}
    custody.save(tmp_path / "scores.json", scores)
    custody.save(
        tmp_path / "independent-audit.json",
        {
            "verified": True,
            "protected_roots": 1200,
            "verdict": "READY WITH LIMITATIONS",
        },
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        verified_verdict(tmp_path)
