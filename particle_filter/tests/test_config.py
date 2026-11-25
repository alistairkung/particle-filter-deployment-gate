import pytest

from particle_filter.config import load_variant, read_variant


@pytest.mark.parametrize(
    "name,half_life,ablate",
    [
        ("full", 120, False),
        ("regime-conditional-work", 120, True),
        ("half-life-60", 60, False),
        ("half-life-240", 240, False),
    ],
)
def test_reviewed_variants_select_only_the_declared_difference(name, half_life, ablate):
    variant = load_variant(name)
    assert (variant.half_life, variant.replace_cut_work) == (half_life, ablate)


def test_reject_a_modified_primary_half_life(tmp_path):
    candidate = tmp_path / "full.toml"
    candidate.write_text(
        'name = "full"\nhalf_life_seconds = 30\ncut_work = "posterior"\n'
    )
    with pytest.raises(ValueError, match="reviewed"):
        read_variant(candidate)
