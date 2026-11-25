"""Named, reviewed experimental variants of the same forecast pipeline."""

from dataclasses import dataclass
from pathlib import Path
import tomllib

CONFIGS = Path(__file__).with_name("configs")
REVIEWED = {
    "full": (120, "posterior"),
    "regime-conditional-work": (120, "development-given-regime"),
    "half-life-60": (60, "posterior"),
    "half-life-240": (240, "posterior"),
}


@dataclass(frozen=True)
class Variant:
    name: str
    half_life: int
    replace_cut_work: bool


def read_variant(path):
    with path.open("rb") as stream:
        settings = tomllib.load(stream)
    if set(settings) != {"name", "half_life_seconds", "cut_work"}:
        raise ValueError("variant must name its reviewed half-life and cut-work source")
    name, half, source = (
        settings["name"],
        settings["half_life_seconds"],
        settings["cut_work"],
    )
    if (
        type(name) is not str
        or type(half) is not int
        or REVIEWED.get(name) != (half, source)
    ):
        raise ValueError("configuration differs from the reviewed experimental variant")
    return Variant(name, half, source == "development-given-regime")


def load_variant(name):
    if name not in REVIEWED:
        raise ValueError(f"unknown experimental variant: {name}")
    result = read_variant(CONFIGS / f"{name}.toml")
    if result.name != name:
        raise ValueError("configuration filename and variant disagree")
    return result
