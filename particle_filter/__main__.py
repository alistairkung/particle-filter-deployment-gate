"""Run the AIMS5701 development experiment from the repository root."""

import argparse
from pathlib import Path

from .config import REVIEWED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="new directory for development evidence"
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=REVIEWED,
        help="repeat to compare variants (default: all four)",
    )
    parser.add_argument(
        "--particles", type=int, required=True, help="development particle count"
    )
    parser.add_argument(
        "--pairs", type=int, required=True, help="matched forecast pairs per posterior"
    )
    parser.add_argument("--seed", type=int, default=570112)
    args = parser.parse_args()
    from .experiment import run_experiment

    run_experiment(
        args.output,
        args.variant or list(REVIEWED),
        particles=args.particles,
        pairs=args.pairs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
