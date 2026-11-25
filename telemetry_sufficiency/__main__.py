import argparse
from pathlib import Path
from .workflow import execute, STAGES


def main():
    parser = argparse.ArgumentParser(description="AIMS5701 telemetry sufficiency study")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        help="regenerate retained v1 audit or report in a separate directory",
    )
    args = parser.parse_args()
    if args.output is not None:
        from .archive import regenerate

        regenerate(args.stage, args.run.resolve(), args.output, args.workers)
        return
    execute(args.stage, args.run, args.workers)


if __name__ == "__main__":
    main()
