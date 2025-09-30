"""Command-line helper to profile LoRaFlexSim runs with ``cProfile``."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
from pathlib import Path

from loraflexsim import run as run_module


def profile(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile a LoRaFlexSim scenario using cProfile",
        epilog=(
            "Example:\n"
            "  python scripts/profile_simulation.py --output stats.prof -- "
            "--nodes 200 --steps 86400 --fast"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profile.stats"),
        help="Profiling statistics file (default: profile.stats)",
    )
    parser.add_argument(
        "--sort",
        default="cumulative",
        help="Sorting key passed to pstats.Stats.sort_stats",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=40,
        help="Number of lines to display in the console summary",
    )
    parser.add_argument(
        "simulation_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to loraflexsim.run (prefix with --)",
    )
    args = parser.parse_args(argv)

    forwarded = args.simulation_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        run_module.main(forwarded)
    finally:
        profiler.disable()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(args.output))

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats(args.sort)
    stats.print_stats(args.top)
    print(stream.getvalue())
    return 0


def main() -> int:  # pragma: no cover - thin CLI wrapper
    return profile()


if __name__ == "__main__":  # pragma: no cover - manual execution
    raise SystemExit(main())

