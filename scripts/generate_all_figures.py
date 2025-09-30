"""Generate all figures by running predefined simulation scripts.

This utility sequentially executes several simulation and plotting scripts to
reproduce the figures shipped with the project. Parameters such as the number
of nodes, packets per node and the random seed can be provided either on the
command line or via a configuration file.

Examples
--------
Run with explicit arguments::

    python scripts/generate_all_figures.py --nodes 50 --packets 100 --seed 1

Run using a configuration file::

    python scripts/generate_all_figures.py --config my_config.ini

The configuration file should contain a section ``[simulation]`` with keys
``nodes``, ``packets`` and ``seed``.
"""

from __future__ import annotations

import argparse
import configparser
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = ROOT_DIR / "results"


DEFAULTS = {"nodes": 50, "packets": 100, "seed": 1, "area_size": 1000.0}


def load_config(path: str | None) -> dict:
    params = DEFAULTS.copy()
    if path:
        config = configparser.ConfigParser()
        config.read(path)
        if "simulation" in config:
            for key in params:
                if key in config["simulation"]:
                    raw = config["simulation"][key]
                    params[key] = float(raw) if key == "area_size" else int(raw)
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--nodes", type=int, help="Number of nodes")
    parser.add_argument("--packets", type=int, help="Packets per node")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--area-size", type=float, help="Area size for node position plot")
    args = parser.parse_args()

    params = load_config(args.config)
    for key in ("nodes", "packets", "seed", "area_size"):
        value = getattr(args, key.replace("-", "_"), None)
        if value is not None:
            params[key] = value

    python = sys.executable

    subprocess.run(
        [
            python,
            str(SCRIPT_DIR / "run_mobility_multichannel.py"),
            "--nodes",
            str(params["nodes"]),
            "--packets",
            str(params["packets"]),
            "--seed",
            str(params["seed"]),
        ],
        check=True,
    )

    subprocess.run(
        [
            python,
            str(SCRIPT_DIR / "plot_mobility_multichannel.py"),
            str(RESULTS_DIR / "mobility_multichannel.csv"),
            "--formats",
            "png",
            "jpg",
            "svg",
            "eps",
        ],
        check=True,
    )

    subprocess.run(
        [
            python,
            str(SCRIPT_DIR / "run_mobility_latency_energy.py"),
            "--nodes",
            str(params["nodes"]),
            "--packets",
            str(params["packets"]),
            "--seed",
            str(params["seed"]),
        ],
        check=True,
    )

    subprocess.run(
        [
            python,
            str(SCRIPT_DIR / "plot_mobility_latency_energy.py"),
            str(RESULTS_DIR / "mobility_latency_energy.csv"),
        ],
        check=True,
    )

    subprocess.run(
        [
            python,
            str(SCRIPT_DIR / "plot_sf_vs_scenario.py"),
            str(RESULTS_DIR / "mobility_latency_energy.csv"),
        ],
        check=True,
    )

    subprocess.run(
        [
            python,
            str(SCRIPT_DIR / "run_mobility_models.py"),
            "--nodes",
            str(params["nodes"]),
            "--packets",
            str(params["packets"]),
            "--seed",
            str(params["seed"]),
        ],
        check=True,
    )

    subprocess.run(
        [
            python,
            str(SCRIPT_DIR / "plot_mobility_models.py"),
            str(RESULTS_DIR / "mobility_models.csv"),
        ],
        check=True,
    )

    subprocess.run(
        [
            python,
            str(SCRIPT_DIR / "plot_sf_vs_scenario.py"),
            "--by-model",
            str(RESULTS_DIR / "mobility_models.csv"),
        ],
        check=True,
    )

    subprocess.run(
        [
            python,
            str(SCRIPT_DIR / "run_battery_tracking.py"),
            "--nodes",
            str(params["nodes"]),
            "--packets",
            str(params["packets"]),
            "--seed",
            str(params["seed"]),
        ],
        check=True,
    )

    subprocess.run(
        [python, str(SCRIPT_DIR / "plot_battery_tracking.py")],
        check=True,
    )

    subprocess.run(
        [
            python,
            str(SCRIPT_DIR / "plot_node_positions.py"),
            "--num-nodes",
            str(params["nodes"]),
            "--area-size",
            str(params["area_size"]),
            "--seed",
            str(params["seed"]),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
