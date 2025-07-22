import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.simulator import Simulator  # noqa: E402
from VERSION_4.launcher.adr_standard_1 import apply as adr1  # noqa: E402
from VERSION_4.launcher.compare_flora import compare_with_sim  # noqa: E402

CONFIG_PATH = ROOT / "examples" / "flora_scalability.ini"


def run_simulation(runs: int, seed: int | None = None, flora_csv: str | None = None):
    metrics = []
    for i in range(runs):
        sim = Simulator(
            config_file=str(CONFIG_PATH),
            flora_mode=True,
            adr_node=True,
            adr_server=True,
            seed=(seed + i) if seed is not None else None,
        )
        adr1(sim)
        sim.run()
        m = sim.get_metrics()
        metrics.append(m)
        print(f"Run {i+1}/{runs} PDR: {m['PDR']:.2%}")
        if flora_csv:
            try:
                match = compare_with_sim(m, flora_csv)
            except RuntimeError as exc:
                print(f"Cannot compare with FLoRa data: {exc}")
                match = False
            status = "matches" if match else "differs from"
            print(f"-> Metrics {status} FLoRa reference")
    avg: dict[str, float] = {}
    for k, v in metrics[0].items():
        if isinstance(v, (int, float)):
            avg[k] = sum(m[k] for m in metrics) / len(metrics)
    print("Average PDR:", avg.get("PDR", 0.0))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scalability scenario from FLoRa article")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--seed", type=int, help="Base random seed")
    parser.add_argument("--flora-csv", type=str, help="Reference FLoRa CSV")
    args = parser.parse_args()
    run_simulation(args.runs, args.seed, args.flora_csv)


if __name__ == "__main__":
    main()
