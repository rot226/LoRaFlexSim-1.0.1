import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.simulator import Simulator  # noqa: E402
from VERSION_4.launcher.adr_standard_1 import apply as adr1  # noqa: E402
from VERSION_4.launcher.compare_flora import compare_with_sim  # noqa: E402

NODE_POSITIONS = [
    (450.45, 490.0),
    (555.0, 563.0),
    (460.0, 467.0),
    (565.0, 571.0),
    (702.0, 578.0),
]
GW_POSITION = (500.0, 500.0)


def run_simulation(runs: int, seed: int | None = None, flora_csv: str | None = None, degrade: bool = False):
    metrics = []
    for i in range(runs):
        sim = Simulator(
            num_nodes=5,
            num_gateways=1,
            area_size=1000.0,
            transmission_mode="Random",
            packet_interval=100.0,
            packets_to_send=80,
            adr_node=True,
            adr_server=True,
            mobility=False,
            fixed_sf=12,
            fixed_tx_power=14.0,
            flora_mode=True,
            seed=(seed + i) if seed is not None else None,
        )
        # apply ADR 1 settings
        adr1(sim, degrade_channel=degrade)
        # override positions
        gw = sim.gateways[0]
        gw.x, gw.y = GW_POSITION
        for node, pos in zip(sim.nodes, NODE_POSITIONS):
            node.x, node.y = pos
            node.initial_x, node.initial_y = pos
        sim.run()
        m = sim.get_metrics()
        metrics.append(m)
        print(f"Run {i+1}/{runs} PDR: {m['PDR']:.2%}")
        print("SF distribution:", m['sf_distribution'])
        if flora_csv:
            try:
                match = compare_with_sim(m, flora_csv)
            except RuntimeError as exc:
                print(f"Cannot compare with FLoRa data: {exc}")
                match = False
            status = "matches" if match else "differs from"
            print(f"-> Metrics {status} FLoRa reference")
    # compute averages
    avg: dict[str, float] = {}
    for k, v in metrics[0].items():
        if isinstance(v, (int, float)):
            avg[k] = sum(m[k] for m in metrics) / len(metrics)
    print("Average PDR:", avg.get("PDR", 0.0))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run FLoRa-like scenario")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--seed", type=int, help="Base random seed")
    parser.add_argument("--flora-csv", type=str, help="Path to reference FLoRa CSV")
    parser.add_argument("--degrade", action="store_true", help="Apply harsh channel settings")
    args = parser.parse_args()
    run_simulation(args.runs, args.seed, args.flora_csv, args.degrade)


if __name__ == "__main__":
    main()
