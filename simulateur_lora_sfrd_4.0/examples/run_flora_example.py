import argparse
from VERSION_4.launcher.simulator import Simulator
from VERSION_4.launcher.adr_standard_1 import apply as adr1

NODE_POSITIONS = [
    (450.45, 490.0),
    (555.0, 563.0),
    (460.0, 467.0),
    (565.0, 571.0),
    (702.0, 578.0),
]
GW_POSITION = (500.0, 500.0)


def run_simulation(runs: int, seed: int | None = None):
    metrics = []
    for i in range(runs):
        sim = Simulator(
            num_nodes=5,
            num_gateways=1,
            area_size=1000.0,
            transmission_mode="Random",
            packet_interval=100.0,
            packets_to_send=5 * 80,
            adr_node=True,
            adr_server=True,
            mobility=False,
            fixed_sf=12,
            fixed_tx_power=14.0,
            detection_threshold_dBm=-110.0,
            min_interference_time=5.0,
            seed=(seed + i) if seed is not None else None,
        )
        # apply ADR 1 settings
        adr1(sim)
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
    args = parser.parse_args()
    run_simulation(args.runs, args.seed)


if __name__ == "__main__":
    main()
