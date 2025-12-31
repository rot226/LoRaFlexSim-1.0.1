"""Campagne SNIR balayant la fenêtre d'interférence utilisée."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loraflexsim.launcher import MultiChannel, Simulator
from loraflexsim.launcher.non_orth_delta import DEFAULT_NON_ORTH_DELTA

FREQUENCIES_HZ = (
    868_100_000.0,
    868_300_000.0,
    868_500_000.0,
    867_100_000.0,
    867_300_000.0,
    867_500_000.0,
    867_700_000.0,
    867_900_000.0,
)

PACKET_INTERVAL_S = 900.0
PACKETS_PER_NODE = 10
PAYLOAD_BYTES = 20
NUM_NODES = 6000
WINDOW_MODES: tuple[tuple[str, str], ...] = (
    ("packet", "packet"),
    ("preamble", "preamble"),
    ("symbol", "symbol"),
)
OUTPUT_PATH = ROOT_DIR / "data" / "snir_window_cdf.csv"


def _build_multichannel(window_mode: str) -> MultiChannel:
    multichannel = MultiChannel(FREQUENCIES_HZ)
    multichannel.force_non_orthogonal(DEFAULT_NON_ORTH_DELTA)
    for channel in multichannel.channels:
        channel.use_snir = True
        channel.snir_window = window_mode
    return multichannel


def _snir_cdf(values: Iterable[float], attempts: int) -> list[tuple[float, float]]:
    filtered = [v for v in values if v is not None and math.isfinite(v)]
    if attempts <= 0 or not filtered:
        return []
    minimum = math.floor(min(filtered))
    maximum = math.ceil(max(filtered))
    if minimum == maximum:
        return [(float(minimum), len(filtered) / attempts)]
    bin_width = 1.0
    bin_edges = [minimum + i * bin_width for i in range(int((maximum - minimum) / bin_width) + 1)]
    bin_edges.append(maximum)
    counts = [0 for _ in range(len(bin_edges) - 1)]
    for value in filtered:
        index = min(int((value - minimum) / bin_width), len(counts) - 1)
        counts[index] += 1
    cdf: list[tuple[float, float]] = []
    cumulative = 0
    for edge, count in zip(bin_edges, counts):
        cumulative += count
        cdf.append((float(edge), cumulative / attempts))
    return cdf


def _global_stats(simulator: Simulator) -> dict[str, float]:
    sent = sum(node.packets_sent for node in simulator.nodes)
    attempts = sum(node.tx_attempted for node in simulator.nodes)
    delivered = sum(node.rx_delivered for node in simulator.nodes)
    collisions = sum(node.packets_collision for node in simulator.nodes)
    der = delivered / sent if sent else 0.0
    pdr = delivered / attempts if attempts else 0.0
    return {
        "sent": float(sent),
        "attempts": float(attempts),
        "delivered": float(delivered),
        "collisions": float(collisions),
        "der": der,
        "pdr": pdr,
    }


def _run_single(
    window_mode: str,
    *,
    num_nodes: int,
    packets_per_node: int,
    packet_interval: float,
    payload_bytes: int,
) -> list[dict[str, object]]:
    multichannel = _build_multichannel(window_mode)
    simulator = Simulator(
        num_nodes=num_nodes,
        num_gateways=1,
        area_size=5000.0,
        transmission_mode="Random",
        packet_interval=packet_interval,
        first_packet_interval=packet_interval,
        packets_to_send=packets_per_node,
        duty_cycle=0.01,
        mobility=False,
        channels=multichannel,
        channel_distribution="round-robin",
        payload_size_bytes=payload_bytes,
        flora_mode=True,
        seed=1,
    )
    simulator.run()

    stats = _global_stats(simulator)
    events = list(getattr(simulator, "events_log", []) or [])
    snir_values = [entry.get("snir_dB") for entry in events if "snir_dB" in entry]
    cdf = _snir_cdf(snir_values, int(stats["sent"]))

    rows: list[dict[str, object]] = []
    for snir_db, probability in cdf:
        rows.append(
            {
                "window": window_mode,
                "snir_db": snir_db,
                "cdf": probability,
                "der": stats["der"],
                "pdr": stats["pdr"],
                "sent": int(stats["sent"]),
                "delivered": int(stats["delivered"]),
                "collisions": int(stats["collisions"]),
            }
        )
    return rows


def run_campaign(
    *,
    window_modes: Iterable[tuple[str, str]] = WINDOW_MODES,
    num_nodes: int = NUM_NODES,
    packets_per_node: int = PACKETS_PER_NODE,
    packet_interval: float = PACKET_INTERVAL_S,
    payload_bytes: int = PAYLOAD_BYTES,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode, label in window_modes:
        for row in _run_single(
            mode,
            num_nodes=num_nodes,
            packets_per_node=packets_per_node,
            packet_interval=packet_interval,
            payload_bytes=payload_bytes,
        ):
            row["window_label"] = label
            rows.append(row)
    return rows


def write_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "window",
                "window_label",
                "snir_db",
                "cdf",
                "der",
                "pdr",
                "sent",
                "delivered",
                "collisions",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_window_modes(raw: str | None) -> list[tuple[str, str]]:
    if not raw:
        return list(WINDOW_MODES)
    modes: list[tuple[str, str]] = []
    for mode in raw.split(","):
        mode = mode.strip()
        if mode:
            modes.append((mode, mode))
    if not modes:
        raise ValueError("Les fenêtres SNIR doivent contenir au moins un mode valide.")
    return modes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-modes",
        help="Modes de fenêtre SNIR séparés par des virgules (ex: packet,preamble).",
    )
    parser.add_argument(
        "--quick-windows",
        action="store_true",
        help="Exécute rapidement les deux fenêtres principales (preamble vs packet).",
    )
    parser.add_argument("--num-nodes", type=int, default=NUM_NODES)
    parser.add_argument("--packets-per-node", type=int, default=PACKETS_PER_NODE)
    parser.add_argument("--packet-interval", type=float, default=PACKET_INTERVAL_S)
    parser.add_argument("--payload-bytes", type=int, default=PAYLOAD_BYTES)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.quick_windows:
        window_modes = [("packet", "packet"), ("preamble", "preamble")]
    else:
        window_modes = _parse_window_modes(args.window_modes)
    dataset = run_campaign(
        window_modes=window_modes,
        num_nodes=args.num_nodes,
        packets_per_node=args.packets_per_node,
        packet_interval=args.packet_interval,
        payload_bytes=args.payload_bytes,
    )
    write_csv(dataset, args.output)
    print(f"Résultats enregistrés dans {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
