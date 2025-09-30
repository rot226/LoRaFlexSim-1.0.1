"""Comparer les métriques RSSI/SNR LoRaFlexSim avec une trace FLoRa."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.compare_flora import load_flora_rx_stats


PATH_LOSS_PRESETS = {
    "loralognormalshadowing": ("flora", "lognorm"),
    "lorapathlossoulu": ("flora_oulu", "oulu"),
    "lorapathlosshata": ("flora_hata", "hata"),
}


@dataclass
class NodeConfig:
    """Configuration minimale d'un nœud FLoRa."""

    index: int
    x: float
    y: float
    sf: int | None
    tx_power: float | None


@dataclass
class GatewayConfig:
    """Position d'une passerelle FLoRa."""

    index: int
    x: float
    y: float


def _parse_numeric(value: str) -> float:
    """Convertit une valeur FLoRa en float (gère m, kHz, dBm)."""

    compact = value.replace(" ", "").strip()
    if compact.endswith("dBm"):
        return float(compact[:-3])
    if compact.endswith("kHz"):
        return float(compact[:-3]) * 1e3
    if compact.endswith("MHz"):
        return float(compact[:-3]) * 1e6
    if compact.endswith("m"):
        return float(compact[:-1])
    return float(compact)


def _parse_nodes_and_gateways(path: Path) -> tuple[list[NodeConfig], list[GatewayConfig], str, str, float]:
    """Extrait les entités pertinentes d'un INI FLoRa."""

    nodes: dict[int, dict[str, float | int | None]] = {}
    gateways: dict[int, dict[str, float]] = {}
    defaults = {"sf": None, "tx_power": None, "bandwidth": 125e3}
    environment = "flora"
    flora_loss_model = "lognorm"

    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, raw_value = (part.strip() for part in line.split("=", 1))
            value = raw_value.strip().strip('"')

            if key.startswith("**.LoRaMedium.pathLossType"):
                preset = PATH_LOSS_PRESETS.get(value.lower())
                if preset:
                    environment, flora_loss_model = preset
                continue

            if key == "**.sigma":
                # Le shadowing est neutralisé par défaut pour stabiliser la comparaison.
                continue

            if key.startswith("**.loRaNodes["):
                idx_part, rest = key[len("**.loRaNodes[") :].split("]", 1)
                attr = rest.split(".")[-1].lstrip("*")
                target: dict[str, float | int | None]
                if idx_part == "*":
                    target = defaults
                else:
                    index = int(idx_part)
                    target = nodes.setdefault(index, {})
                if attr in {"initialX", "initialY", "initialLoRaTP", "initialLoRaSF", "initialLoRaBW"}:
                    if attr in {"initialLoRaSF"}:
                        target["sf"] = int(_parse_numeric(value))
                    elif attr == "initialLoRaTP":
                        target["tx_power"] = _parse_numeric(value)
                    elif attr == "initialLoRaBW":
                        defaults["bandwidth"] = _parse_numeric(value)
                    elif attr == "initialX":
                        target["x"] = _parse_numeric(value)
                    elif attr == "initialY":
                        target["y"] = _parse_numeric(value)
                continue

            if key.startswith("**.loRaGW["):
                idx_part, rest = key[len("**.loRaGW[") :].split("]", 1)
                index = int(idx_part)
                attr = rest.split(".")[-1].lstrip("*")
                target_gw = gateways.setdefault(index, {})
                if attr in {"initialX", "initialY"}:
                    target_gw["x" if attr == "initialX" else "y"] = _parse_numeric(value)

    node_cfg = [
        NodeConfig(
            index=i,
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            sf=int(data.get("sf")) if data.get("sf") is not None else None,
            tx_power=float(data.get("tx_power")) if data.get("tx_power") is not None else None,
        )
        for i, data in sorted(nodes.items())
    ]
    gateway_cfg = [
        GatewayConfig(
            index=i,
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
        )
        for i, data in sorted(gateways.items())
    ]

    bandwidth = float(defaults["bandwidth"])
    return node_cfg, gateway_cfg, environment, flora_loss_model, bandwidth


def _compute_expected_metrics(
    channel: Channel,
    nodes: Iterable[NodeConfig],
    gateways: Iterable[GatewayConfig],
) -> tuple[float, float]:
    """Calcule les RSSI/SNR moyens par rapport aux passerelles les plus proches."""

    gw_positions = [(gw.x, gw.y) for gw in gateways]
    if not gw_positions:
        raise RuntimeError("Aucune passerelle définie dans l'INI FLoRa")

    samples: list[tuple[float, float]] = []
    for node in nodes:
        tx_power = node.tx_power if node.tx_power is not None else 14.0
        sf = node.sf
        if sf is None:
            # Lorsque le SF initial n'est pas précisé on ignore le nœud.
            continue
        best_rssi = None
        best_pair: tuple[float, float] | None = None
        for gw_x, gw_y in gw_positions:
            distance = math.hypot(node.x - gw_x, node.y - gw_y)
            rssi, snr = channel.compute_rssi(tx_power, distance, sf=sf)
            if best_rssi is None or rssi > best_rssi:
                best_rssi = rssi
                best_pair = (rssi, snr)
        if best_pair is not None:
            samples.append(best_pair)

    if not samples:
        raise RuntimeError("Aucun RSSI/SNR calculé : vérifiez le contenu de l'INI")

    avg_rssi = sum(val[0] for val in samples) / len(samples)
    avg_snr = sum(val[1] for val in samples) / len(samples)
    return avg_rssi, avg_snr


def build_channel(
    environment: str,
    loss_model: str,
    bandwidth: float,
    *,
    seed: int,
    keep_shadowing: bool,
) -> Channel:
    """Instancie un canal LoRaFlexSim paramétré pour la comparaison FLoRa."""

    channel = Channel(
        environment=environment,
        phy_model="flora_full",
        flora_loss_model=loss_model,
        use_flora_curves=True,
        shadowing_std=0.0,
        bandwidth=bandwidth,
        rng=np.random.default_rng(seed),
    )
    if keep_shadowing and channel.environment and channel.environment in Channel.ENV_PRESETS:
        _, sigma, _, _ = Channel.ENV_PRESETS[channel.environment]
        channel.shadowing_std = sigma
    return channel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ini", required=True, type=Path, help="Fichier INI FLoRa à analyser")
    parser.add_argument("--sca", required=True, type=Path, help="Trace FLoRa (.sca ou dossier) à comparer")
    parser.add_argument("--rssi-tol", type=float, default=0.5, help="Tolérance absolue sur le RSSI (dB)")
    parser.add_argument("--snr-tol", type=float, default=0.5, help="Tolérance absolue sur le SNR (dB)")
    parser.add_argument("--seed", type=int, default=0, help="Graine utilisée pour neutraliser les variations internes")
    parser.add_argument(
        "--keep-shadowing",
        action="store_true",
        help="Conserver le shadowing FLoRa (désactivé par défaut pour une comparaison déterministe)",
    )
    args = parser.parse_args(argv)

    nodes, gateways, environment, loss_model, bandwidth = _parse_nodes_and_gateways(args.ini)
    if not nodes:
        raise RuntimeError("Aucun nœud extrait de l'INI FLoRa")

    flora_stats = load_flora_rx_stats(args.sca)

    channel = build_channel(
        environment=environment,
        loss_model=loss_model,
        bandwidth=bandwidth,
        seed=args.seed,
        keep_shadowing=args.keep_shadowing,
    )

    if not args.keep_shadowing:
        channel.shadowing_std = 0.0

    sim_rssi, sim_snr = _compute_expected_metrics(channel, nodes, gateways)
    delta_rssi = abs(sim_rssi - flora_stats["rssi"])
    delta_snr = abs(sim_snr - flora_stats["snr"])

    print(f"FLoRa RSSI moyen : {flora_stats['rssi']:.2f} dBm")
    print(f"LoRaFlexSim RSSI : {sim_rssi:.2f} dBm (Δ={delta_rssi:.2f} dB)")
    print(f"FLoRa SNR moyen : {flora_stats['snr']:.2f} dB")
    print(f"LoRaFlexSim SNR : {sim_snr:.2f} dB (Δ={delta_snr:.2f} dB)")

    ok = delta_rssi <= args.rssi_tol and delta_snr <= args.snr_tol
    print("Statut :", "✅ conforme" if ok else "❌ hors tolérance")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

