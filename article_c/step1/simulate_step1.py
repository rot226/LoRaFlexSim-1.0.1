"""Simulation de l'étape 1 (placeholder)."""

from dataclasses import dataclass
import random
from typing import Iterable

from article_c.common.metrics import packet_delivery_ratio

SF_VALUES = [7, 8, 9, 10, 11, 12]
SNR_THRESHOLDS = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}
RSSI_THRESHOLDS = {7: -123.0, 8: -126.0, 9: -129.0, 10: -132.0, 11: -134.0, 12: -137.0}


@dataclass
class NodeLink:
    snr: float
    rssi: float


@dataclass
class Step1Result:
    sent: int
    received: int

    @property
    def pdr(self) -> float:
        return packet_delivery_ratio(self.received, self.sent)


def _qos_ok(node: NodeLink, sf: int) -> bool:
    return node.snr >= SNR_THRESHOLDS[sf] and node.rssi >= RSSI_THRESHOLDS[sf]


def _adr_smallest_sf(node: NodeLink) -> int:
    """ADR proxy: choisit le plus petit SF satisfaisant les seuils SNR/RSSI."""
    for sf in SF_VALUES:
        if _qos_ok(node, sf):
            return sf
    return SF_VALUES[-1]


def _mixra_h_assign(nodes: Iterable[NodeLink]) -> list[int]:
    """MixRA-H proxy: équilibre la QoS (marge SNR/RSSI) et la charge par SF."""
    assignments: list[int] = []
    loads = {sf: 0 for sf in SF_VALUES}
    for node in nodes:
        candidates = [sf for sf in SF_VALUES if _qos_ok(node, sf)]
        if not candidates:
            sf = SF_VALUES[-1]
            assignments.append(sf)
            loads[sf] += 1
            continue

        best_sf = candidates[0]
        best_score = float("inf")
        for sf in candidates:
            snr_margin = node.snr - SNR_THRESHOLDS[sf]
            rssi_margin = node.rssi - RSSI_THRESHOLDS[sf]
            qos_margin = min(snr_margin, rssi_margin)
            load_penalty = loads[sf]
            score = load_penalty - 0.35 * qos_margin
            if score < best_score:
                best_score = score
                best_sf = sf
        assignments.append(best_sf)
        loads[best_sf] += 1
    return assignments


def _mixra_opt_assign(nodes: Iterable[NodeLink]) -> list[int]:
    """MixRA-Opt proxy: glouton + recherche locale pour réduire les collisions."""
    nodes_list = list(nodes)
    assignments = _mixra_h_assign(nodes_list)

    def objective(loads: dict[int, int]) -> int:
        return sum(load * load for load in loads.values())

    loads = {sf: 0 for sf in SF_VALUES}
    for sf in assignments:
        loads[sf] += 1

    improved = True
    while improved:
        improved = False
        for idx, node in enumerate(nodes_list):
            current_sf = assignments[idx]
            candidates = [sf for sf in SF_VALUES if _qos_ok(node, sf)]
            if not candidates:
                continue
            best_sf = current_sf
            best_obj = objective(loads)
            for sf in candidates:
                if sf == current_sf:
                    continue
                loads[current_sf] -= 1
                loads[sf] += 1
                candidate_obj = objective(loads)
                loads[sf] -= 1
                loads[current_sf] += 1
                if candidate_obj < best_obj:
                    best_obj = candidate_obj
                    best_sf = sf
            if best_sf != current_sf:
                loads[current_sf] -= 1
                loads[best_sf] += 1
                assignments[idx] = best_sf
                improved = True
    return assignments


def _estimate_received(assignments: Iterable[int]) -> int:
    """Approxime les collisions en pénalisant les SF surchargés."""
    loads = {sf: 0 for sf in SF_VALUES}
    for sf in assignments:
        loads[sf] += 1
    capacity_per_sf = 25
    delivered = 0
    for sf, load in loads.items():
        if load <= capacity_per_sf:
            delivered += load
        else:
            delivered += int(capacity_per_sf + (load - capacity_per_sf) * 0.5)
    return delivered


def _generate_nodes(count: int, seed: int) -> list[NodeLink]:
    rng = random.Random(seed)
    nodes: list[NodeLink] = []
    for _ in range(count):
        snr = rng.uniform(-22.0, 5.0)
        rssi = rng.uniform(-140.0, -110.0)
        nodes.append(NodeLink(snr=snr, rssi=rssi))
    return nodes


def run_simulation(
    sent: int = 120,
    algorithm: str = "adr",
    seed: int = 42,
) -> Step1Result:
    """Exécute une simulation minimale.

    Les résultats reposent sur des proxys ADR/MixRA et ne remplacent pas
    l'implémentation complète des algorithmes.
    """
    nodes = _generate_nodes(sent, seed)
    if algorithm == "adr":
        assignments = [_adr_smallest_sf(node) for node in nodes]
    elif algorithm == "mixra_h":
        assignments = _mixra_h_assign(nodes)
    elif algorithm == "mixra_opt":
        assignments = _mixra_opt_assign(nodes)
    else:
        raise ValueError(f"Algorithme inconnu: {algorithm}")
    received = _estimate_received(assignments)
    return Step1Result(sent=sent, received=received)
