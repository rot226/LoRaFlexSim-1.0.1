"""Simulation de l'étape 1 avec proxys ADR/MixRA.

Les implémentations ci-dessous fournissent des heuristiques simples et
reproductibles pour ADR, MixRA-H et MixRA-Opt. Elles servent de substituts
cohérents lorsque les formules exactes ne sont pas disponibles.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

from article_c.common.metrics import packet_delivery_ratio
from article_c.common.utils import assign_clusters, generate_traffic_times

SF_VALUES = [7, 8, 9, 10, 11, 12]
SNR_THRESHOLDS = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}
RSSI_THRESHOLDS = {7: -123.0, 8: -126.0, 9: -129.0, 10: -132.0, 11: -134.0, 12: -137.0}


@dataclass
class NodeLink:
    snr: float
    rssi: float


SF_VALUES = (7, 8, 9, 10, 11, 12)

# Seuils proxy pour SNR/RSSI (inspirés d'ordres de grandeur LoRaWAN).
SNR_THRESHOLDS = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}
RSSI_THRESHOLDS = {7: -123.0, 8: -126.0, 9: -129.0, 10: -132.0, 11: -134.5, 12: -137.0}


@dataclass(frozen=True)
class NodeLink:
    snr: float
    rssi: float


@dataclass
class Step1Result:
    sent: int
    received: int
    node_clusters: list[str]
    node_received: list[bool]

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


def _estimate_received(assignments: Iterable[int], rng: random.Random) -> list[bool]:
    """Approxime les collisions et simule les succès par nœud."""
    loads = {sf: 0 for sf in SF_VALUES}
    assignments_list = list(assignments)
    for sf in assignments_list:
        loads[sf] += 1
    capacity_per_sf = 25
    success_prob_by_sf: dict[int, float] = {}
    for sf, load in loads.items():
        if load <= 0:
            success_prob_by_sf[sf] = 0.0
            continue
        if load <= capacity_per_sf:
            delivered = load
        else:
            delivered = capacity_per_sf + (load - capacity_per_sf) * 0.5
        success_prob_by_sf[sf] = max(0.0, min(1.0, delivered / load))
    return [rng.random() < success_prob_by_sf[sf] for sf in assignments_list]


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
    *,
    duration_s: float = 3600.0,
    traffic_mode: str = "periodic",
    jitter_range_s: float = 5.0,
) -> Step1Result:
    """Exécute une simulation minimale.

    Les résultats reposent sur des proxys ADR/MixRA et ne remplacent pas
    l'implémentation complète des algorithmes.
    """
    rng = random.Random(seed)
    traffic_times = generate_traffic_times(
        sent,
        duration_s=duration_s,
        traffic_mode=traffic_mode,
        jitter_range_s=jitter_range_s,
        rng=rng,
    )
    actual_sent = len(traffic_times)
    nodes = _generate_nodes(actual_sent, seed)
    if algorithm == "adr":
        assignments = [_adr_smallest_sf(node) for node in nodes]
    elif algorithm == "mixra_h":
        assignments = _mixra_h_assign(nodes)
    elif algorithm == "mixra_opt":
        assignments = _mixra_opt_assign(nodes)
    else:
        raise ValueError(f"Algorithme inconnu: {algorithm}")
    node_received = _estimate_received(assignments, rng)
    node_clusters = assign_clusters(actual_sent, rng=rng)
    received = sum(1 for value in node_received if value)
    return Step1Result(
        sent=actual_sent,
        received=received,
        node_clusters=node_clusters,
        node_received=node_received,
    )
