"""Utilitaires pour agréger les métriques du mini banc QoS clusters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


Number = float


@dataclass
class RunMetrics:
    """Métriques résumées pour un couple (scénario, algorithme)."""

    scenario: str
    algorithm: str
    num_nodes: int
    period_s: float
    pdr_global: float
    der_global: float
    collisions: int
    cluster_pdr: Dict[str, float]
    cluster_targets: Dict[str, float]
    cluster_gaps: Dict[str, float]
    snir_values: List[float]
    snir_cdf: List[Tuple[float, float]]
    mixra_solver: str | None

    def to_csv_row(self) -> Dict[str, object]:
        """Aplati les métriques pour écriture CSV."""

        row: Dict[str, object] = {
            "scenario": self.scenario,
            "algorithm": self.algorithm,
            "num_nodes": self.num_nodes,
            "period_s": self.period_s,
            "pdr_global": self.pdr_global,
            "der_global": self.der_global,
            "collisions": self.collisions,
            "mixra_solver": self.mixra_solver or "",
            "cluster_pdr_json": json.dumps(self.cluster_pdr, ensure_ascii=False, sort_keys=True),
            "cluster_targets_json": json.dumps(
                self.cluster_targets, ensure_ascii=False, sort_keys=True
            ),
            "cluster_gaps_json": json.dumps(self.cluster_gaps, ensure_ascii=False, sort_keys=True),
            "snir_samples": len(self.snir_values),
            "snir_cdf_json": json.dumps(self.snir_cdf, ensure_ascii=False),
        }
        return row


def _compute_cdf(values: Sequence[Number]) -> List[Tuple[float, float]]:
    """Calcule la CDF empirique d'une séquence de valeurs."""

    if not values:
        return []
    sorted_values = sorted(float(v) for v in values)
    total = len(sorted_values)
    cdf: List[Tuple[float, float]] = []
    for index, value in enumerate(sorted_values, start=1):
        cdf.append((value, index / total))
    return cdf


def _format_cluster_mapping(mapping: Mapping[int | str, Number]) -> Dict[str, float]:
    formatted: Dict[str, float] = {}
    for key, value in mapping.items():
        formatted[str(key)] = float(value)
    return formatted


def compute_run_metrics(
    *,
    scenario: str,
    algorithm: str,
    base_metrics: Mapping[str, object],
    events: Iterable[Mapping[str, object]],
) -> RunMetrics:
    """Construit la structure :class:`RunMetrics` à partir du simulateur."""

    pdr_global = float(base_metrics.get("PDR", 0.0) or 0.0)
    delivered = float(base_metrics.get("delivered", 0.0) or 0.0)
    attempted = float(base_metrics.get("tx_attempted", 0.0) or 0.0)
    der_global = delivered / attempted if attempted > 0.0 else 0.0
    collisions = int(base_metrics.get("collisions", 0) or 0)
    cluster_pdr_raw = base_metrics.get("qos_cluster_pdr", {}) or {}
    cluster_targets_raw = base_metrics.get("qos_cluster_targets", {}) or {}
    cluster_gaps_raw = base_metrics.get("qos_cluster_pdr_gap", {}) or {}

    cluster_pdr = _format_cluster_mapping(cluster_pdr_raw)
    cluster_targets = _format_cluster_mapping(cluster_targets_raw)
    cluster_gaps = _format_cluster_mapping(cluster_gaps_raw)

    snir_values: List[float] = []
    snir_keys = ("snir_dB", "snir_db", "snr_dB", "snr_db", "snir", "snr")
    for event in events:
        snir_value = None
        for key in snir_keys:
            candidate = event.get(key)
            if candidate is not None:
                snir_value = candidate
                break
        if snir_value is None:
            continue
        try:
            snir_values.append(float(snir_value))
        except (TypeError, ValueError):
            continue
    snir_cdf = _compute_cdf(snir_values)

    mixra_solver = base_metrics.get("mixra_solver")
    if isinstance(mixra_solver, str):
        mixra_solver_str: str | None = mixra_solver
    else:
        mixra_solver_str = None

    result = RunMetrics(
        scenario=scenario,
        algorithm=algorithm,
        num_nodes=int(base_metrics.get("num_nodes", 0) or 0),
        period_s=float(base_metrics.get("packet_interval_s", 0.0) or 0.0),
        pdr_global=pdr_global,
        der_global=der_global,
        collisions=collisions,
        cluster_pdr=cluster_pdr,
        cluster_targets=cluster_targets,
        cluster_gaps=cluster_gaps,
        snir_values=snir_values,
        snir_cdf=snir_cdf,
        mixra_solver=mixra_solver_str,
    )
    return result


def gap_to_target(cluster_id: str, metrics: RunMetrics) -> float:
    """Retourne l'écart PDR - cible pour un cluster donné."""

    actual = metrics.cluster_pdr.get(cluster_id)
    target = metrics.cluster_targets.get(cluster_id)
    if actual is None or target is None:
        return 0.0
    return actual - target


def load_cluster_ids(results: Sequence[RunMetrics]) -> List[str]:
    """Liste triée des identifiants de clusters présents dans les résultats."""

    cluster_ids: set[str] = set()
    for result in results:
        cluster_ids.update(result.cluster_pdr.keys())
    return sorted(cluster_ids, key=lambda value: int(value))

