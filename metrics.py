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
    delivered: int
    attempted: int
    failures_collision: int
    failures_no_signal: int
    baseline_loss_rate: float
    pdr_ci95: float
    der_ci95: float
    cluster_pdr: Dict[str, float]
    cluster_targets: Dict[str, float]
    cluster_gaps: Dict[str, float]
    mean_snir: float
    mean_snr: float
    snir_values: List[float]
    snr_values: List[float]
    snir_by_result: List[Tuple[float, str]]
    snir_cdf: List[Tuple[float, float]]
    snr_cdf: List[Tuple[float, float]]
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
            "delivered": self.delivered,
            "tx_attempted": self.attempted,
            "failures_collision": self.failures_collision,
            "failures_no_signal": self.failures_no_signal,
            "pdr_ci95": self.pdr_ci95,
            "der_ci95": self.der_ci95,
            "collisions": self.collisions,
            "baseline_loss_rate": self.baseline_loss_rate,
            "mixra_solver": self.mixra_solver or "",
            "cluster_pdr_json": json.dumps(self.cluster_pdr, ensure_ascii=False, sort_keys=True),
            "cluster_targets_json": json.dumps(
                self.cluster_targets, ensure_ascii=False, sort_keys=True
            ),
            "cluster_gaps_json": json.dumps(self.cluster_gaps, ensure_ascii=False, sort_keys=True),
            "mean_snir": self.mean_snir,
            "mean_snr": self.mean_snr,
            "snir_samples": len(self.snir_values),
            "snr_samples": len(self.snr_values),
            "snir_cdf_json": json.dumps(self.snir_cdf, ensure_ascii=False),
            "snr_cdf_json": json.dumps(self.snr_cdf, ensure_ascii=False),
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


def _binomial_ci_half_width(successes: Number, total: Number, alpha: float = 0.05) -> float:
    """Retourne une estimation de l'intervalle de confiance (normal approx)."""

    if total <= 0:
        return 0.0
    p_hat = float(successes) / float(total)
    z = 1.96 if abs(alpha - 0.05) < 1e-9 else 1.96
    return z * (p_hat * (1.0 - p_hat) / float(total)) ** 0.5


def compute_run_metrics(
    *,
    scenario: str,
    algorithm: str,
    base_metrics: Mapping[str, object],
    events: Iterable[Mapping[str, object]],
) -> RunMetrics:
    """Construit la structure :class:`RunMetrics` à partir du simulateur."""

    pdr_global = float(base_metrics.get("PDR", 0.0) or 0.0)
    delivered = int(float(base_metrics.get("delivered", 0.0) or 0.0))
    attempted = int(float(base_metrics.get("tx_attempted", 0.0) or 0.0))
    der_global = delivered / attempted if attempted > 0 else 0.0
    collisions = int(base_metrics.get("collisions", 0) or 0)
    cluster_pdr_raw = base_metrics.get("qos_cluster_pdr", {}) or {}
    cluster_targets_raw = base_metrics.get("qos_cluster_targets", {}) or {}
    cluster_gaps_raw = base_metrics.get("qos_cluster_pdr_gap", {}) or {}

    cluster_pdr = _format_cluster_mapping(cluster_pdr_raw)
    cluster_targets = _format_cluster_mapping(cluster_targets_raw)
    cluster_gaps = _format_cluster_mapping(cluster_gaps_raw)

    failures_collision = collisions
    failures_no_signal = int(base_metrics.get("packets_lost_no_signal", 0) or 0)
    baseline_loss_rate = float(base_metrics.get("baseline_loss_rate", 0.0) or 0.0)

    snir_values: List[float] = []
    snr_values: List[float] = []
    snir_by_result: List[Tuple[float, str]] = []
    snir_keys = ("snir_dB", "snir_db", "snir")
    snr_keys = ("snr_dB", "snr_db", "snr")
    for event in events:
        snir_value = None
        for key in snir_keys:
            candidate = event.get(key)
            if candidate is not None:
                snir_value = candidate
                break
        snr_value = None
        for key in snr_keys:
            candidate = event.get(key)
            if candidate is not None:
                snr_value = candidate
                break
        result_str = str(event.get("result", "")).strip()
        if snir_value is None and snr_value is None and not result_str:
            continue
        try:
            snir_float = float(snir_value) if snir_value is not None else float("nan")
        except (TypeError, ValueError):
            snir_float = float("nan")
        try:
            snr_float = float(snr_value) if snr_value is not None else float("nan")
        except (TypeError, ValueError):
            snr_float = float("nan")
        if snir_float == snir_float:
            snir_values.append(snir_float)
            if result_str:
                snir_by_result.append((snir_float, result_str))
        if snr_float == snr_float:
            snr_values.append(snr_float)

        if result_str:
            lowered = result_str.lower()
            if lowered.startswith("success"):
                delivered += 1
            elif "collision" in lowered:
                failures_collision += 1
            elif "nocoverage" in lowered or "no_coverage" in lowered:
                failures_no_signal += 1
            attempted += 1
    snir_cdf = _compute_cdf(snir_values)
    snr_cdf = _compute_cdf(snr_values)
    mean_snir = sum(snir_values) / len(snir_values) if snir_values else 0.0
    mean_snr = sum(snr_values) / len(snr_values) if snr_values else 0.0

    mixra_solver = base_metrics.get("mixra_solver")
    if isinstance(mixra_solver, str):
        mixra_solver_str: str | None = mixra_solver
    else:
        mixra_solver_str = None

    attempted = max(attempted, int(float(base_metrics.get("tx_attempted", 0.0) or 0.0)))
    delivered = min(delivered, attempted)
    failures_collision = min(failures_collision, attempted - delivered)
    failures_no_signal = max(0, attempted - delivered - failures_collision)

    pdr_global = delivered / attempted if attempted > 0 else pdr_global
    der_global = delivered / attempted if attempted > 0 else der_global
    pdr_ci95 = _binomial_ci_half_width(delivered, attempted)
    der_ci95 = _binomial_ci_half_width(delivered, attempted)

    result = RunMetrics(
        scenario=scenario,
        algorithm=algorithm,
        num_nodes=int(base_metrics.get("num_nodes", 0) or 0),
        period_s=float(base_metrics.get("packet_interval_s", 0.0) or 0.0),
        pdr_global=pdr_global,
        der_global=der_global,
        delivered=delivered,
        attempted=attempted,
        failures_collision=failures_collision,
        failures_no_signal=failures_no_signal,
        baseline_loss_rate=baseline_loss_rate,
        pdr_ci95=pdr_ci95,
        der_ci95=der_ci95,
        collisions=collisions,
        cluster_pdr=cluster_pdr,
        cluster_targets=cluster_targets,
        cluster_gaps=cluster_gaps,
        mean_snir=mean_snir,
        mean_snr=mean_snr,
        snir_values=snir_values,
        snr_values=snr_values,
        snir_by_result=snir_by_result,
        snir_cdf=snir_cdf,
        snr_cdf=snr_cdf,
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
