"""CLI pour agréger les métriques QoS à partir des résultats LoRaFlexSim.

Cette implémentation repose sur plusieurs hypothèses concernant la structure des
CSV exportés par le banc de simulation :

* Chaque scénario est stocké dans ``<racine_resultats>/<methode>/<scenario>/``.
* Les transmissions sont listées dans ``packets.csv`` avec au minimum un
  identifiant de cluster (``cluster`` ou ``cluster_id``) et un indicateur de
  succès (colonne booléenne ou statut textuel ``delivered``/``collision``).
* Les caractéristiques des nœuds résident dans ``nodes.csv`` et exposent
  l'énergie consommée (``energy_J`` ou ``energy``) et le facteur d'étalement
  (``sf`` ou ``spreading_factor``).
* Les fichiers contiennent également un identifiant de nœud (``node_id`` ou
  ``device``) permettant de calculer l'indice de Jain à partir du nombre de
  paquets délivrés par nœud.

Lorsque ces colonnes sont absentes, les métriques correspondantes sont marquées
comme indisponibles et un avertissement est émis dans le résumé. Les règles de
validation (PASS/FAIL) supposent qu'un bloc ``evaluation`` existe pour chaque
scénario dans ``scenarios.yaml`` et définit :

* ``mixra_method`` : nom du répertoire contenant les résultats MixRA.
* ``baselines`` : liste des méthodes de référence à comparer (S2).
* ``cluster_targets`` : dictionnaire ``{cluster: objectif_PDR}``.

Ces hypothèses sont documentées afin de faciliter l'alignement avec les exports
réels du simulateur.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml


@dataclass
class MethodScenarioMetrics:
    """Métriques calculées pour un couple (méthode, scénario)."""

    method: str
    scenario: str
    delivered: int
    attempted: int
    cluster_pdr: Dict[str, float]
    pdr_global: Optional[float]
    der_global: Optional[float]
    collisions: Optional[int]
    snir_cdf: List[Tuple[float, float]]
    energy_j: Optional[float]
    jain_index: Optional[float]
    min_sf_share: Optional[float]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Construit l'espace de noms des arguments CLI."""

    parser = argparse.ArgumentParser(
        prog="lfs_metrics",
        description="Analyse les résultats QoS d'un lot de simulations LoRaFlexSim.",
    )
    parser.add_argument(
        "--in",
        dest="root",
        type=Path,
        required=True,
        help="Dossier racine contenant les résultats (sous-forme <méthode>/<scénario>).",
    )
    parser.add_argument(
        "--config",
        dest="config",
        type=Path,
        required=True,
        help="Fichier YAML décrivant les scénarios et cibles QoS.",
    )
    parser.add_argument(
        "--summary",
        dest="summary",
        type=Path,
        default=Path("qos_cli") / "SUMMARY.txt",
        help="Chemin du fichier SUMMARY.txt à produire (défaut : qos_cli/SUMMARY.txt).",
    )
    return parser.parse_args(argv)


def load_yaml_config(path: Path) -> Mapping[str, Mapping[str, object]]:
    """Charge le fichier ``scenarios.yaml`` et retourne son contenu."""

    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    scenarios = content.get("scenarios", {}) if isinstance(content, Mapping) else {}
    return scenarios  # type: ignore[return-value]


def discover_methods(root: Path) -> List[str]:
    """Liste les méthodes disponibles dans le dossier des résultats."""

    if not root.is_dir():
        return []
    methods = [item.name for item in root.iterdir() if item.is_dir()]
    methods.sort()
    return methods


def discover_scenarios_for_method(root: Path, method: str) -> List[str]:
    """Retourne les scénarios disponibles pour une méthode donnée."""

    method_dir = root / method
    if not method_dir.is_dir():
        return []
    scenarios = [item.name for item in method_dir.iterdir() if item.is_dir()]
    scenarios.sort()
    return scenarios


def read_dataframe(path: Path) -> Optional[pd.DataFrame]:
    """Charge un CSV s'il existe, sinon retourne ``None``."""

    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as error:  # pragma: no cover - message informative
        raise RuntimeError(f"Impossible de lire {path}: {error}") from error
    return df


def _find_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _extract_success_series(df: pd.DataFrame) -> Optional[pd.Series]:
    success_columns = [
        "delivered",
        "success",
        "is_delivered",
        "rx_success",
        "successful",
    ]
    for column in success_columns:
        if column in df.columns:
            series = df[column]
            if series.dtype == bool:
                return series.astype(int)
            if pd.api.types.is_numeric_dtype(series):
                return pd.to_numeric(series, errors="coerce")
            return series.astype(str).str.lower().isin({"true", "1", "yes", "delivered", "success"}).astype(int)
    status_column = _find_column(df.columns, ["status", "result", "outcome", "rx_status"])
    if status_column is not None:
        statuses = df[status_column].astype(str).str.lower()
        return statuses.isin({"delivered", "success", "ok", "received"}).astype(int)
    return None


def _extract_collision_series(df: pd.DataFrame) -> Optional[pd.Series]:
    collision_columns = [
        "collision",
        "is_collision",
        "collided",
    ]
    for column in collision_columns:
        if column in df.columns:
            series = df[column]
            if series.dtype == bool:
                return series.astype(int)
            if pd.api.types.is_numeric_dtype(series):
                return pd.to_numeric(series, errors="coerce")
            return series.astype(str).str.lower().isin({"true", "1", "yes"}).astype(int)
    status_column = _find_column(df.columns, ["status", "result", "outcome", "rx_status"])
    if status_column is not None:
        statuses = df[status_column].astype(str).str.lower()
        return statuses.isin({"collision", "collided", "fail_collision"}).astype(int)
    return None


def compute_cluster_pdr(df: Optional[pd.DataFrame]) -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    cluster_column = _find_column(
        df.columns,
        ["cluster", "cluster_id", "clusterId", "ring", "qos_cluster"],
    )
    success_series = _extract_success_series(df)
    if cluster_column is None or success_series is None:
        return {}
    grouped = df.assign(_success=success_series).groupby(cluster_column)["_success"]
    totals = grouped.count()
    successes = grouped.sum()
    result: Dict[str, float] = {}
    for cluster, total in totals.items():
        if total <= 0:
            continue
        result[str(cluster)] = float(successes.get(cluster, 0.0) / total)
    return dict(sorted(result.items(), key=lambda item: item[0]))


def compute_global_ratios(df: Optional[pd.DataFrame]) -> Tuple[Optional[float], Optional[float], int, int]:
    if df is None or df.empty:
        return None, None, 0, 0
    success_series = _extract_success_series(df)
    if success_series is None:
        attempted = len(df.index)
        return None, None, 0, attempted
    successes = int(success_series.sum())
    attempted = int(len(success_series))
    if attempted == 0:
        return None, None, successes, attempted
    ratio = successes / attempted
    return ratio, ratio, successes, attempted


def compute_collisions(df: Optional[pd.DataFrame]) -> Optional[int]:
    if df is None or df.empty:
        return None
    collision_series = _extract_collision_series(df)
    if collision_series is None:
        return None
    return int(pd.to_numeric(collision_series, errors="coerce").fillna(0).sum())


def compute_snir_cdf(df: Optional[pd.DataFrame]) -> List[Tuple[float, float]]:
    if df is None or df.empty:
        return []
    snir_column = _find_column(
        df.columns,
        [
            "snir",
            "snir_db",
            "snir_dB",
            "snr",
            "snr_db",
            "snr_dB",
        ],
    )
    if snir_column is None:
        return []
    values = pd.to_numeric(df[snir_column], errors="coerce").dropna().to_list()
    if not values:
        return []
    minimum = math.floor(min(values))
    maximum = math.ceil(max(values))
    if minimum == maximum:
        return [(float(minimum), 1.0)]
    bin_width = 1.0
    bin_edges = [minimum + i * bin_width for i in range(int((maximum - minimum) / bin_width) + 1)]
    bin_edges.append(maximum)
    counts = [0 for _ in range(len(bin_edges) - 1)]
    for value in values:
        index = min(int((value - minimum) / bin_width), len(counts) - 1)
        counts[index] += 1
    total = sum(counts)
    cdf: List[Tuple[float, float]] = []
    cumulative = 0
    for edge_index, count in enumerate(counts):
        cumulative += count
        upper = bin_edges[edge_index + 1]
        cdf.append((float(upper), cumulative / total if total else 0.0))
    return cdf


def compute_energy(nodes_df: Optional[pd.DataFrame]) -> Optional[float]:
    if nodes_df is None or nodes_df.empty:
        return None
    energy_column = _find_column(
        nodes_df.columns,
        ["energy_J", "energy", "energy_consumed", "consumed_energy_J", "energy_total"],
    )
    if energy_column is None:
        return None
    return float(pd.to_numeric(nodes_df[energy_column], errors="coerce").dropna().sum())


def compute_jain_index(packets_df: Optional[pd.DataFrame]) -> Optional[float]:
    if packets_df is None or packets_df.empty:
        return None
    node_column = _find_column(
        packets_df.columns,
        ["node_id", "device", "device_id", "end_device", "devaddr"],
    )
    if node_column is None:
        return None
    success_series = _extract_success_series(packets_df)
    if success_series is None:
        return None
    per_node = (
        packets_df.assign(_success=success_series).groupby(node_column)["_success"].sum().astype(float)
    )
    values = per_node.values
    if len(values) == 0:
        return None
    numerator = float(values.sum()) ** 2
    denominator = float(len(values)) * float((values ** 2).sum())
    if denominator == 0:
        return None
    return numerator / denominator


def compute_min_sf_share(nodes_df: Optional[pd.DataFrame]) -> Optional[float]:
    if nodes_df is None or nodes_df.empty:
        return None
    sf_column = _find_column(nodes_df.columns, ["sf", "spreading_factor", "SF", "assigned_sf"])
    if sf_column is None:
        return None
    sf_values = pd.to_numeric(nodes_df[sf_column], errors="coerce").dropna()
    if sf_values.empty:
        return None
    min_sf = sf_values.min()
    if min_sf <= 0:
        return None
    share = (sf_values == min_sf).sum() / len(sf_values)
    return float(share)


def load_metrics_for_method_scenario(root: Path, method: str, scenario: str) -> MethodScenarioMetrics:
    scenario_dir = root / method / scenario
    packets_df = read_dataframe(scenario_dir / "packets.csv")
    nodes_df = read_dataframe(scenario_dir / "nodes.csv")

    cluster_pdr = compute_cluster_pdr(packets_df)
    pdr_global, der_global, delivered, attempted = compute_global_ratios(packets_df)
    collisions = compute_collisions(packets_df)
    snir_cdf = compute_snir_cdf(packets_df)
    energy_j = compute_energy(nodes_df)
    jain_index = compute_jain_index(packets_df)
    min_sf_share = compute_min_sf_share(nodes_df)

    return MethodScenarioMetrics(
        method=method,
        scenario=scenario,
        delivered=delivered,
        attempted=attempted,
        cluster_pdr=cluster_pdr,
        pdr_global=pdr_global,
        der_global=der_global,
        collisions=collisions,
        snir_cdf=snir_cdf,
        energy_j=energy_j,
        jain_index=jain_index,
        min_sf_share=min_sf_share,
    )


def load_all_metrics(root: Path) -> Dict[Tuple[str, str], MethodScenarioMetrics]:
    metrics: Dict[Tuple[str, str], MethodScenarioMetrics] = {}
    for method in discover_methods(root):
        for scenario in discover_scenarios_for_method(root, method):
            metrics[(method, scenario)] = load_metrics_for_method_scenario(root, method, scenario)
    return metrics


def evaluate_pass_fail(
    scenario_id: str,
    scenario_cfg: Mapping[str, object],
    metrics_by_method: Mapping[str, MethodScenarioMetrics],
) -> Tuple[str, List[str]]:
    """Retourne le verdict PASS/FAIL et des commentaires pour un scénario."""

    evaluation_cfg = scenario_cfg.get("evaluation", {}) if isinstance(scenario_cfg, Mapping) else {}
    mixra_method = evaluation_cfg.get("mixra_method", "MixRA")
    baselines = evaluation_cfg.get("baselines", [])
    cluster_targets = evaluation_cfg.get("cluster_targets", {})

    comments: List[str] = []
    verdict = "INCONNU"

    mixra_metrics = metrics_by_method.get(str(mixra_method))
    if mixra_metrics is None:
        comments.append("Aucune donnée MixRA trouvée – vérifiez le nom du dossier.")
        return verdict, comments

    if scenario_id in {"S1", "S2"}:
        margin = float(evaluation_cfg.get("mixra_margin", 0.02))
        failing_clusters: List[str] = []
        for cluster, target in cluster_targets.items():
            try:
                target_value = float(target)
            except (TypeError, ValueError):
                continue
            actual = mixra_metrics.cluster_pdr.get(str(cluster))
            if actual is None:
                failing_clusters.append(f"{cluster} (donnée manquante)")
                continue
            if actual < target_value - margin:
                failing_clusters.append(f"{cluster} ({actual:.3f} < {target_value - margin:.3f})")
        if failing_clusters:
            verdict = "FAIL"
            comments.append(
                "Clusters sous le seuil ajusté : " + ", ".join(sorted(failing_clusters))
            )
        else:
            verdict = "PASS"
            comments.append("Tous les clusters respectent la marge de 0.02 sous la cible.")

    if scenario_id == "S3":
        sorted_clusters = sorted(
            ((cluster, float(target)) for cluster, target in cluster_targets.items()),
            key=lambda item: str(item[0]),
        )
        failing_cluster = None
        for cluster, target in sorted_clusters:
            actual = mixra_metrics.cluster_pdr.get(str(cluster))
            if actual is None:
                continue
            if actual < target:
                failing_cluster = (cluster, actual, target)
                break
        if failing_cluster:
            verdict = "FAIL"
            cluster, actual, target = failing_cluster
            comments.append(
                f"Premier cluster sous cible : {cluster} ({actual:.3f} < {target:.3f})."
            )
        else:
            comments.append("Aucun cluster sous la cible déclarée.")
            verdict = "PASS"

    if scenario_id == "S2" and baselines:
        baseline_names = [str(name) for name in baselines]
        baseline_metrics = [metrics_by_method.get(name) for name in baseline_names]
        baseline_metrics = [metric for metric in baseline_metrics if metric is not None]
        if not baseline_metrics:
            comments.append("Aucune baseline valide détectée pour la comparaison PDR global.")
        else:
            mixra_pdr = mixra_metrics.pdr_global
            baseline_pdrs = [metric.pdr_global for metric in baseline_metrics if metric.pdr_global is not None]
            if mixra_pdr is None or not baseline_pdrs:
                comments.append("Comparaison PDR impossible (valeurs manquantes).")
            else:
                min_baseline = min(baseline_pdrs)
                if mixra_pdr >= min_baseline:
                    comments.append(
                        f"MixRA >= baselines (MixRA {mixra_pdr:.3f} vs min baseline {min_baseline:.3f})."
                    )
                    if verdict == "INCONNU":
                        verdict = "PASS"
                else:
                    comments.append(
                        f"MixRA < baselines (MixRA {mixra_pdr:.3f} vs min baseline {min_baseline:.3f})."
                    )
                    verdict = "FAIL"
    if verdict == "INCONNU":
        verdict = "PASS" if not comments else "FAIL"
    return verdict, comments


def format_metrics(metrics: MethodScenarioMetrics) -> str:
    def fmt(value: Optional[float]) -> str:
        return "N/A" if value is None else f"{value:.3f}"

    sf_percentage = (
        "N/A" if metrics.min_sf_share is None else f"{metrics.min_sf_share * 100:.1f}%"
    )
    energy = "N/A" if metrics.energy_j is None else f"{metrics.energy_j:.3f} J"
    collision = "N/A" if metrics.collisions is None else str(metrics.collisions)

    return (
        f"PDR={fmt(metrics.pdr_global)} | DER={fmt(metrics.der_global)} | "
        f"Collisions={collision} | Jain={fmt(metrics.jain_index)} | Energie={energy} | "
        f"SFmin={sf_percentage}"
    )


def write_summary(
    summary_path: Path,
    scenarios_cfg: Mapping[str, Mapping[str, object]],
    all_metrics: Mapping[Tuple[str, str], MethodScenarioMetrics],
) -> None:
    lines: List[str] = ["=== Résumé des résultats QoS ===", ""]
    grouped: Dict[str, Dict[str, MethodScenarioMetrics]] = {}
    for (method, scenario), metrics in all_metrics.items():
        grouped.setdefault(scenario, {})[method] = metrics

    for scenario in sorted(grouped.keys()):
        scenario_cfg = scenarios_cfg.get(scenario, {})
        metrics_by_method = grouped[scenario]
        verdict, comments = evaluate_pass_fail(scenario, scenario_cfg, metrics_by_method)
        lines.append(f"Scenario {scenario} – Verdict : {verdict}")
        if comments:
            for comment in comments:
                lines.append(f"  • {comment}")
        if not metrics_by_method:
            lines.append("  • Aucune métrique disponible.")
        else:
            for method in sorted(metrics_by_method.keys()):
                metric_line = format_metrics(metrics_by_method[method])
                lines.append(f"    - {method}: {metric_line}")
        lines.append("")

    lines.extend(
        [
            "TODO: Ajouter les métriques de latence et de jitter dès qu'elles sont exportées.",
            "TODO: Étendre le résumé avec une visualisation des CDF SNIR par cluster si nécessaire.",
        ]
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    scenarios_cfg = load_yaml_config(args.config)
    all_metrics = load_all_metrics(args.root)
    write_summary(args.summary, scenarios_cfg, all_metrics)


if __name__ == "__main__":  # pragma: no cover - point d'entrée CLI
    main()
