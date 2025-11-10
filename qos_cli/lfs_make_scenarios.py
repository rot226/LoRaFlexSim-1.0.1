"""Utilitaires CLI pour générer et modifier des scénarios QoS."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Tuple

import yaml


DEFAULT_PATH = Path(__file__).with_name("scenarios.yaml")

_CLUSTER_TEMPLATE: Dict[str, Any] = {
    "cluster_low": {
        "share": 0.10,
        "target_pdr": 0.90,
        "nu_max": {
            "SF7": {"CH0": 0.28, "CH1": 0.28, "CH2": 0.28},
            "SF8": {"CH0": 0.24, "CH1": 0.24, "CH2": 0.24},
            "SF9": {"CH0": 0.20, "CH1": 0.20, "CH2": 0.20},
            "SF10": {"CH0": 0.16, "CH1": 0.16, "CH2": 0.16},
            "SF11": {"CH0": 0.12, "CH1": 0.12, "CH2": 0.12},
            "SF12": {"CH0": 0.10, "CH1": 0.10, "CH2": 0.10},
        },
    },
    "cluster_mid": {
        "share": 0.30,
        "target_pdr": 0.80,
        "nu_max": {
            "SF7": {"CH0": 0.20, "CH1": 0.20, "CH2": 0.20},
            "SF8": {"CH0": 0.18, "CH1": 0.18, "CH2": 0.18},
            "SF9": {"CH0": 0.14, "CH1": 0.14, "CH2": 0.14},
            "SF10": {"CH0": 0.10, "CH1": 0.10, "CH2": 0.10},
            "SF11": {"CH0": 0.07, "CH1": 0.07, "CH2": 0.07},
            "SF12": {"CH0": 0.05, "CH1": 0.05, "CH2": 0.05},
        },
    },
    "cluster_high": {
        "share": 0.60,
        "target_pdr": 0.70,
        "nu_max": {
            "SF7": {"CH0": 0.12, "CH1": 0.12, "CH2": 0.12},
            "SF8": {"CH0": 0.10, "CH1": 0.10, "CH2": 0.10},
            "SF9": {"CH0": 0.08, "CH1": 0.08, "CH2": 0.08},
            "SF10": {"CH0": 0.06, "CH1": 0.06, "CH2": 0.06},
            "SF11": {"CH0": 0.04, "CH1": 0.04, "CH2": 0.04},
            "SF12": {"CH0": 0.03, "CH1": 0.03, "CH2": 0.03},
        },
    },
}

_METHODS_DEFAULT: Dict[str, Any] = {
    "available": ["ADR", "APRA_like", "MixRA_H", "MixRA_Opt"],
    "fallback": "Greedy",
}


def _build_evaluation(clusters: Mapping[str, Any], methods: Mapping[str, Any]) -> Dict[str, Any]:
    """Construit la configuration d'évaluation pour un scénario donné."""

    available_methods = []
    if isinstance(methods, Mapping):
        raw_available = methods.get("available", [])
        if isinstance(raw_available, Iterable) and not isinstance(raw_available, (str, bytes)):
            available_methods = [str(method) for method in raw_available]

    preferred_mixra = "MixRA_Opt"
    if preferred_mixra not in available_methods and available_methods:
        preferred_mixra = available_methods[0]

    cluster_targets: Dict[str, Any] = {}
    if isinstance(clusters, Mapping):
        for cluster_name, cluster_cfg in clusters.items():
            if not isinstance(cluster_cfg, Mapping):
                continue
            if "target_pdr" in cluster_cfg:
                cluster_targets[str(cluster_name)] = cluster_cfg["target_pdr"]

    baselines = [method for method in available_methods if method != preferred_mixra]

    return {
        "mixra_method": preferred_mixra,
        "baselines": baselines,
        "cluster_targets": cluster_targets,
    }


def _make_scenario(label: str, description: str, N: int, period: int) -> Dict[str, Any]:
    clusters = copy.deepcopy(_CLUSTER_TEMPLATE)
    methods = copy.deepcopy(_METHODS_DEFAULT)
    evaluation = _build_evaluation(clusters, methods)
    return {
        "label": label,
        "description": description,
        "N": N,
        "period": period,
        "clusters": clusters,
        "methods": methods,
        "evaluation": evaluation,
    }


DEFAULT_SCENARIOS: Dict[str, Any] = {
    "common": {
        "gateway": {
            "id": "GW0",
            "position_m": [0.0, 0.0, 15.0],
            "antennas": 1,
        },
        "channels": {
            "CH0": {"frequency_hz": 868_100_000, "bandwidth_hz": 125_000},
            "CH1": {"frequency_hz": 868_300_000, "bandwidth_hz": 125_000},
            "CH2": {"frequency_hz": 868_500_000, "bandwidth_hz": 125_000},
        },
        "spreading_factors": [7, 8, 9, 10, 11, 12],
        "propagation": {
            "fading": "Rayleigh",
            "capture_threshold_db": 1.0,
        },
        "payload_size_bytes": 20,
        "duty_cycle": 0.01,
    },
    "scenarios": {
        "S0": _make_scenario(
            label="Charge très faible",
            description="Réseau clairsemé avec trafic ponctuel.",
            N=60,
            period=900,
        ),
        "S1": _make_scenario(
            label="Charge modérée",
            description="Réseau équilibré avec trafic planifié.",
            N=120,
            period=600,
        ),
        "S2": _make_scenario(
            label="Charge élevée",
            description="Densité importante avec trafic périodique serré.",
            N=300,
            period=300,
        ),
        "S3": _make_scenario(
            label="Charge critique",
            description="Applications sensibles nécessitant une forte disponibilité.",
            N=180,
            period=120,
        ),
    },
}


def get_default_scenarios() -> Dict[str, Any]:
    """Retourne une copie profonde du modèle de scénarios par défaut."""
    return copy.deepcopy(DEFAULT_SCENARIOS)


def load_scenarios(path: Path, use_default: bool = False) -> Dict[str, Any]:
    """Charge un fichier de scénarios YAML ou retourne la structure par défaut."""
    if use_default:
        return get_default_scenarios()

    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        # TODO: Valider la structure chargée (schéma, types) avant utilisation.
        return data

    return get_default_scenarios()


def parse_set_option(option: str) -> Tuple[str, Any]:
    """Analyse une option --set clé=valeur."""
    if "=" not in option:
        raise ValueError(f"Format invalide pour --set: '{option}' (clé=valeur attendu)")

    raw_key, raw_value = option.split("=", 1)
    key = raw_key.strip()
    if not key:
        raise ValueError(f"La clé fournie est vide dans l'option: '{option}'")

    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:  # pragma: no cover - protection simple
        raise ValueError(f"Valeur YAML invalide pour '{option}': {exc}") from exc

    return key, value


def _ensure_nested_container(container: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Retourne un sous-dictionnaire existant ou en crée un nouveau."""
    value = container.get(key)
    if isinstance(value, dict):
        return value
    if value is None:
        container[key] = {}
        return container[key]
    raise ValueError(
        f"Impossible de définir une valeur imbriquée sous '{key}' car un type non compatible est présent."
    )


def _propagate_cluster_target(
    data: Dict[str, Any], scenario_id: str, cluster_id: str, target_value: Any
) -> None:
    """Maintient la synchronisation entre clusters.*.target_pdr et evaluation.cluster_targets."""

    scenarios = data.get("scenarios")
    if not isinstance(scenarios, dict):
        return
    scenario_cfg = scenarios.get(scenario_id)
    if not isinstance(scenario_cfg, dict):
        return

    evaluation = scenario_cfg.get("evaluation")
    if not isinstance(evaluation, dict):
        clusters = scenario_cfg.get("clusters", {}) if isinstance(scenario_cfg, dict) else {}
        methods = scenario_cfg.get("methods", {}) if isinstance(scenario_cfg, dict) else {}
        evaluation = _build_evaluation(clusters, methods)
        scenario_cfg["evaluation"] = evaluation

    cluster_targets = evaluation.get("cluster_targets")
    if not isinstance(cluster_targets, dict):
        cluster_targets = {}
        evaluation["cluster_targets"] = cluster_targets

    cluster_targets[str(cluster_id)] = target_value


def apply_update(data: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Applique une mise à jour à l'aide d'une clé en notation pointée."""
    segments = [segment.strip() for segment in dotted_key.split(".") if segment.strip()]
    if not segments:
        raise ValueError("La clé en notation pointée est vide.")

    current: Dict[str, Any] = data
    for segment in segments[:-1]:
        current = _ensure_nested_container(current, segment)

    current[segments[-1]] = value
    if (
        len(segments) >= 5
        and segments[0] == "scenarios"
        and segments[2] == "clusters"
        and segments[-1] == "target_pdr"
    ):
        scenario_id, cluster_id = segments[1], segments[3]
        _propagate_cluster_target(data, scenario_id, cluster_id, value)
    # TODO: Ajouter des validations métier sur les valeurs assignées (plages, types, etc.).


def apply_updates(data: Dict[str, Any], updates: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    """Applique une séquence de mises à jour sur la structure de scénarios."""
    for key, value in updates:
        apply_update(data, key, value)
    return data


def write_scenarios(data: Dict[str, Any], path: Path) -> None:
    """Écrit la structure YAML sur le disque avec un ordre lisible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)


def build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur d'arguments de la CLI."""
    parser = argparse.ArgumentParser(description="Génère ou met à jour un fichier de scénarios QoS.")
    parser.add_argument(
        "--new",
        action="store_true",
        help="Réinitialise le fichier avec le contenu par défaut avant d'appliquer les modifications.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Chemin de sortie du fichier YAML (par défaut: qos_cli/scenarios.yaml).",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="clé=valeur",
        help="Définit ou met à jour dynamiquement des paramètres (notation imbriquée supportée).",
    )
    return parser


def main(args: Iterable[str] | None = None) -> None:
    """Point d'entrée principal pour la ligne de commande."""
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    target_path = parsed.out or DEFAULT_PATH
    target_path = target_path if isinstance(target_path, Path) else Path(target_path)

    load_path = target_path if target_path.exists() else DEFAULT_PATH
    scenarios = load_scenarios(load_path, use_default=parsed.new)

    updates = []
    for option in parsed.set:
        try:
            updates.append(parse_set_option(option))
        except ValueError as exc:
            parser.error(str(exc))

    if updates:
        apply_updates(scenarios, updates)

    write_scenarios(scenarios, target_path)
    print(f"Scénarios écrits dans {target_path.resolve()}")


if __name__ == "__main__":  # pragma: no cover - exécution CLI directe
    main()
