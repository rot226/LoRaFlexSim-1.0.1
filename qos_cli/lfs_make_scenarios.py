"""Utilitaires CLI pour générer et modifier des scénarios QoS."""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
from pathlib import Path
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, Iterable as TypingIterable, Tuple

import yaml


DEFAULT_PATH = Path(__file__).with_name("scenarios.yaml")
DEFAULT_SWEEP_PATH = DEFAULT_PATH.with_name("scenarios_sweep.yaml")

_SWEEP_KEY_ALIASES = {"period": "T"}

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
        return _validate_loaded_scenarios(data)

    return get_default_scenarios()


def _validate_loaded_scenarios(data: Any) -> Dict[str, Any]:
    """Valide sommairement la structure YAML chargée depuis un fichier."""

    if data is None:
        return get_default_scenarios()

    if not isinstance(data, Mapping):
        raise ValueError("Le fichier de scénarios doit contenir un mapping YAML racine.")

    if not data:
        return get_default_scenarios()

    common = data.get("common")
    if common is not None and not isinstance(common, Mapping):
        raise ValueError("La section 'common' doit être un mapping.")

    scenarios = data.get("scenarios")
    if scenarios is None:
        raise ValueError("Le fichier de scénarios doit contenir une section 'scenarios'.")
    if not isinstance(scenarios, Mapping):
        raise ValueError(
            "La section 'scenarios' doit être un mapping d'identifiants vers des configurations."
        )

    for scenario_id, scenario_cfg in scenarios.items():
        if not isinstance(scenario_cfg, Mapping):
            raise ValueError(
                f"Le scénario '{scenario_id}' doit être défini sous forme de mapping YAML."
            )

        for required_field in ("label", "description", "N", "period"):
            if required_field not in scenario_cfg:
                raise ValueError(
                    f"Le scénario '{scenario_id}' doit définir le champ '{required_field}'."
                )

        for int_field in ("N", "period"):
            try:
                int(scenario_cfg[int_field])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"La valeur du champ '{int_field}' dans le scénario '{scenario_id}' doit être numérique."
                ) from exc

        clusters = scenario_cfg.get("clusters")
        if clusters is not None and not isinstance(clusters, Mapping):
            raise ValueError(
                f"La section 'clusters' du scénario '{scenario_id}' doit être un mapping."
            )

    return dict(data)


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


def parse_sweep_option(option: str) -> Tuple[str, list[Any]]:
    """Analyse une option --sweep clé=valeur1,valeur2,..."""

    if "=" not in option:
        raise ValueError(
            f"Format invalide pour --sweep: '{option}' (clé=valeur1,valeur2,... attendu)"
        )

    raw_key, raw_values = option.split("=", 1)
    key = raw_key.strip()
    if not key:
        raise ValueError(f"La clé fournie est vide dans l'option de balayage: '{option}'")

    values: list[Any] = []
    for raw_value in raw_values.split(","):
        cleaned = raw_value.strip()
        if not cleaned:
            continue
        try:
            parsed_value = yaml.safe_load(cleaned)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Valeur YAML invalide pour l'option de balayage '{option}': {exc}"
            ) from exc
        values.append(parsed_value)

    if not values:
        raise ValueError(
            f"Aucune valeur exploitable trouvée pour l'option de balayage: '{option}'"
        )

    return key, values


def load_sweep_csv(path: Path) -> list[OrderedDict[str, Any]]:
    """Charge des combinaisons de balayage depuis un fichier CSV."""

    if not path.exists():
        raise ValueError(f"Fichier CSV introuvable: {path}")

    rows: list[OrderedDict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(
                f"Le fichier CSV '{path}' ne contient pas d'en-têtes exploitables."
            )

        ordered_fields = list(reader.fieldnames)
        for csv_row in reader:
            if not any(csv_row.values()):
                continue
            ordered = OrderedDict()
            for field in ordered_fields:
                if field is None:
                    continue
                raw_value = (csv_row.get(field) or "").strip()
                if raw_value == "":
                    raise ValueError(
                        f"La colonne '{field}' contient une valeur vide dans le fichier '{path}'."
                    )
                try:
                    parsed_value = yaml.safe_load(raw_value)
                except yaml.YAMLError as exc:
                    raise ValueError(
                        f"Valeur YAML invalide dans le fichier '{path}' pour la colonne '{field}': {exc}"
                    ) from exc
                ordered[field] = parsed_value
            rows.append(ordered)

    if not rows:
        raise ValueError(f"Aucune combinaison n'a été trouvée dans '{path}'.")

    return rows


def _format_sweep_identifier(key: str, value: Any) -> str:
    alias = _SWEEP_KEY_ALIASES.get(key, key[:1] or key)
    safe_alias = alias.upper()
    safe_value = str(value).replace(" ", "").replace("/", "-")
    safe_value = safe_value.replace(".", "p")
    return f"{safe_alias}{safe_value}"


def build_sweep_identifier(parameters: Sequence[Tuple[str, Any]]) -> str:
    """Construit un identifiant stable pour une combinaison de balayage."""

    segments = [
        _format_sweep_identifier(key, value)
        for key, value in parameters
    ]
    return "S_" + "_".join(segments)


def _coerce_int(value: Any, key: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"La valeur '{value}' pour '{key}' ne peut être convertie en entier.") from exc


def _compose_sweep_label(parameters: Sequence[Tuple[str, Any]]) -> str:
    couples = ", ".join(f"{key}={value}" for key, value in parameters)
    return f"Balayage ({couples})"


def _compose_sweep_description(parameters: Sequence[Tuple[str, Any]]) -> str:
    couples = ", ".join(f"{key}={value}" for key, value in parameters)
    return (
        "Scénario généré automatiquement à partir d'une combinaison de balayage "
        f"avec {couples}."
    )


def generate_sweep_scenarios(
    combinations: TypingIterable[OrderedDict[str, Any]],
) -> Dict[str, Any]:
    """Génère des scénarios à partir des combinaisons de balayage fournies."""

    sweep_scenarios: Dict[str, Any] = {}
    for combo in combinations:
        if not combo:
            continue
        ordered_items = list(combo.items())
        if "N" not in combo:
            raise ValueError(
                "Chaque combinaison de balayage doit définir une valeur pour 'N'."
            )
        if "period" not in combo:
            raise ValueError(
                "Chaque combinaison de balayage doit définir une valeur pour 'period'."
            )

        scenario_id = build_sweep_identifier(ordered_items)
        label = _compose_sweep_label(ordered_items)
        description = _compose_sweep_description(ordered_items)

        scenario = _make_scenario(
            label=label,
            description=description,
            N=_coerce_int(combo["N"], "N"),
            period=_coerce_int(combo["period"], "period"),
        )

        for key, value in combo.items():
            if key in {"label", "description", "evaluation"}:
                continue
            if key not in {"N", "period"}:
                scenario[key] = value

        scenario["evaluation"] = _build_evaluation(
            scenario.get("clusters", {}), scenario.get("methods", {})
        )

        sweep_scenarios[scenario_id] = scenario

    if not sweep_scenarios:
        raise ValueError("Aucun scénario de balayage valide n'a été généré.")

    return sweep_scenarios


def build_sweep_combinations(
    sweep_options: Sequence[Tuple[str, list[Any]]]
) -> list[OrderedDict[str, Any]]:
    """Produit les combinaisons cartésiennes à partir des options --sweep."""

    if not sweep_options:
        return []

    keys = [key for key, _ in sweep_options]
    values_list = [values for _, values in sweep_options]

    combinations: list[OrderedDict[str, Any]] = []
    for product_values in itertools.product(*values_list):
        combinations.append(OrderedDict(zip(keys, product_values)))

    return combinations


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
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="clé=valeurs",
        help=(
            "Décrit une dimension de balayage, par exemple --sweep N=60,120. "
            "Peut être répété pour créer une grille cartésienne."
        ),
    )
    parser.add_argument(
        "--sweep-csv",
        action="append",
        type=Path,
        default=[],
        metavar="fichier",
        help="Charge des combinaisons explicites depuis un fichier CSV (en-têtes requis).",
    )
    parser.add_argument(
        "--sweep-out",
        type=Path,
        help=(
            "Chemin de sortie pour les scénarios de balayage générés (par défaut: qos_cli/scenarios_sweep.yaml)."
        ),
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

    sweep_options: list[Tuple[str, list[Any]]] = []
    for option in parsed.sweep:
        try:
            sweep_options.append(parse_sweep_option(option))
        except ValueError as exc:
            parser.error(str(exc))

    sweep_combinations = build_sweep_combinations(sweep_options)

    csv_combinations: list[OrderedDict[str, Any]] = []
    for csv_path in parsed.sweep_csv:
        try:
            csv_combinations.extend(load_sweep_csv(csv_path))
        except ValueError as exc:
            parser.error(str(exc))

    all_combinations: list[OrderedDict[str, Any]] = []
    if sweep_combinations:
        all_combinations.extend(sweep_combinations)
    if csv_combinations:
        all_combinations.extend(csv_combinations)

    if all_combinations:
        try:
            sweep_scenarios = generate_sweep_scenarios(all_combinations)
        except ValueError as exc:
            parser.error(str(exc))

        base_common = {}
        if isinstance(scenarios, Mapping):
            raw_common = scenarios.get("common")
            if isinstance(raw_common, Mapping):
                base_common = copy.deepcopy(raw_common)

        sweep_data = {
            "common": base_common,
            "scenarios": sweep_scenarios,
        }

        sweep_path = parsed.sweep_out or DEFAULT_SWEEP_PATH
        write_scenarios(sweep_data, sweep_path)

        print("Balayage généré avec les combinaisons suivantes :")
        for combo in all_combinations:
            ordered_items = list(combo.items())
            scenario_id = build_sweep_identifier(ordered_items)
            summary = ", ".join(f"{key}={value}" for key, value in ordered_items)
            print(f"  - {scenario_id}: {summary}")
        print(f"Scénarios de balayage écrits dans {sweep_path.resolve()}")


if __name__ == "__main__":  # pragma: no cover - exécution CLI directe
    main()
