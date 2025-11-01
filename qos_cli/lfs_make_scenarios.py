"""Utilitaires CLI pour générer et modifier des scénarios QoS."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml


DEFAULT_PATH = Path(__file__).with_name("scenarios.yaml")

# Structure par défaut alignée avec le fichier scenarios.yaml historique.
DEFAULT_SCENARIOS: Dict[str, Any] = {
    "common": {
        "simulation_time_s": 3600,
        "payload_size_bytes": 12,
        "frequency_hz": 868_100_000,
        "coding_rate": "4/5",
        "bandwidth_hz": 125_000,
        "duty_cycle": 0.01,
        "acknowledgements": False,
        # TODO: Étendre avec les paramètres radio supplémentaires requis par LoRaFlexSim.
    },
    "scenarios": {
        "S0": {
            "label": "Baseline faibles trafics",
            "description": "Un seul périphérique transmet périodiquement avec un débit faible.",
            "device_count": 1,
            "send_interval_s": 600,
            "spreading_factor": 7,
            "tx_power_dbm": 14,
            # TODO: Ajouter la prise en charge des pertes de propagation si nécessaire.
        },
        "S1": {
            "label": "Densité modérée",
            "description": "Un réseau de taille moyenne avec trafic périodique synchronisé.",
            "device_count": 50,
            "send_interval_s": 300,
            "spreading_factor": 8,
            "tx_power_dbm": 14,
        },
        "S2": {
            "label": "Déploiement massif",
            "description": "Grand nombre de nœuds avec trafic périodique aléatoire.",
            "device_count": 500,
            "send_interval_s": [120, 900],
            "spreading_factor": 9,
            "tx_power_dbm": 16,
            "random_seed": 42,
            # TODO: Clarifier le format attendu pour les intervalles aléatoires.
        },
        "S3": {
            "label": "Scénario critique QoS",
            "description": "Applications critiques avec redondance et contraintes de délai.",
            "device_count": 20,
            "send_interval_s": 120,
            "spreading_factor": 10,
            "tx_power_dbm": 18,
            "redundancy": 2,
            "latency_budget_s": 30,
            # TODO: Confirmer la gestion des contraintes QoS spécifiques dans LoRaFlexSim.
        },
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


def apply_update(data: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Applique une mise à jour à l'aide d'une clé en notation pointée."""
    segments = [segment.strip() for segment in dotted_key.split(".") if segment.strip()]
    if not segments:
        raise ValueError("La clé en notation pointée est vide.")

    current: Dict[str, Any] = data
    for segment in segments[:-1]:
        current = _ensure_nested_container(current, segment)

    current[segments[-1]] = value
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
