"""Préréglages de scénarios pour le banc QoS par clusters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

__all__ = [
    "ScenarioPreset",
    "get_preset",
    "list_presets",
    "describe_presets",
]


@dataclass(frozen=True)
class ScenarioPreset:
    """Description d'un ensemble de combinaisons à exécuter."""

    name: str
    label: str
    description: str
    node_counts: Sequence[int]
    tx_periods: Sequence[float]
    simulation_duration_s: float
    estimated_runtime_min: float

    def combo_count(self) -> int:
        """Retourne le nombre de couples (nœuds, période)."""

        return len(tuple(self.node_counts)) * len(tuple(self.tx_periods))

    def as_dict(self) -> Dict[str, object]:
        """Expose les principaux champs sous forme de dictionnaire."""

        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "node_counts": list(self.node_counts),
            "tx_periods": list(self.tx_periods),
            "simulation_duration_s": float(self.simulation_duration_s),
            "estimated_runtime_min": float(self.estimated_runtime_min),
        }


_PRESETS: Mapping[str, ScenarioPreset] = {
    "quick": ScenarioPreset(
        name="quick",
        label="Scénario rapide",
        description="Première passe pour valider les métriques et générer des graphiques d'aperçu",
        node_counts=(1000, 5000),
        tx_periods=(600.0,),
        simulation_duration_s=4.0 * 3600.0,
        estimated_runtime_min=30.0,
    ),
    "baseline": ScenarioPreset(
        name="baseline",
        label="Campagne intermédiaire",
        description="Évaluation équilibrée couvrant charges faibles et moyennes",
        node_counts=(1000, 5000, 10000),
        tx_periods=(600.0, 300.0),
        simulation_duration_s=12.0 * 3600.0,
        estimated_runtime_min=150.0,
    ),
    "full": ScenarioPreset(
        name="full",
        label="Campagne complète",
        description="Validation exhaustive des charges et périodes spécifiées dans le banc",
        node_counts=(1000, 5000, 10000, 13000, 15000),
        tx_periods=(600.0, 300.0, 150.0),
        simulation_duration_s=24.0 * 3600.0,
        estimated_runtime_min=480.0,
    ),
}


def list_presets() -> Sequence[ScenarioPreset]:
    """Retourne l'ensemble des préréglages disponibles."""

    return tuple(_PRESETS.values())


def get_preset(name: str) -> ScenarioPreset:
    """Récupère un préréglage par son nom (insensible à la casse)."""

    key = name.lower()
    try:
        return _PRESETS[key]
    except KeyError as exc:  # pragma: no cover - erreur explicite pour l'utilisateur
        available = ", ".join(sorted(_PRESETS))
        raise ValueError(f"Preset inconnu '{name}'. Valeurs possibles : {available}") from exc


def describe_presets(algorithm_count: int) -> str:
    """Construit une table texte décrivant les préréglages."""

    lines: List[str] = []
    lines.append("Préréglages disponibles :")
    lines.append("")
    header = (
        "Nom",
        "Description",
        "Couples",
        "Runs (A×C)",
        "Durée simu",
        "Runtime estimé",
    )
    lines.append(" | ".join(header))
    lines.append(" | ".join("---" for _ in header))
    for preset in list_presets():
        combos = preset.combo_count()
        runs = combos * max(1, int(algorithm_count))
        duration_hours = preset.simulation_duration_s / 3600.0
        runtime_hours = preset.estimated_runtime_min / 60.0
        lines.append(
            " | ".join(
                [
                    preset.name,
                    preset.label,
                    str(combos),
                    str(runs),
                    f"{duration_hours:.1f} h",
                    f"~{runtime_hours:.1f} h",
                ]
            )
        )
    lines.append("")
    lines.append(
        "A×C = nombre d'algorithmes (5) multiplié par le nombre de couples "
        "(nœuds, période). Le runtime est donné à titre indicatif sur une "
        "machine 8 cœurs."
    )
    return "\n".join(lines)
