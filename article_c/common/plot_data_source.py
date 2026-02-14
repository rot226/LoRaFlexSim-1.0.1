"""Résolution des sources CSV contractuelles pour les modules de tracé."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import warnings

RowLoader = Callable[..., list[dict[str, object]]]

CONTRACTUAL_SOURCES = {"aggregates", "by_size"}


def load_aggregated_rows_for_source(
    *,
    step_dir: Path,
    source: str,
    step_label: str,
    loader: RowLoader,
    allow_sample: bool,
    csv_name: str = "aggregated_results.csv",
) -> list[dict[str, object]]:
    """Charge les lignes agrégées selon la source contractuelle demandée.

    - aggregates: lit `results/aggregates/<csv_name>`
    - by_size: concatène `results/by_size/size_*/<csv_name>`
    """
    normalized_source = str(source).strip().lower()
    if normalized_source not in CONTRACTUAL_SOURCES:
        warnings.warn(
            (
                f"Source inconnue '{source}' pour {step_label}; "
                "valeurs supportées: aggregates, by_size."
            ),
            stacklevel=2,
        )
        return []

    results_dir = step_dir / "results"
    if normalized_source == "aggregates":
        path = results_dir / "aggregates" / csv_name
        if not path.exists():
            warnings.warn(
                f"CSV {step_label} introuvable ({normalized_source}): {path}",
                stacklevel=2,
            )
            return []
        try:
            return loader(path, allow_sample=allow_sample)
        except (OSError, ValueError) as exc:
            warnings.warn(
                f"CSV {step_label} illisible ({path}): {exc}",
                stacklevel=2,
            )
            return []

    by_size_paths = sorted(results_dir.glob(f"by_size/size_*/{csv_name}"))
    if not by_size_paths:
        warnings.warn(
            (
                f"CSV {step_label} introuvable ({normalized_source}); "
                f"motif attendu: {results_dir / 'by_size' / 'size_*' / csv_name}"
            ),
            stacklevel=2,
        )
        return []

    merged_rows: list[dict[str, object]] = []
    for path in by_size_paths:
        try:
            merged_rows.extend(loader(path, allow_sample=allow_sample))
        except (OSError, ValueError) as exc:
            warnings.warn(
                f"CSV {step_label} ignoré ({path}): {exc}",
                stacklevel=2,
            )
    if not merged_rows:
        warnings.warn(
            f"Aucune donnée exploitable trouvée pour {step_label} avec source={normalized_source}.",
            stacklevel=2,
        )
    return merged_rows
