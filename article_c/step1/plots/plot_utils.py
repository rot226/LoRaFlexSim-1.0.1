"""Utilitaires communs pour configurer les figures du step 1."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
import warnings

import matplotlib.pyplot as plt

from article_c.common.plot_helpers import load_step1_aggregated

from article_c.common.plot_helpers import (
    apply_figure_layout,
    deduplicate_legend_entries,
    fallback_legend_handles,
    legend_margins,
    legend_ncols,
    add_global_legend,
    place_adaptive_legend,
)
from article_c.common.plot_style import FIGURE_MARGINS, LEGEND_STYLE
from article_c.common.plot_style import legend_bbox_to_anchor

_STEP1_AGGREGATED_NAME = "aggregated_results.csv"
_STEP1_MISSING_HINT = (
    "Exécutez le préflight de make_all_plots.py pour générer "
    "results/aggregates/aggregated_results.csv avant les modules."
)


def _load_step1_rows_from_by_size(
    step_dir: Path,
    *,
    allow_sample: bool,
) -> list[dict[str, object]]:
    by_size_paths = sorted(
        (step_dir / "results").glob(f"by_size/size_*/{_STEP1_AGGREGATED_NAME}")
    )
    if not by_size_paths:
        expected_pattern = (
            step_dir / "results" / "by_size" / "size_*" / _STEP1_AGGREGATED_NAME
        )
        raise FileNotFoundError(
            f"CSV introuvable via by_size: motif attendu {expected_pattern}."
        )
    merged_rows: list[dict[str, object]] = []
    for csv_path in by_size_paths:
        merged_rows.extend(load_step1_aggregated(csv_path, allow_sample=allow_sample))
    return merged_rows


def load_step1_rows_with_fallback(
    step_dir: Path,
    *,
    allow_sample: bool = False,
    source: str = "aggregates",
    allow_by_size_fallback: bool = True,
) -> list[dict[str, object]]:
    """Charge les lignes agrégées Step1 depuis la source demandée."""
    normalized_source = str(source).strip().lower()
    if normalized_source not in {"aggregates", "by_size"}:
        raise ValueError(
            "Source CSV non supportée. Utilisez source='aggregates' ou source='by_size'."
        )

    if normalized_source == "by_size":
        return _load_step1_rows_from_by_size(step_dir, allow_sample=allow_sample)

    primary_path = (
        step_dir / "results" / "aggregates" / _STEP1_AGGREGATED_NAME
    )
    if primary_path.is_file():
        return load_step1_aggregated(primary_path, allow_sample=allow_sample)

    missing_aggregate_message = (
        f"CSV introuvable: {primary_path}. {_STEP1_MISSING_HINT}"
    )
    if allow_by_size_fallback:
        warnings.warn(
            (
                f"{missing_aggregate_message} "
                "Tentative de repli sur results/by_size/size_*/aggregated_results.csv."
            ),
            stacklevel=2,
        )
        return _load_step1_rows_from_by_size(step_dir, allow_sample=allow_sample)

    raise FileNotFoundError(missing_aggregate_message)


def _flatten_axes(axes: object) -> list[plt.Axes]:
    if isinstance(axes, plt.Axes):
        return [axes]
    if hasattr(axes, "flat"):
        return list(axes.flat)
    if isinstance(axes, Iterable):
        flattened: list[plt.Axes] = []
        for item in axes:
            if isinstance(item, plt.Axes):
                flattened.append(item)
            elif isinstance(item, Iterable):
                flattened.extend([ax for ax in item if isinstance(ax, plt.Axes)])
        return flattened
    return []


def configure_figure(
    fig: plt.Figure,
    axes: object,
    title: str | None = None,
    legend_loc: str = "right",
    legend_handles: list[object] | None = None,
    legend_labels: list[str] | None = None,
    enable_suptitle: bool = False,
    wspace: float | None = None,
) -> tuple[str, int]:
    """Configure la légende et les marges de la figure (sans titre).

    legend_loc doit valoir "above" (légende au-dessus) ou "right" (à droite).
    """
    if legend_loc not in {"above", "right"}:
        raise ValueError("legend_loc doit valoir 'above' ou 'right'.")

    axes_list = _flatten_axes(axes)
    legend_rows = 1
    final_legend_loc = legend_loc
    if not fig.legends and not any(ax.get_legend() is not None for ax in fig.axes):
        handles: list[object] = []
        labels: list[str] = []
        if legend_handles is not None:
            handles = legend_handles
            if legend_labels is not None:
                labels = legend_labels
            else:
                labels = [handle.get_label() for handle in handles]
        else:
            for ax in axes_list:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    break
        if not handles:
            handles, labels = fallback_legend_handles()
        if handles:
            handles, labels = deduplicate_legend_entries(handles, labels)
        if handles:
            if len(axes_list) > 1:
                add_global_legend(
                    fig,
                    axes_list,
                    legend_loc=legend_loc,
                    handles=handles,
                    labels=labels,
                    use_fallback=False,
                )
                final_legend_loc = legend_loc
            else:
                if legend_loc == "above":
                    ncol = min(
                        len(labels),
                        int(LEGEND_STYLE.get("ncol", len(labels)) or 1),
                    )
                    legend_rows = max(1, math.ceil(len(labels) / max(1, ncol)))
                placement = place_adaptive_legend(
                    fig,
                    axes_list[0],
                    preferred_loc=legend_loc,
                    handles=handles,
                    labels=labels,
                    enable_suptitle=enable_suptitle,
                )
                legend_rows = placement.legend_rows
                final_legend_loc = placement.legend_loc
    legend_in_figure = bool(fig.legends)
    legend_entry_count = 0
    if legend_in_figure:
        legend = fig.legends[0]
        legend_entry_count = len(legend.get_texts())
        legend_cols_default = int(LEGEND_STYLE.get("ncol", 1) or 1)
        legend_cols = legend_ncols(legend, legend_cols_default)
        legend_rows = max(
            1,
            math.ceil(legend_entry_count / max(1, legend_cols)),
        )
    else:
        legend_rows = 1
    adjust_layout_for_legend = legend_in_figure and legend_entry_count > 1

    if final_legend_loc == "above":
        above_margins = (
            {
                **legend_margins("above", legend_rows=legend_rows),
                "bottom": FIGURE_MARGINS["bottom"],
            }
            if adjust_layout_for_legend
            else FIGURE_MARGINS
        )
        if wspace is not None:
            above_margins = {**above_margins, "wspace": wspace}
        apply_figure_layout(
            fig,
            margins=above_margins,
            legend_rows=legend_rows,
            legend_loc=final_legend_loc,
        )
    else:
        if final_legend_loc == "right":
            margins = (
                {
                    **legend_margins("right"),
                    "bottom": FIGURE_MARGINS["bottom"],
                }
                if adjust_layout_for_legend
                else FIGURE_MARGINS
            )
            if wspace is not None:
                margins = {**margins, "wspace": wspace}
            apply_figure_layout(fig, margins=margins, legend_loc=final_legend_loc)
        else:
            margins = FIGURE_MARGINS
            if wspace is not None:
                margins = {**margins, "wspace": wspace}
            apply_figure_layout(fig, margins=margins)
    _ = (title, enable_suptitle)
    return final_legend_loc, legend_rows
