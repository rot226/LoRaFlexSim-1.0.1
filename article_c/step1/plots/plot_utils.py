"""Utilitaires communs pour configurer les figures du step 1."""

from __future__ import annotations

import math
from collections.abc import Iterable

import matplotlib.pyplot as plt

from article_c.common.plot_helpers import (
    apply_figure_layout,
    deduplicate_legend_entries,
    fallback_legend_handles,
    legend_margins,
    legend_ncols,
    place_adaptive_legend,
    suptitle_y_from_top,
)
from article_c.common.plotting_style import FIGURE_MARGINS, LEGEND_STYLE
from article_c.common.plotting_style import legend_bbox_to_anchor


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
) -> None:
    """Configure le titre, la légende et les marges de la figure.

    legend_loc doit valoir "above" (légende au-dessus) ou "right" (à droite).
    """
    if legend_loc not in {"above", "right"}:
        raise ValueError("legend_loc doit valoir 'above' ou 'right'.")

    axes_list = _flatten_axes(axes)
    legend_rows = 1
    if not fig.legends:
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
            if legend_loc == "above":
                ncol = min(len(labels), int(LEGEND_STYLE.get("ncol", len(labels)) or 1))
                legend_rows = max(1, math.ceil(len(labels) / max(1, ncol)))
            placement = place_adaptive_legend(
                fig,
                axes_list[0],
                preferred_loc=legend_loc,
                handles=handles,
                labels=labels,
            )
            legend_rows = placement.legend_rows
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

    if legend_loc == "above":
        above_margins = (
            {
                **legend_margins("above", legend_rows=legend_rows),
                "bottom": FIGURE_MARGINS["bottom"],
            }
            if adjust_layout_for_legend
            else FIGURE_MARGINS
        )
        apply_figure_layout(
            fig,
            margins=above_margins,
            legend_rows=legend_rows,
            legend_loc=legend_loc,
        )
    else:
        apply_figure_layout(
            fig,
            margins=(
                {
                    **legend_margins("right"),
                    "bottom": FIGURE_MARGINS["bottom"],
                }
                if adjust_layout_for_legend
                else FIGURE_MARGINS
            ),
            legend_loc=legend_loc,
        )
    if title:
        fig.suptitle(title, y=suptitle_y_from_top(fig))
