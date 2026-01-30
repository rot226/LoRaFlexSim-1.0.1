"""Utilitaires communs pour configurer les figures du step 1."""

from __future__ import annotations

import math
from collections.abc import Iterable

import matplotlib.pyplot as plt

from article_c.common.plot_helpers import (
    apply_figure_layout,
    fallback_legend_handles,
    legend_margins,
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
    title: str,
    legend_loc: str,
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
            if legend_loc == "above":
                legend_style = {
                    **LEGEND_STYLE,
                    "ncol": min(len(labels), LEGEND_STYLE.get("ncol", 3)),
                }
                legend = fig.legend(handles, labels, **legend_style)
                ncol = int(legend_style.get("ncol", len(labels)) or 1)
                legend_rows = max(1, math.ceil(len(labels) / ncol))
                bbox_to_anchor = legend_bbox_to_anchor(
                    legend=legend,
                    legend_rows=legend_rows,
                )
                legend.set_bbox_to_anchor(bbox_to_anchor)
                apply_figure_layout(
                    fig,
                    bbox_to_anchor=bbox_to_anchor,
                    margins=legend_margins("above", legend_rows=legend_rows),
                    legend_rows=legend_rows,
                )
            else:
                fig.legend(
                    handles,
                    labels,
                    loc="center left",
                    frameon=False,
                )
                apply_figure_layout(fig, bbox_to_anchor=(1.02, 0.5))

    if legend_loc == "above":
        above_margins = {
            "top": legend_margins("above", legend_rows=legend_rows)["top"],
            "bottom": FIGURE_MARGINS["bottom"],
        }
        apply_figure_layout(
            fig,
            margins=above_margins,
            tight_layout={
                "rect": (0, above_margins["bottom"], 1, above_margins["top"])
            },
            legend_rows=legend_rows,
        )
    else:
        apply_figure_layout(
            fig,
            margins={
                "top": FIGURE_MARGINS["top"],
                "bottom": FIGURE_MARGINS["bottom"],
                "right": 0.80,
            },
            tight_layout={
                "rect": (
                    0,
                    FIGURE_MARGINS["bottom"],
                    0.80,
                    FIGURE_MARGINS["top"],
                )
            },
        )
    fig.suptitle(title, y=suptitle_y_from_top(fig))
