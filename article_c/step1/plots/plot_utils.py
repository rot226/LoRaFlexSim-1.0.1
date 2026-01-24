"""Utilitaires communs pour configurer les figures du step 1."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt

from article_c.common.plot_helpers import SUPTITLE_Y, apply_figure_layout
from article_c.common.plotting_style import LEGEND_STYLE


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
) -> None:
    """Configure le titre, la légende et les marges de la figure.

    legend_loc doit valoir "above" (légende au-dessus) ou "right" (à droite).
    """
    if legend_loc not in {"above", "right"}:
        raise ValueError("legend_loc doit valoir 'above' ou 'right'.")

    axes_list = _flatten_axes(axes)
    if not fig.legends:
        handles: list[object] = []
        labels: list[str] = []
        for ax in axes_list:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
        if handles:
            if legend_loc == "above":
                legend_style = {
                    **LEGEND_STYLE,
                    "ncol": min(len(labels), LEGEND_STYLE.get("ncol", 3)),
                }
                bbox_to_anchor = legend_style.pop("bbox_to_anchor", None)
                fig.legend(handles, labels, **legend_style)
                apply_figure_layout(fig, bbox_to_anchor=bbox_to_anchor)
            else:
                fig.legend(
                    handles,
                    labels,
                    loc="center left",
                    frameon=False,
                )
                apply_figure_layout(fig, bbox_to_anchor=(1.02, 0.5))

    fig.suptitle(title, y=SUPTITLE_Y)
    if legend_loc == "above":
        apply_figure_layout(
            fig,
            margins={"top": 0.80},
            tight_layout={"rect": (0, 0, 1, 0.88)},
        )
    else:
        apply_figure_layout(
            fig,
            margins={"top": 0.88, "right": 0.80},
            tight_layout={"rect": (0, 0, 0.80, 1)},
        )
