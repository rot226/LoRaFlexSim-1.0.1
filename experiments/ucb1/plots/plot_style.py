"""Utilitaires de style et de sélection pour les figures UCB1."""
from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plotting_style import apply_base_rcparams

def apply_ieee_style() -> None:
    """Applique un style proche des recommandations IEEE."""
    apply_base_rcparams()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
        }
    )


def top_groups(df: pd.DataFrame, group_cols: Iterable[str], max_groups: int = 3) -> list[tuple]:
    """Retourne les groupes les plus fréquents (max 3) pour limiter le nombre de courbes."""
    if df.empty:
        return []
    sizes = df.groupby(list(group_cols)).size().sort_values(ascending=False)
    top = sizes.head(max_groups).index.tolist()
    normalized: list[tuple] = []
    for item in top:
        if isinstance(item, tuple):
            normalized.append(item)
        else:
            normalized.append((item,))
    return normalized


def filter_top_groups(df: pd.DataFrame, group_cols: Iterable[str], max_groups: int = 3) -> pd.DataFrame:
    """Filtre un DataFrame pour ne conserver que les groupes principaux."""
    groups = top_groups(df, group_cols, max_groups=max_groups)
    if not groups:
        return df
    group_cols_list = list(group_cols)
    if len(group_cols_list) == 1:
        values = [group[0] for group in groups]
        return df[df[group_cols_list[0]].isin(values)]
    groups_df = pd.DataFrame(groups, columns=group_cols_list)
    return df.merge(groups_df, on=group_cols_list, how="inner")
