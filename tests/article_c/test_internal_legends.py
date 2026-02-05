from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from article_c.common.plot_helpers import add_global_legend, find_internal_legends


def test_add_global_legend_clears_axis_legends() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    for index, ax in enumerate(axes, start=1):
        ax.plot([0, 1], [index, index + 1], label=f"Série {index}")
        ax.legend()

    assert find_internal_legends(fig)

    add_global_legend(fig, axes[0], legend_loc="right")

    assert not find_internal_legends(fig)
    plt.close(fig)
