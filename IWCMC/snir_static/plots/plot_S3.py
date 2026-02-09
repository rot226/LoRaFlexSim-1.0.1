"""Trace la figure S3 pour la campagne SNIR statique."""

from __future__ import annotations

import sys
from pathlib import Path

PLOTS_DIR = Path(__file__).resolve().parent
if str(PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PLOTS_DIR))

from plot_snir_static_common import main_for_figure


if __name__ == "__main__":
    main_for_figure("S3")
