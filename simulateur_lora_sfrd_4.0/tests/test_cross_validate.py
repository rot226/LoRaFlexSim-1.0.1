import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("pandas")

from tools.calibrate_flora import cross_validate  # noqa: E402


def test_cross_validate_multi():
    csv1 = Path(__file__).parent / "data" / "flora_sample.csv"
    csv2 = Path(__file__).parent / "data" / "flora_full.csv"
    params, err = cross_validate([csv1, csv2], runs=1, path_loss_values=(2.7,), shadowing_values=(6.0,), seed=0)
    assert params is not None
    assert err >= 0.0
