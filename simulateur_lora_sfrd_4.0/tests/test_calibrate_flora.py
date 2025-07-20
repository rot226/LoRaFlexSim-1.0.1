import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("pandas")

from tools.calibrate_flora import calibrate  # noqa: E402


def test_calibrate_flora_quick():
    ref = Path(__file__).parent / "data" / "flora_full.csv"
    params, err = calibrate(
        ref,
        runs=1,
        path_loss_values=(2.7,),
        shadowing_values=(6.0,),
        seed=0,
        advanced=True,
    )
    assert params is not None
    assert err >= 0.0
