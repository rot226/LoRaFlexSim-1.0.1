import sys
from pathlib import Path
import pytest

pytest.importorskip("pandas")
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.compare_flora_multi import compare_multiple  # noqa: E402


def test_compare_multiple_zero_diff():
    csv = Path(__file__).parent / "data" / "flora_sample.csv"
    df = compare_multiple(csv, [csv])
    assert isinstance(df, pd.DataFrame)
    assert df.loc[0, "PDR_diff"] == 0
    assert df.loc[0, "collisions_diff"] == 0
