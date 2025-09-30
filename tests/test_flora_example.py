import runpy
from pathlib import Path

import pytest

try:
    pytest.importorskip('pandas')
except Exception:
    pytest.skip('pandas import failed', allow_module_level=True)

from loraflexsim.launcher.compare_flora import compare_with_sim


def test_flora_example_matches_flora():
    # Execute the example script as if run directly
    globals_dict = runpy.run_path('examples/run_flora_example.py', run_name='__main__')
    sim = globals_dict['sim']
    metrics = sim.get_metrics()
    flora_csv = Path(__file__).parent / 'data' / 'n100_gw1_expected.csv'
    assert compare_with_sim(metrics, flora_csv)
