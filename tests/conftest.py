import random
import os
import sys
import shutil
import pytest

# Ensure the project root is on the module search path when the package is not
# installed. This allows ``import loraflexsim`` to succeed during
# test collection without requiring an editable installation. A local stub of
# :mod:`numpy` called ``numpy_stub`` lives under ``tests/stubs`` and must appear
# *before* the project root so that unit tests can run without the real
# dependency.
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

STUBS_DIR = os.path.join(ROOT_DIR, "tests", "stubs")
if STUBS_DIR not in sys.path:
    sys.path.insert(0, STUBS_DIR)

import numpy_stub
import scipy

sys.modules.setdefault("numpy", numpy_stub)
sys.modules.setdefault("numpy.random", numpy_stub.random)
sys.modules.setdefault("scipy", scipy)
sys.modules.setdefault("scipy.stats", scipy.stats)

@pytest.fixture(autouse=True)
def _set_seed():
    random.seed(1)


@pytest.fixture(autouse=True)
def _cleanup_tmp_path(tmp_path):
    """Remove temporary files created during tests."""
    yield
    for path in tmp_path.iterdir():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink()
