import os
import random
import shutil
import sys

import pytest

# Ensure the project root is on the module search path when the package is not
# installed. This allows ``import loraflexsim`` to succeed during
# test collection without requiring an editable installation.
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Keep lightweight stubs importable for tests that explicitly use them
# (e.g. ``import numpy_stub``), without shadowing real third-party packages.
STUBS_DIR = os.path.join(ROOT_DIR, "tests", "stubs")
if STUBS_DIR not in sys.path:
    sys.path.insert(0, STUBS_DIR)


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
