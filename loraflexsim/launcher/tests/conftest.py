"""Test configuration for launcher tests.

Ensures that the project root is on ``sys.path`` so that the
``loraflexsim`` package can be imported when tests are executed from
within the ``loraflexsim/launcher`` subpackage.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Determine the project root (three levels up from this file) and insert it
# into ``sys.path`` if it's not already present.  This allows the tests to
# import ``loraflexsim`` without requiring the package to be installed.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
