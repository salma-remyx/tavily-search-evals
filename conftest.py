"""Pytest bootstrap: ensure the repository root is importable.

This repo ships no package metadata, so without this the test modules could
not import top-level packages such as ``utils`` or ``evaluators``.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
