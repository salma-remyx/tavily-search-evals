"""Shared pytest configuration.

Ensures the repository root is importable so tests can do
``import utils`` / ``import evaluators`` regardless of the directory
pytest is invoked from.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
