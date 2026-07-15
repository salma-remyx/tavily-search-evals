"""Pytest configuration: ensure the repository root is importable.

With the default ``prepend`` import mode, pytest inserts the test file's own
directory (``tests/``) onto ``sys.path`` rather than the repository root. This
root-level ``conftest.py`` makes the repo root importable so that ``utils`` and
other top-level packages resolve when tests are collected.
"""
