import os
import sys

# Make the repository root importable so tests can import top-level modules
# (``run_evaluation``) and the ``evaluators`` / ``utils`` packages regardless
# of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
