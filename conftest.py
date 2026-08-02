import os
import sys

# Make the repository root importable so tests can do `import utils...`
# regardless of pytest's import mode or invocation directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
