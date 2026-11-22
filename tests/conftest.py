import sys
from pathlib import Path

# Make `from src import data` work when running `pytest` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
