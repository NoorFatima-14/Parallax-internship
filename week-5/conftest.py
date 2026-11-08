"""pytest setup: make sure week-5's own packages (api, evals, tests) are found first."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
