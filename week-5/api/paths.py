"""Locations of the other weeks, and a helper that makes their packages importable."""

import sys
from pathlib import Path

WEEK5 = Path(__file__).resolve().parents[1]
REPO = WEEK5.parent
WEEK1, WEEK2, WEEK3, WEEK4 = (REPO / f"week-{n}" for n in (1, 2, 3, 4))


def add_other_weeks() -> None:
    """Make `rag` (Week 3) and `nlpmeta` (Week 4) importable.

    The folders are appended, not inserted first, so Week 5's own packages
    (`api`, `evals`, `tests`) always win when names are the same.
    """
    for folder, package in ((WEEK3, "rag"), (WEEK4, "nlpmeta")):
        if not (folder / package).is_dir():
            raise FileNotFoundError(f"{folder} not found. Week 5 needs the week-3 and week-4 folders.")
        if str(folder) not in sys.path:
            sys.path.append(str(folder))
