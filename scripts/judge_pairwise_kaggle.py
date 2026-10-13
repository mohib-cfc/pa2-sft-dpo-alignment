"""CLI wrapper exposing the manual's judge flags (see pa2/judge.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pa2.judge import main  # noqa: E402

if __name__ == "__main__":
    main()
