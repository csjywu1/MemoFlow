"""Evaluate a trained MemoFlow checkpoint."""

from __future__ import annotations

import sys

from src.train import main as train_main


def main() -> None:
    if "--eval-only" not in sys.argv:
        sys.argv.append("--eval-only")
    train_main()


if __name__ == "__main__":
    main()
