"""Single entry point for the handwritten digit recognition project.

Run from the project root:
    python main.py

This performs the full training/evaluation workflow and writes all results.
"""

from __future__ import annotations

from src.train import main


if __name__ == "__main__":
    main()
