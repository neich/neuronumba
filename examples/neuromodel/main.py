"""Neuromodel — entry point.

Run from the repository root:

    python examples/neuromodel/main.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _main() -> None:
    from app import main
    main()


if __name__ == "__main__":
    _main()
