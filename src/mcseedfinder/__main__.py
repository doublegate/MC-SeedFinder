"""Module entry point: enables ``python -m mcseedfinder``."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
