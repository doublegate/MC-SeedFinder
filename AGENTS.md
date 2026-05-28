# Repository Guidelines

## Project Structure & Module Organization

This is a Python package using a `src/` layout. Core code lives in `src/mcseedfinder/`: `cli.py` handles the command line, `finder.py` coordinates searches, `criteria.py` compiles filters, and `java_random.py` plus `structures.py` contain the accuracy-critical Minecraft math. Tests live in `tests/` and currently cover Java RNG behavior, structure placement, and criteria compilation. Example JSON search specs live in `examples/`. Treat `src/mcseedfinder.egg-info/` and `__pycache__/` as generated artifacts.

## Build, Test, and Development Commands

- `pip install -e ".[dev]"`: install the package in editable mode with pytest, coverage, and mypy extras.
- `python -m mcseedfinder --help`: verify the CLI entry point and inspect available flags.
- `python -m mcseedfinder --show-seed 1`: run a quick local smoke check without a full seed search.
- `python -m unittest`: run the current test suite with the standard library runner.
- `pytest`: run the same tests with pytest, useful when using `pytest-cov`.
- `python -m mcseedfinder --config examples/village_outpost.json --count 100000 --workers 4 --max-matches 1`: exercise a representative configured search.

## Coding Style & Naming Conventions

Use Python 3.10+ features and keep runtime dependencies out of core code unless `pyproject.toml` changes intentionally. Follow the existing style: 4-space indentation, `from __future__ import annotations`, type hints on public functions, `snake_case` functions and modules, `PascalCase` classes, and `UPPER_SNAKE_CASE` constants. Prefer small pure functions for deterministic math. Keep comments focused on provenance, reference vectors, or non-obvious algorithm details.

## Testing Guidelines

Tests use `unittest` classes and methods named `test_*`, and they are pytest-compatible. Place new tests in `tests/test_<module>.py`. For `java_random.py` and `structures.py`, add or update canonical vectors when behavior changes; accidental drift there can invalidate every search result. For criteria and CLI behavior, include both positive and negative cases and prefer small seed counts.

## Commit & Pull Request Guidelines

The existing history uses a Conventional Commit style such as `chore(workspace): ...`. Continue with `type(scope): imperative summary`, for example `fix(structures): correct monument spacing`. Pull requests should describe the behavioral change, list commands run, link related issues, and call out any accuracy impact, biome fallback behavior, or CLI compatibility change. Include example commands or output when changing user-facing search behavior.

## Security & Configuration Tips

The package is stdlib-only at runtime. `cubiomes-py` is optional for accurate biome filtering and should remain behind the `accurate` extra. Keep matching seeds on stdout and diagnostics on stderr so scripts can safely pipe results.
