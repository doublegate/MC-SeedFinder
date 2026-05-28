---
name: verify-all
description: Run the full mc-seed-finder verification gate across Python, Rust, and the desktop app. Use before committing, or when asked to verify the whole project builds and tests pass.
---

Run these from the repo root, in order, and report pass/fail for each stage:

1. Python tests: `PYTHONPATH=src python -m unittest`
2. Rust tests: `RUSTC_WRAPPER= cargo test --features pyo3`
3. Desktop build + check: `cd desktop && npm run build && RUSTC_WRAPPER= cargo check --manifest-path src-tauri/Cargo.toml`

Notes:

- Always prefix Rust commands with `RUSTC_WRAPPER=` to disable `sccache` when it is configured but cannot run in this environment.
- Plain `cargo test` does not exercise the PyO3 bindings — the `--features pyo3` flag is required.
- Stop and surface the first failing stage with its output. Do not attempt fixes unless the user asks.
- If the native extension or `cubiomes` submodule is involved, ensure submodules are initialized (`git submodule update --init`) before the Rust stage.
