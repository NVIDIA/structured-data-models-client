# Reference environment (T0.3)

The reference is the branch `rust-core-reference` of `NVIDIA/structured-data-models-client`, checked out as a worktree at `~/code/sdmc-reference`.
It is `main` at `fa0637d` plus the fixes listed in [determinism-fixes.md](determinism-fixes.md); its head is `70aa159`.
The unpatched baseline is tagged locally as `reference-baseline-fa0637d` (`fa0637d495b5098e1cf7295e46ac7546dc0bcb01`); the tag is not pushed, and its GitHub ruleset is still to be made.

## Versions

| Component | Version |
|---|---|
| Platform | macOS 26.7.1, arm64 |
| Python | 3.12.13 (uv-managed CPython) |
| pandas | 3.0.6 |
| NumPy | 2.5.3 |
| PyArrow | 25.0.1 |
| torch | Not installed |
| DuckDB | 1.5.6 |
| ADBC SQLite driver | 1.12.0 |
| psycopg | 3.3.6 |
| databricks-sql-connector | 4.6.0 |
| pytest | 9.1.1 |
| uv | 0.11.24 |

torch is not a dependency of any package, so the reference runs without it.
The "with torch" runs of the repeatability check installed torch 2.14.1 into the environment for the run, after which the environment was rebuilt from scratch without it.
The full `uv pip freeze` (109 lines, sha256 prefix `346b292a6915d641`) is kept with the harness in `~/.local/share/sdm-dbx/rust-core/t03-harness/reference-freeze.txt`.

## How it was installed

These are the commands of `CONTRIBUTING.md`, with uv in place of `venv` and `pip`:

```bash
cd ~/code/sdmc-reference
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python ruff mypy \
  -e './packages/structured-data-models-api[test]' \
  -e './packages/structured-data-models-pql[test]' \
  -e './packages/structured-data-models-client[test]' \
  -e './packages/structured-data-models-connectors[sqlite,duckdb,postgres,databricks,test]'
uv pip install --python .venv/bin/python \
  -e './packages/structured-data-models-relational-engine[test]'
```

The engine's existing native sampler, `relationallib`, is part of today's Python and is not a Rust core module.

## The guard

`scripts/reference_guard.py` on the reference branch exits with status 1 when a Rust core module is imported or importable from the interpreter running it, and with status 0 otherwise.
It checks 24 names: the top-level `sdm_core`, `sdm_py`, `_native` and `sdm_core_py`, and `_native`, `_core`, `sdm_core` and `sdm_py` under each of the five import roots.
Run it at the start of every reference session:

```bash
~/code/sdmc-reference/.venv/bin/python ~/code/sdmc-reference/scripts/reference_guard.py
```

It was shown to work as follows:

- In the clean environment it prints `reference guard: none of 24 names load` and exits 0.
- With a stub package `sdm_core/__init__.py` on `PYTHONPATH` it reports `sdm_core is importable from <path>` and exits 1.
- With a stub `_native.py` on `PYTHONPATH` it reports `_native` and exits 1.
- With an empty `sdm_relational_engine/_native.py` placed in the engine's source for the test, then removed, it reports `sdm_relational_engine._native` and exits 1.
- With a module object for `sdm_py` put in `sys.modules` before it runs, it reports `sdm_py is imported` and exits 1.

## Repeatability

The method and results are in [determinism-fixes.md](determinism-fixes.md#repeatability-check).
In short, on the reference the wire payloads and the ported checks are byte-identical, and the tie and push-down payloads equal once parsed, under `PYTHONHASHSEED=0` and `=1`, with and without torch.
