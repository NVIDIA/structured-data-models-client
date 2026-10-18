# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.4: the SQLite and Postgres contract jobs run in every cell, and
python-old does not run off DuckDB (T2.1a fix 4), so ``compare.py``'s H1
pair (``_cell_pair``) takes python-core alone in a cell other than the
reference's there: its host fields stay reported against the reference,
and every core rule still gates it. On DuckDB the pair still needs both
runners, and where python-old is given off DuckDB the two are still held
to each other."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _compare() -> ModuleType:
    name = 't24_compare'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'scripts/rust_core/compare.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _line(runner: str, cell: str) -> dict[str, Any]:
    return {'identity': {'runner': runner, 'cell': cell}}


def _failures(
    core: dict[str, Any] | None,
    old: dict[str, Any] | None,
    on_duckdb: bool,
) -> list[tuple[str | None, str]]:
    compare = _compare()
    report = compare.Report()
    compare._cell_pair(
        report, {'id': 'a.case'}, core, old, None, on_duckdb=on_duckdb
    )
    return report.failures


def test_python_core_alone_in_another_cell_off_duckdb_is_no_pair_failure() -> (
    None
):
    for cell in ('floor-pandas2.2', 'dbr17-pandas2'):
        assert _failures(_line('python-core', cell), None, False) == []


def test_on_duckdb_the_pair_still_needs_python_old() -> None:
    [(_, text)] = _failures(_line('python-core', 'floor-pandas2.2'), None, True)
    assert 'python-core and python-old are both needed' in text


def test_off_duckdb_a_given_python_old_is_still_held_to_python_core() -> None:
    # Two cells: the pair is still compared, and fails by name.
    [(_, text)] = _failures(
        _line('python-core', 'floor-pandas2.2'),
        _line('python-old', 'dbr17-pandas2'),
        False,
    )
    assert 'python-core and python-old in different cells' in text
