# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Codex round 3 on T3.2 (fix 8): a row read whose inferred Arrow type
changes a decimal's scale is still the rows read, and a changed Rust source
file's inline tests are the package's own tests."""

from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUN = _module(ROOT / 'contract/runners/run_python.py', 't32_r3_run')
CHECK = _module(ROOT / 'contract/runners/capture_check.py', 't32_r3_check')
HANDOFF = _module(ROOT / 'scripts/rust_core/handoff.py', 't32_r3_handoff')
ROWS = [(Decimal('1.0'),), (Decimal('2.00'),)]


class _Driver:
    description = [('amount', 'numeric')]

    def stream(self, *args: object, **kw: object) -> object:
        yield from ROWS


def test_a_row_read_whose_decimal_scale_arrow_changes_is_recorded() -> None:
    log = RUN.DriverLog()
    log.recorder = RUN.Recorder()
    log.stage = 'materialize'
    cursor = RUN._CapturedPsycopg(_Driver(), log)
    assert list(cursor.stream('SELECT amount FROM t')) == ROWS
    table = RUN.rows_table(*log.recorder.kept[(0, 0)])
    assert table.to_pylist() == [
        {'amount': Decimal('1.00')},
        {'amount': Decimal('2.00')},
    ]
    assert RUN.same_rows(table, ROWS)
    result = {**log.statements[0]['results'][0], 'arrow': table}
    line = {
        'case': 'probe@lakebase',
        'statements': [{**log.statements[0], 'results': [result]}],
    }
    RUN.written(line, RUN.Placeheld({}, []))
    # The saved table is held to what was kept: its own digest, recorded.
    assert result['held'] == RUN.saved_digest(result, table)
    # A table that is not the rows read, value for value, is not.
    assert not RUN.same_rows(
        RUN.rows_table(['amount'], [(Decimal('9'),), (Decimal('2'),)]), ROWS
    )


def test_a_changed_source_file_s_inline_tests_are_own_tests(
    tmp_path: Path,
) -> None:
    source = 'rust/crates/sdm-core/src/thing.rs'
    plain = 'rust/crates/sdm-core/src/other.rs'
    for path, text in (
        (source, 'fn f() {}\n#[cfg(test)]\nmod tests {}\n'),
        (plain, 'fn g() {}\n'),
    ):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
    suites = HANDOFF.own_suites(tmp_path, [source, plain])
    assert suites == [
        (
            'cargo',
            'test',
            '--manifest-path',
            'rust/Cargo.toml',
            '-p',
            'sdm-core',
            '--lib',
        )
    ], suites
