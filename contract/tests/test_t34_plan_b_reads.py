# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.4, round 16: the reads plan B adds to a core run (each table read as
it stands at each materialize, the owner's decision of 2026-10-24) are
taken out of a comparison with the old client by the mark the core puts
on them, the fetch's purpose (``locate``), and by nothing else: a
statement of the old client's is never taken out, and one of the core's
the core did not mark is compared, its parameters included.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _compare() -> Any:
    if 'compare' not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            'compare', ROOT / 'scripts/rust_core/compare.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules['compare'] = module
        spec.loader.exec_module(module)
    return sys.modules['compare']


def _statement(
    stage: str, sql: str, parameters: Any = None, purpose: str | None = None
) -> dict[str, Any]:
    statement: dict[str, Any] = {
        'handle': 1,
        'stage': stage,
        'sql': sql,
        'parameters': parameters,
        'results': [],
        'error': None,
    }
    if purpose is not None:
        statement['purpose'] = purpose
    return statement


def _observation(statements: list[dict[str, Any]]) -> dict[str, Any]:
    return {'driver': {'capture': {'statements': statements, 'bindings': []}}}


def _materialize(observation: dict[str, Any]) -> list[tuple[str, Any]]:
    grouped = _compare()._by_stage(observation) or {}
    return [(s['sql'], s['parameters']) for s in grouped.get('materialize', [])]


def test_a_core_run_skipping_the_old_client_s_key_reads_fails_the_comparison() -> (
    None
):
    # Claude's round 16 probe: the old client reads each table's indexes at
    # materialize; a core run reading its catalog instead, then sampling,
    # is not the old client's run.
    described = [
        _statement('from_source', 'PRAGMA table_info("users")'),
        _statement('from_source', 'PRAGMA index_list("users")'),
    ]
    reference = _observation(
        [
            *described,
            _statement('materialize', 'PRAGMA index_list("users")'),
            _statement('materialize', 'SAMPLE X'),
        ]
    )
    core = _observation(
        [
            *described,
            _statement('materialize', 'PRAGMA table_info("users")'),
            _statement('materialize', 'SAMPLE X'),
        ]
    )
    assert _materialize(core) != _materialize(reference)
    assert _compare().counts(core) == _compare().counts(reference)
    assert _materialize(reference) == [
        ('PRAGMA index_list("users")', None),
        ('SAMPLE X', None),
    ]


def test_a_repeated_statement_bound_to_another_table_is_compared() -> None:
    # Codex's round 16 probe: a statement of the core's that repeats one
    # from_source ran, bound to other parameters and not marked as plan B's,
    # is compared, so a read of another schema fails the comparison.
    catalog = 'SELECT table_name FROM information_schema.tables WHERE table_schema = ?'
    described = [_statement('from_source', catalog, ['public'])]
    reference = _observation(
        [*described, _statement('materialize', 'SAMPLE X')]
    )
    core = _observation(
        [
            *described,
            _statement('materialize', catalog, ['wrong_schema']),
            _statement('materialize', 'SAMPLE X'),
        ]
    )
    assert _compare().counts(core) != _compare().counts(reference)
    assert _materialize(core)[0] == (catalog, ['wrong_schema'])


def test_the_reads_the_core_marks_as_plan_b_s_are_taken_out() -> None:
    # The reads the core marks (purpose `locate`) are taken out of its run,
    # whatever their text; every other statement stays.
    reference = _observation(
        [
            _statement('from_source', 'PRAGMA table_info("users")'),
            _statement('materialize', 'SAMPLE X'),
        ]
    )
    core = _observation(
        [
            _statement('from_source', 'PRAGMA table_info("users")'),
            _statement(
                'materialize', 'PRAGMA table_info("users")', purpose='locate'
            ),
            _statement(
                'materialize',
                'SELECT "user_id" FROM "users" LIMIT 1000',
                purpose='locate',
            ),
            _statement('materialize', 'SAMPLE X'),
        ]
    )
    assert _materialize(core) == _materialize(reference)
    assert _compare().counts(core) == _compare().counts(reference)
