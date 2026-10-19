# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1e: a SQLite or Postgres line's effective expectation.

No baseline holds a ``<case>@<database>`` line, so ``ratchet.py
effective`` gave it none and the database jobs held every core runner to
python-ref's outcome, approved entries or not. Its baseline is now
python-ref's committed line of that key, for python-core and rust-host
(rust-host's without the stage and error it does not observe, as an
accepted baseline holds), moved by an approved entry from that outcome as
a DuckDB baseline is; a DuckDB line keeps its held baseline, and a
proposed entry moves nothing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _ratchet() -> ModuleType:
    name = 't21e_ratchet'
    if name not in sys.modules:
        path = ROOT / 'scripts/rust_core/ratchet.py'
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _line(stage: str, code: str, error: str) -> dict[str, Any]:
    return {
        'outcome': {
            'result': 'refusal',
            'stage': stage,
            'code': code,
            'error': error,
        }
    }


def _item(case: str, runner: str, old: str, new: str) -> dict[str, Any]:
    return {
        'case': case,
        'cell': 'ref-pandas3',
        'runner': runner,
        'old': old,
        'new': new,
        'summary': '-',
    }


def test_a_database_line_is_held_to_the_reference_line_moved_by_its_entries() -> (
    None
):
    r = _ratchet()
    refused = 'refusal INVALID_REQUEST at materialize (StructuredDataError)'
    reference = {
        'x': _line('materialize', 'INVALID_REQUEST', 'StructuredDataError'),
        'x@sqlite': _line(
            'materialize', 'INVALID_REQUEST', 'StructuredDataError'
        ),
        'y@postgres': _line(
            'from_source', 'INVALID_REQUEST', 'StructuredDataError'
        ),
    }
    held = {
        'python-core': {
            'x': {
                'projection': {
                    'outcome.result': 'success',
                    'outcome.stage': 'predict',
                    'outcome.error': None,
                    'outcome.code': None,
                }
            }
        }
    }
    register = {
        'entries': [
            {
                'id': 'A',
                'status': 'approved',
                'approved_by': 'reviewer',
                'case_outcomes': [
                    _item(
                        'x@sqlite', 'python-core', refused, 'success at predict'
                    ),
                    _item(
                        'x@sqlite',
                        'rust-host',
                        'refusal INVALID_REQUEST',
                        'success',
                    ),
                ],
            },
            {
                'id': 'B',
                'status': 'proposed',
                'approved_by': None,
                'case_outcomes': [
                    _item(
                        'y@postgres',
                        'python-core',
                        'refusal INVALID_REQUEST at from_source (StructuredDataError)',
                        'success at predict',
                    )
                ],
            },
        ]
    }
    found = r.effective(held, r.approved(register), reference)
    success = {
        'outcome.result': 'success',
        'outcome.stage': 'predict',
        'outcome.error': None,
        'outcome.code': None,
    }
    # Moved by the approved entry, each runner from its own baseline.
    assert found['x@sqlite']['python-core'] == success
    assert found['x@sqlite']['rust-host'] == {
        'outcome.result': 'success',
        'outcome.stage': None,
        'outcome.error': None,
        'outcome.code': None,
    }
    # A proposed entry moves nothing: the reference's line is the baseline.
    assert found['y@postgres'] == {
        'python-core': {
            'outcome.result': 'refusal',
            'outcome.stage': 'from_source',
            'outcome.error': 'StructuredDataError',
            'outcome.code': 'INVALID_REQUEST',
        },
        'rust-host': {
            'outcome.result': 'refusal',
            'outcome.stage': None,
            'outcome.error': None,
            'outcome.code': 'INVALID_REQUEST',
        },
    }
    # DuckDB's line keeps its held baseline, and gets none where none is held.
    assert found['x'] == {'python-core': success}
    # The comparator's --expect reads them by the line's key.
    expect = r.outcomes(found)
    assert expect['x@sqlite']['python-core'] == {
        'result': 'success',
        'stage': 'predict',
        'error': None,
        'code': None,
    }
