# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1d's outcome rule off DuckDB: an S0 case's calls run there through
its ``until``, so an accepted validation followed by ``materialize``'s
error is one error outcome, the last call's; a single call that both
answers and errors, a refused query beside an error, and the same pair on
DuckDB still fail.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _module(path: Path, name: str) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


COMPARE = _module(ROOT / 'scripts/rust_core/compare.py', 't21d_compare')
ERROR = {'kind': 'unsupported', 'code': 'UNSUPPORTED_REQUEST'}


def _failures(
    database: str, accepted: bool | None, materialized: Any
) -> list[str]:
    r"""The outcome rule's failures for an S0 case (``validate_query``
    alone in its core, ``until`` materialize) whose core answered
    ``accepted`` and ``materialized`` beside materialize's error."""
    case = {
        'id': 's0.boundary.deepest@sqlite',
        'until': 'materialize',
        'core': {'validate_query': {}},
    }
    observation = {
        'core': {
            'accepted': accepted,
            'described': None,
            'materialized': materialized,
            'predicted': None,
            'error': ERROR,
            'fetches': [{}],
        },
        'outcome': {'result': 'refusal', 'code': 'UNSUPPORTED_REQUEST'},
    }
    report = COMPARE.Report()
    COMPARE._core_consistent(report, case, 'rust-host', observation, database)
    return [text for _, text in report.failures]


def test_an_accepted_query_and_a_later_call_s_error_is_one_outcome() -> None:
    assert _failures('sqlite', True, None) == []
    both = 's0.boundary.deepest@sqlite: rust-host: core: an answer and an error'
    assert _failures('sqlite', True, {'state': 1}) == [both]
    assert _failures('sqlite', False, None) == [both]
    assert _failures('duckdb', True, None) == [both]
