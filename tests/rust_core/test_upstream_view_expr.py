# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import runpy
from pathlib import Path

from sdm_core import _native

_ROOT = Path(__file__).resolve().parents[2]
_TESTS = runpy.run_path(
    str(
        _ROOT
        / 'packages/structured-data-models-relational-engine/tests/backend/test_view_expr_calls.py'
    )
)
_NAMESPACE = _TESTS['test_the_validator_accepts'].__globals__
_ORACLE_CALL = _NAMESPACE['_untrusted_call_reason']
_ORACLE_UNSAFE = _NAMESPACE['_unsafe_expr_reason']


def _render(function: str, arguments: dict) -> str | None:
    sql, result = _native.dialect_render(
        'databricks', f'view_expr.{function}', json.dumps(arguments)
    )
    assert sql is None
    return json.loads(result)


def _call(expr: str, dialect: str, *, columns) -> str | None:
    expected = _ORACLE_CALL(expr, dialect, columns=columns)
    actual = _render(
        '_untrusted_call_reason',
        {'expr': expr, 'dialect': dialect, 'columns': list(columns)},
    )
    assert actual == expected, (expr, dialect, columns, expected, actual)
    return actual


def _unsafe(expr: str, dialect: str | None = None) -> str | None:
    expected = _ORACLE_UNSAFE(expr, dialect)
    actual = _render('_unsafe_expr_reason', {'expr': expr, 'dialect': dialect})
    assert actual == expected, (expr, dialect, expected, actual)
    return actual


_NAMESPACE['_untrusted_call_reason'] = _call
_NAMESPACE['_unsafe_expr_reason'] = _unsafe
globals().update(
    (name, value) for name, value in _TESTS.items() if name.startswith('test_')
)
