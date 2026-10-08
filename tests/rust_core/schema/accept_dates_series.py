# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P65.dates-series: 100,000 generated date columns give zero mismatches
with pandas 3.0.6.

Each generated column (``_dates.py``, seed 20261007) goes through every
``dates.*`` operation that reads a column, ``time_column.read``,
``infer.timestamp`` (the engine's ``contains_timestamp``) and, where a
conversion works, ``dates.read`` with it: through ``dates_op``, and through
today's Python. The answers, refusals included, must be identical (DV20 pins
parity to the phase venv's pandas).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from _dates import COUNT, columns, inputs
from _schema import (
    difference,
    native,
    python_contains_timestamp,
    python_op,
    rust_dates_op,
)

PANDAS = '3.0.6'


def _python(operation: str, given: dict[str, Any]) -> dict[str, Any]:
    if operation == 'infer.timestamp':
        return python_contains_timestamp(given['values'], given['column'])
    return python_op(operation, given)


def _with_read(
    generated: dict[str, Any],
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    r"""Each operation, its input and today's answer; and ``dates.read`` with
    the conversion that works, where one does."""
    out = []
    for operation, given in inputs(generated):
        want = _python(operation, given)
        out.append((operation, given, want))
        if (
            operation == 'dates.conversion_that_works'
            and want.get('output') is not None
        ):
            arguments = want['output']
            conversion = {
                'utc': bool(arguments.get('utc', False)),
                'dayfirst': arguments.get('dayfirst'),
                'mixed': arguments.get('format') == 'mixed',
            }
            read = {'values': given['values'], 'conversion': conversion}
            out.append(('dates.read', read, _python('dates.read', read)))
    return out


def test_pandas_is_the_pinned_version() -> None:
    import pandas as pd

    assert pd.__version__ == PANDAS, (
        f'parity is pinned to pandas {PANDAS} (DV20); the venv has '
        f'{pd.__version__}'
    )


def test_the_columns_are_varied() -> None:
    r"""The generated columns reach every answer: read and refused, a
    conversion found and none, inferred as dates and not, and a column
    pandas refuses outright."""
    seen: Counter[str] = Counter()
    for generated in columns(3000):
        for operation, _, want in _with_read(generated):
            kind = 'error' if 'error' in want else 'output'
            if operation == 'time_column.read':
                seen[f'read:{kind}'] += 1
            elif operation == 'dates.conversion_that_works':
                seen[
                    'conversion' if want.get('output') is not None else 'none'
                ] += 1
            elif operation == 'infer.timestamp':
                seen[f'infer:{want.get("output", "error")}'] += 1
            elif operation == 'dates.day_comes_first':
                seen[f'dayfirst:{want.get("output")}'] += 1
    assert seen['read:output'] >= 600, seen
    assert seen['read:error'] >= 600, seen
    assert seen['conversion'] >= 450 and seen['none'] >= 450, seen
    assert seen['infer:True'] >= 450 and seen['infer:False'] >= 450, seen
    assert seen['infer:error'] >= 5, seen
    assert seen['dayfirst:True'] >= 150 and seen['dayfirst:False'] >= 150, seen


def test_generated_date_columns() -> None:
    native('dates_op')
    failures = []
    compared = 0
    for i, generated in enumerate(columns(COUNT)):
        for operation, given, want in _with_read(generated):
            compared += 1
            got = rust_dates_op(operation, given)
            problem = difference(want, got)
            if problem:
                failures.append(
                    f'column #{i} {operation} {given!r:.300}: {problem}'
                )
    assert compared >= COUNT * 7
    assert not failures, (
        f'{len(failures)} of {compared} answers differ:\n'
        + '\n'.join(failures[:30])
    )
