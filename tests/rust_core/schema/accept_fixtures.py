# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P65.fixtures, through the binding: every case of ``time-and-keys``,
``relationships``, ``dates`` and ``composite-key``, and every
``wire.infer_dtype`` case, gives its expected output through the Rust core,
with identical warnings.

- ``schema.from_data``: the core's description reduced to the fixture's
  shape, and its warnings' messages, equal the fixture's; the whole outcome,
  warning categories and ``to_schema_definition`` included, equals today's
  Python's.
- ``dates.*`` and ``time_column.read`` through ``dates_op``, ``composite.*``
  through ``composite_op``: the expected output, by value and JSON type, or
  the expected error code and message; and exactly what today's Python gives.
- ``wire.infer_dtype``: the column ``c`` of a one-column table ``t``, inferred
  by the core, is ``""`` when untyped and otherwise its dtype.

``accept_fixtures.rs`` checks the same ``dates`` and ``composite-key`` cases
inside the Rust crate.
"""

from __future__ import annotations

from typing import Any

import pytest
from _schema import (
    difference,
    expected_of,
    fixture_cases,
    native,
    python_fixture_from_data,
    python_op,
    rust_composite_op,
    rust_dates_op,
    rust_fixture_from_data,
    rust_from_data,
    same_as_fixture,
)


def _cases(name: str, operation: str | None = None) -> list[dict[str, Any]]:
    return [
        case
        for case in fixture_cases(name)
        if operation is None or case['operation'] == operation
    ]


SCHEMA_CASES = _cases('time-and-keys') + _cases('relationships')
DATES_CASES = _cases('dates')
COMPOSITE_CASES = _cases('composite-key')
WIRE_CASES = _cases('wire', 'wire.infer_dtype')


def test_every_fixture_file_is_covered() -> None:
    assert len(SCHEMA_CASES) >= 40
    assert {c['operation'] for c in SCHEMA_CASES} == {'schema.from_data'}
    assert len(DATES_CASES) >= 185
    assert {c['operation'] for c in DATES_CASES} <= {
        'dates.day_comes_first',
        'dates.looks_unambiguous',
        'dates.shapes_written',
        'dates.reads_as_a_date',
        'dates.conversion_that_works',
        'dates.read',
        'dates.pattern',
        'dates.conversion_text',
        'time_column.read',
    }
    assert len(COMPOSITE_CASES) >= 40
    assert all(c['operation'].startswith('composite.') for c in COMPOSITE_CASES)
    assert len(WIRE_CASES) >= 12


def _fixture_shape(outcome: dict[str, Any]) -> dict[str, Any]:
    r"""A schema outcome in the shape the fixtures record."""
    schema = outcome['schema']
    return {
        'tables': {
            name: {
                'primary_key': table['primary_key'],
                'time_column': table['time_column'],
                'columns': {
                    column: {'dtype': spec['dtype'], 'stype': spec['stype']}
                    for column, spec in table['columns'].items()
                },
            }
            for name, table in schema['tables'].items()
        },
        'relationships': schema['relationships'],
        'warnings': [w['message'] for w in outcome['warnings']],
    }


@pytest.mark.parametrize(
    'case', SCHEMA_CASES, ids=[c['id'] for c in SCHEMA_CASES]
)
def test_schema_from_data(case: dict[str, Any]) -> None:
    got = rust_fixture_from_data(case['input'])
    assert 'schema' in got, f'the core refused: {got.get("error")}'
    assert difference(case['expected'], _fixture_shape(got)) is None, (
        difference(case['expected'], _fixture_shape(got))
    )
    want = python_fixture_from_data(case['input'])
    assert difference(want, got) is None, difference(want, got)


@pytest.mark.parametrize(
    'case', DATES_CASES, ids=[c['id'] for c in DATES_CASES]
)
def test_dates(case: dict[str, Any]) -> None:
    got = rust_dates_op(case['operation'], case['input'])
    problem = same_as_fixture(expected_of(case), got)
    assert problem is None, problem
    want = python_op(case['operation'], case['input'])
    assert difference(want, got) is None, difference(want, got)


@pytest.mark.parametrize(
    'case', COMPOSITE_CASES, ids=[c['id'] for c in COMPOSITE_CASES]
)
def test_composite_keys(case: dict[str, Any]) -> None:
    got = rust_composite_op(case['operation'], case['input'])
    problem = same_as_fixture(expected_of(case), got)
    assert problem is None, problem
    want = python_op(case['operation'], case['input'])
    assert difference(want, got) is None, difference(want, got)


def _inferred_dtype(outcome: dict[str, Any]) -> Any:
    if 'error' in outcome:
        return outcome['error']
    column = outcome['schema']['tables']['t']['columns']['c']
    return '' if column['untyped'] else column['dtype']


@pytest.mark.parametrize('case', WIRE_CASES, ids=[c['id'] for c in WIRE_CASES])
def test_wire_infer_dtype(case: dict[str, Any]) -> None:
    import pandas as pd
    from _fixture_ops import build_series
    from sdm_client import Schema

    def frames() -> dict[str, Any]:
        return {'t': pd.DataFrame({'c': build_series(case['input']['series'])})}

    from _schema import python_outcome

    python = python_outcome(lambda: Schema.from_data(frames()))
    assert _inferred_dtype(python) == case['expected'], (
        'the one-column route does not give infer_tfm_dtype in Python'
    )
    got = rust_from_data(frames())
    assert _inferred_dtype(got) == case['expected']
    assert difference(python, got) is None, difference(python, got)


@pytest.mark.parametrize(
    ('function', 'args'),
    [
        ('dates_op', ('dates.no_such_operation', '["2021-01-01"]', '{}')),
        ('dates_op', ('dates.day_comes_first', 'not json', '{}')),
        ('dates_op', ('dates.day_comes_first', '["2021-01-01"]', '[]')),
        ('composite_op', ('composite.no_such_operation', '{}')),
        ('composite_op', ('composite.encode_values', 'not json')),
        ('schema_from_source', ('mysql', '{"namespace": {}}', '{}')),
        ('schema_from_source', ('sqlite', 'not json', '{}')),
        ('schema_from_source', ('sqlite', '{"namespace": {}}', '[1]')),
    ],
)
def test_the_binding_refuses_what_it_cannot_read(
    function: str, args: tuple[str, ...]
) -> None:
    module = native(function)
    with pytest.raises(ValueError):
        getattr(module, function)(*args)


def test_schema_from_data_refuses_options_it_cannot_read() -> None:
    import pandas as pd
    from _schema import describer

    module = native('schema_from_data')
    tables, tags = describer().describe_frames(
        {'t': pd.DataFrame({'a': [1, 2]})}
    )
    for options in ('not json', '[]', '{"edits": 5}'):
        with pytest.raises(ValueError):
            module.schema_from_data(tables, tags, options)
    with pytest.raises(ValueError):
        module.schema_from_data(tables, 'not json', '{}')
