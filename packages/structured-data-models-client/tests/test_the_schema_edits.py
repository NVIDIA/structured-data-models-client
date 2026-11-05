# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public schema edits reach Rust and preserve state on refusal."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import (
    PredictiveQuery,
    Relationship,
    Schema,
    StructuredDataError,
)
from sdm_client._rust import edits as on_the_core
from sdm_client._rust import run as runs
from sdm_client.schema import _described_on_the_core

ROOT = Path(__file__).resolve().parents[3]
SETUP = (
    'CREATE TABLE users (user_id BIGINT PRIMARY KEY, age BIGINT, '
    'country VARCHAR, signup TIMESTAMP, seen TIMESTAMP)',
    "INSERT INTO users SELECT i, i % 50, ['US', 'DE', 'IN'][1 + i % 3], "
    "TIMESTAMP '2020-01-01' + to_days(CAST(i AS INTEGER)), "
    "TIMESTAMP '2021-01-01' + to_days(CAST(i AS INTEGER)) FROM range(40) t(i)",
    'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id BIGINT, '
    'price DOUBLE, ts TIMESTAMP, note VARCHAR)',
    "INSERT INTO orders SELECT i, i % 40, i * 1.5, TIMESTAMP '2022-01-01' "
    "+ to_hours(CAST(i AS BIGINT)), 'n' || (i % 4) FROM range(400) t(i)",
)


def _schema() -> Schema:
    connection = duckdb.connect()
    for statement in SETUP:
        connection.execute(statement)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(connection))
    assert _described_on_the_core(schema)
    return schema


def _projection(schema: Schema) -> dict[str, Any]:
    r"""Everything the public schema says, as its objects hold it."""
    return {
        'repr': repr(schema),
        'tables': {
            name: {
                'repr': repr(table),
                'primary_key': table.primary_key,
                'time_column': table.time_column,
                'duplicate_keys': table.duplicate_keys,
                'columns': [
                    (
                        repr(column),
                        column.stype,
                        type(column.stype).__name__,
                        column.stype_declared,
                        column._stype_before_key,
                    )
                    for column in table.columns.values()
                ],
            }
            for name, table in schema.tables.items()
        },
        'links': [repr(link) for link in schema.relationships],
    }


Edit = Callable[[Schema], Any]


def _link(a: str, b: str, c: str, d: str) -> Relationship:
    return Relationship(a, b, c, d)


#: Sequences of edits, each a list of steps the caller makes in order.
SEQUENCES: dict[str, list[Edit]] = {
    'keys': [
        lambda s: setattr(s['users'], 'primary_key', 'age'),
        lambda s: setattr(s['users'], 'primary_key', None),
        lambda s: setattr(s['users'], 'primary_key', ('user_id',)),
        lambda s: setattr(s['users'], 'primary_key', ['user_id', 'age']),
        lambda s: setattr(s['users'], 'primary_key', 'user_id'),
    ],
    'time columns': [
        lambda s: setattr(s['users'], 'time_column', None),
        lambda s: setattr(s['users'], 'time_column', 'seen'),
        lambda s: setattr(s['users'], 'time_column', 'country'),
        lambda s: setattr(s['users'], 'time_column', None),
        lambda s: setattr(s['users']['signup'], 'stype', 'timestamp'),
    ],
    'meanings': [
        lambda s: setattr(s['users']['age'], 'stype', 'categorical'),
        lambda s: setattr(s['orders']['user_id'], 'stype', 'categorical'),
        lambda s: setattr(s, 'relationships', []),
        lambda s: setattr(s['orders']['user_id'], 'stype', 'ID'),
        lambda s: setattr(s['users']['country'], 'stype', 'text'),
        lambda s: setattr(s['orders']['note'], 'stype', 'text'),
    ],
    'left out': [
        lambda s: s['orders'].__delitem__('note'),
        lambda s: s['orders'].__delitem__('user_id'),
        lambda s: s['users'].__delitem__('seen'),
    ],
    'relationships': [
        lambda s: setattr(s, 'relationships', []),
        lambda s: s.relationships.append(
            _link('orders', 'user_id', 'users', 'user_id')
        ),
        lambda s: s.relationships.extend(
            [_link('orders', 'order_id', 'users', 'user_id')]
        ),
        lambda s: s.relationships.pop(),
        lambda s: s.relationships.insert(
            0, _link('orders', 'user_id', 'users', 'user_id')
        ),
        lambda s: s.relationships.remove(
            _link('orders', 'user_id', 'users', 'user_id')
        ),
        lambda s: s.relationships.clear(),
    ],
    'refused': [
        lambda s: setattr(s['users'], 'primary_key', 'nope'),
        lambda s: setattr(s['users'], 'primary_key', 'signup'),
        lambda s: setattr(s['orders'], 'primary_key', []),
        lambda s: setattr(s['orders'], 'primary_key', {'order_id'}),
        lambda s: setattr(s['orders'], 'primary_key', 7),
        lambda s: setattr(s['users'], 'time_column', 'age'),
        lambda s: setattr(s['users'], 'time_column', 'user_id'),
        lambda s: setattr(s['users'], 'time_column', 'nope'),
        lambda s: setattr(s['users'], 'time_column', 7),
        lambda s: setattr(s['users']['age'], 'stype', 'banana'),
        lambda s: setattr(s['users']['country'], 'stype', 'numerical'),
        lambda s: setattr(s['users']['age'], 'stype', 7),
        lambda s: s['orders'].__delitem__('order_id'),
        lambda s: s['orders'].__delitem__('ts'),
        lambda s: s['orders'].__delitem__('nope'),
        lambda s: setattr(
            s, 'relationships', [_link('orders', 'user_id', 'users', 'age')]
        ),
        lambda s: setattr(s, 'relationships', ['orders.user_id']),
        lambda s: s.relationships.append(
            _link('orders', 'note', 'users', 'user_id')
        ),
    ],
}


def _outcome(schema: Schema, edit: Edit) -> Any:
    try:
        with warnings.catch_warnings(record=True) as shown:
            warnings.simplefilter('always')
            edit(schema)
    except StructuredDataError as refused:
        return (type(refused).__name__, refused.code)
    return [str(w.message).split('] ', 1)[-1] for w in shown]


def _walk(schema: Schema, steps: list[Edit]) -> list[Any]:
    walked = []
    for step in steps:
        before = _projection(schema)
        outcome = _outcome(schema, step)
        after = _projection(schema)
        if isinstance(outcome, tuple):
            # Refused at the assignment, the schema left as it was.
            assert after == before
        walked.append((outcome, after))
    return walked


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    r"""The operation of every core run started, in order."""
    seen: list[str] = []
    real = runs.run

    def counting(operation: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        seen.append(operation['op'])
        return real(operation, *args, **kwargs)

    monkeypatch.setattr(runs, 'run', counting)
    return seen


@pytest.mark.parametrize('name', sorted(SEQUENCES))
def test_each_edit_reaches_the_core_and_updates_the_schema(
    name: str, monkeypatch: pytest.MonkeyPatch, started: list[str]
) -> None:
    schema = _schema()
    started.clear()
    core = _walk(schema, SEQUENCES[name])
    # Every step reached the core, refused or not.
    assert started.count('edit_schema') == len(SEQUENCES[name])
    if name == 'refused':
        assert all(isinstance(outcome, tuple) for outcome, _ in core)
    else:
        assert not any(isinstance(outcome, tuple) for outcome, _ in core)
    states = [state for _, state in core]
    if name == 'keys':
        assert [
            state['tables']['users']['primary_key'] for state in states
        ] == [
            'age',
            None,
            ('user_id',),
            ('user_id', 'age'),
            'user_id',
        ]
    elif name == 'time columns':
        assert [
            state['tables']['users']['time_column'] for state in states
        ] == [
            None,
            'seen',
            'country',
            None,
            'signup',
        ]
    elif name == 'relationships':
        # Re-inserting an existing relationship does not duplicate it.
        assert [len(state['links']) for state in states] == [
            0,
            1,
            2,
            1,
            1,
            0,
            0,
        ]
    elif name == 'left out':
        assert 'note' not in schema['orders']
        assert 'user_id' not in schema['orders']
        assert 'seen' not in schema['users']
    elif name == 'meanings':
        assert schema['users']['age'].stype == 'categorical'
        assert schema['users']['country'].stype == 'text'
        assert schema['orders']['note'].stype == 'text'
        assert schema['orders']['user_id'].stype == 'ID'
        assert schema.relationships == []


def test_a_refusal_names_the_field_at_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _schema()
    with pytest.raises(StructuredDataError) as refused:
        schema['users'].primary_key = 'nope'
    assert refused.value.code == 'INVALID_REQUEST'
    assert refused.value.details['field'] == 'primary_key'
    with pytest.raises(StructuredDataError) as refused:
        schema['users']['country'].stype = 'numerical'
    assert refused.value.details['field'] == 'stype'


def test_declared_semantic_types_survive_relationship_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _schema()
    schema['users']['country'].stype = 'text'
    assert schema['users']['country'].stype == 'text'
    schema['orders']['user_id'].stype = 'categorical'
    assert schema['orders']['user_id']._stype_before_key == 'categorical'
    schema.relationships = []
    assert schema['orders']['user_id'].stype == 'categorical'


def test_a_key_and_a_time_column_given_as_numpy_text_are_kept_as_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A name given as ``np.str_``, a ``str`` subclass, is the object each
    # table holds after the edit, on the core as the reference's setters
    # keep it (review, taken ).
    import numpy as np

    schema = _schema()
    schema['orders'].time_column = np.str_('ts')
    assert type(schema['orders'].time_column) is np.str_
    schema['users'].time_column = np.str_('seen')
    assert type(schema['users'].time_column) is np.str_
    assert schema['users'].time_column == 'seen'
    schema['users'].primary_key = np.str_('user_id')
    assert type(schema['users'].primary_key) is np.str_
    schema['users'].time_column = None
    assert schema['users'].time_column is None


def test_a_setter_that_skips_the_core_fails_here(
    monkeypatch: pytest.MonkeyPatch, started: list[str]
) -> None:
    # Seeded: suppressing the native edit must fail the operation-count check.
    monkeypatch.setattr(on_the_core, 'column', lambda *_: False)
    schema = _schema()
    started.clear()
    schema['users']['age'].stype = 'categorical'
    assert started.count('edit_schema') == 0
    with pytest.raises(AssertionError):
        test_each_edit_reaches_the_core_and_updates_the_schema(
            'meanings', monkeypatch, started
        )


def test_manual_schema_changes_can_be_edited_and_materialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    #: the schema is plain data, each column with what the source said
    # of it, so a column retyped by hand or a table pointed at another source
    # table is no longer refused (`schema_edited`): the edit is made and the
    # schema sampled as the old route makes and samples them.
    def outcome(change: Edit) -> Any:
        schema = _schema()
        change(schema)
        try:
            schema['users']['country'].stype = 'text'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                request = PredictiveQuery(
                    'PREDICT SUM(orders.price, 0, 30, days) FOR EACH '
                    'users.user_id',
                    schema,
                ).materialize(indices=[0, 1])
        except StructuredDataError as refused:
            return (type(refused).__name__, refused.code)
        return (
            schema['users']['country'].stype,
            schema['users']['age'].dtype,
            json.loads(json.dumps(request._payloads[0], default=str)),
        )

    for change in (
        lambda s: setattr(s['users']['age'], 'dtype', 'float64'),
        lambda s: setattr(s['orders'], 'source_name', 'users'),
    ):
        core = outcome(change)
        assert 'schema_edited' not in repr(core)


def test_frame_schema_edits_reach_the_core(
    started: list[str],
) -> None:
    import pandas as pd

    frames = Schema.from_data(
        {
            'users': pd.DataFrame({'user_id': [1, 2], 'age': [3, 4]}),
            'orders': pd.DataFrame(
                {'order_id': [1, 2, 3], 'user_id': [1, 2, 2]}
            ),
        }
    )
    frames['users']['age'].stype = 'categorical'
    frames['users'].primary_key = 'user_id'
    frames.relationships = []
    assert started.count('edit_schema') >= 3


#: Edits naming a column by a value no name is, and cannot be looked up:
#: the reference raises Python's TypeError from the lookup, unnamed; the
#: core refuses the value it cannot read, INVALID_REQUEST naming the field
#: (B-16, proposed).
UNHASHABLE: dict[str, Edit] = {
    'time_column': lambda s: setattr(s['users'], 'time_column', ['age']),
    'stype': lambda s: setattr(s['users']['age'], 'stype', ['ID']),
    'column': lambda s: s['orders'].__delitem__(['note']),
    'relationships': lambda s: setattr(
        s,
        'relationships',
        [Relationship('orders', ['user_id'], 'users', 'user_id')],  # type: ignore[arg-type]
    ),
}


@pytest.mark.parametrize('field', sorted(UNHASHABLE))
def test_invalid_names_are_refused_without_mutating_the_schema(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = _schema()
    before = _projection(schema)
    with pytest.raises(StructuredDataError) as refused:
        UNHASHABLE[field](schema)
    assert refused.value.code == 'INVALID_REQUEST'
    assert refused.value.details['field'] == field
    assert _projection(schema) == before


def test_a_key_named_on_a_link_s_foreign_key_is_refused_at_materialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Found by the E49 differential: the reference's engine
    # refuses a link whose foreign key is its own table's key
    # (Graph._edge_error); the core's consistency check now does too.
    query = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
    schema = _schema()
    schema['orders'].primary_key = 'user_id'
    with pytest.raises(StructuredDataError) as refused:
        PredictiveQuery(query, schema).materialize(indices=[0, 1])
    assert refused.value.code == 'INVALID_REQUEST'
