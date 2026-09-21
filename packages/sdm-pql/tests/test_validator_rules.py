# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two rules a query is resolved against, each fixed from a reported bug."""

from __future__ import annotations

import pytest

from predictive_query import RELATIONAL, SchemaDefinition, TableDefinition
from predictive_query.schema.definition import ColumnKey, ColumnKeyGroup
from predictive_query.schema.table import Column
from predictive_query.schema.typing import Dtype, Stype


def _column(name: str, stype: Stype, dtype: Dtype) -> Column:
    return Column(name=name, stype=stype, dtype=dtype)


def _schema(*, second_path: bool) -> SchemaDefinition:
    """`users`, `events` and `att`, with `att` pointing straight at `users`.

    With `second_path`, `events` points at `users` as well. That edge says
    nothing about `att`, so it must not change how `att` is linked.
    """
    tables = {
        'users': TableDefinition(
            cols=[
                _column('user_id', Stype.ID, Dtype.int),
                _column('country', Stype.categorical, Dtype.string),
            ],
            pkey='user_id',
        ),
        'events': TableDefinition(
            cols=[
                _column('event_id', Stype.ID, Dtype.int),
                _column('user_id', Stype.ID, Dtype.int),
                _column('ts', Stype.timestamp, Dtype.time),
            ],
            pkey='event_id',
            time_col='ts',
        ),
        'att': TableDefinition(
            cols=[
                _column('user_id', Stype.ID, Dtype.int),
                _column('event', Stype.ID, Dtype.int),
                _column('ts', Stype.timestamp, Dtype.time),
                _column('kind', Stype.categorical, Dtype.string),
            ],
            time_col='ts',
        ),
    }
    col_groups = [
        ColumnKeyGroup(
            columns=(
                ColumnKey('att', 'user_id'),
                ColumnKey('users', 'user_id'),
            )
        ),
        ColumnKeyGroup(
            columns=(
                ColumnKey('att', 'event'),
                ColumnKey('events', 'event_id'),
            )
        ),
    ]
    if second_path:
        col_groups.append(
            ColumnKeyGroup(
                columns=(
                    ColumnKey('events', 'user_id'),
                    ColumnKey('users', 'user_id'),
                )
            )
        )
    return SchemaDefinition(tables=tables, col_groups=col_groups)


_AGGREGATE = 'PREDICT COUNT(att.*, 0, 7, days) > 0 FOR EACH users.user_id'


@pytest.mark.parametrize('second_path', [False, True])
def test_a_second_path_to_the_entity_table_leaves_the_direct_key_alone(
    second_path,
):
    """`users.user_id` is in one column group per edge that points at it.

    A group for `events -> users` holds no column of `att` and so answers
    "which column of `att` links to `users`" with nothing. That empty answer
    used to replace the real one found in an earlier group, and the join was
    then reported missing -- so adding an edge between two other tables broke
    an aggregation over a third.
    """
    from predictive_query import validate

    resolved, response = validate(
        _AGGREGATE, _schema(second_path=second_path), RELATIONAL
    )
    assert response.ok, [error.message for error in response.errors]
    assert resolved is not None


def test_two_keys_to_the_same_table_are_still_ambiguous():
    """The guard the fix above must not trade away."""
    from predictive_query import validate

    schema = _schema(second_path=False)
    schema.tables['att'].cols.append(_column('owner_id', Stype.ID, Dtype.int))
    schema.col_groups.append(
        ColumnKeyGroup(
            columns=(
                ColumnKey('att', 'owner_id'),
                ColumnKey('users', 'user_id'),
            )
        )
    )
    _, response = validate(_AGGREGATE, schema, RELATIONAL)
    assert not response.ok


def test_is_in_is_refused_and_names_its_replacement():
    """`IS IN` reached the engine and raised `NotImplementedError` there.

    Refusing it in validation makes it the caller's error rather than the
    library's, and the message has to carry the spelling that does work.
    """
    from predictive_query import validate

    _, response = validate(
        'PREDICT COUNT(att.*, 0, 7, days) FOR EACH users.user_id '
        "WHERE users.country IS IN ('GB', 'DE')",
        _schema(second_path=False),
        RELATIONAL,
    )
    assert not response.ok
    message = ' '.join(error.message for error in response.errors)
    assert 'IS IN' in message
    assert '`IN`' in message


def test_in_is_accepted():
    """The premise of the test above: only the deprecated spelling is refused."""
    from predictive_query import validate

    _, response = validate(
        'PREDICT COUNT(att.*, 0, 7, days) FOR EACH users.user_id '
        "WHERE users.country IN ('GB', 'DE')",
        _schema(second_path=False),
        RELATIONAL,
    )
    assert response.ok, [error.message for error in response.errors]
