# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A deprecated string escape keeps its first warning after the core switch."""

from __future__ import annotations

import warnings

import pytest
from adbc_driver_sqlite import dbapi
from sdm_connectors import SQLiteSource

from sdm_client import PredictiveQuery, Schema


@pytest.mark.parametrize(
    ('literal', 'deprecated'),
    [(r"'a\/'", True), (r"'a\\/'", False)],
)
def test_materialize_preserves_the_first_invalid_escape_warning(
    literal: str, deprecated: bool
) -> None:
    with dbapi.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                'CREATE TABLE users (user_id INTEGER PRIMARY KEY, country TEXT)'
            )
            cursor.execute(
                r"INSERT INTO users VALUES (1,'a\/'),(2,'b'),(3,'a\/'),(4,'b')"
            )
        connection.commit()
        schema = Schema.from_source(SQLiteSource(connection))
        query = f'PREDICT users.country = {literal} FOR EACH users.user_id'

        with warnings.catch_warnings(record=True) as core:
            warnings.simplefilter('always')
            PredictiveQuery(query, schema).materialize(indices=[1, 2])

    def invalid_escape(warning: warnings.WarningMessage) -> bool:
        text = str(warning.message)
        return (
            'invalid escape sequence' in text
            or '[INVALID_STRING_ESCAPE]' in text
        )

    core_invalid = [warning for warning in core if invalid_escape(warning)]
    if deprecated:
        assert len(core_invalid) == 1
        assert core_invalid[0].category is DeprecationWarning
        assert str(core_invalid[0].message) == (
            r"[INVALID_STRING_ESCAPE] invalid escape sequence '\/'"
        )
        assert core[0] is core_invalid[0]
    else:
        assert core_invalid == []
