# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A deprecated string escape keeps its first warning after the core switch."""

from __future__ import annotations

import warnings

import pytest
from adbc_driver_sqlite import dbapi
from sdm_connectors import SQLiteSource

from sdm_client import PredictiveQuery, Schema, _routing


@pytest.mark.parametrize(
    ('literal', 'deprecated'),
    [(r"'a\/'", True), (r"'a\\/'", False)],
)
def test_materialize_preserves_the_first_invalid_escape_warning(
    literal: str, deprecated: bool, monkeypatch: pytest.MonkeyPatch
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
        monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
        schema = Schema.from_source(SQLiteSource(connection))
        query = f'PREDICT users.country = {literal} FOR EACH users.user_id'

        def materialize(setting: str | None) -> list[warnings.WarningMessage]:
            if setting is None:
                monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
            else:
                monkeypatch.setenv(_routing.ENVIRONMENT, setting)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                PredictiveQuery(query, schema).materialize(indices=[1, 2])
            return caught

        old = materialize('')
        core = materialize(None)

    old_deprecated = [
        warning for warning in old if warning.category is DeprecationWarning
    ]
    core_deprecated = [
        warning
        for warning in core
        if issubclass(warning.category, DeprecationWarning)
    ]
    if deprecated:
        assert len(old_deprecated) > 1
        assert len(core_deprecated) == 1
        assert str(core_deprecated[0].message) == str(old_deprecated[0].message)
        assert core[0] is core_deprecated[0]
    else:
        assert old_deprecated == core_deprecated == []
    assert [
        warning.category.__name__
        for warning in core
        if not issubclass(warning.category, DeprecationWarning)
    ] == [
        warning.category.__name__
        for warning in old
        if not issubclass(warning.category, DeprecationWarning)
    ]
