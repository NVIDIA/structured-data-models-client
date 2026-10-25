# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A deprecated string escape keeps its first warning after the core switch."""

from __future__ import annotations

import sys
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
        # P3E-9 keeps this one body stable across hosts and runtimes;
        # P3E-2's code prefix and every other warning's words are unchanged.
        body = r"invalid escape sequence '\/'"
        prefix = '[INVALID_STRING_ESCAPE] '
        assert str(core_deprecated[0].message) == prefix + body
        assert core_deprecated[0].category is old_deprecated[0].category
        old_body = str(old_deprecated[0].message)
        if old_body == body:
            assert str(core_deprecated[0].message) == prefix + old_body
        else:
            # The approved difference is Python 3.14's native expansion,
            # including its final space, observed on the old route in CI.
            assert sys.version_info[:2] == (3, 14)
            assert old_body == (
                r'"\/" is an invalid escape sequence. '
                'Such sequences will not work in the future. '
            )
        assert core[0] is core_deprecated[0]
    else:
        assert old_deprecated == core_deprecated == []
