# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P65.recorder-signin (task 6, Phase 5.6 REVIEW-r1 minor 2): the live
recorder signs in to Snowflake with a programmatic access token when
``SNOWFLAKE_TOKEN`` is set, otherwise with ``SNOWFLAKE_PASSWORD``, and
refuses to record when neither is set.

The Snowflake connector's ``connect`` is replaced by a fake that keeps what
it was given; nothing is reached. The values are dummies.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest
from _schema import REPO, TRANSCRIPTS

sys.path.insert(0, str(REPO / 'scripts' / 'parity'))

CREDENTIALS = (
    'SNOWFLAKE_ACCOUNT',
    'SNOWFLAKE_USER',
    'SNOWFLAKE_TOKEN',
    'SNOWFLAKE_PASSWORD',
    'SNOWFLAKE_WAREHOUSE',
    'PARITY_SNOWFLAKE_DATABASE',
)
BASE = {
    'SNOWFLAKE_ACCOUNT': 'dummy-account',
    'SNOWFLAKE_USER': 'dummy-user',
    'SNOWFLAKE_WAREHOUSE': 'dummy-warehouse',
    'PARITY_SNOWFLAKE_DATABASE': 'DUMMY_DB',
}


class _Cursor:
    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, *args: Any, **kwargs: Any) -> _Cursor:
        return self

    def close(self) -> None:
        return None


class _Connection:
    def cursor(self) -> _Cursor:
        return _Cursor()

    def close(self) -> None:
        return None


@pytest.fixture
def signed_in(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    pytest.importorskip('snowflake.connector')
    import snowflake.connector

    for name in CREDENTIALS:
        monkeypatch.delenv(name, raising=False)
    for name, value in BASE.items():
        monkeypatch.setenv(name, value)
    calls: list[dict[str, Any]] = []

    def connect(*args: Any, **kwargs: Any) -> _Connection:
        assert not args, 'connect takes keywords'
        calls.append(kwargs)
        return _Connection()

    monkeypatch.setattr(snowflake.connector, 'connect', connect)
    return calls


def _transcript(name: str) -> dict[str, Any]:
    path = TRANSCRIPTS / name / 'transcript.json'
    transcript = json.loads(path.read_text())
    transcript['data'] = {
        **transcript['data'],
        'namespace': {
            'database': '$PARITY_SNOWFLAKE_DATABASE',
            'schema': 'PARITY_SHOP',
        },
    }
    return transcript


def _record(transcript: dict[str, Any]) -> None:
    from _parity import data as datagen
    from _parity import live

    prepared = live.prepare(transcript)
    frames = datagen.generate(
        prepared['data']['generator'], prepared['data']['seed']
    )
    with live.connection(prepared, frames):
        pass


@pytest.mark.parametrize(
    'scenario', ['snowflake/count', 'snowflake/semantic-view/count']
)
def test_a_token_signs_in_with_the_token(
    signed_in: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    monkeypatch.setenv('SNOWFLAKE_TOKEN', 'dummy-token')
    monkeypatch.setenv('SNOWFLAKE_PASSWORD', 'dummy-password')
    _record(_transcript(scenario))
    assert len(signed_in) == 1
    given = signed_in[0]
    assert given.get('authenticator') == 'PROGRAMMATIC_ACCESS_TOKEN'
    assert given.get('token') == 'dummy-token'
    assert 'password' not in given
    assert given.get('account') == 'dummy-account'
    assert given.get('user') == 'dummy-user'
    assert given.get('warehouse') == 'dummy-warehouse'


def test_a_password_signs_in_without_a_token(
    signed_in: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('SNOWFLAKE_PASSWORD', 'dummy-password')
    _record(_transcript('snowflake/count'))
    assert len(signed_in) == 1
    given = signed_in[0]
    assert given.get('password') == 'dummy-password'
    assert 'token' not in given
    assert 'authenticator' not in given
    assert given.get('account') == 'dummy-account'


def test_neither_is_refused_before_connecting(
    signed_in: list[dict[str, Any]],
) -> None:
    from _parity.runner import CannotRecord

    with pytest.raises(
        CannotRecord, match='SNOWFLAKE_TOKEN or SNOWFLAKE_PASSWORD'
    ):
        _record(_transcript('snowflake/count'))
    assert signed_in == []
