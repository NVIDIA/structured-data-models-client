# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native query validation, missing-runtime errors and date diagnostics.
Public materialization and validation retain explicit outcomes without
migration routing switches or a second Python validator.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from sdm_client import (
    PredictiveQuery,
    Request,
    Schema,
    StructuredDataError,
)
from sdm_client._rust import s0
from sdm_client.query import RELATIONAL

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def core(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record each query validated by Rust."""
    validated: list[str] = []
    real = s0.validate

    def spy(text: str, definition: Any, profile: Any) -> Any:
        validated.append(text)
        return real(text, definition, profile)

    monkeypatch.setattr(s0, 'validate', spy)
    return validated


def _users() -> pd.DataFrame:
    return pd.DataFrame({'id': [1, 2, 3, 4], 'age': [10.5, 20.5, 30.5, 40.5]})


ACCEPTED = 'PREDICT users.age FOR EACH users.id'
REFUSED = 'PREDICT users.nope FOR EACH users.id'


def test_public_materialize_accepts_valid_query_and_refuses_missing_column() -> (
    None
):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(ACCEPTED, {'users': _users()}).materialize(
            indices=[1, 2]
        )
        with pytest.raises(StructuredDataError) as refused:
            PredictiveQuery(REFUSED, {'users': _users()}).materialize(
                indices=[1]
            )
    assert type(request) is Request
    assert type(refused.value) is StructuredDataError
    assert refused.value.code == 'INVALID_REQUEST'


def test_the_core_route_runs_validation_on_the_core(core: list[str]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        PredictiveQuery(ACCEPTED, {'users': _users()}).materialize(indices=[1])
        with pytest.raises(StructuredDataError):
            PredictiveQuery(REFUSED, {'users': _users()}).materialize(
                indices=[1]
            )
    assert core == [ACCEPTED, REFUSED]
    assert 'sdm_client._native' in sys.modules


def _subprocess(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=300,
    )


#: Fails any import of the native module from the moment it is installed.
_BLOCK_NATIVE = """
    import sys

    class Block:
        def find_spec(self, name, path=None, target=None):
            if name == 'sdm_client._native':
                raise ImportError('the native module was imported')
            return None

    sys.meta_path.insert(0, Block())
"""


def test_missing_native_module_is_reported_without_fallback() -> None:
    script = (
        _BLOCK_NATIVE
        + """
    import warnings
    import pandas as pd
    from sdm_client import PredictiveQuery
    from sdm_client._rust import NativeBoundaryError

    warnings.simplefilter('ignore')
    users = pd.DataFrame({'id': [1, 2, 3, 4], 'age': [10.5, 20.5, 30.5, 40.5]})
    try:
        PredictiveQuery('PREDICT users.age FOR EACH users.id', {'users': users}
            ).materialize(indices=[1, 2])
    except NativeBoundaryError as missing:
        print(missing)
    """
    )
    result = _subprocess(script)
    assert result.returncode == 0, result.stderr
    assert 'sdm_client._native' in result.stdout


def _orders_schema() -> Any:
    users = pd.DataFrame({'user_id': [1, 2, 3]})
    orders = pd.DataFrame(
        {
            'order_id': [1, 2, 3],
            'user_id': [1, 2, 2],
            'ts': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data({'users': users, 'orders': orders})


def test_the_core_s_own_refusal_is_not_a_refusal_of_the_query(
    core: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A local zone the run cannot read is the core's refusal of the host's
    # facts, raised as its code maps, never as a refusal of the query; it is
    # read only for a two-digit year (D5), so a query without one is
    # answered.
    from sdm_client._rust import facts

    monkeypatch.setattr(
        facts, 'local_zone', lambda *_: {'iana': 'Mars/Olympus'}
    )
    schema = _orders_schema()
    PredictiveQuery(
        'PREDICT COUNT(orders.* WHERE orders.ts > 2024-01-13, 0, 30, days) '
        'FOR EACH users.user_id',
        schema,
    )._resolve(RELATIONAL)
    text = (
        'PREDICT COUNT(orders.* WHERE orders.ts > 1/2/76, 0, 30, days) '
        'FOR EACH users.user_id'
    )
    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery(text, schema)._resolve(RELATIONAL)
    assert raised.value.code == 'INVALID_REQUEST'
    assert raised.value.details['field'] == 'facts.local_zone'
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


#: The old validator holds a date as a pandas Timestamp. pandas 3 holds any
#: year; pandas 2 (on Python 3.10, which pandas 3 does not support) only
#: 1677 to 2262, so there the old route refuses a date in year 0 or 1 as a
#: query (OutOfBoundsDatetime), and the core, told pandas 2's range by the
#: host, refuses it alike (T1.20 fix 1).
PANDAS_3 = int(pd.__version__.split('.')[0]) >= 3

#: Date literals with a run of five digits or more (D8), which the
#: reference reads through dateutil or refuses: a packed first field of
#: six, eight, twelve and fourteen digits, a long field in the time of day
#: (register DT-9's two queries are the second and the sixth), a long
#: zero-padded field read as year 1 (on pandas 3 only), and two the
#: reference refuses.
PACKED_READ = (
    '230203-06-09',
    '20240101-01-01',
    '202401011230-1-1',
    '20240101123456-1-1',
    '20240101/00/123456',
    '2024-01-01 00000:00:00',
)
PACKED_READ_ON_PANDAS_3 = ('00001/1/2',)
PACKED_REFUSED = ('10000-01-01', '12345-1-1')


def _filtered(condition: str) -> str:
    return (
        f'PREDICT COUNT(orders.* WHERE {condition}, 0, 30, days) '
        'FOR EACH users.user_id'
    )


def _assert_refused_as_packed(
    refused: StructuredDataError, literal: str, query: str
) -> None:
    r"""T1.6c's refusal by name: ``INVALID_REQUEST`` naming ``operation``
    ``validate_query`` and ``rule`` ``date_literal_packed``, with the
    literal's place. From KO2-86 (an error may name a value of the caller's
    own call) it names the query given (``query``) and its packed literal
    (``literal``), exactly, and its message names the literal; no other
    detail holds the literal's text."""
    assert type(refused) is StructuredDataError
    assert refused.code == 'INVALID_REQUEST'
    assert refused.details['rule'] == 'date_literal_packed', refused.details
    assert refused.details['operation'] == 'validate_query'
    assert refused.details['field'] == 'query', refused.details
    for name in ('start_row', 'start_col', 'end_row', 'end_col'):
        assert isinstance(refused.details[name], int), (name, refused.details)
    assert refused.details['literal'] == literal, refused.details
    assert refused.details['query'] == query, refused.details
    assert literal in str(refused), str(refused)
    rest = {
        name: value
        for name, value in refused.details.items()
        if name not in ('literal', 'query')
    }
    assert literal not in repr(rest)


@pytest.mark.parametrize(
    'literal', PACKED_READ + PACKED_READ_ON_PANDAS_3 + PACKED_REFUSED
)
def test_a_packed_date_literal_is_refused_by_name_on_the_core(
    core: list[str], literal: str
) -> None:
    # T1.6c brief, Tests: a D8 literal is refused by name on python-core,
    # through materialize and the validation it runs (register DT-9's two
    # queries included: no longer read, as T1.6b read them), where the
    # reference reads it or refuses it as a query.
    text = _filtered(f'orders.ts > {literal}')
    schema = _orders_schema()
    with pytest.raises(StructuredDataError) as validated:
        PredictiveQuery(text, schema)._resolve(RELATIONAL)
    _assert_refused_as_packed(validated.value, literal, text)
    assert validated.value.details['start_row'] == 1
    assert validated.value.details['start_col'] == 41
    with pytest.raises(StructuredDataError) as materialized:
        PredictiveQuery(text, schema).materialize(indices=[1])
    _assert_refused_as_packed(materialized.value, literal, text)
    assert core == [text, text]


def test_a_packed_date_in_an_array_is_refused_by_name_on_the_core(
    core: list[str],
) -> None:
    text = _filtered('orders.ts IN (2024-01-01, 20240101-01-01)')
    with pytest.raises(StructuredDataError) as refused:
        PredictiveQuery(text, _orders_schema()).materialize(indices=[1])
    _assert_refused_as_packed(refused.value, '20240101-01-01', text)


@pytest.mark.parametrize(
    ('literal', 'accepted'),
    [
        ('2024-01-31', True),
        ('2024-01-31 23:59:59', True),
        ('1/2/24', True),
        ('0000-01-01', PANDAS_3),
        ('0001/1/2', PANDAS_3),
        ('2024-01-01 0000:00:00', True),
        ('13/01/2024', True),
        ('2023-02-29', False),
        ('123-4-567', False),
    ],
)
def test_ordinary_date_literals_have_explicit_outcomes(
    core: list[str],
    literal: str,
    accepted: bool,
) -> None:
    text = _filtered(f'orders.ts > {literal}')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            PredictiveQuery(text, _orders_schema())._resolve(RELATIONAL)
            answer = None
        except StructuredDataError as refused:
            assert 'rule' not in refused.details, refused.details
            answer = refused.code
    assert core == [text]
    assert answer == (None if accepted else 'INVALID_REQUEST')


def test_an_earlier_refusal_wins_over_a_packed_date_on_the_core(
    core: list[str],
) -> None:
    # T1.6c: node order decides. A D9 literal built first refuses the query
    # as on the old route; a D8 literal built first ends the parse, by name.
    schema = _orders_schema()
    with pytest.raises(StructuredDataError) as refused:
        PredictiveQuery(
            _filtered('orders.ts > 2023-02-29 AND orders.ts < 20240101-01-01'),
            schema,
        )._resolve(RELATIONAL)
    assert refused.value.code == 'INVALID_REQUEST'
    assert 'rule' not in refused.value.details
    text = _filtered('orders.ts > 20240101-01-01 AND orders.ts < 2023-02-29')
    with pytest.raises(StructuredDataError) as packed:
        PredictiveQuery(text, schema)._resolve(RELATIONAL)
    _assert_refused_as_packed(packed.value, '20240101-01-01', text)


def test_a_packed_date_needs_no_local_zone_on_the_core(
    core: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # T1.6c: a D8 literal is refused before it is read, so a local zone the
    # run cannot read is never asked for, even for a six-digit packed date,
    # whose two-digit year the reference places.
    from sdm_client._rust import facts

    monkeypatch.setattr(
        facts, 'local_zone', lambda *_: {'iana': 'Mars/Olympus'}
    )
    text = _filtered('orders.ts > 760101-01-01')
    with pytest.raises(StructuredDataError) as refused:
        PredictiveQuery(text, _orders_schema())._resolve(RELATIONAL)
    # The query's own refusal (field query), not the local zone's.
    _assert_refused_as_packed(refused.value, '760101-01-01', text)


def _shown(text: str, schema: Any) -> tuple[list[str], str | None]:
    r"""The warnings ``_resolve`` shows, and the code it refuses with."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            PredictiveQuery(text, schema)._resolve(RELATIONAL)
            code = None
        except StructuredDataError as refused:
            code = refused.code
    return [f'{type(w.message).__name__}: {w.message}' for w in caught], code


def test_day_first_notice_is_preserved_before_later_query_errors(
    core: list[str],
) -> None:
    accepted = (
        'PREDICT COUNT(orders.* WHERE orders.ts > 13/01/2024, 0, 30, days) '
        'FOR EACH users.user_id'
    )
    refused = [
        'PREDICT COUNT(orders.* WHERE missing.ts > 13/01/2024, 0, 30, days) '
        'FOR EACH users.user_id',
        'PREDICT COUNT(orders.* WHERE orders.ts > 13/01/2024 AND orders.ts < '
        '2023-02-29, 0, 30, days) FOR EACH users.user_id',
    ]
    schema = _orders_schema()
    on_core = {text: _shown(text, schema) for text in [accepted, *refused]}
    assert core == [accepted, *refused]
    said = (
        'UserWarning: [DATE_LITERAL_DAY_FIRST] Parsing dates in %d/%m/%Y format when dayfirst=False '
        '(the default) was specified. Pass `dayfirst=True` or specify a '
        'format to silence this warning.'
    )
    assert on_core[accepted] == ([said], None)
    for text in refused:
        assert on_core[text] == ([said], 'INVALID_REQUEST')
