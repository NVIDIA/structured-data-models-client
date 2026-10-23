# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.16, review round 1: an edit reads the schema as it stands, and the
core route needs nothing of the old PQL package or the old engine.

- An edit is checked against the data type the caller's schema holds, as
  the old client's setter checks it, not the one the source gave when the
  schema was described (Codex P1-1).
- A column the source was never asked of is edited by its data type in any
  spelling the old client takes, every timestamp unit among them (P1-2).
- With ``predictive_query`` and ``sdm_relational_engine`` unimportable from
  before the client is first imported, a schema is described, built and
  edited, a query materialized and predicted, all on the core (the
  structure design's rule 11, P1-3).

Each public call runs on both routes, the old client's and the core's with
every core route forced, and the core is held to the old route.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import (
    Column,
    Schema,
    StructuredDataError,
    _routing,
)
from sdm_client.errors import MissingExtraError

ENGINE = importlib.util.find_spec('sdm_relational_engine') is not None
CORE = ','.join(sorted(_routing.CORE_ROUTED))
ROOT = Path(__file__).resolve().parents[3]
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
SHOP = (
    'CREATE TABLE users (user_id BIGINT PRIMARY KEY, age BIGINT)',
    'INSERT INTO users SELECT i, 20 + i % 50 FROM range(40) t(i)',
    'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id BIGINT, '
    'price DOUBLE, ts TIMESTAMP)',
    'INSERT INTO orders SELECT i, i % 40, 10 + i * 1.5, TIMESTAMP '
    "'2023-06-01' + to_days(CAST(i AS INTEGER)) FROM range(400) t(i)",
)


@pytest.fixture(autouse=True)
def _old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)


def _described() -> Schema:
    connection = duckdb.connect()
    for statement in SHOP:
        connection.execute(statement)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_source(
            DuckDBSource(connection), {'users': 'users', 'orders': 'orders'}
        )


def _both(
    monkeypatch: pytest.MonkeyPatch, act: Callable[[], Any]
) -> dict[str, Any]:
    r"""``act``'s outcome on the old route and on the core's; where no
    engine is installed the old route refuses as the missing extra, which is
    held, and the core's outcome stands for both."""
    outcomes = {}
    for route in ('old', 'core'):
        if route == 'core':
            monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
        else:
            monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
        if route == 'old' and not ENGINE:
            with pytest.raises(MissingExtraError):
                act()
            continue
        outcomes[route] = act()
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    outcomes.setdefault('old', outcomes['core'])
    return outcomes


def _refused(error: StructuredDataError) -> tuple[str, str]:
    return (type(error).__name__, error.code)


@pytest.mark.parametrize('meaning', ['text', 'timestamp', 'numerical'])
def test_an_edit_reads_the_data_type_the_schema_holds(
    meaning: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `age` is a BIGINT in the source, retyped `string` by the caller: the
    # old client's setter asks whether a string can mean `meaning`, so text
    # and a timestamp are taken and a number refused.
    def act() -> Any:
        schema = _described()
        schema['users']['age'].dtype = 'string'
        try:
            schema['users']['age'].stype = meaning
        except StructuredDataError as refused:
            return _refused(refused)
        column = schema['users']['age']
        return (
            column.dtype,
            column.stype,
            column.stype_declared,
            schema['users'].time_column,
        )

    outcomes = _both(monkeypatch, act)
    assert outcomes['core'] == outcomes['old']
    if meaning == 'numerical':
        assert outcomes['old'] == ('StructuredDataError', 'INVALID_REQUEST')
    else:
        assert outcomes['old'][:2] == ('string', meaning)


@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_a_column_added_by_hand_is_a_time_column_in_any_timestamp_unit(
    unit: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def act() -> Any:
        schema = _described()
        schema['users'].columns['seen'] = Column(
            'seen', f'timestamp[{unit}]', 'timestamp'
        )
        try:
            schema['users'].time_column = 'seen'
        except StructuredDataError as refused:
            return _refused(refused)
        return schema['users'].time_column, schema['users']['seen'].stype

    outcomes = _both(monkeypatch, act)
    assert outcomes['core'] == outcomes['old'] == ('seen', 'timestamp')


def test_the_core_route_needs_nothing_of_the_old_pql_or_engine(
    tmp_path: Path,
) -> None:
    # The structure design's rule 11: both old packages unimportable from
    # before the client is first imported, in a process of its own (a
    # finder on sys.meta_path is the whole process's), and every core route
    # forced: a schema described, a column added by hand and its meaning
    # set, its key and links edited, the query validated, materialized and
    # predicted.
    script = f"""
    import importlib.util, sys, warnings

    OLD = ('predictive_query', 'sdm_relational_engine')

    class Refused:
        def find_spec(self, name, path=None, target=None):
            if name in OLD or name.startswith(tuple(f'{{o}}.' for o in OLD)):
                raise ImportError(f'{{name}} is not to be imported')

    sys.meta_path.insert(0, Refused())

    import duckdb
    import pandas as pd
    from sdm_client import (
        Column, KumoRelational, PredictiveQuery, Relationship, SDMClient,
        Schema,
    )
    from sdm_connectors import DuckDBSource

    spec = importlib.util.spec_from_file_location(
        'scripted_endpoint',
        {str(ROOT / 'contract/runners/scripted_endpoint.py')!r},
    )
    endpoint = importlib.util.module_from_spec(spec)
    sys.modules['scripted_endpoint'] = endpoint
    spec.loader.exec_module(endpoint)

    connection = duckdb.connect()
    for statement in {list(SHOP)!r}:
        connection.execute(statement)
    connection.execute('ALTER TABLE users ADD COLUMN plan VARCHAR')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(
            DuckDBSource(connection), {{'users': 'users', 'orders': 'orders'}}
        )
        del schema['users']['plan']
        schema['users'].columns['plan'] = Column('plan', 'string', 'text')
        schema['users']['plan'].stype = 'categorical'
        schema['users']['age'].stype = 'categorical'
        schema['orders'].primary_key = 'order_id'
        schema.relationships = [
            Relationship('orders', 'user_id', 'users', 'user_id')
        ]
        with endpoint.serve() as (url, received):
            request = PredictiveQuery({QUERY!r}, schema).materialize(
                indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01')
            )
            with SDMClient(url, max_retries=0) as client:
                result = client.predict(request, model=KumoRelational())
    print(len(result), sorted(n for n in sys.modules if n.startswith(OLD)))
    """
    environment = {
        key: value
        for key, value in os.environ.items()
        if key != _routing.ENVIRONMENT
    }
    done = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**environment, _routing.ENVIRONMENT: CORE},
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ['2', '[]'], done.stdout


def test_the_client_s_profile_words_and_meanings_are_the_language_s() -> None:
    # What the client holds in place of the old PQL package's objects on the
    # core route (rule 11) is the language's, where that package is here to
    # ask: each profile's model words, and each meaning a data type carries,
    # the core's table of it.
    import predictive_query.profiles as profiles
    from predictive_query.schema.typing import Dtype, Stype

    from sdm_client import _native
    from sdm_client.base import PROFILE_MODELS

    assert PROFILE_MODELS == {
        name: getattr(profiles, name).capabilities.model
        for name in PROFILE_MODELS
    }
    language = {
        (stype.value, dtype.value)
        for stype in Stype
        for dtype in Dtype
        if stype.supports_dtype(dtype)
    }
    assert set(map(tuple, _native.MEANINGS)) == language
    # The one table of data type names, which the client's fallback for a
    # build without the core agrees with, and which names each of the
    # language's types by its own name.
    from sdm_client.schema import _WIRE_TO_PQL_DTYPE

    names = dict(_native.DTYPE_NAMES)
    assert {k: names[k] for k in _WIRE_TO_PQL_DTYPE} == _WIRE_TO_PQL_DTYPE
    assert all(
        names[dtype.value] == dtype.value
        for dtype in Dtype
        if dtype.value not in _WIRE_TO_PQL_DTYPE
    )


def _materialized(schema: Schema) -> Any:
    r"""What a materialize of the shop's query answers: the payloads, or the
    refusal's class and code."""
    from sdm_client import PredictiveQuery

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            request = PredictiveQuery(QUERY, schema).materialize(
                indices=[0, 1, 2], anchor_time=pd.Timestamp('2024-03-01')
            )
    except StructuredDataError as refused:
        return _refused(refused)
    import json

    return json.loads(json.dumps(list(request._payloads), default=str))


@pytest.mark.parametrize(
    ('table', 'column', 'dtype', 'meaning'),
    [
        # Across families: a number held as text, text held as a number.
        ('users', 'age', 'string', 'text'),
        ('users', 'age', 'string', 'numerical'),
        ('orders', 'price', 'string', 'categorical'),
        ('users', 'user_id', 'float64', 'ID'),
        ('orders', 'ts', 'string', 'text'),
    ],
)
def test_a_retyped_column_is_edited_and_sampled_as_the_old_client_does(
    table: str,
    column: str,
    dtype: str,
    meaning: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def act() -> Any:
        schema = _described()
        schema[table][column].dtype = dtype
        try:
            schema[table][column].stype = meaning
        except StructuredDataError as refused:
            return ('edit', _refused(refused))
        return ('sampled', _materialized(schema))

    outcomes = _both(monkeypatch, act)
    assert outcomes['core'] == outcomes['old']


#: Each data type name the old client reads (its wire names, every
#: timestamp unit, and the language's own names), and one it knows nothing
#: of, which its checks let through.
NAMES = (
    'int64',
    'int32',
    'int',
    'int16',
    'byte',
    'float64',
    'float32',
    'float',
    'timestamp[s]',
    'timestamp[ms]',
    'timestamp[us]',
    'timestamp[ns]',
    'date',
    'time',
    'string',
    'bool',
    'decimal(9, 2)',
)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('edit', ['key', 'time column', 'meaning', 'link'])
def test_a_column_written_by_hand_is_edited_by_any_name_the_old_client_reads(
    name: str, edit: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A column the source was never asked of, added to a described schema
    # under a data type name the old client reads, then edited.
    from sdm_client import Relationship

    def act() -> Any:
        schema = _described()
        try:
            # A meaning the name's type carries, the old client's own check
            # at construction, before the core sees anything.
            meaning = 'timestamp' if name.startswith(('time', 'date')) else 'ID'
            schema['orders'].columns['extra'] = Column('extra', name, meaning)
            if edit == 'key':
                schema['orders'].primary_key = 'extra'
            elif edit == 'time column':
                schema['orders'].time_column = 'extra'
            elif edit == 'meaning':
                schema['orders']['extra'].stype = 'categorical'
            else:
                schema.relationships = [
                    *schema.relationships,
                    Relationship('orders', 'extra', 'users', 'user_id'),
                ]
        except StructuredDataError as refused:
            return _refused(refused)
        return (
            schema['orders'].primary_key,
            schema['orders'].time_column,
            schema['orders']['extra'].stype,
            len(schema.relationships),
        )

    outcomes = _both(monkeypatch, act)
    assert outcomes['core'] == outcomes['old']


#: Data type names outside the one table of them that the old client still
#: reads by their prefix when it asks whether two columns can join
#: (`_join_family`: a name starting `int` or `float` is a number, one
#: starting `timestamp` a time), and its own odd ones.
UNLISTED = (
    'int8',
    'int128',
    'float16',
    'float8',
    'timestamp[us, UTC]',
    'uint64',
)


@pytest.mark.parametrize('name', UNLISTED)
@pytest.mark.parametrize('how', ['retyped', 'written by hand'])
def test_a_link_reads_any_data_type_name_as_the_old_client_does(
    name: str, how: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Codex's review round 2 of 0c662906: the foreign key of a link named by
    # a data type outside the table, then linked and the query materialized.
    from sdm_client import Relationship

    def act() -> Any:
        schema = _described()
        schema.relationships = []
        if how == 'retyped':
            schema['orders']['user_id'].dtype = name
        else:
            del schema['orders']['user_id']
            schema['orders'].columns['user_id'] = Column('user_id', name, 'ID')
        try:
            schema.relationships = [
                Relationship('orders', 'user_id', 'users', 'user_id')
            ]
        except StructuredDataError as refused:
            return ('link', _refused(refused))
        return ('linked', _materialized(schema))

    outcomes = _both(monkeypatch, act)
    assert outcomes['core'] == outcomes['old']


@pytest.mark.parametrize('meaning', ['categorical', 'ID', 'numerical'])
def test_a_column_named_unsupported_is_checked_as_the_old_client_checks_it(
    meaning: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The language's own name `unsupported`, which the old client checks as
    # the language answers it (not the leniency a name it does not know gets).
    def act() -> Any:
        schema = _described()
        try:
            schema['users'].columns['odd'] = Column(
                'odd', 'unsupported', 'unsupported'
            )
            schema['users']['odd'].stype = meaning
        except StructuredDataError as refused:
            return _refused(refused)
        return schema['users']['odd'].stype

    outcomes = _both(monkeypatch, act)
    assert outcomes['core'] == outcomes['old']


def _said(error: StructuredDataError) -> tuple[str, str, str]:
    r"""A refusal's class, code and words, the words after a leading
    ``[CODE] `` (the core's prefix, P3E-2)."""
    words = str(error)
    if words.startswith(f'[{error.code}] '):
        words = words[len(error.code) + 3 :]
    return type(error).__name__, error.code, words


@pytest.mark.parametrize(
    ('name', 'edit'),
    [
        ('unsupported', 'key'),
        ('unsupported', 'time column'),
        ('byte', 'time column'),
        ('uuid', 'key'),
        ('json', 'link'),
    ],
)
def test_a_table_edit_of_a_column_written_by_hand_is_refused_in_the_old_client_s_words(
    name: str, edit: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Claude's review round 2 of 0c662906 (P1-1, P2-1): the language's own
    # `unsupported` is checked as the language answers it, a name no type
    # has is let through, a link joins names by family, and each refusal
    # quotes the column's own type name.
    from sdm_client import Relationship

    def act() -> Any:
        schema = _described()
        try:
            schema['orders'].columns['extra'] = Column(
                'extra', name, 'unsupported' if name == 'unsupported' else 'ID'
            )
            if edit == 'key':
                schema['orders'].primary_key = 'extra'
            elif edit == 'time column':
                schema['orders'].time_column = 'extra'
            else:
                schema['users']['user_id'].dtype = 'decimal(9, 2)'
                schema.relationships = [
                    Relationship('orders', 'extra', 'users', 'user_id')
                ]
        except StructuredDataError as refused:
            return _said(refused)
        return (
            schema['orders'].primary_key,
            schema['orders'].time_column,
            len(schema.relationships),
        )

    outcomes = _both(monkeypatch, act)
    assert outcomes['core'] == outcomes['old']


@pytest.mark.parametrize(
    ('table', 'column', 'meaning'),
    [('users', 'age', 'text')],
)
def test_a_meaning_the_source_s_type_cannot_carry_is_refused_in_the_old_client_s_words(
    table: str, column: str, meaning: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Claude's review round 2 (P2-2), held by the owner's rule: a column the
    # caller retyped `string` and gave a meaning its source's type cannot
    # carry is refused at materialize naming the column and both types, as
    # the old client's engine names them.
    def act() -> Any:
        schema = _described()
        schema[table][column].dtype = 'string'
        schema[table][column].stype = meaning
        from sdm_client import PredictiveQuery

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                PredictiveQuery(QUERY, schema).materialize(
                    indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01')
                )
        except StructuredDataError as refused:
            return _said(refused)
        return 'sampled'

    outcomes = _both(monkeypatch, act)
    assert outcomes['core'] == outcomes['old']
    assert 'received an incompatible semantic type' in outcomes['old'][2]


def test_a_meaning_the_caller_s_type_cannot_carry_is_refused_in_the_old_client_s_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The caller's own type changed after its meaning was set: the old
    # client's check_consistent refuses at materialize, quoting the type the
    # schema holds.
    def act() -> Any:
        schema = _described()
        schema['users']['age'].stype = 'numerical'
        schema['users']['age'].dtype = 'string'
        from sdm_client import PredictiveQuery

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                PredictiveQuery(QUERY, schema).materialize(
                    indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01')
                )
        except StructuredDataError as refused:
            return _said(refused)
        return 'sampled'

    outcomes = _both(monkeypatch, act)
    assert outcomes['core'] == outcomes['old']


@pytest.mark.parametrize('dtype', ['string'])
def test_a_linked_column_retyped_then_sampled_is_refused_in_the_old_client_s_words(
    dtype: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Claude's review round 3 of 9c061237 (P1): a link's foreign key retyped
    # by the caller after the link was made; the old client's
    # check_consistent refuses at materialize, naming the link and both
    # types, as the relationships edit refuses it.
    def act() -> Any:
        schema = _described()
        schema['orders']['user_id'].dtype = dtype
        from sdm_client import PredictiveQuery

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                PredictiveQuery(QUERY, schema).materialize(
                    indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01')
                )
        except StructuredDataError as refused:
            return _said(refused)
        return 'sampled'

    outcomes = _both(monkeypatch, act)
    assert outcomes['core'] == outcomes['old']
    assert 'cannot match' in outcomes['old'][2]
