# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered diagnostic templates and specific public refusal messages.

Historical whole-census comparisons remain archived for final acceptance.
These permanent checks cover template rendering and named public refusals.
"""

from __future__ import annotations

import json
import re
import string
import warnings
from contextlib import contextmanager
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
    _native,
)
from sdm_client._rust import errors

RANK = (
    'PREDICT LIST_DISTINCT(buys.item_id, 0, 30, days) RANK TOP 2 '
    'FOR EACH users.user_id'
)


@contextmanager
def _ranking_source():
    fixture = Path(__file__).with_name('data') / 'ranking-shop.json'
    with duckdb.connect() as connection:
        for statement in json.loads(fixture.read_text())['statements']:
            connection.execute(statement)
        yield DuckDBSource(connection)


#: The sentences a refusal said before when no template of the
#: host's matched it: a code's sentence, naming nothing of the refusal.
GENERIC = frozenset(
    {
        'the request is not one the core accepts',
        'the request is valid, and not one the core supports yet',
        'a column has a type the core does not take',
        'a value is outside what its column type can hold in the core',
        'the schema is not one the core can work with',
        'the configuration of this call is not one the core accepts',
    }
)


def _table() -> dict[str, Any]:
    table: dict[str, Any] = json.loads(_native.templates())
    return table


def _sentences() -> dict[str, dict[str, Any]]:
    return {entry['name']: entry for entry in _table()['sentences']}


# --- the table and its spellings ------------------------------------------------


def test_the_binding_exports_the_core_s_table() -> None:
    table = _table()
    assert set(table) >= {'sentences', 'advice', 'groups', 'databases'}
    sentences = _sentences()
    assert len(sentences) == len(table['sentences']) >= 150
    for name, entry in sentences.items():
        assert entry['template'], name
        assert entry['kind'] in ('error', 'warning', 'part', 'note'), name
        if entry['kind'] == 'error':
            assert entry['code'] in _native.ERROR_CODES, name
        elif entry['kind'] == 'warning':
            assert entry['code'] == entry['code'].upper(), name
        else:
            # A part's sentence follows an error's of any code; a note of a
            # semantic model's conversion has none.
            assert entry['code'] is None, name
        assert entry['owner'] in (
            'shared',
            'host_dates',
            'host_cast',
            'host_comparison',
        ), name
    # Every error code the core gives is said by a sentence of the table.
    said = {e['code'] for e in sentences.values() if e['kind'] == 'error'}
    assert said == set(_native.ERROR_CODES)


def test_every_advice_has_a_python_spelling() -> None:
    advice = {entry['name']: entry['arguments'] for entry in _table()['advice']}
    assert set(errors.ADVICE) == set(advice)
    for name, arguments in advice.items():
        spelled = errors.ADVICE[name](*[f'{a}_value' for a in arguments])
        assert isinstance(spelled, str), name
        assert spelled, name


def test_every_host_owned_sentence_has_this_host_s_library_words() -> None:
    owned = {
        name
        for name, entry in _sentences().items()
        if entry['owner'] != 'shared'
    }
    assert owned == set(errors.HOST_OWNED)
    assert errors.HOST_OWNED['float32_overflow'] == (
        'overflow encountered in cast'
    )


def _value(name: str, conversion: str | None, spec: str) -> Any:
    r"""A value a placeholder of ``conversion`` and ``spec`` reads."""
    if conversion == 'I':
        return '1234'
    if spec == ',':
        return 1234
    if spec in ('place', 'backend'):
        return 'duckdb'
    if conversion in ('l', 'j', 'c', 'R', 'd'):
        return ['a', 'b', 'c']
    if conversion == 'T':
        return ['12:00:00', 'noon']
    if conversion == 'p':
        return ['a', 1]
    if conversion in ('G', 'S', 'N'):
        return '"given"'
    if conversion == 't':
        return '2024-01-02 03:04:05'
    return f'{name}_value'


def _fields(template: str) -> tuple[dict[str, Any], dict[str, Any]]:
    r"""Fields and host values that fill every placeholder of ``template``."""
    fields: dict[str, Any] = {}
    held: dict[str, Any] = {}
    for _, name, spec, conversion in string.Formatter().parse(template):
        if name is None:
            continue
        if name == 'advice':
            arguments = spec.split('(', 1)[1].rstrip(')')
            for argument in filter(
                None, (a.strip() for a in arguments.split(','))
            ):
                fields.setdefault(argument, ['a', 'b'])
            continue
        base = name.split('.', 1)[0].split('[', 1)[0]
        if (spec or '').startswith('to('):
            fields.setdefault(base, ['a', 'b'])
            fields[spec[3:-1]] = ['c', 'd']
            continue
        value = _value(base, conversion, spec or '')
        if '[' in name and not isinstance(value, list):
            value = [value, 'second']
        if base == 'host':
            held[name.split('.', 1)[1]] = ['gold', 'silver']
        else:
            fields.setdefault(base, value)
    return fields, held


def test_every_sentence_is_rendered_once() -> None:
    for name, entry in _sentences().items():
        fields, held = _fields(entry['template'])
        said = errors.render(name, fields, held)
        assert isinstance(said, str) and said, name
        assert '{advice' not in said and '{host' not in said, (name, said)


def test_the_time_column_taken_as_the_earliest_and_one_refused_as_empty() -> (
    None
):
    # Two sentences, which the core tells apart: the host chooses neither.
    assert errors.render(
        'time_columns_taken_earliest',
        {
            'table': 'orders',
            'column': 'placed',
            'among': ['placed', 'shipped'],
            'other': 'shipped',
        },
    ) == (
        'orders.placed was taken as the time column because its values start '
        "earliest of 'placed' and 'shipped'. If another says when a row became "
        "true, name it: time_columns={'orders': 'shipped'}. If none does, clear "
        "it: time_columns={'orders': None}."
    )
    assert errors.render(
        'time_columns_refused_empty', {'table': 'events', 'column': 'seen'}
    ) == (
        'events.seen was not taken as the time column because no row holds a '
        'value. If it does say when a row became true, name it: '
        "time_columns={'events': 'seen'}."
    )


def test_a_key_that_names_no_column_with_and_without_what_was_given() -> None:
    assert errors.render('key_not_names', {'table': 'orders'}) == (
        "a primary key of 'orders' names a column or a sequence of them, and "
        'the one given is neither'
    )
    assert (
        errors.render(
            'key_not_names_given', {'table': 'orders', 'given': '3.5'}
        )
        == 'a primary key names a column or a sequence of them, got 3.5'
    )


# --- warnings -------------------------------------------------------------------


def _warning(sentence: str, **fields: Any) -> dict[str, Any]:
    r"""A core warning saying ``sentence``, of the code the table gives
    it, with ``fields``, each a text, a list of texts or a count."""

    def encoded(value: Any) -> Any:
        if isinstance(value, list):
            return [encoded(item) for item in value]
        if isinstance(value, int):
            return {'t': 'u64', 'v': str(value)}
        return {'t': 'utf8', 'v': value}

    return {
        'code': _sentences()[sentence]['code'],
        'sentence': sentence,
        'message': 'the core says this naming nothing',
        'fields': {name: encoded(value) for name, value in fields.items()},
    }


def _shown(*given: dict[str, Any], **more: Any) -> list[str]:
    control = json.dumps({'message': 'warnings', 'warnings': list(given)})
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        errors.warn_from_core(control.encode(), {}, **more)
    return [
        str(w.message)
        for w in shown
        if not issubclass(w.category, ResourceWarning)
    ]


def test_time_column_notices_given_together_are_one_warning() -> None:
    assert _shown(
        _warning('time_columns_refused_empty', table='events', column='seen'),
        _warning(
            'time_columns_taken_earliest',
            table='orders',
            column='placed',
            among=['placed', 'shipped'],
            other='shipped',
        ),
    ) == [
        '[TIME_COLUMNS_GUESSED] Some time columns were inferred, or refused, '
        'on a guess about what a column holds:\n'
        '- events.seen was not taken as the time column because no row holds '
        'a value. If it does say when a row became true, name it: '
        "time_columns={'events': 'seen'}.\n"
        '- orders.placed was taken as the time column because its values '
        "start earliest of 'placed' and 'shipped'. If another says when a row "
        "became true, name it: time_columns={'orders': 'shipped'}. If none "
        "does, clear it: time_columns={'orders': None}."
    ]


def test_columns_left_out_of_one_table_are_one_warning() -> None:
    left = 'columns_left_out'
    assert _shown(
        _warning(left, table='blobs', column='tags', dtype='stringlist'),
        _warning(left, table='blobs', column='scores', dtype='floatlist'),
        _warning(left, table='other', column='more', dtype='intlist'),
    ) == [
        '[COLUMNS_LEFT_OUT] blobs leaves out tags (stringlist), scores '
        '(floatlist): a request carries one scalar value per cell, and these '
        'hold none',
        '[COLUMNS_LEFT_OUT] other leaves out more (intlist): a request carries '
        'one scalar value per cell, and these hold none',
    ]


def test_unscored_candidates_are_named_as_the_caller_gave_them() -> None:
    # The core names the candidates by position; only the host holds what
    # the caller gave, which it fills in as a host value.
    remedy = (
        'Gather a wider context with num_context_samples if other entities in '
        'the data answer differently, or ask over a table of (entity, '
        'candidate) pairs, which scores every candidate from its features.'
    )

    given = {1: 'gold', 2: 'silver', 3: 'bronze'}

    def named(fields: dict[str, Any]) -> list[Any]:
        return [given[at] for at in fields['candidates']]

    several = _warning('candidates_unscored_several', candidates=[1, 3])
    one = _warning('candidates_unscored_one', candidates=[2])
    with errors.holding(candidates=named):
        shown = _shown(several, one)
    assert shown == [
        "[CANDIDATES_UNSCORED] candidates 'gold' and 'bronze' are left out of "
        'this ranking, unscored: the context sampled for each holds one '
        'answer only, nearly always because no entity in it linked to the '
        'candidate in the window, and a classification needs two to learn '
        'from. No call was sent for them, so each row ranks only the rest, '
        "and the result's unscored_candidates names them. " + remedy,
        "[CANDIDATES_UNSCORED] candidate 'silver' is left out of this "
        'ranking, unscored: the context sampled for it holds one answer only, '
        'nearly always because no entity in it linked to the candidate in the '
        'window, and a classification needs two to learn from. No call was '
        "sent for it, so each row ranks only the rest, and the result's "
        'unscored_candidates names it. ' + remedy,
    ]


# --- the public calls -----------------------------------------------------------


def test_a_key_check_s_advice_is_python() -> None:
    connection = duckdb.connect()
    connection.execute('CREATE TABLE users (user_id INTEGER, name VARCHAR)')
    connection.execute("INSERT INTO users VALUES (1, 'a'), (1, 'b'), (2, 'c')")
    with pytest.raises(StructuredDataError) as raised:
        Schema.from_source(
            DuckDBSource(connection), primary_keys={'users': 'user_id'}
        )
    assert str(raised.value) == (
        '[INVALID_REQUEST] users.user_id is not unique enough to identify a '
        'row: 1 of 3 rows repeat a value already used, e.g. 1. Name another '
        'key with primary_keys=, or pass check_keys=False to skip (one full '
        'scan per table).'
    )


def _shop() -> DuckDBSource:
    connection = duckdb.connect()
    for statement in (
        'CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, joined '
        'TIMESTAMP)',
        "INSERT INTO customers SELECT i, TIMESTAMP '2020-01-01' + i * "
        'INTERVAL 1 DAY FROM range(30) t(i)',
        'CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id '
        'INTEGER REFERENCES customers (customer_id), placed TIMESTAMP, '
        'shipped TIMESTAMP)',
        "INSERT INTO orders SELECT i, i % 30, TIMESTAMP '2021-01-01' + i * "
        "INTERVAL 1 DAY, TIMESTAMP '2021-01-05' + i * INTERVAL 1 DAY FROM "
        'range(100) t(i)',
        'CREATE TABLE events (event_id INTEGER PRIMARY KEY, customer_id '
        'INTEGER REFERENCES customers (customer_id), seen TIMESTAMP)',
        'INSERT INTO events SELECT i, i % 30, NULL FROM range(10) t(i)',
        'CREATE TABLE blobs (blob_id INTEGER PRIMARY KEY, tags VARCHAR[], '
        'scores DOUBLE[])',
        "INSERT INTO blobs SELECT i, ['a'], [1.0] FROM range(5) t(i)",
    ):
        connection.execute(statement)
    return DuckDBSource(connection)


def test_a_schema_edit_s_advice_is_python_and_notices_group() -> None:
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        Schema.from_source(_shop())
    said = [
        str(w.message)
        for w in shown
        if not issubclass(w.category, ResourceWarning)
    ]
    assert said == [
        '[COLUMNS_LEFT_OUT] blobs leaves out tags (stringlist), scores '
        '(floatlist): a request carries one scalar value per cell, and these '
        'hold none',
        '[TIME_COLUMNS_GUESSED] Some time columns were inferred, or refused, '
        'on a guess about what a column holds:\n'
        '- events.seen was not taken as the time column because no row holds '
        'a value. If it does say when a row became true, name it: '
        "time_columns={'events': 'seen'}.\n"
        '- orders.placed was taken as the time column because its values '
        "start earliest of 'placed' and 'shipped'. If another says when a row "
        "became true, name it: time_columns={'orders': 'shipped'}. If none "
        "does, clear it: time_columns={'orders': None}.",
    ]


def test_a_primary_key_linked_as_a_foreign_key_is_said_as_the_old_client_says_it() -> (
    None
):
    # M15 of review: the old client's engine refuses it
    # (`Graph._edge_error`), and the core said only its code's sentence.
    connection = duckdb.connect()
    for statement in (
        'CREATE TABLE users (user_id INTEGER PRIMARY KEY, age INTEGER)',
        'INSERT INTO users SELECT i, i % 50 FROM range(40) t(i)',
        'CREATE TABLE orders (order_id INTEGER PRIMARY KEY, user_id INTEGER, '
        'amount DOUBLE, ts TIMESTAMP)',
        "INSERT INTO orders SELECT i, i % 40, i * 1.5, TIMESTAMP '2020-01-01' "
        '+ i * INTERVAL 1 DAY FROM range(200) t(i)',
    ):
        connection.execute(statement)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(connection))
    schema.relationships = [
        Relationship('orders', 'order_id', 'users', 'user_id')
    ]
    query = PredictiveQuery(
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id', schema
    )
    with pytest.raises(StructuredDataError) as raised:
        query.materialize(indices=[1, 2])
    assert raised.value.code == 'INVALID_REQUEST'
    assert str(raised.value) == (
        "[INVALID_REQUEST] Cannot treat the primary key of table 'orders' as "
        'a foreign key. Remove either the primary key or the link before '
        'proceeding.'
    )


# --- fix 1: values only this host holds, and the old client's words ---


def _refusal(sentence: str, **fields: Any) -> bytes:
    def encoded(value: Any) -> Any:
        if isinstance(value, list):
            return [encoded(item) for item in value]
        if isinstance(value, int):
            return {'t': 'u64', 'v': str(value)}
        return {'t': 'utf8', 'v': value}

    entry = _sentences()[sentence]
    error = {
        'kind': 'invalid_request'
        if entry['code'] == 'INVALID_REQUEST'
        else 'unsupported',
        'code': entry['code'],
        'sentence': sentence,
        'message': 'm',
        'fields': {name: encoded(value) for name, value in fields.items()},
    }
    return json.dumps({'message': 'error', 'error': error}).encode()


def test_a_candidate_is_named_as_the_caller_gave_it_never_by_its_position() -> (
    None
):
    # The core names a candidate by its position; the sentence names the
    # candidate as given, a value only the host holds (review r1, P1-1).
    for name in (
        'candidate_not_whole',
        'candidate_unwritable',
        'candidate_type',
        'candidate_not_exact',
        'candidate_unreadable',
    ):
        assert '{candidate' not in _sentences()[name]['template'], name
    given = [1.5, 'a,b']

    def candidate(fields: dict[str, Any]) -> Any:
        return given[fields['candidate']]

    with errors.holding(
        candidate=candidate,
        candidate_type=lambda fields: type(candidate(fields)).__name__,
    ):
        whole = errors.error_from_core(
            _refusal('candidate_not_whole', field='candidates', candidate=0),
            {},
        )
        unwritable = errors.error_from_core(
            _refusal('candidate_unwritable', field='candidates', candidate=1),
            {},
        )
    assert str(whole) == '[INVALID_REQUEST] 1.5 cannot name a row to rank'
    assert str(unwritable) == (
        "[INVALID_REQUEST] the candidate 'a,b' cannot be written into a "
        'query: a key holding a quote, a comma, a bracket or nothing at all '
        'would read as part of the question rather than as the row it names'
    )


def test_a_source_with_no_table_is_named_as_the_caller_gave_it() -> None:
    source = DuckDBSource(duckdb.connect())
    with pytest.raises(StructuredDataError) as raised:
        Schema.from_source(source)
    assert str(raised.value) == (
        f'[INVALID_REQUEST] {source!r} holds no table to describe'
    )


def _users() -> Schema:
    connection = duckdb.connect()
    for statement in (
        'CREATE TABLE users (user_id INTEGER PRIMARY KEY, age INTEGER)',
        'INSERT INTO users SELECT i, i % 50 FROM range(40) t(i)',
        'CREATE TABLE orders (order_id INTEGER PRIMARY KEY, user_id INTEGER '
        'REFERENCES users (user_id), ts TIMESTAMP)',
        "INSERT INTO orders SELECT i, i % 40, TIMESTAMP '2020-01-01' + i * "
        'INTERVAL 1 DAY FROM range(200) t(i)',
    ):
        connection.execute(statement)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_source(DuckDBSource(connection))


def test_the_context_size_refused_is_named_as_the_caller_gave_it() -> None:
    # The old client's words, with the value as given, however large
    # (review r1, P1-2 and P1-3).
    query = PredictiveQuery(
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id', _users()
    )
    with pytest.raises(StructuredDataError) as past:
        query.materialize(indices=[1], num_context_samples=10_001)
    assert str(past.value).split('] ', 1)[1] == (
        'num_context_samples is 10,001, and the endpoint accepts at most '
        '10,000 context rows'
    )
    with pytest.raises(StructuredDataError) as below:
        query.materialize(indices=[1], num_context_samples=-(10**100))
    assert str(below.value).split('] ', 1)[1] == (
        'num_context_samples must be at least 1; a model predicts from '
        f'labelled examples, and {-(10**100)} leaves it none'
    )


def test_no_shared_template_spells_a_python_value() -> None:
    # A value a host writes in its own language is its advice or its own
    # library's words, never the shared template's (review r1, P1-4).

    for name, entry in _sentences().items():
        if entry['owner'] != 'shared':
            continue
        # A class an endpoint spells "True" is its data, not a value.
        found = re.search(
            r"(?<![\"'])\b(?:None|True|False)\b(?![\"'])|'NaT'",
            entry['template'],
        )
        assert found is None, name


def test_several_reserved_columns_are_said_as_the_old_client_says_them() -> (
    None
):
    assert errors.render(
        'wire_reserved_several',
        {
            'tables': ['a', 'b'],
            'columns': ['instance_id', '__kumo_anchor_time'],
            'reserved': ['a.instance_id', 'b.__kumo_anchor_time'],
            'renamed': [
                'instance_id_from_the_data',
                'kumo_anchor_time_from_the_data',
            ],
        },
    ) == (
        'a.instance_id and b.__kumo_anchor_time use names this client needs '
        'for the request it builds, so they cannot also be columns of the '
        "data. Rename them, for instance 'instance_id' to "
        "'instance_id_from_the_data', '__kumo_anchor_time' to "
        "'kumo_anchor_time_from_the_data'."
    )


def test_a_ranking_s_candidates_are_refused_in_the_old_client_s_words() -> None:
    # Through the public calls, as the old client says each (review r1,
    # P1-1): the candidate as given and the question as written are the
    # values only the host holds, which the core's sentences name.

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with _ranking_source() as source:
            schema = Schema.from_source(source)
            said = []
            for candidates, text in (
                ([1.5], RANK),
                (["a'b"], RANK),
                (
                    [1, 2],
                    'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
                ),
            ):
                with pytest.raises(StructuredDataError) as raised:
                    PredictiveQuery(text, schema).materialize(
                        indices=[1], candidates=candidates
                    )
                said.append(str(raised.value))
    assert said == [
        '[INVALID_REQUEST] 1.5 cannot name a row to rank',
        '[INVALID_REQUEST] the candidate "a\'b" cannot be written into a '
        'query: a key holding a quote, a comma, a bracket or nothing at all '
        'would read as part of the question rather than as the row it names',
        '[INVALID_REQUEST] candidates are the rows a ranking is over, and '
        "'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id' does "
        'not rank anything. Ask it without candidates, or rank with "RANK TOP '
        'k", which is the question they are for',
    ]


# --- fix 2: the part a refusal happened in is the core's sentence ---


def test_a_part_s_context_is_a_sentence_of_the_core_s() -> None:
    # The part of a question asked in parts a refusal happened in is said by
    # a sentence the core picks, after the refusal's own: a forecast's step,
    # with the remedy where the step's window outran the data, and a
    # ranking's candidate (review r2, P1).
    parts = {
        name: entry
        for name, entry in _sentences().items()
        if entry['kind'] == 'part'
    }
    assert set(parts) == {
        'part_forecast_step',
        'part_forecast_step_outran',
        'part_ranking_candidate',
    }
    error = {
        'kind': 'invalid_request',
        'code': 'INVALID_REQUEST',
        'sentence': 'anchor_entity_not_static',
        'part': 'part_forecast_step_outran',
        'message': 'm',
        'fields': {
            'forecast_step': {'t': 'u32', 'v': '10'},
            'forecast_steps': {'t': 'u32', 'v': '40'},
        },
    }
    made = errors.error_from_core(
        json.dumps({'message': 'error', 'error': error}).encode(), {}
    )
    assert str(made) == (
        "[INVALID_REQUEST] Anchor time 'entity' is only valid for static "
        'predictive queries. This is forecast step 10 of 40. The window '
        "named above is that step's rather than the one written; ask for "
        'fewer than 10 steps if the data does not reach that far.'
    )


def _buys() -> Schema:
    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE users AS SELECT i::BIGINT AS uid FROM range(20) t(i)'
    )
    connection.execute(
        'CREATE TABLE buys AS SELECT i::BIGINT AS bid, (i % 20)::BIGINT AS '
        "uid, TIMESTAMP '2025-06-01' + i * INTERVAL 1 DAY AS stamp "
        'FROM range(300) t(i)'
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_source(
            DuckDBSource(connection),
            primary_keys={'users': 'uid', 'buys': 'bid'},
            time_columns={'buys': 'stamp'},
        )


def test_a_forecast_s_and_a_ranking_s_refusal_name_their_part_as_the_old_client() -> (
    None
):
    # Through the public calls, word for word as the old client says them,
    # each refusal and its part the core's sentences.
    import pandas as pd

    said = []
    schema = _buys()
    for text, more in (
        (
            'PREDICT COUNT(buys.*, 0, 30, days) FORECAST 40 TIMEFRAMES FOR '
            'EACH users.uid',
            {},
        ),
        (
            'PREDICT COUNT(buys.*, 0, 30, days) FORECAST 3 TIMEFRAMES FOR '
            'EACH users.uid',
            {'anchor_time': pd.Timestamp('1900-01-01')},
        ),
    ):
        with pytest.raises(StructuredDataError) as raised:
            PredictiveQuery(text, schema).materialize(indices=[1, 2], **more)
        assert raised.value._sentence is not None
        said.append(str(raised.value))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with _ranking_source() as source:
            ranked = Schema.from_source(source)
            with pytest.raises(StructuredDataError) as raised:
                PredictiveQuery(RANK, ranked).materialize(
                    indices=[1],
                    candidates=[1, 2],
                    anchor_time=pd.Timestamp('1900-01-01'),
                )
            said.append(str(raised.value))
    assert said == [
        '[INVALID_REQUEST] Context anchor timestamp is too early or '
        'aggregation time range is too large. To make this prediction, we '
        "would need data back to '2025-05-31 00:00:00', however, your data "
        "only contains data back to '2025-06-01 00:00:00'. This is forecast "
        "step 10 of 40. The window named above is that step's rather than "
        'the one written; ask for fewer than 10 steps if the data does not '
        'reach that far.',
        "[INVALID_REQUEST] Anchor timestamp '1900-01-01 00:00:00' is before "
        "the earliest timestamp '2025-06-01 00:00:00' in the data. This is "
        'forecast step 1 of 3.',
        "[INVALID_REQUEST] Anchor timestamp '1900-01-01 00:00:00' is before "
        "the earliest timestamp '2023-01-01 00:00:00' in the data. This is "
        'the question asked for candidate 1, one of 2 this ranking is asked '
        'as.',
    ]


@pytest.mark.parametrize(
    ('case', 'specific'),
    [
        ('missing-column', 'users.ghost'),
        ('incompatible-type', 'users.age'),
        ('negative-seed', 'random_seed'),
        ('context-size', 'num_context_samples'),
    ],
)
def test_public_refusals_have_registered_specific_diagnostics(
    case, specific, monkeypatch
):
    from test_schema_source_selection import (
        _described,
        _hand_built,
        _shop,
        materialize,
    )

    from sdm_client import Column

    decoded = []
    original = errors.error_from_core

    def record(*args, **kwargs):
        error = original(*args, **kwargs)
        decoded.append(error)
        return error

    monkeypatch.setattr(errors, 'error_from_core', record)
    with pytest.raises(StructuredDataError) as raised:
        if case == 'missing-column':
            materialize(
                _hand_built(
                    _shop(), ghost=Column('ghost', 'float64', 'numerical')
                )
            )
        elif case == 'incompatible-type':
            schema = _described(_shop())
            schema['users']['age'].dtype = 'string'
            schema['users']['age'].stype = 'categorical'
            materialize(schema)
        else:
            query = PredictiveQuery(
                'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
                _users(),
            )
            options = (
                {'random_seed': -1}
                if case == 'negative-seed'
                else {'num_context_samples': -1}
            )
            query.materialize(indices=[1], **options)
    assert decoded
    assert specific in str(raised.value)
    assert raised.value.code == 'INVALID_REQUEST'
    for error in decoded:
        assert error._sentence in _sentences()
        assert str(error).split('] ', 1)[-1] not in GENERIC
        assert errors.DEFECT.split('{sentence!r}')[1] not in str(error)
