# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The core picks every refusal's and warning's sentence, and this host
fills it (T3.17, the Phase 3 structure design's fix B, invariant 8).

The core sets a sentence, from one closed list, on every error and warning
it gives; each sentence has one shared template, which the binding exports
(``_native.templates()``), and each advice a template names one spelling in
this host (``errors.ADVICE``). This host chooses no sentence: it fills the
core's with its values, as the old client wrote them, and its advice, as
Python is written. A refusal with no sentence is a bug: the generic
sentence is gone, and the census's probes on the core route show none.

Each public call runs on the core route, every core route forced, and is
held to the old client's words.
"""

from __future__ import annotations

import importlib.util
import json
import string
import sys
import tempfile
import warnings
from collections.abc import Iterator
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
    _routing,
)
from sdm_client._rust import errors

ROOT = Path(__file__).resolve().parents[3]
#: Every path this build can run on the core, forced.
CORE = ','.join(sorted(_routing.CORE_ROUTED))
#: The sentences a refusal said before T3.17 when no template of the
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


@pytest.fixture
def core(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    yield


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
        assert entry['kind'] in ('error', 'warning'), name
        if entry['kind'] == 'error':
            assert entry['code'] in _native.ERROR_CODES, name
        else:
            assert entry['code'] == entry['code'].upper(), name
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
    from sdm_client.client import _SCORE_THE_UNSCORED

    given = {1: 'gold', 2: 'silver', 3: 'bronze'}

    def named(fields: dict[str, Any]) -> list[Any]:
        return [given[at] for at in fields['candidates']]

    several = _warning('candidates_unscored_several', candidates=[1, 3])
    one = _warning('candidates_unscored_one', candidates=[2])
    assert _shown(several, one, host={'candidates': named}) == [
        "[CANDIDATES_UNSCORED] candidates 'gold' and 'bronze' are left out of "
        'this ranking, unscored: the context sampled for each holds one '
        'answer only, nearly always because no entity in it linked to the '
        'candidate in the window, and a classification needs two to learn '
        'from. No call was sent for them, so each row ranks only the rest, '
        "and the result's unscored_candidates names them. "
        + _SCORE_THE_UNSCORED,
        "[CANDIDATES_UNSCORED] candidate 'silver' is left out of this "
        'ranking, unscored: the context sampled for it holds one answer only, '
        'nearly always because no entity in it linked to the candidate in the '
        'window, and a classification needs two to learn from. No call was '
        "sent for it, so each row ranks only the rest, and the result's "
        'unscored_candidates names it. ' + _SCORE_THE_UNSCORED,
    ]


# --- the public calls -----------------------------------------------------------


def test_a_key_check_s_advice_is_python(core: None) -> None:
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


def test_a_schema_edit_s_advice_is_python_and_notices_group(
    core: None,
) -> None:
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


def test_a_primary_key_linked_as_a_foreign_key_is_said_as_the_old_client_says_it(
    core: None,
) -> None:
    # M15 of T3.16's review: the old client's engine refuses it
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


def _census() -> Any:
    name = 'refusal_census'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'contract/tools/refusal_census.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_no_refusal_on_the_census_s_core_route_says_a_generic_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    census = _census()
    decoded: list[StructuredDataError] = []
    reading = errors.error_from_core

    def recording(*args: Any, **kwargs: Any) -> StructuredDataError:
        error = reading(*args, **kwargs)
        decoded.append(error)
        return error

    monkeypatch.setattr(errors, 'error_from_core', recording)
    routes = census._routes()
    probes = [
        probe
        for probe in census.probes(ROOT)
        if 'duckdb' in probe.get('databases', census.DATABASES)
    ]
    with tempfile.TemporaryDirectory(prefix='sdm-t317-') as scratch:
        cases = Path(scratch)
        for number, probe in enumerate(probes):
            (cases / f'{number}.json').write_text(
                json.dumps(
                    {
                        'case': census._case(probe, number, routes),
                        'options': probe.get('options', {}),
                        'model': probe.get('model', 'KumoRelational'),
                    }
                )
            )
        census.route(cases, cases / 'out.json', 'duckdb', core=True)
    assert len(decoded) >= 15
    for error in decoded:
        said = str(error).split('] ', 1)[1]
        assert said not in GENERIC, (error.code, error.details)
        assert error._sentence in _sentences(), error.code
