# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.36 fix group 8: predict of a request whose entity-reference column
was edited answers as the reference does.

    python -m pytest contract/tests/test_t136_entity_ref.py -q

Written by the package's independent test author, before the fix.

A materialized request's ``payloads`` are public, and a caller can rename,
move, drop or refill the predicted instance table's
``__kumo_entity_ref_*`` column before ``SDMClient.predict``. The
reference's ``_named_entities`` (``adapters/relational.py``) reads the
first column whose name starts with ``__kumo_entity_ref`` and, finding
none, gives the answers back as the endpoint wrote them: each row's ``id``
is the instance id the endpoint answered under, as text, and no row is
checked against the payload. Finding one, it reads each entity through the
dtype of the key that column's relationship points at, or as written when
no relationship names the column; a null entity is ``None``. The core
(``sdm-request``'s ``predict::layout``) refused the first with
``UNSUPPORTED_REQUEST`` rule ``entity_ref`` and the second and third with
rule ``id_type``, each after the endpoint had been called. The owner's
standing rule (locked check 5) is that the core matches the reference
where the reference succeeds.

Every outcome here is observed, in this process, on both routes: the
default (the reference's code) and ``SDM_CORE_ROUTES`` forcing the core's
schema, materialize and predict DuckDB routes. Each case materializes the
request on the core, edits its one payload, and predicts it against
``contract/runners/scripted_endpoint.py`` (its rule, or a crafted answer
where a case names one). The two routes must give the same outcome: the
same requests received by the endpoint (count, and bodies with
``metadata.request_id``'s value left out, E35), and either the same result
(each item's fields with the type of its ``id`` and ``prediction``, the
result's model, request id, metadata and outputs, and ``to_pandas()``) or
the same refusal (class and code). No expected value is written by hand;
the non-vacuity checks only assert that the reference answered and, where
the column is gone, that its ids are the instance ids the endpoint was
sent.

- ``test_a_missing_entity_reference_falls_back_to_the_instance_ids``:
  refused today by rule ``entity_ref``. The column renamed (to a plain
  name, to the prefix in upper case, to another column's name: ``age``,
  the anchor column, ``instance_id`` after or before the real one), moved
  and renamed, dropped (with its cells, or its name only), on a
  regression, a binary classification and a text-keyed entity; and the
  endpoint answering in reverse order.
- ``test_an_entity_reference_without_a_known_key_reads_as_the_reference_
  does``: refused today by rule ``id_type``. The column renamed to another
  name with the prefix (no relationship names it, so the reference reads
  the entity as written), a second prefixed column inserted before it
  (the first by position is the one read), all of its cells null, one
  null.
- ``test_without_an_entity_reference_the_answer_is_taken_as_written``:
  refused today by ``entity_ref``. With the column dropped the reference
  checks no row of the answer: answers out of range, repeated, missing,
  extra, under another row's id, without an id, carrying a forecast step,
  or none at all are given back as the endpoint wrote them.
- ``test_edits_the_core_already_answers_stay_the_reference_s``: controls
  that pass today (unchanged, the column moved last, a second prefixed
  column after it, other entity values), so the fix cannot lose them.
- ``test_what_the_reference_refuses_the_core_refuses``: negative controls
  that pass today. The reference refuses them, so the core must not
  accept them: the instance id column renamed too (the endpoint's 400), an
  answer whose id is not text or whose row_index is null with the column
  dropped (the response contract's ``INVALID_RESPONSE``), and, with the
  column kept, each malformed answer above (``INVALID_RESPONSE``).

Added after fix 8, at the coordinator's request: the two differences fix 8
registered for the owner, held as their entries in
``docs/design/rust-core/register.yaml`` state them, so each is reported
rather than hidden (a test asserts the reference's outcome, the core's, and
that they differ; if the core comes to match, it fails and the entry goes
with it):

- ``test_p1b_86_*``: with the column renamed or dropped, an answer whose
  ``row_index`` or ``forecast_step`` is ``2**63``, ``2**130`` or
  ``-2**63 - 1`` is passed through by the reference (in its item and in
  ``to_pandas()``) and refused on the core, ``INVALID_RESPONSE`` with field
  ``prediction.row_index`` or ``prediction.forecast_step`` and reason
  ``out_of_range``. Controls: ``-2**63`` and ``2**63 - 1`` the same on both
  routes, and with the column kept both routes refuse.
- ``test_p1b_87_*``: an entity cell of an integer or a float key edited to
  text holding Tangsa or Kawi five (Unicode 14.0 and 15.0) is the number
  this interpreter's ``int``/``float`` reads on the reference and the text
  on the core; where the interpreter reads none (Python 3.10), both give
  the text. Controls: ASCII and Arabic-Indic five read alike on both
  routes, and a text key keeps every digit as text on both.

Not here: a request holding more than one payload has no core route
(``RouteError`` before any call, register ``P1B-4``'s text), whatever its
entity column.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import duckdb
import pandas as pd
import pytest
from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Request,
    Schema,
    SDMClient,
    _routing,
)
from sdm_connectors import DuckDBSource

ROOT = Path(__file__).resolve().parents[2]
MATERIALIZE = ','.join(
    _routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB
)
CORE = ','.join(
    _routing.FROM_SOURCE_DUCKDB
    + _routing.MATERIALIZE_DUCKDB
    + _routing.PREDICT_DUCKDB
)
ANCHOR = pd.Timestamp('2024-03-01')
PREFIX = '__kumo_entity_ref'

#: A users table keyed by text, for an entity read as ``string``.
TEXT_KEYED = [
    "CREATE TABLE users AS SELECT 'u' || i AS user_id,"
    ' (20 + i)::INTEGER AS age FROM range(8) t(i)',
    'CREATE TABLE orders AS SELECT i::BIGINT AS order_id,'
    " 'u' || (i % 8) AS user_id, (10 + i * 1.5)::DOUBLE AS price,"
    " TIMESTAMP '2023-01-01' + to_days((i * 7)::INTEGER) AS ts"
    ' FROM range(80) t(i)',
]


def _endpoint() -> ModuleType:
    name = 'scripted_endpoint'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'contract/runners/scripted_endpoint.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _shop() -> list[str]:
    lines = (
        (ROOT / 'contract/bases.yaml').read_text(encoding='utf-8').splitlines()
    )
    start = lines.index('  shop:') + 1
    statements = []
    for line in lines[start:]:
        if not line.startswith('  - '):
            break
        statements.append(line[len('  - ') :])
    return statements


#: A users table keyed by a double, for an entity read as ``float64``.
FLOAT_KEYED = [
    'CREATE TABLE users AS SELECT i::DOUBLE AS user_id,'
    ' (20 + i)::INTEGER AS age FROM range(8) t(i)',
    'CREATE TABLE orders AS SELECT i::BIGINT AS order_id,'
    ' (i % 8)::DOUBLE AS user_id, (10 + i * 1.5)::DOUBLE AS price,'
    " TIMESTAMP '2023-01-01' + to_days((i * 7)::INTEGER) AS ts"
    ' FROM range(80) t(i)',
]


#: name: (base, query, indices). A regression and a binary classification
#: over ``int64`` users, and a regression over text-keyed and over
#: double-keyed users.
QUERIES: dict[str, tuple[Callable[[], list[str]], str, list[Any]]] = {
    'regression': (
        _shop,
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
        [0, 1],
    ),
    'classification': (
        _shop,
        'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id',
        [3, 5, 6],
    ),
    'text_key': (
        lambda: TEXT_KEYED,
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
        ['u2', 'u7'],
    ),
    'float_key': (
        lambda: FLOAT_KEYED,
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
        [0.0, 1.0],
    ),
}


@pytest.fixture(autouse=True)
def _old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)


def _materialized(monkeypatch: pytest.MonkeyPatch, query: str) -> Request:
    base, text, indices = QUERIES[query]
    monkeypatch.setenv(_routing.ENVIRONMENT, MATERIALIZE)
    connection = duckdb.connect()
    for statement in base():
        connection.execute(statement)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(connection))
        request = PredictiveQuery(text, schema).materialize(
            indices=list(indices), anchor_time=ANCHOR
        )
    monkeypatch.delenv(_routing.ENVIRONMENT)
    assert request.payloads is not None and len(request.payloads) == 1
    return request


# The edits, each of the one payload's predicted instance table.


def _table(request: Request) -> dict[str, Any]:
    assert request.payloads is not None
    return request.payloads[0]['predict']['instance_table']


def _at(table: dict[str, Any]) -> int:
    return [str(n).startswith(PREFIX) for n in table['columns']].index(True)


def _renamed(to: str) -> Callable[[Request], None]:
    def edit(request: Request) -> None:
        table = _table(request)
        table['columns'][_at(table)] = to

    return edit


def _dropped(request: Request) -> None:
    table = _table(request)
    at = _at(table)
    del table['columns'][at]
    for row in table['rows']:
        del row[at]


def _dropped_name_only(request: Request) -> None:
    table = _table(request)
    del table['columns'][_at(table)]


def _moved(to: int) -> Callable[[Request], None]:
    def edit(request: Request) -> None:
        table = _table(request)
        at = _at(table)
        where = to if to >= 0 else len(table['columns']) - 1
        table['columns'].insert(where, table['columns'].pop(at))
        for row in table['rows']:
            row.insert(where, row.pop(at))

    return edit


def _then(*edits: Callable[[Request], None]) -> Callable[[Request], None]:
    def edit(request: Request) -> None:
        for each in edits:
            each(request)

    return edit


def _moved_then_renamed(to: int, name: str) -> Callable[[Request], None]:
    def edit(request: Request) -> None:
        _moved(to)(request)
        table = _table(request)
        table['columns'][to if to >= 0 else len(table['columns']) - 1] = name

    return edit


def _second_reference(before: bool) -> Callable[[Request], None]:
    def edit(request: Request) -> None:
        table = _table(request)
        where = 0 if before else len(table['columns'])
        table['columns'].insert(where, f'{PREFIX}_1')
        for row in table['rows']:
            row.insert(where, '5')

    return edit


def _nulls(rows: slice) -> Callable[[Request], None]:
    def edit(request: Request) -> None:
        table = _table(request)
        at = _at(table)
        for row in table['rows'][rows]:
            row[at] = None

    return edit


def _other_values(request: Request) -> None:
    table = _table(request)
    at = _at(table)
    for row in table['rows']:
        row[at] = 'u0' if isinstance(row[at], str) else 7


def _instance_renamed(request: Request) -> None:
    table = _table(request)
    table['columns'][table['columns'].index('instance_id')] = 'iid'


def _unchanged(request: Request) -> None:
    pass


# Crafted answers: each prediction of a regression's two rows, under the
# instance ids the materialized request asks with (112 and 113).


def _p(row: Any, id: Any, **more: Any) -> dict[str, Any]:  # noqa: A002
    return {'id': id, 'row_index': row, 'prediction': 1.0, **more}


ANSWERS: dict[str, list[dict[str, Any]]] = {
    'reversed': [_p(1, '113'), _p(0, '112')],
    'out_of_range': [_p(0, '112'), _p(5, '113')],
    'negative_row': [_p(0, '112'), _p(-1, '113')],
    'repeated': [_p(0, '112'), _p(0, '112')],
    'unanswered': [_p(0, '112')],
    'extra': [_p(0, '112'), _p(1, '113'), _p(2, '114')],
    'other_row_s_id': [_p(0, '113'), _p(1, '112')],
    'no_id': [_p(0, None), _p(1, '113')],
    'forecast_step': [_p(0, '112', forecast_step=1), _p(1, '113')],
    'none': [],
    'id_not_text': [_p(0, 112), _p(1, 113)],
    'row_index_null': [_p(None, '112'), _p(1, '113')],
}


def _answering(predictions: list[dict[str, Any]]) -> Callable[..., Any]:
    # A stand-in for the scripted endpoint's rule, taking what the endpoint
    # passes the rule: the request, the call's number and whether to answer
    # probabilities and quantiles (the crafted answer is the same for every
    # call).
    def answer(
        method: str,
        path: str,
        body: bytes,
        *,
        call: int = 0,
        probabilities: bool = False,
        quantiles: bool = False,
    ) -> tuple[int, Any]:
        return 200, {
            'object': 'prediction.response',
            'id': 'r1',
            'model': 'kumo-relational',
            'metadata': {},
            'predictions': predictions,
        }

    return answer


def _masked(body: bytes) -> Any:
    parsed = json.loads(body)
    metadata = parsed.get('metadata')
    if isinstance(metadata, dict) and 'request_id' in metadata:
        metadata['request_id'] = type(metadata['request_id']).__name__
    return parsed


def _typed(value: Any) -> Any:
    return (type(value).__name__, value)


def _observed(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    edit: Callable[[Request], None],
    routes: str | None,
    answer: str | None,
) -> dict[str, Any]:
    r"""One route's outcome: what the endpoint received, and the result or
    the refusal."""
    request = _materialized(monkeypatch, query)
    edit(request)
    endpoint = _endpoint()
    rule = endpoint.answer
    if answer is not None:
        endpoint.answer = _answering(ANSWERS[answer])
    with endpoint.serve() as (url, received):
        if routes is None:
            monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
        else:
            monkeypatch.setenv(_routing.ENVIRONMENT, routes)
        try:
            with SDMClient(url, max_retries=0) as client:
                result = client.predict(request, model=KumoRelational())
            frame = result.to_pandas()
        except Exception as error:  # noqa: BLE001 - the refusal is observed
            details = getattr(error, 'details', None)
            outcome: dict[str, Any] = {
                'refused': (
                    type(error).__name__,
                    getattr(error, 'code', None),
                    details.get('rule') if isinstance(details, dict) else None,
                ),
                # Where the refusal names them; not compared between routes.
                'named': {key: details.get(key) for key in ('field', 'reason')}
                if isinstance(details, dict)
                else {},
            }
        else:
            outcome = {
                'items': [
                    {
                        **dataclasses.asdict(item),
                        'id': _typed(item.id),
                        'prediction': _typed(item.prediction),
                    }
                    for item in result.items
                ],
                'result': (
                    result.model,
                    result.request_id,
                    result.metadata,
                    result.outputs,
                ),
                'frame': frame,
            }
        finally:
            monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
            endpoint.answer = rule
    outcome['sent'] = [_masked(body) for _, _, body in received.requests]
    return outcome


def _held_to_the_reference(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    edit: Callable[[Request], None],
    answer: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = _observed(monkeypatch, query, edit, None, answer)
    core = _observed(monkeypatch, query, edit, CORE, answer)
    assert len(core['sent']) == len(reference['sent'])
    assert core['sent'] == reference['sent']
    assert core.get('refused') == reference.get('refused')
    if 'refused' not in reference:
        assert core['items'] == reference['items']
        assert core['result'] == reference['result']
        pd.testing.assert_frame_equal(core['frame'], reference['frame'])
    return reference, core


def _instance_ids(sent: dict[str, Any]) -> list[str]:
    r"""The instance ids the endpoint was asked under, as its answer spells
    them: the first ``instance_id`` column's cells, as text."""
    table = sent['predict']['instance_table']
    at = table['columns'].index('instance_id')
    return [str(row[at]) for row in table['rows']]


MISSING = {
    'renamed': _renamed('entity'),
    'renamed_upper_case': _renamed(PREFIX.upper() + '_0'),
    'renamed_to_a_feature_s_name': _renamed('age'),
    'renamed_to_the_anchor_column': _renamed('__kumo_anchor_time'),
    'renamed_to_instance_id_after_it': _renamed('instance_id'),
    'moved_first_renamed_to_instance_id': _moved_then_renamed(0, 'instance_id'),
    'moved_last_and_renamed': _moved_then_renamed(-1, 'entity'),
    'dropped': _dropped,
    'dropped_name_only': _dropped_name_only,
}
MISSING_CASES = [
    pytest.param('regression', name, id=f'regression-{name}')
    for name in MISSING
] + [
    pytest.param(query, name, id=f'{query}-{name}')
    for query in ('classification', 'text_key')
    for name in ('renamed', 'dropped')
]


@pytest.mark.parametrize(('query', 'edit'), MISSING_CASES)
def test_a_missing_entity_reference_falls_back_to_the_instance_ids(
    monkeypatch: pytest.MonkeyPatch, query: str, edit: str
) -> None:
    reference, _ = _held_to_the_reference(monkeypatch, query, MISSING[edit])
    # Not vacuous: the reference answered, once, with the instance ids.
    assert 'refused' not in reference, reference['refused']
    [sent] = reference['sent']
    assert not any(
        str(n).startswith(PREFIX)
        for n in sent['predict']['instance_table']['columns']
    )
    assert [i['id'] for i in reference['items']] == [
        ('str', id) for id in _instance_ids(sent)
    ]


def test_a_missing_entity_reference_keeps_the_endpoint_s_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, _ = _held_to_the_reference(
        monkeypatch, 'regression', _dropped, 'reversed'
    )
    assert 'refused' not in reference, reference['refused']
    assert [i['row_index'] for i in reference['items']] == [1, 0]


UNKEYED = {
    'renamed_to_another_reference': _renamed(f'{PREFIX}_1'),
    'renamed_to_the_bare_prefix': _renamed(PREFIX),
    'a_second_reference_before_it': _second_reference(before=True),
    'all_null': _nulls(slice(None)),
    'one_null': _nulls(slice(0, 1)),
}
UNKEYED_CASES = [
    pytest.param('regression', name, id=f'regression-{name}')
    for name in UNKEYED
] + [
    pytest.param(query, name, id=f'{query}-{name}')
    for query in ('classification', 'text_key')
    for name in ('renamed_to_another_reference', 'all_null')
]


@pytest.mark.parametrize(('query', 'edit'), UNKEYED_CASES)
def test_an_entity_reference_without_a_known_key_reads_as_the_reference_does(
    monkeypatch: pytest.MonkeyPatch, query: str, edit: str
) -> None:
    reference, _ = _held_to_the_reference(monkeypatch, query, UNKEYED[edit])
    assert 'refused' not in reference, reference['refused']
    assert len(reference['sent']) == 1


TAKEN_AS_WRITTEN = [
    'out_of_range',
    'negative_row',
    'repeated',
    'unanswered',
    'extra',
    'other_row_s_id',
    'no_id',
    'forecast_step',
    'none',
]


@pytest.mark.parametrize('answer', TAKEN_AS_WRITTEN)
def test_without_an_entity_reference_the_answer_is_taken_as_written(
    monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    reference, _ = _held_to_the_reference(
        monkeypatch, 'regression', _dropped, answer
    )
    assert 'refused' not in reference, reference['refused']
    assert len(reference['items']) == len(ANSWERS[answer])


CONTROLS = {
    'unchanged': _unchanged,
    'moved_last': _moved(-1),
    'a_second_reference_after_it': _second_reference(before=False),
    'other_values': _other_values,
}


@pytest.mark.parametrize(
    ('query', 'edit'),
    [
        pytest.param(query, name, id=f'{query}-{name}')
        for query in QUERIES
        for name in CONTROLS
    ],
)
def test_edits_the_core_already_answers_stay_the_reference_s(
    monkeypatch: pytest.MonkeyPatch, query: str, edit: str
) -> None:
    reference, _ = _held_to_the_reference(monkeypatch, query, CONTROLS[edit])
    assert 'refused' not in reference, reference['refused']


REFUSED = [
    pytest.param(_then(_dropped, _instance_renamed), None, id='no_instance_id'),
    pytest.param(_instance_renamed, None, id='kept-no_instance_id'),
    pytest.param(_dropped, 'id_not_text', id='dropped-id_not_text'),
    pytest.param(_dropped, 'row_index_null', id='dropped-row_index_null'),
] + [
    pytest.param(_unchanged, answer, id=f'kept-{answer}')
    for answer in [*TAKEN_AS_WRITTEN, 'id_not_text', 'row_index_null']
]


@pytest.mark.parametrize(('edit', 'answer'), REFUSED)
def test_what_the_reference_refuses_the_core_refuses(
    monkeypatch: pytest.MonkeyPatch,
    edit: Callable[[Request], None],
    answer: str | None,
) -> None:
    reference, core = _held_to_the_reference(
        monkeypatch, 'regression', edit, answer
    )
    assert 'refused' in reference
    assert core['refused'] == reference['refused']
    assert len(reference['sent']) == 1


def test_the_edits_are_what_they_say(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Each edit changes the column it names, and only the predicted
    instance table."""
    request = _materialized(monkeypatch, 'regression')
    table = _table(request)
    assert table['columns'] == [
        'instance_id',
        f'{PREFIX}_0',
        '__kumo_anchor_time',
    ]
    assert request.payloads is not None
    untouched = copy.deepcopy(request.payloads[0])
    for name, edit in {**MISSING, **UNKEYED, **CONTROLS}.items():
        edited = _materialized(monkeypatch, 'regression')
        edit(edited)
        assert edited.payloads is not None
        payload = edited.payloads[0]
        assert {k: v for k, v in payload.items() if k != 'predict'} == {
            k: v for k, v in untouched.items() if k != 'predict'
        }, name
        columns = payload['predict']['instance_table']['columns']
        if name in MISSING:
            assert not any(str(n).startswith(PREFIX) for n in columns), name
        else:
            assert any(str(n).startswith(PREFIX) for n in columns), name


# The registered differences, held as their entries state them (added after
# fix 8, at the coordinator's request). Each test asserts the reference's
# outcome, the core's, and that they differ; where the core comes to match
# the reference, the test fails, and the entry is to be retired with it.

REGISTER = ROOT / 'docs/design/rust-core/register.yaml'


def _registered(entry: str) -> None:
    assert f'- id: {entry}\n' in REGISTER.read_text(encoding='utf-8'), entry


def _answer_with(field: str, value: int) -> list[dict[str, Any]]:
    first = (
        _p(value, '112')
        if field == 'row_index'
        else _p(0, '112', **{field: value})
    )
    return [first, _p(1, '113')]


for _field in ('row_index', 'forecast_step'):
    for _name, _value in {
        '2**63': 2**63,
        '2**130': 2**130,
        '-2**63-1': -(2**63) - 1,
        '-2**63': -(2**63),
        '2**63-1': 2**63 - 1,
    }.items():
        ANSWERS[f'{_field}={_name}'] = _answer_with(_field, _value)

#: P1B-86: answers whose row_index or forecast_step does not fit int64,
#: below -2**63 or at or past 2**63 (the entry measured 2**63, 2**130 and,
#: for forecast_step, 2**63).
P1B_86 = [
    f'{field}={name}'
    for field in ('row_index', 'forecast_step')
    for name in ('2**63', '2**130', '-2**63-1')
]
#: P1B-86's controls: int64's own ends, the same on both routes.
P1B_86_FITS = [
    f'{field}={name}'
    for field in ('row_index', 'forecast_step')
    for name in ('-2**63', '2**63-1')
]


@pytest.mark.parametrize('answer', P1B_86)
@pytest.mark.parametrize('edit', ['dropped', 'renamed'])
def test_p1b_86_a_row_past_int64_is_refused_where_the_reference_passes_it(
    monkeypatch: pytest.MonkeyPatch, edit: str, answer: str
) -> None:
    r"""Register P1B-86 (approved by the owner, E152): with the entity-reference
    column renamed or dropped, the reference gives the answer back as
    written, the number in its item and in ``to_pandas()``; the core refuses
    it, ``INVALID_RESPONSE`` naming the field, reason ``out_of_range``."""
    _registered('P1B-86')
    field = answer.split('=')[0]
    value = ANSWERS[answer][0][field]
    reference = _observed(
        monkeypatch, 'regression', MISSING[edit], None, answer
    )
    core = _observed(monkeypatch, 'regression', MISSING[edit], CORE, answer)
    assert core['sent'] == reference['sent']
    assert len(reference['sent']) == 1
    assert 'refused' not in reference, reference['refused']
    assert reference['items'][0][field] == value
    assert value in reference['frame'][field].tolist()
    assert core.get('refused') == (
        'StructuredDataError',
        'INVALID_RESPONSE',
        None,
    ), core.get('items')
    assert core['named'] == {
        'field': f'prediction.{field}',
        'reason': 'out_of_range',
    }


@pytest.mark.parametrize('answer', P1B_86_FITS)
@pytest.mark.parametrize('edit', ['dropped', 'renamed'])
def test_p1b_86_a_row_at_int64_s_ends_is_the_reference_s(
    monkeypatch: pytest.MonkeyPatch, edit: str, answer: str
) -> None:
    reference, _ = _held_to_the_reference(
        monkeypatch, 'regression', MISSING[edit], answer
    )
    field = answer.split('=')[0]
    assert 'refused' not in reference, reference['refused']
    assert reference['items'][0][field] == ANSWERS[answer][0][field]


@pytest.mark.parametrize('answer', P1B_86)
def test_p1b_86_with_the_column_kept_both_refuse(
    monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    reference, _ = _held_to_the_reference(
        monkeypatch, 'regression', _unchanged, answer
    )
    assert reference['refused'][1] == 'INVALID_RESPONSE'


def _cells(text: str) -> Callable[[Request], None]:
    def edit(request: Request) -> None:
        table = _table(request)
        at = _at(table)
        for row in table['rows']:
            row[at] = text

    return edit


#: P1B-87: entity text holding a decimal digit Unicode added after 13.0:
#: Tangsa five (Unicode 14.0) and Kawi five (15.0), alone, after an ASCII
#: digit, and before a fraction.
LATER_DIGITS = {
    'tangsa_five': '\U00016ac5',
    'kawi_five': '\U00011f55',
    'one_kawi_five': '1\U00011f55',
    'tangsa_five_point_five': '\U00016ac5.5',
}
#: Its controls: Unicode 13.0's own digits (ASCII, Arabic-Indic five).
EARLIER_DIGITS = {'ascii_five': '5', 'arabic_indic_five': '\u0665'}
#: The key each is read back as, and the reading of the reference's
#: ``as_declared`` there (Python's ``int`` or ``float`` on text).
KEYS: dict[str, Callable[[str], Any]] = {'regression': int, 'float_key': float}


def _python_reads(read: Callable[[str], Any], text: str) -> Any:
    r"""The number this interpreter's ``int`` or ``float`` reads ``text``
    as, or ``None`` where it reads none."""
    try:
        return read(text)
    except ValueError:
        return None


@pytest.mark.parametrize('digit', list(LATER_DIGITS))
@pytest.mark.parametrize('query', list(KEYS))
def test_p1b_87_a_later_digit_stays_text_where_python_reads_a_number(
    monkeypatch: pytest.MonkeyPatch, query: str, digit: str
) -> None:
    r"""Register P1B-87 (approved by the owner, E152): an entity cell edited to
    text with a digit Unicode added after 13.0 is read back by the
    reference as the number this interpreter's ``int``/``float`` reads (on
    Python 3.12, the reference's), and given back as the text on the core,
    as Python 3.10 gives it. Where this interpreter reads no number, both
    give the text and there is no difference to report."""
    _registered('P1B-87')
    text = LATER_DIGITS[digit]
    number = _python_reads(KEYS[query], text)
    reference = _observed(monkeypatch, query, _cells(text), None, None)
    core = _observed(monkeypatch, query, _cells(text), CORE, None)
    assert core['sent'] == reference['sent']
    assert 'refused' not in reference, reference['refused']
    assert 'refused' not in core, core['refused']
    as_text = [('str', text)] * len(core['items'])
    assert [i['id'] for i in core['items']] == as_text
    if number is None:
        assert [i['id'] for i in reference['items']] == as_text
        return
    assert [i['id'] for i in reference['items']] == [
        (type(number).__name__, number)
    ] * len(reference['items'])
    assert str(reference['frame']['id'].dtype) != 'str'
    assert str(core['frame']['id'].dtype) == 'str'


@pytest.mark.parametrize(
    ('query', 'digit'),
    [
        *((query, digit) for query in KEYS for digit in EARLIER_DIGITS),
        *(('text_key', digit) for digit in [*EARLIER_DIGITS, *LATER_DIGITS]),
    ],
)
def test_p1b_87_s_controls_are_the_reference_s(
    monkeypatch: pytest.MonkeyPatch, query: str, digit: str
) -> None:
    r"""Unicode 13.0's digits read as the reference reads them, for an
    integer or a float key; a text key keeps every digit as text."""
    text = {**EARLIER_DIGITS, **LATER_DIGITS}[digit]
    reference, _ = _held_to_the_reference(monkeypatch, query, _cells(text))
    assert 'refused' not in reference, reference['refused']
    if query in KEYS:
        number = KEYS[query](text)
        assert [i['id'] for i in reference['items']] == [
            (type(number).__name__, number)
        ] * len(reference['items'])
