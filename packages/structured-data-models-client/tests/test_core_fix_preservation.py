# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import sqlite3
import warnings
from collections import UserDict, UserList
from copy import copy, deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Request,
    Schema,
    SDMClient,
)
from sdm_client.errors import StructuredDataError


@pytest.mark.parametrize('old', [True, False])
@pytest.mark.parametrize('family', ['plain', 'decimal', 'mapping', 'cycle'])
@pytest.mark.parametrize('half', ['context', 'predict'])
@pytest.mark.parametrize('depth', [0, 1, 2])
@pytest.mark.parametrize('gathered', [True, False])
def test_materialized_setters_preserve_typed_gathered_refusal(
    monkeypatch, old, family, half, depth, gathered
):
    from sdm_core import _native
    from sdm_core.answers import HeldValues

    from sdm_client import _impl

    monkeypatch.setattr(_impl, 'OLD', old)
    payload = {
        'task': {
            'kind': 'classification',
            'target': {'column_name': 'label', 'dtype': 'int64'},
        }
    }
    if family == 'decimal':
        payload['extra'] = Decimal('1.25')
    elif family == 'mapping':
        payload = UserDict(payload)
    elif family == 'cycle':
        payload['extra'] = payload
    request = Request.materialized(
        payloads=(payload,),
        schema=Schema({}),
        query='synthetic',
        indices=[],
        entity_table='items',
    )
    if not gathered:
        request._payloads = ()
    for _ in range(depth):
        parent = copy(request)
        parent._payloads = ()
        parent._parts = (request,)
        request = parent
    original = object()
    replacement = object()
    setattr(request, '_' + half, original)
    calls = []
    native = _native.request_op

    def observed(*args):
        calls.append(args)
        return native(*args)

    monkeypatch.setattr(_native, 'request_op', observed)
    if gathered:
        with pytest.raises(StructuredDataError) as caught:
            setattr(request, half, replacement)
        assert type(caught.value) is StructuredDataError
        assert caught.value.code == 'INVALID_REQUEST'
        assert str(caught.value) == (
            '[INVALID_REQUEST] '
            f'{half} cannot be replaced on a request gathered by a predictive '
            'query: its rows are already encoded, and the encoding carries '
            'what a frame does not. Materialize the query again for the rows '
            'you want instead.'
        )
        assert getattr(request, '_' + half) is original
        assert request.payloads[0] is payload
    else:
        setattr(request, half, replacement)
        assert getattr(request, '_' + half) is replacement
    if old:
        assert calls == []
    else:
        assert len(calls) == 1
        assert calls[0][0] == 'refuse_if_gathered'
        assert isinstance(calls[0][3].__self__, HeldValues)


def _sequence_query():
    return PredictiveQuery(
        'PREDICT users.target FOR EACH users.user_id',
        {
            'users': pd.DataFrame(
                {'user_id': [1, 2, 3], 'target': [3.1, 4.2, 5.3]}
            )
        },
    )


@pytest.mark.parametrize(
    ('indices', 'scored', 'context'),
    [
        (range(1, 1), [], [1, 2, 3]),
        (range(1, 2), [1], [2, 3]),
        (range(1, 4, 2), [1, 3], [2]),
        (range(3, 0, -2), [3, 1], [2]),
        (UserList(), [], [1, 2, 3]),
        (UserList([1]), [1], [2, 3]),
        (UserList([3, 1, 3]), [3, 1, 3], [2]),
        ([1], [1], [2, 3]),
        ((1,), [1], [2, 3]),
    ],
)
def test_public_materialize_preserves_sequence_indices(
    indices, scored, context
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        request = _sequence_query().materialize(indices=indices)
    assert [
        (warning.category.__name__, str(warning.message)) for warning in caught
    ] == (
        [
            (
                'SDMWarning',
                'every context row has the same target, so nothing in the '
                'context distinguishes one answer from another, so the answer '
                'cannot say much. If the rows being scored are the ones that '
                'differ, score fewer of them; if target is constant in the whole '
                'table, there is no context that answers this question.',
            )
        ]
        if len(context) == 1
        else []
    )
    assert request.payloads is None
    assert request.predict.instance_table.to_dict('list') == {'user_id': scored}
    assert request.context.instance_table.to_dict('list') == {
        'user_id': context,
        'target': [{1: 3.1, 2: 4.2, 3: 5.3}[key] for key in context],
    }
    assert request.predict.instance_table.dtypes.to_dict() == {
        'user_id': 'int64'
    }
    assert request.context.instance_table.dtypes.to_dict() == {
        'user_id': 'int64',
        'target': 'float64',
    }
    assert request.task.kind == 'regression'
    assert request.task.target.column_name == 'target'
    assert request.task.target.dtype == 'float64'
    assert request.task.target.table_name == 'users'
    assert request.task.options == {}
    assert request.entity_table == 'users'
    assert request.predict_rows == len(scored)
    assert request.context_rows == len(context)


@pytest.mark.parametrize('indices', [range(1, 2), UserList([1])])
def test_sequence_regression_detects_restored_opaque_boundary(
    monkeypatch, indices
):
    from sdm_client import _request_core

    original = _request_core._given

    def opaque(args):
        encoded, frames, _sequence = original(args)
        return encoded, frames, None

    monkeypatch.setattr(_request_core, '_given', opaque)
    with pytest.raises(StructuredDataError) as refused:
        test_public_materialize_preserves_sequence_indices(indices, [1], [2, 3])
    assert refused.value.code == 'INVALID_REQUEST'
    assert str(refused.value) == (
        '[INVALID_REQUEST] indices must be a sequence of entity keys, got '
        + type(indices).__name__
    )


@pytest.mark.parametrize('container', [range, UserList])
def test_sequence_indices_retain_integer_precision(container):
    key = 2**53 + 1
    indices = range(key, key + 1) if container is range else UserList([key])
    query = PredictiveQuery(
        'PREDICT users.target FOR EACH users.user_id',
        {
            'users': pd.DataFrame(
                {'user_id': [key, key + 1, key + 2], 'target': [3.1, 4.2, 5.3]}
            )
        },
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        request = query.materialize(indices=indices)
    assert caught == []
    assert request.predict.instance_table.to_dict('list') == {'user_id': [key]}
    assert request.context.instance_table.to_dict('list') == {
        'user_id': [key + 1, key + 2],
        'target': [4.2, 5.3],
    }


@pytest.mark.parametrize('family', ['ndarray', 'index', 'generator'])
def test_sequence_boundary_does_not_accept_other_iterables(family):
    import numpy as np

    indices = {
        'ndarray': np.array([1]),
        'index': pd.Index([1]),
        'generator': (value for value in [1]),
    }[family]
    with pytest.raises(StructuredDataError) as refused:
        _sequence_query().materialize(indices=indices)
    assert refused.value.code == 'INVALID_REQUEST'
    assert str(refused.value) == (
        '[INVALID_REQUEST] indices must be a sequence of entity keys, got '
        + type(indices).__name__
    )


def test_sequence_indices_refuse_unhashable_values_at_validation():
    with pytest.raises(StructuredDataError) as refused:
        _sequence_query().materialize(indices=UserList([[1]]))
    assert refused.value.code == 'INVALID_REQUEST'
    assert str(refused.value) == (
        '[INVALID_REQUEST] indices must be entity keys, and 1 of the 1 given '
        "cannot be one: ['list']. A key is looked up by value, so it has to be "
        'hashable. This is what a row of json.loads(...) or '
        'frame.to_dict("records") gives you; pass the key field of each '
        "instead, e.g. [row['<key>'] for row in rows]."
    )


class _ObservedSequence(UserList):
    def __init__(self, events, failure=None):
        super().__init__([1])
        self.events = events
        self.failure = failure
        self.iterations = 0

    def __iter__(self):
        self.events.append('iter')
        self.iterations += 1
        if self.failure == self.iterations:
            raise RuntimeError('sequence iteration sentinel')
        return iter(self.data)

    def __len__(self):
        self.events.append('len')
        if self.failure == 'len':
            raise RuntimeError('sequence length sentinel')
        return len(self.data)


@pytest.mark.parametrize(
    ('failure', 'events'),
    [(1, ['iter']), (2, ['iter', 'iter']), ('len', ['iter', 'iter', 'len'])],
)
def test_sequence_protocol_errors_are_not_eager_or_swallowed(failure, events):
    observed = []
    sequence = _ObservedSequence(observed, failure)
    with pytest.raises(RuntimeError, match='sequence .* sentinel'):
        _sequence_query().materialize(indices=sequence)
    assert observed == events


@pytest.mark.parametrize(
    'invalid', ['both', 'model', 'option', 'count', 'query']
)
def test_sequence_protocol_observes_original_validation_order(invalid):
    observed = []
    sequence = _ObservedSequence(observed)
    query = _sequence_query()
    keywords = {}
    if invalid == 'both':
        keywords['predict'] = pd.DataFrame({'user_id': [1]})
    elif invalid == 'model':
        keywords['model'] = 'invalid-model'
    elif invalid == 'option':
        keywords['unknown_option'] = True
    elif invalid == 'count':
        keywords['num_context_samples'] = 0
    else:
        query.text = 'PREDICT users.absent FOR EACH users.user_id'
    with pytest.raises(StructuredDataError):
        query.materialize(indices=sequence, **keywords)
    assert observed == (['iter'] if invalid in {'count', 'query'} else [])


def test_sequence_protocol_has_two_distinct_successful_iterations():
    observed = []
    test_public_materialize_preserves_sequence_indices(
        _ObservedSequence(observed), [1], [2, 3]
    )
    assert observed == ['iter', 'iter', 'len']


def _holdout_request():
    return PredictiveQuery(
        'PREDICT users.label FOR EACH users.user_id',
        {
            'users': pd.DataFrame(
                {
                    'user_id': range(600),
                    'value': [index * 2 for index in range(600)],
                    'label': [
                        'a' if index % 2 else 'b' for index in range(600)
                    ],
                }
            )
        },
    ).materialize(indices=[1, 2], num_context_samples=600)


def _fit_holdout(holdout, **keywords):
    request = _holdout_request()
    sent = []
    events = []

    class Transport:
        def create_session(self, payload):
            events.append('create_session')
            sent.append(deepcopy(payload))
            return {'session_id': 'synthetic-holdout-session'}

        def delete_session(self, session_id):
            assert session_id == 'synthetic-holdout-session'
            events.append('delete_session')

    client = SDMClient.__new__(SDMClient)
    client._start(Transport())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fitted = client.fit(
            request, model='kumo-tabular', holdout=holdout, **keywords
        )
        held_back = fitted._held_back
        assert fitted._pinned['metadata'] == sent[0]['metadata']
        fitted.close()
    assert events == ['create_session', 'delete_session']
    assert len(sent) == 1
    assert re.fullmatch(r'sdm-[0-9a-f]{12}', sent[0]['metadata']['request_id'])
    return (
        sent[0],
        held_back,
        [
            (warning.category.__name__, str(warning.message))
            for warning in caught
        ],
    )


@pytest.mark.parametrize('family', ['list', 'range', 'generator', 'userlist'])
def test_public_fit_excludes_every_labelled_iterable_holdout(family):
    holdout = {
        'list': lambda: list(range(200, 260)),
        'range': lambda: range(200, 260),
        'generator': lambda: (key for key in range(200, 260)),
        'userlist': lambda: UserList(range(200, 260)),
    }[family]()
    payload, held_back, caught = _fit_holdout(holdout)
    assert held_back == 60
    assert payload['context']['instance_table'] == {
        'format': 'arrays',
        'columns': ['user_id', 'value', 'label'],
        'rows': [
            [key, key * 2, 'a' if key % 2 else 'b']
            for key in range(600)
            if key not in {1, 2} and key not in range(200, 260)
        ],
    }
    assert len(payload['context']['instance_table']['rows']) == 538
    assert payload['schema']['instance_table']['primary_key'] == ['user_id']
    assert payload['model'] == 'kumo-tabular'
    assert caught == [
        (
            'SDMWarning',
            '60 of the rows named as holdout were in the context about to be '
            'pinned, carrying the answers this session will be asked for. '
            'They have been taken out of it, so the context is that many '
            'rows smaller than the request gathered.',
        )
    ]


@pytest.mark.parametrize('family', ['range', 'generator'])
def test_iterable_holdout_regression_detects_opaque_boundary(
    monkeypatch, family
):
    from sdm_core import _native

    original = _native.fit_run

    def opaque(state, tables, _protocol):
        return original(state, tables)

    monkeypatch.setattr(_native, 'fit_run', opaque)
    with pytest.raises(AssertionError, match='assert 0 == 60'):
        test_public_fit_excludes_every_labelled_iterable_holdout(family)


@pytest.mark.parametrize('holdout', ['200', b'200'])
def test_iterable_holdout_keeps_string_and_bytes_refusals(holdout):
    with pytest.raises(StructuredDataError) as refused:
        _fit_holdout(holdout)
    assert refused.value.code == 'INVALID_REQUEST'
    assert str(refused.value) == (
        '[INVALID_REQUEST] holdout names the rows this session may score, and '
        f'{holdout!r} is one string rather than a list of keys; '
        f'pass [{holdout!r}] if that is the key you mean'
    )


def test_iterable_holdout_does_not_ask_for_length():
    events = []

    class Holdout:
        def __iter__(self):
            events.append('iter')
            return iter(range(200, 260))

        def __len__(self):
            raise AssertionError('holdout has no length contract')

    _payload, held_back, _warnings = _fit_holdout(Holdout())
    assert held_back == 60
    assert events == ['iter']


def test_iterable_holdout_does_not_swallow_iteration_errors():
    sentinel = RuntimeError('holdout iteration sentinel')

    def holdout():
        yield 200
        raise sentinel

    with pytest.raises(RuntimeError) as refused:
        _fit_holdout(holdout())
    assert refused.value is sentinel


def test_iterable_holdout_is_reusable_only_when_its_protocol_is():
    holdout = (key for key in range(200, 260))
    _first, held_first, _first_warnings = _fit_holdout(holdout)
    second, held_second, second_warnings = _fit_holdout(holdout)
    assert held_first == 60
    assert held_second == 0
    assert second_warnings == []
    assert len(second['context']['instance_table']['rows']) == 598


def _sqlite_latest_request(
    directory: Path, unreadable: bool
) -> tuple[Any, list[dict[str, Any]], list[str]]:
    dbapi = pytest.importorskip('adbc_driver_sqlite.dbapi')
    from sdm_connectors import SQLiteSource

    users = pd.DataFrame({'user_id': range(1, 21), 'age': range(20, 40)})
    orders = pd.DataFrame(
        {
            'order_id': range(1, 201),
            'user_id': [1 + index % 20 for index in range(200)],
            'quantity': [1 + index % 5 for index in range(200)],
            'ordered_at': pd.date_range(
                '2024-01-01', periods=200, freq='D'
            ).strftime('%Y-%m-%d %H:%M:%S'),
        }
    )
    if unreadable:
        orders.loc[199, 'ordered_at'] = 'zz-invalid'
    else:
        orders = orders.iloc[:199]
    path = directory / ('bad.db' if unreadable else 'readable.db')
    with sqlite3.connect(path) as written:
        users.to_sql('users', written, index=False)
        orders.to_sql('orders', written, index=False)
    calls: list[dict[str, Any]] = []
    with dbapi.connect(str(path)) as connection:
        source = SQLiteSource(connection)
        original = source._run

        def observed(sql, params=None):
            result = original(sql, params)
            calls.append(
                {'sql': sql, 'params': params, 'rows': result.to_pylist()}
            )
            return result

        source._run = observed
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            schema = Schema.from_source(
                source,
                primary_keys={'users': 'user_id', 'orders': 'order_id'},
                time_columns={'orders': 'ordered_at'},
            )
            request = PredictiveQuery(
                'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
                schema,
            ).materialize(indices=[1, 2], num_context_samples=3, random_seed=7)
    return request, calls, [str(warning.message) for warning in caught]


def test_unreadable_latest_sqlite_bound_has_exact_readable_rows(tmp_path):
    bad, calls, bad_warnings = _sqlite_latest_request(tmp_path, True)
    readable, readable_calls, readable_warnings = _sqlite_latest_request(
        tmp_path, False
    )
    payload = json.loads(json.dumps(bad.payloads, default=str))
    assert payload == json.loads(json.dumps(readable.payloads, default=str))
    assert bad_warnings == readable_warnings
    bounds = [call for call in calls if 'MAX("ordered_at")' in call['sql']]
    assert len(bounds) == 1
    assert bounds[0]['params'] is None
    assert bounds[0]['rows'] == [
        {
            'table_index': 0,
            'min_date': '2024-01-01 00:00:00',
            'max_date': 'zz-invalid',
        }
    ]
    searches = [call for call in calls if 'ORDER BY 1 DESC' in call['sql']]
    assert len(searches) == 1
    assert searches[0]['sql'] == (
        'SELECT DISTINCT "ordered_at" AS __kumo_time__ FROM "orders" '
        'WHERE "ordered_at" IS NOT NULL ORDER BY 1 DESC LIMIT 1000 OFFSET 0'
    )
    assert searches[0]['params'] is None
    assert searches[0]['rows'] == [
        {'__kumo_time__': 'zz-invalid'},
        *(
            {'__kumo_time__': stamp}
            for stamp in pd.date_range(
                '2024-01-01', periods=199, freq='D'
            ).strftime('%Y-%m-%d %H:%M:%S')[::-1]
        ),
    ]
    assert not any('ORDER BY 1 DESC' in call['sql'] for call in readable_calls)
    assert payload[0]['context']['instance_table']['rows'] == [
        [0, 18, '2024-06-17T00:00:00.000000Z', 2.0],
        [1, 17, '2024-06-17T00:00:00.000000Z', 2.0],
        [2, 13, '2024-06-17T00:00:00.000000Z', 2.0],
    ]
    assert payload[0]['predict']['instance_table']['rows'] == [
        [3, 1, '2024-07-17T00:00:00.000000Z'],
        [4, 2, '2024-07-17T00:00:00.000000Z'],
    ]


@pytest.mark.parametrize('explicit', [True, False])
def test_materialized_payloads_keep_entity_task_and_public_dispatch(explicit):
    payload = {
        'task': {
            'kind': 'regression',
            'target': {'column_name': 'amount', 'dtype': 'float64'},
            'entity_table_names': ['people'],
        },
        'schema': {
            'instance_table': {
                'columns': {
                    'instance_id': {'dtype': 'int64', 'stype': 'ID'},
                    'amount': {'dtype': 'float64', 'stype': 'numerical'},
                },
                'primary_key': 'instance_id',
            }
        },
        'context': {
            'instance_table': {
                'format': 'arrays',
                'columns': ['instance_id', 'amount'],
                'rows': [[0, 1.25], [1, 2.5]],
            }
        },
        'predict': {
            'instance_table': {
                'format': 'arrays',
                'columns': ['instance_id'],
                'rows': [[2]],
            }
        },
    }
    original = deepcopy(payload)
    request = Request.materialized(
        payloads=(payload,),
        schema=Schema({}),
        query='PREDICT people.amount FOR EACH people.person_id',
        indices=[2],
        entity_table='people' if explicit else None,
    )
    assert request.payloads == (original,)
    assert request.entity_table == request.task.entity_table == 'people'
    assert request.task.kind == 'regression'
    assert request.task.target.column_name == 'amount'
    assert request.task.target.dtype == 'float64'
    assert request.context_rows == 2
    assert request.predict_rows == 1
    sent = []

    class Transport:
        def predict(self, body):
            sent.append(deepcopy(body))
            return {
                'object': 'prediction.response',
                'id': 'proof-response',
                'model': 'kumo-relational',
                'metadata': {},
                'predictions': [
                    {'id': '2', 'row_index': 0, 'prediction': 3.75}
                ],
            }

    client = SDMClient.__new__(SDMClient)
    client._start(Transport())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = client.predict(request, model='kumo-relational')
    assert len(sent) == 1
    for section in ('task', 'schema', 'context', 'predict'):
        assert sent[0][section] == original[section]
    assert sent[0]['model'] == 'kumo-relational'
    assert sent[0]['output'] == {
        'fields': ['prediction'],
        'prediction_statistic': 'median',
    }
    assert [item.prediction for item in result.items] == [3.75]
    assert request.payloads == (original,)


@pytest.mark.parametrize('defect', ['drop-payload', 'wrong-entity'])
def test_materialized_preservation_detects_a_broken_constructor(
    monkeypatch, defect
):
    construct = Request.materialized

    def broken(**kwargs):
        request = construct(**kwargs)
        if defect == 'drop-payload':
            request._payloads = ()
        else:
            request.entity_table = 'unrelated'
        return request

    monkeypatch.setattr(Request, 'materialized', broken)
    with pytest.raises(AssertionError):
        test_materialized_payloads_keep_entity_task_and_public_dispatch(False)


def test_live_tables_mutations_reach_the_next_gather():
    from sdm_client.errors import StructuredDataError

    people = pd.DataFrame(
        {
            'person_id': range(20),
            'amount': [float(index) for index in range(20)],
        }
    )
    extra = pd.DataFrame({'extra_id': range(20), 'person_id': range(20)})
    schema = Schema.from_data({'people': people, 'extra': extra})
    tables = schema.tables
    assert tables is schema.tables
    del tables['extra']
    assert list(schema.tables) == ['people']
    with pytest.raises(StructuredDataError, match="no table 'extra'"):
        PredictiveQuery(
            'PREDICT people.amount FOR EACH people.person_id', schema
        ).materialize(indices=[1, 2], random_seed=7, model=KumoRelational())
    schema.relationships = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(
            'PREDICT people.amount FOR EACH people.person_id', schema
        ).materialize(indices=[1, 2], random_seed=7, model=KumoRelational())
    for part in ('context', 'predict'):
        assert 'extra' not in request.payloads[0][part]['related_tables']
    assert 'extra' not in request.payloads[0]['schema']['related_tables']


def test_live_tables_proof_rejects_a_copied_mapping(monkeypatch):
    original = Schema.tables.fget
    monkeypatch.setattr(
        Schema, 'tables', property(lambda self: dict(original(self)))
    )
    with pytest.raises(AssertionError):
        test_live_tables_mutations_reach_the_next_gather()


def test_live_tables_proof_rejects_a_stale_gather_snapshot(monkeypatch):
    from sdm_client import _schema_core

    construct = Schema.from_data
    describe = _schema_core.schema_state
    snapshots = {}

    def capture(*args, **kwargs):
        schema = construct(*args, **kwargs)
        snapshots[id(schema)] = describe(schema)
        return schema

    monkeypatch.setattr(Schema, 'from_data', capture)
    monkeypatch.setattr(
        _schema_core,
        'schema_state',
        lambda schema: snapshots[id(schema)],
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(AssertionError, match='extra'):
            test_live_tables_mutations_reach_the_next_gather()


def test_unseeded_public_query_continues_its_neighbor_stream():
    users = pd.DataFrame({'user_id': [0], 'age': [25]})
    orders = pd.DataFrame(
        {
            'order_id': range(40),
            'user_id': [0] * 40,
            'amount': range(40),
        }
    )
    schema = Schema.from_data({'users': users, 'orders': orders})
    question = 'PREDICT users.age FOR EACH users.user_id'
    query = PredictiveQuery(question, schema)
    payloads = []
    messages = []
    for current in (query, query, PredictiveQuery(question, schema)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            request = current.materialize(
                indices=[0],
                num_context_samples=1,
                num_neighbors=[2],
                random_seed=None,
            )
        payloads.append(json.loads(json.dumps(request.payloads, default=str)))
        messages.append([str(warning.message) for warning in caught])
    first, second, fresh = payloads
    assert first == fresh
    assert first[0]['predict']['related_tables']['orders']['rows'] == [
        [1, 2, 33, 0],
        [1, 3, 10, 0],
    ]
    expected_second = deepcopy(first)
    expected_second[0]['predict']['related_tables']['orders']['rows'] = [
        [1, 2, 9, 0],
        [1, 3, 17, 0],
    ]
    assert second == expected_second
    assert first[0]['context']['instance_table']['rows'] == []
    assert first[0]['predict']['instance_table']['rows'] == [
        [1, 0, '2262-04-11T23:47:16.709551Z']
    ]
    expected_warning = (
        '1 of the rows sampled as context are the rows being scored, '
        'at one moment or another, carrying the value being asked '
        'for. Rows that carry the answer to the question being asked '
        'are not sent as context, so these do not go into it.'
    )
    assert messages == [[expected_warning]] * 3
