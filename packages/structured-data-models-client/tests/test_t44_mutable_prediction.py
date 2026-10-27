# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public mutable-payload prediction against two actual frozen runtimes.

Only HTTP delivery is mocked; actual native preflight and transport JSON
traversals, paid-prefix errors and post-response entity reads remain live.
Materialization stays on the original route to isolate prediction ownership.
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient
from sdm_client.adapters.relational import RelationalAdapter

REFERENCE = json.loads(
    Path(__file__)
    .with_name('t44_mutable_prediction_reference.json')
    .read_text()
)['cases']
TASK_REFERENCE = json.loads(
    Path(__file__)
    .with_name('t44_mutable_prediction_reference.json')
    .read_text()
)['task_cases']
FAULT_REFERENCE = json.loads(
    Path(__file__)
    .with_name('t44_mutable_prediction_reference.json')
    .read_text()
)['fault_cases']


def _request(*, batched=False):
    data = {
        'users': pd.DataFrame(
            {
                'user_id': np.arange(8, dtype='int64'),
                'age': np.arange(20, 28, dtype='int64'),
            }
        ),
        'items': pd.DataFrame(
            {
                'item_id': pd.Series(range(4), dtype='int64'),
                'amount': [3.0, 4.0, 5.0, 6.0],
            }
        ),
        'buys': pd.DataFrame(
            {
                'buy_id': np.arange(120, dtype='int64'),
                'user_id': np.arange(120) % 8,
                'item_id': pd.Series(
                    [(i * 3) % 4 for i in range(120)], dtype='int64'
                ),
                'ts': pd.date_range('2023-01-01', periods=120, freq='5D'),
            }
        ),
    }
    data['users']['label'] = pd.Series(['a', 'b'] * 4, dtype=object)
    schema = Schema.from_data(
        data,
        primary_keys={'users': 'user_id', 'items': 'item_id', 'buys': 'buy_id'},
        time_columns={'users': None, 'items': None, 'buys': 'ts'},
    )
    schema['users']['label'].stype = 'categorical'
    question = PredictiveQuery(
        'PREDICT users.label FOR EACH users.user_id', schema
    )
    first = question.materialize(indices=[0, 1], random_seed=7)
    if not batched:
        return first
    # R5's actual setup performed this second public materialization.
    return PredictiveQuery(first.query, first.schema).materialize(
        indices=[0, 1],
        random_seed=7,
        rows_per_call=1,
    )


@pytest.mark.parametrize('core', [False, True], ids=['original', 'core'])
@pytest.mark.parametrize('case', list(REFERENCE))
def test_live_native_prediction_boundaries(monkeypatch, core, case):
    monkeypatch.setenv(
        'SDM_CORE_ROUTES', 'client.predict/regression/frames' if core else ''
    )
    expected = REFERENCE[case]
    round_, scenario = case.split('/')
    wire = []
    iterations = []
    metadata_items = []
    observed = {}

    class Rows(list):
        calls = 0

        def __iter__(self):
            self.calls += 1
            iterations.append(self.calls)
            if scenario == 'rows-second-call-fails' and self.calls == 2:
                raise ValueError('actual rows iteration 2 failed')
            if scenario == 'rows-fourth-call-fails' and self.calls == 4:
                raise ValueError('actual rows iteration 4 failed')
            if scenario == 'second-fourth-iteration-fails' and self.calls == 4:
                raise ValueError('second batch actual fourth traversal failed')
            return super().__iter__()

    class Metadata(dict):
        def items(self):
            metadata_items.append(True)
            return super().items()

    def post(url, *, data, **kwargs):
        sent = json.loads(data)
        sent['metadata']['request_id'] = '<native-call-id>'
        wire.append(sent)
        table = sent['predict']['instance_table']
        instance = table['columns'].index('instance_id')
        reply = {
            'id': 'native-boundary-test',
            'model': 'kumo-relational',
            'metadata': {},
            'predictions': [
                {'row_index': i, 'id': str(row[instance]), 'prediction': 'a'}
                for i, row in enumerate(table['rows'])
            ],
        }
        if scenario == 'mutate-response-entity':
            native = payload['predict']['instance_table']
            entity = next(
                i
                for i, name in enumerate(native['columns'])
                if name.startswith('__kumo_entity_ref')
            )
            native['rows'][0][entity] = 7
        if scenario == 'first-http-edits-second' and len(wire) == 1:
            context = second['context']['instance_table']
            target = context['columns'].index(
                second['task']['target']['column_name']
            )
            context['rows'][0][target] = 'edited-after-first-http'
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(reply).encode()
        response._content_consumed = True
        return response

    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            request = _request(batched=round_ == 'r5')
            payload = request.payloads[0]
            second = request.payloads[-1]
            context = payload['context']['instance_table']
            target = context['columns'].index(
                payload['task']['target']['column_name']
            )
            if scenario == 'builtin-edit':
                context['rows'][0][target] = 'edited-after-materialize'
            if scenario == 'surrogate-pair-distinct':
                for number, row in enumerate(context['rows']):
                    row[target] = '\ud83d\ude3a' if number % 2 else '\U0001f63a'
            if scenario.startswith('rows-'):
                context['rows'] = Rows(context['rows'])
            if scenario == 'second-fourth-iteration-fails':
                table = second['context']['instance_table']
                table['rows'] = Rows(table['rows'])
            if scenario == 'metadata-items':
                payload['metadata'] = Metadata({'native': 'metadata'})
            if scenario == 'empty-before-bad-json':
                context['rows'] = []
                payload['metadata'] = {'ignored-unserializable': object()}
            if scenario == 'bad-second-json':
                second['metadata'] = {'bad': object()}
            if core:

                def old_policy(*args, **kwargs):
                    raise AssertionError(
                        'native core prediction called old adapter policy'
                    )

                monkeypatch.setattr(
                    RelationalAdapter, 'prediction_plan', old_policy
                )
                monkeypatch.setattr(RelationalAdapter, 'normalize', old_policy)
            with SDMClient('http://localhost:1', max_retries=0) as client:
                client._transport._session.post = post
                result = client.predict(request, model=KumoRelational())
                observed['result'] = json.loads(
                    result.to_pandas().to_json(orient='split')
                )
            observed['outcome'] = 'success'
        except Exception as error:
            observed.update(outcome='error', result=None)
            reported = {
                'type': type(error).__name__,
                'message': str(error),
                'code': getattr(error, 'code', None),
            }
            if round_ == 'r5':
                details = getattr(error, 'details', {})
                paid = details.get('answered')
                reported.update(
                    cause=None
                    if error.__cause__ is None
                    else {
                        'type': type(error.__cause__).__name__,
                        'message': str(error.__cause__),
                    },
                    calls_answered=details.get('calls_answered'),
                    rows_answered=details.get('rows_answered'),
                    answered=None
                    if paid is None
                    else json.loads(paid.to_pandas().to_json(orient='split')),
                )
            observed['error'] = reported
        observed['warnings'] = [
            {'type': type(item.message).__name__, 'message': str(item.message)}
            for item in notices
        ]
    observed.update(
        wire=wire, iterations=iterations, metadata_items=len(metadata_items)
    )
    assert observed == expected


@pytest.mark.parametrize('core', [False, True], ids=['original', 'core'])
@pytest.mark.parametrize('case', list(TASK_REFERENCE))
def test_native_task_string_overrides_remain_at_reached_boundaries(
    monkeypatch, core, case
):
    monkeypatch.setenv(
        'SDM_CORE_ROUTES', 'client.predict/regression/frames' if core else ''
    )
    expected = copy.deepcopy(TASK_REFERENCE[case])
    events = []

    class Kind(str):
        contains_calls = 0

        def __str__(self):
            events.append('kind.str')
            return self

        def __contains__(self, value):
            self.contains_calls += 1
            events.append('kind.contains.' + str(self.contains_calls))
            if case == 'contains-second-fails' and self.contains_calls == 2:
                raise RuntimeError('actual second task containment failed')
            if (
                case
                in (
                    'third-contains-changes-role',
                    'last-contains-skips-class-cast',
                )
                and self.contains_calls == 3
            ):
                return False
            return str.__contains__(self, value)

        def __hash__(self):
            events.append('kind.hash')
            if case == 'hash-fails':
                raise RuntimeError('actual task membership hash failed')
            return str.__hash__(self)

        def __eq__(self, value):
            events.append('kind.eq')
            return str.__eq__(self, value)

        def __ne__(self, value):
            events.append('kind.ne')
            return str.__ne__(self, value)

    def post(url, *, data, **kwargs):
        events.append('http-post')
        sent = json.loads(data)
        table = sent['predict']['instance_table']
        column = table['columns'].index('instance_id')
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            {
                'id': 'kind-reference',
                'model': 'kumo-relational',
                'metadata': {},
                'predictions': [
                    {'row_index': i, 'id': str(row[column]), 'prediction': 'a'}
                    for i, row in enumerate(table['rows'])
                ],
            }
        ).encode()
        response._content_consumed = True
        return response

    observed = {'case': case}
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            request = _request()
            payload = request.payloads[0]
            payload['task']['kind'] = Kind(payload['task']['kind'])
            if case == 'third-contains-changes-role':
                context = payload['context']['instance_table']
                target = context['columns'].index(
                    payload['task']['target']['column_name']
                )
                for row in context['rows']:
                    row[target] = 'a'
            with SDMClient('http://localhost:1', max_retries=0) as client:
                client._transport._session.post = post
                result = client.predict(request, model=KumoRelational())
                observed['result'] = json.loads(
                    result.to_pandas().to_json(orient='split')
                )
            observed['outcome'] = 'success'
        except Exception as error:
            observed.update(
                outcome='error',
                error={
                    'type': type(error).__name__,
                    'message': str(error),
                    'code': getattr(error, 'code', None),
                },
            )
        observed['warnings'] = [
            {'type': type(item.message).__name__, 'message': str(item.message)}
            for item in notices
        ]
    observed['events'] = events
    if core and case == 'third-contains-changes-role':
        # Existing P3E-2: shared core warnings include their selected code.
        warning = expected['warnings'][1]
        warning['message'] = '[CONTEXT_CONSTANT] ' + warning['message']
    assert observed == expected


def _error_record(error, seen=None):
    seen = set() if seen is None else seen
    if id(error) in seen:
        return {'cause_cycle': True}
    seen.add(id(error))
    result = {
        'type': type(error).__name__,
        'message': str(error),
        'args': [str(value) for value in error.args],
        'code': getattr(error, 'code', None),
        'details': getattr(error, 'details', None),
    }
    if error.__cause__ is not None:
        result['cause'] = _error_record(error.__cause__, seen)
    return result


@pytest.mark.parametrize('core', [False, True], ids=['original', 'core'])
@pytest.mark.parametrize('case', list(FAULT_REFERENCE))
def test_native_part_fault_and_instance_comparison_parity(
    monkeypatch, core, case
):
    from test_t44_decomposition_owners import question

    monkeypatch.setenv(
        'SDM_CORE_ROUTES', 'client.predict/regression/frames' if core else ''
    )
    expected = FAULT_REFERENCE[case]
    events = []
    observed = {'case': case}

    class Instance(str):
        def __str__(self):
            events.append('instance.str')
            return self

        def __iter__(self):
            events.append('instance.iter')
            raise RuntimeError('native instance string must not be iterated')

        def __ne__(self, other):
            events.append('instance.ne')
            if case == 'instance-native-inequality':
                raise RuntimeError('actual instance inequality failed')
            return str.__ne__(self, other)

    def post(url, *, data, **kwargs):
        events.append('http-post')
        sent = json.loads(data)
        table = sent['predict']['instance_table']
        at = table['columns'].index('instance_id')
        reply = {
            'id': 'fault-reference',
            'model': 'kumo-relational',
            'metadata': {},
            'predictions': [
                {'row_index': i, 'id': str(row[at]), 'prediction': 'a'}
                for i, row in enumerate(table['rows'])
            ],
        }
        response = requests.Response()
        response.status_code = 200
        if case == 'forecast-window-refusal':
            response.status_code = 422
            reply = {
                'title': 'context window ended',
                'code': 'INVALID_REQUEST',
                'window_outran_the_data': True,
            }
        if case == 'huge-integer-probability':
            reply['predictions'][0]['probabilities'] = {'true': 10**400}
        response._content = json.dumps(reply).encode()
        response._content_consumed = True
        return response

    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            if case.startswith(('forecast-', 'ranking-')):
                ranking = case.startswith('ranking-')
                asked = question('ranking' if ranking else 'forecast')
                how = {
                    'indices': [1, 2],
                    'num_context_samples': 3,
                    'num_neighbors': [3],
                    'num_hops': 1,
                    'random_seed': 7,
                }
                if ranking:
                    how['candidates'] = pd.Series([0, 1])
                request = asked.materialize(**how)
            else:
                request = _request()
                if case.startswith('instance-'):
                    table = request.payloads[0]['predict']['instance_table']
                    at = table['columns'].index('instance_id')
                    table['rows'][0][at] = Instance(str(table['rows'][0][at]))
            with SDMClient('http://localhost:1', max_retries=0) as client:
                client._transport._session.post = post
                model = (
                    'does-not-exist'
                    if case.endswith('unknown-model')
                    else KumoRelational()
                )
                result = client.predict(request, model=model)
                observed['result'] = json.loads(
                    result.to_pandas().to_json(orient='split')
                )
            observed['outcome'] = 'success'
        except Exception as error:
            observed.update(outcome='error', error=_error_record(error))
        observed['warnings'] = [
            {'type': type(item.message).__name__, 'message': str(item.message)}
            for item in notices
        ]
    observed['events'] = events
    assert observed == expected
