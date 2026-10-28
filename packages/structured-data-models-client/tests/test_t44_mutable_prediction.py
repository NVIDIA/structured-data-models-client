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
import dataclasses
import json
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
from t38_forced_routes_record import bare
from t45_frame_routes import FRAME_PREDICT_ROUTES

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
NATIVE_KIND_REFERENCE = json.loads(
    Path(__file__)
    .with_name('t44_mutable_prediction_reference.json')
    .read_text()
)['native_kind_cases'][
    '310' if int(pd.__version__.split('.')[0]) == 2 else '312'
]


@pytest.fixture(autouse=True)
def _core_preparation_owner_guard(monkeypatch, request):
    if getattr(
        request.node, 'callspec', None
    ) is not None and request.node.callspec.params.get('core'):

        def legacy(*args, **kwargs):
            raise AssertionError(
                'the core prediction route called legacy SDMClient._prepare'
            )

        monkeypatch.setattr(SDMClient, '_prepare', legacy)


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
    monkeypatch.setenv('SDM_CORE_ROUTES', FRAME_PREDICT_ROUTES if core else '')
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
    monkeypatch.setenv('SDM_CORE_ROUTES', FRAME_PREDICT_ROUTES if core else '')
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


def _fault_as_recorded(observed, *, core):
    if not core or observed['case'] != 'huge-integer-probability':
        return observed
    # P3E-5 retains the gathered decoder's branchable field/reason policy.
    # Assert that exact addition before comparing every native observation
    # with the unchanged original recording, whose details are empty.
    assert observed['error']['details'] == {
        'field': 'prediction.probabilities',
        'reason': 'not_a_number',
    }
    projected = copy.deepcopy(observed)
    projected['error']['details'] = {}
    return projected


@pytest.mark.parametrize(
    'details',
    [
        {},
        {'field': 'prediction.probabilities'},
        {'field': 'prediction.row_index', 'reason': 'not_a_number'},
        {'field': 'prediction.probabilities', 'reason': 'not_finite'},
        {
            'field': 'prediction.probabilities',
            'reason': 'not_a_number',
            'diagnostic': 'private native diagnostic',
        },
    ],
)
def test_huge_probability_comparison_requires_exact_gathered_details(details):
    observed = copy.deepcopy(FAULT_REFERENCE['huge-integer-probability'])
    observed['error']['details'] = details
    with pytest.raises(AssertionError):
        _fault_as_recorded(observed, core=True)


@pytest.mark.parametrize('changed', ['cause', 'events', 'warnings'])
def test_huge_probability_comparison_preserves_native_observation_record(
    changed,
):
    expected = FAULT_REFERENCE['huge-integer-probability']
    assert expected['error']['details'] == {}
    observed = copy.deepcopy(expected)
    observed['error']['details'] = {
        'field': 'prediction.probabilities',
        'reason': 'not_a_number',
    }
    before = copy.deepcopy(observed)
    assert _fault_as_recorded(observed, core=True) == expected
    assert observed == before
    if changed == 'cause':
        observed['error']['cause']['message'] = 'changed native cause'
    elif changed == 'events':
        observed['events'].append('unexpected native observation')
    else:
        observed['warnings'][0]['message'] = 'changed native warning'
    assert _fault_as_recorded(observed, core=True) != expected


@pytest.mark.parametrize(
    ('core', 'case'),
    [(False, 'huge-integer-probability'), (True, 'forecast-window-refusal')],
)
def test_gathered_details_comparison_is_limited_to_its_core_case(core, case):
    observed = copy.deepcopy(FAULT_REFERENCE[case])
    assert _fault_as_recorded(observed, core=core) is observed


@pytest.mark.parametrize('core', [False, True], ids=['original', 'core'])
@pytest.mark.parametrize('case', list(FAULT_REFERENCE))
def test_native_part_fault_and_instance_comparison_parity(
    monkeypatch, core, case
):
    from test_t44_decomposition_owners import question

    monkeypatch.setenv('SDM_CORE_ROUTES', FRAME_PREDICT_ROUTES if core else '')
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
    assert _fault_as_recorded(observed, core=core) == expected


@pytest.mark.parametrize('core', [False, True], ids=['original', 'core'])
@pytest.mark.parametrize('case', list(NATIVE_KIND_REFERENCE))
def test_public_task_kind_observations_preserve_preparation_order(
    monkeypatch, core, case
):
    monkeypatch.setenv('SDM_CORE_ROUTES', FRAME_PREDICT_ROUTES if core else '')
    expected = copy.deepcopy(NATIVE_KIND_REFERENCE[case])
    events = []
    hashes = 0
    equalities = 0
    retained = RuntimeError('retained native task-kind failure')
    observed = {'case': case}
    request = None

    class ObservedKind(str):
        def __hash__(self):
            nonlocal hashes
            hashes += 1
            events.append({'event': 'kind.hash', 'ordinal': hashes})
            if (
                case
                in (
                    'raise-default-hash',
                    'explicit-raise-capability-hash',
                    'anchored-explicit-raise-admission-hash',
                )
                and hashes == 1
            ) or (case == 'raise-admission-hash' and hashes == 3):
                raise retained
            if case == 'mutate-default-task' and hashes == 1:
                request.task = dataclasses.replace(
                    request.task, kind='regression'
                )
                events.append({'event': 'task-replaced', 'kind': 'regression'})
            if case == 'mutate-admission-payload' and hashes == 3:
                request.payloads[0]['metadata']['kind_observation'] = (
                    'admission'
                )
                events.append({'event': 'payload-mutated'})
            return str.__hash__(self)

        def __eq__(self, other):
            nonlocal equalities
            equalities += 1
            events.append(
                {'event': 'kind.eq', 'other': str(other), 'ordinal': equalities}
            )
            if (
                case == 'raise-capability-equality'
                and other == 'temporal_link_prediction'
            ):
                raise retained
            if (
                case == 'mutate-capability-task'
                and other == 'temporal_link_prediction'
            ):
                request.task = dataclasses.replace(
                    request.task, kind='regression'
                )
                events.append({'event': 'task-replaced', 'kind': 'regression'})
            return str.__eq__(self, other)

    def post(url, *, data, **kwargs):
        events.append({'event': 'http-post'})
        sent = json.loads(data)
        observed['wire'] = sent
        table = sent['predict']['instance_table']
        at = table['columns'].index('instance_id')
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            {
                'id': 'kind-reference',
                'model': 'kumo-relational',
                'metadata': {},
                'predictions': [
                    {'row_index': i, 'id': str(row[at]), 'prediction': 'a'}
                    for i, row in enumerate(table['rows'])
                ],
            }
        ).encode()
        response._content_consumed = True
        return response

    if 'wire' in expected:
        monkeypatch.setattr(
            'sdm_client.adapters.relational._request_id',
            lambda: expected['wire']['metadata']['request_id'],
        )
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            request = _request()
            request.task = dataclasses.replace(
                request.task, kind=ObservedKind(request.task.kind)
            )
            if case == 'anchored-explicit-raise-admission-hash':
                request.task.options['anchor_time_column'] = '__native_anchor'
            outputs = ('prediction',) if 'explicit-' in case else None
            with SDMClient('http://localhost:1', max_retries=0) as client:
                client._transport._session.post = post
                answer = client.predict(
                    request, model=KumoRelational(), outputs=outputs
                )
                observed['result'] = json.loads(
                    answer.to_pandas().to_json(orient='split')
                )
            observed['outcome'] = 'success'
        except Exception as error:
            observed.update(
                outcome='error',
                error={
                    'type': type(error).__name__,
                    'message': str(error),
                    'code': getattr(error, 'code', None),
                    'retained_identity': error is retained,
                },
            )
        observed['warnings'] = [
            {'type': type(item.message).__name__, 'message': str(item.message)}
            for item in notices
        ]
    observed['events'] = events
    assert observed == expected


@pytest.mark.parametrize('core', [False, True], ids=['original', 'core'])
@pytest.mark.parametrize(
    'case',
    ['refused-kind', 'refused-kind-repr-raises', 'refused-kind-repr-mutates'],
)
def test_native_task_refusal_retains_repr_and_fresh_details(
    monkeypatch, core, case
):
    monkeypatch.setenv('SDM_CORE_ROUTES', FRAME_PREDICT_ROUTES if core else '')
    cell = '310' if int(pd.__version__.split('.')[0]) == 2 else '312'
    expected = json.loads(
        Path(__file__)
        .with_name('t44_mutable_prediction_reference.json')
        .read_text()
    )['native_kind_refusal_cases'][cell][case]
    events = []
    retained = RuntimeError('retained native task-kind repr failure')
    request = None
    actual = None
    observed = {'case': case}

    class Kind(str):
        def __hash__(self):
            events.append('kind.hash')
            return str.__hash__(self)

        def __eq__(self, other):
            events.append('kind.eq.' + str(other))
            return str.__eq__(self, other)

        def __repr__(self):
            events.append('kind.repr')
            if case == 'refused-kind-repr-raises':
                raise retained
            if case == 'refused-kind-repr-mutates':
                request.task = dataclasses.replace(
                    request.task, kind='regression'
                )
                events.append('task-replaced.regression')
            return '<native task kind>'

    def post(*args, **kwargs):
        raise AssertionError('a refused task reached HTTP delivery')

    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            request = _request()
            actual = Kind('unsupported_kind')
            request.task = dataclasses.replace(request.task, kind=actual)
            with SDMClient('http://localhost:1', max_retries=0) as client:
                client._transport._session.post = post
                client.predict(request, model=KumoRelational())
            observed['outcome'] = 'success'
        except Exception as error:
            details = getattr(error, 'details', None)
            identity = (
                isinstance(details, dict) and details.get('task') is actual
            )
            observed.update(
                outcome='error',
                error={
                    'type': type(error).__name__,
                    'message': str(error),
                    'code': getattr(error, 'code', None),
                    'retained_identity': error is retained,
                    'details': None
                    if details is None
                    else json.loads(json.dumps(details)),
                    'details_task_identity': identity,
                },
            )
        observed['warnings'] = [
            {'type': type(item.message).__name__, 'message': str(item.message)}
            for item in notices
        ]
    observed['events'] = events
    assert observed == expected


MODEL_OUTPUT_REFERENCE = json.loads(
    Path(__file__)
    .with_name('t44_mutable_prediction_reference.json')
    .read_text()
)['model_output_cases'][
    '310' if int(pd.__version__.split('.')[0]) == 2 else '312'
]


@pytest.mark.parametrize('seed', [0, 1, 8, 12])
def test_native_model_refusal_keeps_complete_journeys_at_each_hash_seed(seed):
    original, core = _model_output_in_fresh_process(seed, [False, True])
    assert core == original


def _model_output_in_fresh_process(seed, cores):
    """Retain native dictionary probes under an explicit child-only seed."""
    script = textwrap.dedent(
        """\
        import json
        import runpy
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(sys.argv[1]).parent))
        scope = runpy.run_path(sys.argv[1])
        observed = []
        for core in json.loads(sys.argv[2]):
            with scope['pytest'].MonkeyPatch.context() as monkeypatch:
                if core:
                    def legacy(*args, **kwargs):
                        raise AssertionError(
                            'the core prediction route called legacy '
                            'SDMClient._prepare'
                        )
                    monkeypatch.setattr(
                        scope['SDMClient'], '_prepare', legacy
                    )
                observed.append(
                    scope['_observe_native_model_and_output_owners'](
                        monkeypatch, core, 'model-equality-refuses'
                    )
                )
        print(json.dumps(observed))
        """
    )
    done = subprocess.run(
        [
            sys.executable,
            '-c',
            script,
            str(Path(__file__).resolve()),
            json.dumps(cores),
        ],
        env={**os.environ, 'PYTHONHASHSEED': str(seed)},
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return json.loads(done.stdout)


@pytest.mark.parametrize('core', [False, True], ids=['original', 'core'])
@pytest.mark.parametrize('case', list(MODEL_OUTPUT_REFERENCE))
def test_native_model_and_output_owners_keep_complete_public_journeys(
    monkeypatch, core, case
):
    expected = copy.deepcopy(MODEL_OUTPUT_REFERENCE[case])
    # The frozen native dict probe has one equality visit. A process seed
    # controls that layout without changing the Name class or its callbacks.
    if case == 'model-equality-refuses':
        record = _model_output_in_fresh_process(0, [core])[0]
    else:
        record = _observe_native_model_and_output_owners(
            monkeypatch, core, case
        )
    assert record == expected


def _observe_native_model_and_output_owners(monkeypatch, core, case):
    from sdm_client.models import Model

    monkeypatch.setenv('SDM_CORE_ROUTES', FRAME_PREDICT_ROUTES if core else '')
    expected = copy.deepcopy(MODEL_OUTPUT_REFERENCE[case])
    events = []
    retained = RuntimeError('retained native model/output failure')
    record = {'case': case}
    request = None

    class Name(str):
        def __new__(cls, value, owner):
            instance = str.__new__(cls, value)
            instance.owner = owner
            return instance

        def __hash__(self):
            events.append(self.owner + '.hash')
            if (
                case in ('model-hash-raises', 'handle-model-hash-raises')
                and self.owner == 'model'
            ) or (case == 'output-hash-raises' and self.owner == 'output'):
                raise retained
            if case == 'model-hash-mutates' and self.owner == 'model':
                request.task = dataclasses.replace(
                    request.task, kind='regression'
                )
                events.append('task-replaced.regression')
            if case == 'output-hash-mutates' and self.owner == 'output':
                request.payloads[0]['metadata']['native_output'] = 'membership'
                events.append('payload-mutated')
            return str.__hash__(self)

        def __eq__(self, other):
            events.append(self.owner + '.eq.' + str(other))
            if case == 'model-equality-refuses' and self.owner == 'model':
                return False
            if case == 'output-equality-mutates' and self.owner == 'output':
                request.task = dataclasses.replace(
                    request.task, kind='regression'
                )
                events.append('task-replaced.regression')
            return str.__eq__(self, other)

        def __lt__(self, other):
            events.append(self.owner + '.lt.' + str(other))
            return str.__lt__(self, other)

        def __repr__(self):
            events.append(self.owner + '.repr.' + str.__str__(self))
            if case in (
                'unknown-model-repr-raises',
                'output-refusal-repr-raises',
            ):
                raise retained
            return '<native ' + self.owner + ' ' + str.__str__(self) + '>'

    class Kind(str):
        hashes = 0

        def __hash__(self):
            self.hashes += 1
            events.append('task.hash.' + str(self.hashes))
            if case == 'task-admission-before-output' and self.hashes == 2:
                raise retained
            return str.__hash__(self)

        def __eq__(self, other):
            events.append('task.eq.' + str(other))
            return str.__eq__(self, other)

    class Names(list):
        def __iter__(self):
            events.append('outputs.iter')
            return list.__iter__(self)

    def post(url, *, data, **kwargs):
        events.append('http-post')
        record['wire'] = json.loads(data)
        table = record['wire']['predict']['instance_table']
        at = table['columns'].index('instance_id')
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            {
                'id': 'model-output-reference',
                'model': 'kumo-relational',
                'metadata': {},
                'predictions': [
                    {'row_index': i, 'id': str(row[at]), 'prediction': 'a'}
                    for i, row in enumerate(table['rows'])
                ],
            }
        ).encode()
        response._content_consumed = True
        return response

    if 'wire' in expected:
        monkeypatch.setattr(
            'sdm_client.adapters.relational._request_id',
            lambda: expected['wire']['metadata']['request_id'],
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            request = _request()
            request.task = dataclasses.replace(
                request.task, kind=Kind(request.task.kind)
            )
            model = Name('kumo-relational', 'model')
            if case == 'unknown-model-repr-raises':
                model = Name('unknown-model', 'model')
            if case == 'handle-model-hash-raises':
                model = Model()
                model._spec = dataclasses.replace(
                    KumoRelational()._spec,
                    model_id=Name('kumo-relational', 'model'),
                )
            outputs = (Name('prediction', 'output'),)
            if case.startswith('output-refusal'):
                outputs = (Name('zzz', 'output'), Name('yyy', 'output'))
            if case == 'output-sequence-order':
                outputs = Names(outputs)
            with SDMClient('http://localhost:1', max_retries=0) as client:
                client._transport._session.post = post
                answer = client.predict(request, model=model, outputs=outputs)
                record['result'] = json.loads(
                    answer.to_pandas().to_json(orient='split')
                )
            record['outcome'] = 'success'
        except Exception as error:
            details = getattr(error, 'details', None)
            unsupported = (
                details.get('unsupported_outputs')
                if isinstance(details, dict)
                else None
            )
            record.update(
                outcome='error',
                error={
                    'type': type(error).__name__,
                    'message': str(error),
                    'code': getattr(error, 'code', None),
                    'retained_identity': error is retained,
                    'details': None
                    if details is None
                    else json.loads(json.dumps(details)),
                    'unknown_model_identity': getattr(error, 'model', None)
                    is model,
                    'unsupported_output_identities': None
                    if unsupported is None
                    else [
                        any(value is original for original in outputs)
                        for value in unsupported
                    ],
                },
            )
        record['warnings'] = [
            {'type': type(item.message).__name__, 'message': str(item.message)}
            for item in caught
        ]
    record['events'] = events
    return record


MODEL_OUTPUT_SHAPE_REFERENCE = json.loads(
    Path(__file__)
    .with_name('t44_mutable_prediction_reference.json')
    .read_text()
)['model_output_shape_cases'][
    '310' if int(pd.__version__.split('.')[0]) == 2 else '312'
]


@pytest.mark.parametrize('core', [False, True], ids=['original', 'core'])
@pytest.mark.parametrize('case', list(MODEL_OUTPUT_SHAPE_REFERENCE))
def test_native_two_paid_call_output_shape_hash_boundary(
    monkeypatch, core, case
):
    monkeypatch.setenv('SDM_CORE_ROUTES', FRAME_PREDICT_ROUTES if core else '')
    expected = copy.deepcopy(MODEL_OUTPUT_SHAPE_REFERENCE[case])
    events = []
    sent = []
    retained = RuntimeError('retained native output-shape hash failure')
    record = {'case': case}
    hashes = 0

    class Output(str):
        def __hash__(self):
            nonlocal hashes
            hashes += 1
            events.append('output.hash.' + str(hashes))
            if hashes == 2 and case == 'shape-hash-raises':
                raise retained
            return str.__hash__(self)

        def __eq__(self, other):
            events.append('output.eq.' + str(other))
            return str.__eq__(self, other)

    def post(url, *, data, **kwargs):
        events.append('http-post.' + str(len(sent) + 1))
        body = json.loads(data)
        sent.append(body)
        table = body['predict']['instance_table']
        at = table['columns'].index('instance_id')
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            {
                'id': 'shape-reference-' + str(len(sent)),
                'model': 'kumo-relational',
                'metadata': {},
                'predictions': [
                    {'row_index': i, 'id': str(row[at]), 'prediction': 'a'}
                    for i, row in enumerate(table['rows'])
                ],
            }
        ).encode()
        response._content_consumed = True
        return response

    identifiers = iter(
        body['metadata']['request_id'] for body in expected['sent']
    )
    monkeypatch.setattr(
        'sdm_client.adapters.relational._request_id', lambda: next(identifiers)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            first = _request()
            request = PredictiveQuery(first.query, first.schema).materialize(
                indices=[0, 1], random_seed=7, rows_per_call=1
            )
            with SDMClient('http://localhost:1', max_retries=0) as client:
                client._transport._session.post = post
                answer = client.predict(
                    request,
                    model=KumoRelational(),
                    outputs=(Output('prediction'),),
                )
                record['result'] = json.loads(
                    answer.to_pandas().to_json(orient='split')
                )
            record['outcome'] = 'success'
        except Exception as error:
            record.update(
                outcome='error',
                error={
                    'type': type(error).__name__,
                    'message': str(error),
                    'code': getattr(error, 'code', None),
                    'retained_identity': error is retained,
                    'details': getattr(error, 'details', None),
                },
            )
        record['warnings'] = [
            {'type': type(item.message).__name__, 'message': str(item.message)}
            for item in caught
        ]
    record.update(events=events, sent=sent, paid_calls=len(sent))
    assert record == expected


DECOMPOSED_OUTPUT_REFERENCE = json.loads(
    Path(__file__)
    .with_name('t44_mutable_prediction_reference.json')
    .read_text()
)['decomposed_output_cases'][
    '310' if int(pd.__version__.split('.')[0]) == 2 else '312'
]


@pytest.mark.parametrize('core', [False, True], ids=['original', 'core'])
@pytest.mark.parametrize('case', list(DECOMPOSED_OUTPUT_REFERENCE))
def test_native_decomposed_output_owners_and_paid_boundaries(
    monkeypatch, core, case
):
    from test_t44_decomposition_owners import question as question_for

    from sdm_client.result import PredictionResult

    monkeypatch.setenv('SDM_CORE_ROUTES', FRAME_PREDICT_ROUTES if core else '')
    expected = copy.deepcopy(DECOMPOSED_OUTPUT_REFERENCE[case])
    if expected['sent']:
        if case.startswith('ranking-'):
            monkeypatch.setattr(
                'sdm_client.adapters.relational._request_id',
                lambda: expected['sent'][0]['metadata']['request_id'],
            )
        else:
            identifiers = iter(
                body['metadata']['request_id'] for body in expected['sent']
            )
            monkeypatch.setattr(
                'sdm_client.adapters.relational._request_id',
                lambda: next(identifiers),
            )
    events = []
    sent = []
    armed = False
    retained = RuntimeError('retained decomposed output observation failure')
    paid_failure = RuntimeError('retained second child delivery failure')
    record = {'case': case}

    class ModelName(str):
        def __hash__(self):
            events.append('model.hash')
            return str.__hash__(self)

        def __eq__(self, other):
            events.append('model.eq.' + str(other))
            return str.__eq__(self, other)

    class Output(str):
        def __hash__(self):
            events.append('output.hash')
            if armed:
                raise retained
            return str.__hash__(self)

        def __eq__(self, other):
            events.append('output.eq.' + str(other))
            if case == 'ranking-output-equality-fails':
                raise retained
            return str.__eq__(self, other)

    class Outputs(list):
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            events.append('outputs.iter.' + str(self.iterations))
            if (
                case == 'forecast-sequence-fourth-fails'
                and self.iterations == 4
            ):
                raise retained
            return list.__iter__(self)

        def __contains__(self, value):
            events.append('outputs.contains.' + str(value))
            if case == 'forecast-explanation-contains-fails':
                raise retained
            return list.__contains__(self, value)

    def post(url, *, data, **kwargs):
        events.append('http-post.' + str(len(sent) + 1))
        body = json.loads(data)
        sent.append(body)
        if case == 'forecast-paid-prefix-owner' and len(sent) == 2:
            raise paid_failure
        table = body['predict']['instance_table']
        at = table['columns'].index('instance_id')
        ranking = case.startswith('ranking-')
        predictions = []
        for i, row in enumerate(table['rows']):
            item = {
                'row_index': i,
                'id': str(row[at]),
                'prediction': True if ranking else 17.0 + i,
            }
            if ranking:
                item['probabilities'] = {'false': 0.3, 'true': 0.7}
            predictions.append(item)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            {
                'id': 'decomposed-owner-' + str(len(sent)),
                'model': 'kumo-relational',
                'metadata': {},
                'predictions': predictions,
            }
        ).encode()
        response._content_consumed = True
        return response

    def inspect_result(answer):
        return {
            'result': json.loads(answer.to_pandas().to_json(orient='split')),
            'tuple_identity': answer.outputs is outputs,
            'member_identities': [
                value is outputs[i] for i, value in enumerate(answer.outputs)
            ],
        }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            ranking = case.startswith('ranking-')
            question = question_for('ranking' if ranking else 'forecast')
            how = {
                'indices': [1, 2],
                'num_context_samples': 3,
                'num_neighbors': [3],
                'num_hops': 1,
                'random_seed': 7,
            }
            if ranking:
                how['candidates'] = pd.Series([0, 1])
            request = question.materialize(**how)
            owner = Output('rankings' if ranking else 'prediction')
            outputs = (
                Outputs(['prediction'])
                if '-sequence-' in case or '-explanation-' in case
                else (owner,)
            )
            with SDMClient('http://localhost:1', max_retries=0) as client:
                client._transport._session.post = post
                answer = client.predict(
                    request, model=ModelName('kumo-relational'), outputs=outputs
                )
            record.update(outcome='success', answer=inspect_result(answer))
            if '-owner-' in case:
                armed = True
                try:
                    PredictionResult.merge([answer, answer])
                    record['later_merge'] = {'outcome': 'success'}
                except Exception as error:
                    record['later_merge'] = {
                        'outcome': 'error',
                        'type': type(error).__name__,
                        'message': str(error),
                        'retained_identity': error is retained,
                    }
                armed = False
        except Exception as error:
            details = getattr(error, 'details', None)
            answered = (
                details.get('answered') if isinstance(details, dict) else None
            )
            record.update(
                outcome='error',
                error={
                    'type': type(error).__name__,
                    'message': str(error),
                    'code': getattr(error, 'code', None),
                    'retained_identity': error is retained,
                    'paid_failure_identity': error is paid_failure,
                    'parts_answered': details.get('parts_answered')
                    if isinstance(details, dict)
                    else None,
                    'answered': None
                    if answered is None
                    else inspect_result(answered),
                },
            )
            if answered is not None:
                armed = True
                try:
                    PredictionResult.merge([answered, answered])
                    record['later_merge'] = {'outcome': 'success'}
                except Exception as later:
                    record['later_merge'] = {
                        'outcome': 'error',
                        'type': type(later).__name__,
                        'message': str(later),
                        'retained_identity': later is retained,
                    }
                armed = False
        record['warnings'] = [
            {'type': type(item.message).__name__, 'message': str(item.message)}
            for item in caught
        ]
    record.update(events=events, sent=sent, paid_calls=len(sent))
    if core:
        for warning in record['warnings']:
            warning['message'] = bare(warning['message'])
    assert record == expected
