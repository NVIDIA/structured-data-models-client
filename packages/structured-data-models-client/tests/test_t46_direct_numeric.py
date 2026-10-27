# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen43e direct Request -> tabular public journeys and read boundaries.

Reference: request.py:372-488; client.py:421-479,767-807;
base.py:218-310; adapters/tabular.py:171-333; wire/dtypes.py.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_t42_frames import projection

from sdm_client import KumoTabular, Request, SDMClient
from sdm_client._routing import RouteError

CORE = ','.join(
    (
        'request.tabular/classification/frames',
        'request.tabular/regression/frames',
        'client.predict/tabular/frames',
    )
)


class Scripted:
    url = 'scripted://direct-numeric'
    serves_sessions = True

    def __init__(self):
        self.calls = []

    def predict(self, payload):
        self.calls.append(copy.deepcopy(payload))
        return {
            'id': 'direct-fixture',
            'model': 'kumo-tabular',
            'metadata': {'source': 'fixture'},
            'predictions': [
                {'row_index': row, 'id': str(row), 'prediction': 10 + row}
                for row in range(
                    len(payload['predict']['instance_table']['rows'])
                )
            ],
        }

    def close(self):
        pass


def fixture(dtype='float64'):
    context = pd.DataFrame(
        {
            'feature': pd.Series([1, 2, 3], dtype=dtype),
            'target': pd.Series([2, 4, 6], dtype=dtype),
        }
    )
    predict = pd.DataFrame({'feature': pd.Series([4, 5], dtype=dtype)})
    context.index = [7, 2, 7]
    predict.index = [9, 1]
    return context, predict


def outcome(
    monkeypatch, frames, *, core, options=None, send=True, outputs=None
):
    monkeypatch.setenv('SDM_CORE_ROUTES', CORE if core else '')
    scripted = Scripted()
    inspected = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            request = Request(
                *frames,
                **(
                    {'target': 'target', 'task': 'regression'} | (options or {})
                ),
            )
            assert request.context.instance_table is frames[0]
            assert request.predict.instance_table is frames[1]
            assert request.payloads is None
            assert request._core is None
            inspected = (
                request.task,
                projection(request.schema),
                request.inference,
            )
            result = None
            if send:
                with SDMClient('http://localhost:1', max_retries=0) as client:
                    client._transport.close()
                    client._transport = scripted
                    result = client.predict(
                        request, model=KumoTabular(), outputs=outputs
                    )
                    result = (
                        result.to_pandas().to_dict(orient='list'),
                        result.outputs,
                        result.metadata,
                    )
            answer = ('ok', result)
        except Exception as error:
            answer = (
                'error',
                type(error),
                getattr(error, 'code', None),
                str(error),
            )
    for body in scripted.calls:
        body['metadata']['request_id'] = '<drawn>'
    return (
        inspected,
        answer,
        [(str(n.message), n.category) for n in caught],
        scripted.calls,
    )


@pytest.mark.parametrize(
    'dtype',
    ['int8', 'int64', 'uint64', 'float32', 'float64', 'Int64', 'Float64'],
)
@pytest.mark.parametrize('task', ['regression', 'classification'])
def test_direct_numeric_constructor_inspection_body_and_result(
    monkeypatch, dtype, task
):
    frames = fixture(dtype)
    old = outcome(monkeypatch, frames, core=False, options={'task': task})
    new = outcome(monkeypatch, frames, core=True, options={'task': task})
    assert new == old
    assert new[1][0] == 'ok'
    body = new[3][0]
    assert body['task']['kind'] == task
    assert body['context']['instance_table']['columns'] == ['feature', 'target']
    assert body['predict']['instance_table']['rows'] == [[4], [5]]
    assert 'inference' not in body
    assert 'prediction_statistic' not in body['output']


@pytest.mark.parametrize(
    'value',
    [
        None,
        True,
        1,
        1.0,
        np.nan,
        [],
        {},
        set(),
        ('target',),
        'missing',
        '\ud800',
    ],
)
def test_direct_ordinary_target_lookup_and_failure_order(monkeypatch, value):
    frames = fixture()
    assert outcome(
        monkeypatch, frames, core=True, options={'target': value}, send=False
    ) == outcome(
        monkeypatch, frames, core=False, options={'target': value}, send=False
    )


@pytest.mark.parametrize('target', [[], {}, '\ud800'])
def test_direct_unknown_task_precedes_label_target_and_buffer_reads(
    monkeypatch, target
):
    from sdm_client._rust import frames as conversion

    frames = fixture()
    frames[0].columns = pd.Index(['same', 'same'], dtype=object)
    old = outcome(
        monkeypatch,
        frames,
        core=False,
        options={'task': 'unknown', 'target': target},
        send=False,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError('early constructor guard extracted Arrow data')

    monkeypatch.setattr(conversion, '_primitive_array', forbidden)
    new = outcome(
        monkeypatch,
        frames,
        core=True,
        options={'task': 'unknown', 'target': target},
        send=False,
    )
    assert new == old
    assert 'task must be one of' in new[1][-1]


@pytest.mark.parametrize(
    'label,target', [(1, 1), (True, 1), (None, None), (np.nan, np.nan)]
)
def test_direct_native_header_membership_precedes_nonstring_schema_guard(
    monkeypatch, label, target
):
    context, predict = fixture()
    context.columns = pd.Index([label, 'target'], dtype=object)
    predict.columns = pd.Index([label], dtype=object)
    options = {'target': target}
    old = outcome(
        monkeypatch, (context, predict), core=False, options=options, send=False
    )
    assert (
        outcome(
            monkeypatch,
            (context, predict),
            core=True,
            options=options,
            send=False,
        )
        == old
    )
    assert 'predict carries the target column' in old[1][-1]


@pytest.mark.parametrize(
    'variant',
    [
        'empty',
        'known_before_extra',
        'scored_target',
        'extra',
        'missing',
        'duplicate',
        'nonstring',
    ],
)
def test_direct_constructor_semantic_guards_match_source_order(
    monkeypatch, variant
):
    context, predict = fixture()
    if variant == 'empty':
        context = context.iloc[:0]
    elif variant == 'known_before_extra':
        context['target'] = np.nan
        predict['extra'] = 1
    elif variant == 'scored_target':
        predict['target'] = 1
    elif variant == 'extra':
        predict['extra'] = 1
    elif variant == 'missing':
        predict = predict.drop(columns='feature')
    elif variant == 'duplicate':
        context.columns = ['target', 'target']
        predict = predict.iloc[:, :0]
    else:
        context.columns = pd.Index([1, 'target'], dtype=object)
        predict.columns = pd.Index([1], dtype=object)
    frames = context, predict
    assert outcome(monkeypatch, frames, core=True, send=False) == outcome(
        monkeypatch, frames, core=False, send=False
    )


@pytest.mark.parametrize(
    'task',
    [
        'binary_classification',
        'multiclass_classification',
        'forecasting',
        'ranking',
        'temporal_link_prediction',
    ],
)
def test_direct_preserves_explicit_task_until_later_model_refusal(
    monkeypatch, task
):
    frames = fixture()
    old = outcome(monkeypatch, frames, core=False, options={'task': task})
    new = outcome(monkeypatch, frames, core=True, options={'task': task})
    assert new == old
    assert new[0][0].kind == task
    assert new[1][0] == 'error'


@pytest.mark.parametrize('count', [10000, 10001])
def test_direct_capacity_precedes_empty_scoring(monkeypatch, count):
    frames = (
        pd.DataFrame({'feature': np.arange(count), 'target': np.arange(count)}),
        pd.DataFrame({'feature': pd.Series([], dtype='int64')}),
    )
    old = outcome(monkeypatch, frames, core=False)
    assert outcome(monkeypatch, frames, core=True) == old
    assert not old[3]
    assert old[1][0] == ('ok' if count == 10000 else 'error')


@pytest.mark.parametrize('wide', [False, True])
def test_direct_large_integer_encoding_and_declared_widening(monkeypatch, wide):
    context, predict = fixture('uint64' if wide else 'int64')
    base = 2**63 if wide else 2**53 + 1
    context['feature'] = pd.Series(
        [base, base + 1, base + 2],
        index=context.index,
        dtype='uint64' if wide else 'int64',
    )
    frames = context, predict
    old = outcome(monkeypatch, frames, core=False)
    assert outcome(monkeypatch, frames, core=True) == old
    assert old[3][0]['context']['instance_table']['rows'][0][0] == str(base)


def test_direct_fresh_frames_task_options_and_schema_on_each_call(monkeypatch):
    records = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', CORE if core else '')
        context, predict = fixture()
        request = Request(context, predict, target='target', task='regression')
        scripted = Scripted()
        with SDMClient('http://localhost:1', max_retries=0) as client:
            client._transport.close()
            client._transport = scripted
            client.predict(request, model='kumo-tabular')
            predict['feature'] = [20.0, 30.0]
            request.task = replace(request.task, kind='classification')
            request.inference['num_estimators'] = 3
            request.schema['instance']['feature'].stype = 'categorical'
            client.predict(request, model=KumoTabular())
        for body in scripted.calls:
            body['metadata']['request_id'] = '<drawn>'
        records.append(scripted.calls)
    assert records[1] == records[0]
    assert records[1][1]['predict']['instance_table']['rows'] == [
        [20.0],
        [30.0],
    ]
    assert records[1][1]['task']['kind'] == 'classification'
    assert records[1][1]['inference'] == {'num_estimators': 3}


def test_direct_successful_unencodable_labels_remain_unfinished_after_guards(
    monkeypatch,
):
    context, predict = fixture()
    context.columns = pd.Index(['\ud800', 'target'], dtype=object)
    predict.columns = pd.Index(['\ud800'], dtype=object)
    assert outcome(monkeypatch, (context, predict), core=False)[1][0] == 'ok'
    monkeypatch.setenv('SDM_CORE_ROUTES', CORE)
    with pytest.raises(RouteError, match='unencodable'):
        Request(context, predict, target='target', task='regression')


def test_direct_core_path_never_invokes_python_semantic_builders(monkeypatch):
    from sdm_client.adapters.tabular import TabularAdapter
    import sdm_client.schema as schema_module

    def forbidden(*args, **kwargs):
        raise AssertionError(
            'direct core journey called a Python domain builder'
        )

    monkeypatch.setenv('SDM_CORE_ROUTES', CORE)
    monkeypatch.setattr(
        schema_module, '_infer_table_and_time_column', forbidden
    )
    monkeypatch.setattr(TabularAdapter, '_envelope', forbidden)
    context, predict = fixture()
    request = Request(context, predict, target='target', task='regression')
    scripted = Scripted()
    with SDMClient('http://localhost:1', max_retries=0) as client:
        client._transport.close()
        client._transport = scripted
        assert len(client.predict(request, model=KumoTabular()).items) == 2
