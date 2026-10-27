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
    monkeypatch,
    frames,
    *,
    core,
    options=None,
    send=True,
    outputs=None,
    model='kumo-tabular',
    transport=None,
    edit=None,
    warning_error=False,
    on_warning=None,
):
    monkeypatch.setenv('SDM_CORE_ROUTES', CORE if core else '')
    scripted = Scripted() if transport is None else transport
    inspected = None
    presented = []
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
            if edit is not None:
                edit(request)
            if warning_error:
                from sdm_client.core.warnings import SDMWarning

                warnings.simplefilter('error', SDMWarning)
            if on_warning is not None:

                def show(
                    message, category, filename, lineno, file=None, line=None
                ):
                    presented.append((str(message), category))
                    on_warning(request, *frames)

                warnings.showwarning = show
            result = None
            if send:
                with SDMClient('http://localhost:1', max_retries=0) as client:
                    client._transport.close()
                    client._transport = scripted
                    result = client.predict(
                        request, model=model, outputs=outputs
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
        [(str(n.message), n.category) for n in caught] + presented,
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
    import sdm_client.schema as schema_module
    from sdm_client.adapters.tabular import TabularAdapter

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


@pytest.mark.parametrize('seed', [2**53, 2**53 + 1, 2**80, 10**400])
def test_direct_flat_inference_precision_is_checked_at_final_body(
    monkeypatch, seed
):
    frames = fixture('int64')
    options = {'inference': {'seed': seed}}
    old = outcome(monkeypatch, frames, core=False, options=options)
    new = outcome(monkeypatch, frames, core=True, options=options)
    assert new == old
    if seed in (2**53, 2**80):
        assert new[3][0]['inference'] == {'seed': seed}
    else:
        assert not new[3]
        assert (
            'cannot be sent as a JSON number without losing precision'
            in new[1][-1]
        )


@pytest.mark.parametrize('odd', [False, True])
def test_direct_integer_to_float_widening_never_rounds_large_integers(
    monkeypatch, odd
):
    context, predict = fixture('int64')
    first = 2**53 + int(odd)
    context['feature'] = [first, first + 2, first + 4]
    predict['feature'] = [4.0, 5.0]
    old = outcome(monkeypatch, (context, predict), core=False)
    assert outcome(monkeypatch, (context, predict), core=True) == old
    if odd:
        assert not old[3]
        assert (
            'cannot be sent as float64 without losing precision' in old[1][-1]
        )


@pytest.mark.parametrize('null', [False, True])
def test_direct_valid_masked_nan_and_null_wire_cells(monkeypatch, null):
    context, _ = fixture('int64')
    values = np.array([1.0, np.nan, 3.0])
    mask = np.array([False, False, null])
    predict = pd.DataFrame(
        {'feature': pd.Series(pd.arrays.FloatingArray(values, mask))}
    )
    old = outcome(monkeypatch, (context, predict), core=False)
    new = outcome(monkeypatch, (context, predict), core=True)
    assert new == old
    assert (
        new[3][0]['schema']['instance_table']['columns']['feature']['dtype']
        == 'float64'
    )
    assert new[3][0]['predict']['instance_table']['rows'][1] == [None]


@pytest.mark.parametrize('wrong', [False, True])
def test_direct_keyed_leak_warning_and_response_attribution(monkeypatch, wrong):
    class Keyed(Scripted):
        def predict(self, payload):
            result = super().predict(payload)
            rows = payload['predict']['instance_table']['rows']
            for item, row in zip(result['predictions'], rows, strict=True):
                item['id'] = str(
                    [4 if wrong and item['row_index'] == 0 else row[0]]
                )
            return result

    frames = (
        pd.DataFrame({'user_id': [1, 2, 3], 'target': [2, 4, 6]}),
        pd.DataFrame({'user_id': [1, 4]}),
    )
    old = outcome(monkeypatch, frames, core=False, transport=Keyed())
    new = outcome(monkeypatch, frames, core=True, transport=Keyed())
    assert new == old
    assert '1 of the 2 rows being scored are also in context' in new[2][0][0]
    if wrong:
        assert new[1][2] == 'INVALID_RESPONSE'
        assert 'pairs row_index 0' in new[1][-1]
    else:
        assert new[1][1][0]['id'] == [1, 4]


@pytest.mark.parametrize(
    'model,outputs', [('unknown-model', []), ('kumo-tabular', [])]
)
def test_direct_model_lookup_precedes_empty_outputs(
    monkeypatch, model, outputs
):
    frames = fixture()
    old = outcome(monkeypatch, frames, core=False, model=model, outputs=outputs)
    assert (
        outcome(monkeypatch, frames, core=True, model=model, outputs=outputs)
        == old
    )
    assert old[1][2] == (
        'UNKNOWN_MODEL' if model == 'unknown-model' else 'INVALID_REQUEST'
    )


def test_direct_boolean_storage_and_supplied_schema_refusal(monkeypatch):
    from sdm_client import Schema

    context, predict = fixture()
    context['flag'] = [True, False, True]
    predict['flag'] = pd.Series([False, None], dtype='boolean')
    old = outcome(monkeypatch, (context, predict), core=False)
    assert outcome(monkeypatch, (context, predict), core=True) == old
    schema = Schema.from_data({'instance': context}, relationships=[])
    predict['feature'] = [True, False]
    old = outcome(
        monkeypatch,
        (context, predict),
        core=False,
        options={'schema': schema},
        send=False,
    )
    assert (
        outcome(
            monkeypatch,
            (context, predict),
            core=True,
            options={'schema': schema},
            send=False,
        )
        == old
    )
    assert "which 'bool' cannot mean" in old[1][-1]


def test_direct_supplied_untyped_schema_uses_scored_type(monkeypatch):
    from sdm_client import Schema

    context, predict = fixture('int64')
    schema = Schema.from_data(
        {
            'instance': pd.DataFrame(
                {
                    'feature': pd.Series([None] * 3, dtype=object),
                    'target': [2, 4, 6],
                }
            )
        },
        relationships=[],
    )
    assert schema['instance']['feature'].untyped
    old = outcome(
        monkeypatch, (context, predict), core=False, options={'schema': schema}
    )
    new = outcome(
        monkeypatch, (context, predict), core=True, options={'schema': schema}
    )
    assert new == old
    assert new[3][0]['schema']['instance_table']['columns']['feature'] == {
        'dtype': 'int64',
        'stype': 'categorical',
    }


def test_direct_declared_meaning_survives_fresh_wide_scoring(monkeypatch):
    records = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', CORE if core else '')
        context, predict = fixture('int64')
        request = Request(context, predict, target='target', task='regression')
        request.schema['instance']['feature'].stype = 'numerical'
        assert request.schema['instance']['feature'].stype_declared
        predict['feature'] = pd.Series(
            [2**63, 2**63 + 1], index=predict.index, dtype='uint64'
        )
        scripted = Scripted()
        with SDMClient('http://localhost:1', max_retries=0) as client:
            client._transport.close()
            client._transport = scripted
            client.predict(request, model=KumoTabular())
        body = scripted.calls[0]
        body['metadata']['request_id'] = '<drawn>'
        records.append(body)
    assert records[1] == records[0]
    assert records[1]['schema']['instance_table']['columns']['feature'] == {
        'dtype': 'string',
        'stype': 'numerical',
    }


@pytest.mark.parametrize(
    'dtype,wire', [('int32', 'int64'), ('float32', 'float32')]
)
def test_direct_supplied_native_schema_retains_its_wire_width(
    monkeypatch, dtype, wire
):
    from sdm_client import Schema

    context, predict = fixture(dtype)
    schema = Schema.from_data({'instance': context}, relationships=[])
    assert schema['instance']['feature'].dtype == wire
    old = outcome(
        monkeypatch, (context, predict), core=False, options={'schema': schema}
    )
    new = outcome(
        monkeypatch, (context, predict), core=True, options={'schema': schema}
    )
    assert new == old
    assert new[1][0] == 'ok'
    assert (
        new[3][0]['schema']['instance_table']['columns']['feature']['dtype']
        == wire
    )


@pytest.mark.parametrize('kind', ['float', 'float-string', 'bool-string'])
@pytest.mark.parametrize('wrong', [False, True])
def test_direct_numeric_key_spellings_are_checked_before_restoration(
    monkeypatch, kind, wrong
):
    class Identified(Scripted):
        def predict(self, payload):
            answer = super().predict(payload)
            rows = payload['predict']['instance_table']['rows']
            for item in answer['predictions']:
                at = item['row_index']
                item['id'] = repr([rows[1 if wrong and at == 0 else at][0]])
            return answer

    if kind == 'bool-string':
        frames = (
            pd.DataFrame({'user_id': [1, 2, 3], 'target': [2, 4, 6]}),
            pd.DataFrame({'user_id': [4, 5]}),
        )
    else:
        frames = (
            pd.DataFrame({'user_id': [1.5, 2.5, 3.5], 'target': [2, 4, 6]}),
            pd.DataFrame({'user_id': [4.5, 5.5]}),
        )

    def edit(request):
        if 'string' in kind:
            request.schema['instance']['user_id'].dtype = 'string'
        if kind == 'bool-string':
            request.predict.instance_table['user_id'] = [True, False]

    old = outcome(
        monkeypatch,
        tuple(frame.copy(deep=True) for frame in frames),
        core=False,
        edit=edit,
        transport=Identified(),
    )
    new = outcome(
        monkeypatch,
        tuple(frame.copy(deep=True) for frame in frames),
        core=True,
        edit=edit,
        transport=Identified(),
    )
    assert new == old
    if wrong:
        assert new[1][2] == 'INVALID_RESPONSE'
        assert 'pairs row_index 0' in new[1][-1]
    else:
        assert new[1][0] == 'ok'


@pytest.mark.parametrize('sign', [1.0, -1.0])
@pytest.mark.parametrize('null', [False, True])
@pytest.mark.parametrize('masked', [False, True])
def test_direct_nullable_integer_cast_preserves_native_failure_boundary(
    monkeypatch, sign, null, masked
):
    from sdm_client.errors import StructuredDataError

    context, _ = fixture('int64')
    values = [1.0, sign * np.inf] + ([None] if null else [])
    predict = pd.DataFrame(
        {'feature': pd.Series(values, dtype='Float64' if masked else 'float64')}
    )
    old = outcome(monkeypatch, (context, predict), core=False)
    new = outcome(monkeypatch, (context, predict), core=True)
    assert new == old
    assert not new[3]
    assert new[1][1] is (
        pd.errors.IntCastingNaNError
        if null and not masked
        else StructuredDataError
    )


def wide_text_fixture():
    return (
        pd.DataFrame(
            {
                'feature': pd.Series(range(2**63, 2**63 + 65), dtype='uint64'),
                'target': np.arange(65),
            }
        ),
        pd.DataFrame(
            {'feature': pd.Series([2**63 + 66, 2**63 + 67], dtype='uint64')}
        ),
    )


@pytest.mark.parametrize('as_error', [False, True])
def test_direct_text_advice_precedes_fallible_row_and_inference_encoding(
    monkeypatch, as_error
):
    from sdm_client.core.warnings import SDMWarning

    frames = wide_text_fixture()
    options = {'inference': {'seed': 2**53 + 1}}
    old = outcome(
        monkeypatch, frames, core=False, options=options, warning_error=as_error
    )
    new = outcome(
        monkeypatch, frames, core=True, options=options, warning_error=as_error
    )
    assert new == old
    assert not new[3]
    if as_error:
        assert new[1][1] is SDMWarning
        assert not new[2]
    else:
        assert new[2] == [
            (
                'kumo-tabular cannot read a text column, so feature travel(s) as categorical instead. ',
                SDMWarning,
            )
        ]
        assert (
            'cannot be sent as a JSON number without losing precision'
            in new[1][-1]
        )


@pytest.mark.parametrize(
    'mutation', ['values', 'reference', 'inference', 'rows', 'task', 'schema']
)
def test_direct_text_advice_preserves_captured_references_and_later_reads(
    monkeypatch, mutation
):
    from sdm_client.request import Split

    def mutate(request, context, predict):
        if mutation == 'values':
            predict['feature'] = pd.Series(
                [2**63 + 88, 2**63 + 89], dtype='uint64'
            )
        elif mutation == 'reference':
            request._predict = Split(
                pd.DataFrame(
                    {
                        'feature': pd.Series(
                            [2**63 + 99, 2**63 + 100], dtype='uint64'
                        )
                    }
                )
            )
        elif mutation == 'inference':
            request.inference['seed'] = 2**53
        elif mutation == 'rows':
            context.loc[len(context)] = {'feature': 2**63 + 65, 'target': 65}
        elif mutation == 'task':
            request.task = replace(request.task, kind='forecasting')
        else:
            request.schema = None

    old = outcome(
        monkeypatch, wide_text_fixture(), core=False, on_warning=mutate
    )
    new = outcome(
        monkeypatch, wide_text_fixture(), core=True, on_warning=mutate
    )
    assert new == old
    assert new[1][0] == 'ok'
    assert len(new[2]) == 1
    body = new[3][0]
    assert body['task']['kind'] == (
        'forecasting' if mutation == 'task' else 'regression'
    )
    assert len(body['context']['instance_table']['rows']) == (
        66 if mutation == 'rows' else 65
    )
    assert body['predict']['instance_table']['rows'][0][0] == str(
        2**63 + (88 if mutation == 'values' else 66)
    )
    if mutation == 'inference':
        assert body['inference'] == {'seed': 2**53}


@pytest.mark.parametrize('mutation', ['reference', 'task'])
def test_direct_leakage_advice_precedes_envelope_reference_capture(
    monkeypatch, mutation
):
    from sdm_client.request import Split

    def inputs():
        return pd.DataFrame(
            {'user_id': [1, 2, 3], 'target': [2, 4, 6]}
        ), pd.DataFrame({'user_id': [1, 4]})

    def mutate(request, context, predict):
        if mutation == 'reference':
            request._predict = Split(pd.DataFrame({'user_id': [8, 9]}))
        else:
            request.task = replace(request.task, kind='forecasting')

    old = outcome(monkeypatch, inputs(), core=False, on_warning=mutate)
    new = outcome(monkeypatch, inputs(), core=True, on_warning=mutate)
    assert new == old
    assert len(new[2]) == 1
    body = new[3][0]
    if mutation == 'reference':
        assert new[1][0] == 'ok'
        assert body['predict']['instance_table']['rows'] == [[8], [9]]
    else:
        assert body['task']['kind'] == 'forecasting'
        assert new[1][2] == 'INVALID_RESPONSE'


@pytest.mark.parametrize('dtype', ['float64', 'Float64'])
def test_direct_nullable_large_finite_cast_widens_without_a_native_warning(
    monkeypatch, dtype
):
    context, _ = fixture('int64')
    predict = pd.DataFrame(
        {'feature': pd.Series([1.0, float(2**64), None], dtype=dtype)}
    )
    old = outcome(monkeypatch, (context, predict), core=False)
    new = outcome(monkeypatch, (context, predict), core=True)
    assert new == old
    assert new[1][0] == 'ok'
    assert not new[2]
    assert (
        new[3][0]['schema']['instance_table']['columns']['feature']['dtype']
        == 'float64'
    )


@pytest.mark.parametrize('as_error', [False, True])
@pytest.mark.parametrize('failing', [False, True])
def test_direct_overlap_only_widens_key_columns(monkeypatch, as_error, failing):
    from sdm_client.core.warnings import SDMWarning

    def inputs():
        return (
            pd.DataFrame(
                {
                    'user_id': [1, 2, 3],
                    'feature': [1, 2, 3],
                    'target': [2, 4, 6],
                }
            ),
            pd.DataFrame(
                {
                    'user_id': [1, 4],
                    'feature': [np.inf, np.nan] if failing else [4, 5],
                }
            ),
        )

    old = outcome(monkeypatch, inputs(), core=False, warning_error=as_error)
    new = outcome(monkeypatch, inputs(), core=True, warning_error=as_error)
    assert new == old
    if as_error:
        assert new[1][1] is SDMWarning
        assert not new[3]
    else:
        assert len(new[2]) == 1
        assert '1 of the 2 rows' in new[2][0][0]
        if failing:
            assert new[1][1] is pd.errors.IntCastingNaNError
            assert not new[3]


@pytest.mark.parametrize('key', [None, 'alternate', ['alternate', 'user_id']])
def test_direct_primary_key_is_read_from_captured_table_after_text_advice(
    monkeypatch, key
):
    class Opaque(Scripted):
        def predict(self, payload):
            answer = super().predict(payload)
            for item in answer['predictions']:
                item['id'] = ['opaque-a', 'opaque-b'][item['row_index']]
            return answer

    def inputs():
        context, predict = wide_text_fixture()
        context.insert(0, 'user_id', np.arange(65))
        context.insert(1, 'alternate', np.arange(100, 165))
        predict.insert(0, 'user_id', [66, 67])
        predict.insert(1, 'alternate', [166, 167])
        return context, predict

    def mutate(request, context, predict):
        table = request.schema['instance']
        # Replacing the Request schema must not replace the captured table.
        request.schema = None
        table.primary_key = key

    old = outcome(
        monkeypatch, inputs(), core=False, on_warning=mutate, transport=Opaque()
    )
    new = outcome(
        monkeypatch, inputs(), core=True, on_warning=mutate, transport=Opaque()
    )
    assert new == old
    assert new[1][0] == 'ok'
    instance = new[3][0]['schema']['instance_table']
    assert instance.get('primary_key') == (
        None if key is None else [key] if isinstance(key, str) else key
    )
    # The earlier column plan remains frozen even when the setter retypes it.
    assert instance['columns']['alternate']['stype'] == 'numerical'
    assert new[1][1][0]['id'] == (
        [166, 167] if key == 'alternate' else ['opaque-a', 'opaque-b']
    )


@pytest.mark.parametrize(
    'names',
    [
        ['__kumo_entity_ref_feature'],
        ['instance_id'],
        ['__kumo_entity_ref_feature', 'instance_id'],
    ],
)
def test_direct_ordinary_feature_names_do_not_acquire_gathered_attribution(
    monkeypatch, names
):
    class Opaque(Scripted):
        def predict(self, payload):
            answer = super().predict(payload)
            for item in answer['predictions']:
                item['id'] = ['opaque-a', 'opaque-b'][item['row_index']]
            return answer

    def inputs():
        return pd.DataFrame(
            {**{name: [1.0, 2.0, 3.0] for name in names}, 'target': [2, 4, 6]}
        ), pd.DataFrame({name: [4.0, 5.0] for name in names})

    old = outcome(monkeypatch, inputs(), core=False, transport=Opaque())
    new = outcome(monkeypatch, inputs(), core=True, transport=Opaque())
    assert new == old
    assert new[1][0] == 'ok'
    assert new[1][1][0]['id'] == (
        [4.0, 5.0] if 'instance_id' in names else ['opaque-a', 'opaque-b']
    )


@pytest.mark.parametrize('form', ['[{}]', '[{},]', ' [ {}, ] ', "['{}',]"])
@pytest.mark.parametrize('wrong', [False, True])
def test_direct_primitive_list_trailing_comma_ids_keep_attribution(
    monkeypatch, form, wrong
):
    class Identified(Scripted):
        def predict(self, payload):
            answer = super().predict(payload)
            for item in answer['predictions']:
                at = item['row_index']
                item['id'] = form.format(5 if wrong or at else 4)
            return answer

    frames = (
        pd.DataFrame({'user_id': [1, 2, 3], 'target': [2, 4, 6]}),
        pd.DataFrame({'user_id': [4, 5]}),
    )
    old = outcome(monkeypatch, frames, core=False, transport=Identified())
    new = outcome(monkeypatch, frames, core=True, transport=Identified())
    assert new == old
    if wrong:
        assert new[1][2] == 'INVALID_RESPONSE'
    else:
        assert new[1][0] == 'ok'
        assert new[1][1][0]['id'] == [4, 5]


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('composite', [False, True])
@pytest.mark.parametrize('as_error', [False, True])
def test_direct_signed_zero_keys_share_native_overlap_identity(
    monkeypatch, reverse, composite, as_error
):
    from sdm_client.core.warnings import SDMWarning

    def inputs():
        context = pd.DataFrame(
            {
                'user_id': [0.0 if reverse else -0.0, 1.0, 2.0],
                'target': [2, 4, 6],
            }
        )
        predict = pd.DataFrame({'user_id': [-0.0 if reverse else 0.0, 3.0]})
        if composite:
            context['part'] = [1, 2, 3]
            predict['part'] = [1, 4]
        return context, predict

    def edit(request):
        if composite:
            request.schema['instance'].primary_key = ['user_id', 'part']

    old = outcome(
        monkeypatch, inputs(), core=False, edit=edit, warning_error=as_error
    )
    new = outcome(
        monkeypatch, inputs(), core=True, edit=edit, warning_error=as_error
    )
    assert new == old
    if as_error:
        assert new[1][1] is SDMWarning
        assert not new[3]
    else:
        assert len(new[2]) == 1
        assert '1 of the 2 rows' in new[2][0][0]


@pytest.mark.parametrize('first', ['nonfinite', 'precision'])
def test_direct_body_serializes_whole_columns_before_rows(monkeypatch, first):
    def inputs():
        context = pd.DataFrame(
            {'a': [1.0, np.inf], 'b': [2**53 + 1, 1], 'target': [1, 2]}
        )
        predict = pd.DataFrame({'a': [3.0, 4.0], 'b': [1.0, 2.0]})
        if first == 'precision':
            context, predict = (
                context[['b', 'a', 'target']],
                predict[['b', 'a']],
            )
        return context, predict

    old = outcome(monkeypatch, inputs(), core=False)
    new = outcome(monkeypatch, inputs(), core=True)
    assert new == old
    assert new[1][2] == 'INVALID_REQUEST'
    assert (
        'not JSON-representable'
        if first == 'nonfinite'
        else 'without losing precision'
    ) in new[1][-1]
    assert not new[3]


@pytest.mark.parametrize(
    ('identifier', 'invalid'),
    [
        ('[+5,]', True),
        ('[-5,]', True),
        ('[0x5,]', True),
        ('[0o5,]', True),
        ('[0b101,]', True),
        ('[5_0,]', True),
        ('[5.0e0,]', True),
        ('[+5.5,]', True),
        ('[True,]', True),
        ('[None,]', True),
        ('[4+1,]', False),
        ('[4,,]', False),
        ('[04,]', False),
    ],
)
def test_direct_primitive_literal_grammar_preserves_opaque_tokens(
    monkeypatch, identifier, invalid
):
    class Identified(Scripted):
        def predict(self, payload):
            answer = super().predict(payload)
            answer['predictions'][0]['id'] = identifier
            answer['predictions'][1]['id'] = '[5]'
            return answer

    frames = (
        pd.DataFrame({'user_id': [1, 2, 3], 'target': [2, 4, 6]}),
        pd.DataFrame({'user_id': [4, 5]}),
    )
    old = outcome(monkeypatch, frames, core=False, transport=Identified())
    new = outcome(monkeypatch, frames, core=True, transport=Identified())
    assert new == old
    if invalid:
        assert new[1][2] == 'INVALID_RESPONSE'
    else:
        assert new[1][0] == 'ok'
        assert new[1][1][0]['id'] == [4, 5]


@pytest.mark.parametrize(
    'form', ["['{}.5',]", '[{}.5,]', '[+{}.5,]', '[{}.5e0,]']
)
@pytest.mark.parametrize('wrong', [False, True])
def test_direct_primitive_literal_float_and_string_float_ids(
    monkeypatch, form, wrong
):
    class Identified(Scripted):
        def predict(self, payload):
            answer = super().predict(payload)
            for item in answer['predictions']:
                at = item['row_index']
                item['id'] = form.format(5 if wrong or at else 4)
            return answer

    def inputs():
        return pd.DataFrame(
            {'user_id': [1.5, 2.5, 3.5], 'target': [2, 4, 6]}
        ), pd.DataFrame({'user_id': [4.5, 5.5]})

    def edit(request):
        if "'" in form:
            request.schema['instance']['user_id'].dtype = 'string'

    old = outcome(
        monkeypatch, inputs(), core=False, edit=edit, transport=Identified()
    )
    new = outcome(
        monkeypatch, inputs(), core=True, edit=edit, transport=Identified()
    )
    assert new == old
    assert new[1][2] == 'INVALID_RESPONSE' if wrong else new[1][0] == 'ok'


@pytest.mark.parametrize('form', ['[({}),]', r"['\x3{}',]"])
@pytest.mark.parametrize('wrong', [False, True])
def test_direct_grouped_or_escaped_primitive_key_ids(monkeypatch, form, wrong):
    class Identified(Scripted):
        def predict(self, payload):
            answer = super().predict(payload)
            for item in answer['predictions']:
                at = item['row_index']
                item['id'] = form.format(5 if wrong or at else 4)
            return answer

    frames = (
        pd.DataFrame({'user_id': [1, 2, 3], 'target': [2, 4, 6]}),
        pd.DataFrame({'user_id': [4, 5]}),
    )
    old = outcome(monkeypatch, frames, core=False, transport=Identified())
    new = outcome(monkeypatch, frames, core=True, transport=Identified())
    assert new == old
    if wrong:
        assert new[1][2] == 'INVALID_RESPONSE'
    else:
        assert new[1][0] == 'ok'
        assert new[1][1][0]['id'] == [4, 5]
