# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen43e whole inference, native ownership and generated response boundaries."""

from __future__ import annotations

import copy
import math
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from test_t46_direct_numeric import CORE

from sdm_client import Column, KumoTabular, Request, Schema, SDMClient, Table

FRAMES = 'schema.from-data/frames/frames,schema.from-data/mapping/frames'


class ObservationFailure(Exception):
    pass


def observe(function):
    try:
        return function()
    except Exception as failure:
        raise ObservationFailure('probe observation failed') from failure


def error(e):
    return {
        'module': type(e).__module__,
        'type': type(e).__name__,
        'code': getattr(e, 'code', None),
        'message': str(e),
        'cause': None
        if e.__cause__ is None
        else {
            'module': type(e.__cause__).__module__,
            'type': type(e.__cause__).__name__,
            'message': str(e.__cause__),
        },
    }


def columns(table):
    return {
        name: {
            'name': col.name,
            'dtype': col.dtype,
            'stype': str(col.stype),
            'declared': col.stype_declared,
            'before': col._stype_before_key,
            'table_is_owner': getattr(col, '_table', None) is table,
        }
        for name, col in table.columns.items()
    }


def schema_state(schema):
    return {
        name: {
            'name': table.name,
            'key': table.primary_key,
            'time': table.time_column,
            'duplicates': table.duplicate_keys,
            'schema_is_owner': getattr(table, '_schema', None) is schema,
            'columns': columns(table),
        }
        for name, table in schema.tables.items()
    }


def result_state(result, frame):
    return {
        'frame': frame.to_dict(orient='list'),
        'dtypes': [str(t) for t in frame.dtypes],
        'model': result.model,
        'request_id': result.request_id,
        'metadata': result.metadata,
        'items': [repr(i) for i in result.items],
    }


def key_boundary(
    label,
    api,
    change,
    *,
    n=100,
    initial_repeat=False,
    time=False,
    error_warning=False,
):
    key = pd.Series([None, *range(1, n)], dtype='Int64')
    if initial_repeat:
        key.iloc[2] = 1
    context = pd.DataFrame(
        {'user_id': key, 'feature': range(n), 'target': range(n, 2 * n)}
    )
    if time:
        context['created_at'] = pd.date_range('2026-01-01', periods=n).as_unit(
            'us'
        )
    predict = pd.DataFrame(
        {'user_id': pd.Series([n, n + 1], dtype='Int64'), 'feature': [n, n + 1]}
    )
    if time:
        predict['created_at'] = pd.date_range('2027-01-01', periods=2).as_unit(
            'us'
        )
    record = {'case': f'{api}/{label}', 'events': []}
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        if error_warning:
            from sdm_client.core.warnings import SDMWarning

            warnings.simplefilter('error', SDMWarning)

        def show(message, category, filename, lineno, file=None, line=None):
            record['events'].append(
                observe(
                    lambda: {
                        'type': category.__name__,
                        'message': str(message),
                        'context_rows': len(context),
                        'columns': list(context.columns),
                        'key_values': context['user_id']
                        .astype(object)
                        .where(context['user_id'].notna(), None)
                        .tolist()
                        if 'user_id' in context
                        else None,
                    }
                )
            )
            event = len(record['events'])
            if change == 'create' and event == 1:
                context.loc[2, 'user_id'] = 1
            if change == 'remove' and event == 1:
                context.loc[2, 'user_id'] = 2
            if change == 'fill-missing' and event == 1:
                context.loc[0, 'user_id'] = n + 2
            if change == 'no-remedy' and event == 1:
                context.loc[2, 'user_id'] = 1
                context.loc[2, ['feature', 'target']] = context.loc[
                    1, ['feature', 'target']
                ]
            if change == 'target-remedy' and event == 1:
                context.loc[2, 'user_id'] = 1
                context.loc[2, 'feature'] = context.loc[1, 'feature']
            if change == 'unusable-remedy' and event == 1:
                context.loc[2, 'user_id'] = 1
                context.loc[0, ['feature', 'target']] = None
            if change == 'drop-key' and event == 1:
                context.drop(columns=['user_id'], inplace=True)
            if change == 'empty' and event == 1:
                context.drop(index=context.index, inplace=True)
            if change == 'boolean-key' and event == 1:
                context['user_id'] = pd.Series(
                    [True, False] * (n // 2), dtype='boolean'
                )
            if change == 'chained' and event == 1:
                context.loc[2, 'user_id'] = 1
            if change == 'chained' and event == 2:
                predict.drop(columns=['feature'], inplace=True)
            if change == 'after-repeat' and event == 1:
                context.loc[2, 'user_id'] = 1
            if change == 'after-repeat' and event == 2:
                context.loc[2, 'user_id'] = 2
            if change == 'time-null' and event == 1:
                context['created_at'] = pd.Series(
                    pd.NaT, index=context.index, dtype='datetime64[us]'
                )
            if change == 'time-after-repeat' and event == 1:
                context.loc[2, 'user_id'] = 1
            if change == 'time-after-repeat' and event == 2:
                context['created_at'] = pd.Series(
                    pd.NaT, index=context.index, dtype='datetime64[us]'
                )
            if change == 'time-drop' and event == 1:
                context.drop(columns=['created_at'], inplace=True)
            if change == 'time-numeric' and event == 1:
                context['created_at'] = range(n)
            if change == 'time-added' and event == 1:
                context['created_at'] = pd.date_range(
                    '2026-01-01', periods=n
                ).as_unit('us')

        warnings.showwarning = show
        try:
            if api == 'request':
                obj = Request(
                    context, predict, target='target', task='regression'
                )
            else:
                schema = Schema.from_data(
                    {'instance': context}, relationships=()
                )
        except ObservationFailure:
            raise
        except Exception as e:
            record.update(ok=False, error=error(e))
        else:
            if api == 'request':
                schema = obj.schema
                record['inspection'] = repr(obj)
            record.update(ok=True, schema=schema_state(schema))
    record['final_context_columns'] = list(context.columns)
    record['final_context_dtypes'] = [str(t) for t in context.dtypes]
    record['final_predict_columns'] = list(predict.columns)
    return record


class Scripted:
    url = 'scripted://direct-boundaries-r7'

    def __init__(self, edit=None):
        self.calls = []
        self.responses = []
        self.edit = edit

    def close(self):
        pass

    def predict(self, payload):
        self.calls.append(copy.deepcopy(payload))
        answer = {
            'id': 'independent-server',
            'model': 'kumo-tabular',
            'metadata': {},
            'predictions': [
                {'row_index': i, 'id': str(i), 'prediction': 10 + i}
                for i in range(2)
            ],
        }
        if self.edit is not None:
            answer = self.edit(answer)
        self.responses.append(copy.deepcopy(answer))
        return answer


def ownership_boundary(mode):
    context = pd.DataFrame(
        {'user_id': [1, 2, 3], 'feature': [2, 3, 4], 'target': [2, 4, 6]}
    )
    predict = pd.DataFrame(
        {
            'user_id': [2 if mode == 'overlap-replacement' else 4, 5],
            'feature': [5, 6],
        }
    )
    record = {'case': 'ownership/' + mode, 'events': []}
    transport = Scripted()
    refs = {}
    with warnings.catch_warnings():
        warnings.simplefilter('always')

        def show(message, category, filename, lineno, file=None, line=None):
            record['events'].append(
                {'type': category.__name__, 'message': str(message)}
            )
            if mode == 'overlap-replacement' and 'also in context' in str(
                message
            ):
                refs['replacement'] = Column('user_id', 'int64', 'numerical')
                request.schema.tables['instance'].columns['user_id'] = refs[
                    'replacement'
                ]

        warnings.showwarning = show
        request = None

        def setup():
            request = Request(
                context, predict, target='target', task='regression'
            )
            destination = request.schema.tables['instance']
            refs['original_table'] = destination
            refs['original_columns'] = destination.columns
            if mode in ['unowned-column', 'bad-nonkey']:
                col = Column(
                    'user_id' if mode == 'unowned-column' else 'feature',
                    'int64' if mode == 'unowned-column' else 'string',
                    'numerical' if mode == 'unowned-column' else 'text',
                )
                if mode == 'bad-nonkey':
                    col.dtype = 'int64'
                refs['replacement'] = col
                destination.columns[col.name] = col
            elif mode in ['borrowed-column', 'borrowed-former-key']:
                name = 'user_id' if mode == 'borrowed-column' else 'feature'
                col = Column(name, 'int64', 'numerical')
                donor = Table(
                    name='donor',
                    data=context.copy(),
                    columns={name: col},
                    primary_key=name if mode == 'borrowed-former-key' else None,
                )
                refs.update(donor=donor, replacement=col)
                destination.columns[name] = col
            elif mode in ['unowned-table', 'borrowed-table']:
                incoming = Table(
                    name='instance',
                    data=context.copy(),
                    columns={
                        name: Column(name, 'int64', 'numerical')
                        for name in context.columns
                    },
                    primary_key='user_id',
                )
                if mode == 'borrowed-table':
                    donor = Schema({'instance': incoming}, ())
                    incoming = donor.tables['instance']
                    refs['donor_schema'] = donor
                incoming.columns['user_id'] = Column(
                    'user_id', 'int64', 'numerical'
                )
                refs['replacement_table'] = incoming
                refs['replacement'] = incoming.columns['user_id']
                request.schema.tables['instance'] = incoming
            elif mode == 'explicit-nonkey-ID':
                destination.columns['feature'] = Column(
                    'feature', 'int64', 'ID'
                )
                refs['replacement'] = destination.columns['feature']
            elif mode == 'release-key':
                donor = Table(
                    name='donor',
                    data=context.copy(),
                    columns={
                        'feature': Column('feature', 'int64', 'numerical')
                    },
                    primary_key='feature',
                )
                refs.update(donor=donor, replacement=donor.columns['feature'])
                destination.columns['feature'] = donor.columns['feature']
            elif mode == 'unknown-model-before-adoption':
                refs['replacement'] = Column('user_id', 'int64', 'numerical')
                destination.columns['user_id'] = refs['replacement']
            return request

        try:
            request = setup()
        except ObservationFailure:
            raise
        except Exception as e:
            record.update(ok=False, failed_operation='setup', error=error(e))
        else:
            record['before'] = schema_state(request.schema)
            try:
                with SDMClient('http://localhost:1', max_retries=0) as client:
                    client._transport.close()
                    client._transport = transport
                    result = client.predict(
                        request,
                        model='unknown'
                        if mode == 'unknown-model-before-adoption'
                        else KumoTabular(),
                    )
                    frame = result.to_pandas()
            except ObservationFailure:
                raise
            except Exception as e:
                record.update(
                    ok=False,
                    failed_operation='predict_or_to_pandas',
                    error=error(e),
                )
            else:
                record.update(ok=True, result=result_state(result, frame))
        if request is not None:
            record['after'] = schema_state(request.schema)
            current = request.schema.tables.get('instance')
            record['identity'] = {
                'same_table_as_original': current is refs.get('original_table'),
                'same_columns_dict': current is not None
                and current.columns is refs.get('original_columns'),
                'same_inserted_table': current is refs.get('replacement_table'),
                'same_inserted_column': current is not None
                and refs.get('replacement') is not None
                and any(
                    c is refs['replacement'] for c in current.columns.values()
                ),
                'live_table_schema_ref': current is not None
                and getattr(current, '_schema', None) is request.schema,
            }
        if 'replacement' in refs:
            c = refs['replacement']
            record['original_inserted_column'] = {
                'stype': str(c.stype),
                'before': c._stype_before_key,
                'owner_is_destination': getattr(c, '_table', None)
                is request.schema.tables['instance'],
                'owner_is_donor': 'donor' in refs
                and getattr(c, '_table', None) is refs['donor'],
                'owner_is_none': getattr(c, '_table', None) is None,
            }
        if 'donor_schema' in refs:
            record['donor_schema'] = schema_state(refs['donor_schema'])
        if 'donor' in refs:
            record['donor_columns'] = columns(refs['donor'])
    record['bodies'] = transport.calls
    record['responses'] = transport.responses
    return record


def response_boundary(label, change):
    context = pd.DataFrame({'user_id': [1, 2, 3], 'target': [2, 4, 6]})
    predict = pd.DataFrame({'user_id': [4, 5]})
    transport = Scripted(change)
    record = {'case': 'response/' + label, 'events': []}
    with warnings.catch_warnings():
        warnings.simplefilter('always')

        def show(message, category, filename, lineno, file=None, line=None):
            record['events'].append(
                {'type': category.__name__, 'message': str(message)}
            )

        warnings.showwarning = show
        try:
            request = Request(
                context, predict, target='target', task='regression'
            )
            with SDMClient('http://localhost:1', max_retries=0) as client:
                client._transport.close()
                client._transport = transport
                result = client.predict(request, model=KumoTabular())
                frame = result.to_pandas()
        except ObservationFailure:
            raise
        except Exception as e:
            record.update(ok=False, error=error(e))
        else:
            record.update(ok=True, result=result_state(result, frame))
    record['bodies'] = transport.calls
    record['responses'] = transport.responses
    return record


def top(field, value, missing=False):
    def edit(answer):
        if missing:
            answer.pop(field, None)
        else:
            answer[field] = value
        return answer

    return edit


def item(fields):
    def edit(answer):
        answer['predictions'][0].update(fields)
        return answer

    return edit


RESPONSE_CASES = []


def response_case(label, change):
    RESPONSE_CASES.append(pytest.param(label, change, id=label))


for value in [None, [], 4, 'bad']:
    response_case(
        'root-' + type(value).__name__, lambda answer, value=value: value
    )
for field in ['id', 'model', 'predictions', 'metadata']:
    response_case(field + '-missing', top(field, None, True))
    for label, value in [('null', None), ('integer', 4), ('array', [])]:
        response_case(field + '-' + label, top(field, value))
for label, value in [
    ('string', 'bad'),
    ('nonempty-object', {'x': {}}),
    ('empty-object', {}),
]:
    response_case('predictions-' + label, top('predictions', value))
for label, value in [
    ('null', None),
    ('integer', 4),
    ('array', []),
    ('string', 'bad'),
]:
    response_case('item-' + label, top('predictions', [value]))
for label, value in [('integer', 4), ('bool', True), ('array', [])]:
    response_case('item-id-' + label, item({'id': value}))
for label, value in [
    ('string', '0'),
    ('bool', False),
    ('fraction', 0.5),
    ('infinity', float('inf')),
    ('wide-integer', 2**200),
    ('wide-whole-float', 1e20),
]:
    response_case('forecast-step-' + label, item({'forecast_step': value}))
for field in ['probabilities', 'scores', 'embeddings', 'quantiles']:
    for label, value in [
        ('string', 'bad'),
        ('bool', True),
        ('object', {'x': 'bad'})
        if field in ['scores', 'embeddings']
        else ('array', []),
    ]:
        response_case(field + '-container-' + label, item({field: value}))
    for label, value in [
        ('bool', True),
        ('text', 'bad'),
        ('null', None),
        ('infinity', float('inf')),
        ('wide-int', 10**400),
        ('wide-text', '1' + '0' * 400),
    ]:
        response_case(
            field + '-member-' + label,
            item(
                {
                    field: {'x': value}
                    if field in ['probabilities', 'quantiles']
                    else [value]
                }
            ),
        )
for field in ['rankings', 'explanation', 'metadata']:
    for label, value in [
        ('integer', 4),
        ('text', 'bad'),
        ('pair-arity', [['x']]),
        ('unhashable', [[[], 1]]),
    ]:
        response_case('item-' + field + '-' + label, item({field: value}))
for first, second in [
    ('id', 'row_index'),
    ('probabilities', 'scores'),
    ('scores', 'rankings'),
    ('rankings', 'embeddings'),
    ('embeddings', 'quantiles'),
    ('embeddings', 'explanation'),
    ('quantiles', 'explanation'),
    ('explanation', 'metadata'),
]:
    bad = {
        'id': 4,
        'row_index': 'bad',
        'probabilities': [],
        'scores': 'bad',
        'rankings': 4,
        'embeddings': 'bad',
        'quantiles': [],
        'explanation': 4,
        'metadata': 4,
    }
    response_case(
        'paired-' + first + '-' + second,
        item({first: bad[first], second: bad[second]}),
    )


def late(answer):
    answer['predictions'][0]['id'] = r"[4+1, '\q']"
    answer['predictions'][1]['embeddings'] = 'bad'
    return answer


response_case('global-late-field-before-ID-warning', late)
response_case('valid-independent-model', top('model', 'independent-model'))
response_case(
    'valid-dict-pair-explanation',
    item(
        {
            'explanation': [['x', 1]],
            'metadata': [['y', 2]],
            'rankings': [[['z', 3]]],
            'embeddings': [1, '2.5'],
            'quantiles': {'0.5': '2.0'},
        }
    ),
)


def comparable(value):
    if isinstance(value, float) and math.isnan(value):
        return ('nonfinite', 'nan')
    if isinstance(value, dict):
        return {key: comparable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [comparable(item) for item in value]
    return value


def typed_value(value):
    if isinstance(value, dict):
        return (
            type(value).__name__,
            [(typed_value(k), typed_value(v)) for k, v in value.items()],
        )
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [typed_value(v) for v in value])
    return (type(value).__name__, value)


MISSING_PREDICTION = object()
PREDICTION_VALUES = [
    pytest.param(MISSING_PREDICTION, None, id='missing'),
    pytest.param(None, None, id='null'),
    pytest.param(False, False, id='false'),
    pytest.param(True, True, id='true'),
    pytest.param(2, 2, id='integer'),
    pytest.param(2**2200, 2**2200, id='wide-integer'),
    pytest.param(-(2**2200), -(2**2200), id='negative-wide-integer'),
    pytest.param(2.0, 2, id='whole-float'),
    pytest.param(1.5, 1.5, id='fraction'),
    pytest.param('0_2', 2, id='integer-text'),
    pytest.param('1_0.5', '1_0.5', id='float-text'),
    pytest.param('unknown', 'unknown', id='other-text'),
    pytest.param('0' * 650 + '2', 2, id='padded-integer-text'),
    pytest.param([], [], id='empty-list'),
    pytest.param([1, 2], [1, 2], id='list'),
    pytest.param({}, {}, id='empty-object'),
    pytest.param({'a': 2}, {'a': 2}, id='object'),
    pytest.param(
        {'a': [2**2200, {'b': -(2**2200)}]},
        {'a': [2**2200, {'b': -(2**2200)}]},
        id='nested-wide',
    ),
]


@pytest.mark.parametrize('value,int_class', PREDICTION_VALUES)
@pytest.mark.parametrize(
    'task,dtype',
    [
        ('regression', 'int64'),
        ('classification', 'int64'),
        ('classification', 'float64'),
        ('classification', 'bool'),
    ],
)
@pytest.mark.parametrize('lower', [False, True])
def test_direct_prediction_value_family_keeps_task_cast_and_catch_policy(
    monkeypatch, value, int_class, task, dtype, lower
):
    # Typed outcomes come from the immutable 144-case frozen public matrix.
    missing = value is MISSING_PREDICTION
    if missing:
        value = None
    observed = []
    saved = sys.get_int_max_str_digits()
    try:
        for core in (False, True):
            sys.set_int_max_str_digits(4300)
            monkeypatch.setenv('SDM_CORE_ROUTES', CORE if core else '')
            target = [True, False, True] if dtype == 'bool' else [2, 4, 6]
            context = pd.DataFrame(
                {
                    'user_id': [1, 2, 3],
                    'feature': [1, 2, 3],
                    'target': pd.Series(target, dtype=dtype),
                }
            )
            scoring = pd.DataFrame({'user_id': [4, 5], 'feature': [4, 5]})

            def edit(answer):
                for at, item in enumerate(answer['predictions']):
                    item['prediction'] = copy.deepcopy(value)
                    if missing:
                        del item['prediction']
                    if lower and at == 0:
                        item['id'] = r"[4+1, '\q']"
                return answer

            transport = Scripted(edit)
            events = []
            result = None
            failure = None
            with warnings.catch_warnings():
                warnings.simplefilter('always')

                def show(
                    message, category, filename, lineno, file=None, line=None
                ):
                    events.append(
                        (str(message), category, sys.get_int_max_str_digits())
                    )
                    if lower and category in (
                        SyntaxWarning,
                        DeprecationWarning,
                    ):
                        sys.set_int_max_str_digits(640)

                warnings.showwarning = show
                try:
                    query = Request(
                        context, scoring, target='target', task=task
                    )
                    with SDMClient(
                        'http://localhost:1', max_retries=0
                    ) as client:
                        client._transport.close()
                        client._transport = transport
                        result = client.predict(query, model=KumoTabular())
                except Exception as exc:
                    failure = exc
                finally:
                    at_end = sys.get_int_max_str_digits()
                    sys.set_int_max_str_digits(4300)
            record = {
                'ok': failure is None,
                'events': events,
                'limit_at_end': at_end,
                'calls': len(transport.calls),
            }
            if failure is not None:
                record['error'] = error(failure)
            else:
                record['predictions'] = [
                    typed_value(item.prediction) for item in result.items
                ]
                record['ids'] = [typed_value(item.id) for item in result.items]
            observed.append(record)
        assert observed[1] == observed[0]
        assert observed[1]['ok']
        assert observed[1]['calls'] == 1
        assert len(observed[1]['events']) == lower
        assert observed[1]['limit_at_end'] == (640 if lower else 4300)
        expected = value
        if task == 'classification' and dtype == 'int64':
            expected = (
                value if lower and value == '0' * 650 + '2' else int_class
            )
        if task == 'classification' and dtype == 'float64':
            if type(value) in (int, float) and abs(value) <= 2:
                expected = 2.0 if value == 2 else value
            if value == '0_2' or value == '0' * 650 + '2':
                expected = 2.0
            if value == '1_0.5':
                expected = 10.5
        assert observed[1]['predictions'] == [typed_value(expected)] * 2
        assert observed[1]['ids'] == [('int', 4), ('int', 5)]
    finally:
        sys.set_int_max_str_digits(saved)


@pytest.mark.parametrize('column', ['target', 'feature'])
@pytest.mark.parametrize('declared', [False, True])
def test_direct_text_meaning_precedes_regression_target_override(
    monkeypatch, column, declared
):
    observed = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', CORE if core else '')
        context = pd.DataFrame(
            {'user_id': [1, 2, 3], 'feature': [1, 2, 3], 'target': [2, 4, 6]}
        )
        predict = pd.DataFrame({'user_id': [4, 5], 'feature': [4, 5]})
        schema = Schema.from_data({'instance': context}, relationships=[])
        schema['instance'][column].dtype = 'string'
        if declared:
            schema['instance'][column].stype = 'text'
        transport = Scripted()
        record = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                request = Request(
                    context,
                    predict,
                    target='target',
                    task='regression',
                    schema=schema,
                )
                with SDMClient('http://localhost:1', max_retries=0) as client:
                    client._transport.close()
                    client._transport = transport
                    result = client.predict(request, model=KumoTabular())
            except Exception as exc:
                record.update(ok=False, error=error(exc))
            else:
                record.update(
                    ok=True,
                    predictions=[
                        typed_value(i.prediction) for i in result.items
                    ],
                )
            record['warnings'] = [(str(w.message), w.category) for w in caught]
        for body in transport.calls:
            body['metadata']['request_id'] = '<drawn>'
        record['bodies'] = transport.calls
        observed.append(record)
    assert observed[1] == observed[0]
    if declared and column == 'target':
        assert observed[1]['ok']
        assert len(observed[1]['bodies']) == len(observed[1]['warnings']) == 1
        body = observed[1]['bodies'][0]
        assert body['schema']['instance_table']['columns']['target'] == {
            'dtype': 'string',
            'stype': 'categorical',
        }
        assert body['context']['instance_table']['rows'] == [
            [1, 1, '2'],
            [2, 2, '4'],
            [3, 3, '6'],
        ]
    else:
        assert not observed[1]['ok']
        assert observed[1]['error']['code'] == 'INVALID_REQUEST'
        assert not observed[1]['bodies']
        assert not observed[1]['warnings']


@pytest.mark.parametrize(
    'family',
    [
        'metadata',
        'prediction',
        'forecast',
        'item-mappings',
        'combined',
        'text-class',
        'numeric-class',
        'small',
    ],
)
@pytest.mark.parametrize('effect', ['none', 'warning', 'lower'])
def test_direct_decoded_result_carriers_survive_later_native_digit_limit(
    monkeypatch, family, effect
):
    wide = 2**2200
    saved = sys.get_int_max_str_digits()
    observed = []
    try:
        for core in (False, True):
            sys.set_int_max_str_digits(4300)
            monkeypatch.setenv('SDM_CORE_ROUTES', CORE if core else '')
            context = pd.DataFrame(
                {
                    'user_id': [1, 2, 3],
                    'feature': [1, 2, 3],
                    'target': [2, 4, 6],
                }
            )
            predict = pd.DataFrame({'user_id': [4, 5], 'feature': [4, 5]})

            def edit(answer):
                first = answer['predictions'][0]
                if effect != 'none':
                    first['id'] = r"[4+1, '\q']"
                if family in ('metadata', 'combined'):
                    answer['metadata'] = {
                        'wide': wide,
                        'nested': [{'wide': -wide}],
                    }
                if family in ('prediction', 'numeric-class', 'combined'):
                    first['prediction'] = wide
                if family in ('forecast', 'combined'):
                    first['forecast_step'] = wide
                if family in ('item-mappings', 'combined'):
                    first.update(
                        metadata={'wide': wide},
                        explanation={'wide': -wide},
                        rankings=[{'wide': wide}],
                    )
                if family == 'text-class':
                    first.update(
                        prediction='0' * 650 + '2',
                        probabilities={'0' * 650 + '2': 0.75, '4': 0.25},
                    )
                return answer

            transport = Scripted(edit)
            record = {'events': []}
            failure = None
            result = None
            with warnings.catch_warnings():
                warnings.simplefilter('always')

                def show(
                    message, category, filename, lineno, file=None, line=None
                ):
                    record['events'].append(
                        (str(message), category, sys.get_int_max_str_digits())
                    )
                    if effect == 'lower' and category in (
                        SyntaxWarning,
                        DeprecationWarning,
                    ):
                        sys.set_int_max_str_digits(640)

                warnings.showwarning = show
                try:
                    request = Request(
                        context,
                        predict,
                        target='target',
                        task='classification'
                        if family in ('text-class', 'numeric-class')
                        else 'regression',
                    )
                    with SDMClient(
                        'http://localhost:1', max_retries=0
                    ) as client:
                        client._transport.close()
                        client._transport = transport
                        result = client.predict(request, model=KumoTabular())
                except Exception as exc:
                    failure = exc
                finally:
                    record['limit_at_end'] = sys.get_int_max_str_digits()
                    sys.set_int_max_str_digits(4300)
            # Observation happens outside the SDK exception block and after
            # restoring the limit. Wide-value formatting cannot fake a failure.
            record['ok'] = failure is None
            if failure is not None:
                record['error'] = error(failure)
            else:
                record['metadata'] = typed_value(result.metadata)
                record['items'] = [
                    {
                        name: typed_value(getattr(item, name))
                        for name in [
                            'row_index',
                            'id',
                            'forecast_step',
                            'prediction',
                            'probabilities',
                            'metadata',
                            'explanation',
                            'rankings',
                        ]
                    }
                    for item in result.items
                ]
            for body in transport.calls:
                body['metadata']['request_id'] = '<drawn>'
            record['bodies'] = transport.calls
            observed.append(record)
        assert observed[1] == observed[0]
        assert observed[1]['ok']
        assert len(observed[1]['bodies']) == 1
        assert observed[1]['limit_at_end'] == (
            640 if effect == 'lower' else 4300
        )
        assert len(observed[1]['events']) == (effect != 'none')
        first = observed[1]['items'][0]
        if family in ('metadata', 'combined'):
            assert observed[1]['metadata'] == typed_value(
                {'wide': wide, 'nested': [{'wide': -wide}]}
            )
        if family in ('prediction', 'numeric-class', 'combined'):
            assert first['prediction'] == ('int', wide)
        if family in ('forecast', 'combined'):
            assert first['forecast_step'] == ('int', wide)
        if family in ('item-mappings', 'combined'):
            assert first['metadata'] == typed_value({'wide': wide})
            assert first['explanation'] == typed_value({'wide': -wide})
            assert first['rankings'] == typed_value(({'wide': wide},))
        if family == 'text-class':
            # Fresh text admission catches ValueError and keeps the string;
            # this is a successful call, unlike a late carrier parse failure.
            expected = '0' * 650 + '2' if effect == 'lower' else 2
            assert first['prediction'] == typed_value(expected)
            assert first['probabilities'] == typed_value(
                {str(expected): 0.75, '4': 0.25}
            )
    finally:
        sys.set_int_max_str_digits(saved)


@pytest.mark.parametrize('api', ['request', 'schema'])
@pytest.mark.parametrize(
    'label,change,n,initial_repeat',
    [
        ('missing-baseline', 'none', 100, False),
        ('create-repeat', 'create', 100, False),
        ('remove-repeat', 'remove', 200, True),
        ('fill-missing', 'fill-missing', 100, False),
        ('no-remedy', 'no-remedy', 100, False),
        ('target-remedy', 'target-remedy', 100, False),
        ('unusable-remedy', 'unusable-remedy', 100, False),
        ('drop-nominee', 'drop-key', 100, False),
        ('empty-after-advice', 'empty', 100, False),
        ('nominee-retyped-bool', 'boolean-key', 100, False),
        ('after-repeat-count-fixed', 'after-repeat', 100, False),
    ],
)
def test_shared_inference_adoption_uses_reached_values_without_renomination(
    monkeypatch, api, label, change, n, initial_repeat
):
    observed = []
    for core in (False, True):
        monkeypatch.setenv(
            'SDM_CORE_ROUTES',
            (CORE if api == 'request' else FRAMES) if core else '',
        )
        observed.append(
            key_boundary(label, api, change, n=n, initial_repeat=initial_repeat)
        )
    assert comparable(observed[1]) == comparable(observed[0])
    assert observed[1]['ok']
    table = observed[1]['schema']['instance']
    assert table['key'] == 'user_id'
    assert table['columns']['user_id']['dtype'] == 'int64'
    assert table['columns']['user_id']['stype'] == 'ID'
    expected = (
        98
        if change == 'boolean-key'
        else 1
        if change
        in (
            'create',
            'no-remedy',
            'target-remedy',
            'unusable-remedy',
            'after-repeat',
        )
        else 0
    )
    assert table['duplicates'] == expected


@pytest.mark.parametrize(
    'change',
    [
        'time-null',
        'time-after-repeat',
        'time-drop',
        'time-numeric',
        'time-added',
    ],
)
def test_shared_time_choice_observes_after_adoption_advice(monkeypatch, change):
    observed = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', FRAMES if core else '')
        observed.append(
            key_boundary(change, 'schema', change, time=change != 'time-added')
        )
    assert comparable(observed[1]) == comparable(observed[0])
    if change == 'time-drop':
        assert observed[1]['error']['type'] == 'KeyError'
    else:
        assert observed[1]['ok']
        assert observed[1]['schema']['instance']['time'] == (
            'created_at' if change == 'time-numeric' else None
        )


@pytest.mark.parametrize('as_error', [False, True])
def test_adoption_warning_effects_precede_later_constructor_guards(
    monkeypatch, as_error
):
    observed = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', CORE if core else '')
        observed.append(
            key_boundary('chain', 'request', 'chained', error_warning=as_error)
        )
    assert comparable(observed[1]) == comparable(observed[0])
    assert not observed[1]['ok']
    if as_error:
        assert observed[1]['error']['type'] == 'SDMWarning'
        assert not observed[1]['events']
    else:
        assert len(observed[1]['events']) == 2
        assert (
            "missing column(s) ['feature']" in observed[1]['error']['message']
        )


@pytest.mark.parametrize(
    'mode',
    [
        'unowned-column',
        'borrowed-column',
        'borrowed-former-key',
        'unowned-table',
        'borrowed-table',
        'explicit-nonkey-ID',
        'release-key',
        'bad-nonkey',
        'overlap-replacement',
        'unknown-model-before-adoption',
    ],
)
def test_direct_consistency_adopts_copies_marks_and_releases_native_metadata(
    monkeypatch, mode
):
    observed = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', CORE if core else '')
        record = ownership_boundary(mode)
        for body in record['bodies']:
            body['metadata']['request_id'] = '<drawn>'
        observed.append(record)
    assert comparable(observed[1]) == comparable(observed[0])
    record = observed[1]
    if mode in ('bad-nonkey', 'unknown-model-before-adoption'):
        assert not record['ok'] and not record['bodies']
    else:
        assert record['ok'] and len(record['bodies']) == 1
        assert record['identity']['live_table_schema_ref']
        assert (
            record['after']['instance']['columns']['user_id']['stype'] == 'ID'
        )
    if mode.startswith('borrowed') or mode == 'release-key':
        assert not record['identity']['same_inserted_column']


@pytest.mark.parametrize('label,change', RESPONSE_CASES)
def test_direct_generated_response_field_family_and_error_order(
    monkeypatch, label, change
):
    observed = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', CORE if core else '')
        record = response_boundary(label, change)
        for body in record['bodies']:
            body['metadata']['request_id'] = '<drawn>'
        observed.append(record)
    assert comparable(observed[1]) == comparable(observed[0])
    assert len(observed[1]['bodies']) == 1
    if label.startswith('forecast-step-wide') or label.startswith('valid-'):
        assert observed[1]['ok']
    else:
        assert not observed[1]['ok']
        assert observed[1]['error']['type'] == 'StructuredDataError'
        assert observed[1]['error']['code'] == 'INVALID_RESPONSE'
    if label.startswith(
        ('item-rankings-', 'item-explanation-', 'item-metadata-')
    ):
        cause = observed[1]['error']['cause']
        assert cause['module'] == 'builtins'
        assert cause['type'] in ('TypeError', 'ValueError')
        assert observed[1]['error']['message'].endswith(cause['message'])
    if label.startswith('item-id-') or label in (
        'forecast-step-string',
        'forecast-step-bool',
        'forecast-step-fraction',
        'forecast-step-infinity',
        'paired-id-row_index',
    ):
        assert observed[1]['error']['cause']['message'].startswith(
            'prediction response has '
        )
    if label == 'global-late-field-before-ID-warning':
        assert not observed[1]['events']


@pytest.mark.parametrize(
    'field,value,cause_name',
    [
        ('id', 4, 'ResponseContractError'),
        ('forecast_step', 'bad', 'ResponseContractError'),
        ('explanation', 4, 'TypeError'),
        ('rankings', 4, 'TypeError'),
        ('metadata', [['x']], 'ValueError'),
        ('row_index', 2, None),
    ],
)
@pytest.mark.parametrize('active', [False, True])
def test_public_predict_run_loop_keeps_selected_causes(
    monkeypatch, field, value, cause_name, active
):
    from sdm_client.errors import StructuredDataError

    observed = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', CORE if core else '')
        context = pd.DataFrame({'user_id': [1, 2, 3], 'target': [2, 4, 6]})
        scoring = pd.DataFrame({'user_id': [4, 5]})
        request = Request(context, scoring, target='target', task='regression')
        transport = Scripted(item({field: value}))

        def call():
            with SDMClient('http://localhost:1', max_retries=0) as client:
                client._transport.close()
                client._transport = transport
                with pytest.raises(StructuredDataError) as caught:
                    client.predict(request, model=KumoTabular())
            return caught.value

        caller = None
        if active:
            try:
                raise LookupError('outer-active-caller')
            except LookupError as outer:
                caller = outer
                failure = call()
        else:
            failure = call()
        observed.append(error(failure))
        assert failure.code == 'INVALID_RESPONSE'
        assert len(transport.calls) == 1
        cause = failure.__cause__
        if cause_name is None:
            assert cause is None
        else:
            assert type(cause).__name__ == cause_name
            assert failure.message.endswith(str(cause))
            assert failure.__context__ is cause
            assert cause.__cause__ is None
            assert cause.__context__ is caller
            assert not cause.__suppress_context__
            assert cause.__traceback__ is not None
        if core:
            assert failure.__suppress_context__
    assert observed[1] == observed[0]


def later_schema_boundary(mode):
    events = []
    if mode.startswith('override'):
        left = pd.DataFrame({'code': range(100), 'feature': range(100, 200)})
        left.loc[2, 'code'] = 1
        right = pd.DataFrame({'code': range(100), 'feature': range(100, 200)})
        frames = {'left': left, 'right': right}
        options = {
            'primary_keys': {'left': 'code', 'right': 'code'},
            'relationships': [],
        }
    else:
        labels = pd.Series([f'A{n:02}' for n in range(40)], dtype='string')
        other = pd.Series([f'B{n % 32:02}' for n in range(40)], dtype='string')
        frame = pd.DataFrame({'label_a': labels, 'label_b': other})
        if mode == 'date-advice-before-category':
            frame['label_a'] = pd.Series(
                pd.date_range('2026-01-01', periods=40).strftime('%Y-%m-%d'),
                dtype='string',
            )
        frames = {'instance': frame}
        options = {'relationships': []}

    def show(message, category, filename, lineno, file=None, line=None):
        events.append(
            observe(
                lambda: {'type': category.__name__, 'message': str(message)}
            )
        )
        if len(events) == 1:
            if mode == 'override-mutates-later':
                right.loc[[2, 3], 'code'] = 1
            if mode == 'category-advice-mutates-later':
                frame['label_b'] = pd.Series(['constant'] * 40, dtype='string')
            if mode == 'date-advice-before-category':
                frame.drop(columns=['label_b'], inplace=True)

    with warnings.catch_warnings():
        warnings.simplefilter('always')
        warnings.showwarning = show
        try:
            schema = Schema.from_data(frames, **options)
        except ObservationFailure:
            raise
        except Exception as failure:
            record = {'case': mode, 'ok': False, 'error': error(failure)}
        else:
            record = {
                'case': mode,
                'ok': True,
                'schema': observe(lambda: schema_state(schema)),
            }
    record['events'] = events
    record['frames'] = observe(
        lambda: {
            name: {
                'columns': list(frame.columns),
                'dtypes': [str(dtype) for dtype in frame.dtypes],
                'values': frame.to_dict(orient='list'),
            }
            for name, frame in frames.items()
        }
    )
    return record


@pytest.mark.parametrize(
    'mode',
    [
        'override-baseline',
        'override-mutates-later',
        'category-advice-mutates-later',
        'date-advice-before-category',
    ],
)
def test_schema_reads_after_later_override_and_final_advice(monkeypatch, mode):
    outcomes = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', FRAMES if core else '')
        outcomes.append(later_schema_boundary(mode))
    assert comparable(outcomes[1]) == comparable(outcomes[0])
    assert outcomes[1]['ok'] is (mode != 'override-mutates-later')
    assert len(outcomes[1]['events']) == (
        2 if mode == 'date-advice-before-category' else 1
    )


@pytest.mark.parametrize(
    'field', ['probabilities', 'scores', 'embeddings', 'quantiles']
)
@pytest.mark.parametrize(
    'value,native_type',
    [
        ('not-a-number', ValueError),
        (None, TypeError),
        (10**400, OverflowError),
        ('NaN', None),
        (True, None),
    ],
)
@pytest.mark.parametrize('active', [False, True])
def test_public_finite_response_keeps_actual_nested_conversion_context(
    monkeypatch, field, value, native_type, active
):
    # Frozen generated _finite wraps a failed native float conversion from
    # None; rejection after a successful conversion has no such inner cause.
    from sdm_client.errors import StructuredDataError

    observed = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', CORE if core else '')
        context = pd.DataFrame({'user_id': [1, 2, 3], 'target': [2, 4, 6]})
        scoring = pd.DataFrame({'user_id': [4, 5]})
        request = Request(context, scoring, target='target', task='regression')
        member = (
            {'x': value} if field in ('probabilities', 'quantiles') else [value]
        )
        transport = Scripted(item({field: member}))

        def call():
            with SDMClient('http://localhost:1', max_retries=0) as client:
                client._transport.close()
                client._transport = transport
                with pytest.raises(StructuredDataError) as caught:
                    client.predict(request, model=KumoTabular())
            return caught.value

        caller = None
        if active:
            try:
                raise LookupError('outer-active-caller')
            except LookupError as outer:
                caller = outer
                failure = call()
        else:
            failure = call()
        assert len(transport.calls) == 1
        assert failure.code == 'INVALID_RESPONSE'
        generated = failure.__cause__
        assert type(generated).__name__ == 'ResponseContractError'
        assert failure.__context__ is generated
        assert failure.__suppress_context__
        assert generated.__cause__ is None
        assert generated.__traceback__ is not None
        if native_type is None:
            assert not generated.__suppress_context__
            assert generated.__context__ is caller
            nested = None
        else:
            assert generated.__suppress_context__
            nested = generated.__context__
            assert type(nested) is native_type
            assert nested.__cause__ is None
            assert not nested.__suppress_context__
            assert nested.__context__ is caller
            assert nested.__traceback__ is not None
        observed.append(
            (error(failure), None if nested is None else str(nested))
        )
    assert observed[1] == observed[0]


@pytest.mark.parametrize(
    'storage', ['float32', 'Float32', 'float64', 'Float64']
)
@pytest.mark.parametrize('two', [False, True])
def test_fresh_time_candidates_preserve_physical_float_width(
    monkeypatch, storage, two
):
    observed = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', FRAMES if core else '')
        frame = pd.DataFrame(
            {
                'user_id': pd.Series([None, *range(1, 100)], dtype='Int64'),
                'event_a': pd.date_range('2024-01-01', periods=100),
            }
        )
        if two:
            frame['event_b'] = pd.date_range('2025-01-01', periods=100)
        events = []

        def show(message, category, filename, lineno, file=None, line=None):
            events.append((str(message), category))
            if len(events) == 1:
                offsets = np.arange(100, dtype='float64') * 86400e9
                frame['event_a'] = pd.Series(1.7e18 + offsets, dtype=storage)
                if two:
                    frame['event_b'] = 1.8e18 + offsets

        with warnings.catch_warnings():
            warnings.simplefilter('always')
            warnings.showwarning = show
            schema = Schema.from_data({'events': frame}, relationships=())
        observed.append(
            (schema['events'].time_column, schema_state(schema), events)
        )
        assert str(frame['event_a'].dtype) == storage
    assert observed[1] == observed[0]
    assert observed[1][0] == 'event_a'
    assert len(observed[1][2]) == 1


@pytest.mark.parametrize(
    'dtype,prediction,key,expected,expected_key',
    [
        ('int64', '10', '10', 10, '10'),
        ('int64', '1_0', '1_0', 10, '10'),
        ('int64', '١٠', '١٠', 10, '10'),
        ('int64', 9.19e18, '1', 9190000000000000000, '1'),
        ('int64', 9.21e18, '1', 9210000000000000000, '1'),
        ('int64', -9.21e18, '1', -9210000000000000000, '1'),
        ('float64', '10.5', '10.5', 10.5, '10.5'),
        ('float64', '1_0.5', '1_0.5', 10.5, '10.5'),
        ('float64', '١٠.5', '١٠.5', 10.5, '10.5'),
    ],
)
def test_direct_classes_retain_native_numeric_grammar_separately_from_gathered(
    monkeypatch, dtype, prediction, key, expected, expected_key
):
    observed = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', CORE if core else '')
        context = pd.DataFrame(
            {'user_id': [1, 2, 3], 'target': pd.Series([2, 4, 6], dtype=dtype)}
        )
        scoring = pd.DataFrame({'user_id': [4, 5]})
        request = Request(
            context, scoring, target='target', task='classification'
        )
        transport = Scripted(
            item({'prediction': prediction, 'probabilities': {key: 0.75}})
        )
        with SDMClient('http://localhost:1', max_retries=0) as client:
            client._transport.close()
            client._transport = transport
            result = client.predict(request, model=KumoTabular())
        assert len(transport.calls) == 1
        first = result.items[0]
        assert type(first.prediction) is type(expected)
        assert first.prediction == expected
        assert first.probabilities == {expected_key: 0.75}
        observed.append((typed_value(first.prediction), first.probabilities))
    assert observed[1] == observed[0]
