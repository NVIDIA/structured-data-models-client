# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from sdm_client import RelationalTask, Relationship, Schema
from sdm_client.errors import StructuredDataError

FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / 'tests/fixtures/custom-examples/examples.json'
    ).read_text()
)


def frame(data):
    result = pd.DataFrame(data['rows'], columns=data['columns'])
    for name in ['opened', 'closed', 'time', 'ANCHOR_TIMESTAMP']:
        if name in result:
            result[name] = pd.to_datetime(result[name])
    return result


def schema_for(case):
    timeless = case.get('schema_variant') == 'timeless'
    tables = {
        name: frame(data)
        for name, data in FIXTURE[
            'timeless_tables' if timeless else 'tables'
        ].items()
    }
    schema = Schema.from_data(tables)
    for name in tables:
        schema[name].primary_key = 'id'
    if not timeless:
        for name, column in FIXTURE['time_columns'].items():
            schema[name].time_column = column
        for name, column in FIXTURE['end_time_columns'].items():
            schema[name].end_time_column = column
        schema.relationships = [
            Relationship(*r) for r in FIXTURE['relationships']
        ]
    return schema


def column(table, name):
    index = table['columns'].index(name)
    return [row[index] for row in table['rows']]


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda c: c['id'])
def test_shared_examples_labels_order_ownership_and_persistence(case):
    schema = schema_for(case)
    context, predict = frame(case['context']), frame(case['predict'])
    request = schema.materialize_examples(
        RelationalTask(**case['task']), context, predict, num_neighbors=[10, 10]
    )
    payload = request.payloads[0]
    expected = case['expected']['context_targets']
    if case['task']['kind'] == 'temporal_link_prediction':
        expected = [[str(v) for v in row] for row in expected]
    assert column(payload['context']['instance_table'], 'TARGET') == expected
    assert len(payload['context']['instance_table']['rows']) == len(context)
    assert payload['inference']['use_prediction_time'] is False
    before = json.dumps(request.payloads, sort_keys=True)
    context.iloc[0, 0] = 999
    predict.iloc[0, 0] = 999
    assert json.dumps(request.payloads, sort_keys=True) == before
    restored = pickle.loads(pickle.dumps(request))
    assert restored.payloads == request.payloads


@pytest.mark.parametrize('large', [False, True])
def test_sliced_link_targets_ignore_hidden_backing_values(large):
    case = next(
        c for c in FIXTURE['cases'] if c['id'] == 'temporal_link_prediction'
    )
    context = pa.Table.from_pandas(frame(case['context']), preserve_index=False)
    dtype = pa.large_list(pa.int64()) if large else pa.list_(pa.int64())
    targets = pa.array(
        [[999]] + context['TARGET'].to_pylist() + [[999]], type=dtype
    )
    padded = pa.concat_tables(
        [context.slice(0, 1), context, context.slice(0, 1)]
    )
    padded = padded.set_column(
        padded.schema.get_field_index('TARGET'), 'TARGET', targets
    )
    request = schema_for(case).materialize_examples(
        RelationalTask(**case['task']),
        padded.slice(1, len(context)),
        frame(case['predict']),
        num_neighbors=[10, 10],
    )
    assert column(
        request.payloads[0]['context']['instance_table'], 'TARGET'
    ) == [[str(x) for x in row] for row in case['expected']['context_targets']]


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_labels_rejected(bad):
    case = FIXTURE['cases'][0]
    context = frame(case['context'])
    context.loc[0, 'TARGET'] = bad
    with pytest.raises(StructuredDataError) as error:
        schema_for(case).materialize_examples(
            RelationalTask(**case['task']), context, frame(case['predict'])
        )
    assert error.value.code != 'INTERNAL_ERROR'


def test_arrow_nonnull_nan_and_invalid_contracts():
    case = FIXTURE['cases'][0]
    schema = schema_for(case)
    context = pa.Table.from_pandas(frame(case['context']), preserve_index=False)
    predict = frame(case['predict'])
    bad = pa.array(
        [float('nan')] + context['TARGET'].to_pylist()[1:], from_pandas=False
    )
    assert bad.null_count == 0
    invalid = context.set_column(
        context.schema.get_field_index('TARGET'), 'TARGET', bad
    )
    for task, train, test, options in [
        (RelationalTask(**case['task']), invalid, predict, {}),
        (
            RelationalTask(**case['task']),
            context,
            predict.assign(ENTITY=999),
            {},
        ),
        (
            RelationalTask('regression', 'accounts', link_target_table='items'),
            context,
            predict,
            {},
        ),
        (
            RelationalTask('regression', 'accounts'),
            context,
            predict,
            {'num_neighbors': []},
        ),
        (
            RelationalTask(
                'forecasting', 'accounts', step_size_ns=0, num_forecasts=2
            ),
            context,
            predict,
            {},
        ),
        (
            RelationalTask(
                'forecasting', 'accounts', step_size_ns=1, num_forecasts=999
            ),
            context,
            predict,
            {},
        ),
        (
            RelationalTask(**case['task']),
            context.append_column('ENTITY', context['ENTITY']),
            predict,
            {},
        ),
    ]:
        with pytest.raises(StructuredDataError) as error:
            schema.materialize_examples(task, train, test, **options)
        assert error.value.code != 'INTERNAL_ERROR'


@pytest.mark.parametrize('provider', ['sqlite', 'duckdb'])
def test_sql_supplied_labels_and_repeated_entities(provider):
    if provider == 'duckdb':
        import duckdb
        from sdm_connectors import DuckDBSource

        connection = duckdb.connect()
        source = DuckDBSource(connection)
    else:
        import adbc_driver_sqlite.dbapi
        from sdm_connectors import SQLiteSource

        connection = adbc_driver_sqlite.dbapi.connect()
        source = SQLiteSource(connection)
    try:
        cursor = connection.cursor()
        cursor.execute(
            'CREATE TABLE accounts(id INTEGER PRIMARY KEY, feature DOUBLE)'
        )
        cursor.executemany(
            'INSERT INTO accounts VALUES (?, ?)',
            [(i, float(i)) for i in range(6)],
        )
        if provider == 'sqlite':
            connection.commit()
        cursor.close()
        schema = Schema.from_source(
            source,
            {'accounts': 'accounts'},
            primary_keys={'accounts': 'id'},
            relationships=[],
        )
        context = pd.DataFrame(
            {'ENTITY': [0, 0, 1, 2], 'TARGET': [2.25, 4.5, 8.75, 9.125]}
        )
        predict = pd.DataFrame({'ENTITY': [3, 3, 4]})
        request = schema.materialize_examples(
            RelationalTask('regression', 'accounts'), context, predict
        )
        assert (
            column(request.payloads[0]['context']['instance_table'], 'TARGET')
            == context.TARGET.tolist()
        )
        assert (
            len(request.payloads[0]['predict']['instance_table']['rows']) == 3
        )
    finally:
        connection.close()


@pytest.mark.live_nim
@pytest.mark.parametrize(
    'kind,horizon',
    [
        ('regression', None),
        ('binary_classification', None),
        ('multiclass_classification', None),
        ('temporal_link_prediction', None),
        ('forecasting', 3),
        ('forecasting', 1),
    ],
)
def test_live_supplied_examples(kind, horizon):
    import os
    from dataclasses import replace

    from sdm_client import KumoRelational, SDMClient

    path = os.environ.get('SDM_EXAMPLES_CONFIG')
    if not path:
        pytest.skip('set SDM_EXAMPLES_CONFIG for live endpoints')
    config = json.loads(Path(path).read_text())
    case = next(c for c in FIXTURE['cases'] if c['id'] == kind)
    task = RelationalTask(**case['task'])
    predictions = case['predict']
    if horizon is not None:
        task = replace(task, num_forecasts=horizon)
        predictions = case['additional_prediction_examples']
    request = schema_for(case).materialize_examples(
        task, frame(case['context']), frame(predictions), num_neighbors=[10, 10]
    )
    endpoint = (
        config['endpoints_by_case']['relational.forecast']['relational']
        if horizon is not None
        else config['endpoints_by_case']['relational.temporal-link'][
            'relational'
        ]
        if kind == 'temporal_link_prediction'
        else config['endpoints']['relational']
    )
    client = (
        SDMClient.for_databricks_serving
        if endpoint['transport'] == 'databricks'
        else SDMClient
    )(**endpoint['client'])
    try:
        result = client.predict(request, model=KumoRelational()).to_pandas()
        assert len(result) == len(predictions['rows']) * (horizon or 1)
        if horizon:
            assert result.forecast_step.tolist() == list(
                range(1, horizon + 1)
            ) * len(predictions['rows'])
        assert result.row_index.tolist() == [
            i
            for i in range(len(predictions['rows']))
            for _ in range(horizon or 1)
        ]
    finally:
        client.close()


def test_supplied_request_uses_existing_fit_predict_close_contract():
    from test_frame_sessions_core import Endpoint

    from sdm_client import KumoRelational, SDMClient

    case = FIXTURE['cases'][0]
    request = schema_for(case).materialize_examples(
        RelationalTask(**case['task']),
        frame(case['context']),
        frame(case['predict']),
    )
    endpoint = Endpoint()
    client = SDMClient.__new__(SDMClient)
    client._start(endpoint)
    try:
        fitted = client.fit(request, model=KumoRelational())
        try:
            assert len(fitted.predict(request)) == len(case['predict']['rows'])
        finally:
            fitted.close()
        assert endpoint.created and endpoint.deleted
    finally:
        client.close()


@pytest.mark.live_nim
@pytest.mark.live_warehouse
@pytest.mark.parametrize(
    'provider', ['postgres', 'databricks-sql', 'snowflake']
)
def test_live_warehouse_supplied_examples(provider):
    import importlib
    import os

    from sdm_connectors import DatabricksSource, PostgresSource, SnowflakeSource

    from sdm_client import KumoRelational, SDMClient

    path = os.environ.get('SDM_EXAMPLES_CONFIG')
    if not path:
        pytest.skip('set SDM_EXAMPLES_CONFIG for live endpoints')
    config = json.loads(Path(path).read_text())
    settings = config['sources'][provider]
    driver, source_type = {
        'postgres': ('psycopg', PostgresSource),
        'databricks-sql': ('databricks.sql', DatabricksSource),
        'snowflake': ('snowflake.connector', SnowflakeSource),
    }[provider]
    with importlib.import_module(driver).connect(
        **settings['connect']
    ) as connection:
        source = source_type(connection, **settings.get('source', {}))
        schema = Schema.from_source(
            source,
            {'users': settings['tables']['users']},
            primary_keys={'users': 'id'},
            relationships=[],
        )
        context = pd.DataFrame(
            {'ENTITY': list(range(12)), 'TARGET': [i * 2.25 for i in range(12)]}
        )
        predict = pd.DataFrame({'ENTITY': [12, 13]})
        request = schema.materialize_examples(
            RelationalTask('regression', 'users'), context, predict
        )
        assert (
            column(request.payloads[0]['context']['instance_table'], 'TARGET')
            == context.TARGET.tolist()
        )
        endpoint = config['endpoints']['relational']
        client = (
            SDMClient.for_databricks_serving
            if endpoint['transport'] == 'databricks'
            else SDMClient
        )(**endpoint['client'])
        try:
            assert len(client.predict(request, model=KumoRelational())) == 2
        finally:
            client.close()


def test_supplied_link_batches_obey_model_limit():
    case = next(
        c for c in FIXTURE['cases'] if c['id'] == 'temporal_link_prediction'
    )
    predicted = pd.concat(
        [frame(case['predict']).iloc[:1]] * 201, ignore_index=True
    )
    request = schema_for(case).materialize_examples(
        RelationalTask(**case['task']),
        frame(case['context']),
        predicted,
        rows_per_call=500,
    )
    assert [
        len(p['predict']['instance_table']['rows']) for p in request.payloads
    ] == [200, 1]
