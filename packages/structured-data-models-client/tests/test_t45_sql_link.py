# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public SQL link journeys on the legacy and core routes, never a live model."""

from __future__ import annotations

import copy
import importlib.util
import json
import warnings
from pathlib import Path

import duckdb
import jsonschema
import pandas as pd
import pytest
import yaml

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Schema,
    SDMClient,
    StructuredDataError,
)
from sdm_client._routing import CORE_ROUTED
from sdm_client.errors import NimRequestError
from sdm_connectors import DuckDBSource

ROOT = Path(__file__).resolve().parents[3]
QUERY = (
    'PREDICT LIST_DISTINCT(buys.item_id, 0, 30, days) FOR EACH users.user_id'
)
RANKINGS = [{'item': 3, 'score': 0.9}, {'item': 1, 'score': 0.4}]


class Scripted:
    serves_sessions = True
    url = 'scripted://t45-sql-link'

    def __init__(self, mutation=None):
        self.calls = []
        self.mutation = mutation

    def predict(self, payload):
        self.calls.append(copy.deepcopy(payload))
        table = payload['predict']['instance_table']
        id_at = table['columns'].index('instance_id')
        response = {
            'id': 't45-sql-link',
            'model': 'kumo-relational',
            'metadata': {},
            'predictions': [
                {
                    'row_index': i,
                    'id': str(row[id_at]),
                    'prediction': None,
                    'rankings': copy.deepcopy(RANKINGS),
                }
                for i, row in enumerate(table['rows'])
            ],
        }
        if self.mutation:
            self.mutation(response)
        return response

    def close(self):
        pass


def connection(extra=()):
    case = json.loads(
        (
            ROOT / 'contract/cases/s1/target.aggregation.list-distinct.json'
        ).read_text()
    )
    bases = yaml.safe_load((ROOT / 'contract/bases.yaml').read_text())
    db = duckdb.connect()
    for statement in (
        bases['bases']['shop'] + case['input']['setup'] + list(extra)
    ):
        db.execute(statement)
    return db


def materialize(db, query=QUERY, **options):
    schema = Schema.from_source(DuckDBSource(db))
    return PredictiveQuery(query, schema).materialize(
        anchor_time=pd.Timestamp('2023-03-01'), **{'indices': [0, 1], **options}
    )


def predict(request, script):
    with SDMClient('http://localhost:1', max_retries=0) as client:
        client._transport.close()
        client._transport = script
        return client.predict(request, model=KumoRelational()).to_pandas()


def route(monkeypatch, core):
    monkeypatch.setenv(
        'SDM_CORE_ROUTES', ','.join(sorted(CORE_ROUTED)) if core else ''
    )


@pytest.mark.parametrize('reverse', [False, True])
def test_both_public_sql_directions_match_frozen_labels_and_decode_rankings(
    monkeypatch, reverse
):
    route(monkeypatch, True)
    query = (
        'PREDICT LIST_DISTINCT(buys.user_id, 0, 30, days) FOR EACH items.item_id'
        if reverse
        else QUERY
    )
    with connection() as db:
        request = materialize(db, query)
        payload = request.payloads[0]
        rhs, key = ('users', 'user_id') if reverse else ('items', 'item_id')
        related = payload['schema']['related_tables']
        assert related[rhs]['primary_key'] == ['instance_id', key]
        assert related[rhs]['columns'][key]['dtype'] == 'int64'
        assert '__node_id' not in related[rhs]['columns']
        assert related['buys']['primary_key'] == ['instance_id', '__node_id']
        assert {
            'source_columns': ['__kumo_entity_ref_1'],
            'target_table': rhs,
            'target_columns': [key],
        } in payload['schema']['relationships']
        vectors = json.loads(
            (ROOT / 'contract/t210-target-vectors.json').read_text()
        )
        assert (
            vectors['reference'] == '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1'
        )
        name = (
            'aggregation.list-distinct'
            + ('.users' if reverse else '')
            + '@duckdb'
        )
        expected = vectors['vectors'][name]['payloads'][0]
        assert payload['task'] == expected['task']
        for half in ('context', 'predict'):
            assert payload[half]['instance_table'] == expected[half]
        script = Scripted()
        answer = predict(request, script)
    assert len(script.calls) == 1
    assert answer['id'].tolist() == [0, 1]
    assert answer['rankings'].tolist() == [RANKINGS, RANKINGS]
    if not reverse:
        assert (
            payload['schema']['instance_table']['columns'][
                '__kumo_entity_ref_1'
            ]['dtype']
            == 'float64'
        )
        rows = payload['context']['instance_table']['rows']
        assert rows[1][2] is None and rows[1][4] == ['1']
        assert rows[5][2] is None and rows[5][4] == ['2']


@pytest.mark.parametrize(
    'query,reason',
    [
        (
            'PREDICT LIST_DISTINCT(buys.user_id, 0, 30, days) FOR EACH users.user_id',
            'creates a loop by pointing to the entity table',
        ),
        (
            'PREDICT LIST_DISTINCT(orders.price, 0, 30, days) FOR EACH users.user_id',
            'Foundation Model queries do not support multilabel tasks',
        ),
    ],
)
def test_public_sql_refuses_loop_and_non_fk_targets_before_payload_construction(
    monkeypatch, query, reason
):
    # Genuine frozen43e public observations reject both inputs in validation,
    # before the internal alias writer and non-FK task diagnostic are reached.
    observed = []
    for core in (False, True):
        route(monkeypatch, core)
        with connection() as db:
            with pytest.raises(StructuredDataError) as error:
                materialize(db, query)
        assert error.value.code == 'INVALID_REQUEST'
        assert reason in str(error.value)
        observed.append((error.value.code, str(error.value)))
    assert observed[1] == observed[0]


@pytest.mark.parametrize(
    'count,limit,sizes',
    [(200, None, [200]), (201, None, [200, 1]), (201, 73, [73, 73, 55])],
)
def test_link_calls_keep_the_200_row_cap_and_repeated_public_entities(
    monkeypatch, count, limit, sizes
):
    indices = [i % 8 for i in range(count)]
    observed = []
    for core in (False, True):
        route(monkeypatch, core)
        with connection() as db:
            options = {'indices': indices}
            if limit is not None:
                options['rows_per_call'] = limit
            request = materialize(db, **options)
            script = Scripted()
            answer = predict(request, script)
        assert [
            len(c['predict']['instance_table']['rows']) for c in script.calls
        ] == sizes
        assert answer['id'].tolist() == indices
        assert answer['rankings'].tolist() == [RANKINGS] * count
        observed.append(request.payloads)
    assert observed[1] == observed[0]


@pytest.mark.parametrize(
    'expression',
    [
        "'key-' || CAST(item_id AS VARCHAR)",
        'item_id + 9007199254740993',
        'CAST(item_id AS DOUBLE) * 0.00000001 + 1.0',
    ],
)
def test_rhs_keys_keep_the_reference_text_large_integer_and_float_identity(
    monkeypatch, expression
):
    outputs = []
    for core in (False, True):
        route(monkeypatch, core)
        extra = [
            f'ALTER TABLE {table} ALTER COLUMN item_id TYPE '
            + (
                'VARCHAR'
                if expression.startswith("'key-")
                else 'DECIMAL(38,2)'
                if 'DECIMAL' in expression
                else 'DOUBLE'
                if 'DOUBLE' in expression
                else 'BIGINT'
            )
            + f' USING {expression}'
            for table in ('items', 'buys')
        ]
        with connection(extra) as db:
            request = materialize(db)
            assert predict(request, Scripted())['id'].tolist() == [0, 1]
            outputs.append(request.payloads)
    assert outputs[1] == outputs[0]


def test_decimal_rhs_identity_keeps_the_existing_approved_exact_key_rule(
    monkeypatch,
):
    # P2B-1 already requires exact decimal keys instead of legacy rounding.
    # A blanket old/core payload equality assertion here would undo that rule.
    register = yaml.safe_load(
        (ROOT / 'docs/design/rust-core/register.yaml').read_text()
    )
    entry = next(e for e in register['entries'] if e['id'] == 'P2B-1')
    assert (entry['status'], entry['approved_by']) == ('approved', 'owner')
    expression = 'CAST(9007199254740993.25 AS DECIMAL(38,2)) + CAST(item_id AS DECIMAL(38,2))'
    setup = [
        f'ALTER TABLE {table} ALTER COLUMN item_id TYPE DECIMAL(38,2) USING {expression}'
        for table in ('items', 'buys')
    ]
    route(monkeypatch, False)
    with connection(setup) as db:
        try:
            legacy = materialize(db).payloads
        except StructuredDataError as error:
            assert error.code == 'INVALID_REQUEST'
            assert any(
                word in str(error).lower()
                for word in ('key', 'duplicat', 'unique')
            )
            legacy = None
    route(monkeypatch, True)
    with connection(setup) as db:
        request = materialize(db)
        payload = request.payloads[0]
        columns = payload['schema']['instance_table']['columns']
        assert columns['__kumo_entity_ref_1']['dtype'] == 'string'
        assert (
            payload['schema']['related_tables']['items']['columns']['item_id'][
                'dtype'
            ]
            == 'string'
        )
        expected = {f'900719925474099{i}.25' for i in range(3, 7)}
        labels = payload['context']['instance_table']
        at = labels['columns'].index('TARGET')
        assert all(set(row[at]) <= expected for row in labels['rows'])
        assert any(row[at] for row in labels['rows'])
        if legacy is not None:
            assert (
                legacy != request.payloads
            )  # The recorded rule preserves extra precision.
        assert predict(request, Scripted())['id'].tolist() == [0, 1]


@pytest.mark.parametrize('lag_timesteps', [1, 1.5, True])
def test_link_labels_do_not_gain_regression_lag_features(
    monkeypatch, lag_timesteps
):
    outputs = []
    for core in (False, True):
        route(monkeypatch, core)
        with connection() as db:
            request = materialize(db, lag_timesteps=lag_timesteps)
            outputs.append(request.payloads)
    assert outputs[1] == outputs[0]
    assert not any(
        name.startswith('__kumo_arl')
        for name in outputs[1][0]['context']['instance_table']['columns']
    )


def test_distinct_labels_keep_executor_order_null_filter_and_window_boundaries(
    monkeypatch,
):
    setup = [
        "INSERT INTO buys VALUES (999000, 0, 0, TIMESTAMP '2023-01-30'), "
        "(999001, 0, 1, TIMESTAMP '2023-02-01'), "
        "(999002, 0, 1, TIMESTAMP '2023-02-02'), "
        "(999003, 0, NULL, TIMESTAMP '2023-02-03'), "
        "(999004, 0, 3, TIMESTAMP '2023-03-01'), "
        "(999005, 0, 2, TIMESTAMP '2023-03-01 00:00:00.000001')",
    ]
    query = 'PREDICT LIST_DISTINCT(buys.item_id WHERE buys.buy_id >= 999000, 0, 30, days) FOR EACH users.user_id'
    outputs = []
    for core in (False, True):
        route(monkeypatch, core)
        with connection(setup) as db:
            request = materialize(db, query)
            outputs.append(request.payloads)
    assert outputs[1] == outputs[0]
    table = outputs[0][0]['context']['instance_table']
    labels = [r[table['columns'].index('TARGET')] for r in table['rows']]
    assert labels
    assert all(
        len(value) == len(set(value)) and set(value) <= {'1', '3'}
        for value in labels
    )
    assert any(set(value) == {'1', '3'} for value in labels)


@pytest.mark.parametrize('mutation', ['rankings', 'id', 'row_index'])
def test_link_replies_keep_existing_malformed_and_attribution_guards(
    monkeypatch, mutation
):
    route(monkeypatch, True)

    def corrupt(response):
        row = response['predictions'][0]
        row[mutation] = {'rankings': 3, 'id': 'another-row', 'row_index': 999}[
            mutation
        ]

    with connection() as db:
        request = materialize(db)
        with pytest.raises(StructuredDataError) as error:
            predict(request, Scripted(corrupt))
    assert error.value.code == 'INVALID_RESPONSE'


def test_an_absent_ranking_output_keeps_the_reference_notice(monkeypatch):
    outputs = []
    for core in (False, True):
        route(monkeypatch, core)

        def omit(response):
            for row in response['predictions']:
                del row['rankings']

        with connection() as db:
            request = materialize(db)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                answer = predict(request, Scripted(omit))
            outputs.append(
                (answer.columns.tolist(), [str(n.message) for n in notices])
            )
    assert outputs[1] == outputs[0]
    assert any('rankings' in notice for notice in outputs[0][1])


@pytest.mark.parametrize(
    'query,options',
    [
        (QUERY, {'num_hops': 1}),
        ('PREDICT LIST_DISTINCT(buys.item_id) FOR EACH users.user_id', {}),
    ],
)
def test_public_sql_keeps_missing_history_and_static_refusals(
    monkeypatch, query, options
):
    errors = []
    for core in (False, True):
        route(monkeypatch, core)
        with connection() as db:
            with pytest.raises(StructuredDataError) as error:
                materialize(db, query, **options)
        errors.append((error.value.code, str(error.value)))
    assert errors[1] == errors[0]


def test_public_original_endpoint_keeps_its_no_classes_refusal(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        't45_endpoint', ROOT / 'contract/runners/scripted_endpoint.py'
    )
    assert spec is not None and spec.loader is not None
    endpoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(endpoint)

    class Original(Scripted):
        def predict(self, payload):
            self.calls.append(copy.deepcopy(payload))
            status, body = endpoint.answer(
                'POST', endpoint.PREDICTIONS, json.dumps(payload).encode()
            )
            assert (status, body) == (400, {'error': 'no_classes'})
            raise NimRequestError(status, code=None, message=json.dumps(body))

    errors = []
    for core in (False, True):
        route(monkeypatch, core)
        script = Original()
        with connection() as db:
            request = materialize(db)
            with pytest.raises(NimRequestError) as error:
                predict(request, script)
        errors.append(
            (error.value.status_code, error.value.code, str(error.value))
        )
        assert len(script.calls) == 1
    assert errors[1] == errors[0]


def test_link_task_and_base_config_are_closed_typed_payload_variants(
    monkeypatch,
):
    route(monkeypatch, True)
    with connection() as db:
        payload = materialize(db).payloads[0]
    schema = json.loads(
        (
            ROOT / 'rust/crates/sdm-contract/schema/state-payload.json'
        ).read_text()
    )
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(payload)
    assert 'top_k' not in payload['task']
    # The existing task omission convention admits no anchor where the
    # materializer names none, without adding a static-link task kind.
    no_anchor = copy.deepcopy(payload)
    del no_anchor['task']['anchor_time_column']
    validator.validate(no_anchor)
    for path, key, value in [
        (('task',), 'kind', 'static_link_prediction'),
        (('task', 'target'), 'dtype', 'int64'),
        (('task', 'target'), 'classes', ['invented']),
        (('inference', 'inference_config'), 'class_shuffle', False),
    ]:
        corrupted = copy.deepcopy(payload)
        target = corrupted
        for part in path:
            target = target[part]
        target[key] = value
        assert list(validator.iter_errors(corrupted)), (path, key)
