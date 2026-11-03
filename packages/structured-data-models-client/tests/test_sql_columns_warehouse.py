# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only calculated SQL values, semantic-view refresh and live prediction."""

import importlib
import json
import os
from pathlib import Path

import pytest

from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient

pytestmark = [pytest.mark.live_warehouse, pytest.mark.live_nim]


@pytest.mark.parametrize(
    'connector',
    [
        'postgres',
        'databricks-sql',
        'snowflake',
        'databricks-metric',
        'snowflake-semantic',
    ],
)
def test_live_calculated_column_values_refresh_and_prediction(connector):
    path = os.environ.get('SDM_SQL_COLUMN_CONFIG')
    if not path:
        pytest.skip(
            'set SDM_SQL_COLUMN_CONFIG to private source/endpoint config'
        )
    from sdm_connectors import DatabricksSource, PostgresSource, SnowflakeSource

    config = json.loads(Path(path).read_text())
    settings = config['sources'][connector]
    driver, source_type = (
        ('psycopg', PostgresSource)
        if connector == 'postgres'
        else ('databricks.sql', DatabricksSource)
        if connector.startswith('databricks')
        else ('snowflake.connector', SnowflakeSource)
    )
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / 'tests/parity/fixtures/shop.json'
        ).read_text()
    )
    expected = {
        row[0]: row[1] * 2 for row in fixture['tables']['users']['rows']
    }
    with importlib.import_module(driver).connect(
        **settings['connect']
    ) as connection:
        source = source_type(connection, **settings.get('source', {}))
        if connector in ('databricks-metric', 'snowflake-semantic'):
            source = source.semantic_model(
                settings['view'], trust_expressions=False
            )
        key, value = (
            ('ID', 'VALUE')
            if connector == 'snowflake-semantic'
            else ('id', 'value')
        )
        schema = Schema.from_source(
            source,
            {'users': settings['tables']['users']},
            primary_keys={'users': key},
            relationships=[],
        )
        quote = '`' if connector.startswith('databricks') else '"'
        changed = schema.with_sql_column(
            'users', 'doubled', f'{quote}{value}{quote} * 2'
        )
        assert 'doubled' not in schema['users'].columns
        changed['users']['doubled'].stype = 'numerical'
        query = PredictiveQuery(
            f'PREDICT users.doubled FOR EACH users.{key}', changed
        )
        for _ in range(2):
            labels = query.labels(100)
            assert (
                dict(zip(labels.ENTITY, labels.TARGET, strict=True)) == expected
            )
        request = query.materialize(
            indices=[0, 1],
            num_context_samples=12,
            random_seed=42,
            model=KumoRelational(),
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
