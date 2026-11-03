# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only expiry checks on existing warehouse fixtures and a real model."""

import importlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient

pytestmark = [pytest.mark.live_warehouse, pytest.mark.live_nim]


@pytest.mark.parametrize('connector', ['databricks-sql', 'snowflake'])
def test_live_end_time_selection_masking_and_prediction(connector):
    path = os.environ.get('SDM_END_TIME_CONFIG')
    if not path:
        pytest.skip('set SDM_END_TIME_CONFIG to private source/endpoint config')
    from sdm_connectors import DatabricksSource, SnowflakeSource

    config = json.loads(Path(path).read_text())
    settings = config['sources'][connector]
    driver, source_type = {
        'databricks-sql': ('databricks.sql', DatabricksSource),
        'snowflake': ('snowflake.connector', SnowflakeSource),
    }[connector]
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / 'tests/parity/fixtures/shop.json'
        ).read_text()
    )
    events = pd.DataFrame(
        fixture['tables']['events']['rows'],
        columns=fixture['tables']['events']['columns'],
    )
    anchor = pd.Timestamp('2024-02-01')
    eligible = set(
        events.loc[pd.to_datetime(events.event_time) > anchor, 'event']
    )
    with importlib.import_module(driver).connect(
        **settings['connect']
    ) as connection:
        source = source_type(connection, **settings.get('source', {}))
        schema = Schema.from_source(
            source,
            {'events': settings['tables']['events']},
            time_columns={'events': None},
        )
        schema['events'].primary_key = 'event'
        schema['events']['feature'].stype = 'numerical'
        schema['events']['event_time'].stype = 'timestamp'
        schema['events'].time_column = None
        schema['events'].end_time_column = 'event_time'
        query = PredictiveQuery(
            'PREDICT events.feature FOR EACH events.event', schema
        )
        labels = query.labels(200, anchor_time=anchor)
        assert set(labels.ENTITY) == eligible
        request = query.materialize(
            indices=[0, 40],
            anchor_time=anchor,
            num_context_samples=8,
            model=KumoRelational(),
        )
        payload = request.payloads[0]
        for side in ('context', 'predict'):
            table = payload[side]['related_tables']['events']
            rows = [
                dict(zip(table['columns'], row, strict=True))
                for row in table['rows']
            ]
            if side == 'context':
                assert {row['event'] for row in rows} <= eligible
                assert rows and all(row['event_time'] is None for row in rows)
            else:
                assert {row['event'] for row in rows} == {0, 40}
                assert all(
                    row['event_time'] is None
                    if row['event'] == 40
                    else str(row['event_time']).startswith('2024-01-01')
                    for row in rows
                )
        endpoint = config['endpoints']['relational']
        client = (
            SDMClient.for_databricks_serving(**endpoint['client'])
            if endpoint['transport'] == 'databricks'
            else SDMClient(**endpoint['client'])
        )
        try:
            result = client.predict(request, model=KumoRelational())
            assert len(result) == 2
        finally:
            client.close()
