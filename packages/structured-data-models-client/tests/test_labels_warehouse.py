# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only labels journeys against the maintained shop warehouse fixture."""

import importlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from sdm_client import PredictiveQuery, Relationship, Schema

pytestmark = pytest.mark.live_warehouse


@pytest.mark.parametrize('connector', ['databricks-sql', 'snowflake'])
def test_live_ground_truth_without_a_model(connector):
    config_path = os.environ.get('SDM_LABELS_WAREHOUSE_CONFIG')
    if not config_path:
        pytest.skip(
            'set SDM_LABELS_WAREHOUSE_CONFIG to a private source configuration'
        )
    from sdm_connectors import DatabricksSource, SnowflakeSource

    settings = json.loads(Path(config_path).read_text())['sources'][connector]
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
    tables = {
        name: pd.DataFrame(table['rows'], columns=table['columns'])
        for name, table in fixture['tables'].items()
    }
    tables['events']['event_time'] = pd.to_datetime(
        tables['events']['event_time']
    )
    anchor = pd.Timestamp('2024-02-01')
    end = anchor + pd.Timedelta(days=30)
    users = tables['users']
    events = tables['events']
    window = events[(events.event_time > anchor) & (events.event_time <= end)]
    expected_scalar = dict(zip(users.id, users.value, strict=True))
    expected_count = {key: int((window.id == key).sum()) for key in users.id}
    expected_links = {
        key: sorted(group.item.unique().tolist())
        for key, group in window.groupby('id')
    }
    with importlib.import_module(driver).connect(
        **settings['connect']
    ) as connection:
        source = source_type(connection, **settings.get('source', {}))
        schema = Schema.from_source(source, settings['tables'])
        for name, key in fixture['primary_keys'].items():
            schema[name].primary_key = key
        schema['events'].time_column = 'event_time'
        schema['users']['value'].stype = 'numerical'
        schema.relationships = [
            Relationship('events', 'id', 'users', 'id'),
            Relationship('events', 'item', 'items', 'item'),
        ]
        for target, expected in (
            ('users.value', expected_scalar),
            ('COUNT(events.*, 0, 30, days)', expected_count),
            ('LIST_DISTINCT(events.item, 0, 30, days)', expected_links),
        ):
            output = PredictiveQuery(
                f'PREDICT {target} FOR EACH users.id', schema
            ).labels(100, anchor_time=anchor)
            actual = {
                row.ENTITY: sorted(row.TARGET.tolist())
                if hasattr(row.TARGET, 'tolist')
                else row.TARGET
                for row in output.itertuples()
            }
            assert actual == expected
