# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repeated public materialization follows current SQL rows and view definitions."""

from __future__ import annotations

import base64
import importlib.util
import json
import re
import warnings
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pytest
from sdm_connectors import DatabricksSource, DuckDBSource

from sdm_client import PredictiveQuery, Schema

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    't310_sampler_warehouse',
    Path(__file__).with_name('test_databricks_from_source.py'),
)
assert _spec is not None and _spec.loader is not None
_warehouse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_warehouse)

QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'


class _MetricWarehouse(_warehouse._CompletingWarehouse):
    """Existing captured source rows, with live local SQL over changed views."""

    def __init__(self) -> None:
        super().__init__()
        case = json.loads(
            (ROOT / 'contract/cases/s2/view.metric.regression.json').read_text()
        )
        self.definition = case['input']['view']['yaml'].format(
            catalog='main', schema='shop'
        )
        self.definition = self.definition.replace(
            'source.price > 20', 'source.order_id < 5'
        )
        captures = json.loads(
            (ROOT / 'contract/view-captures.json').read_text()
        )
        recorded = next(
            c
            for c in captures['cases']
            if c['case'] == 'view.metric.regression'
            and c['leg'] == 'databricks-sql'
        )
        described = next(
            s
            for s in recorded['statements']
            if s['text'].startswith('DESCRIBE TABLE EXTENDED')
        )
        self.view_description = pa.ipc.open_stream(
            base64.b64decode(captures['tables'][described['result']])
        ).read_all()

    def answer(self, operation: str, parameters: list[Any]) -> pa.Table:
        text = _warehouse._normal(operation)
        if text.startswith('DESCRIBE TABLE EXTENDED'):
            self.calls.append((text, list(parameters)))
            rows = self.view_description.to_pylist()
            names = self.view_description.column_names
            for row in rows:
                if row[names[0]] == 'View Text':
                    row[names[1]] = self.definition
                elif row[names[0]] == 'Catalog':
                    row[names[1]] = 'main'
                elif row[names[0]] == 'Database':
                    row[names[1]] = 'shop'
            return pa.Table.from_pylist(
                rows, schema=self.view_description.schema
            )
        if 'information_schema' in text or 'ORDER BY p.o' in text:
            return super().answer(operation, parameters)
        self.calls.append((text, list(parameters)))
        if text.startswith('WITH TMP AS ( SELECT pos AS '):
            return _warehouse._seeded_over(self._db, text, parameters)
        local = re.sub(r'`main`\.`shop`\.', '', text).replace('`', '"')
        return self._db.execute(local, parameters).to_arrow_table()


_MetricConnection = type(
    'Connection', (_MetricWarehouse,), {'__module__': 'databricks.sql.client'}
)


def _payloads(query: PredictiveQuery) -> list[dict[str, Any]]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return query.materialize(indices=[0, 1], random_seed=7).payloads


def test_metric_filter_refresh_reuses_no_previous_sampler_bounds_or_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('SDM_CORE_ROUTES', raising=False)
    connection = _MetricConnection()
    try:
        source = DatabricksSource(
            connection, catalog='main', schema='shop'
        ).semantic_model('view_metric_regression')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(source)
        query = PredictiveQuery(QUERY, schema)
        first = _payloads(query)
        connection.definition = connection.definition.replace(
            'source.order_id < 5', 'source.order_id >= 0'
        )
        second = _payloads(query)
        fresh = _payloads(PredictiveQuery(QUERY, schema))
        assert first != fresh, (
            'the changed valid view must change actual model inputs'
        )
        assert second == fresh, (
            'an existing query must use the same current bounds/counts as a fresh materializer'
        )
    finally:
        connection._db.close()


def test_data_appended_past_unchanged_preview_refreshes_default_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('SDM_CORE_ROUTES', raising=False)
    connection = duckdb.connect()
    try:
        connection.execute(
            'CREATE TABLE users AS SELECT i AS user_id, 20 + i AS age FROM range(4) t(i)'
        )
        connection.execute(
            "CREATE TABLE orders AS SELECT i AS order_id, i % 4 AS user_id, i::DOUBLE AS price, TIMESTAMP '2024-01-01' + i * INTERVAL 1 HOUR AS ts FROM range(1100) t(i)"
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(
                DuckDBSource(connection),
                primary_keys={'users': 'user_id', 'orders': 'order_id'},
                time_columns={'orders': 'ts'},
            )
        query = PredictiveQuery(QUERY, schema)
        first = _payloads(query)
        preview = connection.execute(
            'SELECT * FROM orders LIMIT 1000'
        ).fetchall()
        connection.execute(
            "INSERT INTO orders VALUES (1100, 0, 99999, TIMESTAMP '2025-06-01')"
        )
        assert (
            connection.execute('SELECT * FROM orders LIMIT 1000').fetchall()
            == preview
        )
        second = _payloads(query)
        fresh = _payloads(PredictiveQuery(QUERY, schema))
        assert first != fresh
        assert second == fresh, (
            'same schema and preview do not make old time bounds current'
        )
    finally:
        connection.close()
