# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import runpy
import warnings
from pathlib import Path

import pytest
from sdm_client import Schema, _impl
from sdm_client.errors import StructuredDataError
from sdm_connectors import DatabricksSource

_ROOT = Path(__file__).resolve().parents[2]
_PUBLIC = runpy.run_path(
    str(
        _ROOT
        / 'packages/structured-data-models-client/tests/test_a_metric_view_filter_holds.py'
    )
)
_Warehouse = _PUBLIC['_Warehouse']
_Cursor = _PUBLIC['_Cursor']


class _MetricCursor(_Cursor):
    def execute(self, sql, parameters=None):
        self._warehouse.statements.append(sql)
        assert 'information_schema.columns' in sql
        self._arrow = None
        self._rows = [
            (
                'region',
                'STRING',
                'YES',
                'cat',
                'sch',
                'sales_metrics',
                'METRIC_VIEW',
            )
        ]


class _MetricWarehouse(_Warehouse):
    def cursor(self):
        return _MetricCursor(self)


def test_public_metric_view_refuses_before_sampling(monkeypatch):
    diagnostics = []
    if not _impl.OLD:
        from sdm_client import _schema_core

        original = _schema_core._refused

        def observe(found, caused=None):
            diagnostics.append(found['error'])
            return original(found, caused)

        monkeypatch.setattr(_schema_core, '_refused', observe)
    warehouse = _MetricWarehouse()
    source = DatabricksSource(warehouse, catalog='cat', schema='sch')
    with pytest.raises(StructuredDataError) as caught:
        Schema.from_source(source, tables={'sales': 'Sales_Metrics'})
    error = caught.value
    assert error.code == 'INVALID_REQUEST'
    assert error.details == {'metric_view': 'Sales_Metrics'}
    assert error.message == (
        "'Sales_Metrics' is a metric view, not a table: its measures are "
        'aggregations with no value per row. Read it as a semantic model '
        "instead, with source.semantic_model('Sales_Metrics')."
    )
    if _impl.OLD:
        from sdm_relational_engine import MetricViewAsTableError

        assert isinstance(error.__cause__, MetricViewAsTableError)
        assert error.__cause__.name == 'Sales_Metrics'
        assert "'cat.sch.Sales_Metrics' is a metric view" in str(
            error.__cause__
        )
    else:
        assert error.__cause__ is None
        (diagnostic,) = diagnostics
        (cause,) = diagnostic['chain']
        assert (
            cause['class']
            == 'sdm_relational_engine.exceptions.MetricViewAsTableError'
        )
        assert cause['details'] == {'name': 'Sales_Metrics'}
        assert "'cat.sch.Sales_Metrics' is a metric view" in cause['message']
    assert len(warehouse.statements) == 1


@pytest.mark.parametrize(
    ('expression', 'reason'),
    [
        ("segment IN ('gold')", "calls 'IN'"),
        ("`user` = 'gold'", 'session value `user`'),
        ("user = 'gold'", "session value 'user'"),
        ("my_udf(status) = 'VALID'", "calls 'my_udf'"),
        ('(SELECT 1) = 1', "SQL keyword 'SELECT'"),
    ],
)
def test_public_filter_rejects_before_reading_rows(expression, reason):
    definition = _PUBLIC['_VIEW'].replace(
        _PUBLIC['_FILTER'], json.dumps(expression)
    )
    warehouse = _Warehouse(definition)
    source = DatabricksSource(warehouse, catalog='cat', schema='sch')
    with pytest.raises(ValueError, match=reason):
        Schema.from_source(source.semantic_model('sales_mv'))
    assert warehouse.statements
    assert all(
        sql.startswith('DESCRIBE') or 'information_schema' in sql
        for sql in warehouse.statements
    )


def test_public_ordinary_table_remains_readable(monkeypatch):
    warehouse = _Warehouse()
    source = DatabricksSource(warehouse, catalog='cat', schema='sch')
    namespace = _PUBLIC[
        'test_every_read_of_the_source_table_goes_through_the_filter'
    ].__globals__
    monkeypatch.setitem(
        namespace['_RELATIONS'], namespace['_FACT'], namespace['_ORDERS']
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(
            source, tables={'order_lines': 'order_lines'}
        )
    assert 'quantity' in schema.tables['order_lines'].columns
    assert any('LIMIT 1000' in sql for sql in warehouse.statements)


def test_public_semantic_model_filter_holds_through_materialize():
    _PUBLIC['test_every_read_of_the_source_table_goes_through_the_filter']()


def test_public_dimension_uses_its_own_table_scope(monkeypatch):
    definition = _PUBLIC['_VIEW'].replace(
        'dimensions:',
        "dimensions:\n  - name: misplaced\n    expr: segment IN ('a')\n"
        "  - name: is_gold\n    expr: customers.segment IN ('a')\n",
    )
    namespace = _PUBLIC[
        'test_every_read_of_the_source_table_goes_through_the_filter'
    ].__globals__
    relation = namespace['_CUSTOMERS']
    customers = namespace['_RELATIONS'][relation]
    monkeypatch.setitem(
        namespace['_RELATIONS'],
        relation,
        customers.assign(is_gold=customers['segment'] == 'a'),
    )
    warehouse = _Warehouse(definition)
    source = DatabricksSource(warehouse, catalog='cat', schema='sch')
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        schema = Schema.from_source(source.semantic_model('sales_mv'))
    assert 'is_gold' in schema.tables['customers'].columns
    assert 'misplaced' not in schema.tables['order_lines'].columns
    assert any(
        "'misplaced'" in str(warning.message)
        and "calls 'IN'" in str(warning.message)
        for warning in recorded
    )
    assert not any('misplaced' in sql for sql in warehouse.statements)
    assert any("segment IN ('a')" in sql for sql in warehouse.statements)
