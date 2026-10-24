# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A semantic model on the core route, the Python host's part (T3.5,
design S6): the provider choice names the model (``view``) on the source
it is kept in, a metric view's calls route as their own paths, the
conversion notes the core gives at describe are said as the schema holds
its tables, and the model is named as the old client named it in what the
core says of it (``{host.semantic_model!r}``, old-behaviour rows 5 and 9).

The core's sentences are the table the binding exports; the view entries
these tests read are written here as ``sdm_contract::wording`` writes them
(T3.27), so the tests hold the host's part whatever binding is built.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import pytest
from sdm_connectors import DatabricksSource, SnowflakeSource
from sdm_connectors.backends.spark import SparkConnection

from sdm_client import _routing
from sdm_client._rust import errors, providers, results
from sdm_client._rust.providers.databricks import (
    DatabricksSparkProvider,
    DatabricksSqlProvider,
)
from sdm_client._rust.schema import conversion_notes

#: A notebook's Spark session as ``sdm_connectors`` recognises one.
_SparkSession = type(
    'SparkSession',
    (),
    {
        '__module__': 'pyspark.fake_for_tests',
        'version': '4.0.0',
        'sql': lambda self, *a, **k: None,
    },
)


def _warehouse() -> Any:
    return type('Connection', (), {'__module__': 'databricks.sql.client'})()


def _snowflake() -> Any:
    return type(
        'SnowflakeConnection', (), {'__module__': 'snowflake.connector'}
    )()


def _sources() -> dict[str, Any]:
    warehouse = DatabricksSource(_warehouse(), catalog='main', schema='sales')
    spark = DatabricksSource(_SparkSession(), schema='sales')
    return {
        'warehouse': warehouse,
        'model': warehouse.semantic_model('sales_metrics'),
        'trusted': spark.semantic_model(
            'main.sales.sales_metrics', trust_expressions=True
        ),
    }


def test_a_metric_view_s_choice_names_the_model_on_its_source() -> None:
    sources = _sources()
    choice, provider = providers.for_source(sources['model'])
    assert choice == {
        'provider': 'sql',
        'dialect': 'databricks',
        'namespace': ['main', 'sales'],
        'view': {'name': 'sales_metrics', 'trust_expressions': False},
    }
    assert type(provider) is DatabricksSqlProvider
    assert provider._connection is sources['warehouse'].connection

    choice, provider = providers.for_source(sources['trusted'])
    assert choice == {
        'provider': 'sql',
        'dialect': 'databricks',
        'namespace': [None, 'sales'],
        'view': {
            'name': 'main.sales.sales_metrics',
            'trust_expressions': True,
        },
    }
    assert type(provider) is DatabricksSparkProvider
    assert isinstance(provider._connection, SparkConnection)

    # A table of the warehouse names no model.
    choice, _ = providers.for_source(sources['warehouse'])
    assert 'view' not in choice


def test_a_snowflake_semantic_view_waits_for_the_snowflake_dialect() -> None:
    model = SnowflakeSource(
        _snowflake(), database='SALES', schema='PUBLIC'
    ).semantic_model('SALES_SEMANTICS')
    assert model.kind == 'snowflake-semantic-view'
    with pytest.raises(_routing.RouteError, match='T3.6'):
        providers.for_source(model)
    # Its paths have no core route in this build, so forcing one fails.
    assert not any(
        path.endswith('/snowflake-semantic-view')
        for path in _routing.CORE_ROUTED
    )


def test_a_metric_view_s_calls_route_as_their_own_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _sources()
    groups = (
        _routing.FROM_SOURCE,
        _routing.FROM_SOURCE_TABLES,
        _routing.MATERIALIZE,
        _routing.PREDICT,
    )
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    for paths in groups:
        assert not _routing.on_the_core(sources['model'], paths)
    assert _routing.FROM_SOURCE['databricks-metric-view'] == (
        'schema.from-source/semantic-model/databricks-metric-view',
    )
    forced = (
        _routing.FROM_SOURCE_DATABRICKS_METRIC_VIEW
        + _routing.MATERIALIZE_DATABRICKS_METRIC_VIEW
        + _routing.PREDICT_DATABRICKS_METRIC_VIEW
    )
    assert set(forced) <= _routing.CORE_ROUTED
    assert set(forced) <= set(_routing.PATHS)
    monkeypatch.setenv(_routing.ENVIRONMENT, ','.join(forced))
    for paths in groups:
        # Either connection's model, and never the warehouse's tables.
        assert _routing.on_the_core(sources['model'], paths)
        assert _routing.on_the_core(sources['trusted'], paths)
        assert not _routing.on_the_core(sources['warehouse'], paths)


def _text(value: str) -> dict[str, str]:
    return {'t': 'utf8', 'v': value}


class _Table:
    def __init__(self, time_column: str | None) -> None:
        self.time_column = time_column


class _Schema:
    def __init__(self, tables: dict[str, _Table]) -> None:
        self.tables = tables


def test_the_conversion_notes_say_what_the_schema_still_holds() -> None:
    described = results.decode(
        {
            'op': 'describe_source',
            'description': 3,
            'notes': [
                {
                    'sentence': 'view_measure_note',
                    'fields': {
                        'measure': _text('revenue'),
                        'function': _text('SUM'),
                        'table': _text('orders'),
                        'column': _text('price'),
                    },
                    'table': 'orders',
                },
                # Of a table the schema did not take.
                {
                    'sentence': 'view_measure_note',
                    'fields': {
                        'measure': _text('visits'),
                        'function': _text('COUNT'),
                        'table': _text('users'),
                        'column': _text('*'),
                    },
                    'table': 'users',
                },
                {
                    'sentence': 'view_time_column_note',
                    'fields': {
                        'column': _text('order_date'),
                        'table': _text('orders'),
                        'reasons': _text('it is what x truncates'),
                    },
                    'table': 'orders',
                    'time_column': 'order_date',
                },
                {
                    'sentence': 'view_parameter_note',
                    'fields': {
                        'parameter': _text('floor'),
                        'default': _text('1'),
                    },
                },
            ],
        }
    )
    assert isinstance(described, results.Described)
    assert described.notes[2] == results.ViewNote(
        sentence='view_time_column_note',
        fields={
            'column': _text('order_date'),
            'table': _text('orders'),
            'reasons': _text('it is what x truncates'),
        },
        table='orders',
        time_column='order_date',
    )
    measure = (
        "'revenue' is SUM(orders.price, 0, 30, days) over the next 30 days"
    )
    took = (
        "Took 'order_date' as the time column of table 'orders', since it "
        'is what x truncates'
    )
    parameter = (
        "Read the parameter 'floor' as its default '1', as a query of the "
        'view without arguments does'
    )
    held = _Schema({'orders': _Table('order_date')})
    assert conversion_notes(described.notes, held) == (
        measure,
        took,
        parameter,
    )
    # A time column the table no longer has says nothing it holds.
    moved = _Schema({'orders': _Table('shipped_at')})
    assert conversion_notes(described.notes, moved) == (measure, parameter)
    # Any other source's description carries none.
    assert results.decode({'op': 'describe_source', 'description': 3}) == (
        results.Described(description=3)
    )


def _warning(sentence: str, **fields: str) -> dict[str, Any]:
    return {
        'code': 'VIEW_READ_IN_PART',
        'sentence': sentence,
        'message': 'the core says this naming nothing',
        'fields': {name: _text(value) for name, value in fields.items()},
    }


def test_what_a_model_loses_is_said_once_naming_the_model() -> None:
    control = json.dumps(
        {
            'message': 'warnings',
            'warnings': [
                _warning('view_materialization_ignored'),
                _warning(
                    'view_join_skipped',
                    join='users',
                    reason='its table is empty',
                ),
            ],
        }
    )
    model = _sources()['model']
    with (
        errors.holding(**errors.source_values(model)),
        warnings.catch_warnings(record=True) as shown,
    ):
        warnings.simplefilter('always')
        errors.warn_from_core(control.encode(), {})
    said = [str(w.message) for w in shown]
    assert said == [
        "[VIEW_READ_IN_PART] SemanticModel('sales_metrics' in "
        "DatabricksSource(catalog='main', schema='sales')) has what a graph "
        'cannot hold, so it is read in part:\n'
        "- Ignored the 'materialization' section since it cannot be "
        'represented in a graph\n'
        "- Failed to add join 'users' since its table is empty"
    ]
    trusted = _sources()['trusted']
    assert repr(trusted).endswith(', trust_expressions=True)')
    # A source that is no model holds no model to name.
    assert 'semantic_model' not in errors.source_values(_sources()['warehouse'])
