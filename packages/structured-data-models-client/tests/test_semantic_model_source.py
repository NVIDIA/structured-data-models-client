# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""How a Databricks metric view read as a schema is refused.

The warehouse is the fake in `test_a_metric_view_filter_holds`, which
answers the statements the Databricks provider sends.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from sdm_connectors import DatabricksSource
from test_a_metric_view_filter_holds import _TYPES, _Cursor, _Warehouse

from sdm_client import Schema, StructuredDataError


@pytest.fixture
def source() -> Any:
    return DatabricksSource(_Warehouse(), catalog='cat', schema='sch')


def _schema(source: Any, tables: dict[str, str] | None = None) -> Schema:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return Schema.from_source(source.semantic_model('sales_mv'), tables)


def test_a_table_the_model_does_not_have_is_named(source: Any) -> None:
    with pytest.raises(StructuredDataError, match='not tables of') as caught:
        _schema(source, {'orders': 'orders'})

    assert caught.value.code == 'INVALID_REQUEST'
    assert caught.value.details['absent'] == ['orders']


def test_a_missing_source_table_is_an_invalid_request_naming_the_model(
    source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(_TYPES, 'order_lines')

    with pytest.raises(StructuredDataError) as caught:
        _schema(source)

    assert caught.value.code == 'INVALID_REQUEST'
    assert "order_lines' does not exist" in caught.value.message
    assert caught.value.details['semantic_model'] == 'sales_mv'
    assert caught.value.details['source'] == ['cat', 'sch', 'order_lines']


class _DriverError(Exception):
    r"""Stands in for a warehouse driver's own exception."""


def test_a_failure_that_is_not_a_refusal_is_not_an_invalid_request(
    source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    execute = _Cursor.execute

    def fails(self: _Cursor, sql: str, parameters: Any = None) -> None:
        if 'LIMIT' in sql:
            raise _DriverError('the warehouse went away')
        execute(self, sql, parameters)

    monkeypatch.setattr(_Cursor, 'execute', fails)

    with pytest.raises(StructuredDataError) as caught:
        _schema(source)

    assert caught.value.code == 'PROVIDER_ERROR'
    assert caught.value.details['driver_class'].endswith('._DriverError')
    assert 'the warehouse went away' not in str(caught.value)
