# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from sdm_relational_engine import Dtype, Stype
from sdm_relational_engine.rfm.infer import infer_stype

NOTEBOOK = (
    Path(__file__).parents[3] / 'examples' / 'kumo_relational_quickstart.ipynb'
)


class _Column:
    def __init__(self):
        self.stype = None


class _Table:
    def __init__(self, columns):
        self.columns = {name: _Column() for name in columns}
        self.primary_key = None
        self.time_column = None

    def __getitem__(self, name):
        return self.columns[name]


class _Graph:
    def __init__(self):
        self.tables = {
            'customers': _Table(['CustomerID', 'AccountNumber']),
            'products': _Table(['ProductID', 'Name', 'ProductNumber']),
            'sales_order_headers': _Table(
                [
                    'SalesOrderID',
                    'OrderDate',
                    'SalesOrderNumber',
                    'PurchaseOrderNumber',
                    'AccountNumber',
                    'CreditCardApprovalCode',
                ]
            ),
            'sales_order_details': _Table(
                [
                    'SalesOrderDetailID',
                    'OrderDate',
                    'OrderQty',
                ]
            ),
        }

    def __getitem__(self, name):
        return self.tables[name]


def _schema_correction_source():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    return next(
        ''.join(cell['source'])
        for cell in notebook['cells']
        if cell['cell_type'] == 'code'
        and 'primary_keys =' in ''.join(cell['source'])
    )


def test_adventureworks_identifier_like_strings_infer_as_text():
    values = pd.Series([f'AW{index:08d}' for index in range(200)])

    assert infer_stype(values, 'AccountNumber', Dtype.string) == Stype.text


def _stype_value(value):
    return Stype(value).value


def test_quickstart_applies_reviewed_adventureworks_semantic_types():
    graph = _Graph()

    exec(
        _schema_correction_source(),
        {
            'graph': graph,
            'relational': SimpleNamespace(Stype=Stype, Dtype=Dtype),
        },
    )

    expected_ids = {
        'customers': ['AccountNumber'],
        'products': ['ProductNumber'],
        'sales_order_headers': [
            'SalesOrderNumber',
            'PurchaseOrderNumber',
            'AccountNumber',
            'CreditCardApprovalCode',
        ],
    }
    for table_name, column_names in expected_ids.items():
        for column_name in column_names:
            assert _stype_value(graph[table_name][column_name].stype) == 'ID'
    assert _stype_value(graph['products']['Name'].stype) == 'text'
    assert (
        _stype_value(graph['sales_order_details']['OrderQty'].stype)
        == 'numerical'
    )

    assert graph['customers'].primary_key == 'CustomerID'
    assert graph['products'].primary_key == 'ProductID'
    assert graph['sales_order_headers'].primary_key == 'SalesOrderID'
    assert graph['sales_order_details'].primary_key == 'SalesOrderDetailID'
    assert graph['sales_order_headers'].time_column == 'OrderDate'
    assert graph['sales_order_details'].time_column == 'OrderDate'
