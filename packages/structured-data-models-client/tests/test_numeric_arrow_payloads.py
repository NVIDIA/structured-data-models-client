# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical floating widths survive public relational materialization."""

import pandas as pd
import pyarrow as pa
import pytest

from sdm_client import KumoRelational, PredictiveQuery, Schema


def floating_dtype(kind):
    return {
        'numpy16': 'float16',
        'numpy32': 'float32',
        'numpy64': 'float64',
        'arrow32': pd.ArrowDtype(pa.float32()),
        'arrow64': pd.ArrowDtype(pa.float64()),
        'nullable32': 'Float32',
        'nullable64': 'Float64',
    }[kind]


@pytest.mark.parametrize(
    'kind',
    [
        'numpy32',
        'numpy64',
        'arrow32',
        'arrow64',
        'nullable32',
        'nullable64',
    ],
)
@pytest.mark.parametrize('declared', [False, True])
def test_floating_primary_key_keeps_its_width_and_values(kind, declared):
    identifiers = [1.25, 2.5, 3.75, 4.0]
    frame = pd.DataFrame(
        {
            'product_id': pd.Series(identifiers, dtype=floating_dtype(kind)),
            'category': ['books', 'games', 'tools', 'books'],
        }
    )
    initial = frame.assign(product_id=[1, 2, 3, 4]) if declared else frame
    schema = Schema.from_data(
        {'products': initial}, primary_keys={'products': 'product_id'}
    )
    if declared:
        schema['products'].data = frame
    request = PredictiveQuery(
        'PREDICT products.category FOR EACH products.product_id', schema
    ).materialize(
        model=KumoRelational(), indices=identifiers[:2], random_seed=7
    )
    expected_dtype = (
        'float64' if kind.endswith('64') or kind == 'arrow32' else 'float32'
    )
    seen = set()
    for payload in request.payloads:
        columns = payload['schema']['instance_table']['columns']
        assert columns['__kumo_entity_ref_0']['dtype'] == expected_dtype
        for half in ('context', 'predict'):
            table = payload[half]['instance_table']
            position = table['columns'].index('__kumo_entity_ref_0')
            seen.update(row[position] for row in table['rows'])
    assert seen == set(identifiers)


@pytest.mark.parametrize('kind', ['numpy16', 'numpy32', 'arrow32'])
def test_declared_float_target_keeps_values_and_class_identity(kind):
    frame = pd.DataFrame(
        {
            'product_id': [1, 2, 3, 4],
            'category': ['books', 'games', 'tools', 'books'],
        }
    )
    schema = Schema.from_data(
        {'products': frame}, primary_keys={'products': 'product_id'}
    )
    values = [1.25, 2.5, 3.75, 4.0]
    schema['products'].data = frame.assign(
        category=pd.Series(values, dtype=floating_dtype(kind))
    )
    request = PredictiveQuery(
        'PREDICT products.category FOR EACH products.product_id', schema
    ).materialize(model=KumoRelational(), indices=[1, 2], random_seed=7)
    seen = set()
    for payload in request.payloads:
        target = payload['task']['target']
        assert target['dtype'] == 'float32'
        assert set(target['classes']) == {str(value) for value in values}
        table = payload['context']['instance_table']
        position = table['columns'].index('TARGET')
        seen.update(row[position] for row in table['rows'])
    assert seen == set(values[2:])


@pytest.mark.parametrize(
    ('kind', 'expected'), [('numpy32', 'float64'), ('nullable32', 'float32')]
)
def test_reconstructed_foreign_keys_keep_the_iterated_scalar_width(
    kind, expected
):
    products = pd.DataFrame(
        {'product_id': [1, 2, 3, 4], 'category': ['a', 'b', 'a', 'b']}
    )
    orders = pd.DataFrame(
        {'order_id': list(range(1, 9)), 'product_id': [1, 2, 3, 4] * 2}
    )
    schema = Schema.from_data(
        {'products': products, 'orders': orders},
        primary_keys={'products': 'product_id', 'orders': 'order_id'},
    )
    schema['products'].data = products.assign(
        product_id=pd.Series([1, 2, 3, 4], dtype=floating_dtype(kind))
    )
    request = PredictiveQuery(
        'PREDICT products.category FOR EACH products.product_id', schema
    ).materialize(model=KumoRelational(), indices=[1, 2], random_seed=7)
    for payload in request.payloads:
        columns = payload['schema']['related_tables']['orders']['columns']
        assert columns['product_id']['dtype'] == expected
        values = set()
        for half in ('context', 'predict'):
            frame = payload[half]['related_tables']['orders']
            position = frame['columns'].index('product_id')
            values.update(row[position] for row in frame['rows'])
        assert values == {1, 2, 3, 4}
