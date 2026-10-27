# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public fresh-part ownership cases from both frozen native runtime cells.

The fixture contains both complete part payloads, not just the carrier's
first part, from the original public ranking and forecast calls.
Native conversion reentry is an explicit observer effect.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdm_client import PredictiveQuery, Relationship, Schema
from sdm_client.materializers.relational import RelationalMaterializer

REFERENCE = json.loads(
    Path(__file__).with_name('t44_decomposition_reference.json').read_text()
)['families']
ROUTES = ','.join(
    (
        'schema.from-data/frames/frames',
        'schema.from-data/mapping/frames',
        'query.materialize/indices-binary/frames',
        'query.materialize/indices-regression/frames',
        'query.materialize/indices-ranking/frames',
        'query.materialize/indices-forecast/frames',
    )
)


@pytest.fixture(params=[False, True], ids=['original', 'core'])
def core(request, monkeypatch):
    monkeypatch.setenv(
        'SDM_CORE_ROUTES',
        ROUTES if request.param else '',
    )
    if request.param:

        def old_engine_was_called(*args, **kwargs):
            raise AssertionError('core decomposition called the old engine')

        monkeypatch.setattr(
            RelationalMaterializer, '_get_model', old_engine_was_called
        )
    return request.param


@pytest.fixture(params=['ranking', 'forecast'])
def family(request):
    return request.param


def question(family):
    np.random.seed(123)
    with pd.option_context(
        'future.infer_string', False, 'mode.string_storage', 'python'
    ):
        users = pd.DataFrame({'user_id': range(12), 'segment': list('abc') * 4})
        orders = pd.DataFrame(
            {
                'order_id': range(120),
                'user_id': [i % 12 for i in range(120)],
                'item': [i % 4 for i in range(120)],
                'price': [10.0 + i for i in range(120)],
                'channel': ['web', 'app', 'store'] * 40,
                'ordered_at': pd.date_range(
                    '2026-01-01', periods=120, freq='D'
                ),
                'marker': range(120),
                'note': pd.Series(
                    ['valid note'] * 120,
                    dtype=pd.StringDtype(storage='python'),
                ),
            }
        )
        items = pd.DataFrame({'item': range(4), 'kind': list('abcd')})
        schema = Schema.from_data(
            {'users': users, 'orders': orders, 'items': items},
            primary_keys={
                'users': 'user_id',
                'orders': 'order_id',
                'items': 'item',
            },
            time_columns={'orders': 'ordered_at'},
            relationships=[
                Relationship('orders', 'user_id', 'users', 'user_id'),
                Relationship('orders', 'item', 'items', 'item'),
            ],
        )
    schema['orders']['marker'].stype = 'numerical'
    schema['orders']['note'].stype = 'text'
    target = (
        'LIST_DISTINCT(orders.item, 0, 30, days) RANK TOP 2'
        if family == 'ranking'
        else 'COUNT(orders.*, 0, 30, days) FORECAST 2 TIMEFRAMES'
    )
    return PredictiveQuery(
        'PREDICT ' + target + ' FOR EACH users.user_id', schema
    )


def gather(asked, family, seed):
    options = {
        'indices': [1, 2],
        'num_context_samples': 3,
        'num_neighbors': [3],
        'num_hops': 1,
        'random_seed': seed,
    }
    if family == 'ranking':
        options['candidates'] = pd.Series([0, 1])
    with (
        pd.option_context(
            'future.infer_string', False, 'mode.string_storage', 'python'
        ),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter('ignore')
        request = asked.materialize(**options)
    assert len(request.parts) == 2
    return list(request.payloads or ())


def test_repeated_decomposition_uses_fresh_child_default_seeds(core, family):
    asked = question(family)
    assert gather(asked, family, 7) == REFERENCE[family]['seed7']
    # The parent None is omitted for fresh children, whose default is 42.
    # Ordinary non-decomposed materializers have a different continuation
    # contract, covered in test_t44_retained_materialize_owners.py.
    assert gather(asked, family, None) == REFERENCE[family]['seed42']
    assert gather(asked, family, None) == REFERENCE[family]['seed42']


def test_new_parts_observe_current_source_after_the_parent_was_used(
    core, family
):
    asked = question(family)
    assert gather(asked, family, None) == REFERENCE[family]['seed42']
    source = asked.schema['orders'].data
    source['marker'] = source['marker'] + 1000
    try:
        assert (
            gather(asked, family, None) == REFERENCE[family]['changed_source']
        )
    finally:
        source['marker'] = source['marker'] - 1000
    assert gather(asked, family, 7) == REFERENCE[family]['seed7']


def test_reentry_keeps_complete_inner_and_outer_part_payloads(
    core, family, monkeypatch
):
    asked = question(family)
    original = pd.Series.astype
    inner = []

    def reenter(values, dtype, *args, **kwargs):
        if dtype is str and values.name == 'note' and not inner:
            inner.append(None)
            inner[0] = gather(asked, family, 11)
        return original(values, dtype, *args, **kwargs)

    monkeypatch.setattr(pd.Series, 'astype', reenter)
    outer = gather(asked, family, 7)
    assert inner == [REFERENCE[family]['seed11']]
    assert outer == REFERENCE[family]['seed7']
