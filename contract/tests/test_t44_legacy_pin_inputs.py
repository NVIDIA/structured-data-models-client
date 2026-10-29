# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The historical Rust input pin predates corrected host declaration flags."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

PIN = Path(__file__).parents[1] / 'pinned/base.regression.core.json'
FLAGS = {
    ('orders', 'order_id'): (False, False),
    ('orders', 'user_id'): (False, False),
    ('orders', 'price'): (False, False),
    ('orders', 'ts'): (False, True),
    ('users', 'user_id'): (False, False),
    ('users', 'age'): (False, False),
    ('users', 'country'): (False, False),
}


def corrected_inputs():
    historical = json.loads(PIN.read_text())['inputs']
    actual = copy.deepcopy(historical)
    [materialize] = [item for item in actual if item['op'] == 'materialize']
    for table in materialize['schema']['tables']:
        for column in table['columns']:
            column['untyped'], column['stype_declared'] = FLAGS[
                (table['name'], column['name'])
            ]
    return actual, historical


@pytest.mark.parametrize(
    'changed',
    ['missing', 'type', 'declared', 'untyped', 'query', 'dtype', 'order'],
)
def test_corrected_flag_map_and_every_other_historical_input_remain_independent(
    changed,
):
    actual, historical = corrected_inputs()
    assert_corrected_inputs(actual, historical)
    [materialize] = [item for item in actual if item['op'] == 'materialize']
    columns = materialize['schema']['tables'][0]['columns']
    if changed == 'missing':
        columns[0].pop('untyped')
    elif changed == 'type':
        columns[-1]['stype_declared'] = 1
    elif changed == 'declared':
        columns[-1]['stype_declared'] = False
    elif changed == 'untyped':
        columns[0]['untyped'] = True
    elif changed == 'query':
        materialize['query'] += ' changed'
    elif changed == 'dtype':
        columns[0]['dtype'] = 'float64'
    else:
        columns.reverse()
    before = (copy.deepcopy(actual), copy.deepcopy(historical))
    with pytest.raises(AssertionError):
        assert_corrected_inputs(actual, historical)
    assert (actual, historical) == before


def assert_corrected_inputs(actual, historical):
    """Pin the corrected actual metadata before full historical comparison.

    The old omitted time declaration meant false and was defective.
    It is not asserted semantically equal to the corrected true value.
    """
    shown = copy.deepcopy(actual)
    [current] = [item for item in shown if item['op'] == 'materialize']
    [prior] = [item for item in historical if item['op'] == 'materialize']
    seen = {}
    for table, old_table in zip(
        current['schema']['tables'], prior['schema']['tables'], strict=True
    ):
        for column, old_column in zip(
            table['columns'], old_table['columns'], strict=True
        ):
            key = table['name'], column['name']
            assert key not in seen and key in FLAGS
            assert 'untyped' in column and 'stype_declared' in column
            assert (
                type(column['untyped'])
                is type(column['stype_declared'])
                is bool
            )
            seen[key] = column['untyped'], column['stype_declared']
            assert seen[key] == FLAGS[key]
            assert (
                'untyped' not in old_column
                and 'stype_declared' not in old_column
            )
            column.pop('untyped')
            column.pop('stype_declared')
    assert seen == FLAGS
    assert shown == historical
