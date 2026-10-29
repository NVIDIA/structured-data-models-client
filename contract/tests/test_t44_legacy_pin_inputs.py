# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The historical Rust input pin predates corrected host declaration flags."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from _actual_core import checked_fetch_bindings

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


def assert_corrected_inputs(actual, historical, *, materialized=None):
    """Pin the corrected actual metadata before full historical comparison.

    The old omitted time declaration meant false and was defective.
    It is not asserted semantically equal to the corrected true value.
    """
    shown = copy.deepcopy(actual)
    for start, prior in zip(shown, historical, strict=True):
        if start['op'] == 'predict' and set(start) == {'op', 'model'}:
            # The checked native role observes a model, not a legacy state
            # argument. Its independently observed materialization keeps
            # the pin's whole state obligation without inventing an input.
            assert materialized is not None and materialized == prior['state']
            assert start == {
                key: value for key, value in prior.items() if key != 'state'
            }
    compared = copy.deepcopy(historical)
    for start, prior in zip(shown, compared, strict=True):
        if start['op'] == 'predict' and 'state' not in start:
            del prior['state']
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
    assert shown == compared


def assert_pinned_fetch_shapes(actual, historical, *, observation=None):
    """Compare the old plan-only pin with an explicit typed owner bijection.

    This pin never recorded seed values. Their separate driver/result and
    full request assertions remain mandatory in the route consumers.
    Actual binding facts require their complete observation and validation
    before projecting the property that this historical pin never recorded.
    Every SQL byte and non-owner field remains in this exact comparison.
    This helper is never used for replay or general cross-host acceptance.
    """
    assert type(actual) is list and all(type(fetch) is dict for fetch in actual)
    if observation is None:
        assert not any('bindings' in fetch for fetch in actual), (
            'binding facts require a complete actual observation'
        )
    else:
        checked_fetch_bindings(observation, actual)
    shown = copy.deepcopy(actual)
    for fetch in shown:
        fetch.pop('bindings', None)
    assert historical == json.loads(PIN.read_text())['fetches']
    forward, reverse = {}, {}
    for current, prior in zip(shown, historical, strict=True):
        read, old_read = current['request'], prior['request']
        if read['request'] not in ('lookup', 'neighbors'):
            continue
        assert read['request'] == old_read['request']
        source, target = (
            read['seeds']['attachment'],
            old_read['seeds']['attachment'],
        )
        assert type(source) is type(target) is int
        assert 0 <= source < 2**32 and 0 <= target < 2**32
        assert forward.setdefault(source, target) == target
        assert reverse.setdefault(target, source) == source
        owned = 0
        for statement, old_statement in zip(
            current['plan']['statements'],
            prior['plan']['statements'],
            strict=True,
        ):
            for param, old_param in zip(
                statement['params'], old_statement['params'], strict=True
            ):
                if param['param'] == 'key_set':
                    assert param['keys'] == read['seeds']
                    assert (
                        old_param['param'] == 'key_set'
                        and old_param['keys'] == old_read['seeds']
                    )
                    param['keys']['attachment'] = target
                    owned += 1
        assert owned > 0
        read['seeds']['attachment'] = target
    assert shown == historical


@pytest.mark.parametrize(
    'fault', ['sql', 'owner', 'schema', 'column', 'order', 'reuse', 'negative']
)
def test_historical_plan_pin_keeps_literal_sql_and_typed_owner_correspondence(
    fault,
):
    historical = json.loads(PIN.read_text())['fetches']
    actual = copy.deepcopy(historical)
    seeded = []
    for fetch in actual:
        if fetch['request']['request'] not in ('lookup', 'neighbors'):
            continue
        fetch['request']['seeds']['attachment'] += 4
        for statement in fetch['plan']['statements']:
            for param in statement['params']:
                if param['param'] == 'key_set':
                    param['keys']['attachment'] += 4
        seeded.append(fetch)
    assert len(seeded) == 3
    assert_pinned_fetch_shapes(actual, historical)
    first, second = seeded[:2]
    param = next(
        p
        for s in first['plan']['statements']
        for p in s['params']
        if p['param'] == 'key_set'
    )
    if fault == 'sql':
        first['plan']['statements'][0]['text'] += ' changed'
    elif fault == 'owner':
        param['keys']['attachment'] += 1
    elif fault == 'schema':
        first['request']['seeds']['columns'][0]['nullable'] = False
        param['keys']['columns'][0]['nullable'] = False
    elif fault == 'column':
        first['request']['columns'][0] = 'different_business_column'
    elif fault == 'order':
        actual[-1], actual[-2] = actual[-2], actual[-1]
    else:
        target = (
            -1
            if fault == 'negative'
            else second['request']['seeds']['attachment']
        )
        first['request']['seeds']['attachment'] = target
        param['keys']['attachment'] = target
    before = copy.deepcopy(actual), copy.deepcopy(historical)
    with pytest.raises(AssertionError):
        assert_pinned_fetch_shapes(actual, historical)
    assert (actual, historical) == before


def test_logical_predict_role_retains_the_independent_whole_materialized_state():
    actual, historical = corrected_inputs()
    [predict] = [item for item in actual if item['op'] == 'predict']
    materialized = predict.pop('state')
    assert_corrected_inputs(actual, historical, materialized=materialized)
    changed = copy.deepcopy(materialized)
    changed['query'] += ' different'
    with pytest.raises(AssertionError):
        assert_corrected_inputs(actual, historical, materialized=changed)
