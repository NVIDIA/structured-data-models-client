# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete public SQL journeys consume native entity and what-if operands."""

from __future__ import annotations

import pandas as pd
import pytest
import test_t45_native_sql_labels as public


def _observed(monkeypatch, column, effect, core):
    actual_astype = pd.Series.astype
    events = []

    def converted(self, dtype, *args, **kwargs):
        if self.name == column and dtype == 'float32':
            events.append((self.name, dtype, len(self)))
            if effect == 'failure':
                raise ValueError('actual condition input failed')
            result = actual_astype(self, dtype, *args, **kwargs)
            result.iloc[:] = -1
            return result
        return actual_astype(self, dtype, *args, **kwargs)

    with monkeypatch.context() as selected:
        selected.setattr(pd.Series, 'astype', converted)
        result = public._observe(selected, 'ordinary', core)
    return result, events


@pytest.mark.parametrize('column', ['age', 'amount'])
@pytest.mark.parametrize('effect', ['changed', 'failure'])
def test_each_reached_condition_role_consumes_actual_native_values_and_failure(
    monkeypatch, column, effect
):
    query = (
        'PREDICT LIST_DISTINCT(buys.item_id, 0, 30, days) '
        'FOR EACH users.user_id WHERE users.age > 22 '
        'ASSUMING SUM(buys.amount, 0, 30, days) > 0'
    )
    setup = [
        'ALTER TABLE users ALTER COLUMN age TYPE DOUBLE',
        'ALTER TABLE buys ADD COLUMN amount DOUBLE DEFAULT 1',
    ]
    monkeypatch.setattr(public, '_input', lambda case: (setup, query))
    old = _observed(monkeypatch, column, effect, False)
    current = _observed(monkeypatch, column, effect, True)
    assert old[1]
    assert current[1] == old[1]
    assert old[0]['outcome'] == 'error'
    assert old[0]['stage'] == 'materialize'
    assert current[0] == public._approved_old(old[0], 'ordinary')


def _time_observed(monkeypatch, effect, core):
    getitem, astype = pd.DataFrame.__getitem__, pd.Series.astype
    captured = []
    events = []

    def column(self, key):
        value = getitem(self, key)
        if type(key) is str and key == 'ts':
            captured.append((self, value))
        return value

    def convert(self, dtype, *args, **kwargs):
        if self.name == 'amount' and dtype == 'float32':
            frame, times = captured[-1]
            earlier = times.iloc[0] - pd.Timedelta(days=1000)
            events.append((len(self), len(times)))
            if effect == 'in-place':
                times.iloc[:] = earlier
            else:
                frame['ts'] = earlier
        return astype(self, dtype, *args, **kwargs)

    with monkeypatch.context() as selected:
        selected.setattr(pd.DataFrame, '__getitem__', column)
        selected.setattr(pd.Series, 'astype', convert)
        result = public._observe(selected, 'ordinary', core)
    return result, events


@pytest.mark.parametrize('effect', ['in-place', 'replacement'])
def test_actual_captured_time_owner_remains_authoritative_until_window_selection(
    monkeypatch, effect
):
    query = (
        'PREDICT LIST_DISTINCT(buys.item_id, 0, 30, days) '
        'FOR EACH users.user_id ASSUMING SUM(buys.amount, 0, 30, days) > 0'
    )
    setup = ['ALTER TABLE buys ADD COLUMN amount DOUBLE DEFAULT 1']
    monkeypatch.setattr(public, '_input', lambda case: (setup, query))
    old = _time_observed(monkeypatch, effect, False)
    current = _time_observed(monkeypatch, effect, True)
    assert old[1]
    assert current[1] == old[1]
    assert current[0] == public._approved_old(old[0], 'ordinary')


@pytest.mark.parametrize('role', ['entity', 'target', 'what_if'])
@pytest.mark.parametrize('kind', ['TIMESTAMP', 'TIMESTAMPTZ'])
def test_native_timestamp_condition_family_matches_its_complete_original_journey(
    monkeypatch, role, kind
):
    table, column = (
        ('buys', 'recorded_at') if role == 'target' else ('users', 'created_at')
    )
    setup = [
        f"ALTER TABLE {table} ADD COLUMN {column} {kind} DEFAULT TIMESTAMP '2023-01-01'"
    ]
    target = (
        'buys.item_id WHERE buys.recorded_at != NULL'
        if role == 'target'
        else 'buys.item_id'
    )
    query = (
        f'PREDICT LIST_DISTINCT({target}, 0, 30, days) FOR EACH users.user_id'
    )
    if role == 'entity':
        query += ' WHERE users.created_at != NULL'
    elif role == 'what_if':
        query += ' ASSUMING users.created_at != NULL'
    monkeypatch.setattr(public, '_input', lambda case: (setup, query))
    original = public._observe(monkeypatch, 'ordinary', False)
    current = public._observe(monkeypatch, 'ordinary', True)
    if original['outcome'] == 'error':
        assert original['stage'] == 'materialize'
        assert (
            original['error']['type'] == 'sdm_client.errors.StructuredDataError'
        )
        assert original['error']['code'] == 'INTERNAL_ERROR'
        assert (
            original['error']['cause']['type']
            == 'pyarrow.lib.ArrowNotImplementedError'
        )
        assert (
            original['error']['cause']['text']
            == 'Unsupported cast from list<item: int64> to int64 using function cast_int64'
        )
    else:
        assert original['request']['task']['kind'] == 'temporal_link_prediction'
        assert original['request']['predict_rows'] == 2
        assert original['request']['context_rows'] > 0
    assert current == public._approved_old(original, 'ordinary')
