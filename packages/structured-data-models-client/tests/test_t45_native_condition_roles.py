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
