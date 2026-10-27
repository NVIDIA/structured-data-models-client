# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public ownership twins of the executed frozen CPython 3.10/3.12 witness.

The two frozen result receipts have digests
5ac982b76a9e85a598a9613640e37ff801c40bb5af11af4351ddf32f69549f5a and
9abf91cae7c6f8965ff59e239c0f29ae7a928423e42b62cfeb6544b4c024bcfb.
The native conversion reentry below is explicit observer behavior, not a
claim that an ordinary builtin conversion naturally invokes a callback.
"""

from __future__ import annotations

import warnings

import pandas as pd

from sdm_client import PredictiveQuery, Schema

from test_t44_text_frames import core as core, shop


def schema():
    frames = shop('python')
    frames['users']['age'] = pd.Series(
        [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, None, None], dtype='Float64'
    )
    value = Schema.from_data(
        frames,
        primary_keys={'users': 'user_id', 'orders': 'order_id'},
        time_columns={},
    )
    value['users']['age'].stype = 'numerical'
    value['users']['note'].stype = 'categorical'
    value['orders']['note'].stype = 'text'
    value['orders']['marker'].stype = 'numerical'
    return value


def query(value):
    return PredictiveQuery('PREDICT users.age FOR EACH users.user_id', value)


def gather(asked, seed=None, rows=1):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return asked.materialize(
            indices=[7, 8],
            random_seed=seed,
            rows_per_call=rows,
            num_neighbors=[3],
            num_hops=1,
            num_context_samples=6,
        )


def offsets(request):
    result = []
    for number, payload in enumerate(request.payloads):
        table = payload['predict']['related_tables']['orders']
        marker = table['columns'].index('marker')
        result.append(
            [row[marker] - (48 + 8 * number) for row in table['rows']]
        )
    return result


def test_one_materializer_continues_while_fresh_same_schema_queries_are_independent(
    core,
):
    value = schema()
    shared = query(value)
    assert offsets(gather(shared)) == [[1, 0, 6], [5, 4, 7]]
    assert offsets(gather(shared)) == [[4, 1, 0], [0, 4, 6]]
    assert offsets(gather(query(value))) == [[1, 0, 6], [5, 4, 7]]
    assert offsets(gather(shared, seed=7)) == [[3, 1, 4], [3, 1, 4]]
    assert offsets(gather(shared, seed=7)) == [[3, 1, 4], [3, 1, 4]]
    assert offsets(gather(shared)) == [[2, 1, 5], [4, 0, 5]]


def test_native_conversion_reentry_uses_the_actual_shared_stream_between_draws(
    core, monkeypatch
):
    shared = query(schema())
    original = pd.Series.astype
    inner = []

    def reenter(values, dtype, *args, **kwargs):
        if dtype is str and values.name == 'note' and not inner:
            inner.append(None)
            inner[0] = gather(shared, rows=2)
        return original(values, dtype, *args, **kwargs)

    monkeypatch.setattr(pd.Series, 'astype', reenter)
    outer = gather(shared)
    assert len(inner) == 1
    assert offsets(outer) == [[1, 0, 6], [1, 5, 2]]
    assert offsets(gather(shared)) == [[2, 4, 7], [1, 4, 5]]
