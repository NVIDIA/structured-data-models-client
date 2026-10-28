# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller schema refusals retain exact names before native graph casts."""

from __future__ import annotations

import copy
import warnings
from decimal import Decimal

import pandas as pd
import pytest
from test_t44_live_graph_metadata import REFERENCE, failure, frames
from test_t44_text_frames import core as core

from sdm_client import PredictiveQuery, Schema, StructuredDataError


@pytest.mark.parametrize('mutation', [False, True])
def test_declared_string_refusal_keeps_full_frozen_chain_before_native_callbacks(
    core, mutation
):
    case = next(
        item
        for item in REFERENCE['records']
        if item['declared'] == 'string' and item['mutation'] == mutation
    )
    data = frames()
    schema = Schema.from_data(
        data,
        primary_keys={'users': 'user_id', 'orders': 'order_id'},
        time_columns={},
    )
    schema['users']['signal'].stype = 'numerical'
    trace = []

    class NativeDecimal(Decimal):
        def observe(self, method):
            trace.append({'event': method})
            if mutation:
                schema['users']['signal'].dtype = 'string'
                schema['users']['signal'].stype = 'categorical'

        def __float__(self):
            self.observe('float')
            return super().__float__()

        def __str__(self):
            self.observe('str')
            return super().__str__()

    data['orders']['amount'] = pd.Series(
        [NativeDecimal(str(Decimal(i) + Decimal('0.25'))) for i in range(64)],
        dtype=object,
    )
    schema['orders']['amount'].dtype = 'string'
    asked = PredictiveQuery('PREDICT users.age FOR EACH users.user_id', schema)
    actual = []
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        for _ in range(2):
            start, first_warning = len(trace), len(notices)
            with pytest.raises(StructuredDataError) as raised:
                asked.materialize(
                    indices=[7, 8],
                    num_context_samples=3,
                    num_neighbors=[3],
                    num_hops=1,
                    random_seed=7,
                )
            assert raised.value.__cause__ is None
            assert raised.value.__context__ is None
            actual.append(
                {
                    'outcome': failure(raised.value),
                    'trace': copy.deepcopy(trace[start:]),
                    'warnings': [
                        {
                            'type': type(item.message).__name__,
                            'message': str(item.message),
                        }
                        for item in notices[first_warning:]
                    ],
                    'schema_after': {
                        'dtype': schema['users']['signal'].dtype,
                        'stype': str(schema['users']['signal'].stype),
                        'physical': str(data['users']['signal'].dtype),
                    },
                }
            )
    assert actual == case['calls']
    assert trace == []
    assert not case['effect_reached']
    assert asked._relational_materializer._model is None
    assert asked._relational_materializer._core_sampler_state == {}
