# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-column time decisions through the public frame materializer."""

from __future__ import annotations

import copy
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from t38_forced_routes_record import bare
from t45_frame_routes import FRAME_ROUTES

from sdm_client import PredictiveQuery, Schema

REFERENCE = json.loads(
    Path(__file__)
    .with_name('t44_client_time_columns_reference.json')
    .read_text()
)


@pytest.mark.parametrize('core', [False, True], ids=['original', 'core'])
@pytest.mark.parametrize('case', ['late-deciding', 'feature-iso'])
@pytest.mark.parametrize('storage', ['python', 'pyarrow'])
def test_public_complete_time_columns_match_frozen_payloads(
    monkeypatch, core, case, storage
):
    monkeypatch.setenv('SDM_CORE_ROUTES', FRAME_ROUTES if core else '')
    cell = '310' if sys.version_info[:2] == (3, 10) else '312'
    expected = next(
        record
        for record in REFERENCE['cells'][cell]
        if record['case'] == case and record['storage'] == storage
    )
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        rows = 1005 if case == 'late-deciding' else 24
        users = pd.DataFrame(
            {'user_id': range(16), 'age': np.arange(20, 36, dtype='int64')}
        )
        orders = pd.DataFrame(
            {
                'order_id': range(rows),
                'user_id': np.arange(rows) % 16,
                'amount': np.arange(rows, dtype='float64'),
                'ts': pd.date_range('2024-01-01', periods=rows, freq='h'),
            }
        )
        users['registered'] = pd.date_range('2023-01-01', periods=16, freq='D')
        schema = Schema.from_data(
            {'users': users, 'orders': orders},
            primary_keys={'users': 'user_id', 'orders': 'order_id'},
            time_columns={'users': None, 'orders': 'ts'},
        )
        schema['users']['registered'].stype = 'timestamp'
        if case == 'feature-iso':
            # The corrected feature receipt clears automatic time adoption.
            schema['users'].time_column = None
            values = ['2024-02-01T00:00:00.123456789Z'] * len(users)
            users['registered'] = pd.Series(
                values, dtype=pd.StringDtype(storage=storage)
            )
        else:
            # The decisive original row is beyond both sampling and fetches.
            values = ['01/02/2024'] * rows
            values[-1] = '15/03/2024'
            orders['ts'] = pd.Series(
                values, dtype=pd.StringDtype(storage=storage)
            )
        question = PredictiveQuery(
            'PREDICT users.age FOR EACH users.user_id', schema
        )
        request = question.materialize(
            indices=[14, 15],
            num_context_samples=3,
            num_neighbors=[3],
            random_seed=7,
            anchor_time=pd.Timestamp('2024-04-01'),
        )
        assert type(request.payloads) is tuple
        observed = {
            'case': case,
            'storage': storage,
            'outcome': 'success',
            'payloads': copy.deepcopy(list(request.payloads)),
            'warnings': [
                {
                    'type': type(item.message).__name__,
                    'message': str(item.message),
                }
                for item in notices
            ],
        }
    if core:
        for warning in observed['warnings']:
            warning['message'] = bare(warning['message'])
    assert observed == expected
