# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public LINK parity against a model endpoint with a link prediction head.

Set SDM_LINK_URL for a NIM or SDM_LINK_SERVING_ENDPOINT for Databricks.
Databricks uses the ambient SDK authentication configuration.
No warehouse tables, serving endpoints or sessions are created by these tests.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd
import pytest

from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient
from sdm_client._routing import CORE_ROUTED
from sdm_client.materializers.relational import RelationalMaterializer

pytestmark = pytest.mark.live_nim
URL = os.environ.get('SDM_LINK_URL')
ENDPOINT = os.environ.get('SDM_LINK_SERVING_ENDPOINT')


@pytest.fixture
def link_client():
    if URL and ENDPOINT:
        pytest.fail('set only one LINK endpoint configuration')
    if ENDPOINT:
        client = SDMClient.for_databricks_serving(ENDPOINT, timeout=600.0)
    elif URL:
        client = SDMClient(URL, timeout=600.0)
    else:
        pytest.skip('set SDM_LINK_URL or SDM_LINK_SERVING_ENDPOINT')
    with client:
        yield client


def frames():
    return {
        'users': pd.DataFrame({'user_id': range(8), 'age': range(20, 28)}),
        'items': pd.DataFrame(
            {'item_id': range(4), 'amount': [3.0, 4.0, 5.0, 6.0]}
        ),
        'buys': pd.DataFrame(
            {
                'buy_id': range(120),
                'user_id': [i % 8 for i in range(120)],
                'item_id': [(i * 3) % 4 for i in range(120)],
                'ts': pd.date_range('2023-01-01', periods=120, freq='5D'),
            }
        ),
    }


def forbid_legacy_model(*args, **kwargs):
    raise AssertionError('core LINK called the Python relational engine')


@pytest.mark.parametrize('rows_per_call', [None, 1])
def test_live_link_preserves_request_entities_and_ranked_items(
    monkeypatch, link_client, rows_per_call
):
    observed = []
    indices = [1, 0, 1]
    for core in (False, True):
        with monkeypatch.context() as selected:
            selected.setenv(
                'SDM_CORE_ROUTES',
                ','.join(sorted(CORE_ROUTED)) if core else '',
            )
            if core:
                selected.setattr(
                    RelationalMaterializer, '_get_model', forbid_legacy_model
                )
            schema = Schema.from_data(
                frames(),
                primary_keys={
                    'users': 'user_id',
                    'items': 'item_id',
                    'buys': 'buy_id',
                },
                time_columns={'users': None, 'items': None, 'buys': 'ts'},
            )
            query = PredictiveQuery(
                'PREDICT LIST_DISTINCT(buys.item_id, 0, 30, days) '
                'FOR EACH users.user_id',
                schema,
            )
            options = (
                {}
                if rows_per_call is None
                else {'rows_per_call': rows_per_call}
            )
            request = query.materialize(
                indices=indices,
                anchor_time=pd.Timestamp('2023-03-01'),
                random_seed=7,
                **options,
            )
            payloads = copy.deepcopy(request.payloads)
            result = link_client.predict(
                request, model=KumoRelational()
            ).to_pandas()
            assert result['id'].tolist() == indices
            assert result['row_index'].tolist() == list(range(len(indices)))
            assert len(payloads) == (1 if rows_per_call is None else 3)
            assert request.payloads == payloads
            rankings = result['rankings'].tolist()
            assert all(isinstance(items, list) and items for items in rankings)
            observed.append((payloads, rankings))
    assert observed[1][0] == observed[0][0]
    for original, core in zip(observed[0][1], observed[1][1], strict=True):
        assert [item['item'] for item in core] == [
            item['item'] for item in original
        ]
        np.testing.assert_allclose(
            [item['score'] for item in core],
            [item['score'] for item in original],
            rtol=1e-6,
            atol=1e-7,
        )
