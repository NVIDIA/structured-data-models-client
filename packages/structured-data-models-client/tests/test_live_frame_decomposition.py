# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual endpoint comparison of frame ranking and forecast routes.

Uses the same public questions as the immutable frozen decomposition cases.
No serving endpoint, warehouse table or fitted session is created here.
"""

from __future__ import annotations

import copy
import os
import warnings
from dataclasses import asdict

import pandas as pd
import pytest
from t45_frame_routes import FRAME_ROUTES
from test_t44_decomposition_owners import question

from sdm_client import KumoRelational, SDMClient
from sdm_client.materializers.relational import RelationalMaterializer

pytestmark = pytest.mark.live_nim


def _same_answer(original, migrated):
    """Compare every response value, allowing only model floating precision."""
    if isinstance(original, dict):
        assert isinstance(migrated, dict)
        assert migrated.keys() == original.keys()
        for key, value in original.items():
            _same_answer(value, migrated[key])
    elif isinstance(original, (list, tuple)):
        assert isinstance(migrated, (list, tuple))
        assert len(migrated) == len(original)
        for expected, actual in zip(original, migrated, strict=True):
            _same_answer(expected, actual)
    elif isinstance(original, float):
        assert migrated == pytest.approx(original, rel=1e-6, abs=1e-7)
    else:
        assert migrated == original


@pytest.fixture
def relational_client():
    url = os.environ.get('SDM_RELATIONAL_URL')
    endpoint = os.environ.get('SDM_RELATIONAL_SERVING_ENDPOINT')
    if url and endpoint:
        pytest.fail('set only one relational endpoint configuration')
    if endpoint:
        configured = SDMClient.for_databricks_serving(endpoint, timeout=600.0)
    elif url:
        configured = SDMClient(url=url, timeout=600.0)
    else:
        pytest.skip('set SDM_RELATIONAL_URL or SDM_RELATIONAL_SERVING_ENDPOINT')
    with configured as opened:
        yield opened


@pytest.mark.parametrize('family', ['forecast', 'ranking'])
def test_live_decomposition_agrees_with_the_original_route(
    monkeypatch, relational_client, family
):
    observed = []
    for core in (False, True):
        with monkeypatch.context() as selected:
            selected.setenv('SDM_CORE_ROUTES', FRAME_ROUTES if core else '')
            if core:

                def legacy_engine(*args, **kwargs):
                    raise AssertionError(
                        'core decomposition called the old engine'
                    )

                selected.setattr(
                    RelationalMaterializer, '_get_model', legacy_engine
                )
            options = {
                'indices': [1, 2],
                'num_context_samples': 3,
                'num_neighbors': [3],
                'num_hops': 1,
                'random_seed': 7,
            }
            if family == 'ranking':
                options['candidates'] = pd.Series([0, 1])
            with warnings.catch_warnings(record=True) as delivered:
                warnings.simplefilter('always')
                request = question(family).materialize(**options)
                assert len(request.parts) == 2
                payloads = [
                    copy.deepcopy(part.payloads) for part in request.parts
                ]
                answer = relational_client.predict(
                    request, model=KumoRelational()
                )
            frame = answer.to_pandas()
            assert not frame.empty
            assert set(frame['id']) == {1, 2}
            if family == 'forecast':
                assert set(frame['forecast_step']) == {1, 2}
                assert len(frame) == 4
            else:
                assert len(frame) == 2
                assert all(frame['rankings'].map(bool))
            observed.append(
                {
                    'payloads': payloads,
                    'model': answer.model,
                    'outputs': answer.outputs,
                    'items': [asdict(item) for item in answer.items],
                    'unscored': answer.unscored_candidates,
                    'explanation': answer.explanation,
                    'warnings': [
                        (type(w.message).__name__, str(w.message))
                        for w in delivered
                    ],
                }
            )
    # Payloads are compared exactly; tolerance applies only to actual model
    # output floats and never changes context, ordering or request values.
    assert observed[1].pop('payloads') == observed[0].pop('payloads')
    _same_answer(observed[0], observed[1])
