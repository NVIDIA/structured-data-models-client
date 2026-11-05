# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved requests preserve model payloads and results across client upgrades."""

import json
import pickle
import warnings
from pathlib import Path

import pandas as pd
import pytest
from saved_request_fixtures import ANCHOR, REQUESTS, TABLES, answers, source

from sdm_client import PredictiveQuery, Schema

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/saved-requests'
EXPECTED = json.loads((FIXTURES / 'expected.json').read_text())


@pytest.mark.parametrize('version', ['reference', 'phase-2'])
@pytest.mark.parametrize('name', sorted(EXPECTED['reference']['requests']))
def test_saved_request_predicts_without_its_original_runtime(version, name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = pickle.loads(
            (FIXTURES / f'{version}-{name}.pkl').read_bytes()
        )
    # Saved payloads are sufficient: prediction needs neither the original
    # runtime nor a source connection used when those rows were materialized.
    assert request.schema.source is None
    assert answers(request, name) == EXPECTED[version]['requests'][name]


def test_saved_fixture_inventory():
    for version, record in EXPECTED.items():
        names = {
            p.stem.removeprefix(f'{version}-')
            for p in FIXTURES.glob(f'{version}-*.pkl')
        }
        assert names == set(REQUESTS) == set(record['requests'])
        assert record['protocol'] == 4
    recorded = EXPECTED['reference']['requests']
    assert len(recorded['batched'][0]['bodies']) == 3
    assert recorded['ranking'][0]['answer']['unscored'] == [9]
    assert recorded['regression'][1]['answer']['items'][0]['explanation']
    assert any(
        'quantiles' in notice for notice in recorded['batched'][1]['notices']
    )


def materialized(name):
    text, arguments = REQUESTS[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(source(), TABLES)
        return PredictiveQuery(text, schema).materialize(
            anchor_time=pd.Timestamp(ANCHOR), **arguments
        )


@pytest.mark.parametrize('name', sorted(REQUESTS))
def test_payloads_restore_missing_native_state(name):
    request = materialized(name)
    for held in (request, *(request.parts or ())):
        held._core = None
    assert answers(request, name) == EXPECTED['reference']['requests'][name]


def test_changed_payloads_take_precedence_over_cached_native_state():
    request = materialized('batched')
    assert request._core is not None and request._core.batches
    request._payloads = request._payloads[:2]
    observed = answers(request, 'batched')
    assert len(observed[0]['bodies']) == 2
    assert (
        observed[0]['bodies']
        == EXPECTED['reference']['requests']['batched'][0]['bodies'][:2]
    )
    request._core = None
    assert answers(request, 'batched') == observed
