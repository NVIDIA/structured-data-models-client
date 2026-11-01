# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public frozen parity and wide-date improvement for Arrow row labels."""

import json
from pathlib import Path

import pytest
from tabular_arrow_index_cases import CASES, observe_case
from tabular_pql_reference import runtime_identity
from test_tabular_pql_core import ROUTES


def reference_cases():
    runtime = runtime_identity()
    path = (
        Path(__file__).parent
        / 'data/tabular_arrow_index_reference'
        / (runtime['fingerprint'] + '.json')
    )
    receipt = json.loads(path.read_text())
    assert (
        receipt['reference']['revision']
        == '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1'
    )
    assert receipt['generated_by'] == 'scripts/record_tabular_pql.py'
    assert receipt['group'] == 'arrow_index'
    assert receipt['runtime']['identity'] == runtime['identity']
    assert set(receipt['cases']) == set(CASES)
    return receipt['cases']


@pytest.mark.parametrize('route', ['old', 'core'])
@pytest.mark.parametrize('name', CASES)
def test_arrow_labels_match_frozen_public_journey(monkeypatch, route, name):
    monkeypatch.setenv('SDM_CORE_ROUTES', ROUTES if route == 'core' else '')
    expected = reference_cases()[name]
    actual = observe_case(name)
    assert actual == expected
    assert 'error' not in actual
    assert actual['index_matches_source']
    assert actual['context_ids_sha256'] == actual['encoded_ids_sha256']


@pytest.mark.parametrize('entry', ['indices', 'predict'])
@pytest.mark.parametrize('shape', ['forward', 'reverse'])
def test_wide_dates_avoid_python_date_conversion(monkeypatch, entry, shape):
    monkeypatch.setenv('SDM_CORE_ROUTES', ROUTES)
    # Adding a constant to every non-null physical day preserves its order.
    # The complete ordinary-date journey is independently frozen above.
    actual = observe_case(f'date32_wide/{entry}/{shape}')
    assert actual == reference_cases()[f'date32/{entry}/{shape}']
    assert 'error' not in actual
