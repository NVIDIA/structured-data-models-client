# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit acceptance: SDM_PQL_DECIMAL_REFERENCE=<dir> pytest <this file>."""

import json
import os
from pathlib import Path

import pytest
from tabular_decimal_cases import CASES, observe_case
from tabular_pql_reference import runtime_identity
from test_tabular_pql_core import ROUTES


@pytest.mark.parametrize('route', ['old', 'core'])
@pytest.mark.parametrize('name', CASES)
def test_matches_frozen_decimal_receipt(monkeypatch, route, name):
    directory = Path(os.environ['SDM_PQL_DECIMAL_REFERENCE'])
    runtime = runtime_identity()
    receipt = json.loads(
        (directory / (runtime['fingerprint'] + '.json')).read_text()
    )
    assert (
        receipt['reference']['revision']
        == '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1'
    )
    assert receipt['generated_by'] == 'scripts/record_tabular_pql.py'
    assert receipt['format'] == 1 and receipt['group'] == 'decimal'
    assert receipt['runtime']['identity'] == runtime['identity']
    assert set(receipt['cases']) == set(CASES)
    monkeypatch.setenv('SDM_CORE_ROUTES', ROUTES if route == 'core' else '')
    actual = observe_case(name)
    expected = receipt['cases'][name]
    if (
        route == 'core'
        and name.split('/')[1] == 'predict'
        and actual.get('error') == expected.get('error')
        and actual.get('error', {}).get('type')
        in ('FloatOperation', 'InvalidOperation')
        and actual['flags'] == expected['flags']
    ):
        # Positional filtering avoids pandas' redundant label comparison.
        # The same terminal Decimal error can therefore follow this advisory.
        advice = [
            'SDMWarning',
            'data has 10,001 rows that could be context and the endpoint '
            'accepts 10,000, so 10,000 were sampled. This route takes no '
            'num_context_samples, so choose the context by building the '
            'schema from the rows you want.',
        ]
        actual = {
            **actual,
            'warnings': [item for item in actual['warnings'] if item != advice],
        }
        expected = {
            **expected,
            'warnings': [
                item for item in expected['warnings'] if item != advice
            ],
        }
    assert actual == expected
