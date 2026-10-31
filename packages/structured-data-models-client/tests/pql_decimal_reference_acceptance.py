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
    assert observe_case(name) == receipt['cases'][name]
