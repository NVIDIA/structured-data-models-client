# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit acceptance: SDM_PQL_NUMERIC_REFERENCE=<dir> pytest <this file>."""

import json
import os
from pathlib import Path

import pytest
from tabular_numeric_pure_cases import CASES, observe_case
from tabular_pql_reference import runtime_identity
from test_tabular_pql_core import ROUTES


def _ordinary_warning_outcome(outcome):
    # Owner does not require incidental pandas warning counts/interleaving.
    # Retain exact data/errors, other warnings and the numeric warning sites.
    result = dict(outcome)
    ordinary = set()
    retained = []
    for run in outcome['warnings']:
        event = run['event']
        site = event['site']
        if (
            event['type'] == 'RuntimeWarning'
            and event['message'] == 'overflow encountered in cast'
            and site is not None
            and site['file']
            in (
                'core/indexes/base.py',
                'core/indexes/multi.py',
                'core/sorting.py',
            )
        ):
            ordinary.add((site['file'], site['line']))
        else:
            retained.append(run)
    result['warnings'] = retained
    result['numeric_warning_sites'] = sorted(ordinary)
    return result


@pytest.mark.parametrize('route', ['old', 'core'])
@pytest.mark.parametrize('name', CASES)
def test_matches_frozen_numeric_receipt(monkeypatch, route, name):
    directory = Path(os.environ['SDM_PQL_NUMERIC_REFERENCE'])
    runtime = runtime_identity()
    receipt = json.loads(
        (directory / (runtime['fingerprint'] + '.json')).read_text()
    )
    assert (
        receipt['reference']['revision']
        == '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1'
    )
    assert receipt['generated_by'] == 'scripts/record_tabular_pql.py'
    assert receipt['format'] == 1 and receipt['group'] == 'numeric'
    assert receipt['runtime']['identity'] == runtime['identity']
    assert set(receipt['cases']) == set(CASES)
    monkeypatch.setenv('SDM_CORE_ROUTES', ROUTES if route == 'core' else '')
    actual = observe_case(name)
    expected = receipt['cases'][name]
    if route == 'core':
        assert _ordinary_warning_outcome(actual) == _ordinary_warning_outcome(
            expected
        )
    else:
        assert actual == expected


def test_print_mode_matches_native_stderr(monkeypatch, capfd):
    # NumPy uses C stderr, so FD capture is intentional rather than capsys.
    outcomes = []
    streams = []
    for route in ('old', 'core'):
        monkeypatch.setenv('SDM_CORE_ROUTES', ROUTES if route == 'core' else '')
        outcomes.append(observe_case('float32/indices/print/default/forward'))
        streams.append(capfd.readouterr())
    assert outcomes[0] == outcomes[1]
    assert streams[0] == streams[1]
    assert streams[0].out == ''
    assert all(
        line == 'Warning: overflow encountered in cast'
        for line in streams[0].err.splitlines()
    )
