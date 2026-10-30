# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit same-runtime acceptance for materialization and prediction effects.

Like the count receipt acceptance, this is invoked against an installed wheel
and a matching canonical receipt; it never substitutes another runtime oracle.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from tabular_numeric_effects_cases import CASES, observe_case
from tabular_pql_reference import runtime_identity

ROUTES = ','.join(
    (
        'query.materialize/indices-binary/frames',
        'query.materialize/indices-multiclass/frames',
        'query.materialize/indices-regression/frames',
        'query.materialize/predict-frame/frames',
        'request.tabular/classification/frames',
        'request.tabular/regression/frames',
        'client.predict/tabular/frames',
    )
)


@pytest.fixture(scope='module')
def reference() -> dict[str, Any]:
    runtime = runtime_identity()
    path = (
        Path(__file__).parent
        / 'data/tabular_numeric_effects_reference'
        / (runtime['fingerprint'] + '.json')
    )
    receipt = json.loads(path.read_text())
    assert receipt['runtime'] == runtime
    assert (
        receipt['reference']['revision']
        == '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1'
    )
    assert receipt['group'] == 'numeric-effects'
    assert set(receipt['cases']) == set(CASES)
    return receipt['cases']


@pytest.mark.parametrize('route', ['old', 'core'])
@pytest.mark.parametrize('case', CASES)
def test_native_effects_match_frozen_public_client(
    monkeypatch: pytest.MonkeyPatch,
    reference: dict[str, Any],
    route: str,
    case: str,
) -> None:
    monkeypatch.setenv('SDM_CORE_ROUTES', ROUTES if route == 'core' else '')
    actual = observe_case(case)
    # Raw pandas warnings remain recorded; the owner excludes their incidental
    # presentation from parity. Native events, errors and shared warnings match.
    assert {k: v for k, v in actual.items() if k != 'warnings'} == {
        k: v for k, v in reference[case].items() if k != 'warnings'
    }


def test_callback_codec_error_is_a_suppressible_native_value_error(
    monkeypatch: pytest.MonkeyPatch, reference: dict[str, Any]
) -> None:
    from sdm_client._rust.codec import CodecError

    monkeypatch.setenv('SDM_CORE_ROUTES', ROUTES)
    case = 'value_error/forward/indices'
    actual = observe_case(case, value_error=CodecError)
    assert {k: v for k, v in actual.items() if k != 'warnings'} == {
        k: v for k, v in reference[case].items() if k != 'warnings'
    }
