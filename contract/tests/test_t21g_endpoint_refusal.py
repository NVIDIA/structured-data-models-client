# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1g's outcome rule for an endpoint's refusal at ``predict``: the
client raises it as the endpoint's own refusal, ``NimRequestError`` with no
code of the client's (decision E47), and the core, told the endpoint's
status, reports the same call as ``INVALID_RESPONSE`` (decision E45). The
two agree there: python-core's public outcome against its core's error,
and rust-host's code, its core's, against the reference's outcome and the
case's expectation. Every other pairing still fails: another core code, a
refusal of another class or carrying a code, and one at another stage.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _module(path: Path, name: str) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


COMPARE = _module(ROOT / 'scripts/rust_core/compare.py', 't21g_refusal')
REFUSAL = {
    'result': 'refusal',
    'stage': 'predict',
    'error': 'NimRequestError',
    'code': None,
}
CASE = {
    'id': 'refusal.probe',
    'until': 'predict',
    'core': {'materialize': {}, 'predict': {}},
    'expect': {
        'outcome': 'refusal',
        'stage': 'predict',
        'error': 'NimRequestError',
        'code': None,
    },
}


def _python_core(outcome: dict[str, Any], core_code: str) -> list[str]:
    r"""python-core's outcome rule against its core's error at predict."""
    observation = {
        'core': {
            'accepted': None,
            'described': None,
            'materialized': {'version': 2},
            'predicted': None,
            'error': {'kind': 'model_endpoint', 'code': core_code},
            'fetches': [{}],
        },
        'outcome': outcome,
    }
    report = COMPARE.Report()
    COMPARE._core_consistent(report, CASE, 'python-core', observation)
    return [text for _, text in report.failures]


def _rust_host(code: str | None, wanted: dict[str, Any]) -> list[str]:
    r"""rust-host's outcome, its core's ``code``, against the reference's
    outcome ``wanted`` and against the case expecting it."""
    outcome = {'result': 'refusal', 'stage': None, 'error': None, 'code': code}
    case = {
        **CASE,
        'expect': {
            'outcome': wanted['result'],
            'stage': wanted['stage'],
            'error': wanted['error'],
            'code': wanted['code'],
        },
    }
    report = COMPARE.Report()
    COMPARE._expectation(report, case, 'rust-host', outcome)
    COMPARE._against_reference(
        report, case, 'rust-host', outcome, {'outcome': wanted}
    )
    COMPARE._effective(report, case, 'rust-host', outcome, {}, None)
    return [text for _, text in report.failures]


def test_an_endpoint_s_refusal_is_the_core_s_invalid_response() -> None:
    assert _python_core(REFUSAL, 'INVALID_RESPONSE') == []
    assert _rust_host('INVALID_RESPONSE', REFUSAL) == []


def test_every_other_pairing_still_fails() -> None:
    # Another core code beside the endpoint's refusal.
    assert _python_core(REFUSAL, 'TRANSPORT_ERROR')
    assert _rust_host('TRANSPORT_ERROR', REFUSAL)
    # A refusal of another class, or carrying a code.
    for other in (
        {**REFUSAL, 'error': 'StructuredDataError'},
        {**REFUSAL, 'code': 'INVALID_REQUEST'},
    ):
        assert _python_core(other, 'INVALID_RESPONSE'), other
        assert _rust_host('INVALID_RESPONSE', other), other
    # The same refusal at another stage.
    elsewhere = {**REFUSAL, 'stage': 'materialize'}
    assert _rust_host('INVALID_RESPONSE', elsewhere)
    # A success, against the core's error.
    success = {
        'result': 'success',
        'stage': 'predict',
        'error': None,
        'code': None,
    }
    assert _python_core(success, 'INVALID_RESPONSE')
