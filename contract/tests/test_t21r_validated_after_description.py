# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1r: a ``validate_query`` error after a description ends the call.

A case whose core describes its source and then validates its query may
end with the description and ``validate_query``'s error: a packed date
literal is ``INVALID_REQUEST`` there (DT-5), as in T2.13's case
date.day-first-then-packed. The comparator takes that pair as the
edit_schema branch takes a refused edit (decision E67): the outcome is a
refusal with the error's code, and a different outcome is the
outcome-against-core difference. A description and an error with no
``validate_query`` step still fail as before.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module(path: Path, name: str) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


COMPARE = _module(ROOT / 'scripts/rust_core/compare.py', 't21r_compare')
ID = 'date.day-first-then-packed'
CASE = {
    'id': ID,
    'core': {'describe_source': {}, 'validate_query': {}},
}
ERROR = {
    'kind': 'invalid_request',
    'code': 'INVALID_REQUEST',
    'fields': {
        'operation': 'validate_query',
        'rule': 'date_literal_packed',
        'span': [0, 8],
    },
}
WHO = ('python-core', 'rust-host')


def _failures(who: str, operations: list[str], code: str | None) -> list[str]:
    r"""The outcome rule's failures for a core that answered a
    description and then an error, its calls ``operations``, and an
    outcome refused with ``code``."""
    observation: dict[str, Any] = {
        'identity': {'operations': operations},
        'core': {
            'inputs': [{'op': op} for op in operations],
            'accepted': None,
            'described': {'tables': []},
            'materialized': None,
            'predicted': None,
            'error': ERROR,
            'fetches': [{}],
        },
        'outcome': {'result': 'refusal', 'code': code},
    }
    report = COMPARE.Report()
    COMPARE._core_consistent(report, CASE, who, observation)
    return [text for _, text in report.failures]


@pytest.mark.parametrize('who', WHO)
def test_a_validate_query_error_after_a_description_is_its_refusal(
    who: str,
) -> None:
    validated = ['describe_source', 'validate_query']
    assert _failures(who, validated, 'INVALID_REQUEST') == []


@pytest.mark.parametrize('who', WHO)
def test_an_outcome_unlike_the_validate_query_error_differs(who: str) -> None:
    validated = ['describe_source', 'validate_query']
    assert _failures(who, validated, 'UNSUPPORTED_REQUEST') == [
        f'{ID}: {who}: outcome against core: expected '
        "('refusal', 'INVALID_REQUEST'), observed "
        "('refusal', 'UNSUPPORTED_REQUEST')"
    ]


@pytest.mark.parametrize('who', WHO)
def test_a_description_and_an_error_with_no_validate_query_step_fail(
    who: str,
) -> None:
    wanted = [f'{ID}: {who}: core: not one description or one error']
    assert _failures(who, ['describe_source'], 'INVALID_REQUEST') == wanted
    # Validated and then another call: validate_query did not end it.
    later = ['describe_source', 'validate_query', 'materialize']
    assert _failures(who, later, 'INVALID_REQUEST') == wanted
