# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HOST-5's narrowing of the ``warnings`` rule (T1.29, decision E80) holds
each python-core warning to a python-old warning it may stand for: the T1.29
test author's seeds, from Codex's review of T1.29's first candidate.

The narrowing leaves out a ``DeprecationWarning`` python-old alone shows
whose origin is pandas or pyarrow; a warning the two routes both show is
compared by class, its origin free (the client's own warnings name the
caller's frame, reached from different modules on each route). So a runtime
``DeprecationWarning`` python-old shows is never the match of a
``DeprecationWarning`` of the client's own that python-core shows: that
would let a new client deprecation hide behind the runtime's.

- Still failing with HOST-5 approved: python-core's own
  ``DeprecationWarning`` against python-old's runtime ones only (one, two,
  around a kept warning), directly and through ``_host_fields``.
- Passing with HOST-5 approved (the converse, legitimate): a client
  deprecation both routes show, before, after or between runtime
  deprecations python-old alone shows, its origin the same or another
  non-runtime module.
- Approval still needed: each legitimate case fails unnarrowed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _compare() -> Any:
    if 'compare' in sys.modules:
        return sys.modules['compare']
    spec = importlib.util.spec_from_file_location(
        'compare', ROOT / 'scripts/rust_core/compare.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['compare'] = module
    spec.loader.exec_module(module)
    return module


compare = _compare()

RUNTIME = ('DeprecationWarning', 'pandas')
ARROW = ('DeprecationWarning', 'pyarrow')
CLIENT = ('DeprecationWarning', 'sdm_client')
ENGINE = ('DeprecationWarning', 'sdm_relational_engine')
USER = ('UserWarning', 'sdm_client')


def host(*warnings: tuple[str, str | None]) -> dict[str, Any]:
    return {
        'warnings': [name for name, _ in warnings],
        'origins': [
            {'class': name, 'module': module} for name, module in warnings
        ],
    }


#: (python-core, python-old): a client deprecation python-core alone shows,
#: python-old showing only the runtime's.
HIDDEN = {
    "Codex's: one against two": (host(CLIENT), host(RUNTIME, RUNTIME)),
    'one against one': (host(CLIENT), host(RUNTIME)),
    'one against pyarrow': (host(CLIENT), host(ARROW)),
    'after a kept warning': (
        host(USER, CLIENT),
        host(USER, RUNTIME, RUNTIME),
    ),
    'before a kept warning': (
        host(CLIENT, USER),
        host(RUNTIME, USER),
    ),
    'two against two and one kept': (
        host(CLIENT, CLIENT),
        host(RUNTIME, CLIENT, RUNTIME),
    ),
}

#: (python-core, python-old): a client deprecation both show, python-old
#: also showing the runtime's.
LEGITIMATE = {
    "Codex's converse: retained after the runtime's": (
        host(CLIENT),
        host(RUNTIME, RUNTIME, CLIENT),
    ),
    'retained before the runtime': (host(CLIENT), host(CLIENT, RUNTIME)),
    'retained between': (host(CLIENT), host(ARROW, CLIENT, RUNTIME)),
    'retained, another non-runtime origin': (
        host(CLIENT),
        host(RUNTIME, ENGINE),
    ),
    'retained with a kept warning': (
        host(USER, CLIENT),
        host(RUNTIME, USER, RUNTIME, CLIENT, ARROW),
    ),
}


@pytest.mark.parametrize('name', sorted(HIDDEN))
def test_a_client_deprecation_never_matches_a_runtime_one(name: str) -> None:
    core, old = HIDDEN[name]
    assert not compare._runtime_only(core, old, True)


@pytest.mark.parametrize('name', sorted(LEGITIMATE))
def test_a_retained_client_deprecation_beside_the_runtime_s_passes(
    name: str,
) -> None:
    core, old = LEGITIMATE[name]
    assert compare._runtime_only(core, old, True)
    # Approval still needed.
    assert not compare._runtime_only(core, old, False)


def _observation(warnings: dict[str, Any]) -> dict[str, Any]:
    return {
        'host': warnings,
        'public': {'schema': None, 'result': None, 'payloads': []},
    }


RULES = {
    'schema': 'gate',
    'warnings': 'gate',
    'result': 'gate',
    'payload': 'gate',
}


def _host_fields(core: dict[str, Any], old: dict[str, Any]) -> Any:
    report = compare.Report(RULES)
    report.narrowed = True
    compare._host_fields(
        report,
        {'id': 'base.regression'},
        'python-core',
        _observation(core),
        _observation(old),
        'python-old',
    )
    return report


def test_codex_s_case_fails_the_warnings_rule_through_host_fields() -> None:
    core, old = HIDDEN["Codex's: one against two"]
    report = _host_fields(core, old)
    assert any('host.warnings' in failure for failure in report.failures), (
        report.failures
    )


def test_codex_s_converse_passes_through_host_fields() -> None:
    core, old = LEGITIMATE["Codex's converse: retained after the runtime's"]
    report = _host_fields(core, old)
    assert report.failures == [], report.failures
    assert any('HOST-5' in note for note in report.notes)
