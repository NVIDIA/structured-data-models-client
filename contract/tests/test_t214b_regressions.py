# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public regressions for temporal candidates, semantic types and JSON.

Expected observations are checked into the repository; no old client runs.
PostgreSQL scenarios require SDM_POSTGRES_TEST_DSN, and CI can require them
with SDM_REQUIRE_POSTGRES.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / 'contract/support/temporal_candidate_scenarios.py'


def scenarios() -> ModuleType:
    name = 'temporal_candidate_scenarios'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SCENARIOS)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _dsn() -> str | None:
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
            'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
        )
        return None
    return dsn


@functools.cache
def vectors() -> dict[str, Any]:
    return scenarios().load()


@functools.cache
def core() -> dict[str, Any]:
    """Observe each scenario through the current public client."""
    return scenarios().observe(dsn=_dsn())


def test_scenarios_cover_temporal_identity_and_invalid_values() -> None:
    """The expected results exercise each named edge case."""
    document = vectors()
    parts = {k: v['parts'] for k, v in document['candidates'].items()}
    assert parts == {
        'dates-repeated': 2,
        'dates-distinct': 2,
        'aware-one-instant': 2,
        'aware-distinct': 2,
        'zone-fold': 2,
        'zone-against-utc': 3,
        'same-text-two-zones': 3,
        'times-one-instant': 2,
        'timestamp-beside-datetime64': 2,
        'naive-beside-aware': 3,
    }
    assert document['stype']['refusal']['message'].startswith(
        '[INVALID_REQUEST] 5 is not a semantic type; use one of '
    )
    assert document['where']['runtime_warnings'] == [
        'overflow encountered in cast'
    ]
    sent = json.dumps(document['json']['bodies'])
    assert document['json']['refusal'] is None
    assert '\\ud800' in sent and '\\udc00' in sent


@pytest.mark.parametrize('name', list(scenarios().CANDIDATES))
def test_temporal_candidates_preserve_identity_and_order(
    name: str,
) -> None:
    """Equivalent temporal candidates collapse; distinct instants remain distinct."""

    want = vectors()['candidates'][name]
    got = core()['candidates'][name]
    assert got == want, (
        f'{name}: {json.dumps(got)[:400]} where expected is '
        f'{json.dumps(want)[:400]}'
    )


def test_a_numeric_semantic_type_has_a_useful_refusal() -> None:
    """The refusal names the invalid value and accepted semantic types."""

    assert core()['stype'] == vectors()['stype']


def test_a_filter_reading_a_double_past_float32_warns() -> None:
    """The filter reports overflow and preserves the materialization outcome."""

    assert core()['where'] == vectors()['where']


def test_json_documents_preserve_escaped_lone_surrogates() -> None:
    """PostgreSQL JSON values retain surrogate escapes through prediction."""
    if _dsn() is None:
        pytest.skip('SDM_POSTGRES_TEST_DSN is not configured')

    want = vectors()['json']
    got = core()['json']
    assert got['refusal'] == want['refusal'], got['refusal']
    assert got['bodies'] == want['bodies'], (
        'the model requests differ from the expected JSON'
    )


# --- FLOAT32_OVERFLOW and the host NumPy version ---------------------------


def _codes_reported(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    r"""Each warning code the core reports, as the host takes its
    warnings at ``sdm_client._native``'s boundary (the one function through
    which the host imports the native module)."""
    from sdm_client._rust import codec, run

    imported = run.native
    codes: list[str] = []

    class Handle:
        def __init__(self, handle: Any) -> None:
            self._handle = handle

        def take_warnings(self) -> Any:
            taken = self._handle.take_warnings()
            if taken is not None:
                control, sent = taken
                read = codec.read(control, sent)
                codes.extend(w['code'] for w in read.get('warnings', []))
            return taken

        def __getattr__(self, name: str) -> Any:
            return getattr(self._handle, name)

    class Run:
        def __init__(self, native: Any) -> None:
            self._native = native

        def start(self, message: bytes, attachments: Any) -> Handle:
            return Handle(self._native.Run.start(message, attachments))

    class Native:
        def __init__(self, native: Any) -> None:
            self._native = native
            self.Run = Run(native)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._native, name)

    monkeypatch.setattr(run, 'native', lambda: Native(imported()))
    return codes


@pytest.mark.parametrize(
    ('numpy', 'shown'),
    [('1.23.5', False), ('1.24.0', True), ('2.3.4', True)],
)
def test_float32_overflow_is_shown_where_the_runtime_s_numpy_warns(
    numpy: str, shown: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""The core reports a double past
    float32's range (``FLOAT32_OVERFLOW``) on every runtime, and the Python
    host shows it as ``RuntimeWarning`` ("overflow encountered in cast")
    only where the installed NumPy is 1.24 or later, which first warns on
    such a cast; NumPy 1.23 casts
    silently. The version the host's function reads is set either side of
    1.24; the core's fact is there either way."""
    import numpy as np
    from sdm_client._rust import errors

    monkeypatch.setattr(np, '__version__', numpy)
    assert errors.numpy_warns_on_cast_overflow() is shown
    codes = _codes_reported(monkeypatch)
    observed = scenarios().where()
    assert observed['refusal'] is None, observed
    assert 'FLOAT32_OVERFLOW' in codes, codes
    assert observed['runtime_warnings'] == (
        ['overflow encountered in cast'] if shown else []
    ), (numpy, observed['runtime_warnings'])
