# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A panic in the Rust core reaches a caller as an ``sdm_client`` error.

A panic is a bug in the core, never a refusal of what the caller gave, so
no input is known to cause one; the binding catches each and raises it as
``sdm_core._native.CorePanic``, a ``ValueError``. These tests stand a panic
in for the binding's answer, at a call the host makes directly and at a
step of a run it drives, and hold the public call to raise a
``StructuredDataError`` with the internal code, with nothing of the
binding beneath it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

_native = pytest.importorskip('sdm_core._native')

from sdm_client import PredictiveQuery, Schema, _impl  # noqa: E402
from sdm_client.errors import StructuredDataError  # noqa: E402

PANIC = 'the core panicked; this is a bug in it'
REPO = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.skipif(
    _impl.OLD, reason='the old implementation has no core to panic'
)


def _panic() -> BaseException:
    r"""What the binding raises when the core panics."""
    return getattr(_native, 'CorePanic', ValueError)(PANIC)


def _frames() -> dict[str, pd.DataFrame]:
    users = pd.DataFrame({'user_id': [1, 2, 3], 'age': [20, 30, 40]})
    orders = pd.DataFrame(
        {
            'order_id': range(1, 31),
            'user_id': [1 + i % 3 for i in range(30)],
            'ordered_at': pd.date_range('2024-01-01', periods=30, freq='D'),
        }
    )
    return {'users': users, 'orders': orders}


def _held(error: pytest.ExceptionInfo[StructuredDataError]) -> None:
    raised = error.value
    assert type(raised) is StructuredDataError
    assert raised.code == 'INTERNAL_ERROR'
    assert PANIC in str(raised)
    assert 'report' in str(raised)
    beneath: BaseException | None = raised
    seen = []
    while beneath is not None:
        seen.append(type(beneath).__module__)
        beneath = beneath.__cause__ or (
            None if beneath.__suppress_context__ else beneath.__context__
        )
    assert not [m for m in seen if m.startswith(('sdm_core', 'pyo3'))], seen


@pytest.mark.parametrize('implementation', ['core', 'python'])
def test_recorded_host_events(implementation):
    environment = {**os.environ, 'SDM_CLIENT_IMPLEMENTATION': implementation}
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / 'scripts/parity/make_fixtures.py'),
            '--host-events',
            '--check',
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('defect', ['warning', 'panic'])
def test_host_recordings_detect_seeded_mapping_regressions(defect):
    script = """
import json
import runpy
import sys
from sdm_client import _core
module = runpy.run_path('scripts/parity/make_fixtures.py')
if sys.argv[1] == 'warning':
    _core._category = lambda qualified: UserWarning
else:
    _core.panicked = lambda error: ValueError(str(error))
actual = module['capture_host_events']()
expected = json.loads(module['HOST_EVENTS'].read_text())
assert actual != expected, 'the seeded mapping regression escaped the recording'
"""
    result = subprocess.run(
        [sys.executable, '-c', script, defect],
        cwd=REPO,
        env={**os.environ, 'SDM_CLIENT_IMPLEMENTATION': 'core'},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_binding_names_a_panic_apart_from_a_refusal() -> None:
    assert issubclass(_native.CorePanic, ValueError)
    assert not issubclass(_native.CorePanic, _native.CoreError)


@pytest.mark.parametrize('blocked', ['sdm_client', 'sdm_connectors'])
def test_view_warning_does_not_add_an_install_dependency(blocked):
    script = """
import importlib.abc
import sys
class WithoutOptionalPackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] == sys.argv[1]:
            raise ModuleNotFoundError('optional package deliberately absent')
sys.meta_path.insert(0, WithoutOptionalPackage())
if sys.argv[1] == 'sdm_client':
    from sdm_relational_engine.rfm.graph import ViewConversionWarning
    assert issubclass(ViewConversionWarning, UserWarning)
else:
    import sdm_client
    from sdm_client import _core
    assert _core._category('builtins.UserWarning') is UserWarning
"""
    result = subprocess.run(
        [sys.executable, '-c', script, blocked],
        cwd=REPO,
        env={**os.environ, 'SDM_CLIENT_IMPLEMENTATION': 'core'},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_panic_in_a_call_is_an_internal_error(monkeypatch) -> None:
    def panics(*args: Any) -> Any:
        raise _panic()

    monkeypatch.setattr(_native, 'data_schema', panics)
    with pytest.raises(StructuredDataError) as error:
        Schema.from_data(_frames())
    _held(error)


class _PanickingRun:
    r"""A run whose first step panics, as the binding's run raises it."""

    def next(self) -> Any:
        raise _panic()

    def submit(self, *args: Any) -> None:  # pragma: no cover - never asked
        raise AssertionError('no step was given')


def test_a_panic_in_a_step_of_a_run_is_an_internal_error(monkeypatch) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        query = PredictiveQuery(
            'PREDICT COUNT(orders.*, 0, 7, days) FOR EACH users.user_id',
            Schema.from_data(_frames()),
        )
    monkeypatch.setattr(
        _native, 'materialize_run', lambda *args: _PanickingRun()
    )
    with pytest.raises(StructuredDataError) as error:
        query.materialize(indices=[1], random_seed=7)
    _held(error)
