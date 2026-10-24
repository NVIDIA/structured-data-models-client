# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The existing runner installs the existing guard before any live call."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    path = ROOT / 'contract/runners/run_python.py'
    spec = importlib.util.spec_from_file_location('t310_guard_runner', path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def test_live_guard_wraps_runner_calls_and_is_released_on_failure(
    tmp_path, monkeypatch
):
    run = _runner()
    events = []
    loader = object()
    guard = object()
    helper = SimpleNamespace(
        load_guard=lambda: guard,
        install_guard=lambda leg, seen, actual: (
            events.append(('install', leg, seen is guard, actual is loader))
            or (lambda: events.append('uninstall'))
        ),
    )
    monkeypatch.setattr(run, '_capture_run', lambda: helper)
    monkeypatch.setattr(run, '_sibling', lambda name: loader)
    monkeypatch.setattr(
        run, 'load_cases', lambda paths: [({'id': 'dummy.guard'}, 'digest')]
    )
    monkeypatch.setattr(run, 'inventory_cases', lambda ids: [])
    monkeypatch.setattr(run, 'environment_refusal', lambda env: None)

    def fail(*args, **kwargs):
        assert events == [('install', 'snowflake', True, True)]
        raise run.RunnerFailure('owned test refusal')

    monkeypatch.setattr(run, 'run', fail)
    assert (
        run.main(
            [
                '--mode',
                'python-core',
                '--database',
                'snowflake',
                '--guard-live',
                '--out',
                str(tmp_path / 'out.jsonl'),
                str(tmp_path / 'case.json'),
            ]
        )
        == 1
    )
    assert events[-1] == 'uninstall'


def test_live_guard_option_refuses_an_offline_leg_before_running(tmp_path):
    run = _runner()
    with pytest.raises(SystemExit) as refused:
        run.main(
            [
                '--mode',
                'python-core',
                '--guard-live',
                '--out',
                str(tmp_path / 'out.jsonl'),
            ]
        )
    assert refused.value.code == 2
