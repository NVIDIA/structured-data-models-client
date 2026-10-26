# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The selftest command keeps failures visible in the handoff's short tail."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def command(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/rust_core/selftest.py'
    spec = importlib.util.spec_from_file_location(
        't44_selftest_reporting', path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.l0_deps, 'cargo_metadata', lambda _: {})
    for section in ('freeze_cases', 'scan_cases', 'clippy_case'):
        monkeypatch.setattr(module, section, lambda _: None)
    monkeypatch.setattr(module, 'parser_name_cases', lambda _, __: None)

    def run(results):
        monkeypatch.setattr(
            module, 'deps_cases', lambda found, _: found.extend(results)
        )
        return module.main([])

    return module, run


@pytest.mark.parametrize('failure_count', [1, 6])
def test_early_failures_and_nonzero_exit_survive_the_handoff_tail(
    command, capsys, failure_count
):
    module, run = command
    failures = [
        module.Result(f'early-{n}', False, f'guard refused-{n}')
        for n in range(failure_count)
    ]
    successes = [module.Result(f'later-{n}', True, 'passed') for n in range(40)]
    assert run([*failures, *successes]) == 1
    lines = capsys.readouterr().out.splitlines()
    tail = lines[-25:]
    expected = [
        f'FAIL early-{n}: guard refused-{n}' for n in range(failure_count)
    ]
    assert all(line in tail for line in expected)
    assert [line for line in lines if line.startswith('FAIL ')] == expected
    assert lines[:40] == [f'ok   later-{n}: passed' for n in range(40)]
    assert tail[-1] == f'40 of {40 + failure_count} self-tests as expected'


def test_all_success_output_keeps_its_existing_order_and_exit(command, capsys):
    module, run = command
    results = [
        module.Result('first', True, 'one'),
        module.Result('second', True, 'two'),
        module.Result('third', True, 'three'),
    ]
    assert run(results) == 0
    assert capsys.readouterr().out.splitlines() == [
        'ok   first: one',
        'ok   second: two',
        'ok   third: three',
        '3 of 3 self-tests as expected',
    ]
