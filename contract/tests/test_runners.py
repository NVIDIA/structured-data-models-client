# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The S0 cases on the Python runners, and the runner's controls (T1.10).

Run in an environment with the candidate client (``sdm_client._native``
included), the relational engine and DuckDB, as CI's ``engine`` job has:

    python -m pytest contract/tests -q

The two S0 cases agree on ``python-core`` and ``python-old`` with their
committed reference observation. The runner records a known
``RuntimeError`` refusal as the case's outcome and the comparator accepts
it; the comparator rejects an unexpected ``RuntimeError`` on a case that
should succeed; a process abort, a refused forced route and a failing
reference guard are runner failures that write nothing.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
COMPARE = ROOT / 'scripts/rust_core/compare.py'
CASES = ROOT / 'contract/cases'
REFERENCE = next((ROOT / 'contract/reference').glob('*.jsonl'))
ACCEPTED = CASES / 's0/s0.accept.sum-window.json'


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_python = _module(RUNNER, 'run_python')


def _run(
    mode: str, out: Path, *cases: Path, reference: Path | None = None
) -> subprocess.CompletedProcess[str]:
    arguments = [sys.executable, '-I', str(RUNNER), '--mode', mode]
    if reference is not None:
        arguments += ['--reference', str(reference)]
    arguments += ['--out', str(out), *map(str, cases)]
    return subprocess.run(
        arguments, capture_output=True, text=True, check=False
    )


def _compare(
    cases: Path, *observations: Path, reference: bool = True
) -> subprocess.CompletedProcess[str]:
    arguments = [sys.executable, str(COMPARE), '--cases', str(cases)]
    if reference:
        arguments += ['--reference', str(REFERENCE)]
    arguments += [str(path) for path in observations]
    return subprocess.run(
        arguments, capture_output=True, text=True, check=False
    )


def _write_case(directory: Path, case: dict[str, Any]) -> Path:
    directory.mkdir(exist_ok=True)
    path = directory / f'{case["id"]}.json'
    path.write_text(json.dumps(case), encoding='utf-8')
    return path


def _observations(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_s0_cases_agree_with_the_reference_on_both_python_runners(
    tmp_path: Path,
) -> None:
    core, old = tmp_path / 'python-core.jsonl', tmp_path / 'python-old.jsonl'
    for mode, out in (('python-core', core), ('python-old', old)):
        run = _run(mode, out, CASES)
        assert run.returncode == 0, run.stderr
    # Every case, base.regression's diagnostic one included (T1.11b), whose
    # differences the comparator reports and gates none.
    compared = _compare(CASES, core, old)
    assert compared.returncode == 0, compared.stdout
    assert 'DIFFERS' not in compared.stdout
    operations = {
        'S0': ['validate_query'],
        # From T1.11c base.regression's materialize runs on the core too,
        # and from T1.11d its predict.
        'S1': ['describe_source', 'materialize', 'predict'],
    }
    for observation in _observations(core):
        slice_ = next(
            case['slice']
            for case, _ in run_python.load_cases([CASES])
            if case['id'] == observation['case']
        )
        assert observation['identity']['operations'] == operations[slice_]
        assert observation['identity']['native'] is not None
    for observation in _observations(old):
        assert observation['identity']['native_loaded'] is False


def test_a_known_runtime_error_refusal_is_recorded_as_the_outcome(
    tmp_path: Path,
) -> None:
    inventory = run_python.load_inventory()
    entry = next(
        item for item in inventory['cases'] if item['id'] == 'nested.struct'
    )
    assert entry['expect']['error'] == 'RuntimeError'
    case = run_python.inventory_case(entry, inventory['defaults'])
    # Gated here, so the comparator's acceptance is what is tested.
    case['status'] = 'acceptance'
    cases = tmp_path / 'cases'
    _write_case(cases, case)
    out = tmp_path / 'python-old.jsonl'
    run = _run('python-old', out, cases)
    assert run.returncode == 0, run.stderr
    [observation] = _observations(out)
    assert observation['outcome'] == {
        'result': 'refusal',
        'stage': 'from_source',
        'error': 'RuntimeError',
        'code': None,
    }
    compared = _compare(cases, out, reference=False)
    assert compared.returncode == 0, compared.stdout


def test_an_unexpected_runtime_error_on_a_success_case_is_rejected(
    tmp_path: Path,
) -> None:
    case = json.loads(ACCEPTED.read_text())
    # A struct column the reference's schema inference refuses with a
    # RuntimeError, under a case that expects success.
    case['input']['setup'] = [
        "CREATE OR REPLACE TABLE users AS SELECT *, {'a': user_id::INTEGER} "
        'AS f FROM users'
    ]
    cases = tmp_path / 'cases'
    _write_case(cases, case)
    out = tmp_path / 'python-old.jsonl'
    run = _run('python-old', out, cases)
    assert run.returncode == 0, run.stderr
    [observation] = _observations(out)
    assert observation['outcome']['error'] == 'RuntimeError'
    compared = _compare(cases, out, reference=False)
    assert compared.returncode == 1
    assert (
        f"{case['id']}: python-old: outcome.result: expected 'success', "
        "observed 'refusal'"
    ) in compared.stdout


def test_a_process_abort_fails_the_runner(tmp_path: Path) -> None:
    out = tmp_path / 'python-old.jsonl'
    aborting = textwrap.dedent(
        f"""
        import os, runpy, sys
        runner = runpy.run_path({str(RUNNER)!r}, run_name='run_python')
        import sdm_client
        sdm_client.PredictiveQuery.materialize = lambda *a, **k: os.abort()
        sys.exit(runner['main'](
            ['--mode', 'python-old', '--out', {str(out)!r}, {str(CASES)!r}]
        ))
        """
    )
    run = subprocess.run(
        [sys.executable, '-I', '-c', aborting],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert not out.exists()
    compared = _compare(CASES, out)
    assert compared.returncode == 1
    assert 'runner failure: no observations written' in compared.stdout


def test_a_refused_forced_route_fails_the_runner(tmp_path: Path) -> None:
    case = json.loads(ACCEPTED.read_text())
    # One of S0's two paths alone: one call runs them on one route.
    case['routes'] = ['query.validate/accept/any']
    cases = tmp_path / 'cases'
    _write_case(cases, case)
    out = tmp_path / 'python-core.jsonl'
    out.write_text("an earlier run's observations\n")
    run = _run('python-core', out, cases)
    assert run.returncode == 1
    assert 'runner failure' in run.stderr and 'RouteError' in run.stderr
    # Not even an earlier run's output survives a failed run.
    assert not out.exists()


def test_an_input_the_harness_cannot_build_fails_the_runner(
    tmp_path: Path,
) -> None:
    case = json.loads(ACCEPTED.read_text())
    case['input']['materialize']['anchor_time'] = '2024-13-45'
    cases = tmp_path / 'cases'
    _write_case(cases, case)
    out = tmp_path / 'python-old.jsonl'
    run = _run('python-old', out, cases)
    assert run.returncode == 1
    assert 'its materialize arguments could not be built' in run.stderr
    assert not out.exists()


def _reference(tmp_path: Path, guard: str) -> Path:
    reference = tmp_path / 'reference'
    (reference / 'scripts').mkdir(parents=True)
    (reference / 'scripts/reference_guard.py').write_text(guard)
    return reference


def test_python_ref_stops_when_the_guard_fails(tmp_path: Path) -> None:
    reference = _reference(
        tmp_path,
        'def require_isolation():\n'
        "    raise RuntimeError('the Rust core can be loaded')\n",
    )
    out = tmp_path / 'python-ref.jsonl'
    run = _run('python-ref', out, CASES, reference=reference)
    assert run.returncode == 1
    assert 'the reference guard failed before the first case' in run.stderr
    assert not out.exists()


def test_python_ref_stops_when_sdm_client_is_not_the_references(
    tmp_path: Path,
) -> None:
    reference = _reference(tmp_path, 'def require_isolation():\n    pass\n')
    out = tmp_path / 'python-ref.jsonl'
    run = _run('python-ref', out, CASES, reference=reference)
    assert run.returncode == 1
    assert "sdm_client is not the reference checkout's" in run.stderr
    assert not out.exists()


def test_python_ref_stops_when_the_guard_fails_after_the_last_case(
    tmp_path: Path,
) -> None:
    reference = _reference(
        tmp_path,
        'calls = []\n'
        'def require_isolation():\n'
        '    calls.append(1)\n'
        '    if len(calls) > 1:\n'
        "        raise RuntimeError('the Rust core was loaded')\n",
    )
    out = tmp_path / 'python-ref.jsonl'
    # The candidate's sdm_client stands in for the reference's here, so the
    # check that it is the reference checkout's is replaced.
    script = textwrap.dedent(
        f"""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location('run_python', {str(RUNNER)!r})
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        runner._require_reference_client = lambda reference: None
        sys.exit(runner.main([
            '--mode', 'python-ref', '--reference', {str(reference)!r},
            '--out', {str(out)!r}, {str(CASES)!r},
        ]))
        """
    )
    run = subprocess.run(
        [sys.executable, '-I', '-c', script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 1, run.stderr
    assert 'the reference guard failed after the last case' in run.stderr
    assert not out.exists()
