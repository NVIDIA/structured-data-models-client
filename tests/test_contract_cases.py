# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contract's case, inventory and observation formats (T1.10).

The Phase 1b inventory validates against its schema, and the inventory
loader maps every one of its cases into the case format and back without
loss; the bases the cases build their data from are the inventory's; every
committed case and reference observation validates, and the reference meets
each case's expectation. Running the cases needs the relational engine, so
that lives in ``contract/tests``, which CI's ``engine`` job runs.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / 'contract/runners/run_python.py'
COMPARE = ROOT / 'scripts/rust_core/compare.py'
CASES = ROOT / 'contract/cases'
REFERENCE = ROOT / 'contract/reference'


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location('run_python', RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_python = _runner()


def _inventory() -> dict[str, Any]:
    inventory: dict[str, Any] = run_python.load_inventory()
    return inventory


def test_the_inventory_validates_against_its_schema() -> None:
    inventory = _inventory()
    assert len(inventory['cases']) == 40
    validate = run_python.validator('inventory')
    broken = copy.deepcopy(inventory)
    broken['cases'][0]['seam'] = 'no-such-seam'
    assert run_python._invalid(validate, broken) is not None
    del broken['cases'][0]['seam']
    assert run_python._invalid(validate, broken) is not None


def test_the_inventory_loader_maps_every_case_without_loss() -> None:
    inventory = _inventory()
    defaults = inventory['defaults']
    validate = run_python.validator('case')
    for entry in inventory['cases']:
        case = run_python.inventory_case(entry, defaults)
        assert run_python._invalid(validate, case) is None, entry['id']
        assert run_python.inventory_entry(case, defaults) == entry
    # The mapping keeps what the inventory left to its defaults apart from
    # what it stated, and refuses a case it has no place for.
    entry = next(item for item in inventory['cases'] if 'pql' not in item)
    case = run_python.inventory_case(entry, defaults)
    assert case['input']['pql'] == defaults['pql']
    case['status'] = 'acceptance'
    with pytest.raises(run_python.RunnerFailure):
        run_python.inventory_entry(case, defaults)


def test_the_bases_are_the_inventorys() -> None:
    assert run_python.load_bases() == _inventory()['bases']


def test_every_committed_case_validates_with_a_unique_id() -> None:
    cases = run_python.load_cases([CASES])
    ids = [case['id'] for case, _ in cases]
    assert len(ids) == len(set(ids)) >= 2
    s0 = [case for case, _ in cases if case['slice'] == 'S0']
    assert {case['expect']['outcome'] for case in s0} == {'success', 'refusal'}


def test_the_reference_observations_validate_and_meet_each_expectation(
    tmp_path: Path,
) -> None:
    files = sorted(REFERENCE.glob('*.jsonl'))
    assert len(files) == 1
    reference = files[0]
    validate = run_python.validator('observation')
    observations = [
        json.loads(line) for line in reference.read_text().splitlines()
    ]
    for observation in observations:
        assert run_python._invalid(validate, observation) is None
        identity = observation['identity']
        assert identity['runner'] == 'python-ref'
        assert identity['reference']['commit']['sha'] == reference.stem
        assert identity['native_loaded'] is False
    run = subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            '--cases',
            str(CASES),
            '--reference',
            str(reference),
            str(reference),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    # A corrupted expectation fails.
    corrupted = tmp_path / 'cases'
    corrupted.mkdir()
    for path in CASES.rglob('*.json'):
        case = json.loads(path.read_text())
        if case['expect']['outcome'] == 'refusal':
            case['expect']['code'] = 'INTERNAL_ERROR'
        (corrupted / path.name).write_text(json.dumps(case))
    run = subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            '--cases',
            str(corrupted),
            '--reference',
            str(reference),
            str(reference),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 1
    assert "outcome.code: expected 'INTERNAL_ERROR'" in run.stdout


def test_both_loaders_refuse_an_unknown_key_inside_the_call_arguments(
    tmp_path: Path,
) -> None:
    # sdm-testkit's tests/s0_cases.rs refuses the same key.
    case = json.loads((CASES / 's0/s0.accept.sum-window.json').read_text())
    case['input']['materialize']['ignored_typo'] = True
    path = tmp_path / 'case.json'
    path.write_text(json.dumps(case))
    with pytest.raises(run_python.RunnerFailure, match='ignored_typo|fails'):
        run_python.load_cases([path])


# --- the comparator against seeded observations ---------------------------------


def _reference() -> Path:
    [reference] = sorted(REFERENCE.glob('*.jsonl'))
    return reference


def _seeded(
    tmp_path: Path, change: Any, name: str = 'seeded.jsonl'
) -> subprocess.CompletedProcess[str]:
    r"""compare.py on the reference observations as ``change`` rewrites
    each, the reference itself as the oracle."""
    reference = _reference()
    lines = []
    for line in reference.read_text().splitlines():
        observation = json.loads(line)
        change(observation)
        lines.append(json.dumps(observation) + '\n')
    seeded = tmp_path / name
    seeded.write_text(''.join(lines))
    return subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            '--cases',
            str(CASES),
            '--reference',
            str(reference),
            str(seeded),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _as_old(observation: dict[str, Any]) -> None:
    observation['identity']['runner'] = 'python-old'
    observation['identity']['reference'] = None


def test_the_reference_relabelled_python_old_passes(tmp_path: Path) -> None:
    run = _seeded(tmp_path, _as_old)
    assert run.returncode == 0, run.stdout


def test_an_empty_observations_file_is_a_runner_failure(
    tmp_path: Path,
) -> None:
    empty = tmp_path / 'empty.jsonl'
    empty.write_text('')
    run = subprocess.run(
        [sys.executable, str(COMPARE), '--cases', str(CASES), str(empty)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 1
    assert 'empty.jsonl: runner failure: no valid observations' in run.stdout


def test_a_python_runner_may_not_leave_its_stage_or_class_null(
    tmp_path: Path,
) -> None:
    def blank(observation: dict[str, Any]) -> None:
        _as_old(observation)
        observation['outcome']['stage'] = None
        observation['outcome']['error'] = None

    run = _seeded(tmp_path, blank)
    assert run.returncode == 1
    assert 'outcome.stage: null on a Python runner' in run.stdout
    assert 'outcome.error: a refusal without a class' in run.stdout


def test_an_observation_of_another_case_file_fails(tmp_path: Path) -> None:
    def elsewhere(observation: dict[str, Any]) -> None:
        _as_old(observation)
        observation['identity']['inputs']['case'] = '0' * 64

    run = _seeded(tmp_path, elsewhere)
    assert run.returncode == 1
    assert 'identity.inputs.case: not this case file' in run.stdout

    def other_bases(observation: dict[str, Any]) -> None:
        _as_old(observation)
        observation['identity']['inputs']['bases'] = '0' * 64

    run = _seeded(tmp_path, other_bases)
    assert 'identity.inputs.bases: not this bases file' in run.stdout


def test_a_python_runner_at_another_duckdb_fails(tmp_path: Path) -> None:
    def older(observation: dict[str, Any]) -> None:
        _as_old(observation)
        observation['identity']['versions']['duckdb'] = '1.5.5'

    run = _seeded(tmp_path, older)
    assert run.returncode == 1
    assert 'identity.versions.duckdb' in run.stdout


def _as_core(accepted: bool | None, error: Any = None) -> Any:
    def change(observation: dict[str, Any]) -> None:
        case = json.loads(
            next(
                path
                for path in CASES.rglob('*.json')
                if json.loads(path.read_text())['id'] == observation['case']
            ).read_text()
        )
        spec = case['core']['validate_query']
        identity = observation['identity']
        identity.update(
            runner='python-core',
            reference=None,
            native={'sha256': 'a' * 64},
            native_loaded=True,
            operations=['validate_query'],
            contract=4,
        )
        observation['core'] = {
            'inputs': [
                {
                    'op': 'validate_query',
                    'query': case['input']['pql'],
                    'profile': spec['profile'],
                    'schema': spec['schema'],
                }
            ],
            'accepted': (
                observation['outcome']['result'] == 'success'
                if accepted is None
                else accepted
            ),
            'error': error,
        }

    return change


def test_python_cores_answer_is_held_to_its_own_outcome(
    tmp_path: Path,
) -> None:
    run = _seeded(tmp_path, _as_core(None))
    assert run.returncode == 0, run.stdout
    # Every case's core accepting, whatever the call did.
    run = _seeded(tmp_path, _as_core(True))
    assert run.returncode == 1
    assert (
        's0.refuse.unknown-column: python-core: outcome against core: '
        "expected ('success', None), observed ('refusal', 'INVALID_REQUEST')"
    ) in run.stdout


def test_an_index_is_an_integer_literal_of_any_size_or_text(
    tmp_path: Path,
) -> None:
    # The same literals sdm-testkit's tests/s0_cases.rs reads.
    literals = json.loads(
        (ROOT / 'contract/tests/index_literals.json').read_text()
    )
    case = json.loads((CASES / 's0/s0.accept.sum-window.json').read_text())
    case['input']['materialize']['indices'] = ['@index@']
    template = json.dumps(case)

    def load(literal: str) -> Any:
        path = tmp_path / 'case.json'
        path.write_text(template.replace('"@index@"', literal))
        [(loaded, _)] = run_python.load_cases([path])
        return loaded

    for literal in literals['accepted']:
        loaded = load(literal)
        [index] = loaded['input']['materialize']['indices']
        assert index == json.loads(literal), literal
        assert type(index) is type(json.loads(literal))
        # As a decimal index, every digit is kept.
        loaded['input']['materialize']['indices_as'] = 'decimal'
        if isinstance(index, int):
            [decimal] = run_python._materialize_arguments(
                loaded['input']['materialize']
            )['indices']
            assert decimal == Decimal(literal) and str(decimal) == literal
    for literal in literals['refused']:
        with pytest.raises(run_python.RunnerFailure):
            load(literal)
