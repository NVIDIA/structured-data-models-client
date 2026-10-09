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
