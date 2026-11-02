# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AdventureWorks classification requests declare exactly their context classes."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / 'contract/cases/s1'
IDS = ['base.adventureworks.churn', 'base.adventureworks.color']
TASK = {
    'base.adventureworks.churn': 'binary_classification',
    'base.adventureworks.color': 'multiclass_classification',
}


@pytest.mark.parametrize('case', IDS)
def test_each_request_declares_only_classes_its_context_holds(
    case: str,
    runs,
) -> None:
    # The live endpoint's TASK_SCHEMA_MISMATCH on Color over the full data:
    # a declared class in no context row. Over the fixture the reference's
    # request declares each class its context holds, and no other.
    [payload] = runs[case]['public']['payloads']
    request = payload['parsed']
    task = request['task']
    assert task['kind'] == TASK[case]
    target = task['target']
    context = request['context']['instance_table']
    column = context['columns'].index(target['column_name'])
    held = {json.dumps(row[column]) for row in context['rows']}
    declared = {
        json.dumps(_label(c, target['dtype'])) for c in target['classes']
    }
    assert declared == held - {'null'}, (declared, held)
    if case == 'base.adventureworks.color':
        assert len(declared) == 8


def _label(name: str, dtype: str) -> Any:
    r"""A declared class as the context's cell holds it."""
    return {'true': True, 'false': False}[name] if dtype == 'bool' else name


def _execute(mode: str, cases: list[Path], out: Path) -> dict[str, Any]:
    done = subprocess.run(
        [
            sys.executable,
            '-I',
            str(ROOT / 'contract/runners/run_python.py'),
            '--mode',
            mode,
            '--out',
            str(out),
            *map(str, cases),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return {
        o['case']: o
        for o in (
            json.loads(line)
            for line in out.read_text(encoding='utf-8').splitlines()
        )
    }


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    folder = tmp_path_factory.mktemp('adventureworks')
    return _execute(
        'python-core',
        [CASES / f'{case}.json' for case in IDS],
        folder / 'core.jsonl',
    )


@pytest.mark.parametrize('case', IDS)
def test_adventureworks_predicts_through_native_core(case, runs):
    observed = runs[case]
    assert observed['outcome'] == {
        'result': 'success',
        'stage': 'predict',
        'error': None,
        'code': None,
    }
    assert observed['identity']['native_loaded']
    assert {'describe_source', 'materialize', 'predict'} <= set(
        observed['identity']['operations']
    )
    assert observed['public']['result']['rows'] > 0
    assert observed['core']['error'] is None
