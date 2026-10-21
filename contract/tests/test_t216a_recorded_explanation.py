# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.16a's exit on ``run_python.py`` (the package's test author's): the
runner records a result's explanation and its items' metadata, so the
gate's ``core`` rule compares them across hosts (T2.13's exits; T2.1n
round 1, P3 3). ``sdm-testkit``'s ``tests/t216a_recorded_explanation.rs``
holds ``rust-host`` to the same.

Where it is recorded: ``core.predicted.result.explanation``, the JSON text
the core put on the result's Arrow schema metadata under ``explanation``,
absent where the core gives none; each item's metadata and explanation are
the result rows' ``metadata`` and ``explanation`` columns.

The values are the reference's, from ``contract/t213-option-vectors.json``
(``option_record.py``): a forecast's whole explanation
(``{"by_forecast_step": ...}``) is on the schema metadata only, so
``explain.forecast`` is the case that tells a recorded explanation from
the rows' own column.
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
COMPARE = ROOT / 'scripts/rust_core/compare.py'
CASES = ROOT / 'contract/cases/s1'
VECTORS = ROOT / 'contract/t213-option-vectors.json'
EXPLAINED = ('explain.answered', 'explain.forecast', 'metadata.items-explained')
UNEXPLAINED = ('explain.not-answered', 'metadata.items')
WITH_METADATA = ('metadata.items', 'metadata.items-across-calls')
CASE_IDS = (*EXPLAINED, *UNEXPLAINED, 'metadata.items-across-calls')


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@functools.cache
def vectors() -> dict[str, Any]:
    document = json.loads(VECTORS.read_text(encoding='utf-8'))
    assert document['recorded_by'] == 'contract/tools/option_record.py'
    return document['vectors']


@functools.cache
def core_run() -> dict[str, Any]:
    r"""python-core over the cases on DuckDB: each observation by case, or
    under ``''`` why the run failed whole."""
    with tempfile.TemporaryDirectory(prefix='t216a-') as tmp:
        given = Path(tmp) / 'cases'
        given.mkdir()
        for case_id in CASE_IDS:
            shutil.copy(CASES / f'{case_id}.json', given)
        out = Path(tmp) / 'python-core.jsonl'
        done = subprocess.run(
            [
                sys.executable, '-I', str(RUNNER), '--mode', 'python-core',
                '--out', str(out), str(given),
            ],
            capture_output=True, text=True, cwd=ROOT,
            stdin=subprocess.DEVNULL, check=False, timeout=1800,
        )  # fmt: skip
        if done.returncode != 0:
            return {'': done.stdout + done.stderr}
        return {
            json.loads(line)['case']: json.loads(line)
            for line in out.read_text(encoding='utf-8').splitlines()
        }


def observed(case_id: str) -> dict[str, Any]:
    run = core_run()
    assert '' not in run, run.get('')
    line = run[case_id]
    assert line['outcome']['result'] == 'success', line['outcome']
    return line


def same(a: Any, b: Any) -> bool:
    r"""Two JSON values the same, a number by its value."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(map(same, a, b))
    return a == b


@pytest.mark.parametrize('case_id', EXPLAINED)
def test_the_result_s_explanation_is_recorded(case_id: str) -> None:
    r"""python-core records the explanation the core gave the result, as
    the JSON text of the reference's."""
    wanted = vectors()[case_id]['explanation']
    assert wanted is not None, case_id  # the vector holds one
    result = observed(case_id)['core']['predicted']['result']
    text = result.get('explanation')
    assert isinstance(text, str), f'{case_id}: no explanation recorded'
    assert same(json.loads(text), wanted), (case_id, text, wanted)


@pytest.mark.parametrize('case_id', UNEXPLAINED)
def test_a_result_with_no_explanation_records_none(case_id: str) -> None:
    assert vectors()[case_id]['explanation'] is None
    result = observed(case_id)['core']['predicted']['result']
    assert 'explanation' not in result, result.get('explanation')


@pytest.mark.parametrize('case_id', WITH_METADATA)
def test_each_item_s_metadata_is_recorded(case_id: str) -> None:
    r"""Each item's metadata, the reference's, in the result rows'
    ``metadata`` column, row by row."""
    result = observed(case_id)['core']['predicted']['result']
    names = [column['name'] for column in result['columns']]
    assert 'metadata' in names, names
    at = names.index('metadata')
    cells = [row[at] for row in result['rows']]
    recorded = [
        None if cell is None else json.loads(cell['utf8']) for cell in cells
    ]
    wanted = [pair[1] for pair in vectors()[case_id]['item_metadata']]
    assert len(recorded) == len(wanted)
    for each in wanted:
        assert any(same(each, got) for got in recorded), (each, recorded)


def test_the_observation_schema_takes_the_explanation() -> None:
    validate = _module('t216a_run_python', RUNNER).validator('observation')
    line = copy.deepcopy(observed('explain.answered'))
    assert not list(validate.iter_errors(line))
    # Text only: a parsed value is not what the core gave.
    line['core']['predicted']['result']['explanation'] = {'format': 'x'}
    assert list(validate.iter_errors(line))


def _rust_host(line: dict[str, Any]) -> dict[str, Any]:
    other = copy.deepcopy(line)
    other['identity']['runner'] = 'rust-host'
    return other


@pytest.mark.parametrize('case_id', ['explain.answered', 'explain.forecast'])
def test_the_core_rule_compares_the_explanation_across_hosts(
    case_id: str,
) -> None:
    r"""The two hosts' explanations differing is a ``core`` difference;
    the same, none."""
    compare = _module('t216a_compare', COMPARE)
    case = json.loads((CASES / f'{case_id}.json').read_text(encoding='utf-8'))
    python = observed(case_id)
    agreeing = compare.Report(rules={'core': 'gate'})
    compare._core_agree(agreeing, case, python, _rust_host(python))
    assert not agreeing.failures, agreeing.failures
    rust = _rust_host(python)
    explained = json.loads(rust['core']['predicted']['result']['explanation'])
    explained[
        'by_forecast_step' if 'by_forecast_step' in explained else 'summary'
    ] = 'another host'
    rust['core']['predicted']['result']['explanation'] = json.dumps(explained)
    differing = compare.Report(rules={'core': 'gate'})
    compare._core_agree(differing, case, python, rust)
    assert any('core.predicted' in str(f) for f in differing.failures), (
        differing.failures
    )
