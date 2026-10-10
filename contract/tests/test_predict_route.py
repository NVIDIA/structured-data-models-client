# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""base.regression predicted on the core against the reference (T1.11d).

Run where the contract tests run (CI's ``engine`` job until T1.18):

    python -m pytest contract/tests -q

On ``python-core`` the call's ``predict`` runs on the core: the request
state ``materialize`` answered comes back with the ``predict`` start, the
core writes the call, the client's own transport sends it to the scripted
endpoint, and the core reads the answer and lays out the result. Each
claim and what holds it:

- The model request and its result are the reference's: python-core's
  public projection (outcome at the ``predict`` stage, warning classes,
  schema, every request the endpoint received, parsed with each number as
  its text, the result through ``to_pandas()``: its columns, dtypes, row
  count and rows' digest, and the query-log counts) equals the committed
  reference observation's, with ``capture_check.projection``, whose one
  exclusion is the value of ``metadata.request_id`` (decision E35,
  approved by the owner on 2026-10-10, ``contract/compare-policy.yaml``'s
  one exclusion; the field must be present alike with a value of the
  same type).
- The request the endpoint received is the call the core wrote, and the
  state predict was started with is the one materialize answered: the
  request state crossed runs as the host held it.
- The two core hosts agree: python-core's inputs (predict's included), its
  call, with the same one value left out, and its result are the ones
  ``contract/pinned/base.regression.core.json`` pins, which ``sdm-testkit``'s
  ``tests/duckdb_provider.rs`` holds rust-host to; the pin is checked to
  hold the call's 112 context and 2 predicted rows and the result's two
  rows, each a prediction of 1.0 for its entity.

The case is diagnostic until T1.29 (``compare.py`` reports, and gates
none of, its differences); these tests gate the claims above in CI.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / 'contract/runners'
CASE = ROOT / 'contract/cases/s1/base.regression.json'
REFERENCE = next((ROOT / 'contract/reference').glob('*.jsonl'))
PINNED = ROOT / 'contract/pinned/base.regression.core.json'


def _module(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUNNERS / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


capture_check = _module('capture_check')


def _reference() -> dict[str, Any]:
    for line in REFERENCE.read_text().splitlines():
        observation = json.loads(line)
        if observation['case'] == 'base.regression':
            return observation
    raise AssertionError('no reference observation of base.regression')


def _core(tmp_path: Path) -> dict[str, Any]:
    out = tmp_path / 'python-core.jsonl'
    run = subprocess.run(
        [
            sys.executable,
            '-I',
            str(RUNNERS / 'run_python.py'),
            '--mode',
            'python-core',
            '--out',
            str(out),
            str(CASE),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    [observation] = [json.loads(line) for line in out.read_text().splitlines()]
    return observation


def _float(bits: dict[str, str]) -> float:
    return float(struct.unpack('>d', bytes.fromhex(bits['f64_bits']))[0])


def test_base_regression_completes_a_model_request_and_its_result_on_the_core(
    tmp_path: Path,
) -> None:
    reference = _reference()
    core = _core(tmp_path)
    assert core['identity']['operations'] == [
        'describe_source',
        'materialize',
        'predict',
    ]
    assert core['core']['error'] is None
    assert core['outcome'] == reference['outcome']
    assert core['outcome']['stage'] == 'predict'
    # Not vacuous: one request reached the endpoint, and the result has the
    # reference's two rows of three columns.
    [payload] = core['public']['payloads']
    assert len(payload['parsed']['context']['instance_table']['rows']) == 112
    assert core['public']['result']['rows'] == 2
    assert [c['name'] for c in core['public']['result']['columns']] == [
        'row_index',
        'id',
        'prediction',
    ]
    assert core['public']['result'] == reference['public']['result']
    assert capture_check.projection(core) == capture_check.projection(reference)

    # The request the endpoint received is the call the core wrote, and the
    # state predict started with is the one materialize answered.
    predicted = core['core']['predicted']
    assert predicted['body'] == payload['parsed']
    [started] = [i for i in core['core']['inputs'] if i['op'] == 'predict']
    materialized = core['core']['materialized']
    assert started['state'] == materialized
    assert started['model'] == 'kumo-relational'

    # The two core hosts agree, against the pin rust-host is held to.
    pinned = json.loads(PINNED.read_text())
    assert core['core']['inputs'] == pinned['inputs']
    assert {
        **predicted,
        'body': capture_check.masked(predicted['body']),
    } == pinned['predicted']


def test_the_pin_holds_a_whole_call_and_result() -> None:
    pinned = json.loads(PINNED.read_text())
    predicted = pinned['predicted']
    body = predicted['body']
    assert len(body['context']['instance_table']['rows']) == 112
    assert len(body['predict']['instance_table']['rows']) == 2
    assert body['model'] == 'kumo-relational'
    assert body['output'] == {
        'fields': ['prediction'],
        'prediction_statistic': 'median',
    }
    # E35's one value: present, a string, its value left out.
    assert body['metadata']['request_id'] == {
        'excluded value of type': 'string'
    }
    assert set(body['inference']['inference_config']) <= {
        'kind',
        'num_estimators',
        'column_shuffle',
        'category_shuffle',
        'hop_shuffle',
        'class_shuffle',
    }
    result = predicted['result']
    assert [c['name'] for c in result['columns']] == [
        'row_index',
        'id',
        'prediction',
    ]
    rows = [
        (int(row[0]['i64']), int(row[1]['i64']), _float(row[2]))
        for row in result['rows']
    ]
    assert rows == [(0, 0, 1.0), (1, 1, 1.0)]
    assert [i['op'] for i in pinned['inputs']] == [
        'describe_source',
        'materialize',
        'predict',
    ]
    assert pinned['inputs'][2]['state'] == pinned['materialized']
