# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The public regression workflow preserves native state, payloads and results."""

import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / 'contract/runners'
CASE = ROOT / 'contract/cases/s1/base.regression.json'


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


def test_regression_materializes_and_predicts_through_native_core(tmp_path):
    core = _core(tmp_path)
    assert core['identity']['native_loaded']
    assert core['identity']['operations'] == [
        'describe_source',
        'edit_schema',
        'validate_query',
        'materialize',
        'predict',
        'read_result_items',
    ]
    assert core['core']['error'] is None
    assert core['outcome'] == {
        'result': 'success',
        'stage': 'predict',
        'error': None,
        'code': None,
    }
    [payload] = core['public']['payloads']
    sent = payload['parsed']
    assert len(sent['context']['instance_table']['rows']) == 112
    assert len(sent['predict']['instance_table']['rows']) == 2
    assert isinstance(sent['metadata']['request_id'], str)
    materialized = core['core']['materialized']
    assert materialized['query'] == json.loads(CASE.read_text())['input']['pql']
    for section in ('task', 'schema', 'context', 'predict'):
        assert sent[section] == materialized['payload'][section]
    predicted = core['core']['predicted']
    assert predicted['body'] == sent
    [started] = [i for i in core['core']['inputs'] if i['op'] == 'predict']
    assert started['state'] == materialized
    assert started['model'] == 'kumo-relational'
    assert [c['name'] for c in core['public']['result']['columns']] == [
        'row_index',
        'id',
        'prediction',
    ]
    assert core['public']['result']['rows'] == 2
    result = predicted['result']
    assert [c['name'] for c in result['columns']] == [
        'row_index',
        'id',
        'prediction',
        'answer_index',
    ]
    rows = [
        (
            int(row[0]['i64']),
            int(row[1]['i64']),
            _float(row[2]),
            int(row[3]['i64']),
        )
        for row in result['rows']
    ]
    assert rows == [(0, 0, 1.0, 0), (1, 1, 1.0, 1)]
