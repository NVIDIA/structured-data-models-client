# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require the hosted profile without claiming a full remote model matrix."""

import argparse
import json
from pathlib import Path

from ci_shards import combine
from compare import assess
from hosted_profile import profile

from contract import cells, manifest, records


def require_models(directory, identity):
    receipt = json.loads((directory / 'artifacts.json').read_text())
    smoke = json.loads((directory / 'cpu-model-smoke.json').read_text())
    if receipt != identity or any(
        smoke.get(key) != value
        for key, value in {
            'status': 'passed',
            'device': 'cpu',
            'python_tests': 4,
            'java_tests': 3,
            'gradient_combinations_per_language': 4,
            'java_execution': 'source-equivalent-with-verified-packaged-jni',
        }.items()
    ):
        raise ValueError('CPU smoke did not verify the required CI artifacts')
    return smoke


def aggregate(root, commit, run_id, out, models):
    data = manifest()
    policy = profile(data)
    connectors = [row['connector'] for row in policy['required']]
    combined = out.with_name('materialization-observations.jsonl')
    combine(root, connectors, commit, run_id, combined)
    identity = json.loads(
        (root / ('live-shard-' + connectors[0]) / 'artifacts.json').read_text()
    )
    smoke = require_models(models, identity)
    executions = []
    for row in policy['required']:
        execution = json.loads(
            (
                root / ('live-shard-' + row['connector']) / 'execution.json'
            ).read_text()
        )
        if (
            any(execution.get(key) != value for key, value in row.items())
            or execution.get('commit') != commit
            or execution.get('completed') is not True
            or execution.get('model_calls') != 0
            or execution.get('matrix_mode') != 'materialization-only'
        ):
            raise ValueError('Missing or inconsistent hosted execution receipt')
        executions.append(execution)
    selected = {cell for cell in cells(data) if cell[1] in connectors}
    report = assess(
        data,
        {},
        records(combined),
        live=False,
        selected=selected,
        candidate_commit=commit,
        comparison='current-clients',
    )
    report['execution_profile'] = policy
    report['cpu_model_smoke'] = smoke
    report['executions'] = executions
    report['full_live_model_matrix_claimed'] = False
    out.write_text(json.dumps(report, indent=2) + '\n')
    if not report['scope_complete']:
        raise ValueError(
            'Required hosted materialization cells did not all pass'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.root, args.commit, args.run_id, args.out, args.models)
