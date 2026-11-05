# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run both installed clients against disposable fixtures, without model calls."""

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from hosted_fixtures import parquet, postgres, s3
from hosted_profile import EMULATED, LOCAL, WAREHOUSE, profile
from hosted_warehouses import warehouse


def provision(connector):
    if connector == 'parquet':
        return parquet()
    if connector == 'postgres':
        return postgres()
    if connector in EMULATED:
        return s3(connector)
    if connector in WAREHOUSE:
        return warehouse(connector)
    if connector in LOCAL:
        return contextlib.nullcontext({})
    raise ValueError('Connector is not in the required hosted profile')


def run(connector, root, commit):
    repository = Path(__file__).resolve().parents[2]
    evidence = root / 'evidence'
    private = root / 'private'
    evidence.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    config_path = private / 'fixture-config.json'
    policy = next(
        row for row in profile()['required'] if row['connector'] == connector
    )
    start = time.monotonic()
    completed = False
    try:
        with provision(connector) as config:
            config_path.write_text(json.dumps(config))
            config_path.chmod(0o600)
            commands = {
                'python': [
                    sys.executable,
                    'tests/compatibility/python_runner.py',
                    '--phase',
                    'candidate',
                    '--mode',
                    'local',
                    '--commit',
                    commit,
                    '--connector',
                    connector,
                    '--config',
                    str(config_path),
                    '--provenance',
                    os.environ['PARITY_PROVENANCE'],
                    '--out',
                    str(evidence / 'python.jsonl'),
                ],
                'java': [
                    'mvn',
                    '-B',
                    '-f',
                    'java/pom.xml',
                    '-Dtest=ParityRunnerTest',
                    '-Dsdm.native.library=' + os.environ['PARITY_NATIVE'],
                    '-Dsdm.parity.jar=' + os.environ['PARITY_JAR'],
                    '-Dsdm.parity.provenance='
                    + str(root / 'java-provenance.json'),
                    '-Dsdm.parity.commit=' + commit,
                    '-Dsdm.parity.manifest='
                    + str(repository / 'tests/compatibility/manifest.json'),
                    '-Dsdm.parity.config=' + str(config_path),
                    '-Dsdm.parity.output=' + str(evidence / 'java.jsonl'),
                    '-Dsdm.parity.connectors=' + connector,
                    '-Dsdm.parity.mode=local',
                    'test',
                ],
            }
            if connector == 'databricks-spark':
                commands['java'].insert(
                    -1,
                    '-Dsdm.parity.connect.classpath='
                    + str(root / 'connect-classpath.txt'),
                )
            outcomes = {}
            for language, command in commands.items():
                with (private / (language + '.log')).open('w') as log:
                    outcomes[language] = subprocess.run(
                        command,
                        cwd=repository,
                        stdout=log,
                        stderr=log,
                        timeout=2400,
                        check=False,
                    ).returncode
            if any(outcomes.values()):
                raise RuntimeError(
                    'A client runner failed; private diagnostics retained'
                )
        completed = True
    finally:
        config_path.unlink(missing_ok=True)
        (evidence / 'execution.json').write_text(
            json.dumps(
                {
                    **policy,
                    'commit': commit,
                    'matrix_mode': 'materialization-only',
                    'model_calls': 0,
                    'completed': completed,
                    'wall_seconds': time.monotonic() - start,
                },
                indent=2,
            )
            + '\n'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connector', required=True)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    run(args.connector, args.root.resolve(), args.commit)
