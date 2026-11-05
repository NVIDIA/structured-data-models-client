# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded pretrained CPU coverage, separate from connector materialization."""

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def require_tests(path, expected):
    cases = ET.parse(path).getroot().findall('.//testcase')
    if len(cases) != expected or any(
        case.find(tag) is not None
        for case in cases
        for tag in ('skipped', 'failure', 'error')
    ):
        raise ValueError('Required CPU smoke tests did not all pass')


def run(root):
    private, evidence = root / 'private', root / 'evidence'
    private.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    url = private / 'cpu-url.txt'
    url.unlink(missing_ok=True)
    env = dict(
        os.environ,
        OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',
        CUDA_VISIBLE_DEVICES='',
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        HF_TOKEN='',
        SDM_RUN_LOCAL_MODEL='1',
    )
    start = time.monotonic()
    with (private / 'cpu-model-server.log').open('w') as server_log:
        server = subprocess.Popen(
            [
                sys.executable,
                'tests/compatibility/cpu_model_server.py',
                '--url-file',
                str(url),
            ],
            env=env,
            stdout=server_log,
            stderr=server_log,
        )
        try:
            for _ in range(60):
                if url.is_file():
                    break
                if server.poll() is not None:
                    raise RuntimeError(
                        'CPU model server exited before readiness'
                    )
                time.sleep(1)
            endpoint = url.read_text()
            env.update(
                SDM_GRADIENT_URL=endpoint,
                SDM_TABULAR_LIVE_URL=endpoint,
                SDM_RELATIONAL_URL=endpoint,
            )
            python_xml = evidence / 'cpu-python.xml'
            with (private / 'cpu-python.log').open('w') as log:
                subprocess.run(
                    [
                        sys.executable,
                        '-m',
                        'pytest',
                        'packages/structured-data-models-client/tests/test_gradient_explanations.py::test_pretrained_local_model_gradients',
                        '-q',
                        '--junitxml=' + str(python_xml),
                    ],
                    env=env,
                    stdout=log,
                    stderr=log,
                    timeout=600,
                    check=True,
                )
            require_tests(python_xml, 4)
            with (private / 'cpu-java.log').open('w') as log:
                subprocess.run(
                    [
                        'mvn',
                        '-B',
                        '-f',
                        'java/pom.xml',
                        '-Dtest=GradientExplanationsTest#realModelEndpointGradients,TabularTest#liveTabularDirectAndRepeatedSession,AutomaticBatchSessionTest#livePreparedBatchPrediction',
                        '-Dsdm.native.library=' + os.environ['PARITY_NATIVE'],
                        'test',
                    ],
                    env=env,
                    stdout=log,
                    stderr=log,
                    timeout=600,
                    check=True,
                )
            for name in (
                'GradientExplanationsTest',
                'TabularTest',
                'AutomaticBatchSessionTest',
            ):
                path = Path('java/target/surefire-reports') / (
                    'TEST-com.nvidia.sdm.' + name + '.xml'
                )
                require_tests(path, 1)
            (evidence / 'cpu-model-smoke.json').write_text(
                json.dumps(
                    {
                        'status': 'passed',
                        'device': 'cpu',
                        'python_tests': 4,
                        'java_tests': 3,
                        'gradient_combinations_per_language': 4,
                        'java_execution': 'source-equivalent-with-verified-packaged-jni',
                        'wall_seconds': time.monotonic() - start,
                    },
                    indent=2,
                )
                + '\n'
            )
        finally:
            server.terminate()
            server.wait(timeout=20)
            url.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    run(parser.parse_args().root.resolve())
