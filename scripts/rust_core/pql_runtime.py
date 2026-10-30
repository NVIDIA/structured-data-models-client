#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record frozen public PQL results and judge an installed wheel on native Arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1'


def run(command: list[str], *, env: dict[str, str], cwd: Path) -> None:
    subprocess.run(command, env=env, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if platform.machine() not in ('aarch64', 'arm64'):
        raise SystemExit('this acceptance lane requires native Arm')
    wheel = args.wheel.resolve()
    reference = args.reference.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('PYTEST_') or key in (
            'PYTHONPATH',
            'SDM_CORE_ROUTES',
        ):
            env.pop(key)
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    run([sys.executable, '-m', 'pip', 'check'], env=env, cwd=out)
    freeze = subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze', '--all'], env=env, text=True
    )
    (out / 'requirements.txt').write_text(freeze)
    sha = subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True
    ).strip()
    provenance = json.loads(wheel.with_name(wheel.name + '.json').read_text())
    if (
        provenance['commit'] != sha
        or not provenance['tree_clean']
        or provenance['features']
        or provenance['sha256']
        != hashlib.sha256(wheel.read_bytes()).hexdigest()
    ):
        raise SystemExit(
            'wheel provenance does not identify this clean shipped candidate'
        )
    shutil.copy2(wheel.with_name(wheel.name + '.json'), out / 'wheel.json')
    (out / 'candidate.json').write_text(
        json.dumps(
            {
                'revision': sha,
                'wheel': wheel.name,
                'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
                'python': sys.version,
                'machine': platform.machine(),
                'reference_revision': REFERENCE,
                'scope': 'targeted PQL runtime acceptance, not full CI',
            },
            indent=2,
        )
        + '\n'
    )
    # Freeze the oracle before candidate execution, with only frozen product
    # sources on PYTHONPATH; the recorder itself verifies checkout provenance.
    frozen_env = dict(env)
    frozen_env['PYTHONPATH'] = os.pathsep.join(
        str(reference / 'packages' / package / 'src')
        for package in (
            'structured-data-models-client',
            'structured-data-models-api',
            'structured-data-models-pql',
            'structured-data-models-connectors',
        )
    )
    run(
        [
            sys.executable,
            str(ROOT / 'scripts/record_tabular_pql.py'),
            '--reference-checkout',
            str(reference),
            '--reference-revision',
            REFERENCE,
            '--output-dir',
            str(out / 'reference'),
        ],
        env=frozen_env,
        cwd=out,
    )
    tests = out / 'tests'
    tests.mkdir()
    source = ROOT / 'packages/structured-data-models-client/tests'
    for name in (
        'test_tabular_pql_core.py',
        'tabular_pql_fixtures.py',
        'tabular_pql_reference.py',
    ):
        shutil.copy2(source / name, tests / name)
    shutil.copytree(source / 'data', tests / 'data')
    shutil.rmtree(tests / 'data/tabular_pql_reference')
    shutil.copytree(out / 'reference', tests / 'data/tabular_pql_reference')
    run(
        [
            sys.executable,
            '-c',
            'from pathlib import Path; import hashlib; import sdm_client, sdm_client._native; '
            f'root=Path({str(ROOT)!r}); '
            'assert not Path(sdm_client.__file__).resolve().is_relative_to(root); '
            'assert not Path(sdm_client._native.__file__).resolve().is_relative_to(root); '
            'assert hashlib.sha256(Path(sdm_client._native.__file__).read_bytes()).hexdigest() '
            f'== {provenance["native_sha256"]!r}; '
            'print(sdm_client.__file__, sdm_client._native.__file__)',
        ],
        env=env,
        cwd=out,
    )
    run(
        [
            sys.executable,
            '-m',
            'pytest',
            str(tests / 'test_tabular_pql_core.py'),
            '-q',
            '-o',
            'addopts=',
            '--junitxml',
            str(out / 'results.xml'),
        ],
        env=env,
        cwd=out,
    )
    suites = ElementTree.parse(out / 'results.xml').getroot()
    cases = suites.findall('.//testcase')
    if not cases or any(case.find('skipped') is not None for case in cases):
        raise SystemExit('PQL runtime acceptance requires tests and zero skips')


if __name__ == '__main__':
    main()
