# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reuse one successful full CI run's production artifacts, never rebuild them."""

import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path

NATIVE = 'com/nvidia/sdm/native/linux-x86_64/libsdm_jni.so'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_run(run, workflow, commit):
    if not (
        workflow['path'] == '.github/workflows/ci.yml'
        and run['workflow_id'] == workflow['id']
        and run['path'] == workflow['path']
        and run['head_sha'] == commit
        and run['status'] == 'completed'
        and run['conclusion'] == 'success'
        and run['display_title'] == 'CI'
    ):
        raise ValueError(
            'A successful full CI run of the exact commit is required'
        )
    return int(run['id'])


def gh_json(path):
    return json.loads(
        subprocess.check_output(
            ['gh', 'api', path], text=True, stderr=subprocess.PIPE
        )
    )


def select_run(repository, commit, requested):
    base = f'repos/{repository}/actions'
    workflow = gh_json(base + '/workflows/ci.yml')
    if requested:
        run = gh_json(base + '/runs/' + str(int(requested)))
        return validate_run(run, workflow, commit)
    # Bounded discovery: require an explicit run id if older evidence is desired.
    runs = gh_json(
        base
        + '/workflows/ci.yml/runs?head_sha='
        + commit
        + '&status=success&per_page=100'
    )['workflow_runs']
    for run in runs:
        try:
            return validate_run(run, workflow, commit)
        except ValueError:
            continue
    raise ValueError('No successful full CI run exists for this commit')


def one(paths):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError('Expected exactly one production artifact')
    return paths[0]


def verify(root, commit, run_id):
    wheel = one((root / 'wheels').glob('*.whl'))
    sidecar = one((root / 'wheels').glob('*.whl.json'))
    identity = json.loads(sidecar.read_text())
    if not (
        identity.get('commit') == commit
        and identity.get('tree_clean') is True
        and identity.get('features') == []
        and identity.get('wheel') == wheel.name
        and identity.get('sha256') == digest(wheel)
    ):
        raise ValueError(
            'Wheel must be the clean exact-commit production build'
        )
    with zipfile.ZipFile(wheel) as archive:
        native = one(
            name
            for name in archive.namelist()
            if name.startswith('sdm_client/_native') and name.endswith('.so')
        )
        if hashlib.sha256(archive.read(native)).hexdigest() != identity.get(
            'native_sha256'
        ):
            raise ValueError('Wheel native differs from its sidecar')
    jar = one((root / 'java').glob('*.jar'))
    pom = one((root / 'java').glob('*.pom'))
    java = json.loads((root / 'java/java-artifacts.json').read_text())
    if java.get('commit') != commit or java.get('files') != {
        p.name: digest(p) for p in (jar, pom)
    }:
        raise ValueError(
            'JAR/POM receipt does not match the exact commit artifacts'
        )
    native = root / 'native/linux-x86_64/libsdm_jni.so'
    with zipfile.ZipFile(jar) as archive:
        if archive.read(NATIVE) != native.read_bytes():
            raise ValueError(
                'Packaged JNI differs from the production CI native artifact'
            )
    receipt = {
        'commit': commit,
        'ci_run_id': int(run_id),
        'build': 'production',
        'wheel_sha256': digest(wheel),
        'jar_sha256': digest(jar),
        'native_sha256': digest(native),
    }
    (root / 'evidence/artifacts.json').write_text(
        json.dumps(receipt, indent=2) + '\n'
    )
    return wheel, sidecar, jar, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument(
        '--repository', default=os.environ.get('GITHUB_REPOSITORY')
    )
    parser.add_argument('--run-id', default='')
    parser.add_argument('--installed', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    (root / 'evidence').mkdir(parents=True, exist_ok=True)
    if args.installed:
        from python_runner import artifact

        receipt = json.loads((root / 'evidence/artifacts.json').read_text())
        wheel, sidecar, jar, native = verify(
            root, args.commit, receipt['ci_run_id']
        )
        checkout = Path(__file__).resolve().parents[2]
        head = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=checkout,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=checkout,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        if head != args.commit or dirty:
            raise ValueError(
                'Focused Java tests require the exact clean source commit'
            )
        python = artifact(args.commit, 'candidate', sidecar)
        java = dict(
            commit=args.commit,
            package_hash=digest(jar),
            native_hash=digest(native),
            contract=python['contract'],
        )
        (root / 'java-provenance.json').write_text(json.dumps(java) + '\n')
        receipt.update(
            python=python,
            java=java,
            focused_java={
                'execution': 'source-equivalent',
                'source_commit': head,
                'tree_clean': True,
                'native_hash': digest(native),
            },
        )
        (root / 'evidence/artifacts.json').write_text(
            json.dumps(receipt, indent=2) + '\n'
        )
        return
    run_id = select_run(args.repository, args.commit, args.run_id)
    for name, directory in [
        ('native-wheel', 'wheels'),
        ('java-jar', 'java'),
        ('java-native-linux-x86_64', 'native'),
    ]:
        subprocess.run(
            [
                'gh',
                'run',
                'download',
                str(run_id),
                '--repo',
                args.repository,
                '--name',
                name,
                '--dir',
                str(root / directory),
            ],
            check=True,
            capture_output=True,
        )
    wheel, sidecar, jar, native = verify(root, args.commit, run_id)
    with open(os.environ['GITHUB_ENV'], 'a') as out:
        for key, value in [
            ('PARITY_DIR', root),
            ('PARITY_WHEEL', wheel),
            ('PARITY_PROVENANCE', sidecar),
            ('PARITY_JAR', jar),
            ('PARITY_NATIVE', native),
        ]:
            out.write(f'{key}={value}\n')
    with open(os.environ['GITHUB_OUTPUT'], 'a') as out:
        out.write(f'ci_run_id={run_id}\n')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # GitHub diagnostics can include authenticated URLs; do not echo them.
        raise SystemExit(
            'CI artifact reuse failed ('
            + type(error).__name__
            + '); no acceptance evidence certified'
        ) from None
