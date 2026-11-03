#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build target/candidate Java APIs and run japicmp without JNI or live services."""

import argparse
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=ROOT):
    subprocess.run(args, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--maven', default='mvn')
    args = parser.parse_args()
    base = subprocess.check_output(
        ['git', 'merge-base', args.base, 'HEAD'],
        cwd=ROOT,
        text=True,
    ).strip()
    build = ['-q', '-Dmaven.test.skip=true', 'compile', 'jar:jar']
    baseline = subprocess.check_output(
        ['git', 'ls-tree', '--name-only', base, 'java/pom.xml'],
        cwd=ROOT,
        text=True,
    ).strip()
    if not baseline:
        run(args.maven, '-f', 'java/pom.xml', *build)
        print(
            f'Java API introduced: candidate compiled; no Java baseline at {base}'
        )
        return
    with tempfile.TemporaryDirectory(prefix='sdm-java-api-') as directory:
        target = Path(directory)
        archive = target / 'baseline.tar'
        run(
            'git',
            'archive',
            '--format=tar',
            f'--output={archive}',
            base,
            'java/pom.xml',
            'java/src/main/java',
        )
        with tarfile.open(archive) as bundle:
            bundle.extractall(target, filter='data')
        run(args.maven, '-f', str(target / 'java/pom.xml'), *build)
        jars = list((target / 'java/target').glob('*.jar'))
        if len(jars) != 1:
            raise RuntimeError(f'expected one baseline JAR, got {jars}')
        run(
            args.maven,
            '-f',
            'java/pom.xml',
            '-q',
            '-Papi-compatibility',
            f'-Dsdm.api.baseline={jars[0]}',
            *build,
            'japicmp:cmp',
        )
    print(f'Java source/binary API comparison passed against {base}')


if __name__ == '__main__':
    main()
