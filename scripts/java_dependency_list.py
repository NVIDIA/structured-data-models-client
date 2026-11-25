#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List the third-party JARs a Java build copied, as Maven Central fetches.

Usage: java_dependency_list.py TARGET REPOSITORY

TARGET is the build's target directory, whose lib and jdbc directories hold
the copied runtime dependencies and JDBC drivers. REPOSITORY is the local
Maven repository they were resolved into. Each line printed is a path under
the artifact root, the JAR's SHA-256 and its Maven Central URL, which
java/jar-load/FetchDependencies.java reads. A JAR found nowhere in the
repository with the same bytes is an error.
"""

import hashlib
import sys
from pathlib import Path

CENTRAL = 'https://repo1.maven.org/maven2/'
DIRECTORIES = ('lib', 'jdbc')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(target: Path, repository: Path) -> list[str]:
    copies: dict[str, list[Path]] = {}
    for path in repository.rglob('*.jar'):
        copies.setdefault(path.name, []).append(path)
    listed = []
    for directory in DIRECTORIES:
        for jar in sorted((target / directory).glob('*.jar')):
            digest = sha256(jar)
            sources = [
                path
                for path in copies.get(jar.name, [])
                if sha256(path) == digest
            ]
            if not sources:
                raise SystemExit(f'{jar} is not in {repository}')
            url = CENTRAL + sources[0].relative_to(repository).as_posix()
            listed.append(f'{directory}/{jar.name}\t{digest}\t{url}')
    if not listed:
        raise SystemExit(f'{target} holds no dependencies')
    return listed


def main() -> None:
    target, repository = (Path(argument) for argument in sys.argv[1:3])
    print('\n'.join(lines(target, repository)))


if __name__ == '__main__':
    main()
