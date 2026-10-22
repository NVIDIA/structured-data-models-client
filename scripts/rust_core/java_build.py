#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Builds and tests the Java library locally (T3.11), as CI's Java jobs do.

    java_build.py verify    the native with the test-only operations, put
                            where Maven packages it (``java/target/natives/
                            <platform>/``), then ``mvn verify``: every Java
                            test, with the runtime flags, loading it as the
                            JAR does (affected.py's suite for ``java/**``
                            and ``rust/bindings/sdm-jni/**``)
    java_build.py jar       the same, then the JAR, and prints ``JAR``,
                            ``LIB`` and ``NATIVE`` lines, the arguments of
                            ``contract/runners/run_java.py`` (handoff.py's
                            Java legs)

The JDK is ``SDM_JAVA_HOME``, else ``JAVA_HOME``; Maven is ``SDM_MAVEN``,
else ``mvn`` on the path. Either missing exits 2, naming it. Cargo builds
into ``CARGO_TARGET_DIR`` (else ``rust/target``); Maven into ``java/target``,
which git ignores.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JAVA = ROOT / 'java'
POM = JAVA / 'pom.xml'
JAR = JAVA / 'target' / 'structured-data-models-client-1.0.0.jar'


class Missing(Exception):
    """A tool this needs is not there."""


def java_home() -> Path:
    for name in ('SDM_JAVA_HOME', 'JAVA_HOME'):
        value = os.environ.get(name)
        if value and (Path(value) / 'bin' / 'java').is_file():
            return Path(value)
    raise Missing('no JDK: set SDM_JAVA_HOME (or JAVA_HOME) to a JDK 17 or 21')


def maven() -> str:
    given = os.environ.get('SDM_MAVEN') or shutil.which('mvn')
    if not given:
        raise Missing('no Maven: set SDM_MAVEN, or put mvn on the path')
    return given


def native_name() -> str:
    return 'libsdm_jni.dylib' if sys.platform == 'darwin' else 'libsdm_jni.so'


def platform_name() -> str:
    os_name = 'macos' if sys.platform == 'darwin' else 'linux'
    machine = platform.machine().lower()
    arch = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(machine, machine)
    return f'{os_name}-{arch}'


def run(argv: list[str], **kwargs: object) -> None:
    done = subprocess.run(argv, stdin=subprocess.DEVNULL, check=False, **kwargs)  # type: ignore[call-overload]
    if done.returncode != 0:
        raise SystemExit(f'java_build.py: {argv[0]} failed ({done.returncode})')


def native() -> Path:
    """The native with the test-only operations, built in debug."""
    run(
        [
            'cargo',
            'build',
            '-q',
            '--locked',
            '-p',
            'sdm-jni',
            '--features',
            'test-ops',
        ],
        cwd=ROOT / 'rust',
    )
    target = Path(
        os.environ.get('CARGO_TARGET_DIR') or ROOT / 'rust' / 'target'
    )
    built = target / 'debug' / native_name()
    if not built.is_file():
        raise SystemExit(f'java_build.py: cargo built no {built}')
    return built


def maven_env(home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env['JAVA_HOME'] = str(home)
    env['PATH'] = f'{home / "bin"}{os.pathsep}{env.get("PATH", "")}'
    return env


def staged() -> Path:
    """The test-ops native, built and put where Maven packages it."""
    built = native()
    natives = JAVA / 'target' / 'natives' / platform_name()
    natives.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, natives / native_name())
    return built


def verify() -> None:
    home, mvn = java_home(), maven()
    staged()
    run([mvn, '-q', '-B', '-f', str(POM), 'verify'], env=maven_env(home))


def jar() -> tuple[Path, Path, Path]:
    home, mvn = java_home(), maven()
    built = staged()
    run(
        [mvn, '-q', '-B', '-f', str(POM), '-DskipTests', 'package'],
        env=maven_env(home),
    )
    return JAR, JAVA / 'target' / 'lib', built


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('command', choices=('verify', 'jar'))
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == 'verify':
            verify()
            print('java: mvn verify passed')
        else:
            built_jar, lib, built = jar()
            print(f'JAR {built_jar}')
            print(f'LIB {lib}')
            print(f'NATIVE {built}')
    except Missing as missing:
        print(f'java_build.py: {missing}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
