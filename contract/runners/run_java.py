#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Java runner of the contract cases (design choice 10, "Tests").

    python3 -I contract/runners/run_java.py --out FILE [--jar JAR]
        [--classpath DIR] [--native LIBRARY] [--java JAVA]
        [CASE_OR_DIRECTORY...]

It reads each case with the Python runner's case reader
(``run_python.load_cases``), so both hosts read the same files the same
way, and hands the Java program (``contract/runners/java/ContractRunner.java``, run in
source-file mode against the JAR under test) only the calls to make and
what to make them with. The program makes each through the public Java
API and writes back what it sent the core and what it was answered; this
file turns each answer into an observation in
``contract/observation.schema.json``'s shape, runner ``java``, judged by
the checkpoint's comparator as rust-host's are.

A Java leg runs exactly where the Java host has a provider for it (design
choice 10): S0's cases, which read no data, from T3.11; a case reading a
database only once the Java host reports a provider for that database
(T3.12, T3.13). A case on no Java leg is not run, and is named on stderr.

The Java outcome is the core's answer as the route maps it, as rust-host's
is: an accepted query a success, a refused one ``INVALID_REQUEST``, a raised
``StructuredDataException`` a refusal with its code; the stage and the
exception class are left null (the S0 call is ``validate``, not the public
``materialize`` the reference's outcome is at).

``--jar`` is the JAR under test (default: ``SDM_JAVA_JAR``) and
``--classpath`` the directory holding
its runtime dependencies (Maven's ``target/lib``, default: the JAR's
sibling ``lib``); ``--native`` names a native library to load in place of
the JAR's own (``sdm.native.library``); ``--java`` the ``java`` to run
(default: ``$JAVA_HOME/bin/java``, else ``java`` on the path). Every Java
run takes the two runtime flags choice 10 names (:data:`FLAGS`).

A runner failure (the Java program failing, an answer missing or not
holding to the observation schema) exits non-zero and writes no output, an
earlier run's included. No message text is recorded (decision H17).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PROGRAM = HERE / 'java' / 'ContractRunner.java'

sys.path.insert(0, str(HERE))
import run_python  # noqa: E402

#: The runtime flags of design choice 10: JNI's native access, and the
#: access Arrow Java's memory module needs.
FLAGS = (
    '--enable-native-access=ALL-UNNAMED',
    '--add-opens=java.base/java.nio=ALL-UNNAMED',
)
#: The databases the Java host has a provider for: none before T3.12.
PROVIDERS: tuple[str, ...] = ()
#: How long one Java run may take, in seconds.
TIMEOUT = 900


class RunnerFailure(Exception):
    r"""The runner, not a case, failed; its text never quotes an input."""


def on_java(case: dict[str, Any]) -> bool:
    r"""Whether the case runs on a Java leg: an S0 case (validation, which
    reads no data), or one whose databases the Java host has a provider
    for (none yet)."""
    core = case.get('core') or {}
    if case.get('slice') == 'S0' and 'validate_query' in core:
        return True
    return bool(PROVIDERS) and all(
        database in PROVIDERS for database in run_python.case_databases(case)
    )


def call_of(case: dict[str, Any]) -> dict[str, Any]:
    r"""What the Java program is to call for the case: S0's validation,
    with the case's query, schema and profile."""
    spec = case['core']['validate_query']
    return {
        'op': 'validate_query',
        'case': case['id'],
        'query': case['input']['pql'],
        'schema': spec['schema'],
        'profile': spec['profile'],
    }


def java_binary(given: str | None) -> str:
    if given:
        return given
    home = os.environ.get('JAVA_HOME')
    if home and (Path(home) / 'bin' / 'java').is_file():
        return str(Path(home) / 'bin' / 'java')
    found = shutil.which('java')
    if found is None:
        raise RunnerFailure('no java: pass --java or set JAVA_HOME')
    return found


def classpath(jar: Path, lib: Path | None) -> str:
    directory = lib if lib is not None else jar.parent / 'lib'
    if not jar.is_file():
        raise RunnerFailure(f'{jar}: not a file')
    if not directory.is_dir():
        raise RunnerFailure(
            f'{directory}: no dependency directory (mvn package writes '
            'target/lib)'
        )
    return os.pathsep.join([str(jar), str(directory / '*')])


def run_program(
    calls: list[dict[str, Any]],
    *,
    java: str,
    path: str,
    native: Path | None,
) -> list[dict[str, Any]]:
    r"""The Java program's answer to each call, in order."""
    with tempfile.TemporaryDirectory(prefix='sdm-java-') as scratch:
        plan = Path(scratch) / 'plan.json'
        answers = Path(scratch) / 'answers.jsonl'
        plan.write_text(json.dumps({'calls': calls}), encoding='utf-8')
        command = [java, *FLAGS]
        if native is not None:
            command.append(f'-Dsdm.native.library={native.resolve()}')
        command += ['-cp', path, str(PROGRAM), str(plan), str(answers)]
        try:
            done = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise RunnerFailure('the Java program did not finish') from None
        if done.returncode != 0:
            sys.stderr.write(done.stderr[-4000:])
            raise RunnerFailure(
                f'the Java program failed with exit status {done.returncode}'
            )
        lines = answers.read_text(encoding='utf-8').splitlines()
    read = [json.loads(line) for line in lines]
    if [answer.get('case') for answer in read] != [c['case'] for c in calls]:
        raise RunnerFailure(
            'the Java program did not answer every call in order'
        )
    return read


def observation(
    case: dict[str, Any],
    case_digest: str,
    answer: dict[str, Any],
    commit: dict[str, Any],
) -> dict[str, Any]:
    r"""One observation of ``case`` from the Java program's ``answer``."""
    error = answer['error']
    if error is not None:
        outcome = {
            'result': 'refusal',
            'stage': None,
            'error': None,
            'code': error['code'],
        }
    elif answer['accepted']:
        outcome = {
            'result': 'success',
            'stage': None,
            'error': None,
            'code': None,
        }
    else:
        outcome = {
            'result': 'refusal',
            'stage': None,
            'error': None,
            'code': 'INVALID_REQUEST',
        }
    identity = answer['identity']
    if identity['providers'] != list(PROVIDERS):
        raise RunnerFailure(
            'the Java host reports providers this runner does not run legs for'
        )
    return {
        'format': 'sdm-observation/1',
        'case': case['id'],
        'identity': {
            'runner': 'java',
            'cell': None,
            'platform': identity['platform'],
            'commit': commit,
            'reference': None,
            'native': None,
            'native_loaded': None,
            'operations': answer['operations'],
            'contract': identity['contract'],
            'versions': {'java': identity['java']},
            'inputs': {'case': case_digest, 'bases': None},
        },
        'outcome': outcome,
        'host': None,
        'core': {
            'inputs': answer['inputs'],
            'fetches': [],
            'accepted': answer['accepted'],
            'described': None,
            'materialized': None,
            'predicted': None,
            'error': error,
        },
    }


def run(
    paths: list[Path],
    *,
    jar: Path,
    lib: Path | None = None,
    native: Path | None = None,
    java: str | None = None,
) -> list[dict[str, Any]]:
    r"""The observations of the Java legs of the cases at ``paths``."""
    try:
        cases = run_python.load_cases(paths)
    except run_python.RunnerFailure as failed:
        raise RunnerFailure(str(failed)) from None
    chosen = [(case, digest) for case, digest in cases if on_java(case)]
    for case, _ in cases:
        if not on_java(case):
            print(f'{case["id"]}: no Java leg', file=sys.stderr)
    if not chosen:
        return []
    answers = run_program(
        [call_of(case) for case, _ in chosen],
        java=java_binary(java),
        path=classpath(jar, lib),
        native=native,
    )
    commit = run_python._git(REPO)
    validate = run_python.validator('observation')
    written = []
    for (case, digest), answer in zip(chosen, answers, strict=True):
        made = observation(case, digest, answer, commit)
        problem = run_python._invalid(validate, made)
        if problem is not None:
            raise RunnerFailure(
                f'{case["id"]}: an invalid observation: {problem}'
            )
        written.append(made)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--jar', type=Path)
    parser.add_argument('--classpath', type=Path)
    parser.add_argument('--native', type=Path)
    parser.add_argument('--java')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('cases', nargs='*', type=Path)
    arguments = parser.parse_args(argv)
    arguments.out.unlink(missing_ok=True)
    jar = arguments.jar or (
        Path(os.environ['SDM_JAVA_JAR'])
        if os.environ.get('SDM_JAVA_JAR')
        else None
    )
    if jar is None:
        print(
            'run_java.py: no JAR: pass --jar or set SDM_JAVA_JAR',
            file=sys.stderr,
        )
        return 1
    paths = arguments.cases or [REPO / 'contract/cases/s0']
    try:
        written = run(
            paths,
            jar=jar,
            lib=arguments.classpath,
            native=arguments.native,
            java=arguments.java,
        )
    except RunnerFailure as failed:
        print(f'run_java.py: {failed}', file=sys.stderr)
        return 1
    arguments.out.write_text(
        ''.join(json.dumps(line, sort_keys=True) + '\n' for line in written),
        encoding='utf-8',
    )
    print(f'{len(written)} Java observation(s) written to {arguments.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
