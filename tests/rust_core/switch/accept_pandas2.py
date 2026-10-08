# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P9.pandas2: under Python 3.10 with pandas 2.x, the suite-rest command and
the replay pass, or every difference is listed with its cause.

The second venv (``CONTRACT.md``, "The pandas 2 environment") is made when
missing, and on every run given this checkout: the packages as GitHub's
test job installs them, editable, with the engine, this checkout's binding
built with maturin, and ``pandas<3``. In it the suite-rest command and the
replay of every transcript run on the core (the replay must load the binding
and neither ported package). Every test that fails or errors, and every
transcript that does not replay, must be listed in
``tests/rust_core/shadow/pandas2_differences.json`` with its cause, and
every one listed must still differ.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import pytest
from _switch import (
    CORE,
    HERE,
    PANDAS2_DIFFERENCES,
    REPO,
    SUITE_MARKER,
    SUITE_PATHS,
    add_paths,
    env_for,
    run,
)

VENV = Path(
    os.environ.get(
        'SDM_P9_PANDAS2_VENV', '~/.local/share/sdm-dbx/venv-rc-p9-pd2'
    )
).expanduser()
PYTHON = VENV / 'bin' / 'python'


def _uv() -> str:
    uv = shutil.which('uv')
    assert uv, 'uv is needed to make the pandas 2 venv'
    return uv


@pytest.fixture(scope='module')
def venv() -> Path:
    add_paths()
    from _rc import build_local_wheel

    if not PYTHON.exists():
        made = run([_uv(), 'venv', '--python', '3.10', VENV])
        assert made.returncode == 0, made.stderr
    with tempfile.TemporaryDirectory(prefix='sdm-p9-pd2-wheel-') as tmp:
        wheel = build_local_wheel(Path(tmp))
        installed = run(
            [
                _uv(),
                'pip',
                'install',
                '--python',
                PYTHON,
                '--find-links',
                tmp,
                '--reinstall-package',
                'sdm-core',
                '-e',
                './packages/structured-data-models-api[test]',
                '-e',
                './packages/structured-data-models-pql[test]',
                '-e',
                './packages/structured-data-models-client[test]',
                '-e',
                './packages/structured-data-models-connectors[sqlite,duckdb,postgres,databricks,test]',
                '-e',
                './packages/structured-data-models-relational-engine',
                str(wheel),
                'pandas>=2,<3',
            ],
            cwd=REPO,
        )
        assert installed.returncode == 0, installed.stderr[-4000:]
    versions = run(
        [
            PYTHON,
            '-c',
            'import sys, pandas, sdm_core; '
            'print(sys.version_info[:2], pandas.__version__, sdm_core.__file__)',
        ],
        cwd='/',
    )
    assert versions.returncode == 0, versions.stderr
    assert versions.stdout.startswith('(3, 10)'), versions.stdout
    assert re.search(r'\) 2\.\d+', versions.stdout), versions.stdout
    return VENV


def _env() -> dict[str, str]:
    env = env_for(CORE)
    env['VIRTUAL_ENV'] = str(VENV)
    env['PATH'] = os.pathsep.join([str(VENV / 'bin'), env.get('PATH', '')])
    env.pop('PYTHONPATH', None)
    return env


def _listed() -> dict:
    if not PANDAS2_DIFFERENCES.is_file():
        return {'tests': {}, 'replay': {}}
    data = json.loads(PANDAS2_DIFFERENCES.read_text())
    return {'tests': data.get('tests', {}), 'replay': data.get('replay', {})}


def _judge(kind: str, failing: set[str]) -> list[str]:
    listed = _listed()[kind]
    problems = []
    for name in sorted(failing - set(listed)):
        problems.append(
            f'{kind}: {name} differs under pandas 2 and is not listed'
        )
    for name in sorted(set(listed) - failing):
        problems.append(f'{kind}: {name} is listed and no longer differs')
    for name, cause in sorted(listed.items()):
        if len(str(cause).split()) < 6:
            problems.append(
                f'{kind}: {name}: give its cause in six words or more'
            )
    return problems


def test_the_suite_passes_under_pandas_2_or_each_difference_is_listed(venv):
    proc = run(
        [
            PYTHON,
            '-m',
            'pytest',
            *SUITE_PATHS,
            '-q',
            '-rfE',
            '-m',
            SUITE_MARKER,
            '-p',
            'no:cacheprovider',
        ],
        cwd=REPO,
        env=_env(),
        timeout=5400,
    )
    failing = {
        line.split(' ', 1)[1].split(' - ')[0].strip()
        for line in proc.stdout.splitlines()
        if line.startswith(('FAILED ', 'ERROR '))
    }
    summary = proc.stdout.strip().splitlines()[-1:] or ['']
    assert re.search(r'\d+ passed', summary[0]), (
        proc.stdout[-3000:] + proc.stderr[-3000:]
    )
    if proc.returncode != 0:
        assert failing, proc.stdout[-3000:] + proc.stderr[-3000:]
    problems = _judge('tests', failing)
    assert not problems, '\n'.join(problems[:60])


def test_the_replay_passes_under_pandas_2_or_each_difference_is_listed(venv):
    with tempfile.TemporaryDirectory(prefix='sdm-p9-pd2-replay-') as tmp:
        out = Path(tmp) / 'replay.json'
        proc = run(
            [PYTHON, HERE / '_switch.py', 'cpu', '--impl', CORE, '--out', out],
            cwd=REPO,
            env=_env(),
            timeout=3600,
        )
        assert proc.returncode == 0 and out.is_file(), proc.stderr[-4000:]
        data = json.loads(out.read_text())
    assert data['transcripts'] >= 137, data['transcripts']
    assert data['native_loaded'] and not data['first_imports'], (
        f'the replay did not run on the core: {data["first_imports"]}'
    )
    failing = {line.split(': ', 1)[0] for line in data['failed']}
    problems = _judge('replay', failing)
    assert not problems, '\n'.join(problems[:60])
