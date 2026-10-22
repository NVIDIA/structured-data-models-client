# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T3.11's exit 2: a Java-only difference stops ``handoff.py``'s stamp.

The handoff runs each touched or named case's Java legs beside python-core
(the brief, "Handoff"): here the S0 case T3.0's own handoff test uses, on
a clone of this checkout, through T3.0's helpers
(``tests/test_t30_process.py``). The ``--python`` interpreter is a wrapper
that, when told to, rewrites the Java runner's observation of that case to
a refusal (``INTERNAL_ERROR``) after the runner wrote it, and leaves
python-core's alone: the seeded run gets no stamp, naming the case and the
Java leg; the same run unseeded gets one, so the refusal is the Java leg's
alone.

ASSUMED until the implementer relays it: ``handoff.py`` runs the Java
runner as it runs ``run_python.py``, with ``--python``'s interpreter
(``<python> -I contract/runners/run_java.py ... --out FILE ...``), the JDK
from ``JAVA_HOME`` and the JAR from ``SDM_JAVA_JAR``, which this passes
through.

Like T3.0's handoff test, it needs ``SDM_TEST_CORE_PYTHON`` (an interpreter
holding the candidate's client) and ``SDM_POSTGRES_TEST_DSN`` (the case runs
on Postgres too, and the handoff stamps nothing without a server), and a
JDK; without them it has nothing to hold and returns. CI's ``tooling
(root)`` job has the first two. Gating (``contract/tests/t311_gate.py``):
while ``contract/tests/t311-pending.json`` names it, it must still fail.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEEDED_CASE = 's0.accept.binary-classification'
#: The Java leg named in the handoff's report (a word: the test's own
#: temporary paths hold ``_java_``).
JAVA = re.compile(r'\bjava\b', re.IGNORECASE)
#: These tests copy the checkout and run the checking tools there: the
#: judging-tool job's (T3.0, item 3).
pytestmark = pytest.mark.tooling


def _t30() -> ModuleType:
    r"""T3.0's process tests, whose clone and handoff helpers these use."""
    name = 't311_t30_process'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'tests/test_t30_process.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G = _module('t311_gate', ROOT / 'contract/tests/t311_gate.py')


def _ready() -> bool:
    r"""What the handoff's Java leg needs is here: the core interpreter
    (``SDM_TEST_CORE_PYTHON``), a Postgres server (the case runs there too)
    and a JDK (``JAVA_HOME`` or ``java`` on ``PATH``)."""
    return bool(
        os.environ.get('SDM_TEST_CORE_PYTHON')
        and os.environ.get('SDM_POSTGRES_TEST_DSN')
        and (os.environ.get('JAVA_HOME') or shutil.which('java'))
    )


#: ``--python`` for the handoff: the core interpreter, which, when told to,
#: turns the Java runner's observation of the seeded case into a refusal
#: with INTERNAL_ERROR, after the runner wrote it.
SEEDING_PYTHON = r"""#!{python}
import json, os, subprocess, sys

args = sys.argv[1:]
done = subprocess.run([os.environ['T311_CORE_PYTHON'], *args])
seed = os.environ.get('T311_SEED_CASE')
if (
    seed
    and done.returncode == 0
    and any(a.endswith('run_java.py') for a in args)
    and '--out' in args
):
    path = args[args.index('--out') + 1]
    with open(path) as handle:
        lines = handle.read().splitlines()
    out = []
    for line in lines:
        seen = json.loads(line)
        if seen.get('case', '').split('@')[0] == seed:
            seen['outcome'] = {{
                **seen['outcome'],
                'result': 'refusal',
                'code': 'INTERNAL_ERROR',
            }}
            line = json.dumps(seen, sort_keys=True)
        out.append(line)
    with open(path, 'w') as handle:
        handle.write(''.join(line + '\n' for line in out))
sys.exit(done.returncode)
"""


def _seeding_python(tmp_path: Path) -> Path:
    path = tmp_path / 'seeding-python'
    path.write_text(SEEDING_PYTHON.format(python=sys.executable))
    path.chmod(0o755)
    return path


def _java() -> dict[str, str]:
    r"""What the Java leg needs from this environment, passed through to
    the handoff: ``JAVA_HOME`` and ``SDM_JAVA_JAR``."""
    return {
        key: os.environ[key]
        for key in ('JAVA_HOME', 'SDM_JAVA_JAR')
        if os.environ.get(key)
    }


@G.gated(_ready)
@pytest.mark.parametrize('seeded', [True, False])
def test_t311_a_java_only_difference_stops_the_handoff_stamp(
    tmp_path: Path, seeded: bool
) -> None:
    t30 = _t30()
    core = os.environ['SDM_TEST_CORE_PYTHON']
    dsn = os.environ['SDM_POSTGRES_TEST_DSN']
    java = _java()
    _, work, _ = t30._clone(tmp_path)
    sha = t30._commit(
        work,
        f'{t30.PACKAGE}: a package that touches the binary-classification case',
        {'docs/t399-probe.md': 'A probe for the Java handoff test.\n'},
    )
    env = t30._env(
        tmp_path,
        SDM_PHASE_BRANCH=t30.PHASE,
        T311_CORE_PYTHON=core,
        SDM_POSTGRES_TEST_DSN=dsn,
        **java,
        **({'T311_SEED_CASE': SEEDED_CASE} if seeded else {}),
    )
    done = t30._handoff(
        work, env, 'run', '--package', t30.PACKAGE, '--case', SEEDED_CASE,
        '--python', str(_seeding_python(tmp_path)),
    )  # fmt: skip
    said = done.stdout + done.stderr
    check = t30._handoff(work, env, 'check', sha, timeout=120)
    if seeded:
        assert done.returncode == 1, said[-3000:]
        named = [
            line
            for line in said.splitlines()
            if JAVA.search(line) and 'match' not in line.lower()
        ]
        assert named, said[-3000:]
        assert SEEDED_CASE in said, said[-3000:]
        assert not t30._stamp(work, sha).exists()
        assert check.returncode == 1, check.stdout + check.stderr
    else:
        assert done.returncode == 0, said[-3000:]
        # The Java leg ran: its note names the case.
        assert any(
            JAVA.search(line) and SEEDED_CASE in line
            for line in said.splitlines()
        ), said[-3000:]
        stamp = json.loads(t30._stamp(work, sha).read_text(encoding='utf-8'))
        assert stamp['sha'] == sha and stamp['ok'] is True, stamp
        assert check.returncode == 0, check.stdout + check.stderr
