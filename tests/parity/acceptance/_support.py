# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Helpers shared by the phase 2 acceptance checks.

The interface the checks rely on is written down in ``CONTRACT.md``.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
PARITY = REPO / 'tests' / 'parity'
TRANSCRIPTS = PARITY / 'transcripts'
SCHEMA = PARITY / 'schema' / 'transcript.schema.json'
SCRIPTS = REPO / 'scripts' / 'parity'
RUN_CORPUS = SCRIPTS / 'run_corpus.py'
DIFF = SCRIPTS / 'diff.py'
SCRUB = SCRIPTS / 'scrub.py'
COVERAGE = SCRIPTS / 'coverage.py'
GOLDEN = REPO / 'tests' / 'golden' / 'wire_payloads.json'

TASK_KINDS = (
    'count',
    'classification',
    'regression',
    'forecast',
    'ranking',
    'inactivity',
)
OFFLINE_BACKENDS = ('sqlite', 'duckdb', 'postgres', 'memory')
LIVE_BACKENDS = ('databricks', 'lakebase', 'snowflake')
STEP_OPS = ('execute', 'ingest', 'register', 'commit', 'paramstyle')
NAME = re.compile(r'^[A-Za-z0-9_./#-]+$')

# Environment variables that could carry a credential or reach a server. A
# replay must not need any of them.
_CREDENTIAL_PREFIXES = (
    'DATABRICKS_',
    'SNOWFLAKE_',
    'PG',
    'SDM_POSTGRES',
    'SDM_REQUIRE_POSTGRES',
    'SDM_RELATIONAL_URL',
    'SDM_TABULAR_URL',
    'KUMO_',
    'AWS_',
    'AZURE_',
    'GOOGLE_',
    'LAKEBASE',
)

_NETBLOCK = r"""
import socket

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _refuse(*args, **kwargs):
    raise OSError('network access is blocked during an offline replay')


def _connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _refuse()
    return _real_connect(self, address)


def _connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _refuse()
    return _real_connect_ex(self, address)


socket.socket.connect = _connect
socket.socket.connect_ex = _connect_ex
socket.getaddrinfo = _refuse
socket.gethostbyname = _refuse
socket.create_connection = _refuse
"""


def run(
    *args: Any, env: dict[str, str] | None = None, timeout: float = 900
) -> subprocess.CompletedProcess[str]:
    r"""Run a Python script from the repository root."""
    return subprocess.run(
        [sys.executable, *(str(a) for a in args)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def output(proc: subprocess.CompletedProcess[str]) -> str:
    return f'exit {proc.returncode}\n--- stdout\n{proc.stdout}\n--- stderr\n{proc.stderr}'


def offline_env(home: Path) -> dict[str, str]:
    r"""An environment with no credentials, no database and no network.

    Python sockets to IP addresses and name lookups raise. libpq doesn't use
    Python sockets, so it is pointed at a socket directory that doesn't exist.
    """
    block = home / 'netblock'
    block.mkdir(parents=True, exist_ok=True)
    (block / 'sitecustomize.py').write_text(_NETBLOCK)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_CREDENTIAL_PREFIXES)
    }
    env['HOME'] = str(home)
    env['PGHOST'] = str(home / 'no-postgres-here')
    env['PGPORT'] = '1'
    env['PGCONNECT_TIMEOUT'] = '1'
    env['PYTHONPATH'] = os.pathsep.join(
        p for p in (str(block), env.get('PYTHONPATH', '')) if p
    )
    return env


def listing() -> list[dict[str, Any]]:
    r"""The runner's scenario list, checked against the contract's shape."""
    proc = run(RUN_CORPUS, 'list', '--json', timeout=300)
    assert proc.returncode == 0, output(proc)
    entries = json.loads(proc.stdout)
    assert isinstance(entries, list) and entries, 'list --json gave no array'
    names = [entry['name'] for entry in entries]
    assert len(names) == len(set(names)), 'scenario names are not unique'
    for entry in entries:
        assert NAME.match(entry['name']), entry
        assert entry['status'] in ('recorded', 'pending-live'), entry
        if entry['status'] == 'recorded':
            assert entry['transcript'], entry
            assert (REPO / entry['transcript']).is_file(), entry
        else:
            assert entry['transcript'] is None, entry
    return entries


def recorded() -> list[dict[str, Any]]:
    return [e for e in listing() if e['status'] == 'recorded']


def load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def save(path: Path, transcript: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(transcript, indent=1))


def transcript_files(root: Path = TRANSCRIPTS) -> list[Path]:
    return sorted(root.rglob('*.json'))


def execute_steps(transcript: dict[str, Any]) -> list[int]:
    return [
        i
        for i, step in enumerate(transcript['steps'])
        if step.get('op') == 'execute'
    ]


def arrow_refs(transcript: dict[str, Any]) -> list[str]:
    refs = [
        step['result']
        for step in transcript['steps']
        if isinstance(step.get('result'), str)
    ]
    if isinstance(transcript.get('result'), str):
        refs.append(transcript['result'])
    return refs


def read_arrow(path: Path) -> tuple[Any, str]:
    r"""The table in an Arrow IPC file, and which IPC format it used."""
    import pyarrow as pa

    data = Path(path).read_bytes()
    try:
        return pa.ipc.open_file(pa.BufferReader(data)).read_all(), 'file'
    except pa.ArrowInvalid:
        return pa.ipc.open_stream(pa.BufferReader(data)).read_all(), 'stream'


def write_arrow(path: Path, table: Any, kind: str) -> None:
    import pyarrow as pa

    with pa.OSFile(str(path), 'wb') as sink:
        opener = pa.ipc.new_file if kind == 'file' else pa.ipc.new_stream
        with opener(sink, table.schema) as writer:
            writer.write_table(table)


def copy_corpus(destination: Path) -> Path:
    r"""A scratch copy of the stored transcripts, safe to mutate."""
    shutil.copytree(TRANSCRIPTS, destination)
    return destination


def in_copy(copy_root: Path, transcript: str | Path) -> Path:
    r"""Where a stored transcript (repo-relative path) lives in a copy."""
    return copy_root / (REPO / transcript).relative_to(TRANSCRIPTS)


def find_by_name(root: Path, name: str) -> Path:
    for path in transcript_files(root):
        if load(path).get('name') == name:
            return path
    raise AssertionError(f'no transcript named {name} under {root}')


# Paths into JSON documents.

Path_ = list[Any]


def leaves(value: Any, path: Path_ | None = None) -> list[Path_]:
    r"""Paths of every scalar in *value*, depth first, in document order."""
    path = path or []
    if isinstance(value, dict):
        out: list[Path_] = []
        for key, child in value.items():
            out += leaves(child, [*path, key])
        return out
    if isinstance(value, list):
        out = []
        for index, child in enumerate(value):
            out += leaves(child, [*path, index])
        return out
    return [path]


def get(value: Any, path: Path_) -> Any:
    for part in path:
        value = value[part]
    return value


def put(value: Any, path: Path_, new: Any) -> None:
    get(value, path[:-1])[path[-1]] = new


def changed(value: Any) -> Any:
    r"""A different value of the same JSON type."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.5
    if isinstance(value, str):
        return value + 'x'
    if value is None:
        return 'mutated'
    raise TypeError(type(value))


def mutate_first_leaf(value: Any, skip: tuple[str, ...] = ()) -> Path_:
    r"""Change the first scalar not under a key in *skip*; return its path."""
    for path in leaves(value):
        if not any(part in skip for part in path):
            put(value, path, changed(get(value, path)))
            return path
    raise AssertionError('nothing to mutate')


def deep(value: Any) -> Any:
    return copy.deepcopy(value)


# Replay output.

_LINE = re.compile(r'^(PASS|FAIL) (\S+)(.*)$')


def replay_lines(text: str) -> dict[str, tuple[str, str]]:
    r"""``{name: (PASS|FAIL, rest of the line)}`` from replay's stdout."""
    out = {}
    for line in text.splitlines():
        match = _LINE.match(line.strip())
        if match:
            out[match.group(2)] = (match.group(1), match.group(3))
    return out


def names_step(reason: str, index: int) -> bool:
    return bool(
        re.search(rf'\bstep {index}\b', reason) or f'steps[{index}]' in reason
    )


# Pytest runners.


def run_junit(
    runner: Path, report: Path, env: dict[str, str] | None = None
) -> tuple[int, dict[str, str], str]:
    r"""Run a pytest file; ``(exit, {test name: outcome}, output tail)``."""
    proc = run(
        '-m',
        'pytest',
        runner,
        '-q',
        '-p',
        'no:cacheprovider',
        f'--junitxml={report}',
        env=env,
        timeout=1800,
    )
    outcomes = {}
    if report.exists():
        for case in ET.parse(report).getroot().iter('testcase'):
            outcome = 'passed'
            for tag in ('failure', 'error', 'skipped'):
                if case.find(tag) is not None:
                    outcome = tag
            outcomes[case.get('name', '')] = outcome
    return proc.returncode, outcomes, proc.stdout[-4000:] + proc.stderr[-4000:]


def ran(outcomes: dict[str, str], case_id: str) -> list[str]:
    r"""Outcomes of the tests whose pytest id is *case_id*."""
    return [o for name, o in outcomes.items() if name.endswith(f'[{case_id}]')]
