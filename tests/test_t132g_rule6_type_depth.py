# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32g: rule 6 reads a nested Arrow type text from the reference
observations without recursing past a bound (E49's depth probe, the
subagent's review of part 1, P2).

Written by the package's independent test author. Rule 6 reads each
column's type as pyarrow writes it (``scripts/rust_core/coverage.py``'s
``cell_type``), looking through a dictionary's values and a run-end-encoded
array's values. That text is outside JSON: a reference line holds it. Each
seed below changes the type of ``key.uint8``'s ``user_id`` in its
from_source samples, in a commit whose objects live in a temporary object
directory (nothing of the repository is written), and runs
``coverage.py --check`` on that commit.

- A dictionary 2,000 levels deep, or a run-end-encoded type nested as
  deep, is refused by name: exit 1, with a ``rule 6:`` problem naming
  ``key.uint8`` or a ``read:`` problem, and no traceback or
  ``RecursionError``. Its claims, ``uint8/primary-key`` and
  ``uint8/foreign-key``, do not hold on a type that was not read.
- Nesting three deep is read as the type it holds: a dictionary of a
  dictionary of a dictionary of ``uint8`` (and the same run-end encoded) is
  ``uint8``, so ``key.uint8``'s claims hold and the check passes.

Gating (``contract/tests/t132g_gate.py``): a test named in
``t132g-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
when written. It still runs and must still fail; once it passes it fails,
naming itself, until it is taken out of the list.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / 'scripts/rust_core/coverage.py'
CASE = 'key.uint8'
#: Deep enough to pass any interpreter's default recursion limit.
DEEP = 2000


def _module(name: str, path: Path) -> Any:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _module('t132g_gate', ROOT / 'contract/tests/t132g_gate.py')


def dictionary(values: str, depth: int) -> str:
    r"""``values`` as a dictionary's, nested ``depth`` deep, as pyarrow
    writes a dictionary type."""
    return (
        'dictionary<values=' * depth
        + values
        + ', indices=int32, ordered=0>' * depth
    )


def run_end_encoded(values: str, depth: int) -> str:
    r"""``values`` as a run-end-encoded array's, nested ``depth`` deep, as
    pyarrow writes the type."""
    return (
        'run_end_encoded<run_ends: int32, values: ' * depth
        + values
        + '>' * depth
    )


def _git(env: dict[str, str], *args: str, text_in: str | None = None) -> str:
    return subprocess.run(
        ['git', '-C', str(ROOT), *args],
        input=text_in,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.strip()


def _environment(tmp: Path) -> dict[str, str]:
    common = Path(_git(dict(os.environ), 'rev-parse', '--git-common-dir'))
    if not common.is_absolute():
        common = ROOT / common
    objects = tmp / 'objects'
    objects.mkdir(parents=True)
    return {
        **os.environ,
        'GIT_OBJECT_DIRECTORY': str(objects),
        'GIT_ALTERNATE_OBJECT_DIRECTORIES': str(common / 'objects'),
        'GIT_INDEX_FILE': str(tmp / 'index'),
        'GIT_AUTHOR_NAME': 'seed',
        'GIT_AUTHOR_EMAIL': 'seed@example.invalid',
        'GIT_COMMITTER_NAME': 'seed',
        'GIT_COMMITTER_EMAIL': 'seed@example.invalid',
    }


def _reference_path() -> str:
    names = _git(
        dict(os.environ),
        'ls-tree',
        '--name-only',
        'HEAD',
        'contract/reference/',
    )
    [path] = [n for n in names.splitlines() if n.endswith('.jsonl')]
    return path


def retyped(tmp: Path, arrow: str) -> subprocess.CompletedProcess[str]:
    r"""``coverage.py --check`` of HEAD with ``key.uint8``'s ``user_id``
    sampled at from_source as ``arrow``."""
    env = _environment(tmp)
    path = _reference_path()
    lines = []
    changed = 0
    for text in _git(dict(os.environ), 'show', f'HEAD:{path}').splitlines():
        line = json.loads(text)
        if line['case'] == CASE:
            for statement in line['driver']['capture']['statements']:
                if statement['stage'] != 'from_source':
                    continue
                for result in statement['results']:
                    for field in result.get('schema') or []:
                        if field['name'] == 'user_id':
                            field['type'] = arrow
                            changed += 1
            text = json.dumps(line, sort_keys=True)
        lines.append(text)
    assert changed, 'key.uint8 samples user_id'
    blob = _git(
        env, 'hash-object', '-w', '--stdin', text_in='\n'.join(lines) + '\n'
    )
    _git(env, 'read-tree', 'HEAD')
    _git(env, 'update-index', '--cacheinfo', f'100644,{blob},{path}')
    commit = _git(
        env, 'commit-tree', _git(env, 'write-tree'), '-p', 'HEAD', '-m', 'seed'
    )
    return subprocess.run(
        [
            sys.executable,
            str(COVERAGE),
            '--check',
            '--repo',
            str(ROOT),
            '--rev',
            commit,
            '--checkpoint',
            'HEAD',
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )


def refused_by_name(done: subprocess.CompletedProcess[str]) -> None:
    output = done.stdout + done.stderr
    assert 'Traceback' not in output, output[-2000:]
    assert 'RecursionError' not in output, output[-2000:]
    assert done.returncode == 1, output[-2000:]
    assert any(
        line.startswith('read:')
        or (line.startswith('rule 6:') and CASE in line)
        for line in output.splitlines()
    ), output[-2000:]


NESTINGS: dict[str, Callable[[str, int], str]] = {
    'dictionary': dictionary,
    'run_end_encoded': run_end_encoded,
}


@pytest.mark.parametrize('nesting', list(NESTINGS))
def test_a_type_nested_past_the_bound_is_refused_by_name(
    nesting: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        refused_by_name(retyped(tmp_path, NESTINGS[nesting]('uint8', DEEP)))

    G.held('exits', f'py:{request.node.name}', check)


@pytest.mark.parametrize('nesting', list(NESTINGS))
def test_a_type_nested_three_deep_is_read_as_its_values(
    nesting: str, tmp_path: Path
) -> None:
    done = retyped(tmp_path, NESTINGS[nesting]('uint8', 3))
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_nested_texts_are_pyarrow_s() -> None:
    # The seeds' text is pyarrow's own, so the shallow controls read what a
    # driver returns.
    pa = pytest.importorskip('pyarrow')
    nested = pa.uint8()
    for _ in range(3):
        nested = pa.dictionary(pa.int32(), nested)
    assert str(nested) == dictionary('uint8', 3)
    ree = pa.uint8()
    for _ in range(3):
        ree = pa.run_end_encoded(pa.int32(), ree)
    assert str(ree) == run_end_encoded('uint8', 3)
