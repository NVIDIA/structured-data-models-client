# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32f's RelBench recipe keeps its data out of the tree (exit item (d)),
folded into T1.33.

    python -m pytest contract/tests/test_t133_relbench_recipe.py -q

Written by the package's independent test author from the design's T1.32f
row ("no RelBench or H&M data or payloads committed; only the recipe and
summary results") and the plan ("the recipe keeps the spec and samples
outside the tree; ``--out`` must point outside the worktree"), before the
implementer ports ``contract/tools/relbench.py`` (the prep branch's
abaeaee3 and f23551fe).

The tool writes three kinds of file: ``fetch`` RelBench's tables into
``--cache``, ``sample`` the samples (under the cache on the prep branch)
and the spec, ``run`` the spec and every probe's payloads under ``--out``.
So a worktree path is refused for ``--out`` and for ``--cache`` alike,
before anything is written or fetched, with a non-zero exit.

Each check runs the tool from a copy of the tree in a temporary directory
(``contract/``, the registry the tool reads and the register), made a git
repository, so the worktree is the copy's whichever way the tool finds it
(its own location or ``git rev-parse``) and no check can write into this
checkout. A path is "inside" in any spelling: absolute, relative to a
working directory in the copy, or through ``..``. ``git status --ignored``
of the copy, every untracked and ignored file listed, must stay empty. A
control with the same arguments outside the copy shows that each refusal
is the path's, not the arguments'. Another python stands in for the
reference and candidate environments in ``run``: it records its call and
fails, so ``run`` never reaches the differential. ``fetch`` runs with an
unreachable proxy, so nothing is downloaded whatever the tool does.

The roots alone are not the check: a directory or file below a root
outside the copy that is a symbolic link into it (the samples, the
download directory, the spec or a probe's output) is refused the same way
(T1.33 part 1's review, P2). So is a path spelled with ``~``, which the
tool must expand once and use as checked; a path in another letter case on
a disk that folds case; and a path in any git work tree, this
repository's other worktrees included, compared by file identity (the
orchestrator's rule). Sampling from inside the worktree spills nothing
there either: DuckDB's temporary files go elsewhere (the subagent's
review, P2-1, P2-2 and P3-2).

Statically: the tree tracks no data file (parquet, Arrow, Feather, DuckDB)
and no RelBench spec.

Gating (``contract/tests/t133_gate.py``): a test whose id is in
``t133-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
when written. It still runs and must still fail; once it passes it fails,
naming itself, until it is taken out of the list, which may only shrink.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = Path('contract/tools/relbench.py')
REGISTRY = Path(
    'packages/structured-data-models-relational-engine/src'
    '/sdm_relational_engine/rfm/assets/relbench_registry.json'
)
REGISTER = Path('docs/design/rust-core/register.yaml')
#: The dataset the sampling checks use, the recipe's first.
DATASET = 'rel-hm'
#: Data formats the tree never tracks.
DATA = ('.parquet', '.arrow', '.feather', '.ipc', '.duckdb')
#: No network for fetch, whatever the tool does.
OFFLINE = {
    'HTTPS_PROXY': 'http://127.0.0.1:9',
    'https_proxy': 'http://127.0.0.1:9',
    'HTTP_PROXY': 'http://127.0.0.1:9',
    'http_proxy': 'http://127.0.0.1:9',
    'NO_PROXY': '',
    'no_proxy': '',
}


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _module('t133_gate', ROOT / 'contract/tests/t133_gate.py')


def _load(path: Path) -> ModuleType:
    r"""The tool at ``path``, loaded afresh (each copy reads its own
    registry)."""
    spec = importlib.util.spec_from_file_location('relbench_under_test', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # No bytecode beside a copy's tool: the copy must stay as committed.
    writes, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = writes
    return module


def gated(request: pytest.FixtureRequest, check: Any) -> None:
    r"""Runs ``check`` held to the pending list, as ``py:<test id>``."""
    G.held('exits', f'py:{request.node.name}', check)


# --- the copy -------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            'git',
            '-c',
            'user.name=t133',
            '-c',
            'user.email=t133@example.invalid',
            '-c',
            'commit.gpgsign=false',
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def copy_of_the_tree(tmp: Path) -> Path:
    r"""The tool's part of the tree, copied and committed: a worktree of its
    own."""
    assert (ROOT / TOOL).is_file(), f'{TOOL} is not on the branch'
    repo = tmp / 'worktree'

    def ignored(directory: str, names: list[str]) -> set[str]:
        omitted = shutil.ignore_patterns('__pycache__', '.pytest_cache')(
            directory, names
        )
        # The recipe reads its registry and differential tools, never the
        # independent warehouse replay corpus. Keep every other input.
        if Path(directory) == ROOT / 'contract':
            omitted.add('captures')
        return omitted

    shutil.copytree(
        ROOT / 'contract',
        repo / 'contract',
        ignore=ignored,
    )
    for path in (REGISTRY, REGISTER):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, repo / path)
    _git(repo, 'init', '-q')
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '-m', 'copy')
    return repo.resolve()


def untouched(repo: Path) -> list[str]:
    r"""Every file or directory in the copy that is not as committed:
    ``git status``, ignored files included, and any path, an empty
    directory among them, that no tracked file accounts for."""
    status = _git(
        repo, 'status', '--porcelain', '--ignored', '--untracked-files=all'
    ).splitlines()
    tracked = {Path(p) for p in _git(repo, 'ls-files', '-z').split('\0') if p}
    known = tracked | {d for p in tracked for d in p.parents}
    extra = [
        f'not committed: {path}'
        for path in sorted(
            p.relative_to(repo)
            for p in repo.rglob('*')
            if p.relative_to(repo).parts[0] != '.git'
        )
        if path not in known
    ]
    return status + extra


def tool(
    repo: Path, *args: str, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-I', '-B', str(repo / TOOL), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        env={**os.environ, **(env or {})},
        timeout=300,
    )


def stand_in(tmp: Path) -> tuple[Path, Path]:
    r"""An executable that records its arguments in a log outside the copy,
    then fails; and that log."""
    log = tmp / 'stand-in.log'
    script = tmp / 'stand-in-python'
    script.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\nexit 3\n')
    script.chmod(0o755)
    return script, log


def synthetic_cache(repo: Path, cache: Path) -> None:
    r"""Tiny tables for :data:`DATASET` in the loader's layout under
    ``cache``: every table the recipe reads, with the registry's keys,
    links and time columns, its facts inside the recipe's window."""
    import duckdb

    relbench = _load(repo / TOOL)
    recipe = relbench.RECIPES[DATASET]
    tables = relbench._tables(DATASET)
    where = relbench.directory(cache, DATASET)
    where.mkdir(parents=True)
    since = recipe['window'][0]
    con = duckdb.connect()
    for table, spec in tables.items():
        columns = []
        if spec['primary_key'] is not None:
            columns.append(f'i::BIGINT AS "{spec["primary_key"]}"')
        for column in spec['foreign_keys']:
            columns.append(f'(i % 7)::BIGINT AS "{column}"')
        if spec['time_column'] is not None:
            columns.append(
                f"TIMESTAMP '{since}' + to_days((i % 20)::INTEGER) "
                f'AS "{spec["time_column"]}"'
            )
        for left in recipe.get('exclude', {}).get(table, []):
            columns.append(f'\'free text \' || i AS "{left}"')
        columns.append('(i * 1.5)::DOUBLE AS value')
        path = where / f'{table}.parquet'
        con.execute(
            f'COPY (SELECT {", ".join(columns)} FROM range(40) t(i)) '
            f"TO '{path}' (FORMAT parquet)"
        )


# --- the tree -------------------------------------------------------------------


def test_the_recipe_is_on_the_branch(request: pytest.FixtureRequest) -> None:
    def check() -> None:
        assert (ROOT / TOOL).is_file(), f'{TOOL} is not on the branch'
        relbench = _load(ROOT / TOOL)
        assert DATASET in relbench.RECIPES

    gated(request, check)


def test_the_tree_tracks_no_data_and_no_spec() -> None:
    tracked = subprocess.run(
        ['git', 'ls-files', '-z'],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split('\0')
    data = [p for p in tracked if p.lower().endswith(DATA)]
    assert data == [], data
    specs = []
    for path in tracked:
        if not path.endswith('.json'):
            continue
        try:
            loaded = json.loads((ROOT / path).read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(loaded, dict) and {'relbench', 'sources'} <= set(loaded):
            specs.append(path)
    assert specs == [], specs


@pytest.mark.tooling
def test_the_data_check_finds_a_seeded_file(tmp_path: Path) -> None:
    # Not vacuous: the same reading of a repository that tracks a sample.
    repo = tmp_path / 'seeded'
    (repo / 'contract').mkdir(parents=True)
    (repo / 'contract/customer.parquet').write_bytes(b'PAR1')
    (repo / 'contract/spec.json').write_text(
        json.dumps({'relbench': [DATASET], 'sources': []})
    )
    _git(repo, 'init', '-q')
    _git(repo, 'add', '-A')
    listed = _git(repo, 'ls-files').split()
    assert [p for p in listed if p.lower().endswith(DATA)] == [
        'contract/customer.parquet'
    ]
    assert 'contract/spec.json' in listed


# --- --out ----------------------------------------------------------------------


def _inside(repo: Path, spelling: str, name: str) -> tuple[str, Path]:
    r"""A path inside the copy, as ``spelling`` writes it, and the working
    directory it is read from."""
    if spelling == 'absolute':
        return str(repo / 'contract' / name), repo.parent
    if spelling == 'relative':
        return name, repo / 'contract'
    return str(repo / 'contract' / '..' / name), repo.parent


@pytest.mark.tooling
@pytest.mark.parametrize('spelling', ['absolute', 'relative', 'dotdot'])
def test_sample_refuses_an_out_inside_the_worktree(
    spelling: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        repo = copy_of_the_tree(tmp_path)
        cache = tmp_path / 'cache'
        synthetic_cache(repo, cache)
        out, cwd = _inside(repo, spelling, 'spec.json')
        done = tool(
            repo,
            'sample',
            '--cache',
            str(cache),
            '--out',
            out,
            DATASET,
            cwd=cwd,
        )
        assert done.returncode != 0, 'an out inside the worktree is refused'
        assert untouched(repo) == []
        assert not (cache / 'samples').exists(), 'refused before sampling'
        # The control: the same call with the spec outside the copy runs.
        outside = tmp_path / 'out' / 'spec.json'
        outside.parent.mkdir()
        done = tool(
            repo,
            'sample',
            '--cache',
            str(cache),
            '--out',
            str(outside),
            DATASET,
            cwd=cwd,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        assert outside.is_file()

    gated(request, check)


@pytest.mark.tooling
@pytest.mark.parametrize('spelling', ['absolute', 'relative', 'dotdot'])
def test_run_refuses_an_out_inside_the_worktree(
    spelling: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        repo = copy_of_the_tree(tmp_path)
        cache = tmp_path / 'cache'
        synthetic_cache(repo, cache)
        python, log = stand_in(tmp_path)
        out, cwd = _inside(repo, spelling, 'relbench-run')

        def call(where: str) -> subprocess.CompletedProcess[str]:
            return tool(
                repo,
                'run',
                '--cache',
                str(cache),
                '--reference-python',
                str(python),
                '--candidate-python',
                str(python),
                '--out',
                where,
                DATASET,
                cwd=cwd,
            )

        done = call(out)
        assert done.returncode != 0, 'an out inside the worktree is refused'
        assert untouched(repo) == []
        assert not log.exists(), 'refused before any environment is called'
        # The control: outside the copy, run gets as far as calling the
        # stand-in, which fails it.
        done = call(str(tmp_path / 'out'))
        assert done.returncode != 0
        assert log.is_file(), done.stdout + done.stderr
        assert untouched(repo) == []

    gated(request, check)


# --- --cache --------------------------------------------------------------------


@pytest.mark.tooling
@pytest.mark.parametrize('spelling', ['absolute', 'relative', 'dotdot'])
def test_sample_refuses_a_cache_inside_the_worktree(
    spelling: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    # sample writes the samples, RelBench's rows, beside the tables it reads
    # (under the cache on the prep branch): a cache in the tree would put
    # data there.
    def check() -> None:
        repo = copy_of_the_tree(tmp_path)
        cache, cwd = _inside(repo, spelling, 'relbench-cache')
        synthetic_cache(repo, (cwd / cache).resolve())
        _git(repo, 'add', '-A')
        _git(repo, 'commit', '-q', '-m', 'a cache in the tree, for the check')
        outside = tmp_path / 'spec.json'
        done = tool(
            repo,
            'sample',
            '--cache',
            cache,
            '--out',
            str(outside),
            DATASET,
            cwd=cwd,
        )
        assert done.returncode != 0, 'a cache inside the worktree is refused'
        assert untouched(repo) == []
        assert not outside.exists()

    gated(request, check)


@pytest.mark.tooling
@pytest.mark.parametrize('command', ['fetch', 'run'])
def test_fetch_and_run_refuse_a_cache_inside_the_worktree(
    command: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        repo = copy_of_the_tree(tmp_path)
        python, log = stand_in(tmp_path)
        cache = repo / 'contract' / 'relbench-cache'
        args = ['--cache', str(cache)]
        if command == 'run':
            args += [
                '--reference-python',
                str(python),
                '--candidate-python',
                str(python),
                '--out',
                str(tmp_path / 'out'),
            ]
        done = tool(repo, command, *args, DATASET, cwd=tmp_path, env=OFFLINE)
        assert done.returncode != 0, 'a cache inside the worktree is refused'
        assert untouched(repo) == []
        assert not log.exists(), 'refused before any environment is called'

    gated(request, check)


# --- what sample writes ---------------------------------------------------------


@pytest.mark.tooling
def test_a_sample_outside_the_worktree_writes_nothing_inside_it(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        repo = copy_of_the_tree(tmp_path)
        cache = tmp_path / 'cache'
        synthetic_cache(repo, cache)
        before = sorted(p for p in cache.rglob('*') if p.is_file())
        out = tmp_path / 'out' / 'spec.json'
        out.parent.mkdir()
        done = tool(
            repo,
            'sample',
            '--cache',
            str(cache),
            '--out',
            str(out),
            DATASET,
            cwd=repo,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        assert untouched(repo) == []
        spec = json.loads(out.read_text(encoding='utf-8'))
        assert spec['sources'], 'the spec has its questions'
        # The samples are written, outside the copy, and the spec reads
        # them from there: it names no path inside the worktree.
        written = sorted(p for p in tmp_path.rglob('*.parquet'))
        assert len(written) > len(before)
        assert all(repo not in p.parents for p in written)
        assert str(repo) not in out.read_text(encoding='utf-8')

    gated(request, check)


# --- descendants linked into the worktree (T1.33 part 1's review, P2) -----------
#
# The roots outside the copy, a directory or file below them a symbolic
# link into it: what the tool writes there lands in the worktree. Each is
# refused before anything is written or any environment called.


def _link(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=target.is_dir())


@pytest.mark.tooling
@pytest.mark.parametrize('linked', ['samples', f'samples/{DATASET}'])
def test_sample_refuses_samples_linked_into_the_worktree(
    linked: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        repo = copy_of_the_tree(tmp_path)
        cache = tmp_path / 'cache'
        synthetic_cache(repo, cache)
        _link(cache / linked, repo / 'contract')
        out = tmp_path / 'out' / 'spec.json'
        out.parent.mkdir()
        done = tool(
            repo,
            'sample',
            '--cache',
            str(cache),
            '--out',
            str(out),
            DATASET,
            cwd=tmp_path,
        )
        assert done.returncode != 0, 'samples linked into the worktree'
        assert untouched(repo) == []
        assert not out.exists(), 'refused before the spec is written'

    gated(request, check)


@pytest.mark.tooling
def test_fetch_refuses_downloads_linked_into_the_worktree(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    # The download directory's first part (the repository's owner) a link
    # into the copy: fetch would make the rest there and download into it.
    def check() -> None:
        repo = copy_of_the_tree(tmp_path)
        relbench = _load(repo / TOOL)
        cache = tmp_path / 'cache'
        where = relbench.directory(cache, DATASET).relative_to(cache)
        _link(cache / where.parts[0], repo / 'contract')
        done = tool(
            repo,
            'fetch',
            '--cache',
            str(cache),
            DATASET,
            cwd=tmp_path,
            env=OFFLINE,
        )
        assert done.returncode != 0, 'downloads linked into the worktree'
        assert untouched(repo) == []

    gated(request, check)


@pytest.mark.tooling
@pytest.mark.parametrize(
    'linked',
    ['spec.json', 'old.json', 'core.json', 'corrected.json', 'samples'],
)
def test_run_refuses_outputs_linked_into_the_worktree(
    linked: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    # A probe's output (or the spec) under --out a link to a file in the
    # copy that does not exist yet; or the cache's samples a link into it.
    def check() -> None:
        repo = copy_of_the_tree(tmp_path)
        cache = tmp_path / 'cache'
        synthetic_cache(repo, cache)
        out = tmp_path / 'out'
        out.mkdir()
        if linked == 'samples':
            _link(cache / 'samples', repo / 'contract')
        else:
            _link(out / linked, repo / 'contract' / linked)
        python, log = stand_in(tmp_path)
        done = tool(
            repo,
            'run',
            '--cache',
            str(cache),
            '--reference-python',
            str(python),
            '--candidate-python',
            str(python),
            '--out',
            str(out),
            DATASET,
            cwd=tmp_path,
        )
        assert done.returncode != 0, f'{linked} linked into the worktree'
        assert untouched(repo) == []
        assert not log.exists(), 'refused before any environment is called'

    gated(request, check)


# --- other spellings and other work trees (T1.33 part 1's subagent review) ------


def _home(tmp: Path) -> dict[str, str]:
    r"""A home directory outside the copy, so ``~`` expanded is harmless."""
    home = tmp / 'home'
    home.mkdir(exist_ok=True)
    return {'HOME': str(home)}


@pytest.mark.tooling
@pytest.mark.parametrize('command', ['sample', 'run'])
def test_an_out_spelled_with_a_tilde_writes_nothing_in_the_worktree(
    command: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    # ``--out=~/…`` reaches the tool unexpanded (one argument, no shell):
    # the path checked and the path written are one, so nothing lands in
    # ``<cwd>/~/`` with the copy as the working directory.
    def check() -> None:
        repo = copy_of_the_tree(tmp_path)
        cache = tmp_path / 'cache'
        synthetic_cache(repo, cache)
        env = _home(tmp_path)
        (Path(env['HOME']) / 't133-tilde').mkdir()
        python, _log = stand_in(tmp_path)
        out = '--out=~/t133-tilde/spec.json'
        extra: list[str] = []
        if command == 'run':
            out = '--out=~/t133-tilde-run'
            extra = [
                '--reference-python',
                str(python),
                '--candidate-python',
                str(python),
            ]
        done = tool(
            repo,
            command,
            '--cache',
            str(cache),
            *extra,
            out,
            DATASET,
            cwd=repo,
            env=env,
        )
        assert untouched(repo) == [], done.stdout + done.stderr
        if command == 'sample':
            assert done.returncode == 0, done.stdout + done.stderr
            assert (Path(env['HOME']) / 't133-tilde/spec.json').is_file()

    gated(request, check)


@pytest.mark.tooling
@pytest.mark.parametrize('command', ['fetch', 'sample'])
def test_a_cache_spelled_with_a_tilde_writes_nothing_in_the_worktree(
    command: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        repo = copy_of_the_tree(tmp_path)
        env = _home(tmp_path)
        synthetic_cache(repo, Path(env['HOME']) / 'cache')
        extra = (
            ['--out', str(tmp_path / 'spec.json')]
            if command == 'sample'
            else []
        )
        done = tool(
            repo,
            command,
            '--cache=~/cache',
            *extra,
            DATASET,
            cwd=repo,
            env={**env, **OFFLINE},
        )
        assert untouched(repo) == [], done.stdout + done.stderr
        if command == 'sample':
            assert done.returncode == 0, done.stdout + done.stderr
            assert (Path(env['HOME']) / 'cache/samples').is_dir()

    gated(request, check)


def _folds_case(tmp: Path) -> bool:
    probe = tmp / 'case-probe'
    probe.mkdir(exist_ok=True)
    return (tmp / 'CASE-PROBE').exists()


def _case_kind() -> str:
    import tempfile

    with tempfile.TemporaryDirectory() as where:
        folds = _folds_case(Path(where))
    return 'case-insensitive' if folds else 'case-sensitive'


@pytest.mark.tooling
@pytest.mark.parametrize('disk', [_case_kind()])
def test_an_out_spelled_in_another_case_is_refused(
    disk: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    # The worktree spelled in other letter case: on a disk that folds case
    # (macOS's default) the same directory, compared by identity; on one
    # that does not, another path, which must stay outside the copy.
    def check() -> None:
        repo = copy_of_the_tree(tmp_path)
        cache = tmp_path / 'cache'
        synthetic_cache(repo, cache)
        variant = repo.parent / repo.name.upper() / 'spec.json'
        done = tool(
            repo,
            'sample',
            '--cache',
            str(cache),
            '--out',
            str(variant),
            DATASET,
            cwd=tmp_path,
        )
        assert untouched(repo) == [], done.stdout + done.stderr
        if disk == 'case-insensitive':
            assert done.returncode != 0, 'the worktree in another case'

    gated(request, check)


@pytest.mark.tooling
@pytest.mark.parametrize('other', ['sibling worktree', 'another repository'])
@pytest.mark.parametrize('option', ['out', 'cache'])
def test_a_path_in_any_other_git_work_tree_is_refused(
    option: str, other: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    # The orchestrator's rule: no path inside any git work tree, this
    # repository's other worktrees included.
    def check() -> None:
        repo = copy_of_the_tree(tmp_path)
        tree = tmp_path / 'other'
        if other == 'sibling worktree':
            _git(repo, 'worktree', 'add', '-q', '--detach', str(tree))
        else:
            tree.mkdir()
            _git(tree, 'init', '-q')
            (tree / 'README').write_text('another repository\n')
            _git(tree, 'add', '-A')
            _git(tree, 'commit', '-q', '-m', 'another')
        tree = tree.resolve()
        cache = tmp_path / 'cache'
        synthetic_cache(repo, cache)
        out = tmp_path / 'spec.json'
        if option == 'out':
            out = tree / 'spec.json'
        else:
            target = tree / 'relbench-cache'
            synthetic_cache(repo, target)
            _git(tree, 'add', '-A')
            _git(tree, 'commit', '-q', '-m', 'a cache, for the check')
            cache = target
        done = tool(
            repo,
            'sample',
            '--cache',
            str(cache),
            '--out',
            str(out),
            DATASET,
            cwd=tmp_path,
        )
        assert done.returncode != 0, f'--{option} inside {other}'
        assert untouched(tree) == []
        assert untouched(repo) == []

    gated(request, check)


@pytest.mark.tooling
def test_sampling_spills_nothing_into_the_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    # Run from inside the copy, the sampler's DuckDB spills to its
    # temp_directory (by default '.tmp' under the working directory). A
    # spill is forced on the tool's own connection just before it writes
    # the samples, and the copy read while that connection is open
    # (DuckDB removes its temporary files when it closes).
    import duckdb

    def check() -> None:
        repo = copy_of_the_tree(tmp_path)
        cache = tmp_path / 'cache'
        synthetic_cache(repo, cache)
        relbench = _load(repo / TOOL)
        seen: list[list[str]] = []
        connect = duckdb.connect

        class Spilling:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._con = connect(*args, **kwargs)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._con, name)

            def execute(self, sql: str, *args: Any) -> Any:
                if sql.lstrip().upper().startswith('COPY') and not seen:
                    limit = self._con.execute(
                        "SELECT current_setting('memory_limit')"
                    ).fetchone()[0]
                    self._con.execute("SET memory_limit = '20MB'")
                    self._con.execute(
                        'CREATE TEMP TABLE t133_spill AS SELECT i, '
                        "repeat('x', 50) || i AS s FROM range(2000000) r(i)"
                    )
                    seen.append(untouched(repo))
                    self._con.execute('DROP TABLE t133_spill')
                    self._con.execute(f"SET memory_limit = '{limit}'")
                return self._con.execute(sql, *args)

        monkeypatch.chdir(repo)
        monkeypatch.setattr(duckdb, 'connect', Spilling)
        writes, sys.dont_write_bytecode = sys.dont_write_bytecode, True
        try:
            code = relbench.main(
                [
                    'sample',
                    '--cache',
                    str(cache),
                    '--out',
                    str(tmp_path / 'spec.json'),
                    DATASET,
                ]
            )
        finally:
            sys.dont_write_bytecode = writes
            monkeypatch.undo()
        assert code == 0
        assert seen, 'the sampler wrote no samples'
        assert seen[0] == [], f'spilled into the worktree: {seen[0]}'
        assert untouched(repo) == []

    gated(request, check)
