# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ratchet's seeded checks (T1.14's exit; design, "Reference observations
and the gate"), each on a real git history.

The repository's harness, cases, reference, register and baselines are
copied from this checkout into a temporary repository and committed as two
landing commits, the second one holding ``ratchet.py``, so it is a
checkpoint whose harness judges. Each test commits a candidate on top, seeded
with one fault or one allowed change, and runs ``ratchet.py``'s checks on it,
with observations made from the committed reference observations, relabelled
for each runner (no runner runs here; ``contract/tests/test_gate_controls.py``
runs the real ones through ``gate.sh``). Each check that must fail is shown to
pass first on the unseeded candidate, so a failure is the seed's.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.tooling

ROOT = Path(__file__).resolve().parents[1]
#: What the temporary repository copies: the harness and everything it
#: reads, and a product file the protected list takes out of protection.
COPIED = (
    '.github/workflows/ci.yml',
    'contract',
    'docs/design/rust-core',
    'rust/crates/sdm-core/tests/pql',
    'rust/crates/sdm-testkit/src',
    'scripts/rust_core',
    'packages/structured-data-models-client/src/sdm_client/_routing.py',
    'packages/structured-data-models-client/src/sdm_client/query.py',
)
# The ratchet unit histories never execute a warehouse replay or read its
# per-case snapshots/Arrow tables. Keep the real manifests, cases, reference,
# protection rules and checker sources; capture validation has its own tests.
UNCOPIED_PAYLOADS = (
    'contract/captures/r2/*/*/capture.json',
    'contract/captures/r2/*/*/tables/*.arrows',
)
IDENTITY = ['-c', 'user.name=Gate Test', '-c', 'user.email=gate@example.com']
CELL = 'ref-pandas3'


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered first: a dataclass reads its module from sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ratchet = _module(ROOT / 'scripts/rust_core/ratchet.py', 'ratchet')
#: CI4's pending list (the ``test_ci4_*`` tests).
CI4 = _module(ROOT / 'contract/tests/ci4_gate.py', 'ci4_gate')


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ['git', '-C', str(repo), *IDENTITY, *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def commit(
    repo: Path,
    subject: str = 'T1.15: a change',
    edit: Callable[[Path], None] | None = None,
    trailers: dict[str, str] | None = None,
) -> str:
    r"""Commits ``edit``'s changes (or none) under ``subject``."""
    if edit is not None:
        edit(repo)
    git(repo, 'add', '-A')
    message = subject
    if trailers:
        message += '\n\n' + '\n'.join(f'{k}: {v}' for k, v in trailers.items())
    git(repo, 'commit', '-q', '--allow-empty', '-m', message)
    return git(repo, 'rev-parse', 'HEAD')


def land(
    repo: Path,
    package: str = 'T1.15',
    edit: Callable[[Path], None] | None = None,
    protected: str = 'none',
) -> str:
    r"""A package and its landing commit, as land.py makes it: empty, its
    subject ``Land <ID>: ...``, with the base it landed on."""
    base = git(repo, 'rev-parse', 'HEAD')
    commit(repo, f'{package}: the package', edit)
    return commit(
        repo,
        f'Land {package}: the package',
        trailers={
            'Reviewed-by': f'independent reviewer and Codex, {package}',
            'Protected-paths': protected,
            'Base-SHA': base,
            'Checkpoint-SHA': base,
        },
    )


def build(repo: Path) -> tuple[Path, str]:
    r"""The temporary repository, in ``repo``: a first landing with the
    harness's files but ``ratchet.py``, then a landing adding it, the
    checkpoint; with that landing's SHA."""
    git(repo, 'init', '-q', '-b', 'rust-core-phase-1')
    listed = subprocess.run(
        ['git', '-C', str(ROOT), 'ls-files', '-z', '--', *COPIED],
        capture_output=True,
        check=True,
    ).stdout.split(b'\0')
    for name in filter(None, (item.decode() for item in listed)):
        if any(Path(name).match(pattern) for pattern in UNCOPIED_PAYLOADS):
            continue
        source = ROOT / name
        if source.is_file():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, repo / name)
    # The client tests the scope manifests claim a path by (``tests``,
    # coverage's rule 4 from T3.2), which the copy's tree must define as the
    # candidate's does.
    for manifest in sorted(
        repo.glob('docs/design/rust-core/phase-*/scope-*.yaml')
    ):
        text = manifest.read_text(encoding='utf-8')
        for name in sorted(set(re.findall(r'([\w./-]+\.py)::', text))):
            if (ROOT / name).is_file():
                (repo / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, repo / name)
    ratchet_file = repo / 'scripts/rust_core/ratchet.py'
    kept = ratchet_file.read_bytes()
    ratchet_file.unlink()
    commit(
        repo,
        'Land T1.13: the start',
        trailers={
            'Reviewed-by': 'test',
            'Base-SHA': 'f' * 40,
            'Checkpoint-SHA': 'f' * 40,
        },
    )
    landed = land(
        repo,
        'T1.14',
        lambda r: ratchet_file.write_bytes(kept),
        protected='scripts/rust_core/ratchet.py (new)',
    )
    return repo, landed


@pytest.fixture(scope='module')
def origin(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    return build(tmp_path_factory.mktemp('origin'))


@pytest.fixture
def repo(origin: tuple[Path, str], tmp_path: Path) -> Path:
    r"""A worktree at the checkpoint, for one test's candidate."""
    source, landed = origin
    tree = tmp_path / 'tree'
    # Passed test tmp_path retention may remove a directory while its Git
    # registration remains. This origin belongs only to this module's unit
    # fixtures; pruning missing registrations keeps extant failed evidence.
    git(source, 'worktree', 'prune', '--expire', 'now')
    git(source, 'worktree', 'add', '-q', '--detach', str(tree), landed)
    return tree


def test_the_fixture_reuses_a_removed_worktree_without_pruning_failed_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / 'origin'
    source.mkdir()
    git(source, 'init', '-q', '-b', 'fixture')
    landed = commit(source, 'fixture origin')
    create = repo.__wrapped__
    original = create((source, landed), tmp_path / 'reused')
    failed = create((source, landed), tmp_path / 'failed')
    evidence = failed / 'failure.txt'
    evidence.write_text('preserve this failed test evidence')
    # pytest's passed-only tmp_path cleanup removes the directory without
    # knowing that Git still has a worktree registration for this path.
    shutil.rmtree(original)
    replacement = create((source, landed), tmp_path / 'reused')
    assert replacement == original and replacement.is_dir()
    assert evidence.read_text() == 'preserve this failed test evidence'
    registrations = git(source, 'worktree', 'list', '--porcelain')
    assert f'worktree {replacement}' in registrations
    assert f'worktree {failed}' in registrations


def test_the_synthetic_fixture_omits_only_bulk_r2_capture_inputs(
    origin: tuple[Path, str],
) -> None:
    source, _ = origin
    copied = source / 'contract/captures/r2'
    manifests = sorted(copied.glob('*/manifest.json'))
    expected = sorted((ROOT / 'contract/captures/r2').glob('*/manifest.json'))
    assert manifests and len(manifests) == len(expected)
    for path in manifests:
        assert (
            path.read_bytes() == (ROOT / path.relative_to(source)).read_bytes()
        )
    assert not list(copied.glob('*/*/capture.json'))
    assert not list(copied.glob('*/*/tables/*.arrows'))
    assert (source / 'contract/protected-paths.txt').read_bytes() == (
        ROOT / 'contract/protected-paths.txt'
    ).read_bytes()
    assert (
        source / 'contract/cases/s2/databricks.name.three-part.json'
    ).read_bytes() == (
        ROOT / 'contract/cases/s2/databricks.name.three-part.json'
    ).read_bytes()


def test_an_omitted_capture_path_still_requires_protected_edit_review(
    repo: Path,
) -> None:
    # Use an actual payload path with tiny synthetic bytes. These unit
    # checks exercise Git protection, never the real capture validator.
    original = next((ROOT / 'contract/captures/r2').glob('*/*/tables/*.arrows'))
    path = original.relative_to(ROOT).as_posix()

    def write_capture(text: bytes) -> Callable[[Path], None]:
        def write(tree: Path) -> None:
            target = tree / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text)

        return write

    landed = land(
        repo, 'T1.15', write_capture(b'synthetic baseline'), protected=path
    )
    commit(repo, edit=write_capture(b'synthetic changed capture'))
    assert check(repo, landed, protected='docs only') == [
        f'protected: {path}: changed, and the Reviewed-by and Protected-paths '
        'trailers do not name it'
    ]
    assert check(repo, landed, protected=path) == []


def checkpoint_of(repo: Path) -> str:
    return git(repo, 'rev-parse', 'HEAD')


def reference(repo: Path) -> dict[str, dict[str, Any]]:
    path = next((repo / 'contract/reference').glob('*.jsonl'))
    return {
        line['case']: line
        for line in map(json.loads, path.read_text().splitlines())
    }


def observe(
    repo: Path,
    tmp_path: Path,
    runner: str,
    change: Callable[[dict[str, dict[str, Any]]], None] | None = None,
) -> Path:
    r"""A runner's observations, made from the reference's: relabelled,
    and for rust-host with what it cannot observe left null; where the
    checkpoint's tree holds a baseline of the case for the runner, its
    projected fields are the baseline's (read from this checkout, whose
    files the checkpoint holds), so the unseeded observation is
    what that runner is held to wherever an admission has moved it from
    the reference (an approved transition, or python-old's recorded crash),
    whichever cases are admitted. Only the reference's DuckDB lines (the
    bare case IDs) are replayed: a runner's file is one database's run, and
    the reference's ``<case>@sqlite`` and ``<case>@postgres`` lines
    (T2.1a) are other runs', which a scenario about them adds itself."""
    # This checkout's, which the temporary repository's checkpoint copied:
    # a seed that edits or breaks the candidate's baseline files changes
    # nothing here.
    held = {}
    baselines = ROOT / f'contract/baselines/{CELL}/{runner}.jsonl'
    if baselines.is_file():
        held = {
            line['case']: line['projection']
            for line in map(json.loads, baselines.read_text().splitlines())
        }
    cases = {
        json.loads(path.read_text())['id']: json.loads(path.read_text())
        for path in (repo / 'contract/cases').glob('*/*.json')
    }
    lines = {}
    for case, line in reference(repo).items():
        if '@' in case:
            continue
        line = copy.deepcopy(line)
        line['identity']['runner'] = runner
        line['identity']['reference'] = None
        if runner == 'python-core':
            line['core'] = {'inputs': []}
        if runner == 'rust-host':
            line['outcome']['stage'] = line['outcome']['error'] = None
            line['host'] = line['public'] = None
            line['core'] = {'inputs': []}
            if not set(cases[case].get('core', {})) & {
                'describe_source',
                'materialize',
                'predict',
            }:
                line.pop('driver', None)
        for name, value in held.get(case, {}).items():
            *parents, last = name.split('.')
            target = line
            for part in parents:
                target = target[part]
            target[last] = value
        lines[case] = line
    if change is not None:
        change(lines)
    path = tmp_path / f'{runner}.jsonl'
    path.write_text(''.join(json.dumps(line) + '\n' for line in lines.values()))
    return path


def check(
    repo: Path,
    checkpoint: str,
    observations: tuple[Path, ...] = (),
    protected: str | None = None,
) -> list[str]:
    return ratchet.check(
        repo,
        checkpoint,
        list(observations),
        protected,
        database='duckdb' if observations else None,
    ).problems


def test_duckdb_fixture_declares_its_scope_without_changing_production_default(
    repo: Path, tmp_path: Path
) -> None:
    checkpoint = checkpoint_of(repo)
    commit(repo)
    files = (observe(repo, tmp_path, 'python-core'),)
    assert check(repo, checkpoint, files) == []
    aggregate = ratchet.check(repo, checkpoint, list(files)).problems
    for database in ('sqlite', 'postgres'):
        assert any(
            f'@{database}:' in problem and 'not observed' in problem
            for problem in aggregate
        ), aggregate


def test_one_check_parses_unchanged_reference_bytes_once_for_all_rules(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    landed = checkpoint_of(repo)
    commit(repo)
    original = ratchet.jsonl
    reads = []

    def read(text: str | None, name: str) -> list[dict[str, Any]]:
        if name.startswith('contract/reference/'):
            reads.append(name)
        return original(text, name)

    monkeypatch.setattr(ratchet, 'jsonl', read)
    assert check(repo, landed) == []
    assert reads == [
        path.relative_to(repo).as_posix()
        for path in sorted((repo / 'contract/reference').glob('*.jsonl'))
    ]


def rules(problems: list[str]) -> set[str]:
    return {problem.split(':', 1)[0] for problem in problems}


def append_entry(repo: Path, entry: dict[str, Any]) -> None:
    r"""Appends a register entry (the file ends with its entry list)."""
    path = repo / 'docs/design/rust-core/register.yaml'
    defaults = {
        'title': 'A seeded change',
        'kind': 'change',
        'old_behavior': 'Before.',
        'new_behavior': 'After.',
        'affected_inputs': 'One case.',
        'status': 'approved',
        'approved_by': 'reviewer',
        'decision_ref': None,
        'notes': 'Seeded by the ratchet tests.',
    }
    seeded = {**defaults, **entry}
    if seeded['approved_by'] == 'owner' and seeded['decision_ref'] is None:
        # An owner's entry names the row holding the owner's approval
        # (register.py, T3.2): a seed of one names an owner-approved row.
        seeded['decision_ref'] = {
            'file': 'docs/design/rust-core/phase-1/decisions.md',
            'row': 'E1',
        }
    text = yaml.safe_dump([seeded], sort_keys=False)
    path.write_text(path.read_text() + '\n' + text)


def outcome(
    case: str, runner: str, old: str, new: str, cell: str = CELL
) -> dict[str, str]:
    return {
        'case': case,
        'cell': cell,
        'runner': runner,
        'old': old,
        'new': new,
        'summary': 'seeded',
    }


def set_outcome(case: str, **fields: Any) -> Callable[[dict[str, Any]], None]:
    def change(lines: dict[str, Any]) -> None:
        lines[case]['outcome'].update(fields)

    return change


ACCEPTED = 's0.accept.sum-window'
#: A depth probe: valid JSON and YAML nested 100,000 levels.
DEEP = '[' * 100_000 + ']' * 100_000


# --- the baseline itself -------------------------------------------------------


def test_the_unseeded_candidate_passes_on_every_runner(
    repo: Path, tmp_path: Path
) -> None:
    landed = checkpoint_of(repo)
    commit(repo)
    files = tuple(
        observe(repo, tmp_path, runner)
        for runner in ('python-core', 'python-old', 'rust-host')
    )
    assert check(repo, landed, files) == []


def test_a_changed_answer_passes_only_with_an_exact_reviewer_approved_entry(
    repo: Path, tmp_path: Path
) -> None:
    landed = checkpoint_of(repo)
    old, new = 'success at materialize', 'success at predict'
    files = (
        observe(
            repo, tmp_path, 'python-old', set_outcome(ACCEPTED, stage='predict')
        ),
    )
    commit(repo)
    found = check(repo, landed, files)
    assert any(
        f'{ACCEPTED}: python-old: {old} became {new}' in p for p in found
    ), found
    # Inexact entries do not count: another runner, another cell, another
    # old outcome, an entry still proposed.
    for wrong, extra in (
        (outcome(ACCEPTED, 'rust-host', old, new), {}),
        (outcome(ACCEPTED, 'python-old', old, new, 'dbr17-pandas2'), {}),
        (outcome(ACCEPTED, 'python-old', 'success', new), {}),
        (
            outcome(ACCEPTED, 'python-old', old, new),
            {'status': 'proposed', 'approved_by': None},
        ),
    ):
        git(repo, 'reset', '-q', '--hard', landed)
        commit(
            repo,
            edit=lambda r, w=wrong, e=extra: append_entry(
                r,
                {
                    'id': 'SEED-1',
                    'cases': [ACCEPTED],
                    'case_outcomes': [w],
                    **e,
                },
            ),
        )
        assert any('became' in p for p in check(repo, landed, files)), wrong
    git(repo, 'reset', '-q', '--hard', landed)
    commit(
        repo,
        edit=lambda r: append_entry(
            r,
            {
                'id': 'SEED-1',
                'cases': [ACCEPTED],
                'case_outcomes': [outcome(ACCEPTED, 'python-old', old, new)],
            },
        ),
    )
    found = check(repo, landed, files)
    # Judged against the checkpoint's baseline, the change passes; the
    # candidate's proposed baseline must then say the new outcome, which
    # accept writes from the effective expectation.
    assert not any('became' in p for p in found), found
    assert any(
        'the proposed baseline is not the observation' in p for p in found
    )
    commit(repo, edit=lambda r: accepted(r, landed, files))
    assert check(repo, landed, files) == []


def accepted(repo: Path, checkpoint: str, files: tuple[Path, ...]) -> None:
    r"""Writes the proposed baselines as ``ratchet.py accept --checkpoint``
    does, for the runners observed."""
    for runner, lines in ratchet.accept(repo, list(files), checkpoint).items():
        path = repo / f'contract/baselines/{CELL}/{runner}.jsonl'
        path.write_text(
            ''.join(json.dumps(line, sort_keys=True) + '\n' for line in lines)
        )


def _transition(
    repo: Path, tmp_path: Path, item: dict[str, str] | None, owner: bool = False
) -> tuple[str, tuple[Path, ...]]:
    r"""A candidate whose python-old observation of ACCEPTED moves to
    predict, with ``item`` as an approved entry's one case outcome."""
    landed = checkpoint_of(repo)
    files = (
        observe(
            repo, tmp_path, 'python-old', set_outcome(ACCEPTED, stage='predict')
        ),
    )
    if item is not None:
        commit(
            repo,
            edit=lambda r: append_entry(
                r,
                {
                    'id': 'SEED-1',
                    'cases': [ACCEPTED],
                    'case_outcomes': [item],
                    'approved_by': 'owner' if owner else 'reviewer',
                },
            ),
        )
    return landed, files


def test_accept_writes_the_successor_baseline_of_an_approved_transition_only(
    repo: Path, tmp_path: Path
) -> None:
    old, new = 'success at materialize', 'success at predict'

    def held(found: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
        return next(
            (
                x['projection']
                for x in found['python-old']
                if x['case'] == ACCEPTED
            ),
            None,
        )

    landed, files = _transition(
        repo, tmp_path, outcome(ACCEPTED, 'python-old', old, new)
    )
    # Approved, exact: the successor baseline holds the new outcome.
    assert held(ratchet.accept(repo, list(files), landed)) == {
        'outcome.code': None,
        'outcome.error': None,
        'outcome.result': 'success',
        'outcome.stage': 'predict',
    }
    # Without the checkpoint (the first baselines), only reference equality.
    assert held(ratchet.accept(repo, list(files))) is None
    # A wrong or unapproved transition admits nothing.
    for item in (outcome(ACCEPTED, 'python-old', 'success', new), None):
        git(repo, 'reset', '-q', '--hard', landed)
        landed, files = _transition(repo, tmp_path, item)
        assert held(ratchet.accept(repo, list(files), landed)) is None, item


def test_the_effective_expectation_is_the_baseline_or_its_approved_transition(
    repo: Path, tmp_path: Path
) -> None:
    landed = checkpoint_of(repo)
    target = tmp_path / 'effective.json'
    refusal = 'refusal INVALID_REQUEST at materialize (ValueError)'

    def wanted(item: dict[str, str] | None, owner: bool = False) -> Any:
        git(repo, 'reset', '-q', '--hard', landed)
        _transition(repo, tmp_path, item, owner)
        ratchet.check(repo, landed, [], '', effective_out=target)
        return json.loads(target.read_text())[ACCEPTED]['python-old']

    baseline = {
        'result': 'success',
        'stage': 'materialize',
        'error': None,
        'code': None,
    }
    assert wanted(None) == baseline
    assert wanted(
        outcome(
            ACCEPTED,
            'python-old',
            'success at materialize',
            'success at predict',
        )
    ) == {
        **baseline,
        'stage': 'predict',
    }
    # Not from the baseline's outcome, another runner, a new refusal the
    # reviewer approved, a panic: the baseline stands.
    for item in (
        outcome(ACCEPTED, 'python-old', 'success', 'success at predict'),
        outcome(
            ACCEPTED,
            'python-core',
            'success at materialize',
            'success at predict',
        ),
        outcome(ACCEPTED, 'python-old', 'success at materialize', refusal),
        outcome(
            ACCEPTED,
            'python-old',
            'success at materialize',
            'refusal INTERNAL_ERROR at materialize',
        ),
    ):
        assert wanted(item) == baseline, item
    assert wanted(
        outcome(ACCEPTED, 'python-old', 'success at materialize', refusal),
        owner=True,
    ) == {
        'result': 'refusal',
        'stage': 'materialize',
        'error': 'ValueError',
        'code': 'INVALID_REQUEST',
    }


def test_a_new_refusal_fails_without_the_owner(
    repo: Path, tmp_path: Path
) -> None:
    landed = checkpoint_of(repo)
    old = 'success at materialize'
    new = 'refusal INVALID_REQUEST at materialize (ValueError)'
    files = (
        observe(
            repo,
            tmp_path,
            'python-old',
            set_outcome(
                ACCEPTED,
                result='refusal',
                code='INVALID_REQUEST',
                error='ValueError',
            ),
        ),
    )
    item = outcome(ACCEPTED, 'python-old', old, new)
    commit(
        repo,
        edit=lambda r: append_entry(
            r, {'id': 'SEED-1', 'cases': [ACCEPTED], 'case_outcomes': [item]}
        ),
    )
    found = check(repo, landed, files)
    assert any('only the owner approves it' in p for p in found), found
    assert any('with no approved owner entry' in p for p in found), found
    git(repo, 'reset', '-q', '--hard', landed)
    commit(
        repo,
        edit=lambda r: append_entry(
            r,
            {
                'id': 'SEED-1',
                'cases': [ACCEPTED],
                'case_outcomes': [item],
                'approved_by': 'owner',
            },
        ),
    )
    found = check(repo, landed, files)
    assert not any('owner' in p for p in found), found


def test_a_panic_is_never_accepted(repo: Path, tmp_path: Path) -> None:
    landed = checkpoint_of(repo)
    new = 'refusal INTERNAL_ERROR at materialize (StructuredDataError)'
    files = (
        observe(
            repo,
            tmp_path,
            'python-old',
            set_outcome(
                ACCEPTED,
                result='refusal',
                code='INTERNAL_ERROR',
                error='StructuredDataError',
            ),
        ),
    )
    item = outcome(ACCEPTED, 'python-old', 'success at materialize', new)
    commit(
        repo,
        edit=lambda r: append_entry(
            r,
            {
                'id': 'SEED-1',
                'cases': [ACCEPTED],
                'case_outcomes': [item],
                'approved_by': 'owner',
            },
        ),
    )
    found = check(repo, landed, files)
    assert any('no entry accepts INTERNAL_ERROR' in p for p in found), found
    assert any('INTERNAL_ERROR, a panic' in p for p in found), found


def test_a_removed_case_fails(repo: Path, tmp_path: Path) -> None:
    landed = checkpoint_of(repo)
    files = (observe(repo, tmp_path, 'python-old'),)
    commit(
        repo, edit=lambda r: (r / f'contract/cases/s0/{ACCEPTED}.json').unlink()
    )
    found = check(repo, landed, files)
    assert any(
        f'cases: {ACCEPTED}: removed, and no approved entry states it on python-old'
        in p
        for p in found
    ), found


def test_a_demotion_to_diagnostic_fails(repo: Path, tmp_path: Path) -> None:
    landed = checkpoint_of(repo)

    def demote(r: Path) -> None:
        path = r / f'contract/cases/s0/{ACCEPTED}.json'
        path.write_text(
            path.read_text().replace('"acceptance"', '"diagnostic"', 1)
        )

    commit(repo, edit=demote)
    found = check(repo, landed)
    assert any(f'cases: {ACCEPTED}: diagnostic' in p for p in found), found
    # With an approved entry for each baselined runner, the demotion passes.
    git(repo, 'reset', '-q', '--hard', landed)
    items = [
        outcome(ACCEPTED, runner, old, 'diagnostic')
        for runner, old in (
            ('python-core', 'success at materialize'),
            ('python-old', 'success at materialize'),
            ('rust-host', 'success'),
        )
    ]

    def both(r: Path) -> None:
        demote(r)
        append_entry(
            r, {'id': 'SEED-1', 'cases': [ACCEPTED], 'case_outcomes': items}
        )

    commit(repo, edit=both)
    assert 'cases' not in rules(check(repo, landed))


def test_a_changed_case_input_fails_without_an_entry(repo: Path) -> None:
    landed = checkpoint_of(repo)

    def corrupt(r: Path) -> None:
        path = r / f'contract/cases/s0/{ACCEPTED}.json'
        case = json.loads(path.read_text())
        case['expect']['outcome'] = 'refusal'
        path.write_text(json.dumps(case, indent=1))

    commit(repo, edit=corrupt)
    found = check(repo, landed)
    assert (
        f'cases: {ACCEPTED}: its input changed, and no approved entry names it'
        in found
    )


def test_a_base_changes_only_under_an_entry_and_may_be_added(
    repo: Path,
) -> None:
    r"""A base added passes; a base changed passes only under an approved
    entry the change itself adds (or changes) naming a case built on it.
    From T1.29 the checkpoint's register holds an approved entry naming
    cases built on ``shop`` (P1B-26, the baseline cases' admission): it
    authorized its own change only, so the change below still fails."""
    landed = checkpoint_of(repo)

    def grow(r: Path) -> None:
        path = r / 'contract/bases.yaml'
        bases = yaml.safe_load(path.read_text())
        bases['bases']['seeded'] = ['CREATE TABLE t AS SELECT 1 AS x']
        path.write_text(yaml.safe_dump(bases))

    commit(repo, edit=grow)
    assert 'cases' not in rules(check(repo, landed))
    git(repo, 'reset', '-q', '--hard', landed)

    def change(r: Path) -> None:
        path = r / 'contract/bases.yaml'
        bases = yaml.safe_load(path.read_text())
        bases['bases']['shop'] = bases['bases']['shop'][:1]
        path.write_text(yaml.safe_dump(bases))

    commit(repo, edit=change)
    assert any(
        'base shop: changed or removed' in p for p in check(repo, landed)
    )
    # The same change with its own entry, naming a case built on the base.
    git(repo, 'reset', '-q', '--hard', landed)

    def change_and_entry(r: Path) -> None:
        change(r)
        append_entry(
            r,
            {
                'id': 'SEED-1',
                'cases': [ACCEPTED],
                'case_outcomes': [
                    outcome(
                        ACCEPTED,
                        'python-old',
                        'success at materialize',
                        'success at materialize',
                    )
                ],
            },
        )

    commit(repo, edit=change_and_entry)
    assert 'cases' not in rules(check(repo, landed))


# --- rules and protected paths -------------------------------------------------


def edit_yaml(
    path: str, change: Callable[[dict[str, Any]], None]
) -> Callable[[Path], None]:
    def edit(repo: Path) -> None:
        document = yaml.safe_load((repo / path).read_text())
        change(document)
        (repo / path).write_text(yaml.safe_dump(document, sort_keys=False))

    return edit


def owner_row(repo: Path, row: str, names: str, status: str) -> None:
    path = repo / 'docs/design/rust-core/phase-1/decisions.md'
    path.write_text(
        path.read_text().rstrip('\n')
        + f'\n| {row} | A seeded relaxation of `{names}` | Seeded | {status} |\n'
    )


def test_a_relaxed_rule_fails_without_an_owner_approved_row_at_the_checkpoint(
    repo: Path,
) -> None:
    policy = 'contract/compare-policy.yaml'
    change = 'rule sql.result: gate to diagnostic'

    def relax(document: dict[str, Any], decision: str | None = None) -> None:
        document['rules']['sql.result'] = 'diagnostic'
        if decision:
            document['relaxations'] = [{'change': change, 'decision': decision}]

    landed = checkpoint_of(repo)
    commit(repo, edit=edit_yaml(policy, relax))
    assert f'policy: {change}: a relaxation' in ' '.join(check(repo, landed))
    # A row the package writes itself never authorizes it.
    git(repo, 'reset', '-q', '--hard', landed)
    commit(
        repo,
        edit=lambda r: (
            owner_row(
                r, 'E900', 'compare-policy.yaml', 'approved by owner 2026-10-11'
            ),
            edit_yaml(policy, lambda d: relax(d, 'E900'))(r),
        ),
    )
    assert 'policy' in rules(check(repo, landed))
    # The row at the checkpoint, approved by the owner and naming the file,
    # does; one the owner did not approve, or naming another file, does not.
    for status, names, passes in (
        ('approved by owner 2026-10-11', 'compare-policy.yaml', True),
        ('owner to confirm', 'compare-policy.yaml', False),
        ('approved by owner 2026-10-11', 'availability.yaml', False),
    ):
        git(repo, 'reset', '-q', '--hard', landed)
        checkpoint = land(
            repo,
            'T1.15',
            lambda r, s=status, n=names: owner_row(r, 'E900', n, s),
        )
        commit(
            repo,
            'T1.16: relaxed',
            edit_yaml(policy, lambda d: relax(d, 'E900')),
        )
        assert ('policy' not in rules(check(repo, checkpoint))) is passes, (
            status
        )


@pytest.mark.parametrize(
    'relaxation',
    [
        'rule removed',
        'exclusion added',
        'exclusion widened',
        'ordered statement dropped',
        'reach dropped',
        'baseline field dropped',
    ],
)
def test_each_relaxation_of_the_policy_fails(
    repo: Path, relaxation: str
) -> None:
    def relax(document: dict[str, Any]) -> None:
        if relaxation == 'rule removed':
            del document['rules']['sql.count']
        elif relaxation == 'exclusion added':
            document['exclusions'].append(
                {**document['exclusions'][0], 'id': 'more'}
            )
        elif relaxation == 'exclusion widened':
            document['exclusions'][0]['path'] = ['metadata']
        elif relaxation == 'ordered statement dropped':
            document['ordered'] = {}
        elif relaxation == 'reach dropped':
            document['reach'] = {}
        else:
            document['baseline'].remove('outcome.code')

    landed = checkpoint_of(repo)
    commit(repo, edit=edit_yaml('contract/compare-policy.yaml', relax))
    assert 'policy' in rules(check(repo, landed))


def test_a_tightened_rule_passes(repo: Path) -> None:
    def tighten(document: dict[str, Any]) -> None:
        document['rules']['sql.text'] = 'gate'
        document['exclusions'] = []
        document['ordered']['base.regression']['from_source'] = [0]

    landed = checkpoint_of(repo)
    commit(repo, edit=edit_yaml('contract/compare-policy.yaml', tighten))
    assert 'policy' not in rules(check(repo, landed))


@pytest.mark.parametrize(
    'relaxation', ['downgraded', 'dropped', 'added as unavailable']
)
def test_an_availability_relaxation_fails(repo: Path, relaxation: str) -> None:
    def relax(document: dict[str, Any]) -> None:
        runner = document['runners']['python-core']
        if relaxation == 'downgraded':
            runner['core'] = 'unavailable'
        elif relaxation == 'dropped':
            del runner['public']
        else:
            runner['driver.query_log'] = 'unavailable'

    landed = checkpoint_of(repo)
    commit(repo, edit=edit_yaml('contract/availability.yaml', relax))
    assert 'availability' in rules(check(repo, landed))


def test_an_upgraded_availability_passes(repo: Path) -> None:
    def tighten(document: dict[str, Any]) -> None:
        document['runners']['python-core']['driver.query_log'] = 'available'

    landed = checkpoint_of(repo)
    commit(repo, edit=edit_yaml('contract/availability.yaml', tighten))
    assert 'availability' not in rules(check(repo, landed))


def test_an_observation_missing_an_available_field_or_recording_an_unavailable_one_fails(
    repo: Path, tmp_path: Path
) -> None:
    landed = checkpoint_of(repo)
    commit(repo)

    def drop(lines: dict[str, Any]) -> None:
        lines[ACCEPTED]['host'] = None

    def fabricate(lines: dict[str, Any]) -> None:
        lines[ACCEPTED]['outcome']['stage'] = 'materialize'

    def retype(lines: dict[str, Any]) -> None:
        del lines[ACCEPTED]['public']

    for runner, change, wanted in (
        ('python-old', drop, 'host available, and missing'),
        ('python-old', retype, 'public available, and missing'),
        (
            'rust-host',
            fabricate,
            'outcome.stage recorded, and it is unavailable',
        ),
    ):
        found = check(repo, landed, (observe(repo, tmp_path, runner, change),))
        assert any(
            f'availability: {ACCEPTED}: {runner}: {wanted}' in p for p in found
        ), found


def test_a_protected_change_fails_unless_the_landing_s_trailers_name_it(
    repo: Path,
) -> None:
    landed = checkpoint_of(repo)
    path = 'contract/runners/capture_check.py'

    def touch(r: Path) -> None:
        with (r / path).open('a') as handle:
            handle.write('\n# touched\n')

    commit(repo, edit=touch)
    # Before landing, with no text given, the paths are listed, not failed.
    report = ratchet.check(repo, landed, [], None)
    assert (
        f'protected: changed, for the landing to name: {path}' in report.notes
    )
    assert check(repo, landed, protected='docs only') == [
        f'protected: {path}: changed, and the Reviewed-by and Protected-paths '
        'trailers do not name it'
    ]
    assert check(repo, landed, protected=f'{path} (a comment)') == []
    assert (
        check(repo, landed, protected='contract/runners/** (the runners)') == []
    )
    # On the landing commit the trailers are read, whatever is given.
    git(repo, 'reset', '-q', '--hard', landed)
    land(repo, 'T1.15', touch, protected='nothing protected')
    assert any(path in p for p in check(repo, landed, protected=path))
    git(repo, 'reset', '-q', '--hard', landed)
    land(repo, 'T1.15', touch, protected=f'{path} (touched)')
    assert check(repo, landed) == []


def test_a_landing_without_a_reviewed_by_trailer_fails(repo: Path) -> None:
    landed = checkpoint_of(repo)
    commit(repo, 'T1.15: the package')
    commit(
        repo,
        'Land T1.15: the package',
        trailers={
            'Protected-paths': 'none',
            'Base-SHA': landed,
            'Checkpoint-SHA': landed,
        },
    )
    assert check(repo, landed) == [
        'protected: the landing commit has no Reviewed-by trailer'
    ]


def test_a_protected_file_renamed_out_of_protection_or_onto_it_fails(
    repo: Path,
) -> None:
    landed = checkpoint_of(repo)
    protected = 'contract/tests/test_runners.py'
    moved = 'tests/test_runners_moved.py'

    def out(r: Path) -> None:
        (r / 'tests').mkdir(exist_ok=True)
        git(r, 'mv', protected, moved)
        with (r / moved).open('a') as handle:
            handle.write('\n# changed in the move\n')

    commit(repo, edit=out)
    assert check(repo, landed, protected='') == [
        f'protected: {protected}: changed, and the Reviewed-by and '
        'Protected-paths trailers do not name it'
    ]
    git(repo, 'reset', '-q', '--hard', landed)
    seam = 'packages/structured-data-models-client/src/sdm_client/_routing.py'
    onto = 'contract/runners/routing.py'
    commit(repo, edit=lambda r: git(r, 'mv', seam, onto))
    assert check(repo, landed, protected='') == [
        f'protected: {onto}: changed, and the Reviewed-by and '
        'Protected-paths trailers do not name it'
    ]


def test_a_protected_file_with_a_name_past_ascii_fails_unless_named(
    repo: Path,
) -> None:
    r"""Git quotes a path past ASCII under its default ``core.quotePath``,
    so read line by line a new protected file's name never matched the
    list; the ratchet reads git's paths NUL-separated (T1.18)."""
    landed = checkpoint_of(repo)
    added = 'contract/runners/na\u00efve_r\u00e9sum\u00e9.py'
    git(repo, 'config', 'core.quotePath', 'true')

    def write(r: Path) -> None:
        (r / added).write_text('# a new runner file\n', encoding='utf-8')

    commit(repo, edit=write)
    assert check(repo, landed, protected='') == [
        f'protected: {added}: changed, and the Reviewed-by and '
        'Protected-paths trailers do not name it'
    ]
    assert check(repo, landed, protected=f'{added} (new)') == []
    tree = ratchet.Tree(repo, git(repo, 'rev-parse', 'HEAD'))
    assert added in tree.files('contract/runners')


def test_the_protected_list_holds_the_corpus_test_and_the_stream_vectors() -> (
    None
):
    rules = ratchet.patterns(
        (ROOT / 'contract/protected-paths.txt').read_text()
    )
    for path in (
        'rust/crates/sdm-core/tests/pql_corpus.rs',
        'rust/crates/sdm-core/tests/pql/known-differing.json',
        'rust/crates/sdm-engine/tests/rs1.json',
        'rust/crates/sdm-engine/tests/rs1-draws.json',
        'rust/crates/sdm-engine/tests/rs1.rs',
        'rust/crates/sdm-engine/tests/rs1_record.py',
        'contract/request-kinds.json',
    ):
        assert ratchet.protects(rules, path), path


def test_a_change_to_the_routing_seam_is_not_protected(repo: Path) -> None:
    landed = checkpoint_of(repo)
    seam = 'packages/structured-data-models-client/src/sdm_client/_routing.py'
    product = 'packages/structured-data-models-client/src/sdm_client/query.py'

    def touch(r: Path) -> None:
        for name in (seam, product):
            with (r / name).open('a') as handle:
                handle.write('\n# touched\n')

    commit(repo, edit=touch)
    assert check(repo, landed, protected='') == [
        f'protected: {product}: changed, and the Reviewed-by and '
        'Protected-paths trailers do not name it'
    ]


def test_a_protected_paths_list_missing_at_the_checkpoint_fails(
    repo: Path,
) -> None:
    checkpoint = land(
        repo, 'T1.15', lambda r: (r / 'contract/protected-paths.txt').unlink()
    )
    commit(repo)
    assert (
        'protected: no contract/protected-paths.txt at the checkpoint'
        in check(repo, checkpoint, protected='')
    )


# --- what passes -----------------------------------------------------------------


def test_a_new_case_meeting_its_expectation_passes_without_a_checkpoint_baseline(
    repo: Path, tmp_path: Path
) -> None:
    landed = checkpoint_of(repo)
    new = 's0.accept.sum-window-again'

    def add(r: Path) -> None:
        case = json.loads(
            (r / f'contract/cases/s0/{ACCEPTED}.json').read_text()
        )
        case['id'] = new
        (r / f'contract/cases/s0/{new}.json').write_text(
            json.dumps(case, indent=1)
        )

    def observed(lines: dict[str, Any]) -> None:
        lines[new] = copy.deepcopy(lines[ACCEPTED])
        lines[new]['case'] = new

    commit(repo, edit=add)
    files = (observe(repo, tmp_path, 'python-old', observed),)
    # Admitted, it needs the proposed baseline ratchet.py accept writes.
    assert check(repo, landed, files) == [
        f'baselines: {new}: python-old: an acceptance case with no proposed baseline'
    ]

    def accepted(r: Path) -> None:
        path = r / f'contract/baselines/{CELL}/python-old.jsonl'
        line = json.loads(path.read_text().splitlines()[0])
        projection = ratchet.projection(observed_line(r, new), line_fields(r))
        line.update(
            case=new, projection=projection, digest=ratchet.digest(projection)
        )
        path.write_text(path.read_text() + json.dumps(line) + '\n')

    def observed_line(r: Path, case: str) -> dict[str, Any]:
        return {
            json.loads(x)['case']: json.loads(x)
            for x in files[0].read_text().splitlines()
        }[case]

    def line_fields(r: Path) -> list[str]:
        return yaml.safe_load((r / 'contract/compare-policy.yaml').read_text())[
            'baseline'
        ]

    commit(repo, edit=accepted)
    assert check(repo, landed, files) == []


def test_a_branch_side_baseline_edit_has_no_effect(
    repo: Path, tmp_path: Path
) -> None:
    landed = checkpoint_of(repo)
    files = (
        observe(
            repo, tmp_path, 'python-old', set_outcome(ACCEPTED, stage='predict')
        ),
    )

    def follow(r: Path) -> None:
        path = r / f'contract/baselines/{CELL}/python-old.jsonl'
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        for line in lines:
            if line['case'] == ACCEPTED:
                line['projection']['outcome.stage'] = 'predict'
                line['digest'] = ratchet.digest(line['projection'])
        path.write_text(''.join(json.dumps(x) + '\n' for x in lines))

    commit(repo, edit=follow)
    found = check(
        repo,
        landed,
        files,
        protected=f'contract/baselines/{CELL}/python-old.jsonl',
    )
    assert found == [
        f'baselines: {ACCEPTED}: python-old: success at materialize became '
        f'success at predict, with no approved entry stating it: '
        + ratchet.proposal(
            ACCEPTED,
            'python-old',
            'success at materialize',
            'success at predict',
        )
    ]


def test_an_identity_only_change_passes(repo: Path, tmp_path: Path) -> None:
    landed = checkpoint_of(repo)

    def identity(lines: dict[str, Any]) -> None:
        for line in lines.values():
            line['identity']['commit'] = {'sha': '0' * 40, 'clean': False}
            line['identity']['platform'] = 'linux-x86_64'
            line['identity']['versions']['pandas'] = '2.2.3'

    def rerecord(r: Path) -> None:
        path = next((r / 'contract/reference').glob('*.jsonl'))
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        for line in lines:
            line['identity']['commit']['sha'] = '1' * 40
        path.write_text(''.join(json.dumps(x) + '\n' for x in lines))

    commit(repo, edit=rerecord)
    files = (observe(repo, tmp_path, 'python-old', identity),)
    report = ratchet.check(repo, landed, list(files), 'contract/reference/**')
    assert report.problems == []
    assert any('re-recorded, identity only' in note for note in report.notes)


# --- ratchet integrity -------------------------------------------------------------


def _pinned(repo: Path) -> str:
    r"""A landing whose ``KNOWN_DIFFERING`` pins one case, the checkpoint
    the known-differing tests start from (the file is empty from T1.6b)."""
    path = 'rust/crates/sdm-core/tests/pql/known-differing.json'

    def pin(r: Path) -> None:
        document = json.loads((r / path).read_text())
        document['cases'] = [
            {'id': 'literal/pinned', 'parts': ['warnings'], 'why': 'seeded'}
        ]
        (r / path).write_text(json.dumps(document, indent=1))

    return land(repo, 'T1.14', edit=pin, protected=path)


def test_an_id_added_to_known_differing_fails(repo: Path) -> None:
    path = 'rust/crates/sdm-core/tests/pql/known-differing.json'
    landed = _pinned(repo)

    def add(part: bool) -> Callable[[Path], None]:
        def edit(r: Path) -> None:
            document = json.loads((r / path).read_text())
            if part:
                document['cases'][0]['parts'].append('tree')
            else:
                document['cases'].append(
                    {'id': 'literal/new', 'parts': ['tree'], 'why': 'seeded'}
                )
            (r / path).write_text(json.dumps(document, indent=1))

        return edit

    commit(repo, edit=add(False))
    assert 'known-differing: literal/new: added' in check(
        repo, landed, protected=path
    )
    git(repo, 'reset', '-q', '--hard', landed)
    commit(repo, edit=add(True))
    assert any('parts added' in p for p in check(repo, landed, protected=path))
    git(repo, 'reset', '-q', '--hard', landed)

    def shrink(r: Path) -> None:
        document = json.loads((r / path).read_text())
        document['cases'] = []
        (r / path).write_text(json.dumps(document, indent=1))

    commit(repo, edit=shrink)
    assert check(repo, landed, protected=path) == []


@pytest.mark.parametrize(
    ('seed', 'wanted'),
    [
        (
            'missing',
            'no rust/crates/sdm-core/tests/pql/known-differing.json at the candidate',
        ),
        ('no cases key', 'is not a list of cases'),
        ('cases retyped', 'is not a list of cases'),
        ('parts retyped', 'is not a list of cases'),
    ],
)
def test_a_known_differing_file_missing_or_reshaped_fails(
    repo: Path, seed: str, wanted: str
) -> None:
    path = 'rust/crates/sdm-core/tests/pql/known-differing.json'
    landed = _pinned(repo)

    def reshape(r: Path) -> None:
        if seed == 'missing':
            (r / path).unlink()
            return
        document = json.loads((r / path).read_text())
        if seed == 'no cases key':
            del document['cases']
        elif seed == 'cases retyped':
            document['cases'] = {}
        else:
            document['cases'][0]['parts'] = 'warnings'
        (r / path).write_text(json.dumps(document))

    commit(repo, edit=reshape)
    found = check(repo, landed, protected=path)
    assert any(wanted in p for p in found), found


def resolve(repo: Path, **where: str) -> tuple[Any, list[str]]:
    return ratchet.resolve(repo, where.get('base'), where.get('phase_push'))


def test_self_comparison_is_rejected(repo: Path) -> None:
    landed = checkpoint_of(repo)
    candidate = commit(repo)
    found, problems = resolve(repo, base=candidate)
    assert found is None
    assert any(p.startswith('self-comparison') for p in problems)
    found, problems = resolve(repo, base=landed)
    assert problems == [] and found.sha == landed


def test_advancing_the_phase_ref_cannot_erase_a_changed_outcome_or_a_protected_edit(
    repo: Path, tmp_path: Path
) -> None:
    landed = checkpoint_of(repo)
    files = (
        observe(
            repo, tmp_path, 'python-old', set_outcome(ACCEPTED, stage='predict')
        ),
    )

    def touch(r: Path) -> None:
        with (r / 'contract/runners/capture_check.py').open('a') as handle:
            handle.write('\n# touched\n')

    bad = commit(repo, 'T1.15: an unreviewed change', touch)
    # Judged against the base recorded at package start, both fail.
    found = check(repo, landed, files, protected='')
    assert rules(found) == {'baselines', 'protected'}, found
    # The phase branch advanced to the change itself: the base holds the
    # candidate, which would be its own checkpoint.
    assert resolve(repo, base=bad)[0] is None
    # The next package, built on the unreviewed commit, cannot take it for
    # a checkpoint: it is no landing.
    commit(repo, 'T1.16: the next package')
    found, problems = resolve(repo, base=bad)
    assert found is None and any('not a landing commit' in p for p in problems)
    # Nor can a phase push carrying it land.
    found, problems = resolve(repo, phase_push=landed)
    assert found is None and any('its tip is none' in p for p in problems)


def test_a_rebased_package_is_judged_against_the_new_tip(repo: Path) -> None:
    landed = checkpoint_of(repo)
    commit(repo, 'T1.16: the package, on the old tip')
    old = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'checkout', '-q', '--detach', landed)
    other = land(repo, 'T1.15')
    # Not rebased: judged at the old merge base, and told to rebase.
    git(repo, 'checkout', '-q', '--detach', old)
    found, problems = resolve(repo, base=other)
    assert problems == [] and found.sha == landed
    assert any('rebase onto it' in note for note in found.notes)
    # Rebased onto the other landing: judged against the new tip.
    git(repo, 'rebase', '-q', other)
    found, problems = resolve(repo, base=other)
    assert problems == [] and found.sha == other and found.notes == []


def test_a_push_that_lands_a_stack_is_judged_against_the_landing_below(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # T3.0: a push may land a stack of up to SDM_STACK_MAX packages, each
    # landing on the one before it; the top is judged against its own
    # Base-SHA, the landing below it.
    monkeypatch.delenv('SDM_STACK_MAX', raising=False)
    landed = checkpoint_of(repo)
    first = land(repo, 'T1.15')
    found, problems = resolve(repo, phase_push=landed)
    assert problems == [] and found.sha == landed
    land(repo, 'T1.16')
    found, problems = resolve(repo, phase_push=landed)
    assert problems == [] and found.sha == first
    monkeypatch.setenv('SDM_STACK_MAX', '1')
    found, problems = resolve(repo, phase_push=landed)
    assert found is None
    assert problems == [
        'a push to the phase branch lands at most 1 packages (a stack, '
        'T3.0): it holds 2 landing commits'
    ]
    # One landing whose recorded base is not the previous tip fails too.
    found, problems = resolve(repo, phase_push=first)
    assert problems == [] and found.sha == first
    found, problems = resolve(repo, phase_push=landed + '~1')
    assert found is None


@pytest.mark.parametrize(
    ('status', 'approved'),
    [
        ('approved by owner 2026-10-12', True),
        ('approved by owner 2026-10-12 (T1.7b check-in)', True),
        (
            'approved by owner 2026-10-12 (T1.7b check-in), no longer marked pending',
            True,
        ),
        ('approved by owner 2026-10-12 (declined)', False),
        ('approved by owner 2026-10-12, pending', False),
        (
            'approved by owner 2026-10-12 (proposed 2026-10-13, not yet answered)',
            False,
        ),
    ],
)
def test_u4_an_approval_its_own_qualifier_takes_back_is_none(
    status: str, approved: bool
) -> None:
    # U4 (carried into Phase 3): the qualifier right after the owner's date
    # decides; a declined, pending or unanswered one is no approval.
    assert ratchet.owner_approved(status) is approved


def test_a_package_above_an_unlanded_train_is_judged_by_the_train(
    repo: Path,
) -> None:
    # T3.0 (Codex r1 P2): before a stack lands, its top's run is pushed
    # while the phase branch still holds H. The base recorded at push time
    # is the stack's tail, the train's landing L, so the top is judged with
    # the train's harness, each landing's checkpoint the one below it.
    head = checkpoint_of(repo)
    train = land(repo, 'T1.15')
    candidate = commit(repo, 'T1.16: a case the train reads')
    git(
        repo, 'push', '-q', '.', f'{head}:refs/remotes/origin/rust-core-phase-1'
    )
    env = {
        'GITHUB_REF_NAME': 'rust-core-phase-1-lane-a',
        'GITHUB_EVENT_NAME': 'push',
    }
    assert ratchet.stack_tail(repo, head, candidate) == train
    assert ratchet.push_base(env, repo) == train
    found, problems = resolve(repo, base=train)
    assert problems == [] and found.sha == train
    # A landing that does not stack on the one below it is no tail.
    git(repo, 'checkout', '-q', '--detach', head)
    commit(repo, 'T1.15: the train again')
    stray = commit(
        repo,
        'Land T1.15: the train',
        trailers={
            'Reviewed-by': 'r',
            'Protected-paths': 'none',
            'Base-SHA': candidate,
            'Checkpoint-SHA': candidate,
        },
    )
    above = commit(repo, 'T1.16: above it')
    assert ratchet.stack_tail(repo, head, above) == head
    assert stray != head


def test_a_landing_commit_is_judged_against_its_recorded_base(
    repo: Path,
) -> None:
    landed = checkpoint_of(repo)
    tip = land(repo, 'T1.15')
    # In CI on the lane after landing, the phase tip is the landing itself.
    found, problems = resolve(repo, base=tip)
    assert problems == [] and found.sha == landed


# --- CI4: a run's checkpoint while the phase branch moves (E131) ---------------
#
# The independent test author's cases. A run reads the phase branch's tip
# when each job starts; a stacked package landing mid-run moves it. Every
# job of a run must reach the same checkpoint, so the candidate's checkpoint
# is the same before and after that landing, and nothing an unlanded commit
# on the phase branch reaches becomes a checkpoint. A phase push and a
# landing commit's recorded Base-SHA are read from commits, not the live
# branch, and resolve as before E131. Each is held by ``ci4_gate``.


def landing_on(
    repo: Path,
    at: str,
    package: str,
    base: str,
    subject: str = 'the package',
    moved_from: str | None = None,
) -> str:
    r"""An empty landing commit on ``at``, as land.py makes it, recording
    ``base``; with CI3's ``Moved-from`` when the package was moved."""
    git(repo, 'checkout', '-q', '--detach', at)
    found = {
        'Reviewed-by': f'independent reviewer and Codex, {package}',
        'Protected-paths': 'none',
        'Base-SHA': base,
        'Checkpoint-SHA': base,
    }
    if moved_from is not None:
        found['Moved-from'] = moved_from
    return commit(repo, f'Land {package}: {subject}', trailers=found)


def work(repo: Path, at: str, subject: str, name: str) -> str:
    r"""A package's commit on ``at``, adding an unprotected file."""
    git(repo, 'checkout', '-q', '--detach', at)
    return commit(repo, subject, lambda r: (r / f'{name}.txt').write_text(name))


def moved_copy(repo: Path, at: str, sha: str) -> str:
    r"""``sha`` cherry-picked onto ``at`` (CI3's move)."""
    git(repo, 'checkout', '-q', '--detach', at)
    git(repo, 'cherry-pick', '--allow-empty', sha)
    return git(repo, 'rev-parse', 'HEAD')


def checkpoint_at(repo: Path, candidate: str, **where: str) -> str:
    r"""The candidate's checkpoint against ``base`` or ``phase_push``,
    which must resolve."""
    git(repo, 'checkout', '-q', '--detach', candidate)
    found, problems = resolve(repo, **where)
    assert problems == [] and found is not None, (where, problems)
    return str(found.sha)


def refused_at(repo: Path, candidate: str, **where: str) -> list[str]:
    git(repo, 'checkout', '-q', '--detach', candidate)
    found, problems = resolve(repo, **where)
    assert found is None and problems, (where, found)
    return list(problems)


def ci4(request: pytest.FixtureRequest, check: Callable[[], None]) -> None:
    CI4.held('ratchet', f'py:{request.node.name}', check)


def test_ci4_a_stacked_candidate_keeps_its_checkpoint_when_its_package_lands(
    repo: Path, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        l0 = checkpoint_of(repo)
        a1 = work(repo, l0, 'T1.15: the package', 'a1')
        a2 = work(repo, a1, 'T1.15 fix 1: its review', 'a2')
        b1 = work(repo, a2, 'T1.16: stacked on T1.15', 'b1')
        # Before: the phase branch is still at the older landing.
        assert checkpoint_at(repo, b1, base=l0) == l0
        # T1.15 lands on its own commits while b1's run goes on.
        la = landing_on(repo, a2, 'T1.15', l0)
        assert checkpoint_at(repo, b1, base=la) == l0
        # And another package lands on top of it before the last job starts.
        c1 = work(repo, la, 'T1.17: another lane', 'c1')
        lc = landing_on(repo, c1, 'T1.17', la)
        assert checkpoint_at(repo, b1, base=lc) == l0
        # A candidate stacked on the package's first commit only, too.
        b2 = work(repo, a1, 'T1.18: stacked on T1.15 part way', 'b2')
        assert checkpoint_at(repo, b2, base=l0) == l0
        assert checkpoint_at(repo, b2, base=la) == l0

    ci4(request, check)


def test_ci4_a_landing_on_an_unlanded_phase_commit_is_still_refused(
    repo: Path, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        l0 = checkpoint_of(repo)
        # A commit pushed to the phase branch by hand, never landed.
        u = work(repo, l0, 'T1.15: pushed to the phase branch unlanded', 'u')
        on_u = work(repo, u, 'T1.16: built on it', 'on_u')
        p = work(repo, u, 'T1.17: a package on it', 'p')
        stacked = work(repo, p, 'T1.18: stacked on T1.17', 'stacked')
        for candidate in (on_u, stacked):
            problems = refused_at(repo, candidate, base=u)
            assert any('not a landing commit' in x for x in problems)
        # T1.17 lands with the unlanded commit for its Base-SHA.
        lp = landing_on(repo, p, 'T1.17', u)
        for candidate in (on_u, stacked):
            problems = refused_at(repo, candidate, base=lp)
            assert any('not a landing commit' in x for x in problems), (
                candidate,
                problems,
            )

    ci4(request, check)


def test_ci4_a_stack_landed_as_moved_copies_resolves_as_before(
    repo: Path, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        l0 = checkpoint_of(repo)
        a1 = work(repo, l0, 'T1.15: the package', 'a1')
        b1 = work(repo, a1, 'T1.16: stacked on T1.15', 'b1')
        # Another lane lands first, so T1.15 is moved over it (CI3).
        x1 = work(repo, l0, 'T1.17: another lane', 'x1')
        l1 = landing_on(repo, x1, 'T1.17', l0)
        a1m = moved_copy(repo, l1, a1)
        b1m = moved_copy(repo, a1m, b1)
        # Before the moved copies land.
        on_old = checkpoint_at(repo, b1, base=l1)
        on_copy = checkpoint_at(repo, b1m, base=l1)
        assert (on_old, on_copy) == (l0, l1)
        la = landing_on(repo, a1m, 'T1.15', l1, moved_from=a1)
        # After: the same checkpoints, and a candidate on the original
        # commits is still told to rebase.
        assert checkpoint_at(repo, b1, base=la) == l0
        git(repo, 'checkout', '-q', '--detach', b1)
        found, _ = resolve(repo, base=la)
        assert any('rebase onto it' in note for note in found.notes)
        assert checkpoint_at(repo, b1m, base=la) == l1
        # T1.16 moved over the landing (CI3's chain) is judged on it.
        b1mm = moved_copy(repo, la, b1m)
        git(repo, 'checkout', '-q', '--detach', b1mm)
        found, problems = resolve(repo, base=la)
        assert problems == [] and found.sha == la and found.notes == []

    ci4(request, check)


def test_ci4_a_landing_on_the_line_the_phase_branch_lacks_is_passed_over(
    repo: Path, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        l0 = checkpoint_of(repo)
        a1 = work(repo, l0, 'T1.15: the package', 'a1')
        # A landing commit the lane keeps but the phase branch never took.
        kept = landing_on(repo, a1, 'T1.15', l0, 'the package, not pushed')
        b1 = work(repo, kept, 'T1.16: stacked on that landing', 'b1')
        assert checkpoint_at(repo, b1, base=l0) == l0
        # The phase branch takes another landing of the same commits.
        la = landing_on(repo, a1, 'T1.15', l0, 'the package, landed')
        assert la != kept
        assert checkpoint_at(repo, b1, base=la) == l0

    ci4(request, check)


def test_ci4_a_phase_push_and_a_recorded_base_resolve_as_before(
    repo: Path, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        l0 = checkpoint_of(repo)
        a1 = work(repo, l0, 'T1.15: the package', 'a1')
        la = landing_on(repo, a1, 'T1.15', l0)
        b1 = work(repo, a1, 'T1.16: stacked on T1.15', 'b1')
        # A landing recording a base it does not descend from: its
        # recorded base is not the live branch, so no later landing
        # stands in for the merge base, which is no landing commit.
        bad = landing_on(repo, b1, 'T1.16', la)
        problems = refused_at(repo, bad, base=la)
        assert any('not a landing commit' in x for x in problems), problems
        assert refused_at(repo, bad, phase_push=la) == [
            'the push is not a fast-forward of the previous tip'
        ]
        # The same package moved onto the landing resolves on it, both ways.
        b1m = moved_copy(repo, la, b1)
        lb = landing_on(repo, b1m, 'T1.16', la, moved_from=b1)
        assert checkpoint_at(repo, lb, phase_push=la) == la
        c1 = work(repo, lb, 'T1.17: the next package', 'c1')
        lc = landing_on(repo, c1, 'T1.17', lb)
        assert checkpoint_at(repo, lb, base=lc) == la
        assert checkpoint_at(repo, lb, base=l0) == la
        # A phase push from a tip that is no landing stays refused.
        u = work(repo, la, 'T1.18: pushed unlanded', 'u')
        lu = landing_on(repo, u, 'T1.18', u)
        problems = refused_at(repo, lu, phase_push=u)
        assert any('not a landing commit' in x for x in problems), problems

    ci4(request, check)


# --- reference immutability ---------------------------------------------------------


def edit_reference(
    change: Callable[[list[dict[str, Any]]], None],
) -> Callable[[Path], None]:
    def edit(repo: Path) -> None:
        path = next((repo / 'contract/reference').glob('*.jsonl'))
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        change(lines)
        path.write_text(''.join(json.dumps(x) + '\n' for x in lines))

    return edit


@pytest.mark.parametrize(
    ('seed', 'wanted'),
    [
        ('outcome changed', 'its line changed'),
        ('a result row changed', 'its line changed'),
        ('a payload field added', 'its line changed'),
        ('removed', 'its line removed'),
    ],
)
def test_a_reference_line_is_never_changed_or_removed(
    repo: Path, seed: str, wanted: str
) -> None:
    landed = checkpoint_of(repo)

    def change(lines: list[dict[str, Any]]) -> None:
        line = next(x for x in lines if x['case'] == 'base.regression')
        if seed == 'outcome changed':
            line['outcome']['stage'] = 'materialize'
        elif seed == 'a result row changed':
            line['public']['result']['rows'] += 1
        elif seed == 'a payload field added':
            parsed = line['public']['payloads'][0]['parsed']
            parsed['seeded'] = {'number': '1'}
        else:
            lines.remove(line)

    commit(repo, edit=edit_reference(change))
    found = check(repo, landed, protected='contract/reference/**')
    assert any(f'reference: base.regression: {wanted}' in p for p in found), (
        found
    )


def test_a_reference_line_may_be_added_and_changes_under_an_approved_entry(
    repo: Path,
) -> None:
    landed = checkpoint_of(repo)

    def add(lines: list[dict[str, Any]]) -> None:
        line = copy.deepcopy(lines[0])
        line['case'] = 'a.new-case'
        lines.append(line)

    commit(repo, edit=edit_reference(add))
    assert 'reference' not in rules(
        check(repo, landed, protected='contract/reference/**')
    )
    git(repo, 'reset', '-q', '--hard', landed)

    def both(r: Path) -> None:
        edit_reference(
            lambda lines: next(x for x in lines if x['case'] == ACCEPTED)[
                'outcome'
            ].update(stage='predict')
        )(r)
        append_entry(
            r,
            {
                'id': 'SEED-1',
                'cases': [ACCEPTED],
                'case_outcomes': [
                    outcome(
                        ACCEPTED,
                        'python-ref',
                        'success at materialize',
                        'success at predict',
                    )
                ],
            },
        )

    commit(repo, edit=both)
    assert 'reference' not in rules(
        check(repo, landed, protected='contract/reference/**')
    )


def test_a_new_reference_commit_is_the_owner_s(repo: Path) -> None:
    landed = checkpoint_of(repo)

    def rename(r: Path) -> None:
        path = next((r / 'contract/reference').glob('*.jsonl'))
        path.rename(path.with_name('0' * 40 + '.jsonl'))

    commit(repo, edit=rename)
    assert any(
        'new reference commit' in p for p in check(repo, landed, protected='')
    )


# --- the register through the ratchet, and unreadable inputs -------------------------


def test_a_removed_register_id_fails(repo: Path) -> None:
    landed = checkpoint_of(repo)

    def drop(r: Path) -> None:
        path = r / 'docs/design/rust-core/register.yaml'
        text = path.read_text()
        start = text.index('- id: PQL-2\n')
        path.write_text(text[:start])

    commit(repo, edit=drop)
    assert any('PQL-2: removed' in p for p in check(repo, landed, protected=''))


@pytest.mark.parametrize(
    ('path', 'text', 'wanted'),
    [
        (
            'docs/design/rust-core/register.yaml',
            DEEP,
            ('nested past what the parser reads',),
        ),
        (
            'contract/availability.yaml',
            DEEP,
            ('nested past what the parser reads',),
        ),
        (
            'contract/availability.yaml',
            'runners: [1, 2]\n',
            ('not a file of its format',),
        ),
        (
            f'contract/baselines/{CELL}/python-old.jsonl',
            DEEP + '\n',
            # Python 3.14's json reads this depth; then it is no object.
            ('not JSON it reads', 'not an object'),
        ),
        (
            f'contract/baselines/{CELL}/python-old.jsonl',
            '{"case": "x"}\n',
            ('a line is not a baseline',),
        ),
        (
            f'contract/baselines/{CELL}/python-old.jsonl',
            '',
            ('the checkpoint holds no baselines of python-old',),
        ),
        (
            'rust/crates/sdm-core/tests/pql/known-differing.json',
            '[1]',
            ('is not a list of cases',),
        ),
    ],
    # Short IDs: pytest puts the test's ID in each subprocess's environment,
    # and a 200,000-character one is past what exec takes.
    ids=[
        'register-deep',
        'availability-deep',
        'availability-retyped',
        'baseline-deep',
        'baseline-retyped',
        'baseline-empty',
        'known-differing-retyped',
    ],
)
def test_an_unreadable_or_retyped_file_is_refused_by_name(
    repo: Path, tmp_path: Path, path: str, text: str, wanted: tuple[str, ...]
) -> None:
    landed = checkpoint_of(repo)
    checkpoint = land(
        repo, 'T1.15', lambda r: (r / path).write_text(text), protected=path
    )
    commit(repo)
    files = (observe(repo, tmp_path, 'python-old'),)
    found = ratchet.check(repo, checkpoint, list(files), '').problems
    assert any(w in p for p in found for w in wanted), found
    assert landed != checkpoint


def test_an_observation_file_nested_too_deep_or_empty_is_refused_by_name(
    repo: Path, tmp_path: Path
) -> None:
    landed = checkpoint_of(repo)
    commit(repo)
    deep = tmp_path / 'deep.jsonl'
    deep.write_text(DEEP + '\n')
    wide = tmp_path / 'wide.jsonl'
    wide.write_text(json.dumps({f'k{n}': n for n in range(100_000)}) + '\n')
    empty = tmp_path / 'empty.jsonl'
    empty.write_text('')
    for path, wanted in (
        (deep, ('not JSON it reads', 'not an object')),
        (wide, ('an observation of no runner',)),
        (empty, ('no observations',)),
    ):
        found = check(repo, landed, (path,))
        assert any(w in p for p in found for w in wanted), found


def test_ci_judges_a_phase_push_from_its_previous_tip_and_a_lane_from_the_phase_tip(
    tmp_path: Path,
) -> None:
    event = tmp_path / 'event.json'
    before = 'a' * 40
    event.write_text(json.dumps({'before': before}))
    push = {'GITHUB_EVENT_NAME': 'push', 'GITHUB_EVENT_PATH': str(event)}
    base = '--base origin/rust-core-phase-1'
    assert ratchet.ci_base(
        {**push, 'GITHUB_REF_NAME': 'rust-core-phase-1'}
    ) == (f'--phase-push {before}')
    lane = {**push, 'GITHUB_REF_NAME': 'rust-core-phase-1-lane-1a'}
    assert ratchet.ci_base(lane) == base
    # A local replay has no CI environment and judges against the tip of
    # the phase SDM_PHASE_BRANCH names (T2.0).
    assert ratchet.ci_base({'SDM_PHASE_BRANCH': 'rust-core-phase-1'}) == base
    # Every phase alike (T2.0): Phase 2's push, lane and local replay.
    phase_2 = {**push, 'GITHUB_REF_NAME': 'rust-core-phase-2'}
    assert ratchet.ci_base(phase_2) == f'--phase-push {before}'
    assert (
        ratchet.ci_base({**push, 'GITHUB_REF_NAME': 'rust-core-phase-2-lane-a'})
        == '--base origin/rust-core-phase-2'
    )
    assert (
        ratchet.ci_base({'SDM_PHASE_BRANCH': 'rust-core-phase-2'})
        == '--base origin/rust-core-phase-2'
    )
    # A name that is no phase branch is refused, never defaulted.
    with pytest.raises(ratchet.Unreadable):
        ratchet.ci_base({'SDM_PHASE_BRANCH': 'rust-core-phase-two'})
    # No ratchet runs elsewhere, nor on another event.
    assert ratchet.ci_base({**push, 'GITHUB_REF_NAME': 'main'}) == ''
    assert ratchet.ci_base({**lane, 'GITHUB_EVENT_NAME': 'pull_request'}) == ''
    # A phase push whose event names no previous tip is refused by name.
    for broken in ('{}', '{"before": 7}', '[' * 100_000 + ']' * 100_000):
        event.write_text(broken)
        with pytest.raises(ratchet.Unreadable):
            ratchet.ci_base({**push, 'GITHUB_REF_NAME': 'rust-core-phase-1'})


@pytest.mark.parametrize('phase', ['1', '3', '10'])
def test_a_dispatched_phase_uses_its_landing_checkpoint_and_rejects_an_unlanded_head(
    tmp_path: Path, phase: str
) -> None:
    repo = tmp_path / 'dispatch'
    repo.mkdir()
    branch = f'rust-core-phase-{phase}'
    git(repo, 'init', '-q', '-b', branch)
    baseline = commit(
        repo,
        f'Land T{phase}.0: the previous reviewed landing',
        trailers={
            'Reviewed-by': 'independent reviewers',
            'Base-SHA': 'f' * 40,
            'Checkpoint-SHA': 'f' * 40,
        },
    )
    land(repo, f'T{phase}.1')
    env = {
        **os.environ,
        'GITHUB_EVENT_NAME': 'workflow_dispatch',
        'GITHUB_REF_NAME': branch,
        'SDM_PHASE_BRANCH': branch,
    }
    command = [sys.executable, str(ROOT / 'scripts/rust_core/ratchet.py')]
    where = subprocess.run(
        [*command, 'where', '--repo', str(repo)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert where.stdout.strip() == '--base HEAD'

    def checkpoint() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                *command,
                'checkpoint',
                '--repo',
                str(repo),
                *where.stdout.split(),
            ],
            env=env,
            capture_output=True,
            text=True,
        )

    found = checkpoint()
    assert found.returncode == 0, found.stdout + found.stderr
    assert found.stdout.split()[:2] == [baseline, baseline]
    # A nightly must not turn an ordinary unlanded phase head into its own
    # checkpoint, even though dispatch has no push-event previous tip.
    commit(repo, f'T{phase}.2: unlanded work')
    refused = checkpoint()
    assert refused.returncode == 1
    assert 'self-comparison' in refused.stderr


def test_dispatch_retains_lane_bases_and_keeps_unrelated_events_out() -> None:
    dispatch = {'GITHUB_EVENT_NAME': 'workflow_dispatch'}
    assert (
        ratchet.ci_base(
            {**dispatch, 'GITHUB_REF_NAME': 'rust-core-phase-3-topic-T3.9'}
        )
        == '--base origin/rust-core-phase-3'
    )
    for event, ref in [
        ('workflow_dispatch', 'main'),
        ('pull_request', 'rust-core-phase-3'),
        ('schedule', 'main'),
    ]:
        assert (
            ratchet.ci_base(
                {'GITHUB_EVENT_NAME': event, 'GITHUB_REF_NAME': ref}
            )
            == ''
        )


compare = _module(ROOT / 'scripts/rust_core/compare.py', 'compare_for_ratchet')


@pytest.mark.parametrize(
    ('text', 'wanted'),
    [
        (DEEP, 'not JSON it reads'),
        ('[]', 'not effective expectations'),
        (
            '{"a.case": {"python-old": {"rows": 1}}}',
            'not effective expectations',
        ),
        ('{"a.case": {"python-old": []}}', 'not effective expectations'),
        ('{', 'not JSON it reads'),
    ],
    ids=['deep', 'a-list', 'another-field', 'retyped', 'truncated'],
)
def test_the_comparator_refuses_effective_expectations_of_another_shape(
    tmp_path: Path, text: str, wanted: str
) -> None:
    path = tmp_path / 'effective.json'
    path.write_text(text)
    # Python 3.14's json reads the deep list; it is then no mapping.
    with pytest.raises(ValueError, match=wanted if text != DEEP else 'not '):
        compare.expectations(path)
