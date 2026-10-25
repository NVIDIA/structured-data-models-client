# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The coverage check, ``scripts/rust_core/coverage.py --check`` (T1.15's
exit; scope-format.md, "Checks", rules 1 to 5 and 8; design, "Coverage
generator"; decision E66(f)), each rule on a real git history.

    coverage.py --check [--repo DIR] [--rev REV] --checkpoint REV

The candidate is ``--rev`` (``HEAD`` by default) of ``--repo``, read from
git, committed content only; rule 8 compares it with the manifests at
``--checkpoint`` and reads that commit's phase-1 ``decisions.md``. Exit 0
when every rule holds, 1 on any problem, each on a line of its own starting
``rule N:``, ``schema:`` or ``read:``, 2 on a usage error.

- T1.15's exit: ``--check`` passes on this repository, and an unresolved ID
  and a dropped path each make it fail.
- Each rule fails on a seeded fault of its own, on a temporary repository
  holding a copy of the manifests, their schema, the phase-1 decisions, the
  case corpus and the reference observations (rule 6's evidence, T1.32g),
  committed as the checkpoint; the unseeded candidate passes first, so a
  failure is the seed's.
- Rule 6 in the copy: a claiming case whose reference line is gone, or a
  copy with no reference observations at all, fails closed.
- Rule 8 passes only under a row of the checkpoint's decisions approved by
  the owner and naming the ID; a row the candidate adds never does.
- E49: a manifest or case missing, empty, retyped, nested 100,000 deep or
  100,000 wide is refused by name, exit 1, never a traceback.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.tooling

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / 'scripts/rust_core/coverage.py'
spec = importlib.util.spec_from_file_location(
    'ratchet_tests', ROOT / 'tests/test_contract_ratchet.py'
)
assert spec is not None and spec.loader is not None
helpers = importlib.util.module_from_spec(spec)
sys.modules['ratchet_tests'] = helpers
spec.loader.exec_module(helpers)
git, commit, land = helpers.git, helpers.commit, helpers.land

PHASE0 = 'docs/design/rust-core/phase-0'
PYTHON = f'{PHASE0}/scope-python.yaml'
JAVA = f'{PHASE0}/scope-java.yaml'
SCHEMA = f'{PHASE0}/scope.schema.json'
DECISIONS = 'docs/design/rust-core/phase-1/decisions.md'
CASES = 'contract/cases'
#: What the temporary repository copies: what the check reads, and the
#: case schema beside the corpus.
#: The reference observations rule 6 holds the cases' cell claims to.
REFERENCE = 'contract/reference'
COPIED = (
    PYTHON,
    JAVA,
    SCHEMA,
    DECISIONS,
    CASES,
    'contract/case.schema.json',
    REFERENCE,
)
#: The problem lines the check prints, every rule's (rule 6's among them,
#: from T1.32g: a problem of any rule counts, so none passes unseen).
PROBLEM = re.compile(r'^(rule [1-8]|schema|read):')
#: An acceptance case that claims cells.
CLAIMING = 'base.regression'
#: An S1 path no case names, and a cell no case names.
PATH = 'source.open/explicit/duckdb'
#: An S1 path no case claims: from T3.2 every DuckDB case claims ``PATH``,
#: the explicit opening of its source (coverage.py's ``claims``).
UNCLAIMED = 'source.open/auto/duckdb'
CELL = 'int8/feature'
#: An accepted cell no case names.
ACCEPTED_CELL = 'float64/feature'
#: The S0 paths, and a case that names one of them.
ACCEPT = 'query.validate/accept/any'
REFUSE = 'query.validate/refuse/any'
CASE = 's0.accept.sum-window'
#: A path that is no path of scope-python.yaml, of an operation, mode and
#: backend it declares.
ABSENT = 'query.validate/accept/duckdb'
APPROVED = 'approved by owner 2026-10-12'


def build(repo: Path) -> tuple[Path, str]:
    r"""The temporary repository: the checked files, committed as one
    landing, the checkpoint; with that landing's SHA."""
    git(repo, 'init', '-q', '-b', 'rust-core-phase-1')
    listed = subprocess.run(
        ['git', '-C', str(ROOT), 'ls-files', '-z', '--', *COPIED],
        capture_output=True,
        check=True,
    ).stdout.split(b'\0')
    # The seeded tests judge their own fixture: the corpus's cases the
    # committed reference observes (rule 6's evidence), never a case added
    # beside them without a reference line yet (a dry run's stubs, T3.2),
    # whose rule 6 line every exact problem list would then carry.
    recorded = {
        json.loads(text)['case'].partition('@')[0]
        for path in sorted((ROOT / REFERENCE).glob('*.jsonl'))
        for text in path.read_text(encoding='utf-8').splitlines()
        if text.strip()
    }
    for name in filter(None, (item.decode() for item in listed)):
        source = ROOT / name
        if (
            name.startswith(CASES + '/')
            and name.endswith('.json')
            and json.loads(source.read_text(encoding='utf-8')).get('id')
            not in recorded
        ):
            continue
        if source.is_file():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_bytes(source.read_bytes())
    # The client tests the manifests claim a path by (``tests``, rule 4
    # from T3.2), which the copy's tree must define as the candidate's does.
    for manifest in (PYTHON, JAVA):
        text = (ROOT / manifest).read_text(encoding='utf-8')
        claimed = re.findall(r'([\w./-]+\.py)::', text) + re.findall(
            r'(java/src/test/java/[\w./-]+\.java)#', text
        )
        for name in sorted(set(claimed)):
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_bytes((ROOT / name).read_bytes())
    landed = commit(
        repo,
        'Land T1.14: the start',
        trailers={'Reviewed-by': 'test', 'Base-SHA': 'f' * 40},
    )
    return repo, landed


@pytest.fixture(scope='module')
def origin(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Path, str]]:
    owned = tmp_path_factory.mktemp('origin')
    try:
        yield build(owned)
    finally:
        shutil.rmtree(owned)


@pytest.fixture
def repo(origin: tuple[Path, str], tmp_path: Path) -> Path:
    r"""A worktree at the checkpoint, for one test's candidate."""
    source, landed = origin
    tree = tmp_path / 'tree'
    # Passed-only tmp_path retention can remove this path but leave its Git
    # registration. Prune missing worktrees in this fixture's own origin,
    # preserving every extant worktree and its failed-test evidence.
    git(source, 'worktree', 'prune', '--expire', 'now')
    git(source, 'worktree', 'add', '-q', '--detach', str(tree), landed)
    return tree


def run(*arguments: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    # Not vacuous: a missing script exits 2 as well.
    assert COVERAGE.is_file(), f'{COVERAGE} is not built'
    return subprocess.run(
        [sys.executable, str(COVERAGE), *arguments],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
    )


def check(
    repo: Path, checkpoint: str, rev: str | None = None
) -> subprocess.CompletedProcess[str]:
    where = ['--rev', rev] if rev is not None else []
    return run(
        '--check',
        '--repo',
        str(repo),
        *where,
        '--checkpoint',
        checkpoint,
        cwd=repo,
    )


def problems(done: subprocess.CompletedProcess[str]) -> list[str]:
    output = done.stdout + done.stderr
    # Refused by name, never a crash.
    assert 'Traceback' not in output, output
    assert 'RecursionError' not in output, output
    return [line for line in output.splitlines() if PROBLEM.match(line)]


def rules(found: list[str]) -> set[str]:
    return {line.split(':', 1)[0] for line in found}


def failing(
    done: subprocess.CompletedProcess[str], rule: str, name: str
) -> list[str]:
    r"""The check failed, with a problem of ``rule`` naming ``name``."""
    found = problems(done)
    assert done.returncode == 1, done.stdout + done.stderr
    assert any(
        line.startswith(f'{rule}:') and name in line for line in found
    ), found
    return found


def passing(done: subprocess.CompletedProcess[str]) -> None:
    found = problems(done)
    assert done.returncode == 0, done.stdout + done.stderr
    assert found == [], found


def manifest(
    path: str, change: Callable[[dict[str, Any]], None]
) -> Callable[[Path], None]:
    def edit(repo: Path) -> None:
        document = yaml.safe_load((repo / path).read_text(encoding='utf-8'))
        change(document)
        (repo / path).write_text(
            yaml.safe_dump(document, sort_keys=False), encoding='utf-8'
        )

    return edit


def case_file(repo: Path, name: str) -> Path:
    [found] = list((repo / CASES).glob(f'*/{name}.json'))
    return found


def case(name: str, change: Callable[[dict[str, Any]], None]) -> Any:
    def edit(repo: Path) -> None:
        path = case_file(repo, name)
        document = json.loads(path.read_text(encoding='utf-8'))
        change(document)
        path.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')

    return edit


def corpus(repo: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding='utf-8'))
        for path in sorted((repo / CASES).glob('**/*.json'))
    ]


def generated(repo: Path, path: str) -> list[str]:
    r"""The list ``--write`` would write for ``path``: the acceptance cases
    naming it, sorted by ID."""
    return sorted(
        each['id']
        for each in corpus(repo)
        if each['status'] == 'acceptance' and path in each['paths']
    )


def entry(document: dict[str, Any], key: str, name: str) -> dict[str, Any]:
    [found] = [item for item in document[key] if item['id'] == name]
    return found


def drop(key: str, name: str) -> Callable[[dict[str, Any]], None]:
    def change(document: dict[str, Any]) -> None:
        before = len(document[key])
        document[key] = [x for x in document[key] if x['id'] != name]
        assert len(document[key]) == before - 1, name

    return change


def update(
    key: str, name: str, **fields: Any
) -> Callable[[dict[str, Any]], None]:
    def change(document: dict[str, Any]) -> None:
        entry(document, key, name).update(fields)

    return change


def owner_row(row: str, names: str, status: str = APPROVED) -> Any:
    def edit(repo: Path) -> None:
        path = repo / DECISIONS
        path.write_text(
            path.read_text(encoding='utf-8').rstrip('\n')
            + f'\n| {row} | Seeded: {names} goes | Seeded | {status} |\n',
            encoding='utf-8',
        )

    return edit


# --- T1.15's exit ------------------------------------------------------------


def test_check_passes_on_this_repository() -> None:
    r"""Exit: ``--check`` passes on the repository (its HEAD, judged against
    itself as the checkpoint)."""
    passing(run('--check', '--repo', str(ROOT), '--checkpoint', 'HEAD'))


def test_the_unseeded_copy_passes(repo: Path) -> None:
    r"""The control every seed below builds on."""
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo)
    passing(check(repo, landed))


def test_an_unresolved_case_id_fails(repo: Path) -> None:
    r"""Exit: an unresolved ID makes it fail (rule 4)."""
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(
        repo,
        edit=manifest(PYTHON, update('paths', ACCEPT, cases=['no.such-case'])),
    )
    found = failing(check(repo, landed), 'rule 4', 'no.such-case')
    assert rules(found) == {'rule 4'}, found


def test_a_dropped_path_fails(repo: Path) -> None:
    r"""Exit: a dropped path makes it fail (rule 8)."""
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo, edit=manifest(PYTHON, drop('paths', PATH)))
    found = failing(check(repo, landed), 'rule 8', PATH)
    assert rules(found) == {'rule 8'}, found


# --- rule 6: the evidence the copy carries -------------------------------------


def _without_line(name: str) -> Callable[[Path], None]:
    def edit(repo: Path) -> None:
        [path] = list((repo / REFERENCE).glob('*.jsonl'))
        lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
        kept = [x for x in lines if json.loads(x)['case'] != name]
        assert len(kept) == len(lines) - 1, name
        path.write_text(''.join(kept), encoding='utf-8')

    return edit


def test_a_claim_whose_reference_line_is_gone_fails(repo: Path) -> None:
    r"""T1.32g: the copy carries the reference so the claims pass on their
    evidence, never because it is missing: without the case's line, its
    claims fail rule 6."""
    claiming = json.loads(case_file(repo, CLAIMING).read_text(encoding='utf-8'))
    assert claiming['cells'], CLAIMING
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo, edit=_without_line(CLAIMING))
    found = failing(check(repo, landed), 'rule 6', CLAIMING)
    assert rules(found) == {'rule 6'}, found


def test_a_copy_without_the_reference_fails_closed(repo: Path) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')

    def edit(repo: Path) -> None:
        for path in (repo / REFERENCE).glob('*.jsonl'):
            path.unlink()

    commit(repo, edit=edit)
    done = check(repo, landed)
    found = problems(done)
    assert done.returncode == 1, done.stdout + done.stderr
    assert rules(found) & {'rule 6', 'read'}, found


# --- rule 4: every ID resolves, and the lists are generated --------------------


@pytest.mark.parametrize(
    ('field', 'name'),
    [('paths', ABSENT), ('routes', ABSENT), ('cells', 'float16/feature')],
)
def test_a_case_naming_what_the_manifest_lacks_fails(
    repo: Path, field: str, name: str
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo, edit=case(CASE, lambda c: c[field].append(name)))
    found = failing(check(repo, landed), 'rule 4', name)
    # A cell claimed is also held to the case's evidence, which shows no
    # such cell (rule 6, T1.32g).
    wanted = {'rule 4', 'rule 6'} if field == 'cells' else {'rule 4'}
    assert rules(found) == wanted, found


def test_a_listed_case_whose_file_is_gone_fails(repo: Path) -> None:
    r"""The checkpoint lists the generated cases; the candidate removes one
    of them from the corpus and leaves the list."""
    landed = git(repo, 'rev-parse', 'HEAD')
    listed = generated(repo, ACCEPT)
    checkpoint = land(
        repo,
        'T1.16',
        manifest(PYTHON, update('paths', ACCEPT, cases=listed)),
    )
    passing(check(repo, landed))
    commit(repo, edit=lambda r: case_file(r, listed[0]).unlink())
    failing(check(repo, checkpoint), 'rule 4', listed[0])


def test_the_generated_list_passes(repo: Path) -> None:
    r"""The control: a committed list that is exactly the generated one,
    the acceptance cases naming the path, sorted by ID."""
    landed = git(repo, 'rev-parse', 'HEAD')
    listed = generated(repo, ACCEPT)
    assert len(listed) > 2
    commit(repo, edit=manifest(PYTHON, update('paths', ACCEPT, cases=listed)))
    passing(check(repo, landed))


@pytest.mark.parametrize('how', ['a subset', 'out of order', 'a diagnostic'])
def test_a_hand_edited_list_fails(repo: Path, how: str) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    listed = generated(repo, ACCEPT)
    diagnostic = sorted(
        each['id']
        for each in corpus(repo)
        if each['status'] == 'diagnostic' and ACCEPT in each['paths']
    )
    assert diagnostic, 'no diagnostic case names the path'
    edited = {
        'a subset': listed[:1],
        'out of order': listed[::-1],
        'a diagnostic': sorted([*listed, diagnostic[0]]),
    }[how]
    commit(repo, edit=manifest(PYTHON, update('paths', ACCEPT, cases=edited)))
    found = failing(check(repo, landed), 'rule 4', ACCEPT)
    assert rules(found) == {'rule 4'}, found


#: A cell both acceptance and diagnostic cases name (from T1.29, which
#: admitted the baseline cases).
KEYED_CELL = 'int64/primary-key'


def _cell_cases(repo: Path, status: str) -> list[str]:
    r"""The cases of ``status`` naming :data:`KEYED_CELL`, sorted by ID:
    for ``acceptance``, the list ``--write`` would write for the cell."""
    return sorted(
        each['id']
        for each in corpus(repo)
        if each['status'] == status and KEYED_CELL in each['cells']
    )


def test_the_generated_cell_list_passes(repo: Path) -> None:
    r"""The control for the cell lists (T1.29 tests): a committed list that
    is exactly the generated one, the acceptance cases naming the cell,
    passes, so the seeds below fail on their edit alone."""
    landed = git(repo, 'rev-parse', 'HEAD')
    listed = _cell_cases(repo, 'acceptance')
    assert len(listed) > 1, 'no two acceptance cases name the cell'
    commit(
        repo,
        edit=manifest(PYTHON, update('cells', KEYED_CELL, cases=listed)),
    )
    passing(check(repo, landed))


def _demotable(repo: Path, listed: list[str]) -> str:
    r"""An acceptance case of ``listed`` whose every path another
    acceptance case also names, so demoting it to diagnostic leaves the
    active slice's paths covered (rule 5) and changes no claim (rule 6)."""
    accepted = [each for each in corpus(repo) if each['status'] == 'acceptance']
    for name in reversed(listed):
        mine = next(each for each in accepted if each['id'] == name)
        others = {
            path
            for each in accepted
            if each['id'] != name
            for path in each['paths']
        }
        if set(mine['paths']) <= others:
            return name
    raise AssertionError(f'no case naming {KEYED_CELL} can be demoted')


@pytest.mark.parametrize(
    'how',
    ['a diagnostic', 'a diagnostic made here', 'a subset', 'out of order'],
)
def test_a_hand_edited_cell_list_fails(repo: Path, how: str) -> None:
    r"""A cell's list that is not the generated one fails rule 4, even when
    every case it holds names the cell: a diagnostic case naming it (the
    seed T1.15 wrote, when no acceptance case named a cell), one acceptance
    case of the several naming it, or the generated cases out of order.

    The diagnostic is one of the corpus's where a diagnostic case names the
    cell; where none does (the families admitted), or always for ``a
    diagnostic made here``, the candidate demotes one of the acceptance
    cases naming it and lists that, so admissions never leave the seed
    without its fixture."""
    landed = git(repo, 'rev-parse', 'HEAD')
    listed = _cell_cases(repo, 'acceptance')
    assert len(listed) > 2, 'no three acceptance cases name the cell'
    diagnostic = _cell_cases(repo, 'diagnostic')
    edits: list[Callable[[Path], None]] = []
    if how == 'a diagnostic made here' or (
        how == 'a diagnostic' and not diagnostic
    ):
        demoted = _demotable(repo, listed)
        edits.append(case(demoted, lambda c: c.update(status='diagnostic')))
        diagnostic = [demoted]
    edited = {
        'a diagnostic': diagnostic[:1],
        'a diagnostic made here': diagnostic[:1],
        'a subset': listed[:1],
        'out of order': listed[::-1],
    }[how]
    edits.append(manifest(PYTHON, update('cells', KEYED_CELL, cases=edited)))

    def edit(tree: Path) -> None:
        for each in edits:
            each(tree)

    commit(repo, edit=edit)
    found = failing(check(repo, landed), 'rule 4', KEYED_CELL)
    assert rules(found) == {'rule 4'}, found


# --- rules 1 to 3: the manifests' own references ------------------------------


@pytest.mark.parametrize(
    ('key', 'name'),
    [
        ('backends', 'duckdb'),
        ('operations', 'query.materialize'),
        ('paths', PATH),
        ('cells', CELL),
    ],
)
def test_a_duplicate_id_fails(repo: Path, key: str, name: str) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')

    def duplicate(document: dict[str, Any]) -> None:
        document[key].append(dict(entry(document, key, name)))

    commit(repo, edit=manifest(PYTHON, duplicate))
    found = failing(check(repo, landed), 'rule 1', name)
    assert rules(found) == {'rule 1'}, found


@pytest.mark.parametrize(
    ('path', 'name'),
    [
        (PYTHON, 'query.validate/maybe/any'),
        (PYTHON, 'no.such-op/accept/any'),
        (JAVA, 'query.validate/accept/frames'),
    ],
    ids=[
        'an undeclared mode',
        'an undeclared operation',
        'an undeclared backend',
    ],
)
def test_a_path_naming_what_is_not_declared_fails(
    repo: Path, path: str, name: str
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(
        repo,
        edit=manifest(
            path,
            lambda d: d['paths'].append(
                {'id': name, 'slice': 'S3', 'cases': []}
            ),
        ),
    )
    found = failing(check(repo, landed), 'rule 2', name)
    assert rules(found) == {'rule 2'}, found


def test_a_new_path_of_what_is_declared_passes(repo: Path) -> None:
    r"""The control for rule 2: adding a path is allowed."""
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(
        repo,
        edit=manifest(
            PYTHON,
            lambda d: d['paths'].append(
                {'id': ABSENT, 'slice': 'S3', 'cases': []}
            ),
        ),
    )
    passing(check(repo, landed))


def _unused_backend(document: dict[str, Any]) -> None:
    document['backends'].append({'id': 'frames', 'what': 'Seeded, no path'})


def _unused_mode(document: dict[str, Any]) -> None:
    entry(document, 'operations', 'query.validate')['modes'].append('unused')


def _unused_operation(document: dict[str, Any]) -> None:
    document['operations'].append(
        {'id': 'no.paths', 'api': None, 'layer': 'core', 'modes': ['only']}
    )


@pytest.mark.parametrize(
    ('path', 'change', 'name'),
    [
        (JAVA, _unused_backend, 'frames'),
        (PYTHON, _unused_mode, 'unused'),
        (PYTHON, _unused_operation, 'no.paths'),
    ],
    ids=['a backend', 'a mode', 'an operation'],
)
def test_a_declared_id_no_path_uses_fails(
    repo: Path, path: str, change: Callable[[dict[str, Any]], None], name: str
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo, edit=manifest(path, change))
    found = failing(check(repo, landed), 'rule 3', name)
    assert rules(found) == {'rule 3'}, found


# --- rule 5: S0, in Python only --------------------------------------------------


def test_an_s0_path_left_without_an_acceptance_case_fails(repo: Path) -> None:
    r"""Every acceptance case stops naming the S0 path in ``paths``; a
    diagnostic case still names it, and counts toward no coverage, and the
    routes that name it count for nothing either."""
    landed = git(repo, 'rev-parse', 'HEAD')

    def uncover(r: Path) -> None:
        for path in sorted((r / CASES).glob('**/*.json')):
            document = json.loads(path.read_text(encoding='utf-8'))
            if (
                document['status'] == 'acceptance'
                and REFUSE in document['paths']
            ):
                document['paths'].remove(REFUSE)
                path.write_text(
                    json.dumps(document, indent=2) + '\n', encoding='utf-8'
                )

    commit(repo, edit=uncover)
    still = [each for each in corpus(repo) if REFUSE in each['paths']]
    assert still and all(each['status'] == 'diagnostic' for each in still)
    assert any(
        REFUSE in each['routes']
        for each in corpus(repo)
        if each['status'] == 'acceptance'
    )
    found = failing(check(repo, landed), 'rule 5', REFUSE)
    assert rules(found) == {'rule 5'}, found


def test_a_path_moved_into_s0_without_an_acceptance_case_fails(
    repo: Path,
) -> None:
    r"""Moving a slice earlier is allowed by rule 8, and the path then needs
    S0's coverage."""
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo, edit=manifest(PYTHON, update('paths', UNCLAIMED, slice='S0')))
    found = failing(check(repo, landed), 'rule 5', UNCLAIMED)
    assert rules(found) == {'rule 5'}, found


def test_java_and_later_slices_are_held_structurally_only(repo: Path) -> None:
    r"""The committed manifests pass with Java's S0 paths, every later slice
    and every cell uncovered: S0 asks no cell, and Java no coverage."""
    landed = git(repo, 'rev-parse', 'HEAD')
    java = yaml.safe_load((repo / JAVA).read_text(encoding='utf-8'))
    s0 = [p for p in java['paths'] if p['slice'] == 'S0']
    assert s0 and all(p['cases'] == [] for p in s0)
    commit(repo)
    passing(check(repo, landed))


# --- rule 8: nothing dropped, delayed or unsettled without the owner ------------


@pytest.mark.parametrize(
    ('path', 'change', 'name'),
    [
        (PYTHON, drop('cells', CELL), CELL),
        (JAVA, drop('paths', PATH), PATH),
        (PYTHON, update('paths', PATH, slice='S2'), PATH),
        (
            PYTHON,
            update('cells', ACCEPTED_CELL, expect='unknown'),
            ACCEPTED_CELL,
        ),
    ],
    ids=[
        'a dropped cell',
        'a dropped Java path',
        'a slice moved later',
        'an expectation turned unknown',
    ],
)
def test_a_regression_of_the_manifest_fails(
    repo: Path, path: str, change: Callable[[dict[str, Any]], None], name: str
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo, edit=manifest(path, change))
    found = failing(check(repo, landed), 'rule 8', name)
    assert rules(found) == {'rule 8'}, found


@pytest.mark.parametrize(
    'change',
    [
        update('paths', 'source.open/explicit/snowflake', slice='S1'),
        update('cells', 'decimal256/feature', expect='accept'),
    ],
    ids=['a slice moved earlier', 'an unknown cell settled'],
)
def test_a_tightening_passes(
    repo: Path, change: Callable[[dict[str, Any]], None]
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    document = yaml.safe_load((repo / PYTHON).read_text(encoding='utf-8'))
    assert entry(document, 'paths', 'source.open/explicit/snowflake')[
        'slice'
    ] in ('S2', 'S3')
    assert entry(document, 'cells', 'decimal256/feature')['expect'] == 'unknown'
    commit(repo, edit=manifest(PYTHON, change))
    passing(check(repo, landed))


@pytest.mark.parametrize(
    ('change', 'name'),
    [
        (drop('paths', PATH), PATH),
        (drop('cells', CELL), CELL),
        (update('paths', PATH, slice='S3'), PATH),
        (update('cells', ACCEPTED_CELL, expect='unknown'), ACCEPTED_CELL),
    ],
    ids=['a dropped path', 'a dropped cell', 'a later slice', 'unknown'],
)
def test_an_owner_approved_checkpoint_row_naming_the_id_allows_it(
    repo: Path, change: Callable[[dict[str, Any]], None], name: str
) -> None:
    checkpoint = land(repo, 'T1.16', owner_row('E900', f'`{name}`'))
    commit(repo, 'T1.17: the regression', manifest(PYTHON, change))
    passing(check(repo, checkpoint))


@pytest.mark.parametrize(
    ('status', 'names'),
    [
        ('owner to confirm', f'`{PATH}`'),
        ('orchestrator, owner to confirm', f'`{PATH}`'),
        (APPROVED, '`source.open/auto/duckdb`'),
        (APPROVED, PATH),
    ],
    ids=['not approved', "the orchestrator's", 'another ID', 'no backticks'],
)
def test_a_checkpoint_row_that_does_not_approve_the_id_does_not_allow_it(
    repo: Path, status: str, names: str
) -> None:
    checkpoint = land(repo, 'T1.16', owner_row('E900', names, status))
    commit(repo, 'T1.17: the regression', manifest(PYTHON, drop('paths', PATH)))
    found = failing(check(repo, checkpoint), 'rule 8', PATH)
    assert rules(found) == {'rule 8'}, found


def test_a_row_the_candidate_adds_never_allows_it(repo: Path) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')

    def both(r: Path) -> None:
        owner_row('E900', f'`{PATH}`')(r)
        manifest(PYTHON, drop('paths', PATH))(r)

    commit(repo, edit=both)
    found = failing(check(repo, landed), 'rule 8', PATH)
    assert rules(found) == {'rule 8'}, found


def test_decisions_missing_at_the_checkpoint_allow_nothing(repo: Path) -> None:
    r"""Fail closed: with no decisions file at the checkpoint, a dropped
    path still fails."""
    checkpoint = land(repo, 'T1.16', lambda r: (r / DECISIONS).unlink())
    commit(repo, 'T1.17: the regression', manifest(PYTHON, drop('paths', PATH)))
    done = check(repo, checkpoint)
    found = problems(done)
    assert done.returncode == 1, done.stdout + done.stderr
    assert any(line.startswith('rule 8:') and PATH in line for line in found), (
        found
    )


# --- what is judged: the committed candidate -----------------------------------


def test_only_committed_content_is_judged(repo: Path) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo)
    manifest(PYTHON, drop('paths', PATH))(repo)
    passing(check(repo, landed))


def test_rev_names_the_candidate(repo: Path) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    seeded = commit(repo, edit=manifest(PYTHON, drop('paths', PATH)))
    git(repo, 'checkout', '-q', '--detach', landed)
    failing(check(repo, landed, rev=seeded), 'rule 8', PATH)
    passing(check(repo, landed, rev=landed))


@pytest.mark.parametrize(
    'arguments',
    [
        ['--check'],
        ['--check', '--checkpoint'],
        ['--check', '--checkpoint', 'HEAD', '--no-such-option'],
    ],
    ids=['no checkpoint', 'a checkpoint without a value', 'an unknown option'],
)
def test_a_usage_error_exits_2(repo: Path, arguments: list[str]) -> None:
    done = run(*arguments, '--repo', str(repo), cwd=repo)
    assert done.returncode == 2, done.stdout + done.stderr
    assert 'Traceback' not in done.stdout + done.stderr


# --- E49: missing, empty, retyped and deep evidence, refused by name -----------

DEEP_JSON = '[' * 100_000 + ']' * 100_000
WIDE_JSON = '{' + ','.join(f'"k{i}": {i}' for i in range(100_000)) + '}'


def _write(path: str, text: str) -> Callable[[Path], None]:
    def edit(repo: Path) -> None:
        (repo / path).write_text(text, encoding='utf-8')

    return edit


def _remove(path: str) -> Callable[[Path], None]:
    def edit(repo: Path) -> None:
        git(repo, 'rm', '-q', '-r', path)

    return edit


def _restore(path: str) -> Callable[[Path], None]:
    def edit(repo: Path) -> None:
        (repo / path).write_bytes((ROOT / path).read_bytes())

    return edit


def _retyped(document: dict[str, Any]) -> None:
    document['paths'] = 'query.validate/accept/any'


@pytest.mark.parametrize(
    ('edit', 'name', 'prefixes'),
    [
        (_remove(PYTHON), 'scope-python.yaml', ('read',)),
        (_remove(JAVA), 'scope-java.yaml', ('read',)),
        (_remove(SCHEMA), 'scope.schema.json', ('read',)),
        (_write(PYTHON, ''), 'scope-python.yaml', ('read', 'schema')),
        (_write(JAVA, 'paths: [\n'), 'scope-java.yaml', ('read',)),
        (manifest(PYTHON, _retyped), 'scope-python.yaml', ('read', 'schema')),
        (
            _write(PYTHON, '[' * 100_000 + ']' * 100_000),
            'scope-python.yaml',
            ('read',),
        ),
        (_write(SCHEMA, ''), 'scope.schema.json', ('read',)),
        (_write(SCHEMA, DEEP_JSON), 'scope.schema.json', ('read',)),
    ],
    ids=[
        'python manifest missing',
        'java manifest missing',
        'schema missing',
        'manifest empty',
        'manifest not YAML',
        'manifest retyped',
        'manifest nested 100,000 deep',
        'schema empty',
        'schema nested 100,000 deep',
    ],
)
def test_a_manifest_or_schema_it_cannot_read_is_refused_by_name(
    repo: Path,
    edit: Callable[[Path], None],
    name: str,
    prefixes: tuple[str, ...],
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo, edit=edit)
    done = check(repo, landed)
    found = problems(done)
    assert done.returncode == 1, done.stdout + done.stderr
    assert any(
        line.split(':', 1)[0] in prefixes and name in line for line in found
    ), found


def _case_text(text: str) -> Callable[[Path], None]:
    def edit(repo: Path) -> None:
        case_file(repo, CASE).write_text(text, encoding='utf-8')

    return edit


@pytest.mark.parametrize(
    'edit',
    [
        _case_text(''),
        _case_text('{"id": '),
        _case_text('[]'),
        case(CASE, lambda c: c.update(paths=ACCEPT)),
        case(CASE, lambda c: c.update(cells={'int64/feature': True})),
        case(CASE, lambda c: c.update(status=1)),
        case(CASE, lambda c: c.pop('id')),
        _case_text(DEEP_JSON),
        _case_text(WIDE_JSON),
    ],
    ids=[
        'empty',
        'not JSON',
        'not an object',
        'paths a string',
        'cells an object',
        'status a number',
        'no id',
        'nested 100,000 deep',
        '100,000 wide',
    ],
)
def test_a_case_it_cannot_read_is_refused_by_name(
    repo: Path, edit: Callable[[Path], None]
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo, edit=edit)
    done = check(repo, landed)
    found = problems(done)
    assert done.returncode == 1, done.stdout + done.stderr
    assert any(
        line.startswith('read:') and f'{CASE}.json' in line for line in found
    ), found


def test_no_corpus_fails_closed(repo: Path) -> None:
    r"""With every case gone, the S0 paths are uncovered: the check fails,
    by name, and does not pass on nothing."""
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo, edit=_remove(CASES))
    done = check(repo, landed)
    found = problems(done)
    assert done.returncode == 1, done.stdout + done.stderr
    assert rules(found) & {'read', 'rule 5'}, found


def test_a_checkpoint_it_cannot_read_fails_closed(repo: Path) -> None:
    r"""Rule 8 has nothing to hold the candidate to: refused, not passed."""
    checkpoint = land(repo, 'T1.16', _remove(PYTHON))
    commit(repo, 'T1.17: the manifest back', _restore(PYTHON))
    done = check(repo, checkpoint)
    found = problems(done)
    assert done.returncode == 1, done.stdout + done.stderr
    assert any(
        line.startswith('read:') and 'scope-python.yaml' in line
        for line in found
    ), found
