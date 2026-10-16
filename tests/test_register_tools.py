# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``scripts/rust_core/orchestration/register_tools.py`` on small synthetic
registers, decisions files and pending files: the list of proposed entries
and their approvers, and the three-way merge by entry and by group, called
as functions, as a git merge driver and from a conflicted index."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/rust_core/orchestration/register_tools.py'


def _tools() -> ModuleType:
    spec = importlib.util.spec_from_file_location('register_tools', SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tools = _tools()

HEAD = '# The register.\n\nformat: sdm-register/2\n\nentries:\n\n'


def _entry(entry_id: str, title: str, status: str = 'approved', **extra) -> str:
    lines = [f'- id: {entry_id}', f'  title: {title}', '  kind: keep']
    lines.append(f'  status: {status}')
    for key, value in extra.items():
        lines.append(f'  {key}: {value}')
    return '\n'.join(lines) + '\n'


def _register(*blocks: str, head: str = HEAD) -> str:
    return head + '\n'.join(blocks)


BANNER = (
    '# --- Keeps ---------------------------------------------------------\n'
)
A, B, C = _entry('A-1', 'One'), _entry('A-2', 'Two'), _entry('A-3', 'Three')
BASE = _register(BANNER, A, B, C)


def _merge(base: str, ours: str, theirs: str) -> tuple[str, list[str]]:
    return tools.merge_register(base, ours, theirs)


# --- register.yaml ----------------------------------------------------------


def test_identical_sides_come_back_byte_for_byte() -> None:
    assert _merge(BASE, BASE, BASE) == (BASE, [])


def test_each_side_keeps_its_own_change_to_a_different_entry() -> None:
    ours = _register(BANNER, _entry('A-1', 'One, ours'), B, C)
    theirs = _register(BANNER, A, B, _entry('A-3', 'Three, theirs'))
    text, conflicts = _merge(BASE, ours, theirs)
    assert conflicts == []
    assert text == _register(
        BANNER, _entry('A-1', 'One, ours'), B, _entry('A-3', 'Three, theirs')
    )


def test_an_entry_they_add_lands_after_its_neighbour_on_their_side() -> None:
    new = _entry('A-9', 'Nine')
    ours = _register(BANNER, A, B, C, _entry('A-4', 'Four'))
    theirs = _register(BANNER, A, new, B, C)
    text, conflicts = _merge(BASE, ours, theirs)
    assert conflicts == []
    assert text == _register(BANNER, A, new, B, C, _entry('A-4', 'Four'))


def test_both_sides_appending_keeps_both() -> None:
    ours = _register(BANNER, A, B, C, _entry('A-4', 'Four'))
    theirs = _register(BANNER, A, B, C, _entry('A-5', 'Five'))
    text, conflicts = _merge(BASE, ours, theirs)
    assert conflicts == []
    assert text == _register(
        BANNER, A, B, C, _entry('A-4', 'Four'), _entry('A-5', 'Five')
    )


def test_a_removal_on_one_side_is_kept() -> None:
    text, conflicts = _merge(BASE, BASE, _register(BANNER, A, C))
    assert conflicts == []
    assert text == _register(BANNER, A, C)


def test_a_change_to_an_entry_the_other_side_removed_conflicts() -> None:
    ours = _register(BANNER, A, C)
    theirs = _register(BANNER, A, _entry('A-2', 'Two, theirs'), C)
    text, conflicts = _merge(BASE, ours, theirs)
    assert conflicts == ['A-2']
    assert '>>>>>>> theirs (A-2)' in text and 'Two, theirs' in text


def test_both_sides_changing_one_entry_differently_conflicts() -> None:
    ours = _register(BANNER, A, _entry('A-2', 'Two, ours'), C)
    theirs = _register(BANNER, A, _entry('A-2', 'Two, theirs'), C)
    text, conflicts = _merge(BASE, ours, theirs)
    assert conflicts == ['A-2']
    assert (
        '<<<<<<< ours (A-2)\n' + _entry('A-2', 'Two, ours') + '=======\n'
    ) in text
    assert _entry('A-2', 'Two, theirs') + '>>>>>>> theirs (A-2)\n' in text
    assert text.startswith(_register(BANNER, A))


def test_the_same_change_on_both_sides_is_not_a_conflict() -> None:
    both = _register(BANNER, A, _entry('A-2', 'Two, both'), C)
    assert _merge(BASE, both, both) == (both, [])


def test_a_change_to_the_head_merges_as_one_block() -> None:
    theirs = _register(BANNER, A, B, C, head='# Edited.\n' + HEAD)
    ours = _register(BANNER, A, B, C, _entry('A-4', 'Four'))
    text, conflicts = _merge(BASE, ours, theirs)
    assert conflicts == []
    assert text == _register(
        BANNER, A, B, C, _entry('A-4', 'Four'), head='# Edited.\n' + HEAD
    )


def test_a_new_banner_section_they_add_keeps_its_place() -> None:
    banner = '# --- Drops ---------------------------------------------------\n'
    theirs = _register(BANNER, A, B, C, banner, _entry('D-1', 'Drop'))
    ours = _register(BANNER, _entry('A-1', 'One, ours'), B, C)
    text, conflicts = _merge(BASE, ours, theirs)
    assert conflicts == []
    assert text == _register(
        BANNER, _entry('A-1', 'One, ours'), B, C, banner, _entry('D-1', 'Drop')
    )


# --- pending JSON -----------------------------------------------------------

PENDING = {
    'about': 'Tests that failed when written.',
    'pending': {'cases': ['a', 'b', 'c']},
    'authored': {'cases': ['a', 'b', 'c']},
}


def _copy(value: dict) -> dict:
    return json.loads(json.dumps(value))


def test_pending_groups_take_both_sides_removals_and_additions() -> None:
    ours, theirs = _copy(PENDING), _copy(PENDING)
    ours['pending']['cases'].remove('a')
    theirs['pending']['cases'].remove('c')
    theirs['authored']['cases'].append('d')
    theirs['pending']['cases'].append('d')
    merged, conflicts = tools.merge_pending(PENDING, ours, theirs)
    assert conflicts == []
    assert merged['pending'] == {'cases': ['b', 'd']}
    assert merged['authored'] == {'cases': ['a', 'b', 'c', 'd']}
    assert merged['about'] == PENDING['about']


def test_a_group_they_add_is_merged_in_and_about_follows_its_changer() -> None:
    theirs = _copy(PENDING)
    theirs['pending']['exits'] = ['x']
    theirs['authored']['exits'] = ['x']
    theirs['about'] = 'Rewritten.'
    merged, conflicts = tools.merge_pending(PENDING, _copy(PENDING), theirs)
    assert conflicts == []
    assert merged['pending'] == {'cases': ['a', 'b', 'c'], 'exits': ['x']}
    assert merged['about'] == 'Rewritten.'


def test_a_group_emptied_and_dropped_on_one_side_goes() -> None:
    theirs = _copy(PENDING)
    del theirs['pending']['cases']
    merged, conflicts = tools.merge_pending(PENDING, _copy(PENDING), theirs)
    assert conflicts == []
    assert merged['pending'] == {}


def test_about_changed_on_both_sides_keeps_ours_and_conflicts() -> None:
    ours, theirs = _copy(PENDING), _copy(PENDING)
    ours['about'], theirs['about'] = 'Ours.', 'Theirs.'
    merged, conflicts = tools.merge_pending(PENDING, ours, theirs)
    assert conflicts == ['about']
    assert merged['about'] == 'Ours.'


# --- the command line -------------------------------------------------------


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, '-I', str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def test_merge_driver_writes_over_ours(tmp_path: Path) -> None:
    base, ours, theirs = (tmp_path / n for n in ('O', 'A', 'B'))
    base.write_text(BASE)
    ours.write_text(_register(BANNER, A, B, C, _entry('A-4', 'Four')))
    theirs.write_text(_register(BANNER, A, C))
    path = 'docs/design/rust-core/register.yaml'
    out = _run('resolve', str(base), str(ours), str(theirs), '--path', path)
    assert out.returncode == 0, out.stderr
    assert ours.read_text() == _register(BANNER, A, C, _entry('A-4', 'Four'))


def test_merge_driver_reports_a_conflict_with_exit_one(tmp_path: Path) -> None:
    files = {n: tmp_path / f'{n}.json' for n in ('O', 'A', 'B')}
    for name, about in (('O', 'base'), ('A', 'ours'), ('B', 'theirs')):
        files[name].write_text(json.dumps({**PENDING, 'about': about}))
    out = _run('resolve', *(str(files[n]) for n in 'OAB'), '--path', 'x.json')
    assert out.returncode == 1
    assert 'conflict in about' in out.stderr
    assert json.loads(files['A'].read_text())['about'] == 'ours'


def test_merge_driver_refuses_a_side_that_is_not_json(tmp_path: Path) -> None:
    files = [tmp_path / n for n in ('O', 'A', 'B')]
    for f in files:
        f.write_text(json.dumps(PENDING))
    files[2].write_text('{not json')
    out = _run('resolve', *map(str, files), '--path', 'x-pending.json')
    assert out.returncode == 2
    assert 'theirs is not JSON' in out.stderr


def test_resolve_from_a_conflicted_index(tmp_path: Path) -> None:
    env = {**os.environ, 'GIT_CONFIG_GLOBAL': os.devnull}
    env['GIT_CONFIG_NOSYSTEM'] = '1'

    def git(*args: str, data: str | None = None) -> str:
        return subprocess.run(
            ['git', *args],
            cwd=tmp_path,
            env=env,
            input=data,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    git('init', '-q')
    path = 'register.yaml'
    sides = (
        BASE,
        _register(BANNER, _entry('A-1', 'Ours'), B, C),
        _register(BANNER, A, B, _entry('A-3', 'Theirs')),
    )
    blobs = [git('hash-object', '-w', '--stdin', data=s) for s in sides]
    info = ''.join(
        f'100644 {blob} {stage}\t{path}\n'
        for stage, blob in enumerate(blobs, start=1)
    )
    git('update-index', '--index-info', data=info)
    out = _run('resolve', '--index', path, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / path).read_text() == _register(
        BANNER, _entry('A-1', 'Ours'), B, _entry('A-3', 'Theirs')
    )


@pytest.mark.parametrize('args', [('resolve', 'one', 'two'), ('resolve',)])
def test_resolve_needs_three_files(args: tuple[str, ...]) -> None:
    out = _run(*args)
    assert out.returncode == 2
    assert 'BASE OURS THEIRS' in out.stderr


# --- proposed ---------------------------------------------------------------


def test_proposed_lists_approvers_named_and_guessed(tmp_path: Path) -> None:
    register = tmp_path / 'register.yaml'
    register.write_text(
        _register(
            BANNER,
            A,
            _entry(
                'P-1',
                'Named reviewer',
                status='proposed',
                notes='>-\n    Recommend approval. Approver: reviewer.',
            ),
            _entry(
                'P-2',
                'Refuses',
                status='proposed',
                new_behavior='The core refuses an input the reference accepts.',
            ),
            _entry('P-3', 'Plain', status='proposed'),
            _entry(
                'P-4',
                'Named owner',
                status='proposed',
                notes='Approver: owner',
                decision_ref='\n    file: decisions.md\n    row: E7',
            ),
        )
    )
    rows = {r['id']: r for r in tools.proposed_entries(register.read_text())}
    assert sorted(rows) == ['P-1', 'P-2', 'P-3', 'P-4']
    assert rows['P-1']['approver'] == 'reviewer'
    assert rows['P-1']['why'] == 'named in notes'
    assert rows['P-2']['approver'] == 'owner'
    assert rows['P-2']['why'] == 'refuses an accepted input (guess)'
    assert rows['P-3']['approver'] == 'reviewer'
    assert rows['P-3']['why'].endswith('(guess)')
    assert rows['P-4']['approver'] == 'owner'
    assert rows['P-4']['decision_row'] == 'E7'

    decisions = tmp_path / 'decisions.md'
    decisions.write_text(
        '# Decisions\n\n| # | Decision | Reason | Status |\n|---|---|---|---|\n'
        '| E1 | Do x | why | orchestrator, owner to confirm |\n'
        '| E2 | Do y | why | owner to confirm; confirmed by owner 2026-10-15 |\n'
        '| E3 | Do z | why | approved by owner |\n'
        '\n## Open decisions\n\n- Q9: which way.\n\n## Later\n\n- not open\n'
    )
    out = _run(
        'proposed',
        '--register',
        str(register),
        '--decisions',
        str(decisions),
        '--approver',
        'owner',
    )
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert [line.split('\t')[0] for line in lines[:2]] == ['P-2', 'P-4']
    assert lines[2] == '2 proposed: 2 owner'
    assert lines[3:] == [
        f'{decisions}: E1: orchestrator, owner to confirm',
        f'{decisions}: open: Q9: which way.',
    ]


def test_help_names_the_merge_driver_setup() -> None:
    out = _run('--help')
    assert out.returncode == 0
    assert 'merge=sdm-register' in out.stdout
    assert 'resolve %O %A %B --path %P' in out.stdout
