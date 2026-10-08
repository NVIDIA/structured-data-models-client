# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P9.shadow: the old Python path and the core give the same, over every
transcript and every public fixture case (``CONTRACT.md``, "The shadow
comparison").

Each path runs in a fresh process of its own, chosen by the selector. For a
transcript, compared: whether it replays as stored, every statement (SQL and
parameters), every ``Source._run`` call, the payloads (request id
normalized), the error (class, code, message, details, status), what the
result holds beyond its frame, the warnings (category, text, order) and the
result's frame. For a fixture case: its output or its error, its warnings
and its ``Source._run`` calls. Only an entry of ``fix_changes.json`` may
differ, and each entry must.
"""

from __future__ import annotations

from typing import Any

import pytest
from _switch import (
    CORE,
    ENGINE_FAILED,
    FIX_IDS,
    OLD,
    base_sha,
    by_key,
    capture,
    core_in_use,
    first_difference,
    fix_changes,
    git,
    public_fixture_cases,
    transcripts,
)

TRANSCRIPT_KEYS = (
    'check',
    'crashed',
    'divergence',
    'statements',
    'source_runs',
    'payloads',
    'error',
    'answer',
    'warnings',
    'result',
)
FIXTURE_KEYS = ('output', 'error', 'crashed', 'warnings', 'source_runs')


def _plain_warnings(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    if isinstance(out.get('warnings'), list):
        out['warnings'] = [
            {'category': w['category'], 'message': w['message']}
            for w in out['warnings']
        ]
    return out


def difference(
    old: dict[str, Any], core: dict[str, Any], keys: Any
) -> str | None:
    old, core = _plain_warnings(old), _plain_warnings(core)
    for key in keys:
        found = first_difference(old.get(key), core.get(key), key)
        if found:
            return found
    return None


@pytest.fixture(scope='module')
def captures() -> dict[str, Any]:
    return {
        (impl, what): capture(impl, what)
        for impl in (OLD, CORE)
        for what in ('transcripts', 'fixtures')
    }


def test_the_two_paths_are_the_old_python_and_the_core(captures):
    r"""The comparison means something only if the two processes ran two
    implementations: the core path loaded the binding and neither ported
    package, the old path loaded the engine."""
    problems = []
    for what in ('transcripts', 'fixtures'):
        problems += [
            f'core {what}: {p}' for p in core_in_use(captures[(CORE, what)])
        ]
    old = captures[(OLD, 'transcripts')]
    if 'sdm_relational_engine' not in old['first_imports']:
        problems.append(
            'the old path never imported sdm_relational_engine: the selector '
            "did not choose today's Python"
        )
    assert not problems, '\n'.join(problems)


def test_every_transcript_gives_the_same_on_both_paths(captures):
    old = by_key(captures[(OLD, 'transcripts')], 'name')
    core = by_key(captures[(CORE, 'transcripts')], 'name')
    assert len(old) == len(core) == len(transcripts()) >= 137, (
        len(old),
        len(core),
        len(transcripts()),
    )
    listed = fix_changes()
    differences = []
    for name, theirs in sorted(old.items()):
        if name in listed:
            continue
        found = difference(theirs, core[name], TRANSCRIPT_KEYS)
        if found:
            differences.append(f'{name}: {found}')
    succeeded = [
        n
        for n, r in core.items()
        if r.get('error') is None and not r.get('crashed') and n not in listed
    ]
    assert not differences, (
        f'{len(differences)} transcripts differ between the paths:\n'
        + '\n'.join(differences[:40])
    )
    assert len(succeeded) >= 100, f'only {len(succeeded)} transcripts answered'


def test_every_public_fixture_case_gives_the_same_on_both_paths(captures):
    old = by_key(captures[(OLD, 'fixtures')], 'id')
    core = by_key(captures[(CORE, 'fixtures')], 'id')
    cases = public_fixture_cases()
    assert len(old) == len(core) == len(cases) >= 439, (
        len(old),
        len(core),
        len(cases),
    )
    listed = fix_changes()
    differences = []
    for case_id, theirs in old.items():
        if case_id in listed:
            continue
        found = difference(theirs, core[case_id], FIXTURE_KEYS)
        if found:
            differences.append(f'{case_id}: {found}')
    answered = [r for r in core.values() if 'output' in r]
    assert not differences, (
        f'{len(differences)} fixture cases differ between the paths:\n'
        + '\n'.join(differences[:40])
    )
    assert len(answered) >= 150, f'only {len(answered)} cases gave an output'


def test_each_listed_change_is_a_fix_and_really_changes(captures):
    r"""``fix_changes.json``: every entry names a transcript or a public
    fixture case, a fix of brief task 7b, and why, in five words or more; the
    answer really differs between the paths; the core's is no unforeseen
    engine failure and replays as stored; and the CHANGELOG says so."""
    listed = fix_changes()
    old_t = by_key(captures[(OLD, 'transcripts')], 'name')
    core_t = by_key(captures[(CORE, 'transcripts')], 'name')
    old_f = by_key(captures[(OLD, 'fixtures')], 'id')
    core_f = by_key(captures[(CORE, 'fixtures')], 'id')
    problems = []
    for key, entry in sorted(listed.items()):
        if not isinstance(entry, dict):
            problems.append(f'{key}: not an object')
            continue
        if entry.get('fix') not in FIX_IDS:
            problems.append(
                f'{key}: fix {entry.get("fix")!r} is not one of {sorted(FIX_IDS)}'
            )
        if len(str(entry.get('why', '')).split()) < 5:
            problems.append(f'{key}: "why" has fewer than five words')
        if key in core_t:
            old, core, keys = old_t[key], core_t[key], TRANSCRIPT_KEYS
            if core.get('check') is not None:
                problems.append(
                    f'{key}: the core does not replay it as stored: {core["check"]}'
                )
        elif key in core_f:
            old, core, keys = old_f[key], core_f[key], FIXTURE_KEYS
        else:
            problems.append(
                f'{key}: no transcript or public fixture case has this name'
            )
            continue
        if difference(old, core, keys) is None:
            problems.append(f'{key}: listed, but both paths give the same')
        error = core.get('error') or {}
        if ENGINE_FAILED in str(error.get('message', '')) or core.get(
            'crashed'
        ):
            problems.append(
                f'{key}: the core still fails unforeseen: {error or core.get("crashed")}'
            )
    if listed and not git('diff', f'{base_sha()}...HEAD', '--', 'CHANGELOG.md'):
        problems.append('answers change, and CHANGELOG.md does not say so')
    assert not problems, '\n'.join(problems)
