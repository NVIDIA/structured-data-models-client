# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference read reuse preserves committed snapshots and refusal rules."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE = 's0.accept.sum-window'
REFERENCE = 'contract/reference/recorded.jsonl'
pytestmark = pytest.mark.tooling


def _module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f'reference_reads_{name}', ROOT / f'scripts/rust_core/{name}.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def ratchet() -> ModuleType:
    return _module('ratchet')


@pytest.fixture(scope='module')
def register() -> ModuleType:
    return _module('register')


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        [
            'git',
            '-C',
            str(repo),
            '-c',
            'user.name=Gate Test',
            '-c',
            'user.email=gate@example.com',
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo: Path) -> str:
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '--signoff', '--allow-empty', '-m', 'fixture')
    return _git(repo, 'rev-parse', 'HEAD')


def _write(repo: Path, path: str, text: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')


def _line(case: str = CASE, **outcome: Any) -> dict[str, Any]:
    return {
        'case': case,
        'outcome': {'result': 'success', 'stage': 'materialize'} | outcome,
    }


def _reference(repo: Path, lines: list[dict[str, Any]]) -> None:
    _write(repo, REFERENCE, ''.join(json.dumps(line) + '\n' for line in lines))


def _register_document() -> dict[str, Any]:
    return {
        'format': 'sdm-register/2',
        'entries': [
            {
                'id': 'SEED-1',
                'title': 'Reference outcome guards',
                'kind': 'change',
                'old_behavior': 'The committed reference outcome',
                'new_behavior': 'A seeded transition',
                'affected_inputs': 'The seeded case on every database',
                'cases': [CASE, f'{CASE}@sqlite', f'{CASE}@postgres'],
                'status': 'approved',
                'approved_by': 'reviewer',
                'decision_ref': None,
                'notes': '',
                'case_outcomes': [
                    {
                        'case': key,
                        'cell': 'ref-pandas3',
                        'runner': 'python-core',
                        'old': 'none',
                        'new': new,
                        'summary': 'A seeded outcome',
                    }
                    for key, new in (
                        (CASE, 'refusal INVALID_REQUEST at predict'),
                        (f'{CASE}@sqlite', 'success at predict'),
                        (
                            f'{CASE}@postgres',
                            'refusal INTERNAL_ERROR at predict',
                        ),
                    )
                ],
            }
        ],
    }


def test_register_derives_all_outcome_facts_from_one_fresh_reference_read(
    register: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = f'contract/cases/s0/{CASE}.json'
    _write(tmp_path, case, (ROOT / case).read_text(encoding='utf-8'))
    _write(
        tmp_path,
        str(register.INVENTORY),
        f'cases:\n  - id: {CASE}\n',
    )
    _reference(
        tmp_path,
        [
            _line(),
            _line(f'{CASE}@sqlite', result='refusal', code='INVALID_REQUEST'),
            _line(
                f'{CASE}@postgres', result='refusal', code='UNSUPPORTED_REQUEST'
            ),
        ],
    )
    # The register's existing tolerant reference reader ignores these rows.
    with (tmp_path / REFERENCE).open('a', encoding='utf-8') as stream:
        stream.write('not JSON\n{"case":"ignored","outcome":[]}\n')
    original = register._reference_lines
    reads = []

    def read(root: Path) -> dict[str, Any]:
        reads.append(root)
        return original(root)

    monkeypatch.setattr(register, '_reference_lines', read)
    document = _register_document()
    before = copy.deepcopy(document)
    first = register.problems(document, tmp_path)
    assert reads == [tmp_path]
    assert document == before
    assert any(f'{CASE} on python-core: refuses an input' in p for p in first)
    assert any(
        f'{CASE}@postgres on python-core: no entry accepts' in p for p in first
    )
    assert not any('schema:' in p or '@sqlite' in p for p in first)

    # The same filesystem path now records a refusal. No prior invocation's
    # successful outcome may keep requiring owner approval for this transition.
    _reference(
        tmp_path,
        [
            _line(result='refusal', code='INVALID_REQUEST', stage='predict'),
            _line(f'{CASE}@sqlite', result='refusal', code='INVALID_REQUEST'),
            _line(
                f'{CASE}@postgres', result='refusal', code='UNSUPPORTED_REQUEST'
            ),
        ],
    )
    second = register.problems(document, tmp_path)
    assert reads == [tmp_path, tmp_path]
    assert not any(
        f'{CASE} on python-core: refuses an input' in p for p in second
    )
    assert any(
        f'{CASE}@postgres on python-core: no entry accepts' in p for p in second
    )
    assert document == before


def _history(repo: Path) -> str:
    _git(repo, 'init', '-q')
    _reference(repo, [_line()])
    _write(repo, f'contract/cases/s0/{CASE}.json', json.dumps({'id': CASE}))
    return _commit(repo)


def test_one_check_shares_reference_parsing_between_rules_and_identical_blobs(
    ratchet: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _history(tmp_path)
    _write(
        tmp_path, 'unrelated.txt', 'A new commit with unchanged reference bytes'
    )
    second = _commit(tmp_path)
    original = ratchet.jsonl
    reads = []

    def read(text: str | None, name: str) -> list[dict[str, Any]]:
        if name.startswith('contract/reference/'):
            reads.append(name)
        return original(text, name)

    monkeypatch.setattr(ratchet, 'jsonl', read)
    cache = ratchet._ReadCache()
    checkpoint = ratchet.Tree(tmp_path, first, reads=cache)
    candidate = ratchet.Tree(tmp_path, second, reads=cache)
    report = ratchet.Report()
    ratchet.check_reference(report, checkpoint, candidate, [], set())
    ratchet.check_keys(report, candidate)
    assert ratchet.reference_lines(checkpoint) == {CASE: _line()}
    assert ratchet.reference_lines(candidate) == {CASE: _line()}
    assert report.problems == []
    assert reads == [REFERENCE]
    # Decoded data is local to a check, not retained by a process-global cache.
    assert ratchet.reference_lines(ratchet.Tree(tmp_path, second)) == {
        CASE: _line()
    }
    assert reads == [REFERENCE, REFERENCE]


def test_changed_reference_commits_and_resets_keep_their_own_snapshot(
    ratchet: ModuleType, tmp_path: Path
) -> None:
    first = _history(tmp_path)
    cache = ratchet._ReadCache()
    checkpoint = ratchet.Tree(tmp_path, first, reads=cache)
    assert ratchet.reference_lines(checkpoint) == {CASE: _line()}
    _reference(tmp_path, [_line(stage='predict')])
    second = _commit(tmp_path)
    changed = ratchet.Tree(tmp_path, second, reads=cache)
    report = ratchet.Report()
    ratchet.check_reference(report, checkpoint, changed, [], set())
    assert report.problems == [
        f'reference: {CASE}: its line changed, and no approved entry names '
        'the case on python-ref'
    ]
    _git(tmp_path, 'reset', '-q', '--hard', first)
    restored = ratchet.Tree(
        tmp_path, _git(tmp_path, 'rev-parse', 'HEAD'), reads=cache
    )
    report = ratchet.Report()
    ratchet.check_reference(report, checkpoint, restored, [], set())
    assert report.problems == []
    assert ratchet.reference_lines(restored) == {CASE: _line()}
    assert ratchet.reference_lines(changed) == {CASE: _line(stage='predict')}


def test_cached_reference_parse_does_not_replace_baseline_validation(
    ratchet: ModuleType, tmp_path: Path
) -> None:
    _history(tmp_path)
    baseline = 'contract/baselines/ref-pandas3/python-core.jsonl'
    _write(
        tmp_path, baseline, (tmp_path / REFERENCE).read_text(encoding='utf-8')
    )
    second = _commit(tmp_path)
    candidate = ratchet.Tree(tmp_path, second)
    assert ratchet.reference_lines(candidate) == {CASE: _line()}
    with pytest.raises(
        ratchet.Unreadable, match=f'{baseline}: a line is not a baseline'
    ):
        ratchet.baselines(candidate)


@pytest.mark.parametrize(
    'text', ['not JSON\n', '[]\n', '[' * 2000 + '0' + ']' * 2000]
)
def test_rejected_reference_blobs_keep_each_path_diagnostic_and_reset_is_fresh(
    ratchet: ModuleType, tmp_path: Path, text: str
) -> None:
    first = _history(tmp_path)
    cache = ratchet._ReadCache()
    assert ratchet.reference_lines(
        ratchet.Tree(tmp_path, first, reads=cache)
    ) == {CASE: _line()}
    other = 'contract/reference/also-recorded.jsonl'
    _write(tmp_path, REFERENCE, text)
    _write(tmp_path, other, text)
    second = _commit(tmp_path)
    candidate = ratchet.Tree(tmp_path, second, reads=cache)
    for path in (REFERENCE, other):
        with pytest.raises(ratchet.Unreadable, match=f'{path}:1:'):
            candidate.json_lines(path)
    _git(tmp_path, 'reset', '-q', '--hard', first)
    assert ratchet.reference_lines(
        ratchet.Tree(tmp_path, first, reads=cache)
    ) == {CASE: _line()}
