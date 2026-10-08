# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P2.replay: every recorded scenario replays through today's Python.

A replay that passes proves nothing unless it would fail on a difference. So
besides the clean replay, these checks change one thing in a scratch copy of
the corpus and require the replay to fail: one byte of SQL, a parameter, the
step order, a payload value, the decoded result, the results fed back to the
run. Changing only ``request_id`` must still pass.
"""

from __future__ import annotations

import json
import subprocess
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from _support import (
    LIVE_BACKENDS,
    OFFLINE_BACKENDS,
    REPO,
    RUN_CORPUS,
    TRANSCRIPTS,
    arrow_refs,
    changed,
    copy_corpus,
    execute_steps,
    find_by_name,
    get,
    in_copy,
    leaves,
    listing,
    load,
    mutate_first_leaf,
    names_step,
    offline_env,
    output,
    put,
    read_arrow,
    recorded,
    replay_lines,
    run,
    save,
    write_arrow,
)


def _replay(
    corpus: Path | None = None,
    only: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    args: list[Any] = [RUN_CORPUS, 'replay']
    if corpus is not None:
        args += ['--corpus', corpus]
    if only:
        args += ['--only', ','.join(only)]
    return run(*args, env=env, timeout=900)


def _git_status() -> str:
    return subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=all'],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture(scope='module')
def scenarios() -> list[dict[str, Any]]:
    return recorded()


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    return offline_env(tmp_path / 'home')


def _pick(
    scenarios: list[dict[str, Any]], backend: str, min_execute: int = 0
) -> dict[str, Any]:
    for entry in scenarios:
        if entry['backend'] != backend:
            continue
        transcript = load(REPO / entry['transcript'])
        if len(execute_steps(transcript)) >= min_execute:
            return entry
    raise AssertionError(f'no {backend} scenario with {min_execute} SQL steps')


def _fails(
    proc: subprocess.CompletedProcess[str], name: str
) -> tuple[str, str]:
    lines = replay_lines(proc.stdout)
    assert proc.returncode != 0, f'replay passed after a change\n{output(proc)}'
    assert name in lines, f'no PASS/FAIL line for {name}\n{output(proc)}'
    assert lines[name][0] == 'FAIL', output(proc)
    return lines[name]


# The clean replay.


def test_every_recorded_scenario_replays(
    scenarios: list[dict[str, Any]], env: dict[str, str]
) -> None:
    before = _git_status()
    proc = _replay(env=env)
    assert proc.returncode == 0, output(proc)
    lines = replay_lines(proc.stdout)
    for entry in scenarios:
        assert lines.get(entry['name'], ('missing',))[0] == 'PASS', (
            f'{entry["name"]} did not replay\n{output(proc)}'
        )
    assert _git_status() == before, 'the replay changed the worktree'


def test_every_scenario_fails_once_changed(
    scenarios: list[dict[str, Any]], env: dict[str, str], tmp_path: Path
) -> None:
    r"""Change every transcript at once; the replay must name each one.

    A transcript with SQL gets one trailing space on its last SQL step; one
    without gets a changed payload value, or, when it posted only to the
    session routes, a changed session request value, or, when it sent
    nothing, a changed error message.
    """
    corpus = copy_corpus(tmp_path / 'corpus')
    expected_step = {}
    for entry in scenarios:
        path = in_copy(corpus, entry['transcript'])
        transcript = load(path)
        steps = execute_steps(transcript)
        if steps:
            last = steps[-1]
            transcript['steps'][last]['sql'] += ' '
            expected_step[entry['name']] = last
        elif transcript['payloads']:
            mutate_first_leaf(transcript['payloads'], skip=('request_id',))
        elif requests := [
            response['request']
            for response in transcript.get('responses', [])
            if isinstance(response.get('request'), dict)
        ]:
            mutate_first_leaf(requests, skip=('request_id',))
        else:
            transcript['error']['message'] += 'x'
        save(path, transcript)
    proc = _replay(corpus, env=env)
    assert proc.returncode != 0, output(proc)
    lines = replay_lines(proc.stdout)
    for entry in scenarios:
        name = entry['name']
        assert lines.get(name, ('missing',))[0] == 'FAIL', (
            f'{name} still replays after a change\n{output(proc)}'
        )
        if name in expected_step:
            assert names_step(lines[name][1], expected_step[name]), (
                f'{name}: expected step {expected_step[name]}: {lines[name]}'
            )


# SQL and parameters, byte for byte.


@pytest.mark.parametrize('backend', ['sqlite', 'duckdb', 'postgres'])
def test_a_changed_sql_character_fails_at_that_step(
    scenarios: list[dict[str, Any]],
    env: dict[str, str],
    tmp_path: Path,
    backend: str,
) -> None:
    entry = _pick(scenarios, backend, min_execute=3)
    corpus = copy_corpus(tmp_path / 'corpus')
    path = in_copy(corpus, entry['transcript'])
    transcript = load(path)
    middle = execute_steps(transcript)[len(execute_steps(transcript)) // 2]
    sql = transcript['steps'][middle]['sql']
    # Swap the case of one letter, keeping the length.
    at = next(i for i, c in enumerate(sql) if c.isalpha())
    transcript['steps'][middle]['sql'] = (
        sql[:at] + sql[at].swapcase() + sql[at + 1 :]
    )
    save(path, transcript)
    _, reason = _fails(_replay(corpus, [entry['name']], env), entry['name'])
    assert names_step(reason, middle), reason


def test_changed_parameters_fail_at_that_step(
    scenarios: list[dict[str, Any]], env: dict[str, str], tmp_path: Path
) -> None:
    for entry in scenarios:
        transcript = load(REPO / entry['transcript'])
        with_params = [
            i
            for i in execute_steps(transcript)
            if leaves(transcript['steps'][i].get('params'))
            and transcript['steps'][i].get('params') not in (None, [], {})
        ]
        if with_params:
            break
    else:
        raise AssertionError('no recorded SQL step has parameters')
    corpus = copy_corpus(tmp_path / 'corpus')
    path = in_copy(corpus, entry['transcript'])
    transcript = load(path)
    index = with_params[0]
    mutate_first_leaf(transcript['steps'][index]['params'])
    save(path, transcript)
    _, reason = _fails(_replay(corpus, [entry['name']], env), entry['name'])
    assert names_step(reason, index), reason


def test_swapped_steps_fail_at_the_first_of_them(
    scenarios: list[dict[str, Any]], env: dict[str, str], tmp_path: Path
) -> None:
    entry = _pick(scenarios, 'sqlite', min_execute=3)
    corpus = copy_corpus(tmp_path / 'corpus')
    path = in_copy(corpus, entry['transcript'])
    transcript = load(path)
    steps = transcript['steps']
    pairs = [
        (a, b)
        for a, b in pairwise(execute_steps(transcript))
        if b == a + 1 and steps[a]['sql'] != steps[b]['sql']
    ]
    assert pairs, 'no two adjacent, different SQL steps'
    a, b = pairs[0]
    steps[a], steps[b] = steps[b], steps[a]
    save(path, transcript)
    _, reason = _fails(_replay(corpus, [entry['name']], env), entry['name'])
    assert names_step(reason, a), reason


def test_a_dropped_step_fails(
    scenarios: list[dict[str, Any]], env: dict[str, str], tmp_path: Path
) -> None:
    entry = _pick(scenarios, 'duckdb', min_execute=2)
    corpus = copy_corpus(tmp_path / 'corpus')
    path = in_copy(corpus, entry['transcript'])
    transcript = load(path)
    del transcript['steps'][execute_steps(transcript)[-1]]
    save(path, transcript)
    _fails(_replay(corpus, [entry['name']], env), entry['name'])


# Results fed back to the run.


@pytest.mark.parametrize('backend', ['sqlite', 'duckdb', 'postgres'])
def test_the_run_reads_the_recorded_results(
    scenarios: list[dict[str, Any]],
    env: dict[str, str],
    tmp_path: Path,
    backend: str,
) -> None:
    r"""Shift every recorded SQL result; the replay must notice.

    A replay that queried a database instead of serving the recordings would
    still pass.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    entry = _pick(scenarios, backend, min_execute=1)
    corpus = copy_corpus(tmp_path / 'corpus')
    path = in_copy(corpus, entry['transcript'])
    transcript = load(path)
    shifted = 0
    for index in execute_steps(transcript):
        ref = transcript['steps'][index].get('result')
        if not isinstance(ref, str):
            continue
        table, kind = read_arrow(path.parent / ref)
        columns = []
        for column in table.columns:
            if pa.types.is_integer(column.type):
                column = pc.add(column, pa.scalar(1000, column.type))
                shifted += 1
            elif pa.types.is_floating(column.type):
                column = pc.add(column, pa.scalar(1000.5, column.type))
                shifted += 1
            elif pa.types.is_string(column.type) or pa.types.is_large_string(
                column.type
            ):
                column = pc.binary_join_element_wise(
                    column, pa.scalar('~', column.type), ''
                )
                shifted += 1
            columns.append(column)
        write_arrow(
            path.parent / ref,
            pa.Table.from_arrays(columns, schema=table.schema),
            kind,
        )
    assert shifted, f'{entry["name"]} recorded no SQL result to change'
    _fails(_replay(corpus, [entry['name']], env), entry['name'])


# Payloads: parsed-equal, only request_id normalized.


@pytest.mark.parametrize('backend', OFFLINE_BACKENDS)
def test_a_changed_payload_value_fails(
    scenarios: list[dict[str, Any]],
    env: dict[str, str],
    tmp_path: Path,
    backend: str,
) -> None:
    entry = _pick(scenarios, backend)
    corpus = copy_corpus(tmp_path / 'corpus')
    path = in_copy(corpus, entry['transcript'])
    transcript = load(path)
    mutate_first_leaf(transcript['payloads'], skip=('request_id',))
    save(path, transcript)
    _fails(_replay(corpus, [entry['name']], env), entry['name'])


def test_an_integer_turned_float_fails(
    scenarios: list[dict[str, Any]], env: dict[str, str], tmp_path: Path
) -> None:
    r"""3 and 3.0 parse to different JSON numbers; the wire keeps them apart."""
    entry = _pick(scenarios, 'sqlite')
    corpus = copy_corpus(tmp_path / 'corpus')
    path = in_copy(corpus, entry['transcript'])
    transcript = load(path)
    ints = [
        p
        for p in leaves(transcript['payloads'])
        if type(get(transcript['payloads'], p)) is int and 'request_id' not in p
    ]
    assert ints, 'no integer in the payloads'
    put(
        transcript['payloads'],
        ints[0],
        float(get(transcript['payloads'], ints[0])),
    )
    save(path, transcript)
    _fails(_replay(corpus, [entry['name']], env), entry['name'])


def test_only_request_id_is_normalized(
    scenarios: list[dict[str, Any]], env: dict[str, str], tmp_path: Path
) -> None:
    corpus = copy_corpus(tmp_path / 'corpus')
    touched = []
    for entry in scenarios:
        path = in_copy(corpus, entry['transcript'])
        transcript = load(path)
        hits = [
            p for p in leaves(transcript['payloads']) if p[-1] == 'request_id'
        ]
        for p in hits:
            put(
                transcript['payloads'],
                p,
                changed(get(transcript['payloads'], p)),
            )
        if hits:
            save(path, transcript)
            touched.append(entry['name'])
        if len(touched) == 3:
            break
    assert touched, 'no payload carries a request_id'
    proc = _replay(corpus, touched, env)
    assert proc.returncode == 0, output(proc)


# The decoded result.


@pytest.mark.parametrize('backend', OFFLINE_BACKENDS)
def test_a_changed_decoded_result_fails(
    scenarios: list[dict[str, Any]],
    env: dict[str, str],
    tmp_path: Path,
    backend: str,
) -> None:
    import pyarrow as pa

    entry = _pick(scenarios, backend)
    corpus = copy_corpus(tmp_path / 'corpus')
    path = in_copy(corpus, entry['transcript'])
    transcript = load(path)
    target = path.parent / transcript['result']
    table, kind = read_arrow(target)
    rows = table.to_pylist()
    first = rows[0]
    for key, value in first.items():
        if isinstance(value, (bool, int, float, str)):
            first[key] = changed(value)
            break
    else:
        raise AssertionError(f'nothing to change in {target.name}')
    write_arrow(target, pa.Table.from_pylist(rows, schema=table.schema), kind)
    _fails(_replay(corpus, [entry['name']], env), entry['name'])


# Recording.


def _comparable(transcript: dict[str, Any], root: Path) -> dict[str, Any]:
    payloads = json.loads(json.dumps(transcript['payloads']))
    for payload in payloads:
        payload.get('metadata', {}).pop('request_id', None)
    return {
        'steps': [
            (s['op'], s.get('sql'), json.dumps(s.get('params'), sort_keys=True))
            for s in transcript['steps']
        ],
        'payloads': json.dumps(payloads, sort_keys=True),
        'result': read_arrow(root / transcript['result'])[0].to_pylist(),
    }


@pytest.mark.parametrize('backend', [*OFFLINE_BACKENDS, 'tabular'])
def test_recording_again_reproduces_the_stored_transcript(
    scenarios: list[dict[str, Any]], tmp_path: Path, backend: str
) -> None:
    if backend == 'tabular':
        candidates = [e for e in scenarios if e['name'].startswith('tabular/')]
    else:
        candidates = [e for e in scenarios if e['backend'] == backend]
    assert candidates, f'no recorded {backend} scenario'
    entry = candidates[0]
    before = _git_status()
    corpus = tmp_path / 'fresh'
    corpus.mkdir()
    proc = run(
        RUN_CORPUS,
        'record',
        '--corpus',
        corpus,
        '--only',
        entry['name'],
        timeout=900,
    )
    assert proc.returncode == 0, output(proc)
    assert _git_status() == before, 'recording changed the worktree'
    fresh = find_by_name(corpus, entry['name'])
    stored = REPO / entry['transcript']
    assert _comparable(load(fresh), fresh.parent) == _comparable(
        load(stored), stored.parent
    ), f'{entry["name"]} records differently from the stored transcript'


def test_a_live_scenario_without_credentials_refuses_cleanly(
    tmp_path: Path,
) -> None:
    r"""Recording a live scenario needs its warehouse. The live scenarios
    were pending when this check was written and are recorded now; recording
    one again without credentials must still refuse cleanly."""
    live = [e for e in listing() if e['backend'] in LIVE_BACKENDS]
    assert {e['backend'] for e in live} == set(LIVE_BACKENDS), live
    corpus = tmp_path / 'live'
    corpus.mkdir()
    env = offline_env(tmp_path / 'home')
    for entry in {e['backend']: e for e in live}.values():
        proc = run(
            RUN_CORPUS,
            'record',
            '--corpus',
            corpus,
            '--only',
            entry['name'],
            env=env,
            timeout=300,
        )
        assert proc.returncode != 0, output(proc)
        assert 'Traceback (most recent call last)' not in proc.stderr, output(
            proc
        )
        written = [p for p in corpus.rglob('*') if p.is_file()]
        assert not written, f'a failed live recording wrote {written}'


def test_the_transcripts_directory_is_the_default_corpus() -> None:
    for entry in recorded():
        assert (
            (REPO / entry['transcript'])
            .resolve()
            .is_relative_to(TRANSCRIPTS.resolve())
        ), entry


def test_every_transcript_is_listed() -> None:
    listed = {(REPO / e['transcript']).resolve() for e in recorded()}
    stored = {p.resolve() for p in TRANSCRIPTS.rglob('*.json')}
    assert stored == listed, sorted(map(str, stored ^ listed))


def test_arrow_files_are_referenced() -> None:
    referenced = set()
    for path in TRANSCRIPTS.rglob('*.json'):
        for ref in arrow_refs(load(path)):
            referenced.add((path.parent / ref).resolve())
    stray = {p.resolve() for p in TRANSCRIPTS.rglob('*.arrow')} - referenced
    assert not stray, sorted(map(str, stray))
