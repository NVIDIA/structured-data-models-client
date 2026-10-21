# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T2.16a's exits on ``scripts/rust_core/heldout.py`` (the package's test
author's), on made-up sets only: no held-out case is read.

- ``counts`` counts each ``<id>@<database>`` leg (the held-out record's
  item 2): a panic, a missing leg or an unapproved new refusal on any
  database's leg is a no-go; a case counts once against the threshold
  however many of its legs differ; ``--approved`` takes ``<id>`` (every
  leg) or ``<id>@<database>`` (that leg alone); ``counts.md`` names each
  differing leg by its key and ``report`` gives each leg a row.
- ``heldout_run.sh --databases`` runs the legs on each database named.
- ``convert`` (the held-out record's item 1): a set in the held-out
  format (c8075715) into the head's case format, into a new directory,
  the original never written.
- Check 9's two dummy bundles (``dummy panic``, ``dummy over-threshold``),
  each made and counted for real in a disposable directory: a no-go.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Any

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = ROOT / 'scripts/rust_core/heldout.py'
RUNNER = ROOT / 'scripts/rust_core/heldout_run.sh'
PREFIX = 'dummy.'
SECRET_PQL = 'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id'
#: The held-out set's format commit (heldout-record.md, T2.1j's landing).
HELD_OUT_FORMAT = 'c80757150ab6daae04e94b1455fb9a5e912779e4'
DATABASES = ('duckdb', 'sqlite', 'postgres')

CASE: dict[str, Any] = {
    'format': 'sdm-case/1',
    'id': 'dummy.one',
    'status': 'diagnostic',
    'slice': 'S1',
    'paths': ['query.validate/accept/any'],
    'routes': ['query.validate/accept/any', 'query.validate/refuse/any'],
    'cells': [],
    'input': {
        'base': 'shop',
        'pql': SECRET_PQL,
        'materialize': {'indices': [0, 1], 'anchor_time': '2024-03-01'},
    },
    'until': 'materialize',
    'expect': {'outcome': 'success', 'stage': 'materialize'},
    'why': 'a made-up case for the T2.16a tools test',
    'source': 'tests/test_t216a_heldout_legs.py',
}


def tool(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-I', str(TOOL), *map(str, args)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )


def write_case(directory: pathlib.Path, case: dict) -> None:
    (directory / f'{case["id"]}.json').write_text(json.dumps(case, indent=2))


def key(cid: str, database: str) -> str:
    return cid if database == 'duckdb' else f'{cid}@{database}'


def observation(cid, result='success', stage='materialize', code=None):
    return {
        'case': cid,
        'outcome': {
            'result': result,
            'stage': stage,
            'error': None if result == 'success' else 'StructuredDataError',
            'code': code,
        },
    }


def jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


# --- counts on every database's legs ---------------------------------------

IDS = ('dummy.a', 'dummy.b', 'dummy.c')
REFUSED = ('refusal', 'materialize', 'INVALID_REQUEST')
OTHER_REFUSAL = ('refusal', 'materialize', 'UNSUPPORTED_REQUEST')


def legs_run(
    tmp_path: pathlib.Path,
    core: dict[str, tuple] | None = None,
    rust: dict[str, tuple] | None = None,
    ref: dict[str, tuple] | None = None,
    drop: dict[str, set[str]] | None = None,
) -> pathlib.Path:
    r"""A run directory whose every runner observed each case on all three
    databases, each leg keyed as the runners key it. ``core``, ``rust`` and
    ``ref`` give a leg's outcome (result, stage, code) where it is not the
    plain success; ``drop`` leaves a runner's legs out."""
    run = tmp_path / 'run'
    (run / 'cases').mkdir(parents=True)
    for cid in IDS:
        write_case(run / 'cases', {**copy.deepcopy(CASE), 'id': cid})
    keys = [key(c, d) for c in IDS for d in DATABASES]

    def lines(given: dict[str, tuple] | None, runner: str) -> list[dict]:
        out = []
        for k in keys:
            if k in (drop or {}).get(runner, set()):
                continue
            out.append(observation(k, *(given or {}).get(k, ())))
        return out

    jsonl(run / 'python-ref.jsonl', lines(ref, 'python-ref'))
    plain = {**(ref or {})}
    jsonl(
        run / 'python-core.jsonl',
        lines({**plain, **(core or {})}, 'python-core'),
    )
    jsonl(run / 'python-old.jsonl', lines(plain, 'python-old'))
    jsonl(
        run / 'rust-host.jsonl', lines({**plain, **(rust or {})}, 'rust-host')
    )
    (run / 'compare.txt').write_text('')
    return run


def counts(run: pathlib.Path, threshold: int, *approved: str, tmp=None):
    args = ['counts', run, '--prefix', PREFIX, '--threshold', threshold]
    if approved:
        listed = run.parent / 'approved.txt'
        listed.write_text(
            '# approved before the run\n' + '\n'.join(approved) + '\n'
        )
        args += ['--approved', listed]
    return tool(*args)


def test_every_leg_agreeing_is_within_the_threshold(tmp_path):
    run = legs_run(tmp_path)
    done = counts(run, 0)
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'within the threshold' in done.stdout
    assert 'cases: 3' in (run / 'counts.md').read_text()


@pytest.mark.parametrize('database', ['sqlite', 'postgres'])
@pytest.mark.parametrize('runner', ['python-core', 'rust-host'])
def test_a_panic_on_a_database_leg_is_a_no_go(tmp_path, runner, database):
    leg = key('dummy.b', database)
    crash = {leg: ('refusal', 'materialize', 'INTERNAL_ERROR')}
    run = legs_run(
        tmp_path,
        core=crash if runner == 'python-core' else None,
        rust=crash if runner == 'rust-host' else None,
    )
    done = counts(run, 10, 'dummy.b')
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'NO-GO' in done.stdout
    assert leg in done.stdout
    assert leg in (run / 'counts.md').read_text()


@pytest.mark.parametrize('runner', ['python-core', 'rust-host'])
def test_a_leg_the_reference_has_and_a_core_runner_lacks_is_a_panic(
    tmp_path, runner
):
    leg = key('dummy.c', 'postgres')
    run = legs_run(tmp_path, drop={runner: {leg}})
    done = counts(run, 10, 'dummy.c')
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'NO-GO' in done.stdout
    assert leg in done.stdout


def test_python_old_s_legs_are_reported_not_counted(tmp_path):
    leg = key('dummy.a', 'sqlite')
    run = legs_run(tmp_path, drop={'python-old': {leg}})
    done = counts(run, 0)
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_new_refusal_on_a_database_leg_is_a_no_go_until_that_leg_is_approved(
    tmp_path,
):
    leg = key('dummy.a', 'sqlite')
    run = legs_run(tmp_path, core={leg: REFUSED})
    done = counts(run, 4)
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'new refusals with no approval: 1' in done.stdout
    assert leg in done.stdout
    # Another leg's approval does not approve it.
    done = counts(run, 4, key('dummy.a', 'postgres'))
    assert done.returncode == 1, done.stdout
    # That leg's approval does, and the bare id approves every leg.
    assert counts(run, 4, leg).returncode == 0
    assert counts(run, 4, 'dummy.a').returncode == 0


def differing(*legs: str) -> dict[str, tuple]:
    r"""Each leg refused by the reference and refused otherwise by the
    core: a difference, not a new refusal."""
    return {leg: OTHER_REFUSAL for leg in legs}


def test_a_case_counts_once_however_many_of_its_legs_differ(tmp_path):
    legs = [key('dummy.a', d) for d in DATABASES]
    ref = {leg: REFUSED for leg in legs}
    run = legs_run(
        tmp_path, ref=ref, core=differing(*legs), rust=differing(*legs)
    )
    done = counts(run, 1)
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'within the threshold' in done.stdout
    assert counts(run, 0).returncode == 1
    # counts.md names each differing leg by its key.
    text = (run / 'counts.md').read_text()
    for leg in legs:
        assert leg in text, leg


def test_two_cases_differing_on_other_databases_are_two(tmp_path):
    legs = [key('dummy.a', 'sqlite'), key('dummy.b', 'postgres')]
    ref = {leg: REFUSED for leg in legs}
    run = legs_run(tmp_path, ref=ref, core=differing(*legs))
    assert counts(run, 2).returncode == 0
    done = counts(run, 1)
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'NO-GO' in done.stdout


def test_a_case_is_unapproved_while_any_differing_leg_is(tmp_path):
    legs = [key('dummy.a', 'sqlite'), key('dummy.a', 'postgres')]
    ref = {leg: REFUSED for leg in legs}
    run = legs_run(tmp_path, ref=ref, rust=differing(*legs))
    assert counts(run, 0, legs[0]).returncode == 1
    assert counts(run, 0, *legs).returncode == 0
    assert counts(run, 0, 'dummy.a').returncode == 0


def test_a_missing_core_leg_is_still_partial(tmp_path):
    run = legs_run(tmp_path)
    (run / 'rust-host.jsonl').unlink()
    done = tool(
        'counts', run, '--prefix', PREFIX, '--threshold', 4,
        '--missing', 'rust-host=exit 101',
    )  # fmt: skip
    assert done.returncode == 3, done.stdout + done.stderr
    assert 'PARTIAL' in done.stdout


def test_report_gives_each_leg_a_row(tmp_path):
    leg = key('dummy.b', 'sqlite')
    run = legs_run(tmp_path, core={leg: REFUSED})
    done = tool('report', run, '--prefix', PREFIX)
    assert done.returncode == 0, done.stdout + done.stderr
    rows = [
        line.split('|')[1].strip()
        for line in (run / 'report.md').read_text().splitlines()
        if line.startswith('| dummy.')
    ]
    assert sorted(rows) == sorted(key(c, d) for c in IDS for d in DATABASES)
    assert SECRET_PQL not in done.stdout


# --- heldout_run.sh --databases ----------------------------------------------


def dry_run(tmp_path: pathlib.Path, *extra: str) -> str:
    if shutil.which('bash') is None:
        pytest.skip('no bash')
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    for cid in ('dummy.one', 'dummy.two'):
        write_case(bundle, {**copy.deepcopy(CASE), 'id': cid})
    frozen = tool('digests', bundle, '--prefix', PREFIX, '--write')
    assert frozen.returncode == 0, frozen.stdout + frozen.stderr
    wheel = tmp_path / 'wheels' / 'sdm_client-0-py3-none-any.whl'
    wheel.parent.mkdir()
    wheel.write_bytes(b'not a real wheel')
    args = [
        'bash', str(RUNNER), '--bundle', str(bundle), '--prefix', PREFIX,
        '--bundle-sha', (bundle / 'BUNDLE_SHA256').read_text().strip(),
        '--inputs-sha', (bundle / 'INPUTS_SHA256').read_text().strip(),
        '--candidate', str(tmp_path / 'candidate'),
        '--reference', str(tmp_path / 'reference'), '--reference-sha', '0' * 40,
        '--wheel', str(wheel), '--out', str(tmp_path / 'out'),
        '--threshold', '4', *extra,
    ]  # fmt: skip
    done = subprocess.run(
        args, capture_output=True, text=True, stdin=subprocess.DEVNULL,
        timeout=120, cwd=tmp_path,
        env={'PATH': '/usr/bin:/bin', 'HELDOUT_TOOL_PYTHON': sys.executable},
    )  # fmt: skip
    assert not (tmp_path / 'out').exists()
    return done.stdout + done.stderr


def test_the_runner_runs_the_legs_on_every_database_by_default(tmp_path):
    text = dry_run(tmp_path)
    assert '--database sqlite' in text, text
    assert '--database postgres' in text, text
    assert SECRET_PQL not in text


def test_the_runner_runs_only_the_databases_named(tmp_path):
    text = dry_run(tmp_path, '--databases', 'duckdb')
    assert '== 7. THE run' in text, text
    assert '--database sqlite' not in text, text
    assert '--database postgres' not in text, text


# --- check 9's dummy bundles ---------------------------------------------------


@pytest.mark.parametrize('kind', ['panic', 'over-threshold'])
@pytest.mark.parametrize('threshold', [None, 2])
def test_check_9_s_dummy_bundles_are_each_a_no_go(tmp_path, kind, threshold):
    # A disposable directory: made here, counted for real, never reused.
    out = tmp_path / 'disposable' / kind
    given = [] if threshold is None else ['--threshold', threshold]
    made = tool('dummy', kind, out, '--prefix', PREFIX, *given)
    assert made.returncode == 0, made.stdout + made.stderr
    done = tool(
        'counts', out, '--prefix', PREFIX, '--threshold',
        4 if threshold is None else threshold,
    )  # fmt: skip
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'NO-GO' in done.stdout
    assert 'NO-GO' in (out / 'counts.md').read_text()
    if kind == 'panic':
        assert 'panics: 0' not in done.stdout, done.stdout
    else:
        # No panic and no new refusal: the threshold alone makes it a
        # no-go, with more differing cases than it allows.
        assert 'panics: 0' in done.stdout, done.stdout
        assert 'new refusals with no approval: 0' in done.stdout, done.stdout


def test_a_dummy_bundle_is_never_written_over_a_run(tmp_path):
    out = tmp_path / 'disposable'
    out.mkdir()
    (out / 'counts.md').write_text('an earlier run\n')
    done = tool('dummy', 'panic', out, '--prefix', PREFIX)
    assert done.returncode != 0
    assert (out / 'counts.md').read_text() == 'an earlier run\n'


# --- convert: the held-out format to the head's ------------------------------


def _schema_at(rev: str) -> dict:
    done = subprocess.run(
        ['git', '-C', str(ROOT), 'show', f'{rev}:contract/case.schema.json'],
        capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    if done.returncode != 0:
        pytest.skip(f'{rev[:8]} is not in this clone')
    return json.loads(done.stdout)


def _validator(schema: dict):
    jsonschema = pytest.importorskip('jsonschema')
    return jsonschema.Draft202012Validator(schema)


#: Synthetic cases in the held-out format, one per kind of input it holds:
#: a plain one, a ranking's, a request asked in several calls, and a
#: predict with a failed call and probabilities.
FORMAT_CASES: dict[str, dict[str, Any]] = {
    'dummy.plain': {},
    'dummy.ranking': {
        'materialize': {
            'indices': [0],
            'anchor_time': '2024-03-01',
            'candidates': [1, 2, '3'],
            'top_k': 2,
        },
    },  # fmt: skip
    'dummy.calls': {
        'materialize': {
            'indices': [0, 1, 2],
            'anchor_time': '2024-03-01',
            'rows_per_call': 2,
        },
    },  # fmt: skip
    'dummy.predict': {'fail_call': 1, 'probabilities': True},
}


def format_bundle(tmp_path: pathlib.Path) -> pathlib.Path:
    directory = tmp_path / 'frozen'
    directory.mkdir()
    old = _validator(_schema_at(HELD_OUT_FORMAT))
    for cid, extra in FORMAT_CASES.items():
        case = copy.deepcopy(CASE)
        case['id'] = cid
        case['input'].update(copy.deepcopy(extra))
        if 'fail_call' in extra:
            case['until'] = 'predict'
            case['input']['pql'] = (
                'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
            )
        errors = [e.message for e in old.iter_errors(case)]
        assert not errors, (cid, errors)  # the held-out format's own
        # Written as an author writes: not the converter's layout.
        (directory / f'{cid}.json').write_text(
            json.dumps(case, indent=4) + '\n'
        )
    (directory / 'README.md').write_text('a dummy set; format c8075715\n')
    frozen = tool('digests', directory, '--prefix', PREFIX, '--write')
    assert frozen.returncode == 0, frozen.stdout + frozen.stderr
    return directory


def _snapshot(directory: pathlib.Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


def _digest_lines(text: str) -> dict[str, str]:
    found = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ('BUNDLE_SHA256', 'INPUTS_SHA256'):
            found[parts[0]] = parts[1]
    return found


def test_convert_writes_the_head_s_format_into_a_new_directory(tmp_path):
    frozen = format_bundle(tmp_path)
    before = _snapshot(frozen)
    out = tmp_path / 'converted'
    done = tool('convert', frozen, out, '--prefix', PREFIX)
    assert done.returncode == 0, done.stdout + done.stderr
    assert _snapshot(frozen) == before  # the original is never written
    head = _validator(
        json.loads((ROOT / 'contract/case.schema.json').read_text())
    )
    for cid in FORMAT_CASES:
        written = out / f'{cid}.json'
        assert not list(head.iter_errors(json.loads(written.read_text()))), cid
        # No field the held-out format holds moved by the head (ranking and
        # multi-call options stay in input.materialize): copied byte for byte.
        assert written.read_bytes() == before[f'{cid}.json'], cid
    assert (out / 'README.md').read_bytes() == before['README.md']
    assert SECRET_PQL not in done.stdout + done.stderr


def test_convert_prints_the_new_directory_s_digests(tmp_path):
    frozen = format_bundle(tmp_path)
    out = tmp_path / 'converted'
    done = tool('convert', frozen, out, '--prefix', PREFIX)
    assert done.returncode == 0, done.stdout + done.stderr
    printed = _digest_lines(done.stdout)
    recorded = _digest_lines(tool('digests', out, '--prefix', PREFIX).stdout)
    assert set(printed) == {'BUNDLE_SHA256', 'INPUTS_SHA256'}, done.stdout
    assert printed == recorded
    # Nothing moved, so the inputs are the frozen set's.
    frozen_inputs = (frozen / 'INPUTS_SHA256').read_text().strip()
    assert printed['INPUTS_SHA256'] == frozen_inputs


@pytest.mark.parametrize('where', ['non-empty', 'the original'])
def test_convert_writes_only_into_a_new_or_empty_directory(tmp_path, where):
    frozen = format_bundle(tmp_path)
    before = _snapshot(frozen)
    if where == 'non-empty':
        out = tmp_path / 'converted'
        out.mkdir()
        (out / 'kept.txt').write_text('kept\n')
    else:
        out = frozen
    done = tool('convert', frozen, out, '--prefix', PREFIX)
    assert done.returncode != 0, done.stdout
    assert _snapshot(frozen) == before
    if where == 'non-empty':
        assert [p.name for p in out.iterdir()] == ['kept.txt']


def test_convert_refuses_a_case_the_head_s_format_does_not_take(tmp_path):
    frozen = format_bundle(tmp_path)
    broken = copy.deepcopy(CASE)
    broken['id'] = 'dummy.broken'
    broken['input']['not_a_field'] = SECRET_PQL
    write_case(frozen, broken)
    done = tool('convert', frozen, tmp_path / 'converted', '--prefix', PREFIX)
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'dummy.broken' in done.stdout + done.stderr
    assert SECRET_PQL not in done.stdout + done.stderr


def test_a_converted_set_validates_and_builds(tmp_path):
    for module in ('duckdb', 'pyarrow', 'jsonschema', 'yaml'):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f'{module} is not importable')
    frozen = format_bundle(tmp_path)
    out = tmp_path / 'converted'
    assert tool('convert', frozen, out, '--prefix', PREFIX).returncode == 0
    done = tool('validate', out, '--prefix', PREFIX)
    assert done.returncode == 0, done.stdout + done.stderr
    assert f'cases {len(FORMAT_CASES)} problems 0' in done.stdout
