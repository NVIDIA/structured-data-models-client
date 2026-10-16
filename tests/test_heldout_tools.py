# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``scripts/rust_core/heldout.py`` and ``heldout_run.sh`` on a made-up
held-out set: validate, freeze and check the digests, regenerate expects,
report and count a run with one leg missing, and the runner's dry run and
its refusal of a second run. Every case here is a dummy built in tmp_path;
no held-out case is read."""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = ROOT / 'scripts/rust_core/heldout.py'
RUNNER = ROOT / 'scripts/rust_core/heldout_run.sh'
PREFIX = 'dummy.'
SECRET_PQL = 'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id'

CASE = {
    'format': 'sdm-case/1',
    'id': 'dummy.one',
    'status': 'diagnostic',
    'slice': 'S0',
    'paths': ['query.validate/accept/any'],
    'routes': ['query.validate/accept/any', 'query.validate/refuse/any'],
    'cells': [],
    'input': {
        'base': 'shop',
        'setup': ['CREATE TABLE dummy_extra AS SELECT 1 AS x'],
        'pql': SECRET_PQL,
        'materialize': {'indices': [0, 1], 'anchor_time': '2024-03-01'},
    },
    'until': 'materialize',
    'expect': {'outcome': 'success', 'stage': 'materialize'},
    'why': 'a made-up case for the tools test',
    'source': 'tests/test_heldout_tools.py',
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


def bundle(tmp_path: pathlib.Path) -> pathlib.Path:
    directory = tmp_path / 'bundle'
    directory.mkdir()
    for cid in ('dummy.one', 'dummy.two'):
        write_case(directory, {**copy.deepcopy(CASE), 'id': cid})
    (directory / 'README.md').write_text('a dummy set\n')
    return directory


def frozen(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str, str]:
    directory = bundle(tmp_path)
    done = tool('digests', directory, '--prefix', PREFIX, '--write')
    assert done.returncode == 0, done.stdout + done.stderr
    return (
        directory,
        (directory / 'BUNDLE_SHA256').read_text().strip(),
        (directory / 'INPUTS_SHA256').read_text().strip(),
    )


def observation(cid: str, result: str, stage: str, code=None, error=None):
    return {
        'case': cid,
        'outcome': {
            'result': result,
            'stage': stage,
            'error': error,
            'code': code,
        },
    }


def jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


# --- validate ----------------------------------------------------------------


def _needs_validate_deps() -> None:
    for module in ('duckdb', 'pyarrow', 'jsonschema', 'yaml'):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f'{module} is not importable')


def test_validate_accepts_a_good_set(tmp_path):
    _needs_validate_deps()
    done = tool('validate', bundle(tmp_path), '--prefix', PREFIX)
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'cases 2 problems 0' in done.stdout


def test_validate_names_problems_without_printing_the_case(tmp_path):
    _needs_validate_deps()
    directory = bundle(tmp_path)
    broken = copy.deepcopy(CASE)
    broken['id'] = 'dummy.broken'
    broken['input']['setup'] = ['CREATE TABLE secret_table_name AS SELECT nope']
    write_case(directory, broken)
    renamed = {**copy.deepcopy(CASE), 'id': 'dummy.two'}
    (directory / 'dummy.three.json').write_text(json.dumps(renamed))
    (directory / 'other.json').write_text(json.dumps(CASE))
    done = tool('validate', directory, '--prefix', PREFIX)
    assert done.returncode == 1
    assert 'SQL dummy.broken.json' in done.stdout
    assert 'id used twice' in done.stdout
    assert 'NAME other.json' in done.stdout
    assert 'secret_table_name' not in done.stdout
    assert SECRET_PQL not in done.stdout


# --- digests -----------------------------------------------------------------


def test_digests_freeze_once_and_hold_the_set(tmp_path):
    directory, bundle_sha, inputs_sha = frozen(tmp_path)
    done = tool(
        'digests', directory, '--prefix', PREFIX, '--bundle-sha', bundle_sha,
        '--inputs-sha', inputs_sha, '--cases', 2,
    )  # fmt: skip
    assert done.returncode == 0, done.stdout
    assert f'BUNDLE_SHA256 {bundle_sha} ok' in done.stdout
    # Freezing again never overwrites the record.
    again = tool('digests', directory, '--prefix', PREFIX, '--write')
    assert 'already recorded' in again.stdout
    assert (directory / 'BUNDLE_SHA256').read_text().strip() == bundle_sha
    # A wrong count is a mismatch.
    assert (
        tool('digests', directory, '--prefix', PREFIX, '--cases', 3).returncode
        == 1
    )


def test_digests_inputs_ignore_expect_and_bundle_does_not(tmp_path):
    directory, bundle_sha, inputs_sha = frozen(tmp_path)
    path = directory / 'dummy.one.json'
    case = json.loads(path.read_text())
    case['expect'] = {'outcome': 'refusal', 'stage': 'materialize'}
    path.write_text(json.dumps(case))
    done = tool('digests', directory, '--prefix', PREFIX)
    assert done.returncode == 1
    assert f'INPUTS_SHA256 {inputs_sha} ok' in done.stdout
    assert 'BUNDLE_SHA256' in done.stdout and 'MISMATCH' in done.stdout
    case['why'] = 'changed'
    path.write_text(json.dumps(case))
    done = tool(
        'digests', directory, '--prefix', PREFIX, '--inputs-sha', inputs_sha
    )
    assert 'INPUTS_SHA256' in done.stdout
    assert f'INPUTS_SHA256 {inputs_sha} ok' not in done.stdout


# --- expect and same ---------------------------------------------------------


def test_expect_is_rewritten_from_the_reference(tmp_path):
    directory = bundle(tmp_path)
    ref = tmp_path / 'python-ref.jsonl'
    jsonl(
        ref,
        [
            observation('dummy.one', 'success', 'materialize'),
            observation(
                'dummy.two', 'refusal', 'from_source', 'INVALID_REQUEST',
                'StructuredDataError',
            ),
        ],
    )  # fmt: skip
    out = tmp_path / 'run' / 'cases'
    done = tool('expect', directory, ref, out, '--prefix', PREFIX)
    assert done.returncode == 0, done.stdout + done.stderr
    assert '1 differ' in done.stdout
    two = json.loads((out / 'dummy.two.json').read_text())
    assert two['expect'] == {
        'outcome': 'refusal',
        'stage': 'from_source',
        'error': 'StructuredDataError',
        'code': 'INVALID_REQUEST',
    }
    guesses = json.loads((tmp_path / 'run' / 'writer-guesses.json').read_text())
    assert guesses['dummy.two'] == CASE['expect']
    # Never into a directory that holds files.
    again = tool('expect', directory, ref, out, '--prefix', PREFIX)
    assert again.returncode != 0
    assert tool('same', ref, ref).returncode == 0


# --- report and counts -------------------------------------------------------


def run_dir(tmp_path: pathlib.Path, core: list[dict]) -> pathlib.Path:
    run = tmp_path / 'run'
    (run / 'cases').mkdir(parents=True)
    ids = ('dummy.a', 'dummy.b', 'dummy.c')
    for cid in ids:
        write_case(run / 'cases', {**copy.deepcopy(CASE), 'id': cid})
    ref = [observation(c, 'success', 'materialize') for c in ids]
    jsonl(run / 'python-ref.jsonl', ref)
    jsonl(run / 'python-core.jsonl', core)
    jsonl(run / 'python-old.jsonl', ref)
    jsonl(run / 'rust-host.jsonl', ref)
    (run / 'compare.txt').write_text(
        'diagnostic: outcome: dummy.b: python-core: differs\n'
    )
    return run


def test_counts_hold_the_threshold_and_approvals(tmp_path):
    core = [
        observation('dummy.a', 'success', 'materialize'),
        observation(
            'dummy.b', 'refusal', 'materialize', 'INVALID_REQUEST',
            'StructuredDataError',
        ),
        observation('dummy.c', 'success', 'materialize'),
    ]  # fmt: skip
    run = run_dir(tmp_path, core)
    done = tool('counts', run, '--prefix', PREFIX, '--threshold', 4)
    assert done.returncode == 1, done.stdout  # a new refusal, unapproved
    assert 'new refusals with no approval: 1' in done.stdout
    approved = tmp_path / 'approved.txt'
    approved.write_text('# approved before the run\ndummy.b\n')
    done = tool(
        'counts', run, '--prefix', PREFIX, '--threshold', 1,
        '--approved', approved,
    )  # fmt: skip
    assert done.returncode == 0, done.stdout
    assert 'within the threshold' in done.stdout
    done = tool(
        'counts', run, '--prefix', PREFIX, '--threshold', 0,
        '--approved', tmp_path / 'approved.txt',
    )  # fmt: skip
    assert done.returncode == 0  # approved differences never count


def test_counts_a_panic_is_a_no_go(tmp_path):
    core = [
        observation('dummy.a', 'success', 'materialize'),
        observation('dummy.b', 'success', 'materialize'),
    ]  # dummy.c: no observation
    run = run_dir(tmp_path, core)
    done = tool('counts', run, '--prefix', PREFIX, '--threshold', 10)
    assert done.returncode == 1
    assert "panics: 1 ['dummy.c']" in done.stdout


def test_report_and_counts_tolerate_one_missing_leg(tmp_path):
    core = [observation(f'dummy.{c}', 'success', 'materialize') for c in 'abc']
    run = run_dir(tmp_path, core)
    (run / 'rust-host.jsonl').rename(tmp_path / 'aside.jsonl')
    assert tool('report', run, '--prefix', PREFIX).returncode != 0
    done = tool(
        'report', run, '--prefix', PREFIX, '--missing', 'rust-host=exit 101'
    )
    assert done.returncode == 0, done.stdout + done.stderr
    report = (run / 'report.md').read_text()
    assert 'the rust-host leg is missing (exit 101)' in report
    assert '| dummy.b | success | success | success | NOT RUN | 1 | |' in report
    done = tool(
        'counts', run, '--prefix', PREFIX, '--threshold', 4,
        '--missing', 'rust-host=exit 101',
    )  # fmt: skip
    assert done.returncode == 3
    assert 'PARTIAL' in done.stdout


# --- heldout_run.sh ----------------------------------------------------------


def runner(tmp_path, *extra, stamp=None, out_arg=None):
    if shutil.which('bash') is None:
        pytest.skip('no bash')
    directory, bundle_sha, inputs_sha = frozen(tmp_path)
    wheel = tmp_path / 'wheels' / 'sdm_client-0-py3-none-any.whl'
    wheel.parent.mkdir()
    wheel.write_bytes(b'not a real wheel')
    out = tmp_path / 'out'
    args = [
        'bash', str(RUNNER), '--bundle', str(directory), '--prefix', PREFIX,
        '--bundle-sha', bundle_sha, '--inputs-sha', inputs_sha, '--cases', '2',
        '--candidate', str(tmp_path / 'candidate'),
        '--reference', str(tmp_path / 'reference'), '--reference-sha', '0' * 40,
        '--wheel', str(wheel), '--out', out_arg or str(out), '--threshold', '4',
        *(['--stamp', str(stamp)] if stamp else []), *extra,
    ]  # fmt: skip
    env = {'PATH': '/usr/bin:/bin', 'HELDOUT_TOOL_PYTHON': sys.executable}
    done = subprocess.run(
        args, capture_output=True, text=True, stdin=subprocess.DEVNULL,
        timeout=120, env=env, cwd=tmp_path,
    )  # fmt: skip
    return done, out


def test_runner_dry_run_prints_the_plan_and_writes_nothing(tmp_path):
    done, out = runner(tmp_path)
    text = done.stdout + done.stderr
    # The dummy candidate and reference are not real: the dry run says so,
    # and still checks the digests and prints every step.
    assert done.returncode == 1, text
    assert 'BUNDLE_SHA256' in text and 'MISMATCH' not in text
    assert '!! the reference checkout is not at' in text
    for step in ('== 0.', '== 4.', '== 7. THE run', '== 9.'):
        assert step in text
    assert 'dry run: ' in text and 'precondition(s) failed' in text
    assert not out.exists()
    assert SECRET_PQL not in text


def test_runner_refuses_a_second_run_and_runs_nothing(tmp_path):
    stamp = tmp_path / 'ran-once'
    stamp.write_text('once\n')
    done, out = runner(tmp_path, stamp=stamp)
    assert 'the held-out cases have run once already' in done.stdout
    done, out = runner_again(tmp_path, stamp)
    assert done.returncode == 1
    assert 'nothing run' in done.stdout
    assert '== 1.' not in done.stdout
    assert not out.exists()


def test_runner_makes_a_relative_out_absolute(tmp_path):
    # rust-host runs from inside the candidate: a relative --out would
    # point elsewhere there, after the stamp.
    done, out = runner(tmp_path, out_arg='out')
    text = done.stdout + done.stderr
    absolute = tmp_path.resolve() / 'out'
    assert f'run dir {absolute}' in done.stdout, text
    assert f'--out {absolute}/python-core.jsonl' in done.stdout, text
    assert ' --out out/' not in done.stdout
    assert not out.exists()


def test_runner_help_prints_the_whole_header():
    if shutil.which('bash') is None:
        pytest.skip('no bash')
    done = subprocess.run(
        ['bash', str(RUNNER), '--help'], capture_output=True, text=True,
        stdin=subprocess.DEVNULL, timeout=60, env={'PATH': '/usr/bin:/bin'},
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert '--bundle DIR' in done.stdout
    assert 'HELDOUT_STAMP' in done.stdout
    assert '3 a run with one leg missing (partial).' in done.stdout


def runner_again(tmp_path, stamp):
    """--execute on a fresh tmp set beside the first, the same stamp."""
    second = tmp_path / 'second'
    second.mkdir()
    return runner(second, '--execute', stamp=stamp)


def test_runner_refuses_missing_options():
    if shutil.which('bash') is None:
        pytest.skip('no bash')
    done = subprocess.run(
        ['bash', str(RUNNER), '--prefix', PREFIX], capture_output=True,
        text=True, stdin=subprocess.DEVNULL, timeout=60,
        env={'PATH': '/usr/bin:/bin'},
    )  # fmt: skip
    assert done.returncode == 2
    assert 'missing --bundle' in done.stderr
