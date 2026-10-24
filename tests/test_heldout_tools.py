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
    assert f'{absolute}/python-core.jsonl' in done.stdout, text
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


# --- Phase 3 warehouse and Java coverage -------------------------------------


def test_expect_uses_a_warehouse_case_s_own_first_reference_leg(tmp_path):
    original = tmp_path / 'warehouse-original'
    original.mkdir()
    case = {
        **copy.deepcopy(CASE),
        'warehouses': ['databricks-sql', 'snowflake'],
        'slice': 'S2',
    }
    write_case(original, case)
    ref = tmp_path / 'reference.jsonl'
    jsonl(
        ref,
        [
            observation(case['id'] + '@snowflake', 'success', 'predict'),
            observation(
                case['id'] + '@databricks-sql',
                'refusal',
                'materialize',
                'INVALID_REQUEST',
                'StructuredDataError',
            ),
        ],
    )
    out = tmp_path / 'normalized'
    done = tool('expect', original, ref, out, '--prefix', PREFIX)
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads((out / (case['id'] + '.json')).read_text())['expect'] == {
        'outcome': 'refusal',
        'stage': 'materialize',
        'code': 'INVALID_REQUEST',
        'error': 'StructuredDataError',
    }
    assert (
        tool('digests', original, '--prefix', PREFIX)
        .stdout.split('INPUTS_SHA256 ')[1]
        .split()[0]
        == tool('digests', out, '--prefix', PREFIX)
        .stdout.split('INPUTS_SHA256 ')[1]
        .split()[0]
    )


def test_counts_retain_hyphenated_warehouse_legs_and_count_payload_differences(
    tmp_path,
):
    run = run_dir(tmp_path, [])
    for path in (run / 'cases').glob('*.json'):
        case = json.loads(path.read_text())
        case['slice'] = 'S2'
        case['warehouses'] = ['databricks-sql']
        path.write_text(json.dumps(case))
    rows = [
        observation(
            'dummy.' + letter + '@databricks-sql', 'success', 'materialize'
        )
        for letter in 'abc'
    ]
    for name in ('python-ref', 'python-core', 'rust-host'):
        jsonl(run / (name + '.jsonl'), rows)
    jsonl(run / 'python-old.jsonl', [])
    (run / 'compare.txt').write_text(
        'diagnostic: payload: dummy.b@databricks-sql: python-core: differs\n'
    )
    done = tool(
        'counts',
        run,
        '--prefix',
        PREFIX,
        '--threshold',
        0,
        '--databases',
        'databricks-sql',
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'databricks-sql 3' in done.stdout
    assert "with no approval: 1 ['dummy.b']" in done.stdout
    reported = tool(
        'report', run, '--prefix', PREFIX, '--databases', 'databricks-sql'
    )
    assert reported.returncode == 0, reported.stdout + reported.stderr
    assert (
        '| dummy.b@databricks-sql | success | success | not run there | success | 1 |'
        in (run / 'report.md').read_text()
    )


@pytest.mark.parametrize('count', [3, 5])
def test_counts_union_payload_and_outcome_cases_across_hosts_once(
    tmp_path, count
):
    ids = ['dummy.' + letter for letter in 'abcde'[:count]]
    rows = [observation(cid, 'success', 'materialize') for cid in ids]
    run = run_dir(tmp_path, rows)
    for cid in ids:
        write_case(run / 'cases', {**copy.deepcopy(CASE), 'id': cid})
    for who in ('python-core', 'rust-host', 'python-old'):
        jsonl(run / (who + '.jsonl'), rows)
    reference = [
        *rows[:-1],
        observation(
            ids[-1],
            'refusal',
            'materialize',
            'INVALID_REQUEST',
            'StructuredDataError',
        ),
    ]
    jsonl(run / 'python-ref.jsonl', reference)
    (run / 'compare.txt').write_text(
        ''.join(
            f'diagnostic: payload: {cid}: python-core: differs\n'
            for cid in ids[:-1]
        )
        + f'diagnostic: result: {ids[0]}: rust-host: differs\n'
    )
    done = tool('counts', run, '--prefix', PREFIX, '--threshold', count - 1)
    assert done.returncode == 1, done.stdout + done.stderr
    assert f'with no approval: {count} {ids!r}' in done.stdout
    allowed = tool('counts', run, '--prefix', PREFIX, '--threshold', count)
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr


def test_java_authored_subset_is_counted_and_cannot_be_missing(tmp_path):
    rows = [
        observation('dummy.' + letter, 'success', 'materialize')
        for letter in 'abc'
    ]
    run = run_dir(tmp_path, rows)
    for path in (run / 'cases').glob('*.json'):
        case = json.loads(path.read_text())
        case['core'] = {'validate_query': {}}
        path.write_text(json.dumps(case))
    subset = tmp_path / 'java-case-ids.txt'
    subset.write_text('dummy.a\n')
    jsonl(run / 'java.jsonl', [])
    done = tool(
        'counts',
        run,
        '--prefix',
        PREFIX,
        '--threshold',
        4,
        '--databases',
        'duckdb',
        '--java-case-ids',
        subset,
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert "panics: 1 ['dummy.a']" in done.stdout
    jsonl(run / 'java.jsonl', [rows[0]])
    assert (
        tool(
            'counts',
            run,
            '--prefix',
            PREFIX,
            '--threshold',
            4,
            '--databases',
            'duckdb',
            '--java-case-ids',
            subset,
        ).returncode
        == 0
    )
    subset.write_text('dummy.unknown\n')
    assert (
        tool(
            'counts',
            run,
            '--prefix',
            PREFIX,
            '--threshold',
            4,
            '--databases',
            'duckdb',
            '--java-case-ids',
            subset,
        ).returncode
        != 0
    )


def test_counts_fail_when_a_required_reference_line_is_missing(tmp_path):
    rows = [
        observation('dummy.' + letter, 'success', 'materialize')
        for letter in 'abc'
    ]
    run = run_dir(tmp_path, rows)
    jsonl(run / 'python-ref.jsonl', rows[:2])
    done = tool(
        'counts',
        run,
        '--prefix',
        PREFIX,
        '--threshold',
        4,
        '--databases',
        'duckdb',
    )
    assert done.returncode != 0
    assert 'reference' in done.stdout + done.stderr


@pytest.mark.parametrize('kind', ['panic', 'over-threshold'])
def test_phase3_counts_keep_the_existing_no_go_controls(tmp_path, kind):
    run = tmp_path / 'control'
    made = tool('dummy', kind, run, '--prefix', PREFIX, '--threshold', 4)
    assert made.returncode == 0, made.stdout + made.stderr
    counted = tool('counts', run, '--prefix', PREFIX, '--threshold', 4)
    assert counted.returncode == 1, counted.stdout + counted.stderr


def test_runner_dry_mode_does_not_call_a_case_reader(tmp_path):
    sentinel = tmp_path / 'tool-sentinel'
    sentinel.write_text('#!/bin/sh\necho SEALED_READER_WAS_CALLED\nexit 99\n')
    sentinel.chmod(0o700)
    done, _ = runner(tmp_path, '--tool-python', str(sentinel))
    assert done.returncode == 1
    assert 'precondition(s) failed' in done.stdout
    assert 'SEALED_READER_WAS_CALLED' not in done.stdout + done.stderr


def test_runner_stamps_before_loading_or_digesting_any_case():
    text = RUNNER.read_text()
    executed_stamp = text.index('do_ stamp_once')
    assert executed_stamp < text.index('do_ copy_bundle')
    assert executed_stamp < text.index('do_ "${heldout[@]}" digests')
    # The preflight no longer invokes candidate-owned digest tooling.
    preflight = text[
        text.index('echo "== preconditions') : text.index('snapshot()')
    ]
    assert '"${heldout[@]}" digests' not in preflight


def test_comparator_failure_cannot_be_counted_as_a_clean_run(tmp_path):
    rows = [
        observation('dummy.' + letter, 'success', 'materialize')
        for letter in 'abc'
    ]
    run = run_dir(tmp_path, rows)
    (run / 'compare.txt').write_text(
        'Traceback (most recent call last):\nRuntimeError: comparator failed\n'
    )
    done = tool('counts', run, '--prefix', PREFIX, '--threshold', 4)
    assert done.returncode != 0
    assert 'comparison' in (done.stdout + done.stderr).lower()


def test_execute_makes_global_stamp_before_its_first_case_tool(tmp_path):
    """Actual shell flow, with dummy checkouts and a reader that only checks ordering."""
    for name in ('git', 'bash'):
        if shutil.which(name) is None:
            pytest.skip(f'no {name}')
    candidate, reference = tmp_path / 'candidate', tmp_path / 'reference'
    for checkout in (candidate, reference):
        checkout.mkdir()
        subprocess.run(['git', 'init', '-q', str(checkout)], check=True)
        (checkout / 'dummy').write_text('owned test checkout\n')
        if checkout == candidate:
            (checkout / 'contract/runners').mkdir(parents=True)
            (checkout / 'contract/runners/run_python.py').write_text(
                '# dummy\n'
            )
            (checkout / 'docs/design/rust-core').mkdir(parents=True)
            (checkout / 'docs/design/rust-core/register.yaml').write_text(
                'entries: []\n'
            )
        subprocess.run(['git', '-C', str(checkout), 'add', '.'], check=True)
        subprocess.run(
            [
                'git',
                '-C',
                str(checkout),
                '-c',
                'user.name=Tool test',
                '-c',
                'user.email=test@example.com',
                'commit',
                '-q',
                '--signoff',
                '-m',
                'dummy',
            ],
            check=True,
        )
    reference_sha = subprocess.check_output(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True
    ).strip()
    dummy = tmp_path / 'dummy-bundle'
    dummy.mkdir()
    stamp = pathlib.Path(str(dummy.resolve()) + '.started')
    binary = tmp_path / 'bin'
    binary.mkdir()
    python = binary / 'python'
    python.write_text('#!/bin/sh\nexit 0\n')
    python.chmod(0o700)
    cargo = binary / 'cargo'
    cargo.write_text('#!/bin/sh\nexit 0\n')
    cargo.chmod(0o700)
    checker = binary / 'case-reader'
    checker.write_text(
        f'#!{sys.executable}\nfrom pathlib import Path\nassert Path({str(stamp)!r}).is_file(), "reader ran before marker"\nprint("verified marker precedes case reader")\nraise SystemExit(1)\n'
    )
    checker.chmod(0o700)
    done = subprocess.run(
        [
            'bash',
            str(RUNNER),
            '--bundle',
            str(dummy),
            '--prefix',
            PREFIX,
            '--bundle-sha',
            '0' * 64,
            '--inputs-sha',
            '0' * 64,
            '--candidate',
            str(candidate),
            '--reference',
            str(reference),
            '--reference-sha',
            reference_sha,
            '--python',
            str(python),
            '--reference-python',
            str(python),
            '--tool-python',
            str(checker),
            '--databases',
            'duckdb',
            '--out',
            str(tmp_path / 'out'),
            '--threshold',
            '4',
            '--execute',
        ],
        capture_output=True,
        text=True,
        env={'PATH': str(binary) + ':/usr/bin:/bin'},
        timeout=30,
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'verified marker precedes case reader' in done.stdout
    assert stamp.is_file()
    assert not list((tmp_path / 'out/cases-orig').glob('*.json'))


def _heldout_module():
    spec = importlib.util.spec_from_file_location('heldout_review_edges', TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_java_comparison_selects_exact_admitted_case_keys_per_database(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    held = _heldout_module()
    run = tmp_path / 'comparison'
    (run / 'cases').mkdir(parents=True)
    for relative, name in [
        ('s0/s0.accept.binary-classification.json', 'dummy.s0'),
        ('s1/base.multiclass.json', 'dummy.s1'),
    ]:
        case = json.loads((ROOT / 'contract/cases' / relative).read_text())
        write_case(
            run / 'cases',
            {
                **case,
                'id': name,
                'status': 'diagnostic',
                'databases': ['duckdb', 'sqlite', 'postgres'],
            },
        )
    subset = tmp_path / 'java.txt'
    subset.write_text('dummy.s0\ndummy.s1\n')
    all_keys = [
        name + suffix
        for name in ('dummy.s0', 'dummy.s1')
        for suffix in ('', '@sqlite', '@postgres')
    ]
    admitted = {'dummy.s0', 'dummy.s1', 'dummy.s1@sqlite', 'dummy.s1@postgres'}
    for who in ('python-ref', 'python-core', 'rust-host', 'python-old', 'java'):
        keys = admitted if who == 'java' else all_keys
        jsonl(
            run / (who + '.jsonl'),
            [observation(key, 'success', 'predict') for key in keys],
        )
    seen = {}

    def compare(paths, reference, files, **kwargs):
        if any(path.name == 'java.jsonl' for path in files):
            db = kwargs['database']
            loader = held._head_loader(ROOT / 'contract')
            cases = loader.load_cases(paths)
            keys = {
                loader.case_key(case['id'], db)
                for case, _ in cases
                if db in loader.case_legs(case)
            }
            expected = {key for key in admitted if held.database_of(key) == db}
            assert keys == expected
            for path in (reference, *files):
                assert set(held._observations(path)) == expected
            seen[db] = keys
        return SimpleNamespace(notes=[], texts=[], failures=[])

    monkeypatch.setattr(
        held, '_comparison_tool', lambda: SimpleNamespace(compare=compare)
    )
    assert (
        held.cmd_compare(
            SimpleNamespace(
                run_dir=run,
                prefix=PREFIX,
                databases='duckdb,sqlite,postgres',
                java_case_ids=subset,
            )
        )
        == 0
    )
    assert seen == {
        'duckdb': {'dummy.s0', 'dummy.s1'},
        'sqlite': {'dummy.s1@sqlite'},
        'postgres': {'dummy.s1@postgres'},
    }


@pytest.mark.parametrize(
    'rule',
    [
        'sql.text',
        'sql.parameters',
        'sql.bindings',
        'payload',
        'result',
        'schema',
    ],
)
def test_h11_uses_the_comparator_policy_for_five_reported_differences(
    tmp_path, rule
):
    held = _heldout_module()
    comparator = held._comparison_tool()
    ids = ['dummy.' + letter for letter in 'abcde']
    rows = [observation(cid, 'success', 'materialize') for cid in ids]
    run = run_dir(tmp_path, rows)
    for cid in ids:
        write_case(run / 'cases', {**copy.deepcopy(CASE), 'id': cid})
    for who in ('python-ref', 'python-core', 'python-old', 'rust-host'):
        jsonl(run / (who + '.jsonl'), rows)
    report = comparator.Report(comparator._capture_check().policy()['rules'])
    for cid in ids:
        report.fail(f'{cid}: python-core: observed values differ', rule)
    report.demote(
        0
    )  # held-out case diagnostics, without changing a rule's class
    (run / 'compare.txt').write_text('\n'.join(report.notes) + '\n')
    done = tool('counts', run, '--prefix', PREFIX, '--threshold', 4)
    assert done.returncode == (0 if rule.startswith('sql.') else 1), (
        done.stdout + done.stderr
    )
