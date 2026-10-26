# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The tooling train's own checks (T3.2), the implementer's; the
acceptance tests are the test author's.

- the runner ``java`` in the ratchet, and a captures-only commit standing
  for the candidate it was recorded from (the stand-in's ``live_only``
  checks went with the stand-in, B-lite, 2026-10-23);
- the owner's approval read by allowlist (U4), and an entry approved by
  the owner naming a row the owner approved;
- coverage: a leg claims its own path, a named client test may claim one.
"""

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
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COMPARE = _module(ROOT / 'scripts/rust_core/compare.py', 't32_compare')
RATCHET = _module(ROOT / 'scripts/rust_core/ratchet.py', 't32_ratchet')
REGISTER = _module(ROOT / 'scripts/rust_core/register.py', 't32_register')
COVERAGE = _module(ROOT / 'scripts/rust_core/coverage.py', 't32_coverage')
CHECK = _module(ROOT / 'contract/runners/capture_check.py', 't32_capture_check')


# --- the ratchet and the locked run -----------------------------------------------------


def test_java_is_a_ratchet_runner_on_every_leg() -> None:
    assert 'java' in RATCHET.RUNNERS and 'java' in RATCHET.DATABASE_RUNNERS


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env={
            'GIT_AUTHOR_NAME': 't',
            'GIT_AUTHOR_EMAIL': 't@example.invalid',
            'GIT_COMMITTER_NAME': 't',
            'GIT_COMMITTER_EMAIL': 't@example.invalid',
            'GIT_CONFIG_GLOBAL': '/dev/null',
            'GIT_CONFIG_NOSYSTEM': '1',
            'PATH': '/usr/bin:/bin',
        },
    ).stdout.strip()


def _manifest(path: Path, commit: str, verdict: str | None) -> Path:
    path.write_text(
        json.dumps(
            {
                'format': 'sdm-capture/1',
                'recording_commit': commit,
                'verdicts': {}
                if verdict is None
                else {'c@databricks-sql': verdict},
            }
        )
    )
    return path


def test_a_captures_only_commit_stands_for_the_candidate_it_was_recorded_from(
    tmp_path: Path,
) -> None:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'main')
    (repo / 'README').write_text('x\n')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-q', '-m', 'final candidate')
    final = _git(repo, 'rev-parse', 'HEAD')
    captures = repo / 'contract' / 'captures'
    captures.mkdir(parents=True)
    _manifest(captures / 'e1.json', final, 'equal')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-q', '-m', 'captures')
    assert final in CHECK.accepted_recordings(repo)
    (repo / 'README').write_text('y\n')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-q', '-m', 'a product change after the run')
    assert final not in CHECK.accepted_recordings(repo)


# --- the owner's approval -----------------------------------------------------------------


@pytest.mark.parametrize(
    ('status', 'approved'),
    [
        ('approved by owner 2026-10-22', True),
        ('approved by owner 2026-10-22 (T3.2)', True),
        ('approved by owner 2026-10-22 (relayed by the orchestrator)', True),
        ('approved by owner 2026-10-22 (with KO3-17)', True),
        ('approved by owner 2026-10-22 (declined)', False),
        ('approved by owner 2026-10-22, pending', False),
        (
            'approved by owner 2026-10-22 (proposed by T3.2, not yet answered)',
            False,
        ),
        ('approved by owner 2026-10-22 (if the tests pass)', False),
        ('approved by owner 2026-10-22 (maybe)', False),
        ('not approved by owner 2026-10-22', False),
    ],
)
def test_the_owner_s_approval_is_read_by_allowlist(
    status: str, approved: bool
) -> None:
    assert RATCHET.owner_approved(status) is approved


def test_every_approval_the_decisions_files_hold_still_reads_as_one() -> None:
    r"""The allowlist takes no approval the denylist took: every row whose
    status the forms before T3.2 read as approved still is."""
    for path in sorted(
        (ROOT / 'docs/design/rust-core').glob('phase-*/decisions.md')
    ):
        for row, (_, status) in CHECK.decision_rows(path.read_text()).items():
            if RATCHET.OWNER.fullmatch(status.strip()) and not (
                RATCHET.WITHDRAWN.search(status)
                or RATCHET.NOT_YET.search(status)
            ):
                assert RATCHET.owner_approved(status), (path.parent.name, row)


def _entry(row: str) -> dict[str, Any]:
    return {
        'id': 'P3A-99',
        'status': 'approved',
        'approved_by': 'owner',
        'decision_ref': {
            'file': 'docs/design/rust-core/phase-3/decisions.md',
            'row': row,
        },
    }


def test_an_owner_entry_names_a_row_the_owner_approved() -> None:
    assert REGISTER.owner_row_problem(_entry('KO3-45'), ROOT) is None
    problem = REGISTER.owner_row_problem(_entry('KO3-999'), ROOT)
    assert problem is not None and 'no row the owner approved' in problem
    missing = copy.deepcopy(_entry('KO3-45'))
    missing['decision_ref'] = None
    assert 'no decision_ref' in (
        REGISTER.owner_row_problem(missing, ROOT) or ''
    )


def test_a_warehouse_line_key_is_a_corpus_line() -> None:
    assert REGISTER.ON_DATABASE.fullmatch('base.binary@databricks-spark')
    assert not REGISTER.ON_DATABASE.fullmatch('base.binary@bigquery')


# --- coverage ---------------------------------------------------------------------------------


def test_a_leg_claims_its_own_path() -> None:
    case = json.loads((ROOT / 'contract/cases/s1/base.binary.json').read_text())
    claimed = COVERAGE.claims(case)['paths']
    for path in case['paths']:
        if path.endswith('/duckdb'):
            head = path.rpartition('/')[0]
            for leg in ('sqlite', 'postgres', 'databricks-sql', 'lakebase'):
                assert f'{head}/{leg}' in claimed
    s2 = {
        **case,
        'slice': 'S2',
        'warehouses': ['snowflake'],
        'paths': ['schema.from-source/all-tables/snowflake'],
    }
    assert COVERAGE.claims(s2)['paths'] == {
        'schema.from-source/all-tables/snowflake',
        'source.open/explicit/snowflake',
    }


# --- the orchestration fixes the dry run found -------------------------------------------


def test_a_stopped_handoff_runs_its_cleanup(tmp_path: Path) -> None:
    r"""A handoff stopped by SIGTERM leaves through its ``finally`` blocks,
    so the previous round's worktree it made is removed and pruned."""
    script = tmp_path / 'stopped.py'
    marker = tmp_path / 'cleaned'
    script.write_text(
        'import os, signal, sys\n'
        f'sys.path.insert(0, {str(ROOT / "scripts/rust_core")!r})\n'
        'import handoff\n'
        'handoff.finish_on_signals()\n'
        'try:\n'
        '    os.kill(os.getpid(), signal.SIGTERM)\n'
        '    signal.pause()\n'
        'finally:\n'
        f'    open({str(marker)!r}, "w").close()\n'
    )
    done = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert marker.exists(), done.stderr
    assert 'stopped by signal' in done.stderr


def test_clean_tmp_keeps_a_directory_a_daemon_listens_in(
    tmp_path: Path,
) -> None:
    r"""A directory holding a listening Unix socket (the Codex daemon's) is
    in use however old its files are; an old one holding none is stale."""
    import os
    import socket
    import tempfile

    # A Unix socket's path holds at most 107 bytes, which a deep pytest
    # temporary directory passes: the root is a short one of /tmp's.
    # Under /tmp itself, whatever TMPDIR names.
    short = tempfile.TemporaryDirectory(prefix='t32-', dir='/tmp')
    root = Path(short.name) / 'r'
    live, dead = root / 'daemon', root / 'old'
    live.mkdir(parents=True)
    dead.mkdir()
    (dead / 'file').write_text('x')
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(live / 's'))
        server.listen(1)
        for path in (live / 's', live, dead / 'file', dead):
            os.utime(path, (0, 0))
        done = subprocess.run(
            [
                'bash',
                str(ROOT / 'scripts/rust_core/orchestration/clean_tmp.sh'),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            stdin=subprocess.DEVNULL,
            env={
                **os.environ,
                'SDM_TMP_ROOT': str(root),
                'SDM_TARGET_ROOT': str(tmp_path / 'none-target'),
                'SDM_CELLS_ROOT': str(tmp_path / 'none-cells'),
                'SDM_SCRATCH': '',
            },
            check=False,
        )
    finally:
        server.close()
        short.cleanup()
    assert f'skip listening {live}' in done.stdout, done.stdout
    assert f'would remove {dead}' in done.stdout, done.stdout


# --- P2-T2.21-1: an excused digest still holds the values -------------------------------


def _public(digest: str, text: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {'columns': [], 'rows': 1, 'digest': digest}
    if text is not None:
        result['text_digest'] = text
    return {
        'public': {'schema': None, 'payloads': [], 'result': result},
        'host': {'warnings': []},
    }


def test_an_excused_result_digest_still_holds_the_rows_as_text() -> None:
    report = COMPARE.Report(
        {
            'result': 'gate',
            'schema': 'gate',
            'payload': 'gate',
            'warnings': 'gate',
        }
    )
    case = {'id': 'k@postgres'}
    report.more_exact = {
        ('k@postgres', 'python-core'): ('public.result/digest',)
    }
    COMPARE._host_fields(
        report,
        case,
        'python-core',
        _public('a' * 64, 'c' * 64),
        _public('b' * 64, 'c' * 64),
    )
    assert report.failures == []
    COMPARE._host_fields(
        report,
        case,
        'python-core',
        _public('a' * 64, 'c' * 64),
        _public('b' * 64, 'd' * 64),
    )
    assert report.failures and 'text_digest' in report.texts[0]
    # A line recorded before the text digest, its typed digest excused:
    # nothing holds its values, so it fails until recorded again (r1, P2-4).
    later = COMPARE.Report({'result': 'gate'})
    later.more_exact = report.more_exact
    COMPARE._host_fields(
        later,
        case,
        'python-core',
        _public('a' * 64, 'c' * 64),
        _public('b' * 64, None),
    )
    assert [rule for rule, _ in later.failures] == ['result']
    assert 'record that line again' in later.texts[0]


def test_the_census_real_data_mode_counts_hits_and_needs_its_control(
    tmp_path: Path,
) -> None:
    r"""Locked check 2's report: each refusal on a real source counted by
    leg, and the run failing when its seeded control is not refused."""
    census = _module(ROOT / 'contract/tools/refusal_census.py', 't32_census')
    cases = tmp_path / 'real'
    cases.mkdir()
    (cases / 'base.binary.json').write_text(
        (ROOT / 'contract/cases/s1/base.binary.json').read_text()
    )
    refused = ROOT / 'contract/cases/s1/decimal.compare-finer-eq.json'
    found = census.real_data(cases, 'duckdb', refused)
    assert found['sources'] == 1 and found['database'] == 'duckdb'
    assert found['control']['outcome']['result'] == 'refusal'
    assert any(refused.stem in hit['cases'] for hit in found['hits'])
    with pytest.raises(census.CensusError, match='the control'):
        census.real_data(
            cases, 'duckdb', ROOT / 'contract/cases/s1/base.binary.json'
        )


# --- Java's lines (found by T3.11) ----------------------------------------------------


def test_a_java_line_names_java_s_version_and_no_duckdb() -> None:
    run = _module(ROOT / 'contract/runners/run_python.py', 't32_own_run_python')
    validate = run.validator('observation')
    reference = ROOT / 'contract/reference'
    [path] = sorted(reference.glob('*.jsonl'))
    line = json.loads(path.read_text().splitlines()[0])
    java = copy.deepcopy(line)
    java['identity'].update(runner='java', reference=None)
    java['identity']['versions'] = {'java': '21.0.4'}
    assert run._invalid(validate, java) is None
    python = copy.deepcopy(line)
    python['identity']['versions'] = {'java': '21.0.4'}
    assert run._invalid(validate, python) is not None


def test_java_like_rust_host_observes_no_stage_or_class() -> None:
    expect = {
        'outcome': 'refusal',
        'stage': 'from_source',
        'error': 'X',
        'code': 'C',
    }
    seen = {'result': 'refusal', 'stage': None, 'error': None, 'code': 'C'}
    for who, failing in (
        ('java', False),
        ('rust-host', False),
        ('python-core', True),
    ):
        report = COMPARE.Report({'outcome': 'gate'})
        COMPARE._expectation(report, {'id': 'c', 'expect': expect}, who, seen)
        assert bool(report.failures) is failing, (who, report.texts)
    assert RATCHET.UNOBSERVED['java'] == RATCHET.UNOBSERVED['rust-host']


# --- the full dry run's findings (fix 1) ------------------------------------------------


DRYRUN = _module(ROOT / 'scripts/rust_core/gate_dryrun.py', 't32_own_dryrun')


def _kind(case_id: str) -> Any:
    return DRYRUN.Kind(name=case_id, case={'id': case_id})


def test_the_dry_run_reads_failure_words_outside_a_kind_s_id() -> None:
    refused = _kind('dryrun.endpoint.only-call-refused')
    other = _kind('dryrun.switch.edit')
    DRYRUN.attribute(
        'gate',
        'ok   dryrun.endpoint.only-call-refused judged\n'
        'FAILS dryrun.switch.edit: a missing column\n',
        [refused, other],
        True,
    )
    assert refused.gaps == []
    assert other.gaps == ['gate: FAILS dryrun.switch.edit: a missing column']


def test_a_stub_s_missing_reference_line_is_noted_not_a_gap() -> None:
    kind = _kind('dryrun.switch.edit')
    DRYRUN.attribute(
        'coverage',
        'rule 6: case dryrun.switch.edit: claims int64/primary-key with no '
        'reference observation of the case to hold them to\n'
        'DIFFERS dryrun.switch.edit: no reference observation\n',
        [kind],
        True,
    )
    assert kind.gaps == [] and len(kind.notes) == 2


def test_the_ratchet_names_a_runner_that_wrote_no_observations(
    tmp_path: Path,
) -> None:
    with pytest.raises(RATCHET.Unreadable, match='no observations written'):
        RATCHET.observations([tmp_path / 'python-core.jsonl'])


def test_a_refused_case_names_the_field_the_format_does_not_have() -> None:
    run = _module(ROOT / 'contract/runners/run_python.py', 't32_own_run_python')
    case = json.loads((ROOT / 'contract/cases/s1/base.binary.json').read_text())
    case['input']['pickle'] = 'request'
    problem = run._invalid(run.validator('case'), case)
    assert problem is not None and 'pickle not a field there' in problem


def test_the_report_list_s_forms_run(tmp_path: Path) -> None:
    for command in (
        [
            str(ROOT / 'contract/tools/refusal_census.py'),
            '--real-data',
            '--leg',
            'sqlite',
            '--help',
        ],
        [str(ROOT / 'scripts/rust_core/coverage.py'), '--gate', 'S2', '--help'],
    ):
        done = subprocess.run(
            [sys.executable, *command],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
            check=False,
        )
        assert done.returncode == 0, done.stdout + done.stderr


GUARD = _module(
    ROOT / 'scripts/rust_core/orchestration/guard_cmd.py', 't32_own_guard'
)


@pytest.mark.parametrize(
    'command',
    [
        'git -c core.hooksPath=/dev/null commit -m x',
        'X=1 git -c core.hooksPath=/dev/null commit -q --allow-empty -m x',
        'git --config-env=core.hooksPath=HP commit -m x',
        'git --config-env core.hooksPath=HP commit -m x',
        "GIT_CONFIG_PARAMETERS=\"'core.hooksPath'='/dev/null'\" git commit -m x",
        'GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.hooksPath GIT_CONFIG_VALUE_0=/dev/null git push',
        'env GIT_CONFIG_KEY_0=core.hooksPath git commit -m x',
    ],
)
def test_the_guard_refuses_every_way_to_set_the_hooks_path(
    command: str,
) -> None:
    verdict = GUARD.check_command(command, '.', None)
    assert verdict is not None and 'hook' in verdict.rule


def test_the_guard_lets_other_git_configuration_through() -> None:
    assert (
        GUARD.check_command('git -c user.name=x commit -m y', '.', None) is None
    )


def test_a_leg_s_source_open_claim_follows_how_the_case_opens_it() -> None:
    case = json.loads((ROOT / 'contract/cases/s1/base.binary.json').read_text())
    s2 = {
        **case,
        'slice': 'S2',
        'warehouses': ['snowflake'],
        'paths': ['schema.from-source/all-tables/snowflake'],
    }
    assert 'source.open/explicit/snowflake' in COVERAGE.claims(s2)['paths']
    connected = {**s2, 'input': {**s2['input'], 'connect': 'snowpark'}}
    paths = COVERAGE.claims(connected)['paths']
    assert 'source.open/auto/snowflake' in paths
    assert 'source.open/explicit/snowflake' not in paths
    assert 'source.open/explicit/duckdb' in COVERAGE.claims(case)['paths']


def test_an_old_reference_line_without_the_text_digest_still_matches() -> None:
    r"""The committed reference lines predate the text digest: a core line
    holding public.result_text against one without it is compared on the
    rest; two lines that both hold it are held to it (P2-T2.21-1)."""
    case = {'id': 'k'}
    rules = {
        'result': 'gate',
        'schema': 'gate',
        'payload': 'gate',
        'warnings': 'gate',
    }
    old = _public('b' * 64, None)
    new = _public('b' * 64, None)
    new['public']['result_text'] = 'c' * 64
    report = COMPARE.Report(rules)
    COMPARE._host_fields(report, case, 'python-core', new, old)
    assert report.failures == []
    other = _public('b' * 64, None)
    other['public']['result_text'] = 'd' * 64
    report = COMPARE.Report(rules)
    COMPARE._host_fields(report, case, 'python-core', new, other)
    assert report.failures and 'text_digest' in report.texts[0]


# --- the sampled Snowflake case (option C) --------------------------------------------

OPTION_C = '2026-10-22 evening: Snowflake sampling, option C'
SAMPLING = 'SELECT * FROM "orders" SAMPLE ROW (5 ROWS)'


def _sampled_line(sampled_rows: int, result_rows: int) -> dict[str, Any]:
    statements = [
        {
            'stage': 'materialize',
            'sql': 'SELECT 1',
            'parameters': None,
            'error': None,
            'results': [{'rows': 1}],
        },
        {
            'stage': 'materialize',
            'sql': SAMPLING,
            'parameters': None,
            'error': None,
            'results': [{'rows': sampled_rows}],
        },
    ]
    return {
        'driver': {
            'capture': {
                'statements': statements,
                'bindings': [],
                'incomplete': [],
            },
            'query_log': {'statements': 2, 'bindings': 0, 'differing': []},
        },
        'public': {'result': {'rows': result_rows, 'digest': str(result_rows)}},
    }


def test_the_sampled_case_is_held_to_its_sampling_statement_and_row_count(
    tmp_path: Path,
) -> None:
    entry = {
        'leg': 'snowflake',
        'stage': 'materialize',
        'statement': 1,
        'decision': OPTION_C,
    }
    document = yaml.safe_load(
        (ROOT / 'contract/compare-policy.yaml').read_text()
    )
    assert document['sampled'] == {
        'snowflake.sampled.entity-table': {
            **entry,
            'statement': 3,
        }
    }
    path = tmp_path / 'compare-policy.yaml'
    for sampled, ok in (
        ({'s.sampled': entry}, True),
        ({'s.sampled': entry, 's.other': entry}, False),
        ({'s.sampled': {**entry, 'leg': 'databricks-sql'}}, False),
        ({'s.sampled': {**entry, 'decision': 'the implementer'}}, False),
    ):
        path.write_text(yaml.safe_dump({**document, 'sampled': sampled}))
        if ok:
            assert CHECK.policy(path)['sampled'] == sampled
        else:
            with pytest.raises(CHECK.PolicyError, match='sampled'):
                CHECK.policy(path)
    # A case added to the section is a relaxation the ratchet names.
    assert RATCHET.policy_relaxations(
        document, {**document, 'sampled': {'s.sampled': entry}}
    ) == ['sampled s.sampled added']

    rules = dict(CHECK.policy()['rules'])
    reference = _sampled_line(5, 5)
    report = COMPARE.Report(rules)
    report.fail('rows differ as a multiset', 'sql.result')
    report.fail('public.result/digest: differs', 'result')
    report.fail('outcome differs', 'outcome')
    COMPARE.sampled_judgment(
        report, 0, 's.sampled@snowflake', entry,
        {'python-core': _sampled_line(5, 5)}, reference,
    )  # fmt: skip
    # The sample's other rows are reported; the outcome still gates.
    assert report.failures == [('outcome', 'outcome differs')], report.failures
    assert any('rows differ as a multiset' in n for n in report.notes)
    for line, rule in (
        (_sampled_line(4, 5), 'sql.result'),
        (_sampled_line(5, 6), 'result'),
    ):
        report = COMPARE.Report(rules)
        COMPARE.sampled_judgment(
            report, 0, 's.sampled@snowflake', entry, {'python-core': line},
            reference,
        )  # fmt: skip
        assert [r for r, _ in report.failures] == [rule], report.failures


def _sampled_report(line: dict[str, Any], reference: dict[str, Any]) -> Any:
    report = COMPARE.Report(dict(CHECK.policy()['rules']))
    COMPARE.sampled_judgment(
        report,
        0,
        's.sampled@snowflake',
        {
            'leg': 'snowflake',
            'stage': 'materialize',
            'statement': 1,
            'decision': OPTION_C,
        },
        {'python-core': line},
        reference,
    )
    return report


def test_a_sample_read_is_matched_after_freshness_queries_in_the_same_call() -> (
    None
):
    reference = _sampled_line(5, 5)
    line = _sampled_line(5, 5)
    statements = line['driver']['capture']['statements']
    statements.insert(1, copy.deepcopy(statements[0]))
    line['driver']['query_log']['statements'] = len(statements)
    statements[2]['sql'] = '\n  ' + SAMPLING + '  \n'
    assert not _sampled_report(line, reference).failures


@pytest.mark.parametrize(
    'fault',
    ['missing', 'ambiguous', 'other-stage', 'query', 'parameters', 'failed'],
)
def test_the_sample_match_refuses_missing_ambiguous_or_changed_reads(
    fault: str,
) -> None:
    reference = _sampled_line(5, 5)
    line = _sampled_line(5, 5)
    statements = line['driver']['capture']['statements']
    if fault == 'missing':
        statements.pop()
    elif fault == 'ambiguous':
        statements.append(copy.deepcopy(statements[1]))
    elif fault == 'other-stage':
        statements[1]['stage'] = 'from_source'
    elif fault == 'query':
        statements[1]['sql'] = SAMPLING.replace('5 ROWS', '6 ROWS')
    elif fault == 'parameters':
        statements[1]['parameters'] = [{'type': 'builtins.int', 'value': '5'}]
    else:
        statements[1]['error'] = 'ProgrammingError'
    line['driver']['query_log']['statements'] = len(statements)
    assert any(
        rule == 'sql.result'
        for rule, _ in _sampled_report(line, reference).failures
    )


def test_sample_query_matching_preserves_literal_whitespace_and_parameter_types() -> (
    None
):
    reference = _sampled_line(5, 5)
    line = _sampled_line(5, 5)
    for item in (line, reference):
        statement = item['driver']['capture']['statements'][1]
        statement['sql'] += " WHERE 'a b' = ?"
        statement['parameters'] = [{'type': 'builtins.str', 'value': "'a b'"}]
    assert not _sampled_report(line, reference).failures
    changed = copy.deepcopy(line)
    changed['driver']['capture']['statements'][1]['sql'] = changed['driver'][
        'capture'
    ]['statements'][1]['sql'].replace("'a b'", "'ab'")
    assert _sampled_report(changed, reference).failures
    changed = copy.deepcopy(line)
    changed['driver']['capture']['statements'][1]['parameters'] = [
        {'type': 'builtins.bytes', 'value': "b'a b'"}
    ]
    assert _sampled_report(changed, reference).failures


@pytest.mark.parametrize('rows', [None, True, -1])
def test_sample_counts_need_actual_nonnegative_integer_evidence(
    rows: Any,
) -> None:
    reference = _sampled_line(5, 5)
    line = _sampled_line(5, 5)
    for item in (line, reference):
        item['driver']['capture']['statements'][1]['results'][0]['rows'] = rows
    assert _sampled_report(line, reference).failures


@pytest.mark.parametrize(
    'fault', ['capture-off', 'incomplete', 'query-log', 'missing-result']
)
def test_sampled_input_completeness_and_public_result_counts_stay_strict(
    fault: str,
) -> None:
    reference = _sampled_line(5, 5)
    line = _sampled_line(5, 5)
    if fault == 'capture-off':
        line['driver']['capture'] = None
    elif fault == 'incomplete':
        line['driver']['capture']['incomplete'] = ['an unrecorded read']
    elif fault == 'query-log':
        line['driver']['query_log'] = None
    else:
        line['public']['result'] = None
    assert _sampled_report(line, reference).failures


@pytest.mark.parametrize(
    'rule', ['outcome', 'route', 'inputs', 'reach', 'schema', 'warnings']
)
def test_all_original_sampled_case_semantic_rules_keep_gating(
    rule: str,
) -> None:
    report = COMPARE.Report(dict(CHECK.policy()['rules']))
    report.fail('retained semantic difference', rule)
    COMPARE.sampled_judgment(
        report,
        0,
        's.sampled@snowflake',
        {
            'leg': 'snowflake',
            'stage': 'materialize',
            'statement': 1,
            'decision': OPTION_C,
        },
        {'python-core': _sampled_line(5, 5)},
        _sampled_line(5, 5),
    )
    assert (rule, 'retained semantic difference') in report.failures


def test_the_test_author_template_names_the_gate_module_pattern() -> None:
    # Two test authors held pending tests with xfail and skip, which
    # l0_scan.py refuses under contract/ (2026-10-22).
    text = (
        ROOT / 'docs/design/rust-core/orchestration/briefs/test-author.md'
    ).read_text(encoding='utf-8')
    assert 'contract/tests/t132g_gate.py' in text
    assert 'held(group, name, check)' in text
    assert 'as the orchestrator says' not in text and 'strict xfail' not in text
