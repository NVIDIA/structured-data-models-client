# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The tooling train's own checks (T3.2), the implementer's; the
acceptance tests are the test author's.

- the ``live_only`` outcome: the policy's ``standin`` section held to the
  owner's approval, a live-only mark with no stand-in reason refused, a
  flagged case passing when it matches and live only when it differs, the
  ratchet leaving a live-only line to E-1, and the locked run's verdict
  check;
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
CODES = ('B1', 'B2', 'B4')


# --- the policy's stand-in section ----------------------------------------------------


def _policy(tmp_path: Path, standin: Any) -> Path:
    document = yaml.safe_load(
        (ROOT / 'contract/compare-policy.yaml').read_text()
    )
    document['standin'] = standin
    path = tmp_path / 'compare-policy.yaml'
    path.write_text(yaml.safe_dump(document))
    return path


def test_the_policy_holds_the_owner_s_stand_in_codes_and_rules() -> None:
    policy = CHECK.policy()
    assert policy['standin'] == {
        'codes': ['B1', 'B2', 'B4'],
        'diagnostic': ['sql.count', 'sql.result'],
    }


@pytest.mark.parametrize(
    'standin',
    [
        {'codes': ['B1', 'B2', 'B3', 'B4'], 'diagnostic': ['sql.count']},
        {'codes': ['B1', 'B2', 'B4'], 'diagnostic': ['outcome']},
        {'codes': ['B1', 'B2', 'B4']},
        None,
    ],
)
def test_a_stand_in_section_past_the_owner_s_approval_is_refused(
    tmp_path: Path, standin: Any
) -> None:
    with pytest.raises(CHECK.PolicyError, match='standin'):
        CHECK.policy(_policy(tmp_path, standin))


def test_a_stand_in_code_or_rule_added_is_a_relaxation() -> None:
    old = {'standin': {'codes': ['B1'], 'diagnostic': ['sql.count']}}
    new = {
        'standin': {
            'codes': ['B1', 'B2'],
            'diagnostic': ['sql.count', 'sql.result'],
        }
    }
    assert RATCHET.policy_relaxations(old, new) == [
        'standin code B2 added',
        'standin diagnostic rule sql.result added',
    ]
    assert RATCHET.policy_relaxations(new, old) == []


# --- the comparator's live_only -------------------------------------------------------


def _verdict(live_only: str | None, flagged: list[str]) -> dict[str, Any]:
    return {'standin': {'live_only': live_only, 'flagged': flagged}}


def test_a_live_only_mark_with_no_stand_in_reason_fails() -> None:
    report = COMPARE.Report()
    code, flagged = COMPARE._standin(
        report,
        'c@databricks-sql',
        'python-core',
        _verdict('B9', []),
        CODES,
        True,
    )
    assert code is None and report.failures
    assert 'no stand-in reason' in report.texts[0]


def test_a_stand_in_leg_without_the_verdict_fails_and_elsewhere_one_fails() -> (
    None
):
    report = COMPARE.Report()
    COMPARE._standin(report, 'c@snowflake', 'rust-host', {}, CODES, True)
    COMPARE._standin(
        report, 'c', 'python-core', _verdict(None, []), CODES, False
    )
    assert len(report.failures) == 2


def test_a_listed_code_makes_the_case_live_only_and_keeps_runner_failures() -> (
    None
):
    report = COMPARE.Report({'outcome': 'gate'})
    code, _ = COMPARE._standin(
        report, 'c@lakebase', 'python-core', _verdict('B1', []), CODES, True
    )
    assert code == 'B1'
    report.fail('the outcome differs', 'outcome')
    report.fail('runner failure: no observation')
    COMPARE._live_only(report, 'c@lakebase', 'python-core', code, 0, 'why')
    assert report.live_only == {'c@lakebase': 'B1'}
    assert report.texts == ['runner failure: no observation']


def test_a_flagged_answer_passes_when_equal_and_is_live_only_when_not() -> None:
    report = COMPARE.Report({'outcome': 'gate'})
    _, flagged = COMPARE._standin(
        report, 'c@snowflake', 'rust-host', _verdict(None, ['B4']), CODES, True
    )
    assert flagged == ('B4',) and not COMPARE._differences(report, 0)
    report.fail('the outcome differs', 'outcome')
    assert COMPARE._differences(report, 0)


# --- the ratchet and the locked run -----------------------------------------------------


def test_the_ratchet_leaves_a_live_only_line_to_e1() -> None:
    line = _verdict('B2', [])
    assert RATCHET.live_only(line, CODES) == 'B2'
    assert RATCHET.live_only(line, ()) is None
    assert RATCHET.live_only(_verdict(None, ['B4']), CODES) is None
    assert RATCHET.standin_codes(CHECK.policy()) == CODES
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


def test_a_live_only_item_needs_its_e1_verdict_on_the_final_candidate(
    tmp_path: Path,
) -> None:
    final = 'f' * 40
    live = {'c@databricks-sql': 'B1'}
    none = _manifest(tmp_path / 'none.json', final, None)
    assert 'no E-1 verdict' in COMPARE.verdict_problems(live, [none], final)[0]
    other = _manifest(tmp_path / 'other.json', '0' * 40, 'equal')
    assert 'another commit' in COMPARE.verdict_problems(live, [other], final)[0]
    differs = _manifest(tmp_path / 'differs.json', final, 'differs')
    assert 'differs' in COMPARE.verdict_problems(live, [differs], final)[0]
    equal = _manifest(tmp_path / 'equal.json', final, 'equal')
    assert COMPARE.verdict_problems(live, [equal], final) == []
    deep = tmp_path / 'deep.json'
    deep.write_text('[' * 200_000 + ']' * 200_000)
    assert (
        'unreadable' in COMPARE.verdict_problems(live, [deep, equal], final)[0]
    )


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

    root = tmp_path / 'tmp-root'
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
    # A line recorded before the text digest: named, not compared.
    later = COMPARE.Report({'result': 'gate'})
    later.more_exact = report.more_exact
    COMPARE._host_fields(
        later,
        case,
        'python-core',
        _public('a' * 64, 'c' * 64),
        _public('b' * 64, None),
    )
    assert later.failures == [] and any(
        'record it again' in n for n in later.notes
    )


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
            'snowflake',
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
