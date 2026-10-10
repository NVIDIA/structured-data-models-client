# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The comparator's payload and result rules (T1.11e).

Run where the contract tests run (CI's ``engine`` job until T1.18):

    python -m pytest contract/tests -q

- H9 measured: on ``base.regression`` the reference, ``python-core`` and
  ``python-old`` each run 10 statements in ``Schema.from_source``, 6 in
  ``materialize`` and none in ``predict``, and every statement's result is
  the reference's (``sql.count`` and ``sql.result``); field by field, the
  case differs from the reference only in the statements' raw text and the
  relations bound, which H9 leaves diagnostic. ``rust-host``'s are held to
  the same by ``sdm-testkit``'s ``tests/duckdb_provider.rs`` and by
  ``compare.py`` in ``rust-test``.
- On ``python-core``, capture on and off give the same public projection.
- Each rule fails on a seeded fault of its own, under an acceptance copy of
  the case, and a diagnostic rule only reports; in the diagnostic case a
  difference is reported, but not reaching the result fails (``reach``).
- The policy: the one exclusion is E35's, owner-approved and value only;
  an unapproved, widened or unknown exclusion, an unknown rule or a missing
  one is refused; each rule is the gate or the diagnostic the design says.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / 'contract/runners'
CASES = ROOT / 'contract/cases'
CASE = CASES / 's1/base.regression.json'
REFERENCE = next((ROOT / 'contract/reference').glob('*.jsonl'))
POLICY = ROOT / 'contract/compare-policy.yaml'
#: H9's measured counts on base.regression, per public call (T1.11e).
COUNTS = {'from_source': 10, 'materialize': 6, 'predict': 0}


def _module(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_python = _module('run_python', RUNNERS / 'run_python.py')
capture_check = _module('capture_check', RUNNERS / 'capture_check.py')
compare = _module('compare', ROOT / 'scripts/rust_core/compare.py')


def _reference() -> dict[str, Any]:
    for line in REFERENCE.read_text().splitlines():
        observation = json.loads(line)
        if observation['case'] == 'base.regression':
            return observation
    raise AssertionError('no reference observation of base.regression')


def _run(mode: str, out: Path, *extra: str) -> list[dict[str, Any]]:
    run = subprocess.run(
        [
            sys.executable,
            '-I',
            str(RUNNERS / 'run_python.py'),
            '--mode',
            mode,
            *extra,
            '--out',
            str(out),
            str(CASE.parent),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    return [json.loads(line) for line in out.read_text().splitlines()]


@pytest.fixture(scope='module')
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    r"""base.regression on python-core (with and without the capture) and
    python-old."""
    where = tmp_path_factory.mktemp('runs')
    paths = {
        'python-core': where / 'python-core.jsonl',
        'python-core-off': where / 'python-core-off.jsonl',
        'python-old': where / 'python-old.jsonl',
    }
    _run('python-core', paths['python-core'])
    _run('python-core', paths['python-core-off'], '--no-capture')
    _run('python-old', paths['python-old'])
    return paths


# --- the measurement ----------------------------------------------------------


def test_h9_holds_on_base_regression_on_the_python_runners(
    runs: dict[str, Path], tmp_path: Path
) -> None:
    reference = tmp_path / 'python-ref.jsonl'
    reference.write_text(json.dumps(_reference()) + '\n', encoding='utf-8')
    report = compare.compare(
        [CASE.parent],
        reference,
        [runs['python-core'], runs['python-old']],
    )
    assert report.failures == [], report.texts
    measured = {
        line.split(': ')[1]: line.split(': ', 2)[2]
        for line in report.info
        if line.startswith('base.regression: ')
    }
    assert measured == {
        who: str(COUNTS) for who in ('python-ref', 'python-core', 'python-old')
    }
    # Field by field, the case differs only where H9 leaves it diagnostic:
    # the statements' raw text (execution-scoped key-set names) and the
    # relations bound (the core's seeds under role names, in another
    # column order); every statement's result is the reference's.
    differing = {
        note.split(':', 1)[0]
        for note in report.notes
        if 'base.regression' in note
    }
    assert differing <= {
        'diagnostic rule sql.text',
        'diagnostic rule sql.bindings',
    }, report.notes
    # Not vacuous: every statement had its result compared, the reference's
    # first among them.
    reference = _reference()
    statements = reference['driver']['capture']['statements']
    assert len(statements) == sum(COUNTS.values())
    assert all(len(s['results']) == 1 for s in statements)
    assert all(not s['results'][0]['values']['unencodable'] for s in statements)


def test_capture_on_and_off_agree_on_python_core(runs: dict[str, Path]) -> None:
    on = [json.loads(x) for x in runs['python-core'].read_text().splitlines()]
    off = [
        json.loads(x) for x in runs['python-core-off'].read_text().splitlines()
    ]
    assert capture_check.check(on, off, ['base.regression']) == []
    [observation] = on
    assert observation['identity']['operations'] == [
        'describe_source',
        'materialize',
        'predict',
    ]
    assert compare.counts(observation) == COUNTS


# --- seeded faults --------------------------------------------------------------


Fault = Callable[[dict[str, Any]], None]


def _compared(
    tmp_path: Path,
    fault: Fault,
    *,
    status: str = 'acceptance',
    who: str = 'python-old',
    policy: Path | None = None,
) -> Any:
    r"""The reference's own observation of base.regression, as ``who``
    observed it but for ``fault``, compared under a copy of the case with
    ``status``."""
    case = json.loads(CASE.read_text())
    case['status'] = status
    cases = tmp_path / 'cases'
    cases.mkdir(exist_ok=True)
    written = cases / CASE.name
    written.write_text(json.dumps(case), encoding='utf-8')
    digest = run_python.digest(written.read_bytes())
    reference = _reference()
    reference['identity']['inputs']['case'] = digest
    observation = copy.deepcopy(reference)
    observation['identity']['runner'] = who
    observation['identity']['reference'] = None
    fault(observation)
    files = []
    for name, item in (('reference', reference), (who, observation)):
        path = tmp_path / f'{name}.jsonl'
        path.write_text(json.dumps(item) + '\n', encoding='utf-8')
        files.append(path)
    return compare.compare([cases], files[0], files[1:], policy)


def _rules(report: Any) -> set[str | None]:
    return {rule for rule, _ in report.failures}


def _statement(observation: dict[str, Any], position: int) -> dict[str, Any]:
    statement: dict[str, Any] = observation['driver']['capture']['statements'][
        position
    ]
    return statement


def _payload(observation: dict[str, Any]) -> dict[str, Any]:
    parsed: dict[str, Any] = observation['public']['payloads'][0]['parsed']
    return parsed


def test_the_unchanged_reference_passes(tmp_path: Path) -> None:
    report = _compared(tmp_path, lambda o: None)
    assert report.failures == [], report.texts


def _first_number(value: Any) -> dict[str, str]:
    r"""The first float token of a parsed payload, depth first."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if set(item) == {'number'} and '.' in item['number']:
                return item
            stack.extend(reversed(list(item.values())))
        elif isinstance(item, list):
            stack.extend(reversed(item))
    raise AssertionError('no float token')


def test_payload_numbers_compare_by_kind_and_exact_value(
    tmp_path: Path,
) -> None:
    def respelled(observation: dict[str, Any]) -> None:
        token = _first_number(_payload(observation))
        token['number'] = token['number'] + '0'

    # The same value, spelled with a trailing zero: equal.
    assert _compared(tmp_path, respelled).failures == []

    def changed(observation: dict[str, Any]) -> None:
        token = _first_number(_payload(observation))
        token['number'] = token['number'] + '1'

    assert _rules(_compared(tmp_path, changed)) == {'payload'}

    def integer(observation: dict[str, Any]) -> None:
        _first_number(_payload(observation))['number'] = '1'

    def float_one(observation: dict[str, Any]) -> None:
        _first_number(_payload(observation))['number'] = '1.0'

    # 1 and 1.0 are one value of two kinds: they differ.
    one = _compared(tmp_path, integer)
    assert 'payload' in _rules(one)
    assert _rules(_compared(tmp_path, float_one)) == {'payload'}


def test_only_the_request_id_s_value_is_left_out(tmp_path: Path) -> None:
    def another(observation: dict[str, Any]) -> None:
        _payload(observation)['metadata']['request_id'] = 'sdm-000000000000'

    assert _compared(tmp_path, another).failures == []

    def deleted(observation: dict[str, Any]) -> None:
        del _payload(observation)['metadata']['request_id']

    def retyped(observation: dict[str, Any]) -> None:
        _payload(observation)['metadata']['request_id'] = {'number': '7'}

    def other_field(observation: dict[str, Any]) -> None:
        _payload(observation)['metadata']['source_query'] = 'PREDICT x'

    for fault in (deleted, retyped, other_field):
        assert _rules(_compared(tmp_path, fault)) == {'payload'}, fault


@pytest.mark.parametrize(
    ('rule', 'fault'),
    [
        (
            'result',
            lambda o: o['public']['result'].update(digest='0' * 64),
        ),
        (
            'result',
            lambda o: o['public']['result']['columns'][0].update(
                dtype='object'
            ),
        ),
        (
            'schema',
            lambda o: o['public']['schema']['tables'][0]['columns'][0].update(
                dtype='float64'
            ),
        ),
        ('warnings', lambda o: o['host'].update(warnings=[])),
    ],
)
def test_each_host_field_rule_fails_its_fault(
    tmp_path: Path, rule: str, fault: Fault
) -> None:
    assert _rules(_compared(tmp_path, fault)) == {rule}


def _values(observation: dict[str, Any], position: int) -> dict[str, Any]:
    values: dict[str, Any] = _statement(observation, position)['results'][0][
        'values'
    ]
    return values


def test_sql_count_fails_a_missing_statement(tmp_path: Path) -> None:
    def dropped(observation: dict[str, Any]) -> None:
        statements = observation['driver']['capture']['statements']
        statements.pop(14)
        observation['driver']['query_log']['statements'] -= 1

    report = _compared(tmp_path, dropped)
    assert _rules(report) == {'sql.count'}
    assert any('materialize: 5 statements' in t for t in report.texts)


@pytest.mark.parametrize(
    'fault',
    [
        # Another multiset of rows, an unordered statement (_by_pkey).
        lambda o: _values(o, 14).update(unordered='0' * 64),
        # Another order, the one ordered statement (_by_fkey).
        lambda o: _values(o, 15).update(ordered='0' * 64),
        # A column renamed, a type changed, a row count.
        lambda o: _values(o, 1).update(columns=['x'] * 6),
        lambda o: _values(o, 1).update(
            types=['BIGINT', *_values(o, 1)['types'][1:]]
        ),
        lambda o: _statement(o, 3)['results'][0]['schema'][0].update(
            type='int32'
        ),
        lambda o: _statement(o, 3)['results'][0].update(rows=79),
        # A value the canonical form does not hold is never equal.
        lambda o: _values(o, 0).update(unencodable=['builtins.object']),
        # A statement that failed.
        lambda o: _statement(o, 2).update(error='Error'),
    ],
)
def test_sql_result_fails_each_fault(tmp_path: Path, fault: Fault) -> None:
    assert _rules(_compared(tmp_path, fault)) == {'sql.result'}


def test_order_counts_only_where_the_policy_orders(tmp_path: Path) -> None:
    def reordered(observation: dict[str, Any]) -> None:
        # _by_pkey's rows in another order: its order is not defined.
        _values(observation, 14).update(ordered='0' * 64)

    assert _compared(tmp_path, reordered).failures == []


def test_text_parameters_and_bindings_only_report(tmp_path: Path) -> None:
    def relabelled(observation: dict[str, Any]) -> None:
        _statement(observation, 13)['sql'] = 'SELECT 1'
        _statement(observation, 13)['parameters'] = [
            {'type': 'builtins.int', 'value': '1'}
        ]
        observation['driver']['capture']['bindings'][0]['values'][
            'unordered'
        ] = '0' * 64

    report = _compared(tmp_path, relabelled)
    assert report.failures == []
    rules = {note.split(':', 1)[0] for note in report.notes}
    assert rules == {
        'diagnostic rule sql.text',
        'diagnostic rule sql.parameters',
        'diagnostic rule sql.bindings',
    }


@pytest.mark.parametrize(
    'fault',
    [
        lambda o: o['driver'].update(capture=None),
        lambda o: o['driver']['capture'].update(incomplete=['arrow']),
        lambda o: o['driver']['query_log'].update(statements=15),
    ],
)
def test_no_record_or_an_incomplete_one_fails(
    tmp_path: Path, fault: Fault
) -> None:
    assert _rules(_compared(tmp_path, fault)) == {'sql.count'}


def test_a_vacuous_reference_fails(tmp_path: Path) -> None:
    def emptied(item: dict[str, Any]) -> None:
        item['driver']['capture']['statements'] = []

    # The reference's record emptied, the runner's intact.
    case = json.loads(CASE.read_text())
    case['status'] = 'acceptance'
    (tmp_path / 'cases').mkdir()
    written = tmp_path / 'cases' / CASE.name
    written.write_text(json.dumps(case))
    reference = _reference()
    reference['identity']['inputs']['case'] = run_python.digest(
        written.read_bytes()
    )
    observation = copy.deepcopy(reference)
    observation['identity'].update(runner='python-old', reference=None)
    emptied(reference)
    paths = []
    for name, item in (('ref', reference), ('old', observation)):
        paths.append(tmp_path / f'{name}.jsonl')
        paths[-1].write_text(json.dumps(item) + '\n')
    report = compare.compare([tmp_path / 'cases'], paths[0], paths[1:])
    assert 'sql.count' in _rules(report)


def test_the_diagnostic_case_reports_but_gates_reaching_its_result(
    tmp_path: Path,
) -> None:
    def differs(observation: dict[str, Any]) -> None:
        _values(observation, 14).update(unordered='0' * 64)

    report = _compared(tmp_path, differs, status='diagnostic')
    assert report.failures == []
    assert any(
        note.startswith('diagnostic: sql.result: ') and 'rows differ' in note
        for note in report.notes
    ), report.notes

    def refused(observation: dict[str, Any]) -> None:
        observation['outcome'] = {
            'result': 'refusal',
            'stage': 'predict',
            'error': 'NimRequestError',
            'code': 'INVALID_REQUEST',
        }
        observation['public']['result'] = None

    report = _compared(tmp_path, refused, status='diagnostic')
    assert _rules(report) == {'reach'}

    def no_result(observation: dict[str, Any]) -> None:
        observation['public']['result'] = None

    report = _compared(tmp_path, no_result, status='diagnostic')
    assert _rules(report) == {'reach'}


# --- the policy -------------------------------------------------------------------


#: Each rule as the design classes it (design, "Observations and
#: comparator"; decision H9): only the reference's raw SQL text, typed
#: parameters and bindings are diagnostic.
CLASSES = {
    'outcome': 'gate',
    'route': 'gate',
    'inputs': 'gate',
    'reach': 'gate',
    'schema': 'gate',
    'warnings': 'gate',
    'result': 'gate',
    'payload': 'gate',
    'core': 'gate',
    'sql.count': 'gate',
    'sql.result': 'gate',
    'sql.text': 'diagnostic',
    'sql.parameters': 'diagnostic',
    'sql.bindings': 'diagnostic',
}


def test_the_policy_as_the_design_says() -> None:
    policy = capture_check.policy()
    assert policy['rules'] == CLASSES
    assert set(CLASSES) == set(compare.RULES)
    # The one exclusion, E35's, approved by the owner on 2026-10-10: the
    # value of metadata.request_id in a payload, presence and type kept.
    assert policy['exclusions'] == [
        {
            'id': 'request-id',
            'decision': 'E35',
            'status': 'approved by owner 2026-10-10',
            'in': 'payload',
            'path': ['metadata', 'request_id'],
            'keeps': ['presence', 'type'],
        }
    ]
    assert capture_check.RANDOM_PATHS == (('metadata', 'request_id'),)
    assert policy['ordered'] == {'base.regression': {'materialize': [5]}}
    assert policy['reach'] == {'base.regression': 'predict'}


def _policy_with(tmp_path: Path, change: Callable[[str], str]) -> Path:
    path = tmp_path / 'compare-policy.yaml'
    path.write_text(change(POLICY.read_text()), encoding='utf-8')
    return path


@pytest.mark.parametrize(
    'change',
    [
        # Unapproved, widened past value only, of something else, unknown.
        lambda t: t.replace(
            'status: approved by owner 2026-10-10', 'status: proposed'
        ),
        lambda t: t.replace('keeps: [presence, type]', 'keeps: []'),
        lambda t: t.replace('in: payload', 'in: result'),
        lambda t: t.replace(
            '  sql.bindings: diagnostic', '  sql.bindings: off'
        ),
        lambda t: t.replace('    materialize: [5]', '    sample: [5]'),
    ],
)
def test_a_policy_it_cannot_apply_is_refused(
    tmp_path: Path, change: Callable[[str], str]
) -> None:
    path = _policy_with(tmp_path, change)
    assert path.read_text() != POLICY.read_text()
    with pytest.raises(capture_check.PolicyError):
        capture_check.policy(path)
    report = _compared(tmp_path, lambda o: None, policy=path)
    assert any('compare-policy.yaml' in text for text in report.texts)


@pytest.mark.parametrize(
    'change',
    [
        lambda t: t.replace('  sql.bindings: diagnostic\n', ''),
        lambda t: t.replace(
            '  sql.bindings: diagnostic\n',
            '  sql.bindings: diagnostic\n  sql.other: gate\n',
        ),
    ],
)
def test_the_rules_must_be_the_ones_it_runs(
    tmp_path: Path, change: Callable[[str], str]
) -> None:
    path = _policy_with(tmp_path, change)
    assert path.read_text() != POLICY.read_text()
    report = _compared(tmp_path, lambda o: None, policy=path)
    assert any('its rules are not the ones' in text for text in report.texts)


def test_a_gate_made_diagnostic_stops_gating(tmp_path: Path) -> None:
    # The mechanism the ratchet will read as a relaxation (T1.14): with the
    # rule diagnostic, its fault only reports, which the pinned classes
    # above refuse in this repository.
    path = _policy_with(
        tmp_path,
        lambda t: t.replace('  payload: gate', '  payload: diagnostic'),
    )

    def changed(observation: dict[str, Any]) -> None:
        token = _first_number(_payload(observation))
        token['number'] = token['number'] + '1'

    report = _compared(tmp_path, changed, policy=path)
    assert report.failures == []
    assert any(n.startswith('diagnostic rule payload') for n in report.notes)
