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
            str(CASE),
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
        [CASE],
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


# --- fix round 1 (T1.11e review) ----------------------------------------------


@pytest.mark.parametrize(
    'fault',
    [
        # A Python driver's read without its DuckDB types (fetchall).
        lambda o: _values(o, 1).update(types=None),
        # An Arrow read without its schema.
        lambda o: _statement(o, 3)['results'][0].pop('schema'),
        # A Python driver's Arrow read without its DuckDB types.
        lambda o: _values(o, 3).update(types=None),
    ],
)
def test_missing_type_evidence_fails_never_skips(
    tmp_path: Path, fault: Fault
) -> None:
    report = _compared(tmp_path, fault)
    assert _rules(report) == {'sql.result'}, report.texts
    assert any('recorded no' in text for text in report.texts)


def _rust_read(read: dict[str, Any]) -> dict[str, Any]:
    r"""A reference row read as the pure-Rust host would record the same
    result: read as Arrow, with the Arrow types DuckDB reads its columns
    as, and no DuckDB type names."""
    values = dict(read['values'], types=None)
    schema = [
        {'name': name, 'type': compare.arrow_name(kind)}
        for name, kind in zip(
            read['values']['columns'], read['values']['types'], strict=True
        )
    ]
    return {
        'method': 'query_arrow',
        'rows': read['rows'],
        'schema': schema,
        'values': values,
    }


def test_rust_host_types_are_held_to_the_reference_s_row_reads() -> None:
    r"""The reference reads its catalog as rows, so it has no Arrow schema;
    rust-host has no DuckDB type names. The reference's names, as the
    Arrow types DuckDB reads them as, are held to rust-host's schema: an
    int64 read as int32 fails (the subagent's P3-1 repro)."""
    statement = _statement(_reference(), 9)
    assert statement['results'][0]['method'] == 'fetchall'
    rust = dict(statement, results=[_rust_read(statement['results'][0])])
    assert rust['results'][0]['schema'][1]['type'] == 'int64'

    def failures(ours: dict[str, Any]) -> list[str]:
        report = compare.Report({'sql.result': 'gate'})
        compare._results(
            report, 'at', ours, statement, False, ('rust-host', 'python-ref')
        )
        return report.texts

    assert failures(rust) == []
    narrowed = copy.deepcopy(rust)
    narrowed['results'][0]['schema'][1]['type'] = 'int32'
    assert any('differs from the types' in t for t in failures(narrowed))
    # A type with no Arrow name is never taken as equal.
    unnamed = copy.deepcopy(statement)
    unnamed['results'][0]['values']['types'][1] = 'HUGEINT'
    report = compare.Report({'sql.result': 'gate'})
    compare._results(
        report, 'at', rust, unnamed, False, ('rust-host', 'python-ref')
    )
    assert any('no Arrow name' in t for t in report.texts)


def test_duckdb_type_names_map_to_the_arrow_types_read() -> None:
    r""":func:`compare.arrow_name` against every vector whose columns it
    names: DuckDB's names from the description, pyarrow's from the Arrow
    read of the same statement."""
    vectors = json.loads(
        (ROOT / 'contract/result-vectors.json').read_text(encoding='utf-8')
    )['vectors']
    mapped = 0
    for vector in vectors:
        for kind, field in zip(
            vector['values']['types'], vector['arrow'], strict=True
        ):
            name = compare.arrow_name(kind)
            if name is not None:
                assert name == field['type'], (kind, field)
                mapped += 1
    assert mapped >= 30


def test_payload_zero_keeps_its_sign(tmp_path: Path) -> None:
    def signed(observation: dict[str, Any]) -> None:
        _first_number(_payload(observation))['number'] = '-0.0'

    def unsigned(observation: dict[str, Any]) -> None:
        _first_number(_payload(observation))['number'] = '0.0'

    def respelled(observation: dict[str, Any]) -> None:
        _first_number(_payload(observation))['number'] = '-0e0'

    # The candidate's 0.0 against a reference whose token is -0.0: the
    # reference is the fault's base, so apply -0.0 to both, then 0.0.
    case = json.loads(CASE.read_text())
    case['status'] = 'acceptance'
    cases = tmp_path / 'cases'
    cases.mkdir()
    (cases / CASE.name).write_text(json.dumps(case))
    reference = _reference()
    reference['identity']['inputs']['case'] = run_python.digest(
        (cases / CASE.name).read_bytes()
    )
    signed(reference)

    def compared(fault: Fault) -> Any:
        observation = copy.deepcopy(reference)
        observation['identity'].update(runner='python-old', reference=None)
        fault(observation)
        paths = []
        for name, item in (('ref', reference), ('old', observation)):
            paths.append(tmp_path / f'{name}.jsonl')
            paths[-1].write_text(json.dumps(item) + '\n')
        return compare.compare([cases], paths[0], paths[1:])

    assert _rules(compared(unsigned)) == {'payload'}
    # -0e0 is -0.0 spelled otherwise: equal.
    assert compared(respelled).failures == []
    assert compared(signed).failures == []


def test_the_policy_s_own_exclusions_are_applied(tmp_path: Path) -> None:
    def another_id(observation: dict[str, Any]) -> None:
        _payload(observation)['metadata']['request_id'] = 'sdm-000000000000'

    # Under the repository's policy, E35's exclusion leaves the value out.
    assert _compared(tmp_path, another_id).failures == []
    # Under a policy with no exclusion, the same change differs.
    none = _policy_with(
        tmp_path,
        lambda t: (
            t[: t.index('exclusions:')]
            + 'exclusions: []\n\n'
            + t[t.index('# Statements whose rows') :]
        ),
    )
    assert capture_check.policy(none)['exclusions'] == []
    report = _compared(tmp_path, another_id, policy=none)
    assert _rules(report) == {'payload'}, report.texts


@pytest.mark.parametrize(
    'change',
    [
        # A strict prefix of the path E35 approves: all of metadata.
        lambda t: t.replace('path: [metadata, request_id]', 'path: [metadata]'),
        # A decision that is no row of decisions.md.
        lambda t: t.replace('decision: E35', 'decision: E99').replace(
            'path: [metadata, request_id]', 'path: [context]'
        ),
        # No decision.
        lambda t: t.replace('decision: E35', "decision: ''"),
        # A row that does not name the path.
        lambda t: t.replace('path: [metadata, request_id]', 'path: [context]'),
        # A row whose status is not the exclusion's.
        lambda t: t.replace('2026-10-10', '2099-01-01'),
    ],
)
def test_an_exclusion_the_owner_did_not_approve_is_refused(
    tmp_path: Path, change: Callable[[str], str]
) -> None:
    path = _policy_with(tmp_path, change)
    assert path.read_text() != POLICY.read_text()
    with pytest.raises(capture_check.PolicyError):
        capture_check.policy(path)


def test_an_arrow_read_without_a_description_is_incomplete() -> None:
    import pyarrow as pa

    class Undescribed:
        def execute(self, query: str, parameters: Any = None) -> Any:
            return self

        def to_arrow_table(self) -> Any:
            return pa.table({'a': [1]})

    log = run_python.DriverLog()
    proxy = run_python.capture(Undescribed(), log)
    proxy.execute('SELECT 1 AS a').to_arrow_table()
    assert log.incomplete == ['to_arrow_table:no_description']


# --- fix round 2 (T1.11e review) ----------------------------------------------


def test_a_core_arrow_read_is_held_to_the_reference_s_row_types() -> None:
    r"""python-core reads the catalog as Arrow and carries DuckDB's names
    too; its schema is still held to the reference row read's names as the
    Arrow types DuckDB reads them as: a string read as large_string fails
    (the subagent's P3-A repro)."""
    statement = _statement(_reference(), 0)
    assert statement['results'][0]['method'] == 'fetchall'
    core = copy.deepcopy(statement)
    read = _rust_read(statement['results'][0])
    read.update(
        method='to_arrow_table',
        values=dict(
            read['values'], types=statement['results'][0]['values']['types']
        ),
    )
    core['results'] = [read]

    def failures(ours: dict[str, Any]) -> list[str]:
        report = compare.Report({'sql.result': 'gate'})
        compare._results(
            report, 'at', ours, statement, False, ('python-core', 'python-ref')
        )
        return report.texts

    assert read['schema'][0]['type'] == 'string'
    assert failures(core) == []
    widened = copy.deepcopy(core)
    widened['results'][0]['schema'][0]['type'] = 'large_string'
    assert any('differs from the types' in t for t in failures(widened))


def test_an_owner_approved_row_approves_no_other_exclusion(
    tmp_path: Path,
) -> None:
    r"""E2 is owner-approved and names `predict`, but approves no
    exclusion; the loader's allow-list refuses it (the subagent's P3-B
    repro)."""
    extra = (
        '  - id: predict\n'
        '    decision: E2\n'
        '    status: approved by owner 2026-10-08\n'
        '    in: payload\n'
        '    path: [predict]\n'
        '    keeps: [presence, type]\n'
    )
    path = _policy_with(
        tmp_path,
        lambda t: t.replace(
            '    keeps: [presence, type]\n',
            '    keeps: [presence, type]\n' + extra,
            1,
        ),
    )
    assert 'decision: E2' in path.read_text()
    with pytest.raises(capture_check.PolicyError, match='approves no'):
        capture_check.policy(path)
