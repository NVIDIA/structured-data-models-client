# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.10's exits on ``python-core`` (the package's test author's): the PQL
target rules ``aggregation``, ``open_window``, ``condition``,
``logical_target``, ``target_filter`` and ``static_target``.

    SDM_POSTGRES_TEST_DSN=... python -m pytest \
        contract/tests/test_t210_targets.py -q

Written from the design's T2.10 row ("label vectors from the reference;
cases per rule") and the package brief, before the implementation.
``sdm-testkit``'s ``tests/t210_targets.rs`` holds ``rust-host`` to the
same cases. Three kinds of evidence, all from the reference (python-ref
at ``contract/reference-env.txt``'s commit):

- the label vectors, ``contract/t210-target-vectors.json``
  (``contract/tools/target_record.py``): 32 scenarios, one target under
  one rule each, on DuckDB, SQLite and Postgres: the outcome, each
  request's task and its context's and predicted rows' instance tables
  (the labels), the rows each call scores and the result; python-core is
  observed by the same code (``observe``);
- the S1 cases, each on all three databases through ``predict``: one or
  two per rule (``target.*``), a ranking over a DECIMAL key
  (``rank.decimal-key``, T2.8's review) and one over a key named with a
  dot (``rank.odd-column``, T2.18's review), and the filter role of
  T2.18's column names (``name.*-filter``, back from their deferral),
  with python-ref's committed lines;
- the filter role's column-name vectors in
  ``contract/column-name-vectors.json`` (T2.18's recorder): a column
  named with a dot, a backtick, a line feed or a carriage return filtering
  the target's aggregation, each with its twin named ``a_b``.

Where the reference refuses only because it cannot say a column's name
(``rank.odd-column``, ``name.*-filter``), the core's answer is the twin's
(KO2-36: column names should not matter), and the case's outcome
difference must be named by an approved register entry citing no
temporary approval, as T2.18's are.

``open_window`` and ``static_target`` are refused by the query validator
on both routes before the rule is reached from the public calls (the
refusal census's probes say so), so their case and vectors are guards:
they pass now and must keep passing.

Gating: ``t210-pending.json``'s ``pending`` names, as ``py:<case key>``
(the case's line against python-ref's), ``py-labels:<scenario>@<database>``
(the label vectors) and ``py-name:<vector id>`` (the filter role's
column-name vectors), the keys whose test failed when written. Each must
still fail (strict); once one passes it fails, naming itself, until it is
taken out. ``pending`` stays within ``authored``. Where
``SDM_POSTGRES_TEST_DSN`` names no server and ``SDM_REQUIRE_POSTGRES`` is
not set, the Postgres keys have nothing to hold and return without
asserting; with it set, a missing server fails (H26).
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
CASES = ROOT / 'contract/cases/s1'
PENDING = Path(__file__).with_name('t210-pending.json')
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'
NAME_VECTORS = ROOT / 'contract/column-name-vectors.json'
DATABASES = ('duckdb', 'sqlite', 'postgres')
#: The register's umbrellas: approved as E92's Phase 2 class, they excuse
#: no case (T1.33).
UMBRELLAS = ('P1B-1', 'P1B-2', 'P1B-3', 'P1B-4')
#: T2.18's filter-role cases, back from contract/deferred/.
NAME_FILTERS = (
    'name.backtick-filter',
    'name.dot-filter',
    'name.line-break-filter',
)
#: The cases the reference refuses only for a column name it cannot say:
#: the core answers them as the reference answers their twins (KO2-36).
NAMED = ('rank.odd-column', *NAME_FILTERS)
#: The guards: the query validator refuses them on both routes first.
GUARDS = ('target.open-window', 'target.static.other-table')
#: The rule of the filter role's vectors (column_names_record.ROLES).
FILTER_ROLE = 'filter'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('t210_run_python', RUNNER)


def recorder() -> ModuleType:
    return _module('target_record', ROOT / 'contract/tools/target_record.py')


def names() -> ModuleType:
    return _module(
        'column_names_record', ROOT / 'contract/tools/column_names_record.py'
    )


R = recorder()
CASE_IDS = (*sorted(R.CASE_OF), *NAME_FILTERS)


def case_key(case_id: str, database: str) -> str:
    return runner().case_key(case_id, database)


CASE_KEYS = tuple(case_key(c, d) for d in DATABASES for c in CASE_IDS)
LABEL_KEYS = tuple(R.key(s, d) for d in DATABASES for s in R.SCENARIOS)


@functools.cache
def name_vectors() -> dict[str, Any]:
    return json.loads(NAME_VECTORS.read_text(encoding='utf-8'))


def _filter_vectors() -> list[dict[str, Any]]:
    return [
        v
        for v in name_vectors()['vectors']
        if v['role'] == FILTER_ROLE and v['character'] is not None
    ]


NAME_IDS = tuple(v['id'] for v in _filter_vectors())


def held(name: str, check: Callable[[], None]) -> None:
    r"""Runs ``check``: it must pass, or, while ``name`` is pending, fail."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    if name not in gating['pending']:
        check()
        return
    try:
        check()
    except Exception:  # noqa: BLE001 - pending: failing, as when written
        return
    raise AssertionError(
        f'{name} passes now: take it out of '
        'contract/tests/t210-pending.json pending'
    )


def _server() -> bool:
    r"""Whether a Postgres server is named; where none is, one must not be
    required (H26)."""
    if os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


def test_the_pending_list_only_shrinks() -> None:
    r"""Control: ``pending`` names only keys the test author wrote, and
    ``authored`` is every case's key on each database on python-core and
    on DuckDB and SQLite on rust-host (its Postgres pass replays
    python-core's fetches, which ``cross-host (postgres)`` holds), every
    label vector's key, and every filter-role column-name vector."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert set(gating['pending']) <= set(gating['authored'])
    assert set(gating['authored']) == (
        {f'py:{k}' for k in CASE_KEYS}
        | {f'rust:{k}' for k in CASE_KEYS if not k.endswith('@postgres')}
        | {f'py-labels:{k}' for k in LABEL_KEYS}
        | {f'py-name:{v}' for v in NAME_IDS}
    )
    assert len(set(gating['authored'])) == len(gating['authored'])


# --- the cases and the vectors are what they name ---------------------------------


@functools.cache
def cases() -> dict[str, tuple[dict[str, Any], str]]:
    r"""T2.10's cases by id, each with its digest, as the runner loads
    them."""
    loaded = runner().load_cases([CASES / f'{c}.json' for c in CASE_IDS])
    return {case['id']: (case, digest) for case, digest in loaded}


def test_each_rule_has_its_cases_and_scenarios() -> None:
    r"""Exit "cases per rule": every rule the core refuses a target under
    has an S1 case and two scenarios or more (one where the validator
    refuses first), each case through ``predict`` on all three databases,
    its input the recorder's scenario of the same name; and the deferred
    filter role is back, its directory gone."""
    assert set(R.RULES) == {
        'aggregation',
        'open_window',
        'condition',
        'logical_target',
        'target_filter',
        'static_target',
    }
    by_rule: dict[str, list[str]] = {rule: [] for rule in R.RULES}
    for scenario_id, spec in R.SCENARIOS.items():
        by_rule[spec['rule']].append(scenario_id)
    cased = {R.SCENARIOS[s]['rule'] for s in R.CASE_OF.values()}
    assert cased == set(R.RULES)
    for rule, scenarios in by_rule.items():
        wanted = 1 if rule in ('open_window', 'static_target') else 2
        assert len(scenarios) >= wanted, rule
    for case_id, scenario_id in R.CASE_OF.items():
        case, _ = cases()[case_id]
        assert case['status'] == 'acceptance' and case['slice'] == 'S1'
        assert case['until'] == 'predict', case_id
        assert runner().case_databases(case) == DATABASES, case_id
        assert case['input'] == R.as_case(scenario_id)['input'], case_id
    assert set(vectors()['vectors']) == set(LABEL_KEYS)
    assert not (ROOT / 'contract/deferred').exists()
    for case_id in NAME_FILTERS:
        case, _ = cases()[case_id]
        assert 'WHERE orders.`' in case['input']['pql'], case_id
        assert runner().case_databases(case) == DATABASES, case_id


def test_the_rankings_are_the_reviews() -> None:
    r"""The two rankings carried here: over a DECIMAL(18,0) key, items'
    and buys' item_id (T2.8's review P3-1), and over a key named
    ``i.d``, declared, with its twin named ``i_d`` (T2.18's review P3-1);
    each with candidates 0, 1 and 2, RANK TOP 2, and the endpoint asked
    for probabilities."""
    decimal, _ = cases()['rank.decimal-key']
    assert all('DECIMAL(18,0)' in s for s in decimal['input']['setup'])
    odd, _ = cases()['rank.odd-column']
    twin = R.SCENARIOS['rank.odd-column.twin']
    assert R.SCENARIOS['rank.odd-column']['twin'] == 'rank.odd-column.twin'
    assert odd['input']['setup'] == R.declared('"i.d"')
    assert twin['setup'] == R.declared('i_d')
    assert odd['input']['pql'].replace('`i.d`', 'i_d') == twin['pql']
    for case in (decimal, odd):
        given = case['input']
        assert ' RANK TOP 2 ' in given['pql']
        assert given['materialize']['candidates'] == [0, 1, 2]
        assert given['probabilities'] is True


@functools.cache
def reference_lines() -> dict[str, dict[str, Any]]:
    r"""python-ref's committed lines, by key."""
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    commit = next(
        line.split(': ', 1)[1].strip()
        for line in env.splitlines()
        if line.startswith('reference: ')
    )
    path = ROOT / f'contract/reference/{commit}.jsonl'
    return {
        json.loads(line)['case']: json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
    }


def _expected(case: dict[str, Any]) -> tuple[Any, ...]:
    expect = case['expect']
    return (
        expect['outcome'],
        expect['stage'],
        expect.get('error'),
        expect.get('code'),
    )


def _outcome(outcome: dict[str, Any]) -> tuple[Any, ...]:
    return (
        outcome['result'],
        outcome['stage'],
        outcome.get('error'),
        outcome.get('code'),
    )


@pytest.mark.parametrize('case_key', CASE_KEYS)
def test_python_ref_s_line_is_at_its_case_file(case_key: str) -> None:
    r"""Exit "cases per rule" on all three databases: python-ref's line
    for the case on the database, recorded at the case file, ends as the
    case expects."""
    case, digest = cases()[case_key.split('@')[0]]
    line = reference_lines().get(case_key)
    assert line is not None, f'{case_key}: no python-ref line'
    assert line['identity']['runner'] == 'python-ref'
    assert line['identity']['inputs']['case'] == digest
    assert _outcome(line['outcome']) == _expected(case)


@functools.cache
def vectors() -> dict[str, Any]:
    return R.load()


def as_json(value: Any) -> Any:
    r"""A request the runner parsed with every number as its text
    (``{'number': <text>}``, ``run_python._payload``) as JSON reads it,
    so it compares with the recorder's."""
    if isinstance(value, dict):
        if set(value) == {'number'} and isinstance(value['number'], str):
            return json.loads(value['number'])
        return {name: as_json(item) for name, item in value.items()}
    if isinstance(value, list):
        return [as_json(item) for item in value]
    return value


def sent_labels(line: dict[str, Any]) -> list[Any]:
    r"""What each request a line's endpoint received says of its labels
    (``target_record.labels``)."""
    return [
        R.labels(as_json(payload['parsed']))
        for payload in line['public']['payloads']
    ]


def _labels(payloads: list[Any] | None) -> list[list[Any]]:
    r"""Each request's context labels (its ``TARGET`` column)."""
    out = []
    for payload in payloads or []:
        context = payload.get('context')
        if context:
            at = context['columns'].index('TARGET')
            out.append([row[at] for row in context['rows']])
    return out


@pytest.mark.parametrize('label_key', LABEL_KEYS)
def test_the_vectors_are_the_reference_s_and_not_vacuous(
    label_key: str,
) -> None:
    r"""Exit "label vectors from the reference": recorded from the
    reference this tree names, by the recorder here, its scenario's rule;
    every scenario the reference answers labels its context with more than
    one value, so a wrong label shows; and a case's scenario sends the
    requests python-ref's line holds (its task, context and predicted rows
    the same)."""
    document = vectors()
    assert document['recorded_by'] == 'contract/tools/target_record.py'
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {document["reference"]}' in env
    vector = document['vectors'].get(label_key)
    assert vector is not None, f'{label_key}: no vector'
    scenario_id, _, database = label_key.partition('@')
    assert vector['rule'] == R.SCENARIOS[scenario_id]['rule']
    if vector['outcome']['result'] == 'success':
        labels = _labels(vector['payloads'])
        assert labels, label_key
        assert any(
            len({json.dumps(label) for label in each}) > 1 for each in labels
        ), f'{label_key}: every context row has the same label'
    twin = R.SCENARIOS[scenario_id].get('twin')
    if twin is not None:
        assert vector['outcome']['result'] == 'refusal', label_key
        theirs = document['vectors'][R.key(twin, database)]
        assert theirs['outcome']['result'] == 'success', label_key
    cased = {s: c for c, s in R.CASE_OF.items()}
    if scenario_id in cased and vector['outcome']['result'] == 'success':
        line = reference_lines()[case_key(cased[scenario_id], database)]
        found = differences(vector['payloads'], sent_labels(line))
        assert not found, f'{label_key}: not the line: ' + '; '.join(found)


# --- python-core against the reference ---------------------------------------------


def same_outcome(want: tuple[Any, ...], got: tuple[Any, ...]) -> bool:
    r"""Two outcomes the same, but that the core route raises a refusal
    coded ``UNSUPPORTED_REQUEST`` as ``UnsupportedRequestError``, a
    subclass of the reference's ``StructuredDataError`` (register P1B-20,
    approved by the owner, E78)."""
    if want == got:
        return True
    return (
        want[0] == got[0] == 'refusal'
        and want[1:2] == got[1:2]
        and want[3] == got[3] == 'UNSUPPORTED_REQUEST'
        and (want[2], got[2])
        == ('StructuredDataError', 'UnsupportedRequestError')
    )


def differences(want: Any, got: Any, at: str = '') -> list[str]:
    r"""Where ``got`` differs from ``want``, a few paths."""
    if isinstance(want, dict) and isinstance(got, dict):
        found = []
        for name in sorted(set(want) | set(got)):
            found += differences(want.get(name), got.get(name), f'{at}/{name}')
        return found[:8]
    if (
        isinstance(want, list)
        and isinstance(got, list)
        and len(want) == len(got)
    ):
        found = []
        for i, (w, g) in enumerate(zip(want, got, strict=True)):
            found += differences(w, g, f'{at}[{i}]')
        return found[:8]
    if want != got:
        if isinstance(want, list) and isinstance(got, list):
            return [
                f'{at}: {len(got)} items where the reference has {len(want)}'
            ]
        return [f'{at}: {got!r:.80} where the reference has {want!r:.80}']
    return []


def payloads(line: dict[str, Any]) -> list[Any]:
    r"""Each request the scripted endpoint received, parsed, as the
    ``payload`` rule compares it: its request id left out."""
    out = []
    for payload in line['public']['payloads']:
        parsed = copy.deepcopy(payload.get('parsed'))
        if isinstance(parsed, dict) and isinstance(
            parsed.get('metadata'), dict
        ):
            parsed['metadata'].pop('request_id', None)
        out.append(parsed)
    return out


@functools.cache
def core_run(database: str) -> dict[str, Any]:
    r"""python-core over T2.10's cases on ``database``, as that database's
    job runs them: each observation by its key, or under ``''`` why the
    run failed whole."""
    with tempfile.TemporaryDirectory(prefix='t210-') as tmp:
        out = Path(tmp) / 'python-core.jsonl'
        done = subprocess.run(
            [
                sys.executable,
                '-I',
                str(RUNNER),
                '--mode',
                'python-core',
                '--database',
                database,
                '--out',
                str(out),
                *(str(CASES / f'{c}.json') for c in CASE_IDS),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=1800,
        )
        if done.returncode != 0:
            return {'': done.stdout + done.stderr}
        return {
            json.loads(line)['case']: json.loads(line)
            for line in out.read_text(encoding='utf-8').splitlines()
        }


def _approved() -> list[dict[str, Any]]:
    entries = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))['entries']
    return [
        entry
        for entry in entries
        if entry.get('status') == 'approved' and entry['id'] not in UMBRELLAS
    ]


def _row(entry: dict[str, Any]) -> str | None:
    ref = entry.get('decision_ref')
    return ref.get('row') if isinstance(ref, dict) else None


def registered_key(key: str) -> str | None:
    r"""The approved entry, no umbrella and citing no temporary approval,
    naming the line key ``key`` in its ``cases`` or ``case_outcomes``."""
    register = _module('t210_register', ROOT / 'scripts/rust_core/register.py')
    temporary = tuple(register.TEMPORARY)
    for entry in _approved():
        named = set(entry.get('cases') or []) | {
            item.get('case') for item in entry.get('case_outcomes') or []
        }
        if key in named and _row(entry) not in temporary:
            return str(entry['id'])
    return None


@pytest.mark.parametrize('case_key', CASE_KEYS)
def test_python_core_answers_each_case_as_the_reference_does(
    case_key: str,
) -> None:
    r"""Exits "cases per rule" and the rules lifted: python-core runs the
    case on the database on the core route (the native module loaded,
    each of the case's core operations run), and its outcome, each
    request the scripted endpoint received (the labels among them) and
    its result through ``to_pandas()`` are python-ref's. A case the
    reference refuses only for a column name it cannot say is answered
    through ``predict`` as its twin is, each request the twin's labels
    (``rank.odd-column``), and an approved register entry citing no
    temporary approval names its key (KO2-36, as T2.18's cases)."""
    case_id, _, database = case_key.partition('@')
    database = database or 'duckdb'
    if database == 'postgres' and not _server():
        return

    def check() -> None:
        run = core_run(database)
        assert '' not in run, run.get('')
        line = run.get(case_key)
        assert line is not None, f'{case_key}: not observed'
        case, _ = cases()[case_id]
        identity = line['identity']
        assert identity['native_loaded'] and line['core'] is not None
        missing = sorted(set(case['core']) - set(identity['operations']))
        assert not missing, f'{case_key}: core operations not run: {missing}'
        want = reference_lines()[case_key]
        got = _outcome(line['outcome'])
        if case_id in NAMED:
            assert got[:2] == ('success', 'predict'), (
                f'{case_key}: {got} where the twin answers through predict'
            )
            assert registered_key(case_key), (
                f'{case_key}: no approved, permanent entry names it'
            )
            twin = R.SCENARIOS.get(case_id, {}).get('twin')
            if twin is not None:
                theirs = vectors()['vectors'][R.key(twin, database)]
                found = differences(theirs['payloads'], sent_labels(line))
                assert not found, f"{case_key}: not the twin's: " + '; '.join(
                    found
                )
            return
        assert same_outcome(_outcome(want['outcome']), got), (
            f'{case_key}: {got} where python-ref has '
            f'{_outcome(want["outcome"])}'
        )
        found = differences(payloads(want), payloads(line))
        assert not found, f'{case_key}: the payload differs: ' + '; '.join(
            found
        )
        found = differences(want['public']['result'], line['public']['result'])
        assert not found, f'{case_key}: the result differs: ' + '; '.join(found)

    held(f'py:{case_key}', check)


@functools.cache
def core_labels(label_key: str) -> dict[str, Any]:
    r"""python-core's observation of the scenario, by the recorder's code."""
    scenario_id, _, database = label_key.partition('@')
    return R.observe(scenario_id, database, core=True)


@pytest.mark.parametrize('label_key', LABEL_KEYS)
def test_python_core_labels_each_target_as_the_reference_does(
    label_key: str,
) -> None:
    r"""Exit "label vectors from the reference", each rule lifted:
    python-core, on the core route, answers the scenario as the reference
    does: its outcome; each request's task and its context's and
    predicted rows' instance tables (the labels, row by row); the rows
    each call scores; the result through ``to_pandas()``. A scenario the
    reference refuses only for a column name it cannot say is answered
    as its twin (KO2-36)."""
    scenario_id, _, database = label_key.partition('@')
    if database == 'postgres' and not _server():
        return

    def check() -> None:
        twin = R.SCENARIOS[scenario_id].get('twin')
        want = vectors()['vectors'][
            R.key(twin, database) if twin else label_key
        ]
        got = core_labels(label_key)
        outcome = (_outcome(want['outcome']), _outcome(got['outcome']))
        assert same_outcome(*outcome), (
            f'{label_key}: {outcome[1]} where python-ref has {outcome[0]}'
        )
        found = differences(want['payloads'], got['payloads'], '/payloads')
        found += differences(want['calls'], got['calls'], '/calls')
        found += differences(want['result'], got['result'], '/result')
        assert not found, f'{label_key}: ' + '; '.join(found)

    held(f'py-labels:{label_key}', check)


# --- the filter role of any column name (T2.18's vectors) --------------------------


def as_named(value: Any, twin: dict[str, Any], vector: dict[str, Any]) -> Any:
    r"""The twin's answer read for ``vector``: the twin's query text as the
    vector's, and the twin's column name as the vector's, in every key and
    text (``test_t218_column_names.as_named``)."""
    if isinstance(value, dict):
        return {
            as_named(key, twin, vector): as_named(item, twin, vector)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [as_named(item, twin, vector) for item in value]
    if value == twin['pql']:
        return vector['pql']
    if value == twin['name']:
        return vector['name']
    return value


@functools.cache
def _name_answer(vector_id: str) -> dict[str, Any]:
    r"""The vector through the public calls with its database's
    ``from_source`` and ``materialize`` routes forced onto the core."""
    from sdm_client import _routing

    vector = next(v for v in _filter_vectors() if v['id'] == vector_id)
    database = vector['database']
    routes = (
        *_routing.FROM_SOURCE[database],
        *_routing.MATERIALIZE[database],
        *_routing.S0,
    )
    previous = os.environ.get(_routing.ENVIRONMENT)
    os.environ[_routing.ENVIRONMENT] = ','.join(dict.fromkeys(routes))
    try:
        found: dict[str, Any] = names().run(vector)
        return found
    finally:
        if previous is None:
            os.environ.pop(_routing.ENVIRONMENT, None)
        else:
            os.environ[_routing.ENVIRONMENT] = previous


def test_the_filter_role_s_vectors_are_back() -> None:
    r"""Control: the filter role's fifteen vectors (four characters and the
    twin on each database) are in the column-name vectors again, as T2.18's
    recorder writes them; on the reference each twin answers and each
    character is refused at ``materialize``."""
    expected = [v for v in names().vectors() if v['role'] == FILTER_ROLE]
    recorded = [
        v for v in name_vectors()['vectors'] if v['role'] == FILTER_ROLE
    ]
    assert [v['id'] for v in recorded] == [v['id'] for v in expected]
    assert len(recorded) == 15
    for vector in recorded:
        result = vector['reference']['outcome']['result']
        assert result == (
            'success' if vector['character'] is None else 'refusal'
        ), vector['id']


@pytest.mark.parametrize('vector_id', NAME_IDS)
def test_a_filter_names_any_column(vector_id: str) -> None:
    r"""The filter role of T2.18's "a query can name any column", carried
    here (its ``target_filter``): python-core's ``Schema.from_source`` and
    ``materialize`` of a target filtered by a column named with a dot, a
    backtick, a line feed or a carriage return answer as its twin does on
    the reference: the schema, the outcome, the warnings and the request's
    payloads, the twin's name and query read as the vector's."""
    vector = next(v for v in _filter_vectors() if v['id'] == vector_id)
    if vector['database'] == 'postgres' and not _server():
        return

    def check() -> None:
        by_id = {v['id']: v for v in name_vectors()['vectors']}
        twin = by_id[vector['twin']]
        got = _name_answer(vector_id)
        assert got['schema'] == as_named(
            twin['reference']['schema'], twin, vector
        ), f"{vector_id}: not the twin's schema"
        assert got['outcome'] == twin['reference']['outcome'], (
            f'{vector_id}: {got["outcome"]}'
        )
        assert got['payloads'] == as_named(
            twin['reference']['payloads'], twin, vector
        ), f"{vector_id}: not the twin's request"
        assert got['sdm_warnings'] == twin['reference']['sdm_warnings'], (
            f'{vector_id}: {got["sdm_warnings"]}'
        )

    held(f'py-name:{vector_id}', check)
