# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contract's case, inventory and observation formats (T1.10).

The Phase 1b inventory validates against its schema, and the inventory
loader maps every one of its cases into the case format and back without
loss; the bases the cases build their data from are the inventory's; every
committed case and reference observation validates, and the reference meets
each case's expectation. Running the cases needs the relational engine, so
that lives in ``contract/tests``, which CI's ``engine`` job runs.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / 'contract/runners/run_python.py'
COMPARE = ROOT / 'scripts/rust_core/compare.py'
CASES = ROOT / 'contract/cases'
REFERENCE = ROOT / 'contract/reference'
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location('run_python', RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_python = _runner()


def _inventory() -> dict[str, Any]:
    inventory: dict[str, Any] = run_python.load_inventory()
    return inventory


def test_the_inventory_validates_against_its_schema() -> None:
    inventory = _inventory()
    assert len(inventory['cases']) == 40
    validate = run_python.validator('inventory')
    broken = copy.deepcopy(inventory)
    broken['cases'][0]['seam'] = 'no-such-seam'
    assert run_python._invalid(validate, broken) is not None
    del broken['cases'][0]['seam']
    assert run_python._invalid(validate, broken) is not None


def test_the_inventory_loader_maps_every_case_without_loss() -> None:
    inventory = _inventory()
    defaults = inventory['defaults']
    validate = run_python.validator('case')
    for entry in inventory['cases']:
        case = run_python.inventory_case(entry, defaults)
        assert run_python._invalid(validate, case) is None, entry['id']
        assert run_python.inventory_entry(case, defaults) == entry
    # The mapping keeps what the inventory left to its defaults apart from
    # what it stated, and refuses a case it has no place for.
    entry = next(item for item in inventory['cases'] if 'pql' not in item)
    case = run_python.inventory_case(entry, defaults)
    assert case['input']['pql'] == defaults['pql']
    case['status'] = 'acceptance'
    with pytest.raises(run_python.RunnerFailure):
        run_python.inventory_entry(case, defaults)


#: The inventory cases whose call the contract cannot carry to materialize,
#: so their core is describe_source alone (T1.21e): materialize's indices
#: are at least one, and this names none (the empty-null refusal map's
#: gap). From T1.32a the decimal keys' entities are carried, each a scalar
#: of the type the host's values have.
_DESCRIBE_ONLY = {
    'empty.indices',
}
#: A call's core operations, in order, and the one each public call ends on.
_ORDER = ('describe_source', 'materialize', 'predict')
_STAGE_OPERATION = {
    'from_source': 'describe_source',
    'materialize': 'materialize',
    'predict': 'predict',
}


def _written(index: dict[str, Any]) -> str:
    r"""An entity scalar's value as the call writes it: a decimal's unscaled
    digits with the point shifted by its scale, by hand, so no decimal
    context rounds a wide one; any other's ``v``."""
    if index['t'] not in ('d128', 'd256'):
        return str(index['v'])
    digits, scale = index['v'], index['s']
    sign = '-' if digits.startswith('-') else ''
    digits = digits.lstrip('-')
    if scale <= 0:
        return sign + digits + '0' * -scale if digits != '0' else '0'
    digits = digits.rjust(scale + 1, '0')
    return f'{sign}{digits[:-scale]}.{digits[-scale:]}'


def _anchor(text: str) -> dict[str, Any]:
    import datetime

    when = datetime.datetime.fromisoformat(text)
    epoch = datetime.datetime(1970, 1, 1)
    nanos = (when - epoch) // datetime.timedelta(microseconds=1) * 1000
    return {'t': 'ts', 'v': str(nanos), 'unit': 'ns', 'zone': None}


def _refused_before_reading(case_id: str) -> bool:
    r"""Whether the reference refused ``case_id``'s query before reading:
    INVALID_REQUEST at ``materialize``, its observation holding no
    statement of ``materialize`` (the T1.32d family test's
    ``QUERY_REFUSALS`` rule)."""
    for line in _reference().read_text(encoding='utf-8').splitlines():
        observed = json.loads(line)
        if observed['case'] != case_id:
            continue
        outcome = observed['outcome']
        statements = observed['driver']['capture']['statements']
        return (
            outcome['result'] == 'refusal'
            and outcome['stage'] == 'materialize'
            and outcome['code'] == 'INVALID_REQUEST'
            and not [s for s in statements if s['stage'] == 'materialize']
        )
    return False


def _held_to_its_input(case: dict[str, Any]) -> list[str]:
    r"""Each way a case's core operations depart from its own input: the
    source described is the one ``from_source`` names, materialize's
    entities and anchor time are the call's, and predict follows
    materialize."""
    given, core = case['input'], case['core']
    problems = []
    source = core['describe_source']['source']
    named = given.get('from_source', {})
    if source != {
        'tables': list(named.get('tables', [])),
        'primary_keys': {
            t: {'value': [k]} for t, k in named.get('primary_keys', {}).items()
        },
        'time_columns': {
            t: {'value': c} for t, c in named.get('time_columns', {}).items()
        },
    }:
        problems.append('describe_source is not of its source')
    materialize = core.get('materialize')
    # The operations a case carries are its call's, in order, from
    # describe_source. A diagnostic case carries its whole call (bar
    # _DESCRIBE_ONLY's); an acceptance case only those its call reaches on
    # the core, since compare.py's inputs rule holds a core runner to every
    # operation a case names, and at least those the reference reached
    # (its expect's stage). Which exactly, its family's admission test
    # says (the nested family's REACHED).
    carried = tuple(op for op in _ORDER if op in core)
    if carried != _ORDER[: len(carried)]:
        problems.append('the operations carried are not the call in order')
    if 'validate_query' in core:
        # In place of materialize, where the reference refused the query
        # before reading (as the T1.32d family test allows): python-core
        # validates it before the core's materialize would run.
        if not (
            case['status'] == 'acceptance'
            and carried == _ORDER[:1]
            and _refused_before_reading(case['id'])
        ):
            problems.append('validate_query where no query is refused first')
    elif case['id'] in _DESCRIBE_ONLY and case['status'] != 'acceptance':
        # Admitted, the host carries its call (no entity) to materialize,
        # which refuses it as the reference does; the rule below holds it.
        if carried != _ORDER[:1]:
            problems.append('materialize present where it cannot be')
    elif case['status'] != 'acceptance':
        if carried != _ORDER:
            problems.append('a diagnostic case carries less than its call')
    elif (
        len(carried)
        < _ORDER.index(_STAGE_OPERATION[case['expect']['stage']]) + 1
    ):
        problems.append("the operations stop before the reference's stage")
    if materialize is not None:
        call = given['materialize']
        if [_written(index) for index in materialize['indices']] != [
            str(index) for index in call['indices']
        ]:
            problems.append("materialize's indices are not the call's")
        wanted = _anchor(call['anchor_time']) if 'anchor_time' in call else None
        if materialize['anchor_time'] != wanted:
            problems.append("materialize's anchor time is not the call's")
    if core.get('predict', {'model': 'kumo-relational'}) != {
        'model': 'kumo-relational'
    }:
        problems.append("predict's model is not the call's")
    return problems


#: The inventory cases T1.29 admitted, before the families' test authors
#: wrote admission exits.
ADMITTED_AT_T129 = frozenset(
    {'base.regression', 'base.binary', 'base.multiclass'}
)
#: A family's admission exit, as its pending list names it: the test id of
#: a ``test_each_..._is_admitted`` parametrized by the case.
_ADMISSION_EXIT = re.compile(r'^py:test_each_[a-z_]*is_admitted\[(.+)\]$')


def _admission_exits(
    directory: Path = ROOT / 'contract/tests',
) -> dict[str, bool]:
    r"""Each case a family's test author wrote an admission exit for
    (``contract/tests/t132*-pending.json``'s ``authored``), and whether
    that exit has passed (it is no longer ``pending``). A pending list is
    the test author's and only shrinks (each family's ``only_shrinks``), so
    a case is admitted when its exit has cleared, and only then."""
    found: dict[str, bool] = {}
    for path in sorted(directory.glob('t132*-pending.json')):
        loaded = json.loads(path.read_text(encoding='utf-8'))

        def ids(group: dict[str, Any]) -> set[str]:
            return {
                name
                for names in group.values()
                if isinstance(names, list)
                for name in names
                if isinstance(name, str)
            }

        pending = ids(loaded['pending'])
        for name in sorted(ids(loaded['authored'])):
            match = _ADMISSION_EXIT.match(name)
            if match:
                case = match.group(1)
                assert case not in found, f'{case}: two admission exits'
                found[case] = name not in pending
    return found


def admitted_inventory_cases() -> set[str]:
    r"""The inventory cases that are acceptance cases: T1.29's three, and
    each case whose family's admission exit has cleared."""
    return set(ADMITTED_AT_T129) | {
        case for case, cleared in _admission_exits().items() if cleared
    }


#: T2.0c's S1 cases for exact decimals (KO2-50), none in the inventory.
T20C_DECIMAL = (
    'decimal.compare-finer-gt',
    'decimal.compare-finer-eq',
    'decimal.compare-finer-negative',
    'decimal.compare-past-precision',
    'decimal.compare-past-precision-gt',
    'decimal.compare-same-scale',
    'decimal.sum-10-2',
    'decimal.feature-inexact-38-18',
    'decimal.key-wide',
    'decimal.target-regression-collide',
)

#: T2.19's two-path sampling cases (main's eb6c68eb), none in the inventory.
T219_SHAPES = (
    'shape.chain.addresses',
    'shape.diamond.addresses',
    'shape.diamond-from-source',
    'shape.one-column-two-tables',
    'shape.same-table-twice.buyers',
    'shape.same-table-twice.customers',
    'shape.star.customers',
    'shape.star.products',
    'shape.star.stores',
    'shape.two-paths.keyed',
    'shape.two-paths.keyless',
)

#: T2.8c's multi-call predict cases (main's 1fc5a686), none in the
#: inventory: the call's ``rows_per_call`` and the endpoint's failed call
#: (``input.fail_call``) are inputs no inventory entry carries.
T28C_MULTICALL = (
    'multicall.body-30mib.one-row',
    'multicall.body-30mib.several-rows',
    'multicall.calls.100',
    'multicall.calls.101',
    'multicall.entities-per-call.10001',
    'multicall.entities.100000',
    'multicall.entities.100001',
    'multicall.fail-call-0',
    'multicall.fail-call-2',
    'multicall.forecast.batched',
    'multicall.forecast.fail-call-1',
    'multicall.forecast.one-call',
    'multicall.rows-per-call-3',
    'multicall.rows-per-call.above-the-cap',
    'multicall.rows-per-call.bool',
    'multicall.rows-per-call.fraction',
    'multicall.rows-per-call.negative',
    'multicall.rows-per-call.past-int64',
    'multicall.rows-per-call.text',
    'multicall.rows-per-call.whole-float',
    'multicall.rows-per-call.zero',
)


#: T2.18's S1 cases for any column name (KO2-36), none in the inventory:
#: each of a dot, a backtick and a line break in a column's name, as a
#: feature the query does not name, the target, the entity (users' declared
#: primary key) and an aggregation's column. The filter role waits for
#: T2.10 (contract/deferred/t210-column-filter/).
T218_NAMES = tuple(
    f'name.{name}-{role}'
    for name in ('dot', 'backtick', 'line-break')
    for role in ('feature', 'target', 'entity', 'aggregation')
)


def test_every_inventory_case_is_a_committed_case_as_the_inventory_has_it() -> (
    None
):
    # T1.21e: the 40 cases are contract cases, each the inventory's as its
    # loader maps it (ids, bases, setup, from_source, query, materialize,
    # expect), diagnostic, with the core operations its call starts.
    # base.regression, a case since T1.10, keeps the inventory's input and
    # expectation and adds its routes.
    inventory = _inventory()
    defaults = inventory['defaults']
    # T1.24b's edit cases, S1 cases of their own (moved here at T1.27), are
    # no inventory case.
    committed = {
        case['id']: case
        for case, _ in run_python.load_cases([CASES / 's1'])
        if not case['id'].startswith('edit.')
    }
    # T1.33 (T1.32f's real-data part) adds the notebook's churn and Color
    # questions over the AdventureWorks base, no inventory case either.
    # T2.0c adds the exact-decimal cases (KO2-50), no inventory case either;
    # each is an acceptance case carrying its call as far as the reference
    # goes. T2.19 adds the two-path sampling cases, acceptance cases through
    # predict, no inventory case either. T2.18 adds the column-name cases
    # (KO2-36), each carrying its call through predict; so does T2.8c its
    # multi-call cases, each an acceptance case.
    assert set(committed) == {entry['id'] for entry in inventory['cases']} | {
        'base.adventureworks',
        'base.adventureworks.churn',
        'base.adventureworks.color',
        *T20C_DECIMAL,
        *T219_SHAPES,
        *T218_NAMES,
        *T28C_MULTICALL,
    }
    for case_id in T28C_MULTICALL:
        assert committed[case_id]['status'] == 'acceptance', case_id
        assert _held_to_its_input(committed[case_id]) == [], case_id
    for case_id in (*T20C_DECIMAL, *T218_NAMES):
        assert committed[case_id]['status'] == 'acceptance', case_id
        assert _held_to_its_input(committed[case_id]) == [], case_id
    for case_id in T219_SHAPES:
        assert committed[case_id]['status'] == 'acceptance', case_id
        assert committed[case_id]['until'] == 'predict', case_id
    # From T1.29 the inventory's three baseline cases are acceptance cases
    # with their routes and paths, keeping its input and expectation; from
    # T1.32 so is each case whose family's admission exit has cleared, and
    # no other (admitted_inventory_cases). Every admission exit names an
    # inventory case.
    assert set(_admission_exits()) <= {
        entry['id'] for entry in inventory['cases']
    }
    admitted = admitted_inventory_cases()
    for entry in inventory['cases']:
        case = committed[entry['id']]
        assert case['status'] == (
            'acceptance' if entry['id'] in admitted else 'diagnostic'
        ), entry['id']
        assert _held_to_its_input(case) == [], entry['id']
        if entry['id'] in admitted:
            mapped = run_python.inventory_case(entry, defaults)
            for key in ('input', 'expect', 'seam', 'cells', 'until'):
                assert case[key] == mapped[key], key
            continue
        held = {key: value for key, value in case.items() if key != 'core'}
        assert run_python.inventory_entry(held, defaults) == entry, entry['id']

    # Seeded: a case changed alone departs from the inventory, and a core
    # changed alone from its input. The fixture is time.text-declared's
    # file in the inventory's own form (diagnostic, no routes, no paths),
    # whether or not it is admitted: an admitted case is no inventory
    # entry, and may carry less than its call.
    def fixture() -> dict[str, Any]:
        found = copy.deepcopy(committed['time.text-declared'])
        found.update(status='diagnostic', routes=[], paths=[])
        assert set(found['core']) == set(_ORDER), 'not the whole call'
        assert _held_to_its_input(found) == [], 'the fixture holds'
        return found

    case = fixture()
    case['input']['setup'].pop()
    held = {key: value for key, value in case.items() if key != 'core'}
    assert run_python.inventory_entry(held, defaults) != next(
        entry for entry in inventory['cases'] if entry['id'] == case['id']
    )
    for change in (
        lambda core: core['describe_source']['source']['time_columns'].clear(),
        lambda core: core['materialize']['indices'].pop(),
        lambda core: core['materialize'].__setitem__('anchor_time', None),
        lambda core: core.pop('predict'),
        lambda core: core.pop('materialize'),
    ):
        case = fixture()
        change(case['core'])
        assert _held_to_its_input(case) != [], change
    # An acceptance case carries its call up to where the core stops and
    # not short of the reference's stage; a diagnostic one its whole call;
    # any, the call in order.
    case = copy.deepcopy(committed['nested.map'])
    assert case['expect']['stage'] == 'from_source'
    case['core'].pop('predict', None)
    case['core'].pop('materialize', None)
    case['status'] = 'diagnostic'
    assert _held_to_its_input(case) != []
    case['status'] = 'acceptance'
    assert _held_to_its_input(case) == []
    case = copy.deepcopy(committed['time.text-us-date'])
    assert case['expect']['stage'] == 'materialize'
    case['status'] = 'acceptance'
    assert _held_to_its_input(case) == []
    case['core'].pop('predict', None)
    assert _held_to_its_input(case) == []
    case['core'].pop('materialize', None)
    assert _held_to_its_input(case) != []
    case = copy.deepcopy(committed['time.text-us-date'])
    case['status'] = 'acceptance'
    case['core'].pop('materialize', None)
    assert _held_to_its_input(case) != []
    case['core'].pop('describe_source')
    with pytest.raises(KeyError):
        _held_to_its_input(case)
    # validate_query in place of materialize: for an admitted case whose
    # query the reference refused before reading, and no other.
    assert _refused_before_reading('refuse.unknown-column')
    assert not _refused_before_reading('time.text-us-date')
    validated = {'profile': {'name': 'RELATIONAL'}, 'schema': {}}
    case = copy.deepcopy(committed['refuse.unknown-column'])
    case['core'].pop('predict', None)
    case['core'].pop('materialize', None)
    case['core']['validate_query'] = validated
    case['status'] = 'acceptance'
    assert _held_to_its_input(case) == []
    case['status'] = 'diagnostic'
    assert _held_to_its_input(case) != []
    case = copy.deepcopy(committed['time.text-declared'])
    case['core']['validate_query'] = validated
    case['status'] = 'acceptance'
    assert _held_to_its_input(case) != [], 'beside materialize'
    case = copy.deepcopy(committed['time.text-us-date'])
    case['core'].pop('predict', None)
    case['core'].pop('materialize', None)
    case['core']['validate_query'] = validated
    case['status'] = 'acceptance'
    assert _held_to_its_input(case) != [], 'the reference read data first'
    # Without it, the refused-first case stops short of materialize.
    case = copy.deepcopy(committed['refuse.unknown-column'])
    case['core'].pop('validate_query', None)
    case['core'].pop('predict', None)
    case['core'].pop('materialize', None)
    case['status'] = 'acceptance'
    assert _held_to_its_input(case) != []


def test_the_admitted_set_follows_the_families_admission_exits(
    tmp_path: Path,
) -> None:
    # Seeded: an admission exit admits its case once it has left pending,
    # and not before; another exit admits nothing; an exit written twice
    # is refused.
    def written(name: str, pending: list[str], authored: list[str]) -> None:
        (tmp_path / name).write_text(
            json.dumps(
                {
                    'pending': {'exits': pending, 'rust-host': []},
                    'authored': {'exits': authored, 'rust-host': ['x']},
                }
            ),
            encoding='utf-8',
        )

    cleared = 'py:test_each_case_is_admitted[key.uint8]'
    waiting = 'py:test_each_nested_case_is_admitted[nested.map]'
    other = 'py:test_null_decimal_18_2_is_cleared'
    written('t132x-pending.json', [waiting], [cleared, waiting, other])
    assert _admission_exits(tmp_path) == {
        'key.uint8': True,
        'nested.map': False,
    }
    written('t132y-pending.json', [], [cleared])
    with pytest.raises(AssertionError, match='two admission exits'):
        _admission_exits(tmp_path)
    # The committed lists: T1.29's three, and only cleared exits beside.
    exits = _admission_exits()
    assert exits, 'no admission exit read'
    assert admitted_inventory_cases() == set(ADMITTED_AT_T129) | {
        case for case, done in exits.items() if done
    }


#: A statement that names the time zone, in any case and spacing.
_ZONE = re.compile(r'time\s*_?\s*zone', re.IGNORECASE)
#: The one form a case may set it in: for the whole database, so every
#: cursor the source reads through takes it, not the process's local zone.
_GLOBAL_ZONE = re.compile(
    r"^\s*SET\s+GLOBAL\s+TimeZone\s*(=|TO)\s*'[^']+'\s*;?\s*$",
    re.IGNORECASE,
)


def _zone_problems(where: str, statements: list[str]) -> list[str]:
    r"""Each statement of ``statements`` that names the time zone other
    than as ``SET GLOBAL TimeZone = '...'``."""
    return [
        f'{where}: {statement!r}'
        for statement in statements
        if _ZONE.search(statement) and not _GLOBAL_ZONE.match(statement)
    ]


def test_a_case_sets_its_time_zone_for_every_cursor() -> None:
    # The source reads through cursors, which take the process's local zone
    # unless the database's is set: a session SET applies to the setup's
    # connection alone, so the reference recorded the recorder's machine
    # zone (time.timestamptz, found by the bundle's landing CI). Every case,
    # inventory entry and refusal-map probe setup that names the zone sets
    # it with SET GLOBAL, so no answer depends on the recorder's TZ.
    problems = []
    named = 0
    for case, _ in run_python.load_cases([CASES]):
        setup = case['input'].get('setup', [])
        named += sum(bool(_ZONE.search(each)) for each in setup)
        problems += _zone_problems(case['id'], setup)
    for entry in _inventory()['cases']:
        problems += _zone_problems(
            f'inventory {entry["id"]}', entry.get('setup', [])
        )
    for path in sorted((ROOT / 'contract/refusal-maps').glob('*.json')):
        for probe in json.loads(path.read_text(encoding='utf-8'))['probes']:
            for runner in ('python-core', 'rust-host'):
                spec = probe.get(runner) or {}
                setup = (
                    (spec.get('input') or {}).get('setup')
                    or spec.get('setup')
                    or []
                )
                problems += _zone_problems(
                    f'{path.name} {probe["id"]} {runner}', setup
                )
    assert problems == []
    assert named, 'no case names the zone: the check reads nothing'
    # Seeded: every other way to name it fails, and the global form passes.
    for statement in (
        "SET TimeZone = 'America/New_York'",
        "SET SESSION TimeZone = 'UTC'",
        "set timezone to 'UTC'",
        'RESET TimeZone',
        "SET LOCAL TimeZone = 'UTC'",
        "SELECT set_config('TimeZone', 'UTC')",
        "SET GLOBAL TimeZone = 'UTC'; SET TimeZone = 'Asia/Tokyo'",
    ):
        assert _zone_problems('seed', [statement]), statement
    for statement in (
        "SET GLOBAL TimeZone = 'America/New_York'",
        "set global timezone to 'UTC'",
        'ALTER TABLE orders ALTER ts TYPE TIMESTAMPTZ',
    ):
        assert not _zone_problems('seed', [statement]), statement


#: The setups a vector file's tests run as written: each list (or single
#: statement) under one of these keys, at any depth.
_VECTOR_SETUP_KEYS = ('setup', 'witness')
#: Per-connection zone statements still in a vector file, each named by
#: file, vector id and statement. type-set-vectors.json's witnesses are
#: read through the source's cursors, so they reach the process's zone,
#: not the one named (another test author owns that file); drop an entry
#: once its vector sets the zone with SET GLOBAL.
_VECTOR_ZONE_PENDING = {
    _zone_problems(f'type-set-vectors.json {vector}', [statement])[0]
    for vector, statement in (
        ('timestamp.us.utc', "SET TimeZone = 'UTC'"),
        ('timestamp.us.named-zone', "SET TimeZone = 'America/Los_Angeles'"),
    )
}


#: A plain per-connection zone statement: the caller's connection alone.
_SESSION_ZONE = re.compile(
    r"^\s*SET\s+(SESSION\s+)?TimeZone\s*(=|TO)\s*'[^']+'\s*;?\s*$",
    re.IGNORECASE,
)


def _vector_zone_problems(where: str, statements: list[str]) -> list[str]:
    r""":func:`_zone_problems`, but a plain session ``SET TimeZone`` is
    allowed in a setup that also sets the database's zone with ``SET
    GLOBAL``: every cursor then takes the global zone, and the session
    statement only sets the setup's own connection, as a vector may do on
    purpose to show that the source does not read in it
    (t136-host2-zone-vectors.json's session.* and verbatim.* vectors). Any
    other form (RESET, LOCAL, set_config, a compound statement) is still a
    problem, global zone or not."""
    pinned = any(_GLOBAL_ZONE.match(each) for each in statements)
    return _zone_problems(
        where,
        [
            each
            for each in statements
            if not (pinned and _SESSION_ZONE.match(each))
        ],
    )


def _vector_setups(node: Any, where: str) -> list[tuple[str, list[str]]]:
    r"""Each setup ``node`` holds under :data:`_VECTOR_SETUP_KEYS`, with
    the id of the nearest enclosing object that has one."""
    found: list[tuple[str, list[str]]] = []
    if isinstance(node, dict):
        where = str(node.get('id', where))
        for key, value in node.items():
            if key in _VECTOR_SETUP_KEYS and isinstance(value, (str, list)):
                statements = [value] if isinstance(value, str) else value
                found.append(
                    (where, [s for s in statements if isinstance(s, str)])
                )
            found += _vector_setups(value, where)
    elif isinstance(node, list):
        for each in node:
            found += _vector_setups(each, where)
    return found


def test_a_vector_sets_its_time_zone_for_every_cursor() -> None:
    # T1.36 C16: a vector's setup runs on one connection and the source
    # reads through cursors, which take the process's local zone unless the
    # database's is set, so a per-connection SET TimeZone made the recorded
    # reference answer depend on the recorder's TZ (empty-null-vectors.json
    # null.time-timestamptz: 361 order rows under UTC or Los Angeles, 359
    # under Asia/Kolkata). Every vector file's setups name the zone only as
    # SET GLOBAL TimeZone, as the cases do.
    problems = set()
    named = 0
    for path in sorted((ROOT / 'contract').glob('*-vectors.json')):
        document = json.loads(path.read_text(encoding='utf-8'))
        for vector, setup in _vector_setups(document, '?'):
            named += sum(bool(_ZONE.search(each)) for each in setup)
            problems.update(
                _vector_zone_problems(f'{path.name} {vector}', setup)
            )
    assert problems == _VECTOR_ZONE_PENDING
    assert named > len(_VECTOR_ZONE_PENDING), 'no vector sets its zone'
    # Seeded: a session SET passes only beside a global one, in the same
    # setup; nothing else does.
    pinned = "SET GLOBAL TimeZone = 'UTC'"
    for setup in (
        ["SET TimeZone = 'America/New_York'"],
        ["SET SESSION TimeZone = 'UTC'"],
        [pinned, 'RESET TimeZone'],
        [pinned, "SET LOCAL TimeZone = 'UTC'"],
        [pinned, "SELECT set_config('TimeZone', 'UTC')"],
        [pinned, "SET GLOBAL TimeZone = 'UTC'; SET TimeZone = 'Asia/Tokyo'"],
    ):
        assert _vector_zone_problems('seed', setup), setup
    for setup in (
        [pinned, "SET TimeZone = 'America/New_York'"],
        ["SET TimeZone='America/New_York'", pinned],
        [pinned, "set session timezone to 'UTC'"],
        [pinned],
    ):
        assert not _vector_zone_problems('seed', setup), setup


def test_the_bases_are_the_inventorys_and_adventureworks() -> None:
    # The inventory's bases, and E27's AdventureWorks base beside them, in
    # addition to the inventory (E31; T1.21b).
    bases = run_python.load_bases()
    assert 'adventureworks' not in _inventory()['bases']
    assert {
        name: statements
        for name, statements in bases.items()
        if name != 'adventureworks'
    } == _inventory()['bases']
    assert bases['adventureworks'][0].startswith(
        'CREATE TABLE adventureworks_customers'
    )


def test_every_committed_case_validates_with_a_unique_id() -> None:
    cases = run_python.load_cases([CASES])
    ids = [case['id'] for case, _ in cases]
    assert len(ids) == len(set(ids)) >= 2
    s0 = [case for case, _ in cases if case['slice'] == 'S0']
    assert {case['expect']['outcome'] for case in s0} == {'success', 'refusal'}


def _holds_packed_date(query: str) -> bool:
    r"""Whether ``query`` holds a date literal with a run of five digits or
    more (D8): a ``DATE`` token, three runs of digits joined by one of
    ``/``, ``-`` or ``.``, with such a run, or followed by a ``TIME`` token,
    three runs joined by ``:``, with one."""
    words = re.split(r'[\s(),]', query)

    def runs(word: str, separators: str) -> list[int] | None:
        sep = next((c for c in word if c in separators), None)
        if sep is None:
            return None
        parts = word.split(sep)
        if len(parts) != 3 or not all(
            part.isascii() and part.isdigit() for part in parts
        ):
            return None
        return [len(part) for part in parts]

    for i, word in enumerate(words):
        date = runs(word, '/-.')
        if date is None:
            continue
        time = runs(words[i + 1], ':') if i + 1 < len(words) else None
        if any(n >= 5 for n in date + (time or [])):
            return True
    return False


def test_the_s0_boundary_cases_sit_on_their_boundaries() -> None:
    # T1.12's boundary cases, held to the shapes their `why` states.
    cases = {
        case['id']: case for case, _ in run_python.load_cases([CASES / 's0'])
    }

    def pql(name: str) -> str:
        text: str = cases[f's0.boundary.{name}']['input']['pql']
        return text

    # Python's 4,300-digit limit, counting leading zeros, on both sides.
    for name, digits in (('digits-4300', 4300), ('digits-4301', 4301)):
        literal = pql(name).split('orders.price > ')[1].split(',')[0]
        assert literal == '0' * (digits - 1) + '1'
    # Past the 64 KB control cap (decision H5).
    assert len(pql('start-past-64kb').encode()) > 64 * 1024
    # Below the depth the reference reaches before RecursionError (about
    # 461 in the runner on Python 3.12) and the core's MAX_DEPTH (999).
    deepest = pql('deepest')
    assert deepest.count('(' * 400) == 1 and '(' * 401 not in deepest
    # The two date cases with a run of five digits or more (D8's packed
    # date and DT-9's long field) stay diagnostic (a changed case file
    # needs an approved entry). From T1.6c the core refuses such a literal
    # by name where the reference reads it, so no acceptance case may
    # expect it to succeed: each S0 case holding one is diagnostic, under a
    # register entry, or expects the refusal.
    packed = {
        case['id']: case
        for case in cases.values()
        if _holds_packed_date(case['input']['pql'])
    }
    assert {
        's0.boundary.date-packed',
        's0.boundary.date-long-field',
    } <= set(packed)
    for case in packed.values():
        if case['status'] == 'diagnostic':
            assert case.get('register'), case['id']
        else:
            assert case['expect']['outcome'] == 'refusal', case['id']
            assert case['expect'].get('code') == 'INVALID_REQUEST', case['id']
    # Escapes as written in the query text, a backslash and its letters: a
    # \\u escape and no surrogate in the accepted case; an escaped pair and
    # a lone escaped surrogate in the two cases register PQL-1 holds.
    escapes = pql('escapes')
    assert '\\u00e9' in escapes and '\\ud' not in escapes
    assert escapes.isascii()
    assert '\\ud83d\\ude00' in pql('escape-surrogate-pair')
    lone = pql('escape-lone-surrogate')
    assert '\\ud83d' in lone and '\\ude' not in lone
    # Every other diagnostic S0 case names a register entry that waits for
    # approval, or one the owner approved whose admission of the case is a
    # later change's (PQL-1 and PQL-2, approved on 2026-10-13, E93).
    register = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))
    status = {
        entry['id']: (entry['status'], entry['approved_by'])
        for entry in register['entries']
    }
    for case in cases.values():
        if case['id'] in packed:
            assert case.get('register', 'none') in status, case['id']
        elif case['status'] == 'diagnostic':
            assert status[case['register']] in (
                ('proposed', None),
                ('approved', 'owner'),
            ), case['id']
    assert {
        case['id'] for case in cases.values() if case.get('register') == 'PQL-2'
    } == {'s0.boundary.depth-999'}
    # 999 levels exactly: one more is the core's first refusal.
    depth_999 = pql('depth-999')
    assert depth_999.count('(' * 999) == 1 and '(' * 1000 not in depth_999


def test_the_reference_observations_validate_and_meet_each_expectation(
    tmp_path: Path,
) -> None:
    files = sorted(REFERENCE.glob('*.jsonl'))
    assert len(files) == 1
    reference = files[0]
    validate = run_python.validator('observation')
    observations = [
        json.loads(line) for line in reference.read_text().splitlines()
    ]
    for observation in observations:
        assert run_python._invalid(validate, observation) is None
        identity = observation['identity']
        assert identity['runner'] == 'python-ref'
        assert identity['reference']['commit']['sha'] == reference.stem
        assert identity['native_loaded'] is False
    # Every case's expectation is its reference's outcome, each case read
    # here whatever compare.py can read of it.
    by_case = {o['case']: o['outcome'] for o in observations}
    for case, _ in run_python.load_cases([CASES]):
        outcome = by_case[case['id']]
        expect = case['expect']
        assert expect['outcome'] == outcome['result'], case['id']
        assert expect['stage'] == outcome['stage'], case['id']
        if outcome['result'] == 'refusal':
            assert expect.get('error') == outcome['error'], case['id']
            assert expect.get('code') == outcome['code'], case['id']
    # And compare.py holds the reference to them, as a runner too, on the
    # cases whose reference can stand in for one (_not_synthesizable).
    cases, kept = _synthesizable(tmp_path)
    run = subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            '--cases',
            str(cases),
            '--reference',
            str(kept),
            str(kept),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    # A corrupted expectation fails.
    corrupted = tmp_path / 'cases'
    corrupted.mkdir()
    for path in cases.rglob('*.json'):
        case = json.loads(path.read_text())
        if case['expect']['outcome'] == 'refusal':
            case['expect']['code'] = 'INTERNAL_ERROR'
        (corrupted / path.name).write_text(json.dumps(case))
    run = subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            '--cases',
            str(corrupted),
            '--reference',
            str(kept),
            str(kept),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 1
    assert "outcome.code: expected 'INTERNAL_ERROR'" in run.stdout


def test_both_loaders_refuse_an_unknown_key_inside_the_call_arguments(
    tmp_path: Path,
) -> None:
    # sdm-testkit's tests/s0_cases.rs refuses the same key.
    case = json.loads((CASES / 's0/s0.accept.sum-window.json').read_text())
    case['input']['materialize']['ignored_typo'] = True
    path = tmp_path / 'case.json'
    path.write_text(json.dumps(case))
    with pytest.raises(run_python.RunnerFailure, match='ignored_typo|fails'):
        run_python.load_cases([path])


def _changed(
    case: dict[str, Any], path: list[Any], value: Any
) -> dict[str, Any]:
    changed = copy.deepcopy(case)
    at = changed
    for key in path[:-1]:
        at = at[key]
    at[path[-1]] = copy.deepcopy(value)
    return changed


MALFORMED = ROOT / 'contract/tests/malformed-cases.json'


def test_a_case_nested_past_the_parser_s_depth_is_refused_by_name(
    tmp_path: Path,
) -> None:
    # sdm-testkit's loader refuses it by serde_json's depth limit; the Python
    # loader refused nothing and raised RecursionError (T1.21c's probe).
    case = json.loads(
        (ROOT / 'contract/cases/s1/base.adventureworks.json').read_text()
    )
    case['expect'] = {
        'outcome': 'refusal',
        'stage': 'predict',
        'error': 'E',
        'code': None,
    }
    deep = 100_000
    path = tmp_path / 'case.json'
    path.write_text(
        json.dumps(case).replace(
            '"code": null', '"code": ' + '[' * deep + ']' * deep
        )
    )
    with pytest.raises(run_python.RunnerFailure, match='not a case'):
        run_python.load_cases([path])
    path.write_text('{"format": ')
    with pytest.raises(run_python.RunnerFailure, match='not a case'):
        run_python.load_cases([path])


def test_both_loaders_refuse_the_same_malformed_edits_and_relationships(
    tmp_path: Path,
) -> None:
    # sdm-testkit's tests/s0_cases.rs loads the same changes with the Rust
    # loader, which must refuse and take the same ones (T1.21b).
    listed = json.loads(MALFORMED.read_text())
    case = json.loads((ROOT / listed['case']).read_text())
    assert len(listed['refused']) >= 54
    path = tmp_path / 'case.json'
    path.write_text(json.dumps(case))
    assert run_python.load_cases([path])
    for change in listed['refused']:
        path.write_text(
            json.dumps(_changed(case, change['path'], change['value']))
        )
        with pytest.raises(run_python.RunnerFailure, match='not a case'):
            run_python.load_cases([path])
    for change in listed['accepted']:
        path.write_text(
            json.dumps(_changed(case, change['path'], change['value']))
        )
        assert run_python.load_cases([path]), change['why']
    assert listed['omitted']
    for change in listed['omitted']:
        omitted = copy.deepcopy(case)
        at = omitted
        for key in change['path'][:-1]:
            at = at[key]
        del at[change['path'][-1]]
        path.write_text(json.dumps(omitted))
        assert run_python.load_cases([path]), change['why']


# --- the comparator against seeded observations ---------------------------------


def _reference() -> Path:
    [reference] = sorted(REFERENCE.glob('*.jsonl'))
    return reference


#: The public call whose result each core operation gives.
_STAGE_OF = {
    'describe_source': 'from_source',
    'validate_query': 'materialize',
    'materialize': 'materialize',
    'predict': 'predict',
}


def _unencodable(observation: dict[str, Any]) -> bool:
    r"""Whether a statement's result holds a value the canonical form
    does not, which compare.py's ``sql.result`` never takes as equal, even
    to itself."""
    return any(
        result['values']['unencodable']
        for statement in observation['driver']['capture']['statements']
        for result in statement.get('results') or []
        if 'values' in result
    )


def _not_synthesizable() -> dict[str, str]:
    r"""The acceptance cases whose reference observation cannot stand in
    for another runner's, by why: the core reaches another call than the
    reference did (an approved register entry, other than the placeholder
    P1B-1, must list the case, so the difference is held, not hidden), or
    the reference read a value the canonical form does not hold. Each is
    held by the gate's own runs, never by these syntheses; a diagnostic
    case is compared whatever it is, its differences reported."""
    reference = {
        json.loads(line)['case']: json.loads(line)
        for line in _reference().read_text(encoding='utf-8').splitlines()
    }
    entries = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))['entries']
    registered = {
        case
        for entry in entries
        if entry['id'] != 'P1B-1' and entry.get('status') == 'approved'
        for case in entry.get('cases') or []
    }
    found = {}
    for case, _ in run_python.load_cases([CASES]):
        if case['status'] != 'acceptance' or case['id'] not in reference:
            continue
        observed = reference[case['id']]
        carried = [op for op in _STAGE_OF if op in case.get('core', {})]
        reaches = _STAGE_OF[carried[-1]] if carried else None
        stage = observed['outcome']['stage']
        if reaches is not None and reaches != stage:
            assert case['id'] in registered, (
                f'{case["id"]}: the core reaches {reaches}, the reference '
                f'{stage}, and no approved entry lists it'
            )
            found[case['id']] = (
                f'the core reaches {reaches}, the reference {stage}'
            )
        elif _unencodable(observed):
            found[case['id']] = 'values the canonical form does not hold'
    return found


def _synthesizable(tmp_path: Path) -> tuple[Path, Path]:
    r"""The cases and the reference observations, as their files are,
    but those :func:`_not_synthesizable` names, and the reference's DuckDB
    lines only (the bare case IDs): the syntheses are compared in a DuckDB
    run, and a ``<case>@sqlite`` or ``<case>@postgres`` line is its
    database's run's."""
    left_out = _not_synthesizable()
    root = Path(tempfile.mkdtemp(prefix='synthesizable-', dir=tmp_path))
    cases = root / 'cases'
    cases.mkdir()
    for path in sorted(CASES.rglob('*.json')):
        if json.loads(path.read_bytes())['id'] not in left_out:
            (cases / path.name).write_bytes(path.read_bytes())
    reference = _reference()
    kept = root / 'reference' / reference.name
    kept.parent.mkdir()
    kept.write_text(
        ''.join(
            line + '\n'
            for line in reference.read_text(encoding='utf-8').splitlines()
            if '@' not in (case := json.loads(line)['case'])
            and case not in left_out
        ),
        encoding='utf-8',
    )
    return cases, kept


def _seeded(
    tmp_path: Path, change: Any, name: str = 'seeded.jsonl'
) -> subprocess.CompletedProcess[str]:
    r"""compare.py on the reference observations as ``change`` rewrites
    each, the reference itself as the oracle, over the cases whose
    reference can stand in for another runner (:func:`_synthesizable`)."""
    cases, reference = _synthesizable(tmp_path)
    lines = []
    for line in reference.read_text().splitlines():
        observation = json.loads(line)
        change(observation)
        lines.append(json.dumps(observation) + '\n')
    seeded = tmp_path / name
    seeded.write_text(''.join(lines))
    return subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            '--cases',
            str(cases),
            '--reference',
            str(reference),
            str(seeded),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_syntheses_leave_out_only_cases_held_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Seeded on fixtures of its own, whatever the corpus admits: an
    # admitted case whose core reaches another call than the reference is
    # left out under an approved entry listing it, and fails without one;
    # an admitted case whose reference holds a value the canonical form
    # does not is left out; a diagnostic one never is, nor an admitted one
    # that is neither.
    module = sys.modules[__name__]
    short, unreadable, diagnostic, plain = (
        'base.regression',
        'base.binary',
        'base.multiclass',
        'base.adventureworks',
    )
    cases = tmp_path / 'cases'
    cases.mkdir()
    for path in sorted(CASES.rglob('*.json')):
        case = json.loads(path.read_bytes())
        if case['id'] in (short, unreadable, plain):
            case['status'] = 'acceptance'
        if case['id'] == diagnostic:
            case['status'] = 'diagnostic'
        if case['id'] == short:
            # The core stopping at from_source, the reference at predict.
            case['core'] = {'describe_source': case['core']['describe_source']}
        (cases / path.name).write_text(json.dumps(case), encoding='utf-8')
    reference = tmp_path / _reference().name
    lines = []
    for line in _reference().read_text(encoding='utf-8').splitlines():
        observed = json.loads(line)
        assert observed['outcome']['stage'] == 'predict' or (
            observed['case'] not in (short, unreadable, diagnostic, plain)
        )
        if observed['case'] in (unreadable, diagnostic):
            statement = next(
                each
                for each in observed['driver']['capture']['statements']
                if any('values' in r for r in each.get('results') or [])
            )
            result = next(r for r in statement['results'] if 'values' in r)
            result['values']['unencodable'] = ['arrow:seeded']
        lines.append(json.dumps(observed) + '\n')
    reference.write_text(''.join(lines), encoding='utf-8')
    monkeypatch.setattr(module, 'CASES', cases)
    monkeypatch.setattr(module, '_reference', lambda: reference)
    register = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))
    register['entries'].append(
        {
            'id': 'SEEDED-1',
            'status': 'approved',
            'cases': [short],
        }
    )
    approved = tmp_path / 'approved.yaml'
    approved.write_text(yaml.safe_dump(register), encoding='utf-8')
    monkeypatch.setattr(module, 'REGISTER', approved)
    left_out = _not_synthesizable()
    assert left_out[short] == (
        'the core reaches from_source, the reference predict'
    )
    assert left_out[unreadable] == 'values the canonical form does not hold'
    assert diagnostic not in left_out
    assert plain not in left_out
    # No approved entry lists it: the seeded one gone, and the case taken
    # out of any other's list.
    register['entries'].pop()
    for entry in register['entries']:
        entry['cases'] = [c for c in entry.get('cases') or [] if c != short]
    unapproved = tmp_path / 'unapproved.yaml'
    unapproved.write_text(yaml.safe_dump(register), encoding='utf-8')
    monkeypatch.setattr(module, 'REGISTER', unapproved)
    with pytest.raises(AssertionError, match='no approved entry lists it'):
        _not_synthesizable()


def _as_old(observation: dict[str, Any]) -> None:
    observation['identity']['runner'] = 'python-old'
    observation['identity']['reference'] = None


def test_the_reference_relabelled_python_old_passes(tmp_path: Path) -> None:
    run = _seeded(tmp_path, _as_old)
    assert run.returncode == 0, run.stdout


def test_an_empty_observations_file_is_a_runner_failure(
    tmp_path: Path,
) -> None:
    empty = tmp_path / 'empty.jsonl'
    empty.write_text('')
    run = subprocess.run(
        [sys.executable, str(COMPARE), '--cases', str(CASES), str(empty)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 1
    assert 'empty.jsonl: runner failure: no valid observations' in run.stdout


def test_a_python_runner_may_not_leave_its_stage_or_class_null(
    tmp_path: Path,
) -> None:
    def blank(observation: dict[str, Any]) -> None:
        _as_old(observation)
        observation['outcome']['stage'] = None
        observation['outcome']['error'] = None

    run = _seeded(tmp_path, blank)
    assert run.returncode == 1
    assert 'outcome.stage: null on a Python runner' in run.stdout
    assert 'outcome.error: a refusal without a class' in run.stdout


def test_an_observation_of_another_case_file_fails(tmp_path: Path) -> None:
    def elsewhere(observation: dict[str, Any]) -> None:
        _as_old(observation)
        observation['identity']['inputs']['case'] = '0' * 64

    run = _seeded(tmp_path, elsewhere)
    assert run.returncode == 1
    assert 'identity.inputs.case: not this case file' in run.stdout

    def other_bases(observation: dict[str, Any]) -> None:
        _as_old(observation)
        observation['identity']['inputs']['bases'] = '0' * 64

    run = _seeded(tmp_path, other_bases)
    assert 'identity.inputs.bases: not this bases file' in run.stdout


def test_a_python_runner_at_another_duckdb_fails(tmp_path: Path) -> None:
    def older(observation: dict[str, Any]) -> None:
        _as_old(observation)
        observation['identity']['versions']['duckdb'] = '1.5.5'

    run = _seeded(tmp_path, older)
    assert run.returncode == 1
    assert 'identity.versions.duckdb' in run.stdout


def _as_core(accepted: bool | None, error: Any = None) -> Any:
    def change(observation: dict[str, Any]) -> None:
        case = json.loads(
            next(
                path
                for path in CASES.rglob('*.json')
                if json.loads(path.read_text())['id'] == observation['case']
            ).read_text()
        )
        identity = observation['identity']
        identity.update(
            runner='python-core',
            reference=None,
            native={'sha256': 'a' * 64},
            native_loaded=True,
            operations=sorted(case['core']),
            contract=8,
        )
        if 'describe_source' in case['core']:
            # A description, whatever the call did after it, and from T1.11c
            # a request state, for a case that materializes on the core.
            inputs = [
                {'op': 'describe_source', **case['core']['describe_source']}
            ]
            described: dict[str, Any] = {'tables': [], 'links': []}
            materialized = None
            if 'materialize' in case['core']:
                # From T1.21d the schema materialize samples is the host's
                # own description; the case names none.
                inputs.append(
                    {
                        'op': 'materialize',
                        'query': case['input']['pql'],
                        'schema': described,
                        **case['core']['materialize'],
                    }
                )
                materialized = {
                    'version': 1,
                    'query': case['input']['pql'],
                    'payload': {},
                }
            # From T1.11d, a prediction, for a case that predicts on the
            # core.
            predicted = None
            refused = None
            if 'predict' in case['core']:
                inputs.append(
                    {
                        'op': 'predict',
                        **case['core']['predict'],
                        'state': materialized,
                    }
                )
                # A result with a row: from T1.11e the comparator's
                # `reach` rule gates base.regression reaching its result.
                # The call the core wrote is the request the endpoint
                # received, the reference's, as the payload rule holds an
                # acceptance case to it (from T1.29).
                received = (observation.get('public') or {}).get(
                    'payloads'
                ) or []
                outcome = observation['outcome']
                if received:
                    predicted = {
                        'body': received[0]['parsed'],
                        'model': 'kumo-relational',
                        'request_id': 'r1',
                        'metadata': {},
                        'result': {
                            'columns': [{'name': 'ENTITY', 'type': 'int64'}],
                            'rows': [[{'i64': '0'}]],
                        },
                    }
                elif outcome['result'] == 'refusal':
                    # Refused at predict before any request was sent (T2.0c's
                    # one-class comparisons): no prediction and no request,
                    # the core's error the refusal's code, as a real
                    # python-core run records it.
                    refused = {
                        'kind': outcome['code']
                        .lower()
                        .removesuffix('_request'),
                        'code': outcome['code'],
                    }
            # A query validated in place of materialize (from T1.32g:
            # refuse.unknown-column), answered as the call was.
            validated = None
            if 'validate_query' in case['core']:
                spec = case['core']['validate_query']
                inputs.append(
                    {
                        'op': 'validate_query',
                        'query': case['input']['pql'],
                        'profile': spec['profile'],
                        'schema': spec['schema'],
                    }
                )
                validated = (
                    observation['outcome']['result'] == 'success'
                    if accepted is None
                    else accepted
                )
            observation['core'] = {
                'inputs': inputs,
                # A description is built from fetched rows: from T1.29 the
                # core rule fails one with no fetch recorded.
                'fetches': [{'request': {}, 'plan': {}}],
                'accepted': validated,
                'described': described,
                'materialized': materialized,
                'predicted': predicted,
                'error': refused,
            }
            return
        spec = case['core']['validate_query']
        observation['core'] = {
            'inputs': [
                {
                    'op': 'validate_query',
                    'query': case['input']['pql'],
                    'profile': spec['profile'],
                    'schema': spec['schema'],
                }
            ],
            'fetches': [],
            'accepted': (
                observation['outcome']['result'] == 'success'
                if accepted is None
                else accepted
            ),
            'described': None,
            'materialized': None,
            'predicted': None,
            'error': error,
        }

    return change


def test_python_cores_answer_is_held_to_its_own_outcome(
    tmp_path: Path,
) -> None:
    run = _seeded(tmp_path, _as_core(None))
    assert run.returncode == 0, run.stdout
    # Every case's core accepting, whatever the call did.
    run = _seeded(tmp_path, _as_core(True))
    assert run.returncode == 1
    assert (
        's0.refuse.unknown-column: python-core: outcome against core: '
        "expected ('success', None), observed ('refusal', 'INVALID_REQUEST')"
    ) in run.stdout


def test_materialize_samples_the_runner_s_own_description(
    tmp_path: Path,
) -> None:
    # From T1.21d no case names the schema materialize samples: a core
    # runner's is its own description, where the case makes no edit. The
    # seeded fault: a schema with a link the description does not hold.
    def other_schema(observation: dict[str, Any]) -> None:
        _as_core(None)(observation)
        for each in observation['core']['inputs']:
            if each['op'] == 'materialize':
                each['schema'] = {
                    'tables': [],
                    'links': [[{'table': 'a', 'column': 'b'}] * 2],
                }

    run = _seeded(tmp_path, _as_core(None))
    assert 'core.inputs.materialize.schema' not in run.stdout, run.stdout

    def no_description(observation: dict[str, Any]) -> None:
        _as_core(None)(observation)
        observation['core']['described'] = None

    for seeded in (other_schema, no_description):
        run = _seeded(tmp_path, seeded)
        assert (
            'inputs: base.regression: python-core: '
            'core.inputs.materialize.schema'
        ) in run.stdout, run.stdout


def test_an_index_is_an_integer_literal_of_any_size_or_text(
    tmp_path: Path,
) -> None:
    # The same literals sdm-testkit's tests/s0_cases.rs reads.
    literals = json.loads(
        (ROOT / 'contract/tests/index_literals.json').read_text()
    )
    case = json.loads((CASES / 's0/s0.accept.sum-window.json').read_text())
    case['input']['materialize']['indices'] = ['@index@']
    template = json.dumps(case)

    def load(literal: str) -> Any:
        path = tmp_path / 'case.json'
        path.write_text(template.replace('"@index@"', literal))
        [(loaded, _)] = run_python.load_cases([path])
        return loaded

    for literal in literals['accepted']:
        loaded = load(literal)
        [index] = loaded['input']['materialize']['indices']
        assert index == json.loads(literal), literal
        assert type(index) is type(json.loads(literal))
        # As a decimal index, every digit is kept.
        loaded['input']['materialize']['indices_as'] = 'decimal'
        if isinstance(index, int):
            [decimal] = run_python._materialize_arguments(
                loaded['input']['materialize']
            )['indices']
            assert decimal == Decimal(literal) and str(decimal) == literal
    for literal in literals['refused']:
        with pytest.raises(run_python.RunnerFailure):
            load(literal)


def test_an_inherited_index_is_held_to_the_same_rule(tmp_path: Path) -> None:
    inventory = yaml.safe_load(run_python.INVENTORY.read_text())
    inheriting = next(
        entry for entry in inventory['cases'] if 'materialize' not in entry
    )

    def load(indices: list[Any]) -> dict[str, Any]:
        inventory['defaults']['materialize']['indices'] = indices
        path = tmp_path / 'inventory.yaml'
        path.write_text(yaml.safe_dump(inventory))
        loaded: dict[str, Any] = run_python.load_inventory(path)
        return loaded

    with pytest.raises(run_python.RunnerFailure, match='defaults'):
        load([1.0])
    loaded = load([2**70])
    case = run_python.inventory_case(
        next(e for e in loaded['cases'] if e['id'] == inheriting['id']),
        loaded['defaults'],
    )
    [index] = case['input']['materialize']['indices']
    assert type(index) is int and index == 2**70
    path = tmp_path / 'case.json'
    path.write_text(json.dumps(case))
    [(read, _)] = run_python.load_cases([path])
    assert read['input']['materialize']['indices'] == [2**70]
