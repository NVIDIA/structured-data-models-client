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
import subprocess
import sys
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
#: are one integer or text type, at least one, and these name none, or
#: entities of a decimal key (the keys and empty-null refusal maps' gaps).
_DESCRIBE_ONLY = {
    'empty.indices',
    'key.hugeint',
    'key.decimal-18-0',
    'key.decimal-38-10',
}


def _anchor(text: str) -> dict[str, Any]:
    import datetime

    when = datetime.datetime.fromisoformat(text)
    epoch = datetime.datetime(1970, 1, 1)
    nanos = (when - epoch) // datetime.timedelta(microseconds=1) * 1000
    return {'t': 'ts', 'v': str(nanos), 'unit': 'ns', 'zone': None}


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
            t: [k] for t, k in named.get('primary_keys', {}).items()
        },
        'time_columns': named.get('time_columns', {}),
    }:
        problems.append('describe_source is not of its source')
    materialize = core.get('materialize')
    if (materialize is None) != (case['id'] in _DESCRIBE_ONLY):
        problems.append('materialize present where it cannot be, or absent')
    if materialize is not None:
        call = given['materialize']
        if [index['v'] for index in materialize['indices']] != [
            str(index) for index in call['indices']
        ]:
            problems.append("materialize's indices are not the call's")
        wanted = _anchor(call['anchor_time']) if 'anchor_time' in call else None
        if materialize['anchor_time'] != wanted:
            problems.append("materialize's anchor time is not the call's")
    if ('predict' in core) != (materialize is not None) or core.get(
        'predict', {'model': 'kumo-relational'}
    ) != {'model': 'kumo-relational'}:
        problems.append('predict does not follow materialize')
    return problems


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
    committed = {
        case['id']: case for case, _ in run_python.load_cases([CASES / 's1'])
    }
    assert set(committed) == {entry['id'] for entry in inventory['cases']} | {
        'base.adventureworks'
    }
    for entry in inventory['cases']:
        case = committed[entry['id']]
        assert case['status'] == 'diagnostic', entry['id']
        assert _held_to_its_input(case) == [], entry['id']
        if entry['id'] == 'base.regression':
            mapped = run_python.inventory_case(entry, defaults)
            for key in ('input', 'expect', 'seam', 'cells', 'until'):
                assert case[key] == mapped[key], key
            continue
        held = {key: value for key, value in case.items() if key != 'core'}
        assert run_python.inventory_entry(held, defaults) == entry, entry['id']
    # Seeded: a case changed alone departs from the inventory, and a core
    # changed alone from its input.
    case = copy.deepcopy(committed['time.text-declared'])
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
        case = copy.deepcopy(committed['time.text-declared'])
        change(case['core'])
        assert _held_to_its_input(case) != [], change


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
    # The two date cases DT-9 held back stay diagnostic (a changed case
    # file needs an approved entry), and from T1.6b each has an acceptance
    # twin of the same input: D8's packed date and DT-9's long field.
    dated = [case for case in cases.values() if case.get('register') == 'DT-9']
    assert len(dated) == 2
    assert {case['status'] for case in dated} == {'diagnostic'}
    for case in dated:
        name = case['id'].removeprefix('s0.boundary.')
        twin = cases[f's0.accept.{name}']
        assert twin['status'] == 'acceptance' and 'register' not in twin
        assert twin['input'] == case['input'], name
        assert twin['core'] == case['core'], name
        assert twin['expect'] == {'outcome': 'success', 'stage': 'materialize'}
    # Escapes as written in the query text, a backslash and its letters: a
    # \\u escape and no surrogate in the accepted case; an escaped pair and
    # a lone escaped surrogate in the two cases register PQL-1 holds.
    escapes = pql('escapes')
    assert '\\u00e9' in escapes and '\\ud' not in escapes
    assert escapes.isascii()
    assert '\\ud83d\\ude00' in pql('escape-surrogate-pair')
    lone = pql('escape-lone-surrogate')
    assert '\\ud83d' in lone and '\\ude' not in lone
    # Every other diagnostic S0 case names a proposed register entry.
    register = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))
    status = {entry['id']: entry['status'] for entry in register['entries']}
    for case in cases.values():
        if case['status'] == 'diagnostic' and case.get('register') != 'DT-9':
            assert status[case['register']] == 'proposed', case['id']
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
    run = subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            '--cases',
            str(CASES),
            '--reference',
            str(reference),
            str(reference),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    # A corrupted expectation fails.
    corrupted = tmp_path / 'cases'
    corrupted.mkdir()
    for path in CASES.rglob('*.json'):
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
            str(reference),
            str(reference),
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


def _seeded(
    tmp_path: Path, change: Any, name: str = 'seeded.jsonl'
) -> subprocess.CompletedProcess[str]:
    r"""compare.py on the reference observations as ``change`` rewrites
    each, the reference itself as the oracle."""
    reference = _reference()
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
            str(CASES),
            '--reference',
            str(reference),
            str(seeded),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


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
                predicted = {
                    'body': {},
                    'model': 'kumo-relational',
                    'request_id': 'r1',
                    'metadata': {},
                    'result': {
                        'columns': [{'name': 'ENTITY', 'type': 'int64'}],
                        'rows': [[{'i64': '0'}]],
                    },
                }
            observation['core'] = {
                'inputs': inputs,
                'fetches': [],
                'accepted': None,
                'described': described,
                'materialized': materialized,
                'predicted': predicted,
                'error': None,
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
