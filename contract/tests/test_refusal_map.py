# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E27's AdventureWorks case: its fixture, and its refusal map on python-core
(T1.21b); and the Phase 1b inventory's family maps (T1.21e).

Run where the contract tests run (CI's ``engine`` job until T1.18):

    python -m pytest contract/tests -q

- The fixture: the ``adventureworks`` base of ``contract/bases.yaml`` holds
  1,000 rows in all, at most 2,000, as its README's table says, and the
  case asks for the product the notebook asks for, the most units sold.
- The map (``contract/refusal-maps/base.adventureworks.json``): every entry
  names its rule and operation, an owning task among T1.21c, T1.21d,
  T1.22 to T1.27, T1.28a and T1.28b (E50, and T1.28's split at T1.12b) and
  a register entry that is proposed, and the line
  it cites holds the rule;
  each probe's input is a case input, its own edits aside; the line it
  cites holds the rule (a core refusal) or the text it names (a bridge's or
  the seam's); and every probe is the case with the removals it names
  (``refusal_map.bound``), so a change to the case alone fails, seeded.
- The run: every python-core probe of the map, run in process with its
  routes forced, is the map's (``refusal_map.check``): a refusal the map
  does not list fails, and so does an entry no probe reaches. Where the
  probe reaches the core, the core operations it starts are the rust-host
  probe's, so the two runners are held to one map with the same inputs
  (``sdm-testkit``'s ``tests/refusal_map.rs`` runs rust-host's).
- The check fails closed: a seeded fault for each way the evidence can be
  missing, empty or retyped, and for an unlisted refusal and an entry
  nothing hits; the map's reader refuses JSON nested past what it reads,
  and a wide object, by name.
- The family maps (``contract/refusal-maps/<seam>.json``, T1.21e): one per
  seam of the inventory, its cases every committed case of that seam;
  every entry owned (T1.21c to T1.27, T1.28a, T1.28b, T1.32a to T1.32e;
  the writer refuses any other owner), its line cited,
  and either under a proposed register entry or the reference's own
  refusal, held to the case's reference observation (seeded); every probe
  bound to its case (seeded: a case changed alone, a case left without a
  probe, a probe of another family's case); each map held on python-core
  (seeded: an unlisted refusal, a stale entry, retyped evidence) and
  ``refusal_map.write``'s output from that run, so no map is edited by
  hand; the writer reports a new refusal and drops a lifted entry.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import duckdb
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / 'contract/runners'
CASE = ROOT / 'contract/cases/s1/base.adventureworks.json'
MAP = ROOT / 'contract/refusal-maps/base.adventureworks.json'
PIN = ROOT / 'contract/pinned/base.adventureworks.probes.json'
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'
README = ROOT / 'contract/fixtures/adventureworks/README.md'


def _module(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUNNERS / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_python = _module('run_python')
refusal_map = _module('refusal_map')


def _case() -> dict[str, Any]:
    [(case, _)] = run_python.load_cases([CASE])
    return case


def _map() -> dict[str, Any]:
    document: dict[str, Any] = refusal_map.load(MAP)
    return document


# --- the fixture ------------------------------------------------------------------


def _adventureworks() -> Any:
    connection = duckdb.connect()
    for statement in run_python.load_bases()['adventureworks']:
        connection.execute(statement)
    return connection


def test_the_fixture_holds_its_readme_s_rows_and_at_most_2000() -> None:
    connection = _adventureworks()
    counts = {
        table: connection.execute(
            f'SELECT count(*) FROM adventureworks_{table}'
        ).fetchone()[0]
        for table in (
            'customers',
            'products',
            'sales_order_headers',
            'sales_order_details',
        )
    }
    assert sum(counts.values()) == 1000 <= 2000
    readme = README.read_text(encoding='utf-8')
    for table, count in counts.items():
        assert f'| `adventureworks_{table}` | {count} |' in readme
    # The licence travels with the data, as the source carries it.
    licence = (README.parent / 'LICENSE.txt').read_text(encoding='utf-8')
    assert 'MIT License.' in licence
    assert 'Copyright (c) Microsoft Corporation' in licence


def test_the_case_asks_for_the_product_most_sold() -> None:
    # The notebook's own query for top_product, over the fixture.
    [(top,)] = (
        _adventureworks()
        .execute(
            'SELECT ProductID FROM adventureworks_sales_order_details '
            'GROUP BY ProductID ORDER BY sum(OrderQty) DESC, ProductID LIMIT 1'
        )
        .fetchall()
    )
    assert _case()['input']['materialize']['indices'] == [top]


# --- the map ----------------------------------------------------------------------


def test_every_entry_names_its_rule_operation_owner_and_proposed_entry() -> (
    None
):
    document = _map()
    assert document['case'] == _case()['id']
    register = {
        entry['id']: entry
        for entry in yaml.safe_load(REGISTER.read_text(encoding='utf-8'))[
            'entries'
        ]
    }
    for entry in document['entries']:
        assert refusal_map.OWNER.fullmatch(entry['owner']), entry['id']
        # Never self-approved: the entry the refusal is under waits for the
        # owner.
        assert register[entry['register']]['status'] == 'proposed', entry['id']
        path, _, lines = entry['where'].partition(':')
        assert (ROOT / path).is_file(), entry['id']
        first, last = re.match(r'(\d+)(?:-(\d+))?', lines).groups()
        cited = (
            (ROOT / path)
            .read_text()
            .splitlines()[int(first) - 1 : int(last or first)]
        )
        if path.startswith('rust/'):
            # A core refusal: the line cited names the rule.
            assert any(
                f'not_yet("{entry["rule"]}")' in text for text in cited
            ), entry['id']
        else:
            # A refusal of the Python bridge or the seam: the lines cited
            # hold what the entry says they do.
            assert entry['holds'], entry['id']
            assert any(entry['holds'] in text for text in cited), entry['id']
    for gap in document['gaps']:
        assert refusal_map.OWNER.fullmatch(gap['owner'])
    # Each probe's python-core input is a case input, but for the edits
    # only a probe makes (a key or the relationships set).
    validate = run_python.validator('case')
    for probe in document['probes']:
        if 'python-core' in probe:
            case = refusal_map.probe_input(_case(), probe['python-core'])
            assert run_python._invalid(validate, case) is None, probe['id']


def test_the_probes_are_the_case_s_journey_with_the_removals_named() -> None:
    # The first probe of each runner is the case itself, and every other is
    # an earlier probe with the removals it names (refusal_map.bound).
    document = _map()
    assert refusal_map.bound(_case(), document) == []
    first = document['probes'][0]
    assert first['python-core']['input'] == _case()['input']
    assert first['rust-host']['core'] == _case()['core']


def test_a_rust_host_probe_names_no_schema_and_its_twin_s_edits() -> None:
    # The rust-host journey guard (T1.21d): materialize samples rust-host's
    # own description with its twin's edits made, so a probe naming the
    # schema, or edits other than its twin's, fails the binding, seeded.
    document = _map()
    probe = next(
        p
        for p in document['probes']
        if 'materialize' in p.get('rust-host', {}).get('core', {})
    )
    name = probe['id']
    probe['rust-host']['core']['materialize']['schema'] = {
        'tables': [],
        'links': [],
    }
    assert f'{name}/rust-host: names the schema materialize samples' in (
        refusal_map.bound(_case(), document)
    )
    for change in (
        lambda edits: edits.pop(),
        lambda edits: edits.clear(),
        lambda edits: edits.append({'op': 'clear_time_column', 'table': 't'}),
    ):
        document = _map()
        probe = next(p for p in document['probes'] if p['id'] == name)
        change(probe['rust-host']['edits'])
        assert f"{name}/rust-host: not its twin's edits" in (
            refusal_map.bound(_case(), document)
        )
    document = _map()
    probe = next(p for p in document['probes'] if p['id'] == name)
    del probe['rust-host']['edits']
    assert f"{name}/rust-host: not its twin's edits" in (
        refusal_map.bound(_case(), document)
    )


def _case_changed(change: Any) -> dict[str, Any]:
    case = _case()
    change(case)
    return case


@pytest.mark.parametrize(
    'change',
    [
        lambda case: case['input']['edits'].pop(),
        lambda case: case['input'].__setitem__(
            'pql', case['input']['pql'] + ' '
        ),
        lambda case: case['input']['materialize'].pop('inference_config'),
        lambda case: case['input']['from_source']['relationships'].pop(),
        lambda case: case['input']['from_source']['primary_keys'].pop(
            'products'
        ),
        lambda case: case['input']['from_source']['tables'].__setitem__(
            'products', 'other_products'
        ),
        lambda case: case.__setitem__('until', 'materialize'),
        lambda case: case['core']['describe_source']['source'][
            'time_columns'
        ].clear(),
    ],
)
def test_a_change_to_the_case_alone_fails_the_map(change: Any) -> None:
    # Seeded: the case changes and the map does not, so the map would test
    # another journey than the case's; the binding refuses it.
    assert refusal_map.bound(_case_changed(change), _map()) != []


def test_a_probe_without_its_derivation_fails() -> None:
    document = _map()
    del document['probes'][3]['python-core']['derive']
    failures = refusal_map.bound(_case(), document)
    # And every probe after it, which derives from it, fails too.
    assert failures[0] == 'no-table-names/python-core: no derive'
    assert 'no-relationships/python-core: derives from no probe' in failures
    document = _map()
    document['probes'][4]['python-core']['input']['setup'] = ['SELECT 1']
    assert 'no-relationships/python-core: not its derived input' in (
        refusal_map.bound(_case(), document)
    )


def test_the_python_core_run_is_held_to_the_map() -> None:
    document = _map()
    bases = run_python.load_bases()
    observed = {}
    digests: dict[str, dict[str, str]] = {}
    for probe in document['probes']:
        spec = probe.get('python-core')
        if spec is None:
            continue
        seen, capture = refusal_map.observe_python_core(run_python, spec, bases)
        observed[probe['id']] = seen
        rust = probe.get('rust-host')
        if rust is None or not capture.inputs:
            continue
        # Where both hosts reach describe_source, python-core's description
        # and the schema its materialize samples are the pin's, which
        # rust-host is held to (T1.21d).
        if capture.described is not None:
            agreed = {
                'described': refusal_map.digest(capture.described),
                'warnings': refusal_map.digest(capture.warnings),
            }
            for each in capture.inputs:
                if each['op'] == 'materialize':
                    agreed['materialize_schema'] = refusal_map.digest(
                        each['schema']
                    )
            digests[probe['id']] = agreed
        # Where python-core reached the core, it started the operations the
        # rust-host probe starts, with the same inputs.
        started = {each['op']: each for each in capture.inputs}
        assert set(started) <= set(rust['core']), probe['id']
        for op, given in started.items():
            wanted = rust['core'][op]
            assert {key: given[key] for key in wanted} == wanted, probe['id']
            if op == 'materialize':
                assert given['query'] == rust['pql'], probe['id']
    assert refusal_map.check(document, 'python-core', observed) == []
    # The ladder ends at predict's result.
    assert refusal_map.success('predict') in observed.values()
    assert refusal_map.agree(_pin(), digests) == []
    assert len(digests) >= 4 and any(
        'materialize_schema' in each for each in digests.values()
    )


def _pin() -> dict[str, Any]:
    pinned: dict[str, Any] = json.loads(PIN.read_text(encoding='utf-8'))
    return pinned


def test_the_hosts_descriptions_are_held_to_the_pin_fail_closed() -> None:
    # The pin python-core and rust-host are both held to: seeded, each way
    # the evidence can be missing, empty, retyped or other than the pin.
    pin = _pin()
    seen = copy.deepcopy(pin['probes'])
    assert refusal_map.agree(pin, seen) == []
    assert refusal_map.agree(pin, {}) != []
    assert refusal_map.agree({}, seen) == ['the pin lists no probe']
    assert refusal_map.agree({'probes': {}}, seen) == ['the pin lists no probe']
    name = sorted(seen)[0]
    other = copy.deepcopy(seen)
    other[name]['described'] = '0' * 64
    assert refusal_map.agree(pin, other) == [f"{name}: described not the pin's"]
    missing = copy.deepcopy(seen)
    del missing[name]
    assert refusal_map.agree(pin, missing) == [
        f'{name}: pinned, and describe_source not reached'
    ]
    extra = {**copy.deepcopy(seen), 'not-a-probe': seen[name]}
    assert refusal_map.agree(pin, extra) == [
        'not-a-probe: reached describe_source, not pinned'
    ]
    for retyped in (7, None, '', 'abc', ['x'] * 64):
        broken = copy.deepcopy(pin)
        broken['probes'][name]['described'] = retyped
        assert refusal_map.agree(broken, seen) == [
            f'{name}: the pin is not digests'
        ], retyped
    dropped = copy.deepcopy(pin)
    del dropped['probes'][name]['described']
    assert refusal_map.agree(dropped, seen) == [
        f'{name}: the pin is not digests'
    ]
    # The digest is of canonical JSON: key order and spacing do not move it.
    assert refusal_map.digest({'b': 1, 'a': [1, 'é']}) == refusal_map.digest(
        json.loads('{ "a" : [1, "\\u00e9"], "b": 1 }')
    )


# --- the check fails closed --------------------------------------------------------


def _observed() -> dict[str, Any]:
    r"""A run the map passes, built from the map itself."""
    document = _map()
    entries = {entry['id']: entry for entry in document['entries']}
    return {
        probe['id']: (
            refusal_map.success(probe['python-core']['until'])
            if probe['python-core']['expect'] == 'success'
            else copy.deepcopy(
                entries[probe['python-core']['expect']]['observed'][
                    'python-core'
                ]
            )
        )
        for probe in document['probes']
        if 'python-core' in probe
    }


def test_the_built_run_passes_so_the_faults_below_are_the_faults() -> None:
    assert refusal_map.check(_map(), 'python-core', _observed()) == []


def test_a_refusal_the_map_does_not_list_fails() -> None:
    document = _map()
    document['entries'] = [
        entry
        for entry in document['entries']
        if entry['id'] != 'describe_source/text_time'
    ]
    failures = refusal_map.check(document, 'python-core', _observed())
    assert any(
        failure.startswith('no-time-columns: a refusal the map does not list')
        for failure in failures
    ), failures


def test_an_entry_no_probe_hits_fails() -> None:
    document = _map()
    seeded = copy.deepcopy(document['entries'][-1])
    seeded['id'] = 'materialize/seeded'
    seeded['observed']['python-core']['rule'] = 'seeded'
    document['entries'].append(seeded)
    assert refusal_map.check(document, 'python-core', _observed()) == [
        'materialize/seeded: listed for python-core and not hit'
    ]


@pytest.mark.parametrize('retyped', [7, None, '', ['text_time'], {'x': 1}])
def test_retyped_evidence_fails(retyped: Any) -> None:
    observed = _observed()
    observed['no-time-columns']['rule'] = retyped
    assert refusal_map.check(_map(), 'python-core', observed) != []


def test_missing_or_empty_evidence_fails() -> None:
    document = _map()
    assert refusal_map.check(document, 'python-core', {}) == [
        'python-core: nothing was observed'
    ]
    observed = _observed()
    del observed['no-table-names']
    failures = refusal_map.check(document, 'python-core', observed)
    assert 'no-table-names: not observed' in failures
    assert (
        'from_source/relationships: listed for python-core and not hit'
        in failures
    )
    observed = _observed()
    del observed['no-table-names']['rule']
    assert refusal_map.check(document, 'python-core', observed) != []
    observed = _observed()
    observed['not-a-probe'] = refusal_map.success('predict')
    assert refusal_map.check(document, 'python-core', observed) == [
        'not-a-probe: observed, and not a probe of python-core'
    ]
    empty = copy.deepcopy(document)
    empty['probes'] = [
        probe for probe in empty['probes'] if 'python-core' not in probe
    ]
    assert refusal_map.check(empty, 'python-core', _observed()) == [
        'python-core: the map gives this runner no probe'
    ]


def test_a_probe_that_succeeds_or_hits_another_entry_fails() -> None:
    document = _map()
    observed = _observed()
    observed['no-table-names'] = copy.deepcopy(observed['no-time-columns'])
    assert (
        'no-table-names: hit describe_source/text_time, where the map has '
        'from_source/relationships'
    ) in refusal_map.check(document, 'python-core', observed)
    observed = _observed()
    observed['one-hop'] = refusal_map.success('predict')
    assert (
        'one-hop: succeeded, where the map has materialize/text_column'
    ) in refusal_map.check(document, 'python-core', observed)


# --- the map's reader ---------------------------------------------------------------


def test_the_reader_refuses_deep_and_wide_maps_by_name(tmp_path: Path) -> None:
    deep = tmp_path / 'deep.json'
    deep.write_text('[' * 100_000 + ']' * 100_000)
    with pytest.raises(refusal_map.MapFailure, match='nested too deeply'):
        refusal_map.load(deep)
    nested = _map()
    nested['entries'][0]['observed']['python-core']['rule'] = 'NESTED'
    inside = tmp_path / 'inside.json'
    inside.write_text(
        json.dumps(nested).replace('"NESTED"', '[' * 5_000 + ']' * 5_000)
    )
    with pytest.raises(refusal_map.MapFailure, match='neither text nor null'):
        refusal_map.load(inside)
    wide = tmp_path / 'wide.json'
    wide.write_text(json.dumps({f'k{i}': i for i in range(200_000)}))
    with pytest.raises(refusal_map.MapFailure, match='not sdm-refusal-map/1'):
        refusal_map.load(wide)
    for broken in ('[]', '{"format": "sdm-refusal-map/1"}', 'not json'):
        path = tmp_path / 'broken.json'
        path.write_text(broken)
        with pytest.raises(refusal_map.MapFailure):
            refusal_map.load(path)


def test_the_owners_a_map_may_name() -> None:
    # E50 (T1.10e): T1.21c and T1.21d join T1.22 to T1.28; T1.21e adds the
    # inventory's families, T1.32a to T1.32e, and T1.28's halves, which
    # replace it (T1.12b); their neighbours, a landed package, a later one
    # and the T1.28 the split left behind stay refused.
    for task in (
        'T1.21c',
        'T1.21d',
        'T1.22',
        'T1.27',
        'T1.28a',
        'T1.28b',
        'T1.32a',
        'T1.32e',
    ):
        assert refusal_map.OWNER.fullmatch(task), task
    for task in (
        'T1.21',
        'T1.21b',
        'T1.21e',
        'T1.21cd',
        'T1.28',
        'T1.28c',
        'T1.29',
        'T1.32',
        'T1.32f',
        'T1.2',
        '',
    ):
        assert not refusal_map.OWNER.fullmatch(task), task


# --- the inventory's family maps (T1.21e) ------------------------------------------

MAPS = ROOT / 'contract/refusal-maps'
REFERENCE = ROOT / 'contract/reference'


def _family_maps() -> dict[str, dict[str, Any]]:
    return {
        path.stem: refusal_map.load(path)
        for path in sorted(MAPS.glob('*.json'))
        if path != MAP
    }


def _cases() -> dict[str, dict[str, Any]]:
    return {
        case['id']: case
        for case, _ in run_python.load_cases([ROOT / 'contract/cases'])
    }


def _of(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    every = _cases()
    return {case_id: every[case_id] for case_id in document['cases']}


def test_one_family_map_per_seam_holds_every_inventory_case() -> None:
    # Each of the inventory's seams has its map, whose cases are the
    # committed cases of that seam, the inventory's 40 among them, and
    # AdventureWorks has its own.
    maps = _family_maps()
    assert set(maps) == set(refusal_map.FAMILIES)
    inventory = {entry['id'] for entry in run_python.load_inventory()['cases']}
    held: list[str] = []
    for name, document in maps.items():
        assert document['family'] == name
        seam = {
            case_id
            for case_id, case in _cases().items()
            if case.get('seam') == name and case_id != 'base.adventureworks'
        }
        assert set(document['cases']) == seam, name
        held += document['cases']
    assert sorted(held) == sorted(inventory) and len(held) == 40


def _cited(entry: dict[str, Any]) -> list[str]:
    path, _, lines = entry['where'].partition(':')
    assert (ROOT / path).is_file(), entry['id']
    first, last = re.match(r'(\d+)(?:-(\d+))?', lines).groups()
    text: list[str] = (
        (ROOT / path)
        .read_text()
        .splitlines()[int(first) - 1 : int(last or first)]
    )
    return text


def _reference_outcomes() -> dict[str, dict[str, Any]]:
    [reference] = sorted(REFERENCE.glob('*.jsonl'))
    return {
        observation['case']: observation['outcome']
        for observation in map(json.loads, reference.read_text().splitlines())
    }


def _not_the_reference_s(
    document: dict[str, Any],
    entry: dict[str, Any],
    reference: dict[str, dict[str, Any]],
) -> list[str]:
    r"""Each way an entry marked the reference's own refusal is not: it
    names a register entry, no probe hits it on python-core, or a case
    whose probe hits it was not refused by the reference at the same
    stage, with the same class and code."""
    problems = []
    if entry['register'] != 'none':
        problems.append('names a register entry')
    seen = entry['observed'].get('python-core')
    hitting = [
        probe['case']
        for probe in document['probes']
        if probe.get('python-core', {}).get('expect') == entry['id']
    ]
    if seen is None or not hitting:
        problems.append('no python-core probe hits it')
    for case_id in hitting:
        outcome = reference.get(case_id)
        if outcome is None or outcome.get('result') != 'refusal':
            problems.append(f'{case_id}: the reference did not refuse it')
        elif any(
            outcome.get(key) != (seen or {}).get(key)
            for key in ('stage', 'error', 'code')
        ):
            problems.append(f"{case_id}: not the reference's refusal")
    return problems


def test_an_entry_called_the_reference_s_is_held_to_it_fail_closed() -> None:
    document = _family_maps()['keys']
    entry = next(e for e in document['entries'] if e.get('reference'))
    reference = _reference_outcomes()
    assert _not_the_reference_s(document, entry, reference) == []
    case_id = 'key.uint64-above-int64-max'
    for change, problem in (
        (lambda r: r.pop(case_id), 'the reference did not refuse it'),
        (
            lambda r: r[case_id].__setitem__('result', 'success'),
            'the reference did not refuse it',
        ),
        (
            lambda r: r[case_id].__setitem__('code', 'INTERNAL_ERROR'),
            "not the reference's refusal",
        ),
        (
            lambda r: r[case_id].__setitem__('stage', None),
            "not the reference's refusal",
        ),
    ):
        changed = copy.deepcopy(reference)
        change(changed)
        assert _not_the_reference_s(document, entry, changed) == [
            f'{case_id}: {problem}'
        ]
    named = copy.deepcopy(entry)
    named['register'] = 'P1B-3'
    assert _not_the_reference_s(document, named, reference) == [
        'names a register entry'
    ]
    unhit = copy.deepcopy(entry)
    unhit['id'] = 'materialize/unhit'
    assert _not_the_reference_s(document, unhit, reference) == [
        'no python-core probe hits it'
    ]


def test_every_family_entry_is_owned_cited_and_registered_or_the_reference_s() -> (
    None
):
    register = {
        entry['id']: entry
        for entry in yaml.safe_load(REGISTER.read_text(encoding='utf-8'))[
            'entries'
        ]
    }
    reference = _reference_outcomes()
    validate = run_python.validator('case')
    for name, document in _family_maps().items():
        cases = _of(document)
        for entry in document['entries']:
            where = f'{name}: {entry["id"]}'
            assert refusal_map.OWNER.fullmatch(entry['owner']), where
            cited = _cited(entry)
            if 'holds' in entry:
                assert any(entry['holds'] in text for text in cited), where
            else:
                assert any(
                    f'not_yet("{entry["rule"]}")' in text for text in cited
                ), where
            if entry.get('reference') == 'same':
                assert _not_the_reference_s(document, entry, reference) == [], (
                    where
                )
            else:
                # Never self-approved: the entry the refusal is under
                # waits for the owner.
                assert register[entry['register']]['status'] == 'proposed', (
                    where
                )
        for gap in document['gaps']:
            assert refusal_map.OWNER.fullmatch(gap['owner']), name
        for probe in document['probes']:
            if 'python-core' in probe:
                case = refusal_map.probe_input(
                    cases[probe['case']], probe['python-core']
                )
                assert run_python._invalid(validate, case) is None, probe['id']


def test_the_family_maps_probes_are_their_cases_with_the_removals_named() -> (
    None
):
    for name, document in _family_maps().items():
        assert refusal_map.bound(_of(document), document) == [], name


@pytest.mark.parametrize(
    'change',
    [
        lambda case: case['input']['setup'].pop(),
        lambda case: case['input'].__setitem__(
            'pql', case['input']['pql'] + ' '
        ),
        lambda case: case['input']['from_source']['time_columns'].clear(),
        lambda case: case.__setitem__('until', 'materialize'),
        lambda case: case['core']['materialize']['indices'].pop(),
        lambda case: case['core'].pop('predict'),
    ],
)
def test_a_change_to_a_family_case_alone_fails_its_map(change: Any) -> None:
    document = _family_maps()['time']
    cases = _of(document)
    change(cases['time.text-declared'])
    assert refusal_map.bound(cases, document) != []


def test_a_family_map_without_a_case_s_probe_or_with_a_stranger_fails() -> None:
    document = _family_maps()['nested']
    cases = _of(document)
    dropped = copy.deepcopy(document)
    dropped['probes'] = [
        probe for probe in dropped['probes'] if probe['case'] != 'nested.map'
    ]
    failures = refusal_map.bound(cases, dropped)
    assert 'nested.map/python-core: no probe of the case' in failures
    assert 'nested.map/rust-host: no probe of the case' in failures
    stranger = copy.deepcopy(document)
    stranger['probes'][0]['case'] = 'base.regression'
    assert refusal_map.shape(stranger) == 'probes/0 names no case of the family'
    for broken, problem in (
        ({'family': 'no-such-seam'}, 'not a family of the inventory'),
        ({'cases': []}, 'cases are not each a case, once'),
        (
            {'cases': ['nested.map', 'nested.map']},
            'cases are not each a case, once',
        ),
        ({'case': 'nested.map'}, 'neither of one case nor of a family'),
    ):
        assert refusal_map.shape({**document, **broken}) == problem, problem
    single = _map()
    single['probes'][0]['case'] = 'base.regression'
    assert refusal_map.shape(single) == (
        "probes/0 names another case than the map's"
    )


def _python_core_run(
    document: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    r"""Each python-core probe of ``document`` run: what it observed, and
    its observation by its input, routes and stage."""
    bases = run_python.load_bases()
    observed: dict[str, Any] = {}
    by_input: dict[str, Any] = {}
    for probe in document['probes']:
        spec = probe.get('python-core')
        if spec is None:
            continue
        seen, capture = refusal_map.observe_python_core(run_python, spec, bases)
        observed[probe['id']] = seen
        by_input[_spec_key(spec)] = seen
        rust = probe.get('rust-host')
        if rust is None:
            continue
        # Where python-core reached the core, it started the operations the
        # rust-host probe starts, with the same inputs.
        started = {each['op']: each for each in capture.inputs}
        assert set(started) <= set(rust['core']), probe['id']
        for op, given in started.items():
            wanted = rust['core'][op]
            assert {key: given[key] for key in wanted} == wanted, probe['id']
    return observed, by_input


def _spec_key(spec: dict[str, Any]) -> str:
    return json.dumps(
        [spec['input'], spec['routes'], spec['until']], sort_keys=True
    )


@pytest.mark.parametrize('family', refusal_map.FAMILIES)
def test_the_family_map_holds_on_python_core_and_is_the_writer_s(
    family: str,
) -> None:
    document = _family_maps()[family]
    observed, by_input = _python_core_run(document)
    assert refusal_map.check(document, 'python-core', observed) == []
    # Generated, not edited: the writer, given this run, makes the map as
    # committed, its rust-host observations kept.
    written, report = refusal_map.write(
        document, _of(document), lambda spec: by_input[_spec_key(spec)]
    )
    assert report == []
    assert written == document


def test_a_family_map_fails_on_an_unlisted_refusal_and_a_stale_entry() -> None:
    # Seeded on python-core's run of the time family: the entry a probe
    # hits removed, and an entry no probe hits added.
    document = _family_maps()['time']
    entries = {entry['id']: entry for entry in document['entries']}
    observed = {
        probe['id']: (
            refusal_map.success(probe['python-core']['until'])
            if probe['python-core']['expect'] == 'success'
            else copy.deepcopy(
                entries[probe['python-core']['expect']]['observed'][
                    'python-core'
                ]
            )
        )
        for probe in document['probes']
    }
    assert refusal_map.check(document, 'python-core', observed) == []
    unlisted = copy.deepcopy(document)
    unlisted['entries'] = [
        entry
        for entry in unlisted['entries']
        if entry['id'] != 'materialize/wire_dtype'
    ]
    failures = refusal_map.check(unlisted, 'python-core', observed)
    assert any(
        failure.startswith(
            'time.feature-time-of-day: a refusal the map does not list'
        )
        for failure in failures
    ), failures
    stale = copy.deepcopy(document)
    seeded = copy.deepcopy(stale['entries'][0])
    seeded['id'] = 'materialize/stale'
    seeded['observed']['python-core']['rule'] = 'stale'
    stale['entries'].append(seeded)
    assert refusal_map.check(stale, 'python-core', observed) == [
        'materialize/stale: listed for python-core and not hit'
    ]
    for retyped in (7, None, '', ['wire_dtype']):
        changed = copy.deepcopy(observed)
        changed['time.feature-time-of-day']['rule'] = retyped
        assert refusal_map.check(document, 'python-core', changed) != []


def test_the_writer_lists_a_new_refusal_and_drops_a_lifted_entry() -> None:
    # The writer over a seeded run: a refusal no entry lists becomes an
    # entry whose written fields read UNSET, reported; an entry nothing
    # hits any more is dropped, reported.
    document = _family_maps()['nested']
    entries = {entry['id']: entry for entry in document['entries']}
    runs = {}
    for probe in document['probes']:
        spec = probe['python-core']
        runs[_spec_key(spec)] = (
            refusal_map.success(spec['until'])
            if spec['expect'] == 'success'
            else entries[spec['expect']]['observed']['python-core']
        )
    new = {
        'stage': 'materialize',
        'error': 'UnsupportedRequestError',
        'code': 'UNSUPPORTED_REQUEST',
        'operation': 'materialize',
        'rule': 'seeded',
    }
    first = document['probes'][0]['python-core']
    runs[_spec_key(first)] = new
    written, report = refusal_map.write(
        document, _of(document), lambda spec: runs[_spec_key(spec)]
    )
    assert (
        f'{document["probes"][0]["id"]}/python-core: a refusal no entry lists'
        in report
    )
    added = next(
        e for e in written['entries'] if e['id'] == 'materialize/seeded'
    )
    assert added['owner'] == refusal_map.UNSET
    assert 'materialize/seeded: owner is not written' in report
    for probe in document['probes']:
        runs[_spec_key(probe['python-core'])] = refusal_map.success(
            probe['python-core']['until']
        )
    written, report = refusal_map.write(
        document, _of(document), lambda spec: runs[_spec_key(spec)]
    )
    # The rust-host observations stay, so the entry is kept for rust-host
    # alone; with rust-host's run lifted too, it is dropped.
    assert 'python-core' not in written['entries'][0]['observed']
    rust = {
        probe['id']: refusal_map.success(probe['rust-host']['until'])
        for probe in document['probes']
    }
    written, report = refusal_map.write(
        document, _of(document), lambda spec: runs[_spec_key(spec)], rust
    )
    assert written['entries'] == []
    assert 'describe_source/column_type: no probe hits it; dropped' in report
    # Evidence that is not an observation is reported, never taken.
    name = document['probes'][0]['id']
    for retyped in (None, 7, [], {'code': 'X'}, {**new, 'rule': ''}):
        runs[_spec_key(first)] = retyped
        _, report = refusal_map.write(
            document, _of(document), lambda spec: runs[_spec_key(spec)]
        )
        assert any(line.startswith(f'{name}/python-core: ') for line in report)


def _rust_run(document: dict[str, Any]) -> dict[str, Any]:
    r"""rust-host's run of ``document`` as the map records it."""
    entries = {entry['id']: entry for entry in document['entries']}
    return {
        probe['id']: (
            refusal_map.success(probe['rust-host']['until'])
            if probe['rust-host']['expect'] == 'success'
            else entries[probe['rust-host']['expect']]['observed']['rust-host']
        )
        for probe in document['probes']
        if 'rust-host' in probe
    }


def test_supplied_rust_evidence_is_the_whole_of_it() -> None:
    # Codex P2-1 (fix round 1): rust-host evidence supplied empty, short
    # or with a stranger is reported, never filled from the map.
    document = _family_maps()['baseline']
    _, by_input = _python_core_run(document)

    def observe(spec: dict[str, Any]) -> Any:
        return by_input[_spec_key(spec)]

    rust = _rust_run(document)
    written, report = refusal_map.write(document, _of(document), observe, rust)
    assert (written, report) == (document, [])
    probes = sorted(rust)
    _, report = refusal_map.write(document, _of(document), observe, {})
    assert report == [f'{name}/rust-host: not observed' for name in probes]
    short = {name: seen for name, seen in rust.items() if name != probes[0]}
    _, report = refusal_map.write(document, _of(document), observe, short)
    assert report == [f'{probes[0]}/rust-host: not observed']
    extra = {**rust, 'not-a-probe': refusal_map.success('predict')}
    _, report = refusal_map.write(document, _of(document), observe, extra)
    assert report == [
        'not-a-probe/rust-host: observed, and not a probe of rust-host'
    ]
    for retyped in ({}, [], None, 'success'):
        changed = {**rust, probes[0]: retyped}
        _, report = refusal_map.write(document, _of(document), observe, changed)
        assert report and report[0].startswith(f'{probes[0]}/rust-host: '), (
            retyped
        )


@pytest.mark.parametrize(
    ('observed', 'problem'),
    [
        ({'baseline.json': {}}, 'base.binary/rust-host: not observed'),
        ({}, 'no observations of baseline.json'),
        ({'baseline.json': []}, 'no observations of baseline.json'),
    ],
)
def test_the_writer_s_script_fails_on_supplied_rust_evidence_it_lacks(
    tmp_path: Path, observed: Any, problem: str
) -> None:
    # The script, seeded as Codex reproduced it: a copy of the baseline map
    # written with an observations file that holds none of its probes.
    import subprocess

    target = tmp_path / 'baseline.json'
    target.write_text((MAPS / 'baseline.json').read_text())
    evidence = tmp_path / 'observed.json'
    evidence.write_text(json.dumps(observed))
    run = subprocess.run(
        [
            sys.executable,
            '-I',
            str(RUNNERS / 'refusal_map.py'),
            'write',
            str(target),
            '--rust-observed',
            str(evidence),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert problem in run.stdout + run.stderr
    # Subagent Nit-1 (fix round 2): the map is left as it was; what the
    # run would make is beside it, as a draft.
    assert target.read_text() == (MAPS / 'baseline.json').read_text()
    if 'not observed' in problem:
        draft = tmp_path / 'baseline.json.draft'
        assert 'left as it was' in run.stdout
        assert json.loads(draft.read_text())['family'] == 'baseline'


def test_the_writer_s_script_rewrites_a_map_its_run_holds(
    tmp_path: Path,
) -> None:
    # The control: given the whole of its evidence, the script rewrites
    # the map, the same, and leaves no draft.
    import subprocess

    document = _family_maps()['baseline']
    target = tmp_path / 'baseline.json'
    target.write_text((MAPS / 'baseline.json').read_text())
    # A draft an earlier, failed run left beside it, stale once the map is
    # written (T1.22, from T1.21e's review).
    (tmp_path / 'baseline.json.draft').write_text('{}')
    evidence = tmp_path / 'observed.json'
    evidence.write_text(json.dumps({'baseline.json': _rust_run(document)}))
    run = subprocess.run(
        [
            sys.executable,
            '-I',
            str(RUNNERS / 'refusal_map.py'),
            'write',
            str(target),
            '--rust-observed',
            str(evidence),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert target.read_text() == (MAPS / 'baseline.json').read_text()
    assert not (tmp_path / 'baseline.json.draft').exists()


def test_the_writer_s_script_refuses_an_owner_no_map_may_name(
    tmp_path: Path,
) -> None:
    # T1.22 (from T1.21e's review): the writer holds owners to OWNER as the
    # tests do, so a bare T1.28, which T1.12b split into T1.28a and
    # T1.28b, is refused, the map left as it was and the run's map a draft.
    import subprocess

    document = _family_maps()['baseline']
    assert document['entries']
    changed = copy.deepcopy(document)
    changed['entries'][0]['owner'] = 'T1.28'
    assert refusal_map.shape(changed) == (
        'entries/0/owner is not a task a map may name'
    )
    target = tmp_path / 'baseline.json'
    target.write_text(json.dumps(changed, indent=2) + '\n')
    before = target.read_text()
    evidence = tmp_path / 'observed.json'
    evidence.write_text(json.dumps({'baseline.json': _rust_run(document)}))
    run = subprocess.run(
        [
            sys.executable,
            '-I',
            str(RUNNERS / 'refusal_map.py'),
            'write',
            str(target),
            '--rust-observed',
            str(evidence),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0, run.stdout + run.stderr
    assert 'owner is not a task a map may name' in run.stdout
    assert target.read_text() == before
    assert (tmp_path / 'baseline.json.draft').exists()
    # A gap's owner too.
    gapped = copy.deepcopy(document)
    gapped['gaps'] = [{'input': 'x', 'contract': 'y', 'owner': 'T1.28'}]
    assert refusal_map.shape(gapped) == (
        'gaps/0/owner is not a task a map may name'
    )
    # An owner with a trailing newline is not the task (subagent Nit-1,
    # fix round 1): the whole text is matched.
    trailing = copy.deepcopy(document)
    trailing['entries'][0]['owner'] = document['entries'][0]['owner'] + '\n'
    assert refusal_map.shape(trailing) == (
        'entries/0/owner is not a task a map may name'
    )


def test_a_family_all_of_whose_refusals_are_lifted_round_trips(
    tmp_path: Path,
) -> None:
    # Codex P2-2 (fix round 1): every probe succeeding on both hosts, the
    # writer drops every entry; the map it writes loads, holds both runs
    # and is written again unchanged, its probes and coverage kept.
    document = _family_maps()['baseline']
    lifted = {
        runner: {
            probe['id']: refusal_map.success(probe[runner]['until'])
            for probe in document['probes']
            if runner in probe
        }
        for runner in refusal_map.RUNNERS
    }

    def observe(spec: dict[str, Any]) -> Any:
        name = next(
            p['id']
            for p in document['probes']
            if _spec_key(p['python-core']) == _spec_key(spec)
        )
        return lifted['python-core'][name]

    written, report = refusal_map.write(
        document, _of(document), observe, lifted['rust-host']
    )
    assert report == ['materialize/static_target: no probe hits it; dropped']
    assert written['entries'] == [] and written['probes']
    path = tmp_path / 'baseline.json'
    path.write_text(json.dumps(written))
    loaded = refusal_map.load(path)
    for runner in refusal_map.RUNNERS:
        assert refusal_map.check(loaded, runner, lifted[runner]) == []
    assert refusal_map.bound(_of(loaded), loaded) == []
    again, report = refusal_map.write(
        loaded, _of(loaded), observe, lifted['rust-host']
    )
    assert (again, report) == (loaded, [])
    # Still fail-closed: a refusal on the lifted map is one it does not
    # list, and a case left without its probe fails the binding.
    refused = copy.deepcopy(lifted['python-core'])
    refused['base.binary'] = {
        'stage': 'materialize',
        'error': 'UnsupportedRequestError',
        'code': 'UNSUPPORTED_REQUEST',
        'operation': 'materialize',
        'rule': 'static_target',
    }
    assert any(
        failure.startswith('base.binary: a refusal the map does not list')
        for failure in refusal_map.check(loaded, 'python-core', refused)
    )
    loaded['probes'] = loaded['probes'][1:]
    assert refusal_map.bound(_of(loaded), loaded) != []
