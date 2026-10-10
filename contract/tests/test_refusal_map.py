# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E27's AdventureWorks case: its fixture, and its refusal map on python-core
(T1.21b).

Run where the contract tests run (CI's ``engine`` job until T1.18):

    python -m pytest contract/tests -q

- The fixture: the ``adventureworks`` base of ``contract/bases.yaml`` holds
  1,000 rows in all, at most 2,000, as its README's table says, and the
  case asks for the product the notebook asks for, the most units sold.
- The map (``contract/refusal-maps/base.adventureworks.json``): every entry
  names its rule and operation, an owning task among T1.22 to T1.28 and a
  register entry that is proposed, and the line it cites holds the rule;
  each probe's input is a case input, its own edits aside.
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
        assert refusal_map.OWNER.match(entry['owner']), entry['id']
        # Never self-approved: the entry the refusal is under waits for the
        # owner.
        assert register[entry['register']]['status'] == 'proposed', entry['id']
        path, _, lines = entry['where'].partition(':')
        assert (ROOT / path).is_file(), entry['id']
        if path.startswith('rust/'):
            # A core refusal: the line cited names the rule.
            line = int(re.match(r'\d+', lines).group())
            text = (ROOT / path).read_text().splitlines()[line - 1]
            assert f'not_yet("{entry["rule"]}")' in text, entry['id']
    for gap in document['gaps']:
        assert refusal_map.OWNER.match(gap['owner'])
    # Each probe's python-core input is a case input, but for the edits
    # only a probe makes (a key or the relationships set).
    validate = run_python.validator('case')
    for probe in document['probes']:
        if 'python-core' in probe:
            case = refusal_map.probe_input(_case(), probe['python-core'])
            assert run_python._invalid(validate, case) is None, probe['id']


def test_the_python_core_run_is_held_to_the_map() -> None:
    document = _map()
    bases = run_python.load_bases()
    observed = {}
    for probe in document['probes']:
        spec = probe.get('python-core')
        if spec is None:
            continue
        seen, capture = refusal_map.observe_python_core(run_python, spec, bases)
        observed[probe['id']] = seen
        rust = probe.get('rust-host')
        if rust is None or not capture.inputs:
            continue
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
    del observed['no-keys']
    failures = refusal_map.check(document, 'python-core', observed)
    assert 'no-keys: not observed' in failures
    assert (
        'describe_source/time_columns: listed for python-core and not hit'
        in failures
    )
    observed = _observed()
    del observed['no-keys']['rule']
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
    observed['no-keys'] = copy.deepcopy(observed['no-time-columns'])
    assert (
        'no-keys: hit describe_source/text_time, where the map has '
        'describe_source/time_columns'
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
