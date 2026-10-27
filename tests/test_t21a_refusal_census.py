# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1a's acceptance tests for the refusal census and its first run on
DuckDB (T2.0d's row, folded into T2.1a by the owner's lean plan; the
independent test author's).

The design (choice 10): ``contract/tools/refusal_census.py`` lists the
refusal sites generated from the code, never written by hand: every named
rule, every rule-less refusal constructor's call (``unsupported_type``,
``sdm-contract/src/types.rs``; admission's ``unsupported``,
``sdm-host/src/admission.rs``; predict's ``unsupported``,
``sdm-request/src/predict.rs``) and the Python host's own refusals; it
fails when a site has no probe; it runs each probe on both routes
(python-core against the reference) on every database the probe applies
to; a database the core can't run yet is reported as unmeasured, never as
zero hits. Its first run is on DuckDB.

The oracle for "every named rule" and "every rule-less call" is computed
here, independently of the tool, from the same tree: each
``not_yet("<rule>")`` name, and each call of the three constructors, in
``rust/crates/*/src`` outside a file's ``#[cfg(test)]`` part.

Seeded (T2.0d's row): a refusal site with no probe fails, shown for a new
named rule and a new rule-less call, each in a copy of the tree whose
unseeded twin passes. The real-data part (the public calls over the
phase's real datasets, an hour or more) runs at each lane's first landing;
its own seeded faults (a rule-less refusal on a real source, a source left
out) are the implementer's, with that run.

Gating: :data:`PENDING`, a strict xfail; the implementer deletes its one
definition. Unmarked tests are controls that pass today.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'contract/tools/refusal_census.py'


# --- the interface assumed (the implementer matches it, or asks) ---------------
#
#: ``refusal_census.py sites --root DIR``: the sites generated from DIR's
#: code, as JSON on stdout: a list of objects, each with ``id`` (unique),
#: ``rule`` (the named rule, or null), ``code`` (the error code, or null
#: where the site does not fix one) and ``where`` (``<path>:<line>`` of the
#: refusing call, the path relative to DIR).
#: ``refusal_census.py check --root DIR``: exit 1 naming each site of DIR
#: with no probe in DIR's probe list, 0 when every site has one (the census
#: step's site check, run by the gate on the candidate).
#: The first DuckDB run's evidence, :data:`EVIDENCE`: an object with
#: ``database`` (``duckdb``), ``probes`` (each with ``site``, a site's id,
#: and ``outcomes``, an object with both routes, ``python-core`` and
#: ``reference``), ``hits`` (each with ``site`` and either ``rule`` or
#: ``code``, and ``sources``, the count of real sources it hit) and
#: ``unmeasured`` (the databases the core can't run yet, by name).
EVIDENCE = ROOT / 'docs/design/rust-core/phase-2/evidence/census/duckdb.json'
#: The three rule-less refusal constructors, by file and function.
RULELESS = {
    'rust/crates/sdm-contract/src/types.rs': 'unsupported_type',
    'rust/crates/sdm-host/src/admission.rs': 'unsupported',
    'rust/crates/sdm-request/src/predict.rs': 'unsupported',
}
# ---------------------------------------------------------------------------------

NOT_YET = re.compile(r'\bnot_yet\(\s*"([a-z0-9_]+)"')


def _production(text: str) -> str:
    r"""A Rust source's text before its ``#[cfg(test)]`` part."""
    return text.split('#[cfg(test)]', 1)[0]


def _sources(root: Path) -> list[Path]:
    return sorted(root.glob('rust/crates/*/src/**/*.rs'))


def named_rules(root: Path) -> set[str]:
    r"""Every ``not_yet`` rule name in the crates' production code."""
    return {
        name
        for path in _sources(root)
        for name in NOT_YET.findall(_production(path.read_text()))
    }


def ruleless_calls(root: Path) -> set[str]:
    r"""Every call of the three rule-less constructors, as ``path:line``;
    their definitions are not calls."""
    found = set()
    for relative, name in RULELESS.items():
        text = _production((root / relative).read_text())
        for number, line in enumerate(text.splitlines(), 1):
            if re.search(rf'\b{name}\(', line) and not re.search(
                rf'\bfn {name}\(', line
            ):
                found.add(f'{relative}:{number}')
    # unsupported_type is public: a call in another file is a site too.
    for path in _sources(root):
        relative = str(path.relative_to(root))
        if relative in RULELESS:
            continue
        for number, line in enumerate(
            _production(path.read_text()).splitlines(), 1
        ):
            if re.search(r'\bunsupported_type\(', line):
                found.add(f'{relative}:{number}')
    return found


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-I', str(TOOL), *arguments],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _sites(root: Path) -> list[dict[str, Any]]:
    done = _run('sites', '--root', str(root))
    assert done.returncode == 0, done.stderr
    found: list[dict[str, Any]] = json.loads(done.stdout)
    return found


def _copy(tmp_path: Path) -> Path:
    r"""The tree's committed code, the client's source and the contract,
    as the census reads them."""
    root = tmp_path / 'tree'
    listed = subprocess.run(
        [
            'git',
            '-C',
            str(ROOT),
            'ls-files',
            '-z',
            '--',
            'rust/crates',
            'packages/structured-data-models-client/src',
            'contract',
        ],
        capture_output=True,
        check=True,
    ).stdout.split(b'\0')
    for name in filter(None, (item.decode() for item in listed)):
        source = ROOT / name
        if source.is_file():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, root / name)
    return root


def test_the_oracle_finds_the_sites_the_design_names() -> None:
    r"""Control: the oracle is not vacuous: E92's class of rules (61 at
    T1.32d, fewer as packages lift them: 23 after T2.11's ten), and calls
    of all three rule-less constructors."""
    assert len(named_rules(ROOT)) >= 20
    calls = ruleless_calls(ROOT)
    for relative in RULELESS:
        assert any(call.startswith(f'{relative}:') for call in calls), relative


def test_every_site_of_the_code_is_listed() -> None:
    r"""Exit: the list is generated from the code: every named rule, every
    rule-less constructor's call, and the Python host's own refusals."""
    sites = _sites(ROOT)
    ids = [site['id'] for site in sites]
    assert len(ids) == len(set(ids))
    rules = {site['rule'] for site in sites if site['rule']}
    assert named_rules(ROOT) <= rules, sorted(named_rules(ROOT) - rules)
    wheres = {site['where'] for site in sites}
    assert ruleless_calls(ROOT) <= wheres, sorted(ruleless_calls(ROOT) - wheres)
    host = 'packages/structured-data-models-client/src/sdm_client/'
    assert any(where.startswith(host) for where in wheres)


def test_every_site_has_a_probe() -> None:
    r"""Exit: on this tree, the census's site check passes."""
    done = _run('check', '--root', str(ROOT))
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.tooling
@pytest.mark.parametrize(
    ('relative', 'code', 'named'),
    [
        (
            'rust/crates/sdm-engine/src/t21a_seeded.rs',
            'pub fn seeded() -> sdm_contract::CoreError {\n'
            '    crate::not_yet("t21a_seeded_rule")\n}\n',
            't21a_seeded_rule',
        ),
        (
            'rust/crates/sdm-contract/src/t21a_seeded.rs',
            'pub fn seeded(c: &str, t: &DataType) -> CoreError {\n'
            '    crate::unsupported_type(c, t)\n}\n',
            'rust/crates/sdm-contract/src/t21a_seeded.rs:2',
        ),
    ],
    ids=['a named rule', 'a rule-less call'],
)
def test_a_site_with_no_probe_fails_by_name(
    relative: str, code: str, named: str, tmp_path: Path
) -> None:
    r"""Seeded: a refusal site added to the code with no probe fails the
    census's check, naming it; the unseeded copy passes."""
    root = _copy(tmp_path)
    clean = _run('check', '--root', str(root))
    assert clean.returncode == 0, clean.stdout + clean.stderr
    (root / relative).write_text(code)
    seeded = _run('check', '--root', str(root))
    assert seeded.returncode == 1, seeded.stdout + seeded.stderr
    assert named in seeded.stdout + seeded.stderr


def test_the_first_duckdb_run_probed_every_site_on_both_routes() -> None:
    r"""Exit: the first run is on DuckDB; every site the code has now was
    probed there on both routes, each hit is keyed by rule or by site and
    code with its count of real sources, and SQLite and Postgres are
    reported unmeasured, never as zero hits."""
    evidence = json.loads(EVIDENCE.read_text())
    assert evidence['database'] == 'duckdb'
    sites = {site['id'] for site in _sites(ROOT)}
    probed = {}
    for probe in evidence['probes']:
        assert set(probe['outcomes']) >= {'python-core', 'reference'}, probe
        probed.setdefault(probe['site'], probe)
    assert sites <= set(probed), sorted(sites - set(probed))
    for hit in evidence['hits']:
        assert hit['site'] in sites, hit
        assert hit.get('rule') or hit.get('code'), hit
        assert isinstance(hit['sources'], int) and hit['sources'] >= 0, hit
    assert sorted(evidence['unmeasured']) == ['postgres', 'sqlite']


def test_a_case_naming_a_route_the_core_lacks_leaves_the_duckdb_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # T3.20 r3: the census forces onto the core only the routes this
    # build's core has. A case in the tree naming a warehouse route the
    # core has no route for yet (a Lakebase predict) leaves the census's
    # DuckDB probes refusing as they do, never a route error.
    import importlib.util

    spec = importlib.util.spec_from_file_location('census_t320', TOOL)
    assert spec is not None and spec.loader is not None
    census = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = census
    spec.loader.exec_module(census)
    runner = census._runner()
    loaded = runner.load_cases
    lacking = 'client.predict/binary/lakebase'

    def with_a_warehouse_case(paths: Any) -> Any:
        found = list(loaded(paths))
        case, where = found[0]
        found.append(({**case, 'routes': [lacking]}, where))
        return found

    monkeypatch.setattr(runner, 'load_cases', with_a_warehouse_case)
    routes = census._routes()
    assert lacking not in routes and routes
    probes = [
        probe
        for probe in census.probes(ROOT)
        if 'duckdb' in probe.get('databases', census.DATABASES)
        and probe.get('frame') is None
    ][:8]
    cases = tmp_path / 'cases'
    cases.mkdir()
    for number, probe in enumerate(probes):
        (cases / f'{number}.json').write_text(
            json.dumps(
                {
                    'case': census._case(probe, number, routes),
                    'options': probe.get('options', {}),
                    'model': probe.get('model', 'KumoRelational'),
                }
            )
        )
    census.route(cases, tmp_path / 'out.json', 'duckdb', core=True)
    outcomes = json.loads((tmp_path / 'out.json').read_text())
    assert not [o for o in outcomes if o['result'] == 'route_error'], outcomes
    assert any(o['result'] == 'refusal' for o in outcomes), outcomes
