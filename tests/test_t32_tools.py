# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.2's changes to the checking tools other than the case format, the
recording mode and the comparator (the package's test author's), one test
per change:

- ``l0_deps.py`` classifies ``sdm-jni`` as a binding allowed only
  ``sdm-host`` and ``sdm-contract``, absent until it exists: a seeded
  workspace holding it on ``jni`` passes, one holding it with a ``pyo3``
  edge or an edge to ``sdm-core`` fails by name, and one holding an
  unclassified binding fails (the brief's mandatory examples);
- ``l0_scan.py`` scans ``sdm-jni`` for dialect words once it exists, and
  asks nothing of it before;
- ``coverage.py``: a case's database legs claim their own paths
  (P2-T2.16-3), a named client test may claim a path the case format
  cannot express (``state.pickle/request/any``, the dry run's gap), and
  the S1 rule names an S1 path with neither;
- ``register.py`` refuses an ``approved_by: owner`` entry whose decision
  row the owner did not approve, and the ratchet reads an owner's approval
  by allowlist (U4): a qualifier the old denylist missed is no approval;
- the census's real-data mode takes each leg;
- ``gate.sh``: the main step fetches ``origin/main`` itself right before
  it reads it (P2-T2.0f-1); the locked run holds the captures'
  pending-live list; the Java runner runs;
- the comparator hashes result rows as text, so a case whose result
  digest an entry excuses (P2G-3) still has its ids and predictions
  compared (P2-T2.21-1).

Interfaces assumed (T3.2 defines them; this file is their first reader):

- ``coverage.generated(language, manifest, corpus)`` counts each leg a case
  runs on (``databases``, else all three) for the paths it names on DuckDB;
  ``coverage.uncovered(manifests, corpus, language, slice)`` lists the
  paths of ``slice`` with no acceptance case and no client test claiming
  them; a manifest path item may carry ``tests``, pytest node ids
  (``<file>::<test>``) of client tests claiming it, each one that exists;
- ``contract/tools/refusal_census.py real-data --database LEG ...``;
- the observation's ``public.result`` carries ``text_digest``, the SHA-256
  of the result's rows as text, which the ``result`` rule compares where
  ``digest`` is excused.

Each test listed in ``contract/tests/t32-pending.json`` failed when written,
before the implementation, and is held to failing (the ``pending``
decorator, as T2.10's ``held``) until the implementer takes it out of
``pending``; ``pending`` stays within
``authored``, the test author's list.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUST_CORE = ROOT / 'scripts/rust_core'
PENDING_FILE = ROOT / 'contract/tests/t32-pending.json'
WAREHOUSES = ('databricks-sql', 'databricks-spark', 'snowflake', 'lakebase')
#: The L0 scan's test copies the crates into a temporary tree: the judging
#: tools' job runs this file (scripts/rust_core/tooling_tests.py).
pytestmark = pytest.mark.tooling


def _pending_names(key: str) -> set[str]:
    return set(json.loads(PENDING_FILE.read_text(encoding='utf-8'))[key])


def pending(test: Any) -> Any:
    """Holds ``test`` to failing while ``contract/tests/t32-pending.json``
    lists it, as T2.10's ``held`` does: a pending test passes only by
    failing, and once it passes it fails, naming itself, until the
    implementer takes it out of ``pending``."""
    assert test.__name__ in _pending_names('authored'), test.__name__

    @functools.wraps(test)
    def held(*arguments: Any, **keywords: Any) -> None:
        if test.__name__ not in _pending_names('pending'):
            test(*arguments, **keywords)
            return
        try:
            test(*arguments, **keywords)
        except Exception:  # noqa: BLE001 - pending: failing, as when written
            return
        raise AssertionError(
            f'{test.__name__} passes now: take it out of '
            'contract/tests/t32-pending.json pending'
        )

    return held


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _tool(name: str) -> ModuleType:
    r"""A module of ``scripts/rust_core``, its siblings importable."""
    if str(RUST_CORE) not in sys.path:
        sys.path.insert(0, str(RUST_CORE))
    return _module(f't32_tools_{name}', RUST_CORE / f'{name}.py')


# --- l0_deps.py and l0_scan.py ---------------------------------------------------


def _crate_manifest(name: str) -> Path:
    for parent in ('rust/crates', 'rust/bindings'):
        manifest = ROOT / parent / name / 'Cargo.toml'
        if manifest.is_file():
            return manifest
    raise AssertionError(f'no crate {name}')


class Workspace:
    r"""``cargo metadata``'s shape for the workspace as the design draws it,
    built without cargo: every first-party crate ``l0_deps.py`` classifies
    and finds in the tree, at its real manifest, on the first-party edges
    its ``ALLOWED`` gives it; ``without`` leaves a crate out, as if it
    did not exist yet, so ``sdm-jni`` can be seeded whether or not the tree
    holds it."""

    def __init__(self, l0: ModuleType, without: tuple[str, ...] = ()) -> None:
        crates = [
            name
            for name in sorted(l0.CLASSES)
            if name not in without
            and any(
                (ROOT / parent / name / 'Cargo.toml').is_file()
                for parent in ('rust/crates', 'rust/bindings')
            )
        ]
        self.data: dict[str, Any] = {
            'packages': [],
            'workspace_members': [],
            'workspace_root': str(ROOT / 'rust'),
            'resolve': {'nodes': []},
        }
        for name in crates:
            self.package(name, _crate_manifest(name), member=True)
        for name in crates:
            for dep in sorted(l0.ALLOWED.get(name, frozenset())):
                if dep in crates:
                    self.edge(name, dep)

    def _id(self, name: str) -> str:
        return f't32+{name}@0.0.0'

    def package(self, name: str, manifest: Path, member: bool = False) -> None:
        if any(p['name'] == name for p in self.data['packages']):
            return
        self.data['packages'].append(
            {'id': self._id(name), 'name': name, 'manifest_path': str(manifest)}
        )
        self.data['resolve']['nodes'].append({'id': self._id(name), 'deps': []})
        if member:
            self.data['workspace_members'].append(self._id(name))

    def edge(self, crate: str, dep: str) -> None:
        self.package(dep, Path('/nonexistent/Cargo.toml'))
        for node in self.data['resolve']['nodes']:
            if node['id'] == self._id(crate):
                node['deps'].append(
                    {
                        'name': dep.replace('-', '_'),
                        'pkg': self._id(dep),
                        'dep_kinds': [{'kind': None, 'target': None}],
                    }
                )


def _binding(tmp: Path, name: str) -> Path:
    manifest = tmp / name / 'Cargo.toml'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        f'[package]\nname = "{name}"\n\n[lints.rust]\nunsafe_code = "deny"\n',
        encoding='utf-8',
    )
    return manifest


def _l0_problems(l0: ModuleType, workspace: Workspace) -> list[str]:
    return [
        problem
        for problems in l0.run_checks(workspace.data, ROOT).values()
        for problem in problems
    ]


def test_t32_the_seeded_workspace_is_the_design_s_graph() -> None:
    r"""Not vacuous: the workspace built without cargo passes every L0
    check as the tree is, and an unclassified binding in it fails (the
    brief's first ``l0_deps.py`` example, which passes today too)."""
    l0 = _tool('l0_deps')
    assert _l0_problems(l0, Workspace(l0)) == []


@pending
def test_t32_l0_deps_holds_sdm_jni_to_the_host_and_contract(
    tmp_path: Path,
) -> None:
    r"""``sdm-jni`` is a binding of ``l0_deps.py``'s own lists, allowed only
    ``sdm-host`` and ``sdm-contract``, and passes absent: a workspace
    holding it on those and ``jni`` passes; with a ``pyo3`` edge, or an
    edge to ``sdm-core``, it fails naming the edge; an unclassified binding
    fails as unclassified."""
    l0 = _tool('l0_deps')
    assert l0.CLASSES.get('sdm-jni') == l0.BINDING
    assert l0.ALLOWED.get('sdm-jni') == frozenset({'sdm-host', 'sdm-contract'})
    absent = Workspace(l0, without=('sdm-jni',))
    assert 'sdm-jni' not in [p['name'] for p in absent.data['packages']]
    assert _l0_problems(l0, absent) == []

    def with_jni(*extra: str) -> list[str]:
        workspace = Workspace(l0, without=('sdm-jni',))
        manifest = _binding(tmp_path / '-'.join(('jni', *extra)), 'sdm-jni')
        workspace.package('sdm-jni', manifest, True)
        for dep in ('sdm-host', 'sdm-contract', 'jni', *extra):
            workspace.edge('sdm-jni', dep)
        return _l0_problems(l0, workspace)

    assert with_jni() == []
    pyo3 = with_jni('pyo3')
    assert any('sdm-jni reaches pyo3' in p for p in pyo3), pyo3
    core = with_jni('sdm-core')
    assert any('sdm-jni -> sdm-core' in p for p in core), core
    other = Workspace(l0, without=('sdm-jni',))
    other.package('sdm-rb', _binding(tmp_path / 'b', 'sdm-rb'), True)
    other.edge('sdm-rb', 'sdm-host')
    problems = _l0_problems(l0, other)
    assert any('sdm-rb: a workspace crate with no class' in p for p in problems)


@pending
def test_t32_l0_scan_reads_sdm_jni_for_dialect_words_once_it_exists(
    tmp_path: Path,
) -> None:
    r"""``l0_scan.py``'s dialects check holds ``sdm-jni`` as it holds
    ``sdm-py``, once the crate exists, and asks nothing of it before."""
    scan = _tool('l0_scan')
    assert not [p for p in scan.check_dialects(ROOT) if 'sdm-jni' in p]
    for parent in ('rust/crates', 'rust/bindings'):
        shutil.copytree(
            ROOT / parent,
            tmp_path / parent,
            ignore=shutil.ignore_patterns('target', '__pycache__', '.venv'),
        )
    # The copy without sdm-jni, whether or not the tree holds it yet: the
    # check asks nothing of a crate that does not exist.
    for parent in ('rust/crates', 'rust/bindings'):
        if (tmp_path / parent / 'sdm-jni').exists():
            shutil.rmtree(tmp_path / parent / 'sdm-jni')
    assert scan.check_dialects(tmp_path) == []
    source = tmp_path / 'rust/bindings/sdm-jni/src/lib.rs'
    source.parent.mkdir(parents=True)
    source.write_text(
        'pub const DEFAULT_BACKEND: &str = "snowflake";\n', encoding='utf-8'
    )
    (source.parent.parent / 'Cargo.toml').write_text(
        '[package]\nname = "sdm-jni"\n', encoding='utf-8'
    )
    found = scan.check_dialects(tmp_path)
    assert any('sdm-jni/src/lib.rs' in p and 'snowflake' in p for p in found), (
        found
    )


# --- coverage.py ---------------------------------------------------------------


def _manifests() -> dict[str, dict[str, Any]]:
    return {
        language: yaml.safe_load(
            (
                ROOT / f'docs/design/rust-core/phase-0/scope-{language}.yaml'
            ).read_text(encoding='utf-8')
        )
        for language in ('python', 'java')
    }


def _case(name: str) -> dict[str, Any]:
    found: dict[str, Any] = json.loads(
        (ROOT / f'contract/cases/s1/{name}.json').read_text(encoding='utf-8')
    )
    return found


@pending
def test_t32_a_case_s_database_legs_claim_their_own_paths() -> None:
    r"""P2-T2.16-3: a case naming a DuckDB path covers the same path of
    each database it runs on, SQLite's and Postgres's, and not of one its
    ``databases`` leaves out."""
    coverage = _tool('coverage')
    manifest = _manifests()['python']
    case = _case('base.binary')
    path = 'schema.from-source/all-tables/duckdb'
    assert path in case['paths']
    made = coverage.generated('python', manifest, {case['id']: case})['paths']
    for database in ('duckdb', 'sqlite', 'postgres'):
        assert made[path.replace('duckdb', database)] == [case['id']], database
    narrowed = copy.deepcopy(case)
    narrowed['databases'] = ['duckdb']
    narrowed['exclusion'] = 'KO2-19'
    made = coverage.generated('python', manifest, {case['id']: narrowed})[
        'paths'
    ]
    assert made[path] == [case['id']]
    assert made['schema.from-source/all-tables/sqlite'] == []


@pending
def test_t32_a_client_test_may_claim_a_path_the_cases_cannot_express() -> None:
    r"""The dry run's ``input.pickle`` gap: ``state.pickle/request/any`` is
    claimed by a named client test that exists, so the S1 rule does not
    list it; a claim naming no test is refused by name."""
    coverage = _tool('coverage')
    manifests = _manifests()
    [item] = [
        p
        for p in manifests['python']['paths']
        if p['id'] == 'state.pickle/request/any'
    ]
    assert item.get('tests'), 'no client test claims the path'
    for node in item['tests']:
        file, _, test = node.partition('::')
        assert re.search(
            rf'^\s*def {re.escape(test)}\(',
            (ROOT / file).read_text(encoding='utf-8'),
            re.M,
        ), node
    corpus = {'base.binary': _case('base.binary')}
    assert 'state.pickle/request/any' not in coverage.uncovered(
        manifests, corpus, 'python', 'S1'
    )
    report = coverage.Report()
    broken = copy.deepcopy(manifests)
    [claimed] = [
        p
        for p in broken['python']['paths']
        if p['id'] == 'state.pickle/request/any'
    ]
    claimed['tests'] = ['packages/no/such/test_file.py::test_nothing']
    coverage.rule_4(broken, corpus, report)
    assert any('test_nothing' in p for p in report.problems), report.problems


@pending
def test_t32_the_s1_rule_names_an_s1_path_with_no_case() -> None:
    r"""The S1 rule: an S1 path no acceptance case (on any of its legs) and
    no client test covers is named; one a case covers is not."""
    coverage = _tool('coverage')
    manifests = _manifests()
    corpus = {'base.binary': _case('base.binary')}
    empty = coverage.uncovered(manifests, corpus, 'python', 'S1')
    # base.binary covers its three paths on each database, and no other.
    assert 'schema.from-source/all-tables/sqlite' not in empty
    assert 'client.predict/binary/postgres' not in empty
    assert 'query.materialize/indices-ranking/postgres' in empty
    assert 'schema.from-source/tables/postgres' in empty


# --- keys on each leg ------------------------------------------------------------------


@pending
def test_t32_the_ratchet_and_the_register_key_a_line_on_each_leg() -> None:
    r"""``<case>@<leg>`` in ``ratchet.py`` (its key form, which every
    reference line and baseline is held to) and in ``register.py`` (an
    entry naming a line on one leg): each warehouse leg is a key's
    database, a name that is no leg is not."""
    ratchet = _tool('ratchet')
    register = _tool('register')
    form = ratchet.key_form()
    for leg in WAREHOUSES:
        assert form.fullmatch(f'base.binary@{leg}'), leg
        assert ratchet.case_of(f'base.binary@{leg}') == 'base.binary'
        found = register.ON_DATABASE.fullmatch(f'base.binary@{leg}')
        assert found and found['case'] == 'base.binary', leg
    assert not form.fullmatch('base.binary@bigquery')
    assert not register.ON_DATABASE.fullmatch('base.binary@bigquery')


# --- register.py and the owner's approvals -----------------------------------------


def _register() -> dict[str, Any]:
    found: dict[str, Any] = yaml.safe_load(
        (ROOT / 'docs/design/rust-core/register.yaml').read_text(
            encoding='utf-8'
        )
    )
    return found


@pending
def test_t32_an_owner_entry_needs_a_row_the_owner_approved() -> None:
    r"""``register.py`` refuses an entry ``approved_by: owner`` whose
    decision row the owner did not approve (here a superseded row), and
    takes the same entry on a row the owner did."""
    register = _tool('register')
    ratchet = _tool('ratchet')
    check = ratchet.capture_check()
    rows = check.decision_rows(
        (ROOT / 'docs/design/rust-core/phase-1/decisions.md').read_text(
            encoding='utf-8'
        )
    )
    approved = {
        n for n, (_, status) in rows.items() if ratchet.owner_approved(status)
    }
    refused = sorted(
        n
        for n, (_, status) in rows.items()
        if status.startswith('superseded by')
    )
    assert approved and refused
    whole = _register()
    [model] = [
        e
        for e in whole['entries']
        if e.get('approved_by') == 'owner'
        and e['decision_ref']
        and e['decision_ref']['file'].endswith('phase-1/decisions.md')
        and str(e['decision_ref']['row']) in approved
    ][:1]

    def problems(row: str) -> list[str]:
        entry = copy.deepcopy(model)
        entry['id'] = 'TT-901'
        entry['decision_ref'] = {**entry['decision_ref'], 'row': row}
        seeded = copy.deepcopy(whole)
        seeded['entries'].append(entry)
        return [p for p in register.problems(seeded, ROOT) if 'TT-901' in p]

    assert problems(str(model['decision_ref']['row'])) == []
    assert problems(refused[0]), refused[0]


#: Owner approvals as the decisions files write them: each must stay one.
APPROVED = (
    'approved by owner 2026-10-10',
    'approved by owner 2026-10-10 (T1.7b check-in)',
    'approved by owner 2026-10-21 (with KO3-17)',
    'approved by owner 2026-10-12 (relayed by the orchestrator; recorded at '
    "T1.32g, never the implementer's approval)",
    'approved by owner 2026-10-11 (checkpoint 1)',
    'approved by owner 2026-10-17 (round 1 answer 9, round 2 answers 3 and 5)',
    'approved by owner 2026-10-19 (P2G-4, P2G-5 and KO2-89 only)',
    'approved by owner 2026-10-15 (E102)',
    'approved by owner 2026-10-17 (answer 6)',
    'approved by owner 2026-10-10 (T1.11e): the owner\'s written answer was "Approve"',
)
#: Qualifiers that take the approval back, which the denylist (U4) missed.
NOT_APPROVED = (
    'approved by owner 2026-10-21 (on hold)',
    'approved by owner 2026-10-21 (deferred to the check-in)',
    'approved by owner 2026-10-21 (tentatively)',
    'approved by owner 2026-10-21 (conditional on the review)',
    'approved by owner 2026-10-21 (the owner said no)',
    'approved by owner 2026-10-21, maybe',
    'approved by owner 2026-10-21 (declined)',
)


@pending
def test_t32_an_owner_s_approval_is_read_by_allowlist() -> None:
    r"""U4: a qualifier is no approval unless it is one the allowlist
    reads as the owner's; the forms the decisions files use stay
    approvals."""
    ratchet = _tool('ratchet')
    assert [s for s in APPROVED if not ratchet.owner_approved(s)] == []
    assert [s for s in NOT_APPROVED if ratchet.owner_approved(s)] == []


# --- the census, gate.sh and the comparator -----------------------------------------


@pending
def test_t32_the_census_s_real_data_mode_takes_each_leg() -> None:
    r"""Check 2: the census's real-data mode runs per leg on the three
    offline databases; a warehouse leg runs live only (B-lite), so none is
    one of its choices."""
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'contract/tools/refusal_census.py'),
            'real-data',
            '--help',
        ],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    choices = re.search(r'--database \{([^}]*)\}', done.stdout)
    assert choices, done.stdout
    assert set(choices.group(1).split(',')) == {'duckdb', 'sqlite', 'postgres'}


def _function(name: str) -> str:
    r"""``gate.sh``'s shell function ``name``, as written."""
    text = (RUST_CORE / 'gate.sh').read_text(encoding='utf-8')
    found = re.search(rf'^{name}\(\) \{{\n(.*?)^\}}', text, re.M | re.S)
    assert found, f'gate.sh has no {name}()'
    return found.group(1)


@pending
def test_t32_the_main_step_fetches_origin_main_before_reading_it() -> None:
    r"""P2-T2.0f-1: the locked run's main step fetches ``origin/main``
    itself, before it reads ``refs/remotes/origin/main``, so a stale fetch
    cannot pass check 7."""
    body = _function('main_merged')
    fetch = re.search(r'\bfetch\b[^\n]*\borigin\b[^\n]*\bmain\b', body)
    read = body.find('refs/remotes/origin/main')
    assert fetch, 'the main step fetches nothing'
    assert read != -1 and fetch.start() < read


@pending
def test_t32_the_locked_run_holds_pending_live_cases() -> None:
    r"""Locked check 1 in ``gate.sh``: the locked run runs ``capture_check.py
    captures --locked``, so no case is pending live at the final run (the
    E-1 verdicts it also read went with the stand-in, B-lite)."""
    text = (RUST_CORE / 'gate.sh').read_text(encoding='utf-8')
    assert re.search(r'capture_check\.py[^\n]*captures[^\n]*--locked', text)


@pending
def test_t32_the_gate_runs_the_java_legs() -> None:
    r"""``gate.sh`` runs the Java host's runner (``run_java.py``) beside
    DuckDB's runners (its stand-in legs went with the stand-in, B-lite)."""
    body = _function('observe')
    assert re.search(r'^\s*java\)', body, re.M), 'no java runner in observe'
    assert 'run_java.py' in body


@pending
def test_t32_result_rows_are_compared_as_text_where_the_digest_is_excused(
    tmp_path: Path,
) -> None:
    r"""P2-T2.21-1: P2G-3 excuses ``public.result/digest`` for a case whose
    key's dtype the core changes; the rows hashed as text still compare,
    so its ids and predictions are judged again: equal text passes, other
    text fails."""
    compare = _module('t32_tools_compare', RUST_CORE / 'compare.py')
    source = 'base.binary'
    case = _case(source)
    case.pop('core')
    case['id'] = 't32.text.binary'
    cases = tmp_path / 'cases'
    cases.mkdir()
    data = json.dumps(case).encode()
    (cases / 'case.json').write_bytes(data)
    [path] = sorted((ROOT / 'contract/reference').glob('*.jsonl'))
    [line] = [
        json.loads(text)
        for text in path.read_text(encoding='utf-8').splitlines()
        if json.loads(text)['case'] == source
    ]
    line['case'] = case['id']
    line['identity']['inputs']['case'] = hashlib.sha256(data).hexdigest()
    line['public']['result']['text_digest'] = 'a' * 64
    reference = tmp_path / 'reference.jsonl'
    reference.write_text(json.dumps(line) + '\n', encoding='utf-8')
    [p2g3] = [e for e in _register()['entries'] if e['id'] == 'P2G-3']
    entry = copy.deepcopy(p2g3)
    entry['cases'] = [case['id']]
    entry['more_exact'] = [
        {
            'case': case['id'],
            'runners': ['python-core'],
            'fields': ['public.result/digest'],
        }
    ]
    register = tmp_path / 'register.yaml'
    register.write_text(
        yaml.safe_dump({'format': 'sdm-register/2', 'entries': [entry]}),
        encoding='utf-8',
    )

    def judged(text_digest: str) -> list[str]:
        core = copy.deepcopy(line)
        core['identity'].update(
            runner='python-core',
            reference=None,
            native={'sha256': '0' * 64},
            native_loaded=True,
            operations=[],
        )
        core['core'] = {
            'inputs': [],
            'fetches': [],
            'accepted': None,
            'described': None,
            'materialized': None,
            'predicted': None,
            'error': None,
        }
        core['public']['result']['digest'] = '0' * 64
        core['public']['result']['text_digest'] = text_digest
        observed = tmp_path / 'python-core.jsonl'
        observed.write_text(json.dumps(core) + '\n', encoding='utf-8')
        report = compare.compare(
            [cases], reference, [observed], improvements=register
        )
        return list(report.texts)

    assert judged('a' * 64) == []
    differs = judged('b' * 64)
    assert any('text_digest' in text for text in differs), differs


def test_the_locked_run_reads_only_the_capture_manifests(
    tmp_path: Path,
) -> None:
    r"""T3.20: the locked run hands ``capture_check.py captures`` each leg's
    ``manifest.json``, never a capture's own ``capture.json``, which the
    check refuses as a manifest ("no legs or cases")."""
    text = (RUST_CORE / 'gate.sh').read_text(encoding='utf-8')
    [found] = re.findall(r'find "\$candidate/contract/captures"[^)\n]*', text)
    leg = tmp_path / 'contract/captures/s2/snowflake'
    (leg / 'a.case').mkdir(parents=True)
    (leg / 'manifest.json').write_text('{}')
    (leg / 'a.case/capture.json').write_text('{}')
    listed = subprocess.run(
        ['bash', '-c', found.replace('-print0', '-print')],
        env={'candidate': str(tmp_path), 'PATH': os.environ['PATH']},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert listed == [str(leg / 'manifest.json')]
