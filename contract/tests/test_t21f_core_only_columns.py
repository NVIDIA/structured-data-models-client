# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1f: an owner-approved register entry may list, under
``core_only_columns``, columns only the core keeps (KO2-36);
``compare.py`` takes exactly those out of a core runner's side against
the reference's, with the reference's warning of leaving them out, and
compares everything else.

The reference's own observation of ``base.regression``, observed as
``python-core`` but with a column ``a.b`` of ``users`` kept (in the
public schema, the request's tables and the statement result that read
it), and the reference's from_source ``SDMWarning`` of leaving it out,
stands for a case whose column the reference leaves out. The negative
controls: the entry proposed, or backed by no owner-approved row, excuses
nothing; a second extra column the entry does not name still fails.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'contract/cases/s1/base.regression.json'
REFERENCE = next((ROOT / 'contract/reference').glob('*.jsonl'))
#: The rules a kept column makes differ.
RULES = {'schema', 'warnings', 'payload', 'sql.result'}


def _module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_python = _module('run_python', ROOT / 'contract/runners/run_python.py')
compare = _module('compare', ROOT / 'scripts/rust_core/compare.py')


def _keep(observation: dict[str, Any], name: str) -> None:
    r"""``observation`` with a column ``name`` of ``users`` kept, as the
    core keeps one the reference leaves out."""
    tables = observation['public']['schema']['tables']
    [users] = [t for t in tables if t['name'] == 'users']
    users['columns'].append(
        {'dtype': 'int64', 'name': name, 'stype': 'numerical'}
    )
    for each in observation['public']['payloads']:
        parsed = each['parsed']
        for part in ('context', 'predict'):
            section = (
                (parsed.get(part) or {}).get('related_tables') or {}
            ).get('users')
            if section is not None:
                section['columns'].append(name)
                for row in section['rows']:
                    row.append({'number': '7'})
        columns = parsed['schema']['related_tables']['users']['columns']
        columns[name] = {'dtype': 'int64', 'stype': 'numerical'}
    read = next(
        result
        for statement in observation['driver']['capture']['statements']
        for result in statement.get('results') or []
        if 'age' in result['values']['columns']
    )
    read['values']['columns'].append(name)
    read['values']['types'].append('BIGINT')
    read['schema'].append({'name': name, 'type': 'int64'})
    for key in ('ordered', 'unordered'):
        read['values'][key] = name.encode().hex().ljust(64, '0')[:64]


def _failing(
    tmp_path: Path,
    entry: dict[str, Any],
    *,
    also: str | None = None,
    kept: bool = True,
    rows: int = 0,
    cell: str | None = None,
    runtime: str | None = None,
) -> set[str | None]:
    r"""The rules that fail, the core keeping ``a.b`` (and ``also``); with
    ``kept`` false, answering as the reference does, its warning too;
    ``rows`` more rows in the result read with it; in ``cell`` (not the
    reference's), python-core held to a python-old answering as the
    reference does, as that cell's comparison holds it; with ``runtime``,
    both Python runners' warning origins recorded, python-old also showing
    a DeprecationWarning raised from ``runtime``'s files, which HOST-5 leaves
    out where that is pandas or pyarrow."""
    cases = tmp_path / 'cases'
    cases.mkdir(exist_ok=True)
    (cases / CASE.name).write_bytes(CASE.read_bytes())
    reference = next(
        json.loads(line)
        for line in REFERENCE.read_text().splitlines()
        if json.loads(line)['case'] == 'base.regression'
    )
    reference['identity']['inputs']['case'] = run_python.digest(
        CASE.read_bytes()
    )
    observation = copy.deepcopy(reference)
    observation['identity']['runner'] = 'python-core'
    observation['identity']['reference'] = None
    # The reference's warning of the column it leaves out.
    reference['host']['warnings'].insert(0, 'SDMWarning')
    if kept:
        _keep(observation, 'a.b')
    else:
        observation['host']['warnings'].insert(0, 'SDMWarning')
    if also is not None:
        _keep(observation, also)
    if rows:
        read = next(
            result
            for statement in observation['driver']['capture']['statements']
            for result in statement.get('results') or []
            if 'a.b' in result['values']['columns']
        )
        read['rows'] += rows
    runners = [('reference', reference), ('core', observation)]
    if cell is not None:
        old = copy.deepcopy(reference)
        old['identity']['runner'] = 'python-old'
        old['identity']['reference'] = None
        for each in (old, observation):
            each['identity']['cell'] = cell
        if runtime is not None:
            for each in (old, observation):
                each['host']['origins'] = [
                    {'class': name, 'module': 'sdm_client'}
                    for name in each['host']['warnings']
                ]
            old['host']['warnings'].append('DeprecationWarning')
            old['host']['origins'].append(
                {'class': 'DeprecationWarning', 'module': runtime}
            )
        runners.append(('old', old))
    files = []
    for name, item in runners:
        path = tmp_path / f'{name}.jsonl'
        path.write_text(json.dumps(item) + '\n', encoding='utf-8')
        files.append(path)
    register = tmp_path / 'register.yaml'
    register.write_text(yaml.safe_dump({'entries': [entry]}), encoding='utf-8')
    report = compare.compare(
        [cases], files[0], files[1:], improvements=register
    )
    return {rule for rule, _ in report.failures}


def test_an_approved_entry_takes_out_only_the_columns_it_names(
    tmp_path: Path,
) -> None:
    entry = {
        'id': 'P2E-1',
        'status': 'approved',
        'approved_by': 'owner',
        'cases': ['base.regression'],
        # A row of this harness's decisions the owner approved.
        'decision_ref': {
            'file': 'docs/design/rust-core/phase-2/decisions.md',
            'row': 'KO2-36',
        },
        'core_only_columns': ['users.a.b'],
    }
    # Not vacuous: with the entry proposed, the kept column fails each rule.
    proposed = {**entry, 'status': 'proposed', 'approved_by': None}
    assert RULES <= _failing(tmp_path, proposed)
    # Approved by the owner: the named column, and the reference's warning
    # of leaving it out, are taken out; nothing else differs.
    assert not RULES & _failing(tmp_path, entry)
    # A column the entry does not name still fails (the negative control),
    # and so does a column of another table of the same name.
    assert {'schema', 'payload', 'sql.result'} <= _failing(
        tmp_path, entry, also='c.d'
    )
    elsewhere = {**entry, 'core_only_columns': ['orders.a.b']}
    assert {'schema', 'payload', 'sql.result'} <= _failing(tmp_path, elsewhere)
    # Approved only in the candidate's own register, with no owner-approved
    # decision row behind it: nothing is taken out.
    unbacked = {**entry, 'decision_ref': None}
    assert RULES <= _failing(tmp_path, unbacked)
    # A runner answering as the reference does, leaving the column out
    # with its warning, still passes: nothing is taken out of either side.
    assert not RULES & _failing(tmp_path, entry, kept=False)
    # A result read with the column, its row count differing, still fails.
    assert 'sql.result' in _failing(tmp_path, entry, rows=1)
    # In another runtime cell, python-core against python-old, which leaves
    # the column out with its warning as the reference does: taken out the
    # same way; with the entry proposed, it fails there.
    assert not RULES & _failing(tmp_path, entry, cell='dbr17-pandas2')
    assert {'schema', 'warnings', 'payload'} <= _failing(
        tmp_path, proposed, cell='dbr17-pandas2'
    )
    # With HOST-5's runtime deprecation in python-old too, both approved
    # differences compose: the left-out warning's origin goes with it.
    assert not RULES & _failing(
        tmp_path, entry, cell='floor-pandas2.2', runtime='pandas'
    )
    # A deprecation raised from the client's own files is no runtime's.
    assert 'warnings' in _failing(
        tmp_path, entry, cell='floor-pandas2.2', runtime='sdm_client'
    )
    # An entry for another case takes out nothing here.
    other = {**entry, 'cases': ['base.binary']}
    assert RULES <= _failing(tmp_path, other)
