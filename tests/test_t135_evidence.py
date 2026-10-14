# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T1.35 part 1: the evidence-pack generator, held on small seeded inputs.

The independent test author's tests, written before the generator. They
fix the interface below; the implementer builds
``scripts/rust_core/evidence.py`` to it (design.md, "Phase 1b go or no-go
evidence pack", items 2 to 4; t1.35-plan.md section 1).

The command::

    scripts/rust_core/evidence.py --gate-out DIR --gate-json FILE
        --reference FILE --cases DIR --inventory FILE --register FILE
        [--heldout DIR] --out DIR

- ``--gate-out``: what ``gate.sh --locked 1b --out DIR`` kept, with the
  local per-cell runs beside it: every ``*.jsonl`` file is one runner's
  observations (``contract/observation.schema.json``), its runner and
  cell read from each line's ``identity``, never from the file's name;
  ``compare.log``, the ``compare`` step's output (``compare.py``'s lines:
  ``DIFFERS <rule>: <case>: ...``, ``diagnostic rule <rule>: ...``);
  ``ratchet.log``, the ratchet step's notes.
- ``--gate-json``: ``evidence/gate/<sha>.json``, which stays outside the
  tree (design.md:432): the pack names it by its SHA-256 and never copies
  it, and one inside ``--out`` is refused.
- ``--reference``: the python-ref observations
  (``contract/reference/<commit>.jsonl``), each case's reference outcome.
- ``--cases``: the case files (``contract/cases``, read recursively).
- ``--inventory``: ``phase-1b-inventory.yaml``. The per-case rows are its
  cases plus ``base.adventureworks``, no other case.
- ``--register``: ``register.yaml``.
- ``--heldout``: the held-out case files. Their contents never reach the
  pack or the output: digests and counts only.
- ``--out``: ``docs/design/rust-core/phase-1/evidence``. Nothing is written
  anywhere else, and nothing at all when an input is missing.

A missing input fails the run non-zero, naming it (its option or its file
name; a case the inventory lists without a case file, by its ID). A
regression is not a failure of the generator: the pack is written and says
so.

The pack, in ``--out``:

- ``cases.md`` (item 2): a markdown table with a ``class`` column, one row
  per case, the case's ID in its first cell, rows in ID order; columns for
  the ``reference``, ``python-core`` in each cell observed (the header names
  the cell), ``python-old``, ``rust-host``, the ``archive`` (``not
  measured``: 3887762 has no contract runner), the class, and the register
  entry (a header naming ``entry`` or ``register``). The class is exactly
  one of ``correct success``, ``expected refusal``, ``approved
  improvement`` and ``regression``:

  - every runner's outcome equal to the reference's on the fields it
    observes (result, code, and on a Python runner stage and exception
    class): ``correct success`` or ``expected refusal`` as the reference
    succeeded or refused;
  - every difference from the reference stated by an approved entry's
    ``case_outcomes`` item for that case, runner and cell (rust-host's,
    which has no cell, under ``ref-pandas3``), its ``new`` the observed
    outcome as ``ratchet.py``'s ``outcome_text`` writes it, and an entry
    approved by the owner where the reference succeeded and the runner
    refused: ``approved improvement``, naming the entry;
  - anything else, a runner not observing the case included:
    ``regression``. A proposed or rejected entry excuses nothing.

- ``seams.md`` (item 3): a table with a ``deepest`` column, one row per
  case, and a table with an ``unproven`` column, one row per seam, the
  seam's name in the first cell, listing the stages no case of the seam
  reached. The stages, from a core runner's observation: ``driver Arrow``
  (``core.fetches`` not empty), ``admission`` (the core went past fetched
  data to an answer), ``inferred schema`` (``core.described``),
  ``payload`` (``core.materialized``), ``result`` (``core.predicted``'s
  result); a case refused before any of them reached ``refusal``.
- ``comparator.md`` (item 4): a table with a ``rule`` column, one row per
  rule with differences and their count; the register's entries by
  approver, the owner's first.

Gating (``contract/tests/t135_gate.py``): ``t135-pending.json``'s
``pending["exits"]`` lists the tests that failed when written; each must
still fail until the implementer takes its ID out.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'scripts/rust_core/evidence.py'
TEMPLATE = ROOT / 'contract/cases/s1/base.regression.json'
REFERENCE = next(iter(sorted((ROOT / 'contract/reference').glob('*.jsonl'))))


def _gate() -> Any:
    name = 't135_gate'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'contract/tests/t135_gate.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _gate()

CELLS = ('ref-pandas3', 'dbr17-pandas2', 'floor-pandas2.2')
CANDIDATE = 'c0ffee' + '1' * 34
GATE_CANARY = 'GATE-CANARY-K9X4'
HELDOUT_CANARY = 'HELDOUT-CANARY-7Q3W'
STAGES = ('driver arrow', 'admission', 'inferred schema', 'payload', 'result')

#: An outcome: result, stage, exception class, code.
Outcome = tuple[str, str | None, str | None, str | None]

SUCCESS_PREDICT: Outcome = ('success', 'predict', None, None)
SUCCESS_MATERIALIZE: Outcome = ('success', 'materialize', None, None)
REFUSED_NAMED: Outcome = (
    'refusal',
    'from_source',
    'StructuredDataError',
    'INVALID_REQUEST',
)
REFUSED_UNNAMED: Outcome = ('refusal', 'materialize', 'RuntimeError', None)
REFUSED_LATE: Outcome = (
    'refusal',
    'materialize',
    'StructuredDataError',
    'INVALID_REQUEST',
)
CRASHED: Outcome = (
    'refusal',
    'materialize',
    'StructuredDataError',
    'INTERNAL_ERROR',
)


# --- the seeded world -----------------------------------------------------------


def _case(
    seam: str | None, until: str, reference: Outcome, depth: str
) -> dict[str, Any]:
    r"""A case: its seam, how far it runs, the reference's outcome, and how
    deep the core runners get (``none``, ``schema``, ``payload`` or
    ``result``). Every runner answers as the reference unless the world
    says otherwise (``runs``)."""
    return {
        'seam': seam,
        'until': until,
        'reference': reference,
        'depth': depth,
        'runs': {},
    }


def world() -> dict[str, Any]:
    r"""Six inventory cases, base.adventureworks and one case outside both,
    every class once; the time seam reaches nothing past a refusal and the
    keys seam no result."""
    cases = {
        'key.payload': _case(
            'keys', 'materialize', SUCCESS_MATERIALIZE, 'payload'
        ),
        'key.refused': _case('keys', 'from_source', REFUSED_NAMED, 'none'),
        'key.regress': _case(
            'keys', 'materialize', SUCCESS_MATERIALIZE, 'payload'
        ),
        'nested.improved': _case(
            'nested', 'predict', REFUSED_UNNAMED, 'result'
        ),
        'time.only-refused': _case(
            'time', 'from_source', REFUSED_NAMED, 'none'
        ),
        'base.adventureworks': _case(
            'baseline', 'predict', SUCCESS_PREDICT, 'result'
        ),
        's0.extra': _case(None, 'predict', SUCCESS_PREDICT, 'result'),
    }
    # A difference no entry states: rust-host crashes where the reference
    # succeeded.
    cases['key.regress']['runs'][('rust-host', None)] = (
        ('refusal', None, None, 'INTERNAL_ERROR'),
        'schema',
    )
    # An approved improvement: the core runs a case the reference failed
    # unnamed; python-old still fails as the reference does.
    for cell in CELLS:
        cases['nested.improved']['runs'][('python-core', cell)] = (
            SUCCESS_PREDICT,
            'result',
        )
    cases['nested.improved']['runs'][('rust-host', None)] = (
        ('success', None, None, None),
        'result',
    )
    improvement = _entry(
        'ZZ-1',
        'approved',
        'owner',
        [
            *(
                _item(
                    'nested.improved',
                    cell,
                    'python-core',
                    'refusal - at materialize (RuntimeError)',
                    'success at predict',
                )
                for cell in CELLS
            ),
            _item(
                'nested.improved',
                'ref-pandas3',
                'rust-host',
                'refusal -',
                'success',
            ),
        ],
    )
    return {
        'cases': cases,
        'inventory': [
            'key.payload',
            'key.refused',
            'key.regress',
            'nested.improved',
            'time.only-refused',
        ],
        'register': [
            _entry('AA-1', 'approved', 'reviewer', []),
            _entry('MM-1', 'proposed', None, []),
            improvement,
        ],
        'compare': [
            'statements per public call: key.payload: python-core 3, reference 3',
            'diagnostic rule sql.text: key.payload: rust-host: statement 1: differs',
            "DIFFERS outcome: key.regress: rust-host: outcome.code: expected None, observed 'INTERNAL_ERROR'",
            "DIFFERS outcome: key.regress: rust-host: outcome.result: expected 'success', observed 'refusal'",
            'DIFFERS core: key.regress: python-core against rust-host: core.materialized: differs',
            'compare: 3 differences',
        ],
        'heldout': {},
    }


def _entry(
    ident: str,
    status: str,
    approver: str | None,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        'id': ident,
        'title': f'Seeded entry {ident}',
        'kind': 'change',
        'old_behavior': 'The reference fails.',
        'new_behavior': 'The core answers.',
        'affected_inputs': 'The seeded case.',
        'cases': sorted({item['case'] for item in items}),
        'status': status,
        'approved_by': approver,
        'decision_ref': None,
        'notes': 'Seeded for T1.35 part 1.',
    }
    if items:
        entry['case_outcomes'] = items
    else:
        entry['refs'] = ['seeded']
    return entry


def _item(
    case: str, cell: str, runner: str, old: str, new: str
) -> dict[str, Any]:
    return {
        'case': case,
        'cell': cell,
        'runner': runner,
        'old': old,
        'new': new,
        'summary': 'seeded',
    }


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validator(name: str) -> Any:
    import jsonschema
    from referencing import Registry, Resource

    schemas = {
        key: json.loads((ROOT / f'contract/{key}.schema.json').read_text())
        for key in ('case', 'inventory', 'observation')
    }
    registry = Registry().with_resources(
        (schema['$id'], Resource.from_contents(schema))
        for schema in schemas.values()
    )
    cls = jsonschema.validators.validator_for(schemas['case'])
    return cls(schemas[name], registry=registry)


def _valid(name: str, document: Any) -> None:
    errors = [
        f'{"/".join(map(str, e.absolute_path))}: {e.message}'
        for e in _validator(name).iter_errors(document)
    ]
    assert errors == [], f'seeded {name} is not valid: {errors}'


def _case_file(ident: str, case: dict[str, Any], extra: str = '') -> bytes:
    document = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    document['id'] = ident
    document['status'] = 'acceptance'
    if case['seam'] is None:
        document.pop('seam', None)
    else:
        document['seam'] = case['seam']
    document['until'] = case['until']
    if case['until'] != 'predict':
        document['core'].pop('predict', None)
    if case['until'] == 'from_source':
        document['core'].pop('materialize', None)
    result, stage, error, code = case['reference']
    expect: dict[str, Any] = {'outcome': result, 'stage': stage}
    if result == 'refusal':
        expect.update(error=error, code=code)
    document['expect'] = expect
    document['why'] = f'Seeded for T1.35 part 1. {extra}'.strip()
    document['source'] = f'seeded {extra}'.strip()
    if extra:
        document['input']['setup'] = [f"SELECT '{extra}' AS seeded"]
    _valid('case', document)
    return (json.dumps(document, indent=2) + '\n').encode()


def _core(depth: str, outcome: Outcome, query: str) -> dict[str, Any]:
    fetched = depth in ('schema', 'payload', 'result')
    code = outcome[3]
    return {
        'inputs': [
            {
                'op': 'validate_query',
                'query': query,
                'profile': {},
                'schema': {},
            }
        ],
        'fetches': [{'request': {'table': 'users'}, 'plan': {}}]
        if fetched
        else [],
        'accepted': depth != 'none' or code is None,
        'described': {'tables': []} if fetched else None,
        'materialized': {'version': 1, 'query': query, 'payload': {}}
        if depth in ('payload', 'result')
        else None,
        'predicted': {
            'body': {},
            'model': 'kumo-relational',
            'request_id': 'seeded',
            'metadata': {},
            'result': {
                'columns': [{'name': 'id', 'type': 'int64'}],
                'rows': [[{'i64': '0'}], [{'i64': '1'}]],
            },
        }
        if depth == 'result'
        else None,
        'error': None
        if code is None
        else {
            'kind': 'internal'
            if code == 'INTERNAL_ERROR'
            else 'invalid_request',
            'code': code,
        },
    }


def _observation(
    runner: str,
    cell: str | None,
    ident: str,
    case_sha: str,
    outcome: Outcome,
    depth: str,
    query: str,
) -> dict[str, Any]:
    reference = json.loads(
        REFERENCE.read_text(encoding='utf-8').splitlines()[0]
    )['identity']
    core_runner = runner in ('python-core', 'rust-host')
    result, stage, error, code = outcome
    if runner == 'rust-host':
        stage, error = None, None
    line = {
        'format': 'sdm-observation/1',
        'case': ident,
        'identity': {
            'runner': runner,
            'cell': None if runner == 'rust-host' else cell,
            'platform': 'linux-x86_64',
            'commit': {'sha': CANDIDATE, 'clean': True},
            'reference': reference['reference']
            if runner == 'python-ref'
            else None,
            'native': {'sha256': '0' * 64} if runner == 'python-core' else None,
            'native_loaded': None
            if runner == 'rust-host'
            else runner == 'python-core',
            'operations': ['validate_query'] if core_runner else [],
            'contract': 23 if core_runner else None,
            'versions': {'duckdb': '1.5.6'},
            'inputs': {'case': case_sha, 'bases': '0' * 64},
        },
        'outcome': {
            'result': result,
            'stage': stage,
            'error': error,
            'code': code,
        },
        'host': None if runner == 'rust-host' else {'warnings': []},
        'core': _core(depth, outcome, query) if core_runner else None,
    }
    _valid('observation', line)
    return line


def write(state: dict[str, Any], root: Path) -> dict[str, Path]:
    r"""The seeded world as the generator's inputs under ``root``."""
    paths = {
        'gate-out': root / 'gate-out-dir',
        'gate-json': root / 'outside-tree/gate' / f'{CANDIDATE}.json',
        'reference': root / 'ref-0123abc.jsonl',
        'cases': root / 'case-files-dir',
        'inventory': root / 'inventory.yaml',
        'register': root / 'register.yaml',
        'heldout': root / 'heldout-dir',
        'out': root / 'repo/docs/design/rust-core/phase-1/evidence',
    }
    for key in ('gate-out', 'cases', 'heldout'):
        paths[key].mkdir(parents=True)
    paths['gate-json'].parent.mkdir(parents=True)
    (paths['cases'] / 's1').mkdir()
    by_file: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    reference_lines = []
    files = {}
    for ident, case in sorted(state['cases'].items()):
        data = _case_file(ident, case)
        files[ident] = data
        (paths['cases'] / 's1' / f'{ident}.json').write_bytes(data)
    for ident, case in sorted(state['heldout'].items()):
        data = _case_file(ident, case, HELDOUT_CANARY)
        files[ident] = data
        (paths['heldout'] / f'{ident}.json').write_bytes(data)
    every = {**state['cases'], **state['heldout']}
    for ident, case in sorted(every.items()):
        sha = _sha(files[ident])
        query = (
            f'PREDICT {HELDOUT_CANARY}' if ident in state['heldout'] else 'q'
        )
        if ident in state['cases']:
            reference_lines.append(
                _observation(
                    'python-ref',
                    'ref-pandas3',
                    ident,
                    sha,
                    case['reference'],
                    'none',
                    query,
                )
            )
        runs = [('python-core', cell) for cell in CELLS]
        runs += [('python-old', 'ref-pandas3'), ('rust-host', None)]
        for runner, cell in runs:
            outcome, depth = case['runs'].get(
                (runner, cell), (case['reference'], case['depth'])
            )
            if outcome is None:
                continue  # not observed
            by_file.setdefault((runner, cell), []).append(
                _observation(runner, cell, ident, sha, outcome, depth, query)
            )
    for (runner, cell), lines in sorted(
        by_file.items(), key=lambda item: (item[0][0], item[0][1] or '')
    ):
        name = runner if cell in (None, 'ref-pandas3') else f'{runner}-{cell}'
        (paths['gate-out'] / f'{name}.jsonl').write_text(
            ''.join(json.dumps(line, sort_keys=True) + '\n' for line in lines)
        )
    paths['reference'].write_text(
        ''.join(
            json.dumps(line, sort_keys=True) + '\n' for line in reference_lines
        )
    )
    (paths['gate-out'] / 'compare.log').write_text(
        '\n'.join(state['compare']) + '\n'
    )
    (paths['gate-out'] / 'ratchet.log').write_text(
        'ratchet: register: ZZ-1 added: approved, approved_by owner\n'
        'ratchet: ok\n'
    )
    inventory = {
        'format': 'sdm-phase-1b-inventory/1',
        'bases': {'shop': ['CREATE TABLE users AS SELECT 1 AS user_id']},
        'defaults': {
            'pql': 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
            'materialize': {'indices': [0, 1]},
        },
        'cases': [
            {
                'id': ident,
                'seam': state['cases'][ident]['seam'],
                'cells': ['int64/primary-key'],
                'base': 'shop',
                'expect': {
                    key: value
                    for key, value in zip(
                        ('outcome', 'stage', 'error', 'code'),
                        state['cases'][ident]['reference'],
                        strict=True,
                    )
                    if value is not None
                    or (
                        key == 'code'
                        and state['cases'][ident]['reference'][0] == 'refusal'
                    )
                },
                'why': 'seeded',
                'source': 'seeded',
            }
            for ident in state['inventory']
        ],
    }
    _valid('inventory', inventory)
    paths['inventory'].write_text(yaml.safe_dump(inventory, sort_keys=False))
    register = {'format': 'sdm-register/2', 'entries': state['register']}
    _valid_register(register)
    paths['register'].write_text(yaml.safe_dump(register, sort_keys=False))
    gate = {
        'sha': CANDIDATE,
        'halves': {
            '1b': {
                'base': 'b' * 40,
                'cell': 'ref-pandas3',
                'checkpoint': 'a' * 40,
                'checks': {
                    '4': {'result': 'ok', 'steps': ['runners', 'compare']},
                },
                'passed': True,
                'steps': {'compare': 'ok', 'runners': 'ok'},
                'seeded_note': GATE_CANARY,
            }
        },
    }
    paths['gate-json'].write_text(json.dumps(gate, indent=2, sort_keys=True))
    return paths


def _valid_register(register: Any) -> None:
    import jsonschema

    schema = json.loads(
        (ROOT / 'docs/design/rust-core/register.schema.json').read_text()
    )
    errors = [
        e.message
        for e in jsonschema.Draft202012Validator(schema).iter_errors(register)
    ]
    assert errors == [], f'seeded register is not valid: {errors}'


def arguments(paths: dict[str, Path], heldout: bool = False) -> list[str]:
    found = []
    for key in (
        'gate-out',
        'gate-json',
        'reference',
        'cases',
        'inventory',
        'register',
        'out',
    ):
        found += [f'--{key}', str(paths[key])]
    if heldout:
        found += ['--heldout', str(paths['heldout'])]
    return found


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    assert EVIDENCE.is_file(), f'{EVIDENCE.relative_to(ROOT)}: not written yet'
    return subprocess.run(
        [sys.executable, str(EVIDENCE), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
        timeout=120,
    )


def generate(
    tmp: Path, state: dict[str, Any] | None = None, heldout: bool = False
) -> tuple[dict[str, Path], subprocess.CompletedProcess[str]]:
    paths = write(state or world(), tmp)
    done = run(arguments(paths, heldout))
    assert done.returncode == 0, done.stdout + done.stderr
    for name in ('cases.md', 'seams.md', 'comparator.md'):
        assert (paths['out'] / name).is_file(), f'{name} not written'
    return paths, done


# --- reading the pack ----------------------------------------------------------


def _cells(line: str) -> list[str]:
    inner = line.strip()
    inner = inner[1:] if inner.startswith('|') else inner
    inner = inner[:-1] if inner.endswith('|') else inner
    return [
        cell.strip().replace('`', '').replace('**', '').strip()
        for cell in inner.split('|')
    ]


def tables(text: str) -> list[list[list[str]]]:
    r"""Each markdown table as rows of cells, the header first and the
    separator dropped."""
    found: list[list[list[str]]] = []
    block: list[list[str]] = []
    for line in [*text.splitlines(), '']:
        if line.lstrip().startswith('|'):
            cells = _cells(line)
            if not all(re.fullmatch(r':?-{3,}:?', cell) for cell in cells):
                block.append(cells)
        elif block:
            found.append(block)
            block = []
    return found


def table(text: str, *headers: str) -> list[list[str]]:
    r"""The one table with a header cell naming each of ``headers``."""
    hits = [
        rows
        for rows in tables(text)
        if all(
            any(header in cell.lower() for cell in rows[0])
            for header in headers
        )
    ]
    assert len(hits) == 1, f'{len(hits)} tables with columns {headers}'
    return hits[0]


def column(rows: list[list[str]], *words: str) -> int:
    hits = [
        index
        for index, cell in enumerate(rows[0])
        if all(word in cell.lower() for word in words)
    ]
    assert len(hits) == 1, f'{len(hits)} columns naming {words}: {rows[0]}'
    return hits[0]


def row(rows: list[list[str]], first: str) -> list[str]:
    hits = [cells for cells in rows[1:] if cells and cells[0] == first]
    assert len(hits) == 1, f'{len(hits)} rows for {first!r}'
    return hits[0]


def classes(paths: dict[str, Path]) -> list[list[str]]:
    return table((paths['out'] / 'cases.md').read_text(), 'class', 'rust-host')


def class_of(paths: dict[str, Path], case: str) -> str:
    rows = classes(paths)
    return row(rows, case)[column(rows, 'class')].lower()


def entry_of(paths: dict[str, Path], case: str) -> str:
    rows = classes(paths)
    hits = [
        index
        for index, cell in enumerate(rows[0])
        if 'entry' in cell.lower() or 'register' in cell.lower()
    ]
    assert len(hits) == 1, f'{len(hits)} entry columns: {rows[0]}'
    return row(rows, case)[hits[0]]


def pack(paths: dict[str, Path]) -> dict[str, bytes]:
    return {
        str(path.relative_to(paths['out'])): path.read_bytes()
        for path in sorted(paths['out'].rglob('*'))
        if path.is_file()
    }


def tree(root: Path) -> dict[str, str]:
    r"""Every file and directory under ``root``, a file by its digest."""
    found = {}
    for path in sorted(root.rglob('*')):
        name = str(path.relative_to(root))
        if path.is_symlink():
            found[name] = f'link {os.readlink(path)}'
        elif path.is_dir():
            found[name] = 'dir'
        else:
            found[name] = _sha(path.read_bytes())
    return found


# --- item 2: the per-case table ---------------------------------------------------


@pytest.mark.parametrize(
    ('case', 'expected'),
    [
        pytest.param(case, expected, id=case)
        for case, expected in (
            ('key.payload', 'correct success'),
            ('base.adventureworks', 'correct success'),
            ('key.refused', 'expected refusal'),
            ('time.only-refused', 'expected refusal'),
            ('nested.improved', 'approved improvement'),
            ('key.regress', 'regression'),
        )
    ],
)
def test_each_case_is_classed(
    request: pytest.FixtureRequest, tmp_path: Path, case: str, expected: str
) -> None:
    def check() -> None:
        paths, _ = generate(tmp_path)
        assert class_of(paths, case) == expected
        if expected == 'approved improvement':
            assert 'ZZ-1' in entry_of(paths, case)

    G.held('exits', f'py:{request.node.name}', check)


def test_the_rows_are_the_inventory_and_base_adventureworks_in_order(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    def check() -> None:
        paths, _ = generate(tmp_path)
        listed = [cells[0] for cells in classes(paths)[1:]]
        wanted = sorted([*world()['inventory'], 'base.adventureworks'])
        assert listed == wanted

    G.held('exits', f'py:{request.node.name}', check)


def test_the_columns_carry_each_runner_and_cell(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    def check() -> None:
        paths, _ = generate(tmp_path)
        rows = classes(paths)
        cells = row(rows, 'nested.improved')
        assert 'refusal' in cells[column(rows, 'reference')].lower()
        for cell in CELLS:
            assert 'success' in cells[column(rows, 'python-core', cell)].lower()
        assert 'refusal' in cells[column(rows, 'python-old')].lower()
        assert 'success' in cells[column(rows, 'rust-host')].lower()
        archive = column(rows, 'archive')
        assert {r[archive].lower() for r in rows[1:]} == {'not measured'}

    G.held('exits', f'py:{request.node.name}', check)


def _no_entry(state: dict[str, Any]) -> str:
    state['register'] = [e for e in state['register'] if e['id'] != 'ZZ-1']
    return 'nested.improved'


def _proposed(state: dict[str, Any]) -> str:
    for entry in state['register']:
        if entry['id'] == 'ZZ-1':
            entry['status'], entry['approved_by'] = 'proposed', None
    return 'nested.improved'


def _rejected(state: dict[str, Any]) -> str:
    for entry in state['register']:
        if entry['id'] == 'ZZ-1':
            entry['status'] = 'rejected'
    return 'nested.improved'


def _another_outcome(state: dict[str, Any]) -> str:
    for entry in state['register']:
        for item in entry.get('case_outcomes', []):
            if item['runner'] == 'python-core':
                item['new'] = 'success at materialize'
    return 'nested.improved'


def _new_refusal(approver: str) -> Callable[[dict[str, Any]], str]:
    def mutate(state: dict[str, Any]) -> str:
        case = state['cases']['key.payload']
        for cell in CELLS:
            case['runs'][('python-core', cell)] = (REFUSED_LATE, 'schema')
        case['runs'][('rust-host', None)] = (
            ('refusal', None, None, 'INVALID_REQUEST'),
            'schema',
        )
        state['register'].append(
            _entry(
                'RV-1',
                'approved',
                approver,
                [
                    *(
                        _item(
                            'key.payload',
                            cell,
                            'python-core',
                            'success at materialize',
                            'refusal INVALID_REQUEST at materialize (StructuredDataError)',
                        )
                        for cell in CELLS
                    ),
                    _item(
                        'key.payload',
                        'ref-pandas3',
                        'rust-host',
                        'success',
                        'refusal INVALID_REQUEST',
                    ),
                ],
            )
        )
        return 'key.payload'

    return mutate


def _not_observed(
    runner: str, cell: str | None
) -> Callable[[dict[str, Any]], str]:
    def mutate(state: dict[str, Any]) -> str:
        state['cases']['base.adventureworks']['runs'][(runner, cell)] = (
            None,
            'none',
        )
        return 'base.adventureworks'

    return mutate


def _python_old_differs(state: dict[str, Any]) -> str:
    state['cases']['key.refused']['runs'][('python-old', 'ref-pandas3')] = (
        CRASHED,
        'none',
    )
    return 'key.refused'


@pytest.mark.parametrize(
    'mutate',
    [
        pytest.param(_no_entry, id='no-entry'),
        pytest.param(_proposed, id='proposed-entry'),
        pytest.param(_rejected, id='rejected-entry'),
        pytest.param(_another_outcome, id='entry-for-another-outcome'),
        pytest.param(
            _new_refusal('reviewer'), id='new-refusal-by-the-reviewer'
        ),
        pytest.param(_not_observed('rust-host', None), id='rust-host-missing'),
        pytest.param(
            _not_observed('python-core', 'dbr17-pandas2'),
            id='one-cell-missing',
        ),
        pytest.param(_python_old_differs, id='python-old-differs'),
    ],
)
def test_a_regression_is_never_hidden(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], str],
) -> None:
    def check() -> None:
        state = world()
        case = mutate(state)
        paths, _ = generate(tmp_path, state)
        assert class_of(paths, case) == 'regression'
        # The seeded regression with no entry stays, whatever else changed.
        assert class_of(paths, 'key.regress') == 'regression'

    G.held('exits', f'py:{request.node.name}', check)


def test_a_new_refusal_the_owner_approved_is_an_improvement(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    def check() -> None:
        state = world()
        case = _new_refusal('owner')(state)
        paths, _ = generate(tmp_path, state)
        assert class_of(paths, case) == 'approved improvement'
        assert 'RV-1' in entry_of(paths, case)

    G.held('exits', f'py:{request.node.name}', check)


# --- item 3: the per-seam table ---------------------------------------------------


def test_each_case_names_its_deepest_stage(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    def check() -> None:
        paths, _ = generate(tmp_path)
        rows = table((paths['out'] / 'seams.md').read_text(), 'deepest')
        deepest = column(rows, 'deepest')
        for case, stage in (
            ('key.payload', 'payload'),
            ('key.refused', 'refusal'),
            ('nested.improved', 'result'),
            ('time.only-refused', 'refusal'),
            ('base.adventureworks', 'result'),
        ):
            assert row(rows, case)[deepest].lower() == stage, case

    G.held('exits', f'py:{request.node.name}', check)


def test_an_unproven_stage_is_listed(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    def check() -> None:
        paths, _ = generate(tmp_path)
        rows = table((paths['out'] / 'seams.md').read_text(), 'unproven')
        where = column(rows, 'unproven')
        # The time seam's one case is refused before any data: every
        # stage is unproven there.
        time = row(rows, 'time')[where].lower()
        assert [s for s in STAGES if s not in time] == []
        # The keys seam reaches a payload and never a result.
        keys = row(rows, 'keys')[where].lower()
        assert 'result' in keys
        assert [s for s in STAGES[:-1] if s in keys] == []
        # The nested seam reaches its result: nothing unproven.
        nested = row(rows, 'nested')[where].lower()
        assert [s for s in STAGES if s in nested] == []

    G.held('exits', f'py:{request.node.name}', check)


# --- item 4: the comparator summary ---------------------------------------------


def test_the_comparator_report_counts_by_rule_and_lists_the_owner_first(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    def check() -> None:
        paths, _ = generate(tmp_path)
        text = (paths['out'] / 'comparator.md').read_text()
        rows = table(text, 'rule')
        assert '2' in row(rows, 'outcome')[1:]
        assert '1' in row(rows, 'core')[1:]
        assert 'MM-1' in text
        assert text.index('ZZ-1') < text.index('AA-1')

    G.held('exits', f'py:{request.node.name}', check)


# --- what the pack holds and where it goes --------------------------------------


def test_the_pack_is_deterministic_and_sorted(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    def check() -> None:
        first, _ = generate(tmp_path / 'one')
        # The same world elsewhere, its observation lines shuffled and its
        # files named otherwise: the same pack, byte for byte.
        second = write(world(), tmp_path / 'two')
        shuffle = random.Random(1335)
        for path in sorted(second['gate-out'].glob('*.jsonl')):
            lines = path.read_text().splitlines(keepends=True)
            shuffle.shuffle(lines)
            path.unlink()
            (path.parent / f'z-{path.stem[::-1]}.jsonl').write_text(
                ''.join(lines)
            )
        lines = second['reference'].read_text().splitlines(keepends=True)
        shuffle.shuffle(lines)
        second['reference'].write_text(''.join(lines))
        done = run(arguments(second))
        assert done.returncode == 0, done.stdout + done.stderr
        assert pack(first) == pack(second)
        listed = [cells[0] for cells in classes(first)[1:]]
        assert listed == sorted(listed)

    G.held('exits', f'py:{request.node.name}', check)


def test_the_gate_json_is_named_by_its_sha_never_embedded(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    def check() -> None:
        paths, done = generate(tmp_path)
        raw = paths['gate-json'].read_bytes()
        text = b''.join(pack(paths).values()).decode()
        assert _sha(raw) in text
        for leaked in (GATE_CANARY, '"halves"', '"seeded_note"'):
            assert leaked not in text
            assert leaked not in done.stdout + done.stderr

    G.held('exits', f'py:{request.node.name}', check)


def test_a_gate_json_inside_the_pack_is_refused(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    def check() -> None:
        paths = write(world(), tmp_path)
        inside = paths['out'] / 'gate' / paths['gate-json'].name
        inside.parent.mkdir(parents=True)
        inside.write_bytes(paths['gate-json'].read_bytes())
        before = tree(paths['out'])
        done = run(
            [
                *arguments(paths)[:2],
                '--gate-json',
                str(inside),
                *arguments(paths)[4:],
            ]
        )
        assert done.returncode != 0
        assert 'gate' in (done.stdout + done.stderr).lower()
        assert tree(paths['out']) == before

    G.held('exits', f'py:{request.node.name}', check)


def test_held_out_cases_show_digests_and_counts_only(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    def check() -> None:
        state = world()
        for name in ('h11.synth-a', 'h11.synth-b'):
            state['heldout'][name] = _case(
                'keys', 'materialize', SUCCESS_MATERIALIZE, 'payload'
            )
        state['heldout']['h11.synth-b']['runs'][('rust-host', None)] = (
            ('refusal', None, None, 'INVALID_REQUEST'),
            'schema',
        )
        state['compare'][-1:-1] = [
            f"DIFFERS outcome: h11.synth-b: rust-host: outcome.result: expected '{HELDOUT_CANARY}', observed 'refusal'",
            f'diagnostic rule sql.text: h11.synth-a: python-core: {HELDOUT_CANARY}',
        ]
        paths, done = generate(tmp_path, state, heldout=True)
        for name, data in pack(paths).items():
            assert HELDOUT_CANARY not in data.decode(), name
        assert HELDOUT_CANARY not in done.stdout + done.stderr
        # Counted, never shown: the held-out difference is in the rule's
        # count.
        rows = table((paths['out'] / 'comparator.md').read_text(), 'rule')
        assert '3' in row(rows, 'outcome')[1:]

    G.held('exits', f'py:{request.node.name}', check)


def test_nothing_is_written_outside_the_evidence_dir(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    def check() -> None:
        paths = write(world(), tmp_path)
        before = tree(tmp_path)
        done = run(arguments(paths, heldout=True))
        assert done.returncode == 0, done.stdout + done.stderr
        after = tree(tmp_path)
        out = str(paths['out'].relative_to(tmp_path))
        changed = sorted(
            name
            for name in set(before) | set(after)
            if before.get(name) != after.get(name)
            and name != out
            and not name.startswith(out + os.sep)
            and not out.startswith(name + os.sep)
        )
        assert changed == []

    G.held('exits', f'py:{request.node.name}', check)


def test_a_link_in_the_evidence_dir_is_not_written_through(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    def check() -> None:
        paths = write(world(), tmp_path)
        outside = tmp_path / 'outside.md'
        outside.write_text('not the pack\n')
        paths['out'].mkdir(parents=True)
        (paths['out'] / 'cases.md').symlink_to(outside)
        run(arguments(paths))
        assert outside.read_text() == 'not the pack\n'

    G.held('exits', f'py:{request.node.name}', check)


# --- a missing input --------------------------------------------------------------


def _drop(
    key: str, name: str | None = None
) -> Callable[[dict[str, Path]], str]:
    def mutate(paths: dict[str, Path]) -> str:
        target = paths[key] if name is None else paths[key] / name
        if target.is_dir():
            for path in sorted(target.rglob('*'), reverse=True):
                if path.is_dir():
                    path.rmdir()
                else:
                    path.unlink()
            target.rmdir()
        else:
            target.unlink()
        return target.name

    return mutate


def _drop_case_file(paths: dict[str, Path]) -> str:
    (paths['cases'] / 's1' / 'key.refused.json').unlink()
    return 'key.refused'


@pytest.mark.parametrize(
    'mutate',
    [
        pytest.param(_drop('gate-out'), id='gate-out'),
        pytest.param(_drop('gate-out', 'compare.log'), id='compare-log'),
        pytest.param(_drop('gate-out', 'ratchet.log'), id='ratchet-log'),
        pytest.param(_drop('gate-out', 'rust-host.jsonl'), id='rust-host'),
        pytest.param(_drop('gate-out', 'python-old.jsonl'), id='python-old'),
        pytest.param(_drop('gate-json'), id='gate-json'),
        pytest.param(_drop('reference'), id='reference'),
        pytest.param(_drop('cases'), id='cases'),
        pytest.param(_drop('inventory'), id='inventory'),
        pytest.param(_drop('register'), id='register'),
        pytest.param(_drop('heldout'), id='heldout'),
        pytest.param(_drop_case_file, id='an-inventory-case-file'),
    ],
)
def test_a_missing_input_fails_by_name(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    mutate: Callable[[dict[str, Path]], str],
) -> None:
    def check() -> None:
        paths = write(world(), tmp_path)
        name = mutate(paths)
        done = run(arguments(paths, heldout=True))
        assert done.returncode != 0
        said = done.stdout + done.stderr
        runner = name.removesuffix('.jsonl')
        assert name in said or runner in said, said
        written = (
            [p for p in paths['out'].rglob('*') if p.is_file()]
            if paths['out'].exists()
            else []
        )
        assert written == []

    G.held('exits', f'py:{request.node.name}', check)


# --- the gating ----------------------------------------------------------------


def test_the_pending_list_only_shrinks() -> None:
    assert G.only_shrinks() == []
    assert set(G.gating()['pending']) == {'exits'}


def test_the_gate_holds_a_pending_test_to_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A listed test passes only while its check fails, and an unlisted one
    # only while its check passes.
    def passing() -> None:
        pass

    def failing() -> None:
        raise AssertionError('seeded')

    monkeypatch.setattr(G, 'pending', lambda group: {'seeded.listed'})
    G.held('exits', 'seeded.listed', failing)
    with pytest.raises(AssertionError, match='passes now'):
        G.held('exits', 'seeded.listed', passing)
    G.held('exits', 'seeded.unlisted', passing)
    with pytest.raises(AssertionError, match='seeded'):
        G.held('exits', 'seeded.unlisted', failing)


def test_the_seeded_world_is_valid_and_complete(tmp_path: Path) -> None:
    # The fixtures themselves, before any generator: every file is valid to
    # its schema (``write`` checks each), and the world holds each class.
    state = world()
    paths = write(state, tmp_path)
    assert sorted(p.name for p in paths['gate-out'].glob('*.jsonl')) == [
        'python-core-dbr17-pandas2.jsonl',
        'python-core-floor-pandas2.2.jsonl',
        'python-core.jsonl',
        'python-old.jsonl',
        'rust-host.jsonl',
    ]
    assert 'base.adventureworks' not in state['inventory']
    assert 's0.extra' not in state['inventory']
