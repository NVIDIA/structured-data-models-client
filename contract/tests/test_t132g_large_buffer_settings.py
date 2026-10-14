# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32g part 2d, fixes 2 and 4: the large buffers' setting as DuckDB
reads it, or refused by name.

Written by the package's independent test author, after Codex's round 2
and round 3 reviews of part 2d. ``compare.py``'s ``large_buffers`` infers
whether DuckDB writes Arrow's large buffers for a case's statements from
its base's statements and its setup. Parsing SQL to find the setting was
whack-a-mole (a quoted value, a trailing comment, several statements in
one element, then nested block comments and dollar quotes), so the rule
is now strict (the orchestrator's decision):

- a base or setup ELEMENT whose raw text contains
  ``arrow_large_buffer_size`` anywhere (comments, strings and dollar
  quotes included) must be, after trimming whitespace and one optional
  trailing ``;``, exactly one canonical statement, matched in any case:
  ``SET [SESSION|GLOBAL] arrow_large_buffer_size (=|TO)? <bool>``, the
  bool ``true`` or ``false`` unquoted or single-quoted, or ``RESET
  [SESSION|GLOBAL] arrow_large_buffer_size``;
- any other such element fails the comparison by name ("unsupported
  setting statement"); an element without the name is ignored; the last
  canonical statement, the base's and then the setup's, wins. No
  splitting, no comment stripping.

The tests:

- controls on real DuckDB: for each canonical form, in the case's setup
  and in its base, the inferred mode is DuckDB's: the Arrow type of a
  ``VARCHAR`` it returns after running the elements;
- the comparator on the real large-buffer fixture
  (``tests/golden/t132g-later-stage``): a quoted ``'false'`` after the
  case's ``true`` holds its large catalog reads to small types;
- every other element naming the setting, in the setup or the base, fails
  the comparison by name: comments, several statements in one element, a
  quoted ``;``, nested block comments, dollar quotes, the name in a
  comment or a string, a read of it, another value. Fix 2's controls for
  a trailing comment and for several statements in one element now
  expect this instead of a parse: failing closed can only fail a case
  falsely, never pass one falsely, since a refused element infers no mode
  at all.

Gating (``contract/tests/t132g_gate.py``): a test named in
``t132g-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
when written. It still runs and must still fail; once it passes it fails,
naming itself, until it is taken out of the list.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import duckdb
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / 'tests/golden/t132g-later-stage'
LARGE = 't132g.later-key-varchar-large-buffers'
ON = 'SET arrow_large_buffer_size = true'
UNSUPPORTED = 'unsupported setting statement'


def _module(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G = _module('t132g_gate', ROOT / 'contract/tests/t132g_gate.py')
run_python = _module('run_python', ROOT / 'contract/runners/run_python.py')
compare = _module('compare', ROOT / 'scripts/rust_core/compare.py')

#: Each a run of canonical elements (or ones not naming the setting);
#: each ends in a mode DuckDB itself decides (:func:`duckdb_says`). Scopes
#: are not mixed: an unscoped SET is DuckDB's global value, which a
#: ``RESET SESSION`` leaves standing (see the package's report).
SPELLINGS: dict[str, list[str]] = {
    'unquoted true': [ON],
    'nothing': [],
    'elements without the name are ignored': [
        "SELECT 1; SELECT 'a;b' /* c */",
        ON,
        'SELECT 2 -- d',
    ],
    'quoted false after true': [ON, "SET arrow_large_buffer_size = 'false'"],
    'quoted true': ["SET arrow_large_buffer_size = 'true'"],
    'any case and to': ['set Arrow_Large_Buffer_Size to TRUE'],
    'no spaces': ['SET arrow_large_buffer_size=true'],
    'surrounding whitespace and a semicolon': [
        '   SET   arrow_large_buffer_size   =   true   ;  '
    ],
    'session': ["SET SESSION arrow_large_buffer_size = 'true'"],
    'global false after true': [
        ON,
        'SET GLOBAL arrow_large_buffer_size = false',
    ],
    'global to a quoted TRUE': ["SET GLOBAL arrow_large_buffer_size TO 'TRUE'"],
    'reset': [ON, 'RESET arrow_large_buffer_size'],
    'reset in any case with a semicolon': [
        ON,
        'reset ARROW_LARGE_BUFFER_SIZE;',
    ],
    'reset global': [ON, 'RESET GLOBAL arrow_large_buffer_size'],
    'false then true over two elements': [
        'SET arrow_large_buffer_size = false',
        'SET arrow_large_buffer_size = TRUE',
    ],
}


def duckdb_says(steps: list[str]) -> bool:
    r"""Whether DuckDB, after running ``steps`` one element at a time as
    the runners do, writes a ``VARCHAR`` as Arrow's ``large_string``."""
    connection = duckdb.connect()
    for step in steps:
        connection.execute(step)
    table = connection.execute("SELECT 'x'::VARCHAR AS v").to_arrow_table()
    kind = str(table.schema.field('v').type)
    assert kind in ('string', 'large_string'), kind
    return kind == 'large_string'


def inferred(steps: list[str], where: str) -> Any:
    r"""``compare.large_buffers`` for a case running ``steps`` in its
    setup, or in its base (``b``), the setup empty."""
    if where == 'setup':
        return compare.large_buffers({'input': {'setup': steps}}, {})
    return compare.large_buffers(
        {'input': {'base': 'b', 'setup': []}}, {'b': list(steps)}
    )


def test_the_controls_are_not_vacuous() -> None:
    said = {name: duckdb_says(steps) for name, steps in SPELLINGS.items()}
    assert any(said.values()) and not all(said.values())
    assert said['quoted false after true'] is False
    assert said['reset'] is False
    assert said['quoted true'] is True


@pytest.mark.parametrize('where', ['setup', 'base'])
@pytest.mark.parametrize('spelling', sorted(SPELLINGS))
def test_the_inferred_mode_is_the_one_duckdb_uses(
    spelling: str, where: str, request: pytest.FixtureRequest
) -> None:
    steps = SPELLINGS[spelling]
    wanted = duckdb_says(steps)

    def agrees() -> None:
        assert inferred(steps, where) is wanted, (steps, wanted)

    G.held('exits', f'py:{request.node.name}', agrees)


# --- the comparator, on the real fixture -------------------------------------------


def _compared(
    tmp: Path, setup: list[str], bases: dict[str, list[str]] | None = None
) -> Any:
    r"""The large-buffer fixture compared, python-core against the
    reference, its case's setup ``setup`` (and, with ``bases``, its base
    ``b``, the bases file's own bases added)."""
    [case] = [
        json.loads(p.read_text())
        for p in (FIXTURES / 'cases').glob('*.json')
        if p.stem == LARGE
    ]
    case['input']['setup'] = setup
    bases_file = run_python.BASES
    if bases is not None:
        loaded = yaml.safe_load(run_python.BASES.read_text())
        loaded['bases'].update(bases)
        bases_file = tmp / 'bases.yaml'
        bases_file.write_text(yaml.safe_dump(loaded, sort_keys=False))
        case['input']['base'] = 'b'
    directory = tmp / 'cases'
    directory.mkdir(exist_ok=True)
    path = directory / f'{LARGE}.json'
    path.write_text(json.dumps(case, indent=2))
    digests = {
        'case': run_python.digest(path.read_bytes()),
        'bases': run_python.digest(bases_file.read_bytes()),
    }
    files = []
    for who in ('python-ref', 'python-core'):
        [line] = [
            json.loads(text)
            for text in (FIXTURES / f'{who}.jsonl').read_text().splitlines()
            if json.loads(text)['case'] == LARGE
        ]
        line['identity']['inputs'].update(digests)
        written = tmp / f'{who}.jsonl'
        written.write_text(json.dumps(line) + '\n')
        files.append(written)
    return compare.compare(
        [directory], files[0], files[1:], bases_file=bases_file
    )


def _catalog(report: Any) -> list[str]:
    return [
        text
        for rule, text in report.failures
        if rule == 'sql.result'
        and 'Arrow schema differs from the types' in text
    ]


def _rest(setup: list[str]) -> list[str]:
    r"""The fixture case's other setup statements (its columns' types)."""
    [case] = [
        json.loads(p.read_text())
        for p in (FIXTURES / 'cases').glob('*.json')
        if p.stem == LARGE
    ]
    return [s for s in case['input']['setup'] if s != ON] + setup


def test_the_case_s_own_setting_reads_its_large_catalog_reads(
    tmp_path: Path,
) -> None:
    r"""Control: as recorded, the setting on, the catalog reads agree."""
    assert _catalog(_compared(tmp_path, [ON, *_rest([])])) == []


def test_a_quoted_false_after_true_holds_the_large_reads_to_small_types(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    report = _compared(
        tmp_path, [ON, *_rest(["SET arrow_large_buffer_size = 'false'"])]
    )

    def differs() -> None:
        assert _catalog(report), report.failures

    G.held('exits', f'py:{request.node.name}', differs)


# --- an unsupported form fails by name ------------------------------------------------


#: Elements naming the setting that are not one canonical statement; each
#: fails the comparison by name. Fix 2 read the first seven; the rule now
#: refuses them.
FORMS = {
    'a trailing line comment': (
        'SET arrow_large_buffer_size = false -- small again'
    ),
    'a trailing block comment': (
        'SET arrow_large_buffer_size = false /* small again */'
    ),
    'a reset and a select in one element': (
        'RESET arrow_large_buffer_size; SELECT 1'
    ),
    'a quoted semicolon, then a set, in one element': (
        "SELECT 'a;b'; SET arrow_large_buffer_size TO 'true'"
    ),
    'true then false in one element': (
        "SET arrow_large_buffer_size = 'true'; "
        'SET arrow_large_buffer_size = false'
    ),
    'false then true in one element': (
        'SET arrow_large_buffer_size = false; '
        'SET arrow_large_buffer_size = TRUE'
    ),
    'a reset with a comment': (
        'reset ARROW_LARGE_BUFFER_SIZE -- back to small'
    ),
    'a nested block comment': (
        'SELECT 1 /* outer /* inner */ SET arrow_large_buffer_size = true */'
    ),
    'a dollar quote': 'SELECT $$SET arrow_large_buffer_size = true$$',
    'the name in a comment alone': '-- arrow_large_buffer_size\nSELECT 1',
    'the name in a string': "SELECT 'arrow_large_buffer_size'",
    'a number': 'SET arrow_large_buffer_size = 1',
    'another word': "SET arrow_large_buffer_size = 'yes'",
    'a read of it': "SELECT current_setting('arrow_large_buffer_size')",
}


@pytest.mark.parametrize('where', ['setup', 'base'])
@pytest.mark.parametrize('form', sorted(FORMS))
def test_an_unsupported_setting_statement_fails_by_name(
    tmp_path: Path, form: str, where: str, request: pytest.FixtureRequest
) -> None:
    statement = FORMS[form]
    if where == 'setup':
        report = _compared(tmp_path, [ON, *_rest([statement])])
    else:
        shop = yaml.safe_load(run_python.BASES.read_text())['bases']['shop']
        report = _compared(
            tmp_path, _rest([]), bases={'b': [*shop, ON, statement]}
        )

    def named() -> None:
        assert any(UNSUPPORTED in text for _, text in report.failures), (
            report.failures
        )

    G.held('exits', f'py:{request.node.name}', named)
