# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.2's case-format exits on the Python loader (the package's test
author's), beside ``rust/crates/sdm-testkit/tests/t32_case_format.rs``,
which holds the Rust loader to the same vectors. One test per change:

- ``warehouses``, ``input.view``, ``input.connect`` and
  ``input.materialize.num_context_samples`` (design choice 1; the dry
  run's gaps; the S2 test authors' gaps in T3.2's brief): every change of
  ``contract/tests/t32-case-format-vectors.json`` is taken or refused by
  ``run_python.load_cases`` as the file lists it;
- P2-T2.1i-1: ``contract/tests/malformed-cases.json``'s ``refused_texts``
  (a ``fail_call`` of ``-0``, a ``rows_per_call`` of NaN, Infinity or
  -Infinity), which the Python loader took and the Rust loader refused,
  are refused;
- the dry run's format gaps: every stub case of
  ``docs/design/rust-core/phase-3/dryrun-plan.yaml`` loads (exit 2's
  "format" part; the pickled-request kind is claimed by a named client
  test instead, so its stub carries no ``input.pickle``), and the plan
  names Java's packages T3.11 to T3.13 (the brief's 2026-10-22 list).

Each test listed in ``contract/tests/t32-pending.json`` failed when written,
before the implementation, and is held to failing (the ``pending``
decorator, as T2.10's ``held``) until the implementer takes it out of
``pending``; ``pending`` stays within
``authored``, the test author's list.
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'contract/runners/run_python.py'
VECTORS = ROOT / 'contract/tests/t32-case-format-vectors.json'
MALFORMED = ROOT / 'contract/tests/malformed-cases.json'
PLAN = ROOT / 'docs/design/rust-core/phase-3/dryrun-plan.yaml'
PENDING_FILE = ROOT / 'contract/tests/t32-pending.json'


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


def runner() -> ModuleType:
    return _module('t32_format_run_python', RUNNER)


def _refusal(tmp_path: Path, name: str, text: str) -> str | None:
    r"""``load_cases``'s refusal of one case file holding ``text``, or None
    where it takes it."""
    run = runner()
    path = tmp_path / f'{name}.json'
    path.write_text(text, encoding='utf-8')
    try:
        run.load_cases([path])
    except run.RunnerFailure as refused:
        return str(refused)
    return None


def _changed(case: Any, change: dict[str, Any]) -> Any:
    out = copy.deepcopy(case)
    for item in change['set']:
        at = out
        for key in item['path'][:-1]:
            at = at[key]
        at[item['path'][-1]] = copy.deepcopy(item['value'])
    for path in change.get('remove', []):
        at = out
        for key in path[:-1]:
            at = at[key]
        del at[path[-1]]
    return out


def test_t32_the_pending_list_only_shrinks() -> None:
    assert _pending_names('pending') <= _pending_names('authored')


@pending
def test_t32_the_python_loader_takes_and_refuses_the_warehouse_vectors(
    tmp_path: Path,
) -> None:
    r"""Choice 1's fields and the S2 gaps: each accepted vector loads, each
    refused one is refused by name (a runner failure, never a traceback)."""
    vectors = json.loads(VECTORS.read_text(encoding='utf-8'))
    assert len(vectors['accepted']) >= 20 and len(vectors['refused']) >= 20
    problems = []
    for at, change in enumerate(vectors['accepted']):
        text = json.dumps(_changed(vectors['case'], change))
        refused = _refusal(tmp_path, f'accepted-{at}', text)
        if refused is not None:
            problems.append(f'accepted {change["why"]}: {refused}')
    for at, change in enumerate(vectors['refused']):
        text = json.dumps(_changed(vectors['case'], change))
        if _refusal(tmp_path, f'refused-{at}', text) is None:
            problems.append(f'refused {change["why"]}: taken')
    assert not problems, '\n'.join(problems)


@pending
def test_t32_the_python_loader_refuses_the_texts_json_cannot_hold(
    tmp_path: Path,
) -> None:
    r"""P2-T2.1i-1: a ``fail_call`` written ``-0`` and a ``rows_per_call``
    of NaN, Infinity or -Infinity are refused, as the Rust loader refuses
    them; the case they change loads."""
    listed = json.loads(MALFORMED.read_text(encoding='utf-8'))
    case = json.loads((ROOT / listed['case']).read_text(encoding='utf-8'))
    assert _refusal(tmp_path, 'unchanged', json.dumps(case)) is None
    texts = listed['refused_texts']
    assert {item['text'] for item in texts} == {
        '-0',
        'NaN',
        'Infinity',
        '-Infinity',
    }
    placeholder = '\x01t32-the-text\x01'
    taken = []
    for at, change in enumerate(texts):
        out = copy.deepcopy(case)
        where = out
        for key in change['path'][:-1]:
            where = where[key]
        where[change['path'][-1]] = placeholder
        text = json.dumps(out).replace(json.dumps(placeholder), change['text'])
        assert change['text'] in text
        if _refusal(tmp_path, f'text-{at}', text) is None:
            taken.append(change['why'])
    assert not taken, taken


def _stubs() -> list[tuple[str, dict[str, Any]]]:
    plan = yaml.safe_load(PLAN.read_text(encoding='utf-8'))
    return [(kind['name'], kind['case']) for kind in plan['kinds']]


@pending
def test_t32_every_dry_run_stub_loads(tmp_path: Path) -> None:
    r"""Exit 2's format part: each stub case of the dry-run plan is a case
    the loader takes (the gaps "``databases`` names only three databases",
    "``input.view`` and ``input.connect`` not case fields" and
    "``input.pickle``", which a named client test claims instead)."""
    stubs = _stubs()
    assert len(stubs) >= 31
    # Not vacuous: the plan holds the kinds the gaps name.
    assert any('warehouses' in case for _, case in stubs)
    assert any('view' in case['input'] for _, case in stubs)
    assert any('connect' in case['input'] for _, case in stubs)
    refused = {}
    for name, case in stubs:
        problem = _refusal(tmp_path, name, json.dumps(case))
        if problem is not None:
            refused[name] = problem
    assert not refused, '\n'.join(f'{k}: {v}' for k, v in refused.items())


@pending
def test_t32_the_dry_run_plan_names_java_s_packages_as_the_design_does() -> (
    None
):
    r"""The brief's 2026-10-22 list: the plan's Java kinds name T3.11 to
    T3.13 (the design's Java packages), not T3.10 to T3.12."""
    text = PLAN.read_text(encoding='utf-8')
    design = (ROOT / 'docs/design/rust-core/phase-3/design.md').read_text(
        encoding='utf-8'
    )
    for package, what in (
        ('T3.11', 'Java binding, API and Java S0'),
        ('T3.12', 'Java S1 through JDBC'),
        ('T3.13', 'Java on Databricks'),
    ):
        assert f'| {package} | {what} |' in design
    java = {
        kind['name']: kind['case']['why']
        for kind in yaml.safe_load(text)['kinds']
        if kind['name'].startswith('java')
    }
    assert {name: why[why.rindex('(') :] for name, why in java.items()} == {
        'java-s0-validate': '(T3.11)',
        'java-s1-jdbc-duckdb': '(T3.12)',
        'java-databricks-sql': '(T3.13)',
    }


#: A statement naming the time zone, in any case and spacing, and the one
#: form a case may set it in (``tests/test_contract_cases.py``'s check).
_ZONE = re.compile(r'time\s*_?\s*zone', re.IGNORECASE)
_GLOBAL_ZONE = re.compile(
    r"^\s*SET\s+GLOBAL\s+TimeZone\s*(=|TO)\s*'[^']+'\s*;?\s*$",
    re.IGNORECASE,
)
#: A zoned timestamp, as a type or a literal's type.
_ZONED = re.compile(
    r'\btimestamptz\b|\btimestamp\s+with\s+time\s+zone\b', re.IGNORECASE
)


def _zone_problems(where: str, base: list[str], setup: list[str]) -> list[str]:
    r"""C18: a statement of the base or the setup naming the zone other
    than as ``SET GLOBAL TimeZone``, or zoned timestamps made with no zone
    set for the database, so the answer would be the recorder's TZ."""
    statements = [*base, *setup]
    problems = [
        f'{where}: {statement!r} sets the zone for one connection'
        for statement in statements
        if _ZONE.search(_ZONED.sub('', statement))
        and not _GLOBAL_ZONE.match(statement)
    ]
    zoned = any(_ZONED.search(statement) for statement in statements)
    if zoned and not any(_GLOBAL_ZONE.match(s) for s in statements):
        problems.append(f'{where}: zoned timestamps with no zone set')
    return problems


def test_t32_the_zone_check_covers_zoned_cases_and_the_bases() -> None:
    r"""C18's static check, in T3.2's tests: every case making zoned
    timestamps sets the zone with ``SET GLOBAL TimeZone``, and every zone
    setting in ``contract/bases.yaml`` or a case's setup is that form; the
    seeded faults show each part reads what it names."""
    bases = yaml.safe_load(
        (ROOT / 'contract/bases.yaml').read_text(encoding='utf-8')
    )['bases']
    problems = []
    for name, statements in sorted(bases.items()):
        problems += _zone_problems(f'base {name}', list(statements), [])
    zoned = 0
    for path in sorted((ROOT / 'contract/cases').rglob('*.json')):
        case = json.loads(path.read_text(encoding='utf-8'))
        base = list(bases.get(case['input']['base'], []))
        setup = case['input'].get('setup', [])
        zoned += any(_ZONED.search(s) for s in [*base, *setup])
        problems += _zone_problems(case['id'], base, setup)
    assert problems == [], problems
    assert zoned, 'no case makes zoned timestamps: the check reads nothing'
    made = "CREATE TABLE t AS SELECT TIMESTAMPTZ '2024-01-01 00:00:00+00' AS ts"
    assert _zone_problems('seed', [], [made])
    assert not _zone_problems('seed', [], ["SET GLOBAL TimeZone = 'UTC'", made])
    assert not _zone_problems(
        'seed', ["SET GLOBAL TimeZone = 'Asia/Tokyo'"], [made]
    )
    assert _zone_problems('seed', ["SET TimeZone = 'UTC'"], [])
    assert _zone_problems(
        'seed',
        [],
        ['ALTER TABLE orders ALTER ts TYPE TIMESTAMP WITH TIME ZONE'],
    )
