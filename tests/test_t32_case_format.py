# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Case-loader validation and source timezone fixture requirements."""

from __future__ import annotations

import copy
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
