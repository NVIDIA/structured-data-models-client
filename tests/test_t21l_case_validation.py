# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1l's exits (the package's test author's): the case loader's fast
path for long index lists changes no validation result.

``contract/runners/run_python.py``'s ``load_cases`` validates each case
with the case schema (``_invalid``, jsonschema) and then ``_indices``.
jsonschema walks every item of a long ``indices`` list, about 0.9 s for a
case of 100,000 entities, once per corpus load. T2.1l checks the long
homogeneous lists in plain Python with the schema's own per-item rules
(an ``input.materialize.indices`` item is a JSON integer or a string, a
``core.materialize.indices`` item an object) and has jsonschema validate
the case with those lists cut to a few leading items.

The oracle is the loader as it was before T2.1l, frozen here
(:func:`today`): the first jsonschema error of the whole case, its
location and rule, then ``_indices``'s per-item type check, each as
``load_cases`` words it. ``load_cases`` must give the same result for:

- a long list with a bad item at position 50,000, in each kind of list,
  refused by name ("refused by name");
- a long valid list ("accepted");
- every committed case and every change ``malformed-cases.json`` makes
  ("identical for every existing case and malformed entry").

Each passes before T2.1l (the loader is today's) and must pass after it.
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'contract/runners/run_python.py'
CASES = ROOT / 'contract/cases'
MALFORMED = ROOT / 'contract/tests/malformed-cases.json'
TEMPLATE = CASES / 's1/base.regression.json'
#: A list this long is past any cut a fast path makes, and its bad item at
#: :data:`DEEP` is far from the leading items jsonschema still sees.
LONG = 50_010
DEEP = 50_000


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('t21l_run_python', RUNNER)


@functools.cache
def _schema_validator() -> Any:
    r"""The case schema's jsonschema validator, as ``load_cases`` builds
    it."""
    return runner().validator('case')


def _first_schema_error(document: Any) -> str | None:
    r"""Today's ``_invalid``: the first jsonschema error over the whole
    document, its location and rule."""
    error = next(iter(_schema_validator().iter_errors(document)), None)
    if error is None:
        return None
    where = '/'.join(str(part) for part in error.absolute_path) or '(root)'
    named = ''
    if error.validator == 'additionalProperties':
        # From T3.2 the refusal names the keys the format does not have.
        allowed = set(error.schema.get('properties') or {})
        unknown = sorted(str(k) for k in error.instance if k not in allowed)
        named = f' ({", ".join(unknown)} not a field there)' if unknown else ''
    return f'{where} fails {error.validator!r}{named}'


def _typed(given: dict[str, Any]) -> str | None:
    r"""Today's ``_indices``: the types read, past what JSON Schema
    counts."""
    spec = given['materialize']
    for key in ('indices', 'candidates'):
        for position, value in enumerate(spec.get(key, ())):
            if type(value) not in (int, str):
                return f'input/materialize/{key}/{position} is not an integer'
    if 'top_k' in spec and type(spec['top_k']) is not int:
        return 'input/materialize/top_k is not an integer'
    if 'fail_call' in given and type(given['fail_call']) is not int:
        # From T3.2: -0 is refused too, so the words say what is asked.
        return 'input/fail_call is not a whole number written as one'
    return None


def today(name: str, case: Any) -> str | None:
    r"""The loader's result before T2.1l: None for a case it takes, else
    its refusal as ``load_cases`` words it."""
    problem = _first_schema_error(case) or _typed(case['input'])
    return None if problem is None else f'{name}: not a case: {problem}'


def loaded(path: Path) -> str | None:
    r"""``load_cases``'s result for the one case at ``path``: None where it
    takes it (the case as the file holds it), else its refusal."""
    run = runner()
    try:
        [(case, _)] = run.load_cases([path])
    except run.RunnerFailure as refused:
        return str(refused)
    assert case == json.loads(path.read_bytes())
    return None


def _written(tmp_path: Path, name: str, case: Any) -> Path:
    path = tmp_path / f'{name}.json'
    path.write_text(json.dumps(case), encoding='utf-8')
    return path


def _long(count: int = LONG) -> dict[str, Any]:
    r"""``base.regression``'s case asking about ``count`` entities, its
    call's list and its core's list both that long."""
    case = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    case['input']['materialize']['indices'] = list(range(count))
    case['core']['materialize']['indices'] = [
        {'t': 'i64', 'v': str(i)} for i in range(count)
    ]
    return case


#: A bad item at :data:`DEEP`, by list and what it is: the call's list
#: takes a JSON integer or a string, the core's an object.
DEEP_ITEMS = {
    'input-fraction': (('input', 'materialize', 'indices'), 1.5),
    'input-whole-float': (('input', 'materialize', 'indices'), 50_000.0),
    'input-bool': (('input', 'materialize', 'indices'), True),
    'input-null': (('input', 'materialize', 'indices'), None),
    'core-text': (('core', 'materialize', 'indices'), 'x'),
    'core-null': (('core', 'materialize', 'indices'), None),
}


@pytest.mark.parametrize('kind', list(DEEP_ITEMS))
def test_a_bad_item_deep_in_a_long_list_is_refused_by_name(
    kind: str, tmp_path: Path
) -> None:
    r"""Exit "refused by name": an item at position 50,000 of a long list,
    of a type its list does not take, is refused naming that position, as
    the whole-case validation refuses it."""
    where, value = DEEP_ITEMS[kind]
    case = _long()
    at: Any = case
    for key in where:
        at = at[key]
    at[DEEP] = value
    wanted = today(f'{kind}.json', case)
    assert wanted is not None and f'{"/".join(where)}/{DEEP} ' in wanted
    assert loaded(_written(tmp_path, kind, case)) == wanted


def test_a_long_valid_list_is_accepted(tmp_path: Path) -> None:
    r"""Exit "accepted": a case of 50,010 entities, every item of each
    list of its list's type (integers and text in the call's, objects in
    the core's), is taken as it is."""
    case = _long()
    case['input']['materialize']['indices'][7] = 'text-key'
    assert today('long.json', case) is None
    assert loaded(_written(tmp_path, 'long', case)) is None


def test_a_long_list_s_other_errors_come_first_as_today(
    tmp_path: Path,
) -> None:
    r"""Exit "identical": where a long case has more than one error, a bad
    deep item in each list and then a bad ``until`` too, the refusal names
    the error the whole-case validation names first, as today."""
    case = _long()
    case['input']['materialize']['indices'][DEEP] = 1.5
    case['core']['materialize']['indices'][DEEP] = 'x'
    worse = copy.deepcopy(case)
    worse['until'] = 'nowhere'
    for name, each in (('both-lists', case), ('until-too', worse)):
        wanted = today(f'{name}.json', each)
        assert wanted is not None
        assert loaded(_written(tmp_path, name, each)) == wanted


def _committed() -> list[Path]:
    return sorted(CASES.rglob('*.json'))


@pytest.mark.parametrize(
    'path', _committed(), ids=lambda path: path.relative_to(CASES).as_posix()
)
def test_every_committed_case_loads_as_today(path: Path) -> None:
    r"""Exit "identical for every existing case": each committed case's
    result is today's."""
    case = json.loads(path.read_bytes())
    assert loaded(path) == today(path.name, case)


def _changed(case: Any, path: list[Any], value: Any) -> Any:
    changed = copy.deepcopy(case)
    at = changed
    for key in path[:-1]:
        at = at[key]
    at[path[-1]] = copy.deepcopy(value)
    return changed


def _omitted(case: Any, path: list[Any]) -> Any:
    changed = copy.deepcopy(case)
    at = changed
    for key in path[:-1]:
        at = at[key]
    del at[path[-1]]
    return changed


def _malformed() -> list[tuple[str, str, Any]]:
    r"""Each change ``malformed-cases.json`` makes, by its group and
    position, as the case it makes."""
    document = json.loads(MALFORMED.read_text(encoding='utf-8'))
    base = json.loads((ROOT / document['case']).read_text(encoding='utf-8'))
    made = []
    for group in ('refused', 'accepted'):
        for at, change in enumerate(document[group]):
            made.append(
                (
                    group,
                    f'{group}-{at}',
                    _changed(base, change['path'], change['value']),
                )
            )
    for at, change in enumerate(document['omitted']):
        made.append(
            ('omitted', f'omitted-{at}', _omitted(base, change['path']))
        )
    return made


def test_every_malformed_change_loads_as_today(tmp_path: Path) -> None:
    r"""Exit "identical for every ``malformed-cases.json`` entry": each
    refused, accepted and omitted change gives today's result, a refused
    one refused and the others taken."""
    made = _malformed()
    assert {group for group, _, _ in made} == {'refused', 'accepted', 'omitted'}
    differ = []
    for group, name, case in made:
        wanted = today(f'{name}.json', case)
        assert (wanted is not None) == (group == 'refused'), name
        got = loaded(_written(tmp_path, name, case))
        if got != wanted:
            differ.append(f'{name}: {got!r} where today gives {wanted!r}')
    assert not differ, '\n'.join(differ)
