# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What the Phase 5 acceptance checks share: the corpus, today's Python
answer for a query, the Rust answer through ``sdm_core._native``, and the one
comparison between them.

The interface these rely on is ``CONTRACT.md``, beside this file.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CORPUS = REPO / 'tests' / 'parity' / 'pql'
PQL_CRATE = REPO / 'rust' / 'crates' / 'pql'
CODES_PY = REPO / 'packages/sdm-pql/src/predictive_query/schema/codes.py'
# The commit Phase 5 started from (phases/phase-5/BASE in the harness).
BASE = 'b69a834c30b5307ee57a451a51eb65b41abb63ef'

PARSE_SYNTAX = 'PQL_PARSE_SYNTAX'
# Wording that belongs to ANTLR or pydantic, not to a message written for
# PQL's users (brief, task 1): ANTLR's error phrases and token type names, and
# pydantic's validation-error layout and links.
FOREIGN_WORDING = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r'mismatched input',
        r'extraneous input',
        r'input mismatch',
        r'no viable alternative',
        r'token recognition error',
        r"missing \S+ at '",
        r'\bATN\b',
        r'antlr',
        r'RecognitionException',
        r'<EOF>',
        r'\b(FQN_COLUMN|WILDCARD_COLUMN|REL_OP|NEG_INFTY|TIME_UNIT|QUOTED_ID'
        r'|FOR_EACH)\b',
        r'pydantic',
        r'validation errors? for \w+',
        r'value_error',
        r'input_value=',
        r'input_type=',
        r'ArgsKwargs',
    )
)

sys.path.insert(0, str(HERE.parent))


@functools.cache
def corpus_module() -> Any:
    r"""``tests/parity/test_pql_corpus.py``, whose ``resolve`` and ``_same``
    define the corpus comparison."""
    path = REPO / 'tests' / 'parity' / 'test_pql_corpus.py'
    spec = importlib.util.spec_from_file_location('_p5_pql_corpus', path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corpus_cases() -> list[dict[str, Any]]:
    return corpus_module().CASES


def base_cases(name: str) -> list[dict[str, Any]]:
    r"""The cases of corpus file ``name`` as Phase 5 found them."""
    text = subprocess.run(
        ['git', 'show', f'{BASE}:tests/parity/pql/{name}'],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(text)['cases']


def schemas() -> list[dict[str, Any]]:
    r"""Every distinct schema of the corpus, in first-seen order."""
    seen: dict[str, dict[str, Any]] = {}
    for case in base_cases('valid.json') + base_cases('invalid.json'):
        seen.setdefault(
            json.dumps(case['schema'], sort_keys=True), case['schema']
        )
    return list(seen.values())


# --- Today's Python ---------------------------------------------------------


def python_result(
    query: str,
    schema: dict[str, Any],
    profile: str,
    mode: str | None = None,
) -> dict[str, Any]:
    r"""Today's answer, as ``test_pql_corpus.resolve`` gives it, with the
    profile's mode replaced when ``mode`` is given.

    A query today's code raises on, rather than answering, gives
    ``{"exception": ...}``.
    """
    case = {'query': query, 'schema': schema, 'profile': profile}
    try:
        if mode is None:
            got = corpus_module().resolve(case)
        else:
            got = _resolve_in_mode(case, mode)
    except Exception as error:  # noqa: BLE001 - recorded, not hidden
        return {'exception': f'{type(error).__name__}: {error}'}
    return json.loads(json.dumps(got))


def _resolve_in_mode(case: dict[str, Any], mode: str) -> dict[str, Any]:
    import predictive_query

    name = case['profile']
    original = getattr(predictive_query, name)
    replaced = dataclasses.replace(original, mode=predictive_query.Mode(mode))
    # ``resolve`` looks the profile up by name on the package; it is put
    # back however resolution ends.
    setattr(predictive_query, name, replaced)
    try:
        return corpus_module().resolve(case)
    finally:
        setattr(predictive_query, name, original)


def python_date(literal: str) -> dict[str, Any]:
    r"""What today's PQL makes of a date literal: ``pd.to_datetime`` as
    ``Constant.typed_value`` calls it, and the ``UserWarning``\ s it gives.

    ``value`` is ``None`` where the literal is refused, else
    ``YYYY-MM-DDTHH:MM:SS`` (the year padded to four digits), then
    ``.fffffffff`` if pandas gives a fraction of a second, then the UTC offset
    as ``+HH:MM`` (``:SS`` when not whole minutes) if it gives a time zone.
    """
    import pandas as pd

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            stamp = pd.to_datetime(literal)
        except Exception:  # noqa: BLE001 - every failure is a refusal
            stamp = None
    value = None
    if isinstance(stamp, pd.Timestamp) and stamp is not pd.NaT:
        value = (
            f'{stamp.year:04d}-{stamp.month:02d}-{stamp.day:02d}T'
            f'{stamp.hour:02d}:{stamp.minute:02d}:{stamp.second:02d}'
        )
        if stamp.microsecond or stamp.nanosecond:
            value += f'.{stamp.microsecond:06d}{stamp.nanosecond:03d}'
        offset = stamp.utcoffset()
        if offset is not None:
            total = int(offset.total_seconds())
            sign, total = ('+' if total >= 0 else '-'), abs(total)
            value += f'{sign}{total // 3600:02d}:{total % 3600 // 60:02d}'
            if total % 60:
                value += f':{total % 60:02d}'
    given = [
        {
            'category': f'{w.category.__module__}.{w.category.__qualname__}',
            'message': str(w.message),
        }
        for w in caught
        if issubclass(w.category, UserWarning)
    ]
    return {'value': value, 'warnings': given}


# --- The Rust core, through the binding -------------------------------------


_NATIVE: dict[str, Any] = {}


def native() -> Any:
    r"""``sdm_core._native``, freshly built from this checkout and installed
    into this environment, once per process.

    A build that fails, or a module without the Phase 5 functions, fails
    every later call the same way, without building again.
    """
    if 'error' in _NATIVE:
        raise AssertionError(_NATIVE['error'])
    if 'module' in _NATIVE:
        return _NATIVE['module']
    try:
        from _rc import install_local_wheel

        install_local_wheel()
        import importlib

        importlib.invalidate_caches()
        module = importlib.import_module('sdm_core._native')
        for name in ('pql_validate', 'pql_parse_date'):
            assert hasattr(module, name), f'sdm_core._native has no {name}'
    except Exception as error:  # noqa: BLE001 - kept for every later call
        _NATIVE['error'] = f'the binding is not usable: {error}'
        raise AssertionError(_NATIVE['error']) from error
    _NATIVE['module'] = module
    return module


def rust_result(
    query: str,
    schema: dict[str, Any],
    profile: str,
    mode: str | None = None,
) -> dict[str, Any]:
    text = native().pql_validate(query, json.dumps(schema), profile, mode)
    assert isinstance(text, str), f'pql_validate returned {type(text)}'
    return json.loads(text)


def rust_date(literal: str) -> dict[str, Any]:
    text = native().pql_parse_date(literal)
    assert isinstance(text, str), f'pql_parse_date returned {type(text)}'
    return json.loads(text)


# --- The comparison -----------------------------------------------------------


def same(a: Any, b: Any) -> bool:
    return corpus_module()._same(a, b)


def foreign_wording(message: str) -> str | None:
    for pattern in FOREIGN_WORDING:
        found = pattern.search(message)
        if found:
            return found.group(0)
    return None


def mismatch(want: dict[str, Any], got: dict[str, Any]) -> str | None:
    r"""How ``got`` differs from ``want``, or ``None``.

    Both are ``{"tree" | "errors", "warnings"}``, as ``resolve`` gives them.
    Trees compare strictly, by value and JSON type. Errors compare by count,
    then code and span, then message, except that a ``PQL_PARSE_SYNTAX``
    message may differ, but may not carry ANTLR's or pydantic's wording.
    Warnings compare in full.
    """
    keys = {'tree', 'errors'}
    if not isinstance(got, dict) or set(got) - keys - {'warnings'}:
        return f'not a result: {str(got)[:300]}'
    if 'warnings' not in got or len(keys & set(got)) != 1:
        return f'a result has warnings and one of tree or errors: {list(got)}'
    if got['warnings'] != want.get('warnings', []):
        return f'warnings {got["warnings"]} != {want.get("warnings", [])}'
    if 'tree' in want:
        if 'tree' not in got:
            return f'refused: {got["errors"]}'
        if not same(want['tree'], got['tree']):
            from _rc import json_diff

            return f'tree differs: {json_diff(want["tree"], got["tree"])}'
        return None
    if 'errors' not in got:
        return 'resolved, where today it is refused'
    a, b = want['errors'], got['errors']
    if len(a) != len(b):
        return f'{len(b)} errors where today gives {len(a)}: {b}'
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if not isinstance(y, dict) or set(y) != {'code', 'span', 'message'}:
            return f'error {i} is not {{code, span, message}}: {y}'
        if y['code'] != x['code']:
            return f'error {i}: code {y["code"]} != {x["code"]}'
        if y['span'] != x['span']:
            return f'error {i} ({x["code"]}): span {y["span"]} != {x["span"]}'
        if x['code'] == PARSE_SYNTAX:
            if not isinstance(y['message'], str) or not y['message'].strip():
                return f'error {i}: an empty syntax message'
            word = foreign_wording(y['message'])
            if word:
                return (
                    f'error {i}: the syntax message carries {word!r}: '
                    f'{y["message"]!r}'
                )
        elif y['message'] != x['message']:
            return f'error {i} ({x["code"]}): message {y["message"]!r} != {x["message"]!r}'
    return None


def expected(case: dict[str, Any]) -> dict[str, Any]:
    r"""A corpus case's expectation, in the form ``resolve`` gives."""
    out: dict[str, Any] = {'warnings': case.get('expected_warnings', [])}
    if 'expected_tree' in case:
        out['tree'] = case['expected_tree']
    else:
        out['errors'] = case.get('expected_errors') or [case['expected_error']]
    return out
