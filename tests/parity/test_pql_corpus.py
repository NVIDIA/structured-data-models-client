# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Every PQL corpus case, resolved by today's ``predictive_query``.

Each JSON file under ``tests/parity/pql/`` holds ``cases``: a query, the
schema it is resolved against (``SchemaDefinition`` as JSON), the profile, and
either the resolved tree (``expected_tree``) or the diagnostics
(``expected_errors``: code, span and message, in order), and the warnings
resolving it gives (``expected_warnings``, absent when there are none). One
test per case, named by the case's id. ``SDM_PARITY_PQL_DIR`` reads another
directory.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import pytest

CORPUS = Path(
    os.environ.get(
        'SDM_PARITY_PQL_DIR', Path(__file__).resolve().parent / 'pql'
    )
)


def _cases() -> list[dict[str, Any]]:
    out = []
    for path in sorted(CORPUS.rglob('*.json')):
        out += json.loads(path.read_text(encoding='utf-8'))['cases']
    return out


def resolve(case: dict[str, Any]) -> dict[str, Any]:
    r"""``{"tree": ...}`` for a query that resolves, else ``{"errors": ...}``,
    with ``"warnings"``: each ``UserWarning`` resolving it gave, in order.

    Other warnings, such as a dependency's deprecations, are not the
    language's behavior; they are given again rather than recorded.
    """
    import predictive_query
    from predictive_query.schema import json_serde
    from predictive_query.schema.definition import SchemaDefinition

    schema = json_serde.from_json(case['schema'], SchemaDefinition)
    profile = getattr(predictive_query, case.get('profile', 'RELATIONAL'))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        resolved, response = predictive_query.validate(
            case['query'], schema, profile
        )
    given = []
    for warning in caught:
        if issubclass(warning.category, UserWarning):
            category = warning.category
            given.append(
                {
                    'category': f'{category.__module__}.'
                    f'{category.__qualname__}',
                    'message': str(warning.message),
                }
            )
        else:
            warnings.warn_explicit(
                warning.message,
                warning.category,
                warning.filename,
                warning.lineno,
            )
    if resolved is not None:
        return {'tree': json_serde.to_json_dict(resolved), 'warnings': given}
    return {
        'warnings': given,
        'errors': [
            {
                'code': error.code,
                'span': None
                if error.span is None
                else {
                    'start_row': error.span.start_row,
                    'start_col': error.span.start_col,
                    'end_row': error.span.end_row,
                    'end_col': error.span.end_col,
                },
                'message': error.message,
            }
            for error in response.errors
        ],
    }


def _same(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


CASES = _cases()


@pytest.mark.parametrize('case', CASES, ids=[case['id'] for case in CASES])
def test_pql_case(case: dict[str, Any]) -> None:
    got = json.loads(json.dumps(resolve(case)))
    assert got['warnings'] == case.get('expected_warnings', []), (
        f'warnings: {got["warnings"]}'
    )
    if 'expected_tree' in case:
        assert 'tree' in got, f'refused: {got["errors"]}'
        assert _same(got['tree'], case['expected_tree'])
        return
    want = case.get('expected_errors') or [case['expected_error']]
    assert 'errors' in got, 'the query resolved, and the case expects errors'
    assert len(got['errors']) == len(want), got['errors']
    for error, expected in zip(got['errors'], want, strict=True):
        assert error['code'] == expected['code'], error
        assert error['span'] == expected['span'], error
        if 'message' in expected:
            assert error['message'] == expected['message'], error
