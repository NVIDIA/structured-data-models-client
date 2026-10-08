# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single governed old/core fixture expectation difference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CASE_ID = 'request-gaps/client.run/a-tabular-target-dropped-after-its-schema'
FIXTURE_PATH = 'tests/parity/fixtures/request-gaps.json'
FIX_ID = 'dropped-target-or-entity-column'
ENTRY = Path(__file__).with_name('fixture_corrections.json')


def same_json(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate correction key')
        result[key] = value
    return result


def load_correction(path: Path = ENTRY) -> dict[str, Any]:
    entry = json.loads(path.read_text(), object_pairs_hook=unique_object)
    validate_correction(entry)
    return entry


def validate_correction(entry: Any) -> None:
    if not isinstance(entry, dict) or set(entry) != {
        'fixture_path',
        'fix',
        'original_case',
        'core_outcome',
    }:
        raise ValueError('invalid correction entry')
    original = entry['original_case']
    if (
        entry['fixture_path'] != FIXTURE_PATH
        or entry['fix'] != FIX_ID
        or not isinstance(original, dict)
        or original.get('id') != CASE_ID
        or original.get('operation') != 'client.run'
        or 'expected' in original
        or original.get('expected_error')
        != {
            'class': 'builtins.KeyError',
            'code': 'KeyError',
            'message': "'churn'",
            'details_keys': [],
        }
        or original.get('expected_warnings') != []
    ):
        raise ValueError('correction is not the governed original case')
    outcome = entry['core_outcome']
    if not isinstance(outcome, dict) or set(outcome) != {'error', 'warnings'}:
        raise ValueError('corrected outcome must be an error and warnings')
    error = outcome['error']
    if (
        not isinstance(error, dict)
        or set(error) != {'class', 'code', 'message', 'details_keys'}
        or error['class'] != 'sdm_client.errors.StructuredDataError'
        or error['code'] != 'INVALID_REQUEST'
        or not isinstance(error['message'], str)
        or not error['message']
        or error['details_keys'] != []
        or outcome['warnings'] != []
    ):
        raise ValueError('invalid corrected refusal')


def validate_catalog(
    cases: list[dict[str, Any]],
    fixtures: Path,
    excluded: Any = (),
    entry: dict[str, Any] | None = None,
) -> None:
    entry = load_correction() if entry is None else entry
    validate_correction(entry)
    found = [case for case in cases if case.get('id') == CASE_ID]
    if len(found) != 1 or CASE_ID in excluded:
        raise ValueError('corrected case must run exactly once, never excluded')
    if not same_json(found[0], entry['original_case']):
        raise ValueError('original correction case changed')
    document = json.loads((fixtures / 'request-gaps.json').read_text())
    source = [case for case in document['cases'] if case.get('id') == CASE_ID]
    if len(source) != 1 or not same_json(source[0], entry['original_case']):
        raise ValueError('original correction fixture binding changed')


def expected_case(
    case: dict[str, Any],
    *,
    core: bool,
    entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if type(core) is not bool:
        raise ValueError('implementation choice must be explicit')
    entry = load_correction() if entry is None else entry
    validate_correction(entry)
    if case.get('id') != CASE_ID:
        return case
    if not same_json(case, entry['original_case']):
        raise ValueError('original correction case changed')
    if not core:
        return case
    return {
        **case,
        'expected_error': entry['core_outcome']['error'],
        'expected_warnings': entry['core_outcome']['warnings'],
    }
