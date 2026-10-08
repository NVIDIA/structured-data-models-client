# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed genuine replay failures in owned byte snapshots, never canonical files."""

from __future__ import annotations

import copy

from expectations import decode_semantics
from support import Rejected, json_bytes, require, same_json, unique_json


def wrong_public_expectation(
    contents: dict, allowed_ids: set | None = None
) -> dict:
    for name in sorted(contents):
        if not name.startswith(
            'tests/parity/frozen-public/'
        ) or not name.endswith('.json'):
            continue
        document = unique_json(contents[name])
        if document.get('format') != 'p97-public-v2' or not document['cases']:
            continue
        positions = [
            index
            for index, case in enumerate(document['cases'])
            if allowed_ids is None or case['id'] in allowed_ids
        ]
        if not positions:
            continue
        position = positions[0]
        changed = copy.deepcopy(document)
        changed['cases'][position]['expected']['warnings'].append(
            {
                'class': 'P97SeededWrongExpectation',
                'code': None,
                'message': 'acceptance-owned incorrect expectation',
                'details': None,
            }
        )
        require(
            changed['cases'][position]['recipe']
            == document['cases'][position]['recipe']
            and changed['cases'][position]['replay_input']
            == document['cases'][position]['replay_input']
            and changed['cases'][position]['actual_old']
            == document['cases'][position]['actual_old'],
            'negative.input_unchanged',
        )
        return {**contents, name: json_bytes(changed)}
    raise Rejected('negative.no_public_case')


def exercise(
    root, contents: dict, execute, native_ids: set | None = None
) -> dict:
    original = decode_semantics(contents, root)
    changed = wrong_public_expectation(contents, native_ids)
    expected = decode_semantics(changed, root)
    require(expected != original, 'negative.no_expected_change')
    if native_ids is not None:
        original = {
            key: value for key, value in original.items() if key in native_ids
        }
        expected = {
            key: value for key, value in expected.items() if key in native_ids
        }
        require(expected != original, 'negative.no_native_expected_change')
    detected = False
    try:
        result = execute(changed)['result']
        detected = not same_json(result['observations'], expected)
    except Rejected as error:
        require(
            str(error) == 'batch.worker_failed', 'negative.unrelated_failure'
        )
        detected = True
    require(detected, 'negative.wrong_expected_accepted')
    dependencies = sorted(name for name in contents if name.endswith('.arrow'))
    require(bool(dependencies), 'negative.no_dependency')
    missing = {
        name: raw for name, raw in contents.items() if name != dependencies[0]
    }
    missing_detected = False
    try:
        result = execute(missing)['result']
        missing_detected = not same_json(result['observations'], original)
    except Rejected as error:
        require(
            str(error) == 'batch.worker_failed',
            'negative.unrelated_dependency_failure',
        )
        missing_detected = True
    require(missing_detected, 'negative.missing_dependency_accepted')
    return {'wrong_expectation': True, 'missing_dependency': True}
