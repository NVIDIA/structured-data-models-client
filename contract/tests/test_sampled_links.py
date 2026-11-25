# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-column precedence never excuses a missing winning target."""

import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'sampled_links', Path(__file__).with_name('_sampled_links.py')
)
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)


def shared_column():
    section = {
        'instance_table': {'columns': ['instance_id'], 'rows': [[0]]},
        'related_tables': {
            'events': {'columns': ['instance_id', 'cid'], 'rows': [[0, 7]]},
            'first': {'columns': ['instance_id', 'id'], 'rows': [[0, 3]]},
            'last': {'columns': ['instance_id', 'id'], 'rows': [[0, 7]]},
        },
    }
    return {
        'schema': {
            'relationships': [
                {
                    'source_table': 'events',
                    'source_columns': ['cid'],
                    'target_table': target,
                    'target_columns': ['id'],
                }
                for target in ('first', 'last')
            ]
        },
        'context': copy.deepcopy(section),
        'predict': copy.deepcopy(section),
    }


def test_shared_column_uses_last_emitted_relationship():
    body = shared_column()
    CHECKS.assert_local_links(body)
    assert len(body['schema']['relationships']) == 2
    body['schema']['relationships'].reverse()
    with pytest.raises(AssertionError):
        CHECKS.assert_local_links(body)


@pytest.mark.parametrize('rows', [[], [[1, 7]], [[0, 7], [0, 7]]])
def test_winning_target_must_be_unique_and_in_same_instance(rows):
    body = shared_column()
    body['predict']['related_tables']['last']['rows'] = rows
    with pytest.raises(AssertionError):
        CHECKS.assert_local_links(body)
