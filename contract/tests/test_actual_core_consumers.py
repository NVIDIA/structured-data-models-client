# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Older consumer assertions use the actual checked operation products."""

import copy

import pytest
from test_t44_prediction_observation import actual_observation

from _actual_core import _reader, checked_operations


def test_consumer_roles_require_the_real_complete_native_journal():
    core, actual = actual_observation()
    observation = {'identity': {'operations': actual}, 'core': core}
    before = copy.deepcopy(observation)
    assert checked_operations(observation) == [
        'describe_source',
        'materialize',
        'predict',
    ]
    assert observation == before
    broken = copy.deepcopy(observation)
    broken['core']['predictions'] = []
    with pytest.raises(_reader().ObservationError):
        checked_operations(broken)
