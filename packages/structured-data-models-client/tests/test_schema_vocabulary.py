# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable schema vocabulary, without importing the retired Python PQL package."""

import json
from pathlib import Path

from sdm_client import _native
from sdm_client.base import PROFILE_MODELS


def test_semantic_types_preserve_supported_data_types():
    expected = json.loads(
        (Path(__file__).parent / 'data/schema-vocabulary.json').read_text()
    )
    assert sorted(map(list, _native.MEANINGS)) == expected['meanings']


def test_wire_data_type_names_preserve_the_schema_vocabulary():
    expected = json.loads(
        (Path(__file__).parent / 'data/schema-vocabulary.json').read_text()
    )
    assert dict(_native.DTYPE_NAMES) == expected['dtype_names']


def test_model_profile_names_preserve_diagnostics():
    expected = json.loads(
        (Path(__file__).parent / 'data/schema-vocabulary.json').read_text()
    )
    assert PROFILE_MODELS == expected['profile_models']
