# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixture schemas reflect backend storage without mutating shared inputs."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def loader():
    spec = importlib.util.spec_from_file_location(
        'fixture_databases', ROOT / 'contract/runners/databases.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schema_mapping_preserves_unconverted_types_and_source():
    case = json.loads(
        (
            ROOT / 'contract/cases/s0/s0.accept.binary-classification.json'
        ).read_text()
    )
    bases = yaml.safe_load((ROOT / 'contract/bases.yaml').read_text())['bases']
    schema = case['core']['validate_query']['schema']
    original = copy.deepcopy(schema)
    expected = copy.deepcopy(schema)
    for table in expected['tables']:
        for column in table['columns']:
            if column['name'] == 'ts':
                column['dtype'] = 'string'
    assert expected != schema
    databases = loader()
    assert databases.stored_schema(schema, case, bases, 'sqlite') == expected
    assert databases.stored_schema(schema, case, bases, 'duckdb') == original
    assert schema == original
    malformed = copy.deepcopy(schema)
    malformed['tables'][0]['columns'][0]['name'] = 'missing_column'
    with pytest.raises(databases.UnmappedColumn, match='missing_column'):
        databases.stored_schema(malformed, case, bases, 'sqlite')
