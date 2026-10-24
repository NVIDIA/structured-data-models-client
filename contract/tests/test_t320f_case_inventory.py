# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pending warehouse cases retain the existing live write-guard policy."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_odd_columns_use_only_the_existing_databricks_backtick_exclusion():
    spec = importlib.util.spec_from_file_location(
        't320f_captures', ROOT / 'contract/runners/captures.py'
    )
    assert spec is not None and spec.loader is not None
    captures = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = captures
    spec.loader.exec_module(captures)
    case = json.loads(
        (
            ROOT / 'contract/cases/s2/databricks.name.odd-columns.json'
        ).read_text()
    )
    assert '`' in json.dumps(case['input'])
    legs, reason = captures.NOT_LIVE[case['id']]
    assert set(legs) == {'databricks-sql', 'databricks-spark'}
    assert reason == captures.NOT_LIVE['name.backtick-feature'][1]
