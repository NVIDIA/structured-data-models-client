# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.3: lane C's Postgres-family entries name database-scoped lines.

Entries name ``<case>@postgres``, ``<case>@lakebase`` or a task, never a
case bare, so no approved entry excuses the DuckDB line of the same case.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TASK = re.compile(r'T[0-9]+\.[0-9]+[a-z]?')


def test_no_postgres_entry_names_a_case_bare() -> None:
    path = ROOT / 'docs/design/rust-core/register.yaml'
    entries = yaml.safe_load(path.read_text(encoding='utf-8'))['entries']
    lane_c = [e for e in entries if str(e['id']).startswith('P2C-')]
    assert lane_c
    for entry in lane_c:
        for case in entry['cases']:
            assert case.endswith(('@postgres', '@lakebase')) or TASK.fullmatch(
                case
            ), (
                entry['id'],
                case,
            )
