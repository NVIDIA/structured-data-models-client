# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P5.quirks, the Python half: the table the Rust test
``rust/crates/pql/tests/accept_quirks.rs`` reads records, beside each input,
what today's Python gives for it, and has a row for every quirk the brief
lists. The Rust half checks the core gives the same.
"""

from __future__ import annotations

import json

from quirks import QUIRKS, TABLE, record, render


def test_the_table_records_todays_python():
    assert TABLE.is_file(), f'{TABLE} does not exist'
    stored = TABLE.read_text(encoding='utf-8')
    assert stored == render(record()), (
        f"{TABLE} is not what today's Python gives; quirks.py rewrites it"
    )


def test_every_quirk_has_a_row():
    table = json.loads(TABLE.read_text(encoding='utf-8'))
    assert table['quirks'] == QUIRKS
    covered = {row['quirk'] for row in table['queries'] + table['dates']}
    assert set(QUIRKS) <= covered, sorted(set(QUIRKS) - covered)
