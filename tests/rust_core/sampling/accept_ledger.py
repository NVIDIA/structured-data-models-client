# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P7.ledger-sampling (task 9): the ``sampling`` rows this phase ports are
marked in the porting ledger.

The ``labels`` rows are P7.ledger's: ``tools/ledger.py check --phase 7``
requires them verified. The ``sampling`` crate is due at Phase 7.5, so that
check does not look at it; this one requires the building blocks this phase
ports to be ``ported`` or ``verified``, each with a Rust symbol whose last
segment names an item defined under ``rust/``.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import pytest
from _sampling import REPO

HARNESS = Path(
    os.environ.get('SDM_RC_HARNESS', '~/.local/share/sdm-dbx/rust-core')
).expanduser()
LEDGER = HARNESS / 'ledger' / 'ledger.tsv'
RFM = 'packages/sdm-relational-engine/src/sdm_relational_engine/rfm/'

ROWS = (
    ('backend/local/graph_store.py', 'LocalGraphStore.sanitize'),
    ('backend/local/graph_store.py', 'LocalGraphStore.get_pkey_map_dict'),
    ('backend/local/graph_store.py', 'LocalGraphStore.get_time_data'),
    ('backend/local/graph_store.py', 'LocalGraphStore.get_csc'),
    ('backend/local/graph_store.py', '_argsort'),
    ('base/sampler.py', '_normalize_text'),
    ('base/sampler.py', '_pooled_offset'),
    ('base/sampler.py', 'min_date_offset'),
    ('base/sampler.py', 'max_date_offset'),
    ('base/utils.py', 'date_offset_to_seconds'),
    ('base/utils.py', '_elastic_days'),
    ('base/utils.py', 'random_state'),
    ('rfm.py', '_date_offset_to_ns'),
)


def _ledger() -> dict[tuple[str, str], dict[str, str]]:
    assert LEDGER.is_file(), f'no ledger at {LEDGER}; set SDM_RC_HARNESS'
    with LEDGER.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    found = {}
    for row in rows:
        key = (_current_path(row['file']), row['symbol'])
        assert key not in found, 'duplicate projected ledger row'
        found[key] = row
    return found


def _rust_items() -> set[str]:
    names: set[str] = set()
    pattern = re.compile(
        r'\b(?:fn|struct|enum|type|trait|const|static|mod)\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)'
    )
    for path in (REPO / 'rust').rglob('*.rs'):
        if 'target' in path.relative_to(REPO / 'rust').parts:
            continue
        names.update(pattern.findall(path.read_text(encoding='utf-8')))
    return names


@pytest.mark.parametrize(
    ('file', 'symbol'), ROWS, ids=[f'{f}::{s}' for f, s in ROWS]
)
def test_the_row_is_ported(file: str, symbol: str) -> None:
    row = _ledger().get((_current_path(RFM + file), symbol))
    assert row is not None, f'the ledger has no row {file}::{symbol}'
    assert row['status'] in ('ported', 'verified'), (
        f'{file}::{symbol} is {row["status"]!r}, not ported or verified'
    )
    rust = row['rust_symbol'].strip()
    assert rust, f'{file}::{symbol} names no Rust symbol'
    last = re.split(r'::|\.', rust)[-1]
    assert last in _rust_items(), (
        f'{file}::{symbol} names {rust}, but nothing under rust/ defines {last}'
    )


def _current_path(path: str) -> str:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'switch'))
    from _switch import current_package_path

    return current_package_path(path)
