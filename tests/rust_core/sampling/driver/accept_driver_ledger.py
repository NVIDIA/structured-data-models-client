# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P75.triage (task 7): the 65 ``triage:sampling`` rows are decided, each by
a precise line of ``ledger/rules.tsv`` with a reason.

The harness is ``$SDM_RC_HARNESS``, by default
``~/.local/share/sdm-dbx/rust-core``. Each row below, marked
``triage:sampling`` in ``ledger/ledger.tsv`` when these checks were locked
(DV24), must:

- have a destination in ``ledger.tsv`` that is not ``triage:*``;
- get it, by the rules ``tools/ledger.py`` applies (the first line whose
  path and symbol globs match), from a line naming symbols
  (``PATH::SYMBOLS``, with ``SYMBOLS`` not just ``*``) whose third column, its
  reason, is not empty, and whose destination is the one ``ledger.tsv``
  holds.

And no line of ``rules.tsv`` sends anything to ``triage:sampling``. That every
``sampling`` row is ``verified`` with a Rust symbol and fixtures is
``P75.ledger``.
"""

from __future__ import annotations

import csv
import fnmatch
import os
from pathlib import Path

BACKEND = (
    'packages/sdm-relational-engine/src/sdm_relational_engine/rfm/backend/'
)

#: The rows marked ``triage:sampling`` at lock time, by file under BACKEND.
TRIAGED = {
    'databricks/sampler.py': (
        '<module>',
        'DatabricksSampler',
        'DatabricksSampler.__init__',
        'DatabricksSampler.num_rows_dict',
        'DatabricksSampler.backend',
        'DatabricksSampler._projections',
        'DatabricksSampler._flatten',
        'DatabricksSampler._get_min_max_time_dict',
        'DatabricksSampler._sample_entity_table',
        'DatabricksSampler._by_pkey',
        'DatabricksSampler._by_fkey',
        'DatabricksSampler._by_time',
    ),
    'duckdb/sampler.py': (
        '<module>',
        'DuckDBSampler',
        'DuckDBSampler.__init__',
        'DuckDBSampler.backend',
        'DuckDBSampler._time_ref',
        'DuckDBSampler._get_missing_index_dict',
        'DuckDBSampler._indexed_column',
        'DuckDBSampler._by_pkey',
        'DuckDBSampler._by_fkey',
        'DuckDBSampler._by_time',
        'DuckDBSampler._warn_missing_indices',
        'DuckDBSampler._get_min_max_time_dict',
        'DuckDBSampler._sample_entity_table',
    ),
    'postgres/sampler.py': (
        '<module>',
        'PostgresSampler',
        'PostgresSampler.__init__',
        'PostgresSampler.num_rows_dict',
        'PostgresSampler.backend',
        'PostgresSampler._projections',
        'PostgresSampler._rows',
        'PostgresSampler._flatten',
        'PostgresSampler._postgres_type',
        'PostgresSampler._tie_break_term',
        'PostgresSampler._sanitize_output',
        'PostgresSampler._get_min_max_time_dict',
        'PostgresSampler._sample_entity_table',
        'PostgresSampler._by_pkey',
        'PostgresSampler._by_fkey',
        'PostgresSampler._by_time',
    ),
    'snow/sampler.py': (
        '<module>',
        'SnowSampler',
        'SnowSampler.__init__',
        'SnowSampler.num_rows_dict',
        'SnowSampler.backend',
        'SnowSampler._get_min_max_time_dict',
        'SnowSampler._sample_entity_table',
        'SnowSampler._seeded_key_hash',
        'SnowSampler._is_date',
        'SnowSampler._by_pkey',
        'SnowSampler._by_fkey',
        'SnowSampler._by_time',
    ),
    'sqlite/sampler.py': (
        '<module>',
        'SQLiteSampler',
        'SQLiteSampler.__init__',
        'SQLiteSampler._warn_missing_indices',
        'SQLiteSampler.backend',
        'SQLiteSampler._time_spelling',
        'SQLiteSampler._get_min_max_time_dict',
        'SQLiteSampler._seeded_order_by',
        'SQLiteSampler._sample_entity_table',
        'SQLiteSampler._by_pkey',
        'SQLiteSampler._by_fkey',
        'SQLiteSampler._by_time',
    ),
}


def harness() -> Path:
    return Path(
        os.environ.get('SDM_RC_HARNESS', '~/.local/share/sdm-dbx/rust-core')
    ).expanduser()


def rules() -> list[tuple[str, str, str, str, str]]:
    r"""``(line, path glob, symbol glob, destination, reason)``, in order."""
    out = []
    for line in (harness() / 'ledger' / 'rules.tsv').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        pattern, dest, *rest = line.split('\t')
        path_glob, _, symbol_glob = pattern.partition('::')
        out.append(
            (
                line,
                _current_path(path_glob),
                symbol_glob,
                dest,
                rest[0] if rest else '',
            )
        )
    return out


def test_the_triaged_rows_are_decided() -> None:
    assert sum(len(v) for v in TRIAGED.values()) == 65
    table = rules()
    with (harness() / 'ledger' / 'ledger.tsv').open() as f:
        ledger = {}
        for row in csv.DictReader(f, delimiter='\t'):
            key = (_current_path(row['file']), row['symbol'])
            assert key not in ledger, 'duplicate projected ledger row'
            ledger[key] = row
    problems = [
        f'rules.tsv still sends to triage:sampling: {line!r}'
        for line, _, _, dest, _ in table
        if dest == 'triage:sampling'
    ]
    for name, symbols in TRIAGED.items():
        path = _current_path(BACKEND + name)
        for symbol in symbols:
            row = ledger.get((path, symbol))
            if row is None:
                problems.append(f'{name}::{symbol}: not in ledger.tsv')
                continue
            if row['dest'].startswith('triage:'):
                problems.append(f'{name}::{symbol}: still {row["dest"]}')
                continue
            rule = next(
                (
                    r
                    for r in table
                    if fnmatch.fnmatch(path, r[1])
                    and fnmatch.fnmatch(symbol, r[2] or '*')
                ),
                None,
            )
            if rule is None:
                problems.append(f'{name}::{symbol}: no rule matches')
                continue
            line, _, symbol_glob, dest, reason = rule
            if symbol_glob in ('', '*'):
                problems.append(
                    f'{name}::{symbol}: decided by {line!r}, which names no '
                    f'symbols'
                )
            elif not reason.strip():
                problems.append(f'{name}::{symbol}: {line!r} gives no reason')
            elif dest != row['dest']:
                problems.append(
                    f'{name}::{symbol}: ledger.tsv says {row["dest"]}, the '
                    f'rule {line!r} says {dest}'
                )
    assert not problems, f'{len(problems)} problems:\n' + '\n'.join(
        problems[:30]
    )


def _current_path(path: str) -> str:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'switch'))
    from _switch import current_package_path

    return current_package_path(path)
