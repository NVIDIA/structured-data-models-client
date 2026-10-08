# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P55.changelog: the Unreleased section tells users what changed.

``BRIEF.md`` task 5. These check that each entry exists and says the things
the brief requires; whether the wording is right is the reviewer's call.

* Time windows: no longer follow the session's time zone, naming the
  affected column types on each warehouse.
* Scan caps: PostgreSQL and Snowflake read a block sample of large tables;
  on large Databricks tables the sample is bounded but not guaranteed
  repeatable by Databricks.
* The multi-column link check asks the parent rather than comparing samples.
* The Phase 1 ``RandomState`` entry says that on larger tables a seeded
  request now picks different context and test rows, so seeded predictions
  can move slightly.
* No em dash (U+2014) anywhere in the file.
"""

from __future__ import annotations

import re

from _fixes import REPO

CHANGELOG = REPO / 'CHANGELOG.md'


def _unreleased() -> str:
    text = CHANGELOG.read_text()
    start = text.index('## Unreleased')
    following = re.search(r'^## ', text[start + 3 :], re.MULTILINE)
    end = start + 3 + following.start() if following else len(text)
    return text[start:end]


def _paragraphs() -> list[str]:
    r"""Each list item or paragraph of the Unreleased section, on one line."""
    out: list[str] = []
    for block in re.split(r'\n\s*\n|\n(?=- )', _unreleased()):
        out.append(' '.join(block.split()))
    return out


def _find(*words: str) -> list[str]:
    return [
        p
        for p in _paragraphs()
        if all(re.search(w, p, re.IGNORECASE) for w in words)
    ]


def test_the_time_zone_fix_is_described() -> None:
    found = _find(r'time zone', r'timestamptz', r'TIMESTAMP_LTZ', r'Databricks')
    assert found, 'no entry says time windows no longer follow the time zone'
    entry = found[0]
    for word in ('PostgreSQL', 'DuckDB', 'Snowflake', 'TIMESTAMP_TZ', 'UTC'):
        assert re.search(word, entry, re.IGNORECASE), (word, entry)


def test_the_scan_caps_are_described() -> None:
    assert _find(r'PostgreSQL', r'block sample'), (
        'no entry describes the PostgreSQL block sample'
    )
    assert _find(r'Snowflake', r'micro-?partition'), (
        'no entry describes the Snowflake block sample'
    )
    found = _find(r'Databricks', r'repeatable', r'guarantee')
    assert found, (
        'no entry says the Databricks sample is not guaranteed repeatable'
    )
    assert any(re.search(r'\bnot\b', p) for p in found), found


def test_the_link_check_is_described() -> None:
    assert _find(r'(multi-column|composite)', r'\blink', r'\bparent'), (
        'no entry describes the multi-column link check'
    )


def test_the_random_state_entry_is_reworded() -> None:
    found = _find(r'RandomState')
    assert found, 'the RandomState entry is gone'
    entry = found[0]
    for word in (r'seeded', r'context', r'\btest\b', r'slightly', r'larger'):
        assert re.search(word, entry, re.IGNORECASE), (word, entry)


def test_no_em_dash() -> None:
    assert chr(0x2014) not in CHANGELOG.read_text()
