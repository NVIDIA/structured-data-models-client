#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Print every ``PQL_PARSE_SYNTAX`` message of the corpus, as today's
Python recorded it and as the core now gives it, as a Markdown table.

    syntax_messages.py

P5.invalid allows these messages new text, and requires each change to be
listed in the phase report with its old and new text. This prints that list,
for the report and for the reviewer to compare against it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pql import PARSE_SYNTAX, corpus_cases, rust_result  # noqa: E402


def _cell(text: str) -> str:
    return text.replace('|', '\\|').replace('\r', '\\r').replace('\n', '\\n')


def main() -> int:
    print('| Case | Old message | New message |')
    print('|---|---|---|')
    changed = 0
    for case in corpus_cases():
        want = case.get('expected_errors') or []
        if not any(e['code'] == PARSE_SYNTAX for e in want):
            continue
        got = rust_result(
            case['query'], case['schema'], case.get('profile', 'RELATIONAL')
        ).get('errors', [])
        for old, new in zip(want, got, strict=False):
            if old['code'] != PARSE_SYNTAX or old['message'] == new['message']:
                continue
            changed += 1
            print(
                f'| `{case["id"]}` | {_cell(old["message"])} '
                f'| {_cell(new["message"])} |'
            )
    print(f'\n{changed} changed messages', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
