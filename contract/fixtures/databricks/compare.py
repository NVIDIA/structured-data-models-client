# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Compares the core's Databricks statements with the old client's
(T3.4; the brief's exit 2): each golden of
``rust/crates/sdm-data/tests/databricks_goldens.rs`` printed beside the
old client's statement of its kind from ``reference-statements.json``
(the same table where the old client has one), as a unified diff, or
marked as the old client's own statement. Offline; it writes nothing:

    python -I contract/fixtures/databricks/compare.py

The two files beside it are synthetic unit fixtures, frozen, and are not
the recorded-result evidence the live runs give (owner B-lite: no offline
stand-in). ``capture.json`` (``"fixture": true``) holds, per case, the
statements the old client sent at the reference commit (43e55f4f) with
the answers a fake driver replays to the providers' and the catalog
reader's unit tests; ``reference-statements.json`` holds those statements
by request kind. They were written once by T3.4's test author with an
offline stand-in warehouse that answered from DuckDB; that stand-in is
removed (T3.4 fix 1, round 1's Codex P1-6), so the answers' values are
DuckDB's, typed as Unity Catalog types them, and are not evidence of what
a warehouse answers. C-1's live recording replaces both files.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


GOLDENS = (
    ROOT / 'rust' / 'crates' / 'sdm-data' / 'tests' / 'databricks_goldens.rs'
)


def _rust_text(literal: str) -> str:
    r"""A Rust string literal's text, for the escapes the goldens use."""
    out, at = [], 0
    while at < len(literal):
        char = literal[at]
        if char != '\\':
            out.append(char)
            at += 1
            continue
        code = literal[at + 1]
        if code == 'u':
            end = literal.index('}', at)
            out.append(chr(int(literal[at + 3 : end], 16)))
            at = end + 1
            continue
        out.append({'n': '\n', 't': '\t', '"': '"', '\\': '\\'}[code])
        at += 2
    return ''.join(out)


def _goldens() -> list[tuple[dict[str, Any], list[str]]]:
    source = GOLDENS.read_text(encoding='utf-8')
    found = []
    pattern = r'\(\s*r##"(.*?)"##,\s*&\[(.*?)\],?\s*\),'
    for request, body in re.findall(pattern, source, re.S):
        texts = re.findall(r'"((?:[^"\\]|\\.)*)"', body)
        found.append((json.loads(request), [_rust_text(t) for t in texts]))
    return found


def _kind(request: dict[str, Any]) -> str:
    if request['request'] == 'describe':
        return f'describe/{request["part"]}'
    return request['request']


def diff() -> int:
    import difflib

    reference = json.loads(
        (HERE / 'reference-statements.json').read_text(encoding='utf-8')
    )
    for request, statements in _goldens():
        kind = _kind(request)
        candidates = [s['text'] for s in reference['kinds'].get(kind, [])]
        if kind == 'sample_entities':
            candidates += [d['text'] for d in reference['sampling']]
        names = re.findall(
            r'"source":\s*\[[^\]]*"([^"]+)"\]', json.dumps(request)
        )
        named = [c for c in candidates if any(f'`{n}`' in c for n in names)]
        theirs = (named or candidates or [''])[0]
        ours = statements[-1] if statements else ''
        print(f'=== {kind}: {json.dumps(request, sort_keys=True)}')
        if any(
            ' '.join(ours.split()) == ' '.join(c.split()) for c in candidates
        ):
            print("(the old client's statement)")
            continue
        print(
            '\n'.join(
                difflib.unified_diff(
                    theirs.splitlines(),
                    ours.splitlines(),
                    'the old client',
                    'the core',
                    lineterm='',
                )
            )
        )
    return 0


def main() -> int:
    return diff()


if __name__ == '__main__':
    sys.exit(main())
