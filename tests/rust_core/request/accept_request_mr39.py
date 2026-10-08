# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P85.mr39 (task 7): once MR !39 is in the stack, so that
``tests/golden/public_api.json`` no longer lists ``predictive_query``,
``ModelSpec``, ``RuntimeContract`` or ``Request.decomposition`` as public, no
line of the harness's ``ledger/rules.tsv`` whose note calls a symbol public
sends that symbol anywhere on that ground (``CONTRACT.md``, "MR !39").

While the golden lists them all, MR !39 is not in the stack and there is
nothing to fix.
"""

from __future__ import annotations

import fnmatch
import json

import pytest
from _request import REPO, current_package_path, harness

PUBLIC_API = REPO / 'tests' / 'golden' / 'public_api.json'
CLIENT = 'packages/sdm-client/src/sdm_client'
#: Each name MR !39 makes internal: how the golden shows it, and the
#: ledger symbol it is.
NAMES = {
    'ModelSpec': f'{CLIENT}/base.py::ModelSpec',
    'RuntimeContract': f'{CLIENT}/base.py::RuntimeContract',
    'Request.decomposition': f'{CLIENT}/request.py::Request.decomposition',
}


def _internal() -> list[str]:
    r"""The names MR !39 made internal, as far as the golden shows."""
    api = json.loads(PUBLIC_API.read_text(encoding='utf-8'))
    client = api.get('sdm_client', {})
    public = set(client.get('__all__', []))
    request = client.get('members', {}).get('Request', {})
    out = [
        name for name in ('ModelSpec', 'RuntimeContract') if name not in public
    ]
    if 'decomposition' not in request.get('public_attributes', []):
        out.append('Request.decomposition')
    return out


def test_the_rules_call_no_internal_name_public() -> None:
    internal = _internal()
    if not internal and 'predictive_query' in json.loads(
        PUBLIC_API.read_text(encoding='utf-8')
    ):
        pytest.skip('MR !39 is not in the stack: every name is still public')
    rules = harness() / 'ledger' / 'rules.tsv'
    assert rules.is_file(), rules
    problems = []
    for name in internal:
        path, _, symbol = NAMES[name].partition('::')
        for line in rules.read_text().splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            pattern, dest, *note = line.split('\t')
            path_glob, _, symbol_glob = pattern.partition('::')
            if not (
                fnmatch.fnmatch(
                    current_package_path(path), current_package_path(path_glob)
                )
                and fnmatch.fnmatch(symbol, symbol_glob or '*')
            ):
                continue
            if 'public' in ' '.join(note).lower():
                problems.append(
                    f'{name} is internal since MR !39, and the rule '
                    f'{pattern!r} ({dest}) that decides it says: {note}'
                )
            break
    assert not problems, '\n'.join(problems)
