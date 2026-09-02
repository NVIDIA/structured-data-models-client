# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared QA harness: run a probe, record what happened, never stop on failure."""

from __future__ import annotations

import json
import os
import pathlib
import traceback
from contextlib import contextmanager

FINDINGS = pathlib.Path(os.path.expanduser('~/sdm-qa/findings'))
FINDINGS.mkdir(parents=True, exist_ok=True)

TABULAR = os.environ.get('SDM_TABULAR_URL', 'http://localhost:8000')
RELATIONAL = os.environ.get('SDM_RELATIONAL_URL', 'http://localhost:8001')

_results: list[dict] = []


@contextmanager
def probe(area: str, name: str, expect: str):
    entry = {'area': area, 'name': name, 'expect': expect}
    try:
        yield entry
    except BaseException as exc:  # noqa: BLE001 - a probe records, never raises
        entry['status'] = 'RAISED'
        entry['error'] = f'{type(exc).__name__}: {exc}'
        entry['traceback'] = traceback.format_exc()[-1500:]
    else:
        entry.setdefault('status', 'PASS')
    _results.append(entry)
    mark = {'PASS': 'ok  ', 'RAISED': 'RAISE', 'FAIL': 'FAIL'}.get(
        entry['status'], entry['status']
    )
    detail = entry.get('error') or entry.get('detail') or ''
    print(
        f'{mark:5} [{area}] {name}' + (f' -- {detail[:160]}' if detail else '')
    )


def fail(entry: dict, detail: str) -> None:
    entry['status'] = 'FAIL'
    entry['detail'] = detail


def note(entry: dict, detail: str) -> None:
    entry['detail'] = detail


def save(filename: str) -> None:
    path = FINDINGS / filename
    path.write_text(json.dumps(_results, indent=2, default=str))
    bad = [r for r in _results if r['status'] != 'PASS']
    print(f'\n{len(_results)} probes, {len(bad)} not clean -> {path}')
    for r in bad:
        print(f'  {r["status"]:6} [{r["area"]}] {r["name"]}')
