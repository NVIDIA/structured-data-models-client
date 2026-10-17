# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1a's own tests of its judging rules (fix 1): a DuckDB line's capture
is held to DuckDB's query log, and the refusal census counts a rule-less
site reached only when the refusal could have come from it alone."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


capture_check = _module(
    ROOT / 'contract/runners/capture_check.py', 't21a_capture_check'
)
census = _module(ROOT / 'contract/tools/refusal_census.py', 't21a_census')


def _reference_line(case: str) -> dict[str, object]:
    [path] = sorted((ROOT / 'contract/reference').glob('*.jsonl'))
    for text in path.read_text().splitlines():
        line: dict[str, object] = json.loads(text)
        if line['case'] == case:
            return line
    raise AssertionError(case)


def test_a_duckdb_line_without_its_query_log_is_not_complete() -> None:
    r"""Seeded: a null query log passes off DuckDB only; on a DuckDB line
    it fails, where its control, the log as recorded, passes."""
    line = _reference_line('base.regression')
    assert capture_check.complete(line) == []
    nulled = copy.deepcopy(line)
    nulled['driver']['query_log'] = None  # type: ignore[index]
    assert capture_check.complete(nulled) == [
        'a DuckDB observation without its query log'
    ]
    nulled['case'] = 'base.regression@sqlite'
    assert capture_check.complete(nulled) == []


def test_a_refusal_two_sites_share_reaches_neither() -> None:
    r"""Seeded: two rule-less sites whose errors carry the same code and
    fields are not identified, so a refusal with that code and those
    fields reaches neither; a site whose fields are its own is reached,
    and a named rule by its rule."""
    sites = [
        {'id': 'a', 'rule': None, 'code': 'UNSUPPORTED_TYPE', 'where': 'r:1'},
        {'id': 'b', 'rule': None, 'code': 'UNSUPPORTED_TYPE', 'where': 'r:2'},
        {
            'id': 'c',
            'rule': None,
            'code': 'UNSUPPORTED_REQUEST',
            'where': 'r:3',
        },
        {
            'id': 'rule:x',
            'rule': 'x',
            'code': 'UNSUPPORTED_REQUEST',
            'where': 'r:4',
        },
    ]
    probes = [
        {'site': 'a', 'marks': ['column']},
        {'site': 'b', 'marks': ['column']},
        {'site': 'c', 'marks': ['classes']},
        {'site': 'rule:x'},
    ]
    known = census.identifiable(sites, probes)
    assert known == {'c', 'rule:x'}
    refused = {
        'result': 'refusal',
        'code': 'UNSUPPORTED_TYPE',
        'detail_keys': ['column', 'kind'],
    }
    for site in sites[:2]:
        assert not census._reaches(
            site, refused, ['column'], site['id'] in known
        )
    classes = {
        'result': 'refusal',
        'code': 'UNSUPPORTED_REQUEST',
        'detail_keys': ['classes', 'kind'],
    }
    assert census._reaches(sites[2], classes, ['classes'], True)
    assert census._reaches(
        sites[3], {'result': 'refusal', 'rule': 'x'}, [], True
    )
