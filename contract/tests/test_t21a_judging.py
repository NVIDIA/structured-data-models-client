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


def _temporary_flag(locked: str, wheel_run: str) -> str:
    r"""What gate.sh's own text passes the ratchet for the temporary rule,
    the block run as gate.sh runs it."""
    import re
    import subprocess

    text = (ROOT / 'scripts/rust_core/gate.sh').read_text()
    [block] = re.findall(r'^temporary=\(\)\n.*?^fi\n', text, re.M | re.S)
    script = (
        f'locked={locked!r}; wheel_run={wheel_run!r}\n{block}'
        'echo "${temporary[@]+"${temporary[@]}"}"\n'
    )
    done = subprocess.run(
        ['bash', '-c', script],
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    return done.stdout.strip()


def test_only_the_slice_gate_s_locked_run_holds_temporary_entries() -> None:
    r"""Fix 1 (round 1's P1-1): the slice gate's locked run, a Phase 2 half
    on the candidate's CI wheels (``--wheel-run``, as final_gate.sh runs
    it), passes the ratchet ``--locked``; Phase 1's halves, a locked run
    before the slice gate and an unlocked run do not."""
    assert _temporary_flag('2a', '12345') == '--locked'
    assert _temporary_flag('3a', '12345') == '--locked'
    assert _temporary_flag('2a', '') == ''
    assert _temporary_flag('1a', '12345') == ''
    assert _temporary_flag('1b', '') == ''
    assert _temporary_flag('', '') == ''


def test_an_exclusion_that_leaves_no_database_out_is_refused(
    tmp_path: Path,
) -> None:
    r"""Fix 1 (round 1's P3-4): ``databases`` naming all three, with an
    exclusion, is no case; its control, two of them, is one."""
    runner = _module(ROOT / 'contract/runners/run_python.py', 't21a_runner')
    case = json.loads(
        (ROOT / 'contract/cases/s0/s0.accept.sum-window.json').read_text()
    )
    path = tmp_path / 'case.json'
    for named, taken in (
        (['duckdb', 'sqlite'], True),
        (list(runner.DATABASES), False),
    ):
        path.write_text(
            json.dumps({**case, 'databases': named, 'exclusion': 'KO2-19'})
        )
        try:
            runner.load_cases([path])
            loaded = True
        except runner.RunnerFailure as failure:
            assert 'not a case' in str(failure)
            loaded = False
        assert loaded is taken, named
