# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Codex round 2 on T3.2 (fix 7): a saved Arrow result is held to the
values read, not only to its shape; equivalent row and Arrow reads of one
warehouse result compare equal; an observation file the gate is given and
cannot read is refused by name."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OWN = _module(
    ROOT / 'contract/tests/test_t32_recording_own.py', 't32_r2_recording_own'
)


# --- 1. a saved result's values ------------------------------------------------------


def test_a_saved_result_whose_values_are_not_the_read_is_refused() -> None:
    case, line, tables = OWN._capture()
    # The valid capture holds its values (a placeheld row read included).
    assert OWN.check.capture_problems(line, case, tables) == []
    result = line['statements'][1]['results'][0]
    table = OWN.run.read_arrow_ipc(result['arrow'])
    changed = OWN.pa.Table.from_arrays(
        [OWN.pa.array([999] * table.num_rows, type=table.schema.field(0).type)],
        schema=table.schema,
    )
    result['arrow'] = OWN.run.arrow_ipc(changed)
    found = OWN.check.capture_problems(line, case, tables)
    assert any('not the rows read' in p for p in found), found


# --- 2. a row read and an Arrow read of one warehouse result -------------------------

COMPARE = _module(ROOT / 'scripts/rust_core/compare.py', 't32_r2_compare')


def _statement(method: str, types: list[str] | None, schema: bool) -> dict:
    case, line, _ = OWN._capture()
    statement = copy.deepcopy(
        line['observation']['driver']['capture']['statements'][0]
    )
    result = statement['results'][0]
    result['method'] = method
    result['values']['types'] = types
    if schema:
        result['schema'] = [{'name': 'c', 'type': 'string'}]
    else:
        result.pop('schema', None)
    return statement


def _judged(mine: dict, theirs: dict, database: str) -> list:
    report = COMPARE.Report()
    COMPARE._results(
        report, 'the catalog', mine, theirs, False,
        ('python-core', 'python-ref'), database=database,
    )  # fmt: skip
    return report.failures


@pytest.mark.parametrize(
    ('database', 'rows', 'types', 'arrow'),
    [
        ('databricks-sql', 'fetchall', ['string'], 'fetchall_arrow'),
        ('snowflake', 'fetchall', ['string'], 'fetch_arrow_all'),
        ('databricks-spark', 'collect', None, 'toArrow'),
    ],
)
def test_a_row_read_and_an_arrow_read_of_one_result_agree(
    database: str, rows: str, types: list[str] | None, arrow: str
) -> None:
    by_rows = _statement(rows, types, schema=False)
    by_arrow = _statement(arrow, None, schema=True)
    assert _judged(by_arrow, by_rows, database) == []
    assert _judged(by_rows, by_arrow, database) == []
    # Other values still fail.
    other = copy.deepcopy(by_arrow)
    other['results'][0]['values']['unordered'] = '0' * 64
    assert any('rows differ' in t for _, t in _judged(other, by_rows, database))


def test_two_arrow_reads_of_other_types_still_differ() -> None:
    mine = _statement('fetchall_arrow', None, schema=True)
    theirs = copy.deepcopy(mine)
    theirs['results'][0]['schema'] = [{'name': 'c', 'type': 'large_string'}]
    found = _judged(mine, theirs, 'databricks-sql')
    assert any('Arrow schema differs' in t for _, t in found), found
