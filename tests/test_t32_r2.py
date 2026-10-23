# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Codex round 2 on T3.2 (fix 7): a saved Arrow result is held to the
values read, not only to its shape; equivalent row and Arrow reads of one
warehouse result compare equal; an observation file the gate is given and
cannot read is refused by name."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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
