# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1e: a ``more_exact`` item may leave out a value present on one side
and null or absent on the other, by ``null_on_one_side``.

The owner's KO2-61 answers time.feature-time-of-day on SQLite with users
holding no time column where the reference names one; ``more_exact``
excused only a value both sides hold. An owner-approved item marked
``null_on_one_side`` leaves out, at its named paths only, a value on one
side and null or absent on the other; without the mark, at another path,
or under an entry the owner did not approve, the difference is kept.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
KEY = 'time.feature-time-of-day@sqlite'
FIELD = 'public.schema/tables/1/time_column'


def _compare() -> ModuleType:
    name = 't21e_compare_null'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'scripts/rust_core/compare.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _excused(tmp_path: Path, approved_by: str, mark: str) -> tuple[str, ...]:
    path = tmp_path / 'register.yaml'
    path.write_text(
        textwrap.dedent(
            f"""\
            entries:
            - id: X-1
              status: approved
              approved_by: {approved_by}
              decision_ref: {{file: docs/design/rust-core/phase-2/decisions.md, row: KO2-61}}
              more_exact:
              - case: {KEY}
                runners: [python-core]
                fields: ['{FIELD}']
            {mark}
            """
        ),
        encoding='utf-8',
    )
    found: tuple[str, ...] = (
        _compare().more_exact(path).get((KEY, 'python-core'), ())
    )
    return found


def _differs(excused: tuple[str, ...], mine: dict, theirs: dict) -> str | None:
    found: str | None = _compare()._path_of(
        {'tables': [{}, mine]},
        {'tables': [{}, theirs]},
        'public.schema',
        excused,
    )
    return found


def test_a_value_on_one_side_alone_is_left_out_only_where_marked(
    tmp_path: Path,
) -> None:
    marked = _excused(tmp_path, 'owner', '    null_on_one_side: true')
    core = {'name': 'users', 'time_column': None}
    reference = {'name': 'users', 'time_column': 'ts'}
    assert _differs(marked, core, reference) is None
    # Absent on one side, too.
    assert _differs(marked, {'name': 'users'}, reference) is None
    # Another path still differs.
    assert _differs(marked, {**core, 'name': 'u'}, reference)
    # Unmarked, or not the owner's, the value on one side alone differs.
    plain = _excused(tmp_path, 'owner', '')
    assert _differs(plain, core, reference)
    assert _excused(tmp_path, 'reviewer', '    null_on_one_side: true') == ()
    # A dict or a list on the other side is never left out.
    assert _differs(marked, core, {**reference, 'time_column': ['ts']})
