# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32g part 2d, fix 3: a re-recorded reference line is the one a move
is judged on.

Written by the package's independent test author, after the subagent's
second review of part 2d (P3-1). The comparator relaxes from the
candidate's reference line; ``ratchet.py``'s ``moves`` judged the stop
and the outcome an entry's ``old`` must state on the checkpoint's. The
rule (lane 1b's 260ffd60): where this change records a case's reference
line again, the stop and that outcome are the new line's, and only this
change's entries count; without a re-record nothing changes.

The seed (the subagent's probe), on ``tests/test_t132g_later_stage.py``'s
real fixtures: at the checkpoint, the hugeint case is admitted with the
core's success baselines held, its reference refusing at materialize;
the change records the reference again refusing at from_source (an
owner's entry naming it on python-ref), and python-core's public schema
is tampered. Relaxed from from_source, the schema is past the stop and
the tamper passes; held to the reference, it fails ``schema``. So:

- an entry whose ``old`` is the checkpoint's stale outcome does not relax
  (the tamper fails);
- an entry whose ``old`` is the new line's outcome relaxes (clean);
- the same entry at the checkpoint, not this change's, does not relax;
- with no re-record, an entry from the reference's own outcome relaxes
  from materialize, where the schema is still held: the tamper fails.

Gating (``contract/tests/t132g_gate.py``): none of these is pending; each
passed when written.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.gate_tooling

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


moves = _module(
    't132g_later_stage_moves_tests',
    ROOT / 'tests/test_t132g_later_stage_moves.py',
)
later = moves.later

origin, repo = moves.origin, moves.repo
HUGEINT, CORES, WHERE = moves.HUGEINT, moves.CORES, moves.WHERE
entry, item, of, has = moves.entry, moves.item, moves.of, moves.has
_edits, with_entry = moves._edits, moves.with_entry
admitted, holding_the_core = moves.admitted, moves.holding_the_core
set_reference, _re_recorded = moves.set_reference, moves._re_recorded
REFERENCE_TEXT = moves.REFERENCE_TEXT

#: The reference recorded again, refusing at from_source.
NEW_TEXT = 'refusal INVALID_REQUEST at from_source (ValueError)'
NEW_HOST_TEXT = 'refusal INVALID_REQUEST'


def _refusal_at_from_source() -> dict[str, Any]:
    line: dict[str, Any] = moves._refusal_reference('from_source')
    return line


def _tampered(found: Any, cases: dict[str, Any]) -> None:
    r"""python-core's public schema changed: evidence of from_source's end,
    which a reference refusing at materialize produced and one refusing
    at from_source did not."""
    table = found['python-core'][HUGEINT]['public']['schema']['tables'][0]
    table['duplicate_keys'] += 7


def _move(old: str, host_old: str, **fields: Any) -> dict[str, Any]:
    return entry(
        item(HUGEINT, 'python-core') | {'old': old},
        item(HUGEINT, 'rust-host') | {'old': host_old},
        **fields,
    )


def _compared(
    r: Path,
    tmp: Path,
    in_change: Any,
    at_checkpoint: Any = None,
    re_recorded: bool = True,
) -> list[tuple[str | None, str]]:
    refusal = _refusal_at_from_source()

    def change(found: Any, cases: dict[str, Any]) -> None:
        if re_recorded:
            moves._expecting(refusal)(found, cases)
        _tampered(found, cases)

    report = moves.compared(
        r,
        tmp,
        at_checkpoint=_edits(
            admitted(HUGEINT), holding_the_core(HUGEINT), at_checkpoint
        ),
        in_change=_edits(
            set_reference(HUGEINT, refusal) if re_recorded else None,
            with_entry(_re_recorded(REFERENCE_TEXT, NEW_TEXT))
            if re_recorded
            else None,
            in_change,
        ),
        reference=refusal if re_recorded else None,
        change=change,
    )
    found: list[tuple[str | None, str]] = of(report, HUGEINT)
    return found


def test_an_entry_from_the_stale_checkpoint_outcome_does_not_relax(
    repo: Path, tmp_path: Path
) -> None:
    found = _compared(
        repo,
        tmp_path,
        with_entry(_move(REFERENCE_TEXT, 'refusal INTERNAL_ERROR')),
    )
    assert has(found, 'schema', WHERE), found


def test_an_entry_from_the_new_line_s_outcome_relaxes(
    repo: Path, tmp_path: Path
) -> None:
    found = _compared(
        repo, tmp_path, with_entry(_move(NEW_TEXT, NEW_HOST_TEXT))
    )
    assert found == [], found


def test_an_entry_not_from_this_change_does_not_relax(
    repo: Path, tmp_path: Path
) -> None:
    found = _compared(
        repo,
        tmp_path,
        None,
        at_checkpoint=with_entry(_move(NEW_TEXT, NEW_HOST_TEXT, id='SEED-9')),
    )
    assert has(found, 'schema', WHERE), found


def test_without_a_re_record_the_schema_is_still_held(
    repo: Path, tmp_path: Path
) -> None:
    found = _compared(
        repo,
        tmp_path,
        with_entry(_move(REFERENCE_TEXT, 'refusal INTERNAL_ERROR')),
        re_recorded=False,
    )
    assert has(found, 'schema', WHERE), found
    # Relaxed from materialize all the same: nothing past it fails.
    assert not has(found, 'payload', WHERE), found
    assert not has(found, 'result', WHERE), found
