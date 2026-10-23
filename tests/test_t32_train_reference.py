# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A train carries a reference file only when it adds fields to the lines
already there (the coordinator's option 1, 2026-10-22, for T3.2's three
text digests): no line added, removed or reordered, no field removed or
changed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/rust_core'))
import judging  # noqa: E402

PATH = 'contract/reference/0123.jsonl'
RULES = judging.patterns(
    (ROOT / 'scripts/rust_core/judging-paths.txt').read_text(encoding='utf-8')
)
LINES = [
    {'case': 'a', 'public': {'result': {'rows': 1, 'digest': 'x'}}},
    {'case': 'b', 'public': {'result': {'rows': 2, 'digest': 'y'}}},
]


def _text(lines: list[dict[str, object]]) -> str:
    return ''.join(json.dumps(line, sort_keys=True) + '\n' for line in lines)


def _with(**changes: object) -> list[dict[str, object]]:
    lines = json.loads(json.dumps(LINES))
    lines[0]['public'].update(changes)  # type: ignore[union-attr]
    return lines


def test_a_field_added_to_a_line_rides_the_train() -> None:
    new = _text(_with(result_text='t'))
    assert judging.additions_only(_text(LINES), new) is None
    read = {PATH: (_text(LINES), new)}.__getitem__
    assert judging.not_train(RULES, [PATH], read) == []
    # Without the contents to read, a reference file never rides.
    assert judging.not_train(RULES, [PATH]) == [PATH]


@pytest.mark.parametrize(
    ('name', 'new', 'said'),
    [
        (
            'a changed field',
            _with(result={'rows': 9, 'digest': 'x'}),
            'changed',
        ),
        ('a removed field', [{'case': 'a', 'public': {}}, LINES[1]], 'removed'),
        ('a removed line', LINES[:1], 'line(s)'),
        ('an added line', [*LINES, {'case': 'c'}], 'line(s)'),
        ('reordered lines', LINES[::-1], 'changed'),
    ],
)
def test_any_other_change_to_a_reference_file_is_refused(
    name: str, new: list[dict[str, object]], said: str
) -> None:
    found = judging.additions_only(_text(LINES), _text(new))
    assert found is not None and said in found, (name, found)
    read = {PATH: (_text(LINES), _text(new))}.__getitem__
    assert judging.not_train(RULES, [PATH], read) == [PATH]
