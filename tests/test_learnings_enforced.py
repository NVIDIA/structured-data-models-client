# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Every lesson of ``docs/design/rust-core/learnings.md`` names the
mechanism that enforces it (learnings.md, lesson 19; T3.1).

The rules, checked by the plain functions below:

- lessons are the ``### <n>. `` headings, numbered 1..N with no gap or
  repeat;
- each lesson's section, up to the next ``###`` or ``##`` heading, holds
  exactly one line starting with ``**Enforced by:**``;
- every backticked token on that line that looks like a repository path
  exists under the repository root; a bare file name counts only when
  exactly one tracked file has that name;
- at most three lessons say "wording only" in their heading;
- each Enforced-by line names at least one existing path, or says
  "wording only".

The real file is held to them, and seeded copies (changed in memory) show
each rule fails.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEARNINGS = ROOT / 'docs/design/rust-core/learnings.md'

LESSON = re.compile(r'^### (\d+)\. ')
ENFORCED = '**Enforced by:**'
WORDING_ONLY = 'wording only'
MAX_WORDING_ONLY = 3
PATH_SUFFIXES = ('.py', '.sh', '.txt', '.yaml', '.json', '.md')
NOT_A_PATH = (' ', '<', '*', '$', '://')


def tracked_names(root: Path = ROOT) -> dict[str, int]:
    r"""How many tracked files carry each file name."""
    listing = subprocess.run(
        ['git', 'ls-files', '-z'],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout.decode('utf-8')
    counts: dict[str, int] = {}
    for path in filter(None, listing.split('\0')):
        name = path.rsplit('/', 1)[-1]
        counts[name] = counts.get(name, 0) + 1
    return counts


def lessons(text: str) -> list[tuple[int, str, list[str]]]:
    r"""Each lesson's number, heading and section lines."""
    found: list[tuple[int, str, list[str]]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        match = LESSON.match(line)
        if match:
            current = []
            found.append((int(match.group(1)), line, current))
        elif line.startswith(('### ', '## ')):
            current = None
        elif current is not None:
            current.append(line)
    return found


def path_tokens(line: str) -> list[str]:
    r"""The backticked tokens of ``line`` that look like repository paths."""
    tokens = []
    for token in re.findall(r'`([^`]+)`', line):
        if any(bad in token for bad in NOT_A_PATH) or token.startswith('-'):
            continue
        if '/' in token or token.endswith(PATH_SUFFIXES):
            tokens.append(token)
    return tokens


def path_exists(token: str, root: Path, names: dict[str, int]) -> bool:
    if '/' in token:
        return (root / token).exists()
    return names.get(token, 0) == 1


def problems(text: str, root: Path, names: dict[str, int]) -> list[str]:
    r"""Every rule ``text`` breaks, one line each."""
    found = lessons(text)
    out: list[str] = []
    if not found:
        return ['no lessons']
    numbers = [number for number, _, _ in found]
    if numbers != list(range(1, len(numbers) + 1)):
        out.append(f'lesson numbers are {numbers}, not 1..{len(numbers)}')
    wording = [heading for _, heading, _ in found if WORDING_ONLY in heading]
    if len(wording) > MAX_WORDING_ONLY:
        out.append(
            f'{len(wording)} lessons are wording only (at most'
            f' {MAX_WORDING_ONLY}): {wording}'
        )
    for number, _, section in found:
        lines = [line for line in section if line.startswith(ENFORCED)]
        if len(lines) != 1:
            out.append(
                f'lesson {number}: {len(lines)} Enforced-by lines, not one'
            )
            continue
        line = lines[0]
        tokens = path_tokens(line)
        existing = [t for t in tokens if path_exists(t, root, names)]
        for token in tokens:
            if token not in existing:
                out.append(f'lesson {number}: missing path {token}')
        if not existing and WORDING_ONLY not in line:
            out.append(
                f'lesson {number}: names no existing path and is not'
                ' wording only'
            )
    return out


@pytest.fixture(scope='module')
def real() -> str:
    return LEARNINGS.read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def names() -> dict[str, int]:
    return tracked_names()


def test_real_learnings_pass(real: str, names: dict[str, int]) -> None:
    assert len(lessons(real)) >= 20
    assert problems(real, ROOT, names) == []


def _drop_enforced(text: str, number: int) -> str:
    section = _section_line(text, number)
    lines = text.splitlines()
    del lines[section]
    return '\n'.join(lines) + '\n'


def _double_enforced(text: str, number: int) -> str:
    section = _section_line(text, number)
    lines = text.splitlines()
    lines.insert(section + 1, lines[section])
    return '\n'.join(lines) + '\n'


def _name_missing_tool(text: str, number: int) -> str:
    section = _section_line(text, number)
    lines = text.splitlines()
    lines[section] += ' `scripts/rust_core/no_such_tool.py`'
    return '\n'.join(lines) + '\n'


def _repeat_number(text: str, number: int) -> str:
    return text.replace(f'\n### {number + 1}. ', f'\n### {number}. ', 1)


def _four_wording_only(text: str, number: int) -> str:
    r"""Mark headings until exactly four say "wording only"."""
    lines = text.splitlines()
    marked = sum(
        1 for line in lines if LESSON.match(line) and WORDING_ONLY in line
    )
    for i, line in enumerate(lines):
        if marked == MAX_WORDING_ONLY + 1:
            break
        if LESSON.match(line) and WORDING_ONLY not in line:
            lines[i] = f'{line} (wording only)'
            marked += 1
    return '\n'.join(lines) + '\n'


def _section_line(text: str, number: int) -> int:
    r"""The line index of lesson ``number``'s Enforced-by line."""
    lines = text.splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith(f'### {number}. ')
    )
    return next(
        i for i in range(start + 1, len(lines)) if lines[i].startswith(ENFORCED)
    )


SEEDS: dict[str, tuple[Callable[[str, int], str], str]] = {
    'enforced-line-deleted': (_drop_enforced, 'lesson 3: 0 Enforced-by lines'),
    'two-enforced-lines': (_double_enforced, 'lesson 3: 2 Enforced-by lines'),
    'missing-tool': (
        _name_missing_tool,
        'lesson 3: missing path scripts/rust_core/no_such_tool.py',
    ),
    'repeated-number': (_repeat_number, 'lesson numbers are'),
    'four-wording-only': (_four_wording_only, '4 lessons are wording only'),
}


@pytest.mark.parametrize('seed', sorted(SEEDS))
def test_seeded_failure(real: str, names: dict[str, int], seed: str) -> None:
    change, expected = SEEDS[seed]
    seeded = change(real, 3)
    assert seeded != real
    before = problems(real, ROOT, names)
    after = problems(seeded, ROOT, names)
    new = [problem for problem in after if problem not in before]
    assert any(expected in problem for problem in new), new
