# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.1's tests for ``scripts/rust_core/brief_check.py`` (the separate
test author's), run as the orchestrator runs it: a subprocess.

The contract, from the script's docstring: the required headings are the
``## `` lines inside the fenced block of
``docs/design/rust-core/orchestration/briefs/package.md`` (``--template``
names another); a brief passes when it holds each as a whole line outside
any fenced block; exit 0 when every brief passes (``<brief>: ok``), 1
naming each missing heading, 2 when the template or a brief cannot be read
or the template holds no heading inside a fenced block.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/rust_core/brief_check.py'
TEMPLATE = ROOT / 'docs/design/rust-core/orchestration/briefs/package.md'

EXPECTED = [
    'Scope and exits',
    'Threat model',
    'Reference code to port',
    'Host or core',
    'Input axes',
    'When you change X, also update Y',
    'Readers of new or changed files',
    'Judging paths',
    'Owner decisions',
    'Owner questions',
]


def _run(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=ROOT,
    )


def _fenced_block(text: str) -> list[str]:
    r"""The lines of the first fenced block of ``text``."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('```'))
    end = next(
        i for i in range(start + 1, len(lines)) if lines[i].startswith('```')
    )
    return lines[start + 1 : end]


def _brief(headings: list[str], body: str = 'An answer.') -> str:
    parts = ['# T9.9: a package']
    for heading in headings:
        parts += ['', f'## {heading}', body]
    return '\n'.join(parts) + '\n'


def test_real_template_yields_the_ten_headings_in_order(tmp_path: Path) -> None:
    block = _fenced_block(TEMPLATE.read_text(encoding='utf-8'))
    headings = [line[3:] for line in block if line.startswith('## ')]
    assert headings == EXPECTED
    brief = tmp_path / 'brief.md'
    brief.write_text('\n'.join(block) + '\n', encoding='utf-8')
    result = _run(brief)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == f'{brief}: ok'
    # The script reads the same ten from the real template: dropping any
    # one of them fails the brief and names it.
    for heading in EXPECTED:
        lacking = tmp_path / 'lacking.md'
        lacking.write_text(
            _brief([h for h in EXPECTED if h != heading]), encoding='utf-8'
        )
        result = _run(lacking)
        assert result.returncode == 1, heading
        assert heading in result.stdout, heading


def test_missing_heading_is_named_alone(tmp_path: Path) -> None:
    brief = tmp_path / 'brief.md'
    brief.write_text(
        _brief([h for h in EXPECTED if h != 'Host or core']), encoding='utf-8'
    )
    result = _run(brief)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'Host or core' in result.stdout
    for other in EXPECTED:
        if other != 'Host or core':
            assert other not in result.stdout, other


def test_heading_only_inside_a_fence_does_not_count(tmp_path: Path) -> None:
    others = [h for h in EXPECTED if h != 'Threat model']
    text = _brief(others) + '\n```markdown\n## Threat model\nHidden.\n```\n'
    brief = tmp_path / 'brief.md'
    brief.write_text(text, encoding='utf-8')
    result = _run(brief)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'Threat model' in result.stdout


def test_heading_answered_none_with_a_reason_passes(tmp_path: Path) -> None:
    text = _brief(EXPECTED).replace(
        '## Owner questions\nAn answer.',
        '## Owner questions\nnone: every decision this package needs is'
        ' already a row the owner approved.',
    )
    assert 'none: every decision' in text
    brief = tmp_path / 'brief.md'
    brief.write_text(text, encoding='utf-8')
    result = _run(brief)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == f'{brief}: ok'


def test_missing_brief_exits_2(tmp_path: Path) -> None:
    result = _run(tmp_path / 'no-such-brief.md')
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'no-such-brief.md' in result.stderr


def test_template_without_fenced_headings_exits_2(tmp_path: Path) -> None:
    template = tmp_path / 'template.md'
    # Headings outside any fence, and a fence with no heading in it.
    template.write_text(
        '# Template\n\n## Scope and exits\n\n```markdown\nno heading\n```\n',
        encoding='utf-8',
    )
    brief = tmp_path / 'brief.md'
    brief.write_text(_brief(EXPECTED), encoding='utf-8')
    result = _run(brief, '--template', template)
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'template.md' in result.stderr


def test_one_good_one_bad_brief(tmp_path: Path) -> None:
    good = tmp_path / 'good.md'
    good.write_text(_brief(EXPECTED), encoding='utf-8')
    bad = tmp_path / 'bad.md'
    bad.write_text(
        _brief([h for h in EXPECTED if h != 'Judging paths']), encoding='utf-8'
    )
    result = _run(good, bad)
    assert result.returncode == 1, result.stdout + result.stderr
    lines = result.stdout.splitlines()
    assert f'{good}: ok' in lines
    assert any(
        line.startswith(f'{bad}:') and 'Judging paths' in line for line in lines
    )
    assert not any(
        line.startswith(f'{good}:') and 'missing' in line for line in lines
    )
