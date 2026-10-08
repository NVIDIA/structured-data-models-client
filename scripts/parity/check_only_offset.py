# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Check that a re-recorded corpus differs from the approved one only as the
time-zone fix and the PostgreSQL scan cap make it.

    python scripts/parity/check_only_offset.py OLD NEW [BACKEND ...]

``OLD`` and ``NEW`` are transcript directories, such as an approved
``tests/parity/transcripts`` extracted from git and this checkout's. Every
transcript of ``NEW`` that ``OLD`` also holds, of the backends named (all
when none is), must differ from the old one only:

* by ``+00:00`` on bound-shaped strings (``YYYY-MM-DDTHH:MM:SS[.f]``) in a
  step's parameters, inside JSON text too;
* by size-probe steps (an ``execute`` of SQL that reads ``pg_relation_size``
  or ``relpages``, and no inference sample) inserted right before an
  inference sample.

Everything else must be equal: the other keys, every other step, and each
step's result and the decoded answer, whose Arrow files are compared by
content, so renumbered step files still match. A scenario only ``NEW`` holds
is not checked. It prints one line per transcript and exits 0 when every one
passes, 1 when one doesn't, and 2 when it can't read what it was given.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

#: A time bound as the engine writes it, without its offset.
_BOUND = re.compile(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?$')
_OFFSET = '+00:00'
#: What a size probe reads.
_PROBE = re.compile(r'pg_relation_size|relpages')
_SAMPLE = '__kumo_order_'


class Mismatch(Exception):
    r"""How a transcript differs beyond what the fixes allow."""


def strip_offsets(value: Any, count: list[int]) -> Any:
    r"""``value`` with ``+00:00`` taken off every bound-shaped string, inside
    JSON text too; ``count[0]`` counts the strings changed."""
    if isinstance(value, str):
        if value.endswith(_OFFSET) and _BOUND.match(value[: -len(_OFFSET)]):
            count[0] += 1
            return value[: -len(_OFFSET)]
        try:
            inner = json.loads(value)
        except ValueError:
            return value
        if isinstance(inner, (list, dict)):
            before = count[0]
            stripped = strip_offsets(inner, count)
            if count[0] != before:
                return stripped
        return value
    if isinstance(value, list):
        return [strip_offsets(item, count) for item in value]
    if isinstance(value, dict):
        return {key: strip_offsets(item, count) for key, item in value.items()}
    return value


def _parsed(value: Any) -> Any:
    r"""``value`` with JSON text holding an array or object parsed."""
    if isinstance(value, str):
        try:
            inner = json.loads(value)
        except ValueError:
            return value
        return _parsed(inner) if isinstance(inner, (list, dict)) else value
    if isinstance(value, (list, tuple)):
        return [_parsed(item) for item in value]
    if isinstance(value, dict):
        return {key: _parsed(item) for key, item in value.items()}
    return value


def _same_params(new: Any, old: Any) -> bool:
    r"""Parameters equal, JSON text compared by the value it holds."""
    return _parsed(new) == _parsed(old)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same_result(
    new_dir: Path, new_ref: Any, old_dir: Path, old_ref: Any, what: str
) -> None:
    if (new_ref is None) != (old_ref is None):
        raise Mismatch(f'{what}: one recording holds a result, one does not')
    if new_ref is None:
        return
    if _digest(new_dir / new_ref) != _digest(old_dir / old_ref):
        raise Mismatch(f'{what}: the result differs ({new_ref} vs {old_ref})')


def _is_probe(step: dict[str, Any]) -> bool:
    sql = step.get('sql') or ''
    return (
        step.get('op') == 'execute'
        and step.get('error') is None
        and bool(_PROBE.search(sql))
        and _SAMPLE not in sql
    )


def _step_matches(
    new: dict[str, Any], old: dict[str, Any], count: list[int]
) -> bool:
    keep = ('params', 'result')
    if {k: v for k, v in new.items() if k not in keep} != {
        k: v for k, v in old.items() if k not in keep
    }:
        return False
    changed = [0]
    stripped = strip_offsets(new.get('params'), changed)
    if not _same_params(stripped, old.get('params')):
        return False
    count[0] += changed[0]
    return True


def compare(old_dir: Path, new_dir: Path) -> tuple[int, int]:
    r"""Check one transcript against its approved recording; the bounds that
    gained ``+00:00`` and the probes inserted.

    Raises:
        Mismatch: It differs otherwise.
    """
    old = json.loads((old_dir / 'transcript.json').read_text())
    new = json.loads((new_dir / 'transcript.json').read_text())
    for key in sorted(set(old) | set(new)):
        if key in ('steps', 'result'):
            continue
        if old.get(key) != new.get(key):
            raise Mismatch(f'{key!r} differs')
    _same_result(
        new_dir, new.get('result'), old_dir, old.get('result'), 'result'
    )

    count = [0]
    probes = 0
    position = 0
    old_steps, new_steps = old.get('steps', []), new.get('steps', [])
    for index, step in enumerate(new_steps):
        if position < len(old_steps) and _step_matches(
            step, old_steps[position], count
        ):
            _same_result(
                new_dir,
                step.get('result'),
                old_dir,
                old_steps[position].get('result'),
                f'step {index}',
            )
            position += 1
            continue
        following = new_steps[index + 1] if index + 1 < len(new_steps) else {}
        if not _is_probe(step):
            raise Mismatch(
                f'step {index} is neither recorded step {position} with '
                f'+00:00 on its bounds nor a size probe: '
                f'{" ".join(str(step.get("sql")).split())[:200]}'
            )
        if _SAMPLE not in (following.get('sql') or ''):
            raise Mismatch(
                f'step {index} is a size probe not right before an inference '
                'sample'
            )
        probes += 1
    if position != len(old_steps):
        raise Mismatch(f'recorded steps from {position} on are missing')
    return count[0], probes


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    old_root, new_root = Path(argv[1]), Path(argv[2])
    backends = set(argv[3:])
    if not old_root.is_dir() or not new_root.is_dir():
        print(f'not a directory: {old_root} or {new_root}', file=sys.stderr)
        return 2
    failed = 0
    checked = 0
    for path in sorted(new_root.rglob('transcript.json')):
        relative = path.parent.relative_to(new_root)
        if backends and relative.parts[0] not in backends:
            continue
        old_dir = old_root / relative
        if not (old_dir / 'transcript.json').is_file():
            continue
        checked += 1
        try:
            bounds, probes = compare(old_dir, path.parent)
        except Mismatch as error:
            print(f'FAIL {relative}: {error}')
            failed += 1
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f'FAIL {relative}: cannot compare: {error!r}')
            failed += 1
        else:
            print(
                f'PASS {relative}: {bounds} bound(s) +00:00, {probes} probe(s)'
            )
    print(f'{checked - failed} passed, {failed} failed', file=sys.stderr)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
