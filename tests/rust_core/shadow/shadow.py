# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The shadow comparison, to read rather than to gate: every transcript and
every public fixture case through the old path and the core, each in a fresh
process, and every scenario whose answer differs.

    python tests/rust_core/shadow/shadow.py [--what transcripts|fixtures]
                                            [--json FILE]

A capture of one path is ``tests/rust_core/switch/_switch.py capture``, run
with ``SDM_CLIENT_IMPLEMENTATION`` set for that path: payloads with the
request id normalized, the statements and ``Source._run`` calls, the
flattened result, the warnings and where they landed, and the error's class,
code, message, details and chain. A scenario ``fix_changes.json`` lists is
said as changed by its fix, and must differ; any other difference is a
defect of the switch. Exits 0 only when nothing unlisted differs and every
listed scenario does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SWITCH = HERE.parent / 'switch'
if str(SWITCH) not in sys.path:
    sys.path.insert(0, str(SWITCH))

from _switch import (  # noqa: E402
    CORE,
    OLD,
    by_key,
    capture,
    core_in_use,
    fix_changes,
)

#: What is compared, per kind of scenario, as ``accept_shadow.py`` compares.
KEYS = {
    'transcripts': (
        'name',
        (
            'check',
            'crashed',
            'divergence',
            'statements',
            'source_runs',
            'payloads',
            'error',
            'error_chain',
            'answer',
            'warnings',
            'result',
        ),
    ),
    'fixtures': (
        'id',
        (
            'output',
            'error',
            'error_chain',
            'crashed',
            'warnings',
            'source_runs',
        ),
    ),
}


def _plain(record: dict[str, Any]) -> dict[str, Any]:
    r"""A record with each warning's category and text, in order (where it
    landed is ``accept_warnings.py``'s to judge), and its error's chain as
    the user's traceback shows it."""
    out = dict(record)
    # Ported packages' classes are left out, as accept_errors.py leaves them.
    out['error_chain'] = [
        c
        for c in record.get('error_chain') or []
        if not c.startswith(('predictive_query', 'sdm_relational_engine'))
    ]
    out['warnings'] = [
        [w.get('category'), w.get('message')]
        for w in record.get('warnings') or []
    ]
    return out


def _first_difference(a: Any, b: Any, path: str) -> str | None:
    if type(a) is not type(b):
        return f'{path}: {json.dumps(a, default=repr)[:200]} became {json.dumps(b, default=repr)[:200]}'
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f'{path}.{key}: only on the {"core" if key in b else "old path"}'
            found = _first_difference(a[key], b[key], f'{path}.{key}')
            if found:
                return found
        return None
    if isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            found = _first_difference(x, y, f'{path}[{i}]')
            if found:
                return found
        if len(a) != len(b):
            return f'{path}: {len(a)} items became {len(b)}'
        return None
    if a != b:
        return f'{path}: {json.dumps(a, default=repr)[:200]} became {json.dumps(b, default=repr)[:200]}'
    return None


def compare(what: str) -> dict[str, Any]:
    r"""Each scenario of ``what`` that differs, and why, beside the listed
    fixes, and the reasons the core capture did not run on the core."""
    key, keys = KEYS[what]
    old = by_key(capture(OLD, what), key)
    core_data = capture(CORE, what)
    core = by_key(core_data, key)
    listed = fix_changes()
    unlisted: dict[str, str] = {}
    fixed: dict[str, str] = {}
    unchanged_fixes: list[str] = []
    for name in sorted(set(old) | set(core)):
        if name not in old or name not in core:
            unlisted[name] = 'run on one path only'
            continue
        a, b = _plain(old[name]), _plain(core[name])
        found = next(
            (
                d
                for d in (
                    _first_difference(a.get(k), b.get(k), k) for k in keys
                )
                if d
            ),
            None,
        )
        if name in listed:
            if found:
                fixed[name] = f'{listed[name]["fix"]}: {found}'
            else:
                unchanged_fixes.append(name)
        elif found:
            unlisted[name] = found
    return {
        'compared': len(old),
        'unlisted': unlisted,
        'fixed': fixed,
        'unchanged_fixes': unchanged_fixes,
        'not_on_the_core': core_in_use(core_data),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--what', choices=sorted(KEYS), action='append')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args(argv)
    found = {what: compare(what) for what in args.what or sorted(KEYS)}
    failed = False
    for what, report in found.items():
        print(
            f'{what}: {report["compared"]} compared, {len(report["fixed"])} '
            f'changed by a listed fix, {len(report["unlisted"])} differ otherwise'
        )
        for name, why in report['unlisted'].items():
            print(f'  DIFFERS {name}: {why}')
        for name in report['unchanged_fixes']:
            print(f'  LISTED BUT SAME {name}')
        for problem in report['not_on_the_core']:
            print(f'  NOT ON THE CORE {problem.splitlines()[0]}')
        failed |= bool(report['unlisted'] or report['unchanged_fixes'])
    if args.json:
        args.json.write_text(json.dumps(found, indent=1))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
