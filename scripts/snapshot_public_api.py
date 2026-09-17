#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the public surface of every distribution, so a change to it is visible.

What a user can observe is more than the code that produces it: the names a
package exports, the signature of each one, the exception classes and the codes
they carry, and the warning categories a caller can filter on. None of that is
covered by a test that only asserts a prediction comes back, and all of it is
what breaks a downstream caller.

So it is written to ``tests/golden/public_api.json`` and compared on every run.
A diff is not a failure to be silenced -- it is the question "did you mean to
change the contract?", asked at the moment the answer is still cheap.

Run with ``--write`` to re-record after an intended change.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pathlib
import sys

SNAPSHOT = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'tests'
    / 'golden'
    / 'public_api.json'
)

# Only the distributions a user imports directly. The relational engine is
# behind an extra and is recorded when it is installed, never required.
PACKAGES = ('sdm_api', 'sdm_client', 'sdm_connectors', 'predictive_query')


def _signature(obj: object) -> str | None:
    try:
        return str(inspect.signature(obj))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _describe(obj: object) -> dict[str, object]:
    described: dict[str, object] = {'kind': type(obj).__name__}
    if inspect.isclass(obj):
        described['kind'] = 'class'
        described['bases'] = [
            f'{base.__module__}.{base.__qualname__}'
            for base in obj.__bases__
            if base is not object
        ]
        described['signature'] = _signature(obj)
        # Signatures, not just names. Recording only the names let
        # `SDMClient.predict` change shape without the snapshot noticing --
        # which is the method most callers touch first.
        described['public_methods'] = {
            name: _signature(member)
            for name, member in inspect.getmembers(obj, callable)
            if not name.startswith('_')
        }
        # Attributes a caller reads are part of the surface too, and a
        # dataclass field has no entry above.
        described['public_attributes'] = sorted(
            name
            for name, member in inspect.getmembers(obj)
            if not name.startswith('_') and not callable(member)
        )
    elif callable(obj):
        described['kind'] = 'callable'
        described['signature'] = _signature(obj)
    return described


def collect() -> dict[str, object]:
    surface: dict[str, object] = {}
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
        except ImportError:  # pragma: no cover - an extra is not installed
            continue
        exported = getattr(module, '__all__', None)
        if exported is None:
            surface[name] = {'__all__': None}
            continue
        surface[name] = {
            '__all__': sorted(exported),
            'members': {
                member: _describe(getattr(module, member))
                for member in sorted(exported)
                if hasattr(module, member)
            },
        }
    return surface


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--write',
        action='store_true',
        help='re-record the snapshot after an intended contract change',
    )
    args = parser.parse_args()

    current = collect()
    if args.write:
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(
            json.dumps(current, indent=2, sort_keys=True) + '\n'
        )
        print(f'wrote {SNAPSHOT}')
        return 0

    if not SNAPSHOT.exists():
        print(f'no snapshot at {SNAPSHOT}; run with --write', file=sys.stderr)
        return 1

    recorded = json.loads(SNAPSHOT.read_text())
    if recorded == current:
        print('public API matches the recorded snapshot')
        return 0

    print('public API differs from the recorded snapshot', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
