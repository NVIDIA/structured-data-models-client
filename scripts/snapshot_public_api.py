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
import enum
import importlib
import inspect
import json
import pathlib
import sys
from typing import Any

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


FIRST_PARTY_ROOTS = frozenset(
    {
        'sdm_api',
        'sdm_client',
        'sdm_connectors',
        'predictive_query',
        'sdm_relational_engine',
    }
)


def _first_party_attributes(owner: type) -> dict[str, Any]:
    """Every public attribute this project defines on ``owner`` or its bases.

    Read from ``vars()`` along the MRO rather than through
    :func:`inspect.getmembers`, which does not surface the methods defined on
    an ``Enum`` subclass at all -- ``Dtype.is_int`` and the fourteen helpers
    beside it were recorded nowhere while appearing to be covered.

    Reading the class dictionary also keeps inherited builtins out without a
    second filter: ``str.encode`` is in ``vars(str)``, and ``str`` is not one
    of ours.
    """
    found: dict[str, Any] = {}
    for base in reversed(owner.__mro__):
        module = (getattr(base, '__module__', '') or '').split('.')[0]
        if module not in FIRST_PARTY_ROOTS:
            continue
        for name, value in vars(base).items():
            if not name.startswith('_'):
                found[name] = value
    return found


#: Types whose *value* is part of the contract, not just their presence.
#: `__version__` was recorded as `{'kind': 'str'}`, so 1.0.0 becoming anything
#: else moved nothing -- a snapshot of the public surface that could not see
#: the version number in it.
RECORDED_BY_VALUE = (str, int, float, bool, bytes, type(None))


def _describe(obj: object) -> dict[str, object]:
    described: dict[str, object] = {'kind': type(obj).__name__}
    if isinstance(obj, RECORDED_BY_VALUE):
        described['value'] = obj if not isinstance(obj, bytes) else obj.hex()
        return described
    if isinstance(obj, (tuple, frozenset)) and all(
        isinstance(item, RECORDED_BY_VALUE) for item in obj
    ):
        described['value'] = sorted(obj, key=repr)
        return described
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
        #
        # Inherited builtins are excluded. A `StrEnum` otherwise contributes
        # 47 `str` methods, which buries the members that actually matter --
        # and records signatures owned by CPython, so the snapshot would
        # disagree with itself on a different interpreter for reasons that have
        # nothing to do with this library.
        owned = _first_party_attributes(obj)
        described['public_methods'] = {
            name: _signature(getattr(obj, name, value))
            for name, value in sorted(owned.items())
            if callable(value) or isinstance(value, (classmethod, staticmethod))
        }
        if issubclass(obj, enum.Enum):
            # For an enum the members *are* the contract; the methods are not.
            described['members'] = {member.name: member.value for member in obj}
        # Attributes a caller reads are part of the surface too, and a
        # dataclass field has no entry above.
        described['public_attributes'] = sorted(
            name
            for name, value in owned.items()
            if not callable(value)
            and not isinstance(value, (classmethod, staticmethod))
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
