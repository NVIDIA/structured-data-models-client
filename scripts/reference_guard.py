#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refuse to run as the reference if the Rust core can be loaded.

The ``rust-core-reference`` branch is today's Python with only the
determinism fixes, and other environments compare their answers with it.
That comparison means nothing if the reference itself reached the Rust core,
so this exits non-zero when any module the core would ship is already
imported, or could be imported from this interpreter's path.

Run it at the start of every reference session, with the reference's own
interpreter:

    .venv/bin/python scripts/reference_guard.py

A harness that calls the reference in-process should also call
``require_isolation()`` in that same interpreter, so that the check covers
what the harness itself has imported.

``--self-test`` runs the guard against a clean path and against two
negative controls, each in a fresh interpreter, and exits non-zero unless
the clean run passes and both controls fail.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Top-level names the Rust core's Python bindings may take.
_TOP_LEVEL = ('sdm_core', 'sdm_py', '_native', 'sdm_core_py')

# The five import roots, each of which could grow a native submodule.
_IMPORT_ROOTS = (
    'sdm_api',
    'sdm_client',
    'sdm_connectors',
    'sdm_relational_engine',
    'predictive_query',
)
_SUBMODULES = ('_native', '_core', 'sdm_core', 'sdm_py')


def _candidates() -> list[str]:
    names = list(_TOP_LEVEL)
    for root in _IMPORT_ROOTS:
        names.extend(f'{root}.{sub}' for sub in _SUBMODULES)
    return names


def _importable(name: str) -> bool:
    parent, _, _ = name.rpartition('.')
    try:
        if parent and importlib.util.find_spec(parent) is None:
            return False
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def problems() -> list[str]:
    r"""Every Rust core module that is imported or importable here.

    Probing a package-qualified name imports its parent package, and that
    package's initialisation can load a module whose name was already
    checked.  So once every probe has run, the whole candidate set is
    checked against ``sys.modules`` again.
    """
    found = []
    for name in _candidates():
        if name in sys.modules:
            found.append(f'{name} is imported')
        elif _importable(name):
            spec = importlib.util.find_spec(name)
            origin = spec.origin if spec is not None else None
            found.append(f'{name} is importable from {origin}')
    reported = {line.split(' ', 1)[0] for line in found}
    for name in _candidates():
        if name in sys.modules and name not in reported:
            found.append(f'{name} was imported while the guard ran')
    return found


def require_isolation() -> None:
    r"""Raise ``RuntimeError`` if this interpreter can load the Rust core."""
    found = problems()
    if found:
        raise RuntimeError(
            'the reference environment can load the Rust core: '
            + '; '.join(found)
        )


def main() -> int:
    found = problems()
    if found:
        print('the reference environment can load the Rust core:')
        for line in found:
            print(f'  {line}')
        return 1
    print(f'reference guard: none of {len(_candidates())} names load')
    return 0


# Each control: (name, files to write on PYTHONPATH, expected exit status).
_CONTROLS = (
    ('clean run', {}, 0),
    ('importable stub', {'sdm_core.py': ''}, 1),
    (
        'parent import',
        {
            'sdm_api/__init__.py': (
                'import sys, types\n'
                "sys.modules['sdm_core'] = types.ModuleType('sdm_core')\n"
            ),
        },
        1,
    ),
)


def self_test() -> int:
    r"""Run the guard on a clean path and on two negative controls."""
    failures = 0
    for name, files, expected in _CONTROLS:
        with tempfile.TemporaryDirectory() as tmp:
            for rel, text in files.items():
                path = Path(tmp, rel)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            env = dict(os.environ)
            env['PYTHONPATH'] = os.pathsep.join(
                p for p in (tmp, env.get('PYTHONPATH', '')) if p
            )
            run = subprocess.run(
                [sys.executable, __file__],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        ok = run.returncode == expected
        failures += not ok
        status = 'ok' if ok else 'FAILED'
        print(f'{status}: {name}: exit {run.returncode}, expected {expected}')
        for line in run.stdout.splitlines():
            print(f'    {line}')
    return 1 if failures else 0


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        raise SystemExit(self_test())
    raise SystemExit(main())
