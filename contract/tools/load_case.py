#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Loads one case's data into a new SQLite file, for the pure-Rust host.

    python -I contract/tools/load_case.py sqlite CASE.json OUT.sqlite

The case's DuckDB base and setup are built as ``run_python.py`` builds them
(``contract/bases.yaml``), and its tables loaded by the one contract loader
(``contract/runners/databases.py``, T2.1a), so ``sdm-contract-run
--database sqlite`` reads the same loaded database the reference and
python-core read (design choice 2). A case whose data cannot be built
exits 2; one the loader refuses by name (``Unloadable``) exits 3, its name
on standard error; OUT must not exist.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

RUNNERS = Path(__file__).resolve().parents[1] / 'runners'


def _sibling(name: str) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, RUNNERS / f'{name}.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[0] != 'sqlite':
        print(
            'usage: load_case.py sqlite CASE.json OUT.sqlite', file=sys.stderr
        )
        return 64
    runner = _sibling('run_python')
    loader = _sibling('databases')
    case = json.loads(Path(argv[1]).read_text(encoding='utf-8'))
    bases = runner.load_bases(runner.BASES)
    try:
        duck = runner._connection(case, bases)
    except runner.RunnerFailure as failure:
        print(failure, file=sys.stderr)
        return 2
    try:
        loader.load(duck, 'sqlite', Path(argv[2]))
    except loader.Unloadable as refused:
        print(f'unloadable: {refused}', file=sys.stderr)
        return 3
    finally:
        duck.close()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
