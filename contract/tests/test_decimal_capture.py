# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact decimal capture at the Python observation boundary."""

import decimal
import importlib.util
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

WIDE = [
    '12345678901234567890123456789012345678',
    '-99999999999999999999999999999999999999',
    '1234567890123456789012345678.0123456789',
    '-1.2345678901234567890123456789012345678',
    '0.00000000000000000000000000000000000001',
    '1' * 76,
    '-' + '9' * 40 + '.' + '1' * 36,
]


def _digits(text: str, scale: int) -> str:
    sign = '-' if text.startswith('-') else ''
    whole, _, fraction = text.lstrip('-').partition('.')
    digits = (whole + fraction.ljust(scale, '0')).lstrip('0') or '0'
    return sign + digits if digits != '0' else '0'


def _hostile() -> Any:
    context = decimal.Context(prec=3, Emax=5, Emin=-5)
    context.traps[decimal.Inexact] = True
    context.traps[decimal.Rounded] = True
    return decimal.localcontext(context)


def _wide(write: Any) -> list[str]:
    r"""What ``write`` (a value to its scalar) gets wrong of ``WIDE`` in a
    hostile decimal context: precision 3, rounding trapped."""
    import pyarrow as pa

    wrong = []
    for text in WIDE:
        value = Decimal(text)
        kind = pa.array([value]).type
        want = {
            't': 'd128' if pa.types.is_decimal128(kind) else 'd256',
            'p': kind.precision,
            's': kind.scale,
            'v': _digits(text, kind.scale),
        }
        try:
            with _hostile():
                got = write(value)
        except Exception as error:  # noqa: BLE001 - reported
            got = f'raised {type(error).__name__}'
        if got != want:
            wrong.append(f'{text}: {got}, not {want}')
    return wrong


def test_the_runner_writes_wide_decimals_exactly() -> None:
    # The Python runner's record of the entities the host sent
    # (run_python.py's _scalars), held to the same exact digits.
    import pyarrow as pa

    run_python = _module('run_python', ROOT / 'contract/runners/run_python.py')

    def write(value: Decimal) -> Any:
        return run_python._scalars(pa.table({'indices': pa.array([value])}))[0]

    def check() -> None:
        assert _wide(write) == []

    check()


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
