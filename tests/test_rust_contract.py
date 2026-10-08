# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Rust core's contract, read from Python with the standard library.

The scalar edge values the contract encodes decode exactly with only ``json``
and ``decimal``: nothing passes through a double. And the contract lock check
fails a rewritten or removed version entry and a changed schema.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from decimal import Decimal
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
EDGES = ROOT / 'rust/crates/sdm-contract/tests/scalar_edges.json'


def _edges() -> dict[str, dict[str, object]]:
    document = json.loads(EDGES.read_text(encoding='utf-8'))
    assert document['format'] == 'sdm-scalar-edges/1'
    edges: dict[str, dict[str, object]] = document['edges']
    return edges


def _int(scalar: dict[str, object]) -> int:
    text = scalar['v']
    assert isinstance(text, str)
    return int(text)


def _decimal(scalar: dict[str, object]) -> Decimal:
    text, precision, scale = scalar['v'], scalar['p'], scalar['s']
    assert isinstance(text, str) and isinstance(scale, int)
    assert isinstance(precision, int)
    assert len(text.lstrip('-')) <= precision
    # Built from its digits, not by scaleb, which rounds to the context's 28
    # digits.
    unscaled = Decimal(text).as_tuple()
    return Decimal((unscaled.sign, unscaled.digits, -scale))


def _float_hex(scalar: dict[str, object]) -> str:
    text = scalar['v']
    assert isinstance(text, str)
    return float(text).hex()


def test_integer_edges_decode_exactly() -> None:
    edges = _edges()
    assert _int(edges['i8_min']) == -128
    assert _int(edges['i64_min']) == -9223372036854775808
    assert _int(edges['i64_max']) == 9223372036854775807
    assert _int(edges['u64_max']) == 18446744073709551615
    assert _int(edges['u64_above_i64_max']) == 9223372036854775809
    assert _int(edges['date32_min']) == -2147483648
    assert _int(edges['duration_min']) == -9223372036854775808


def test_decimal_edges_decode_exactly() -> None:
    edges = _edges()
    # Literals only: arithmetic on a Decimal rounds to the context's digits.
    assert _decimal(edges['d128_38_0_max']) == Decimal('9' * 38)
    assert _decimal(edges['d128_38_0_min']) == Decimal('-' + '9' * 38)
    assert _decimal(edges['d128_38_10']) == Decimal(
        '-1234567890123456.7890123456'
    )
    assert _decimal(edges['d256_76_0_max']) == Decimal('9' * 76)
    assert _decimal(edges['d256_76_0_min']) == Decimal('-' + '9' * 76)
    assert edges['d256_76_0_max']['p'] == 76
    assert _decimal(edges['d256_76_38']) == Decimal('1' + '0' * 37)


def test_float_edges_decode_exactly() -> None:
    edges = _edges()
    assert _float_hex(edges['f64_neg_zero']) == '-0x0.0p+0'
    assert _float_hex(edges['f64_zero']) == '0x0.0p+0'
    assert _float_hex(edges['f64_tenth']) == (0.1).hex()
    assert _float_hex(edges['f64_max']) == '0x1.fffffffffffffp+1023'
    assert _float_hex(edges['f64_min_subnormal']) == '0x0.0000000000001p-1022'
    assert _float_hex(edges['f64_inf']) == 'inf'
    assert _float_hex(edges['f64_neg_inf']) == '-inf'
    nan = float(str(edges['f64_nan']['v']))
    assert nan != nan
    # The texts Rust writes are ones Decimal reads too, sign of zero kept.
    assert str(Decimal(str(edges['f64_neg_zero']['v']))) == '-0.0'


def test_temporal_edges_decode_exactly() -> None:
    edges = _edges()
    low, high = edges['ts_i64_min'], edges['ts_i64_max']
    assert (low['unit'], low['zone'], _int(low)) == (
        'ns',
        None,
        -9223372036854775808,
    )
    assert (high['unit'], high['zone'], _int(high)) == (
        'us',
        'America/New_York',
        9223372036854775807,
    )
    assert _int(edges['time64_last_ns']) == 86399999999999


def test_no_edge_value_is_a_json_number() -> None:
    for name, scalar in _edges().items():
        assert not isinstance(scalar.get('v'), (int, float)) or isinstance(
            scalar.get('v'), bool
        ), name


def _lock_module() -> ModuleType:
    path = ROOT / 'scripts/rust_core/contract_lock.py'
    spec = importlib.util.spec_from_file_location('contract_lock', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def test_the_lock_check_holds_on_the_committed_lock() -> None:
    # The checkout here may be shallow; rust-lint checks the merge base.
    assert _lock_module().main(['--no-base']) == 0


def test_the_lock_check_fails_without_its_base() -> None:
    assert _lock_module().main(['--base', 'no-such-branch-for-the-lock']) == 1


def test_the_lock_check_fails_an_unreadable_earlier_lock() -> None:
    lock = _lock_module()
    one = b'{"v": 1}\n'
    entry = f'1 sha256:{_digest(one)} sdm-contract-1.json\n'
    problems = lock.check(
        entry,
        [('d' * 40, 'not a lock line\n')],
        {'sdm-contract-1.json': one}.get,
    )
    assert any('cannot be read' in p for p in problems), problems


def test_the_lock_check_fails_a_rewritten_or_removed_entry() -> None:
    lock = _lock_module()
    one, two = b'{"v": 1}\n', b'{"v": 2}\n'
    schemas = {'sdm-contract-1.json': one, 'sdm-contract-2.json': two}
    first = f'1 sha256:{_digest(one)} sdm-contract-1.json\n'
    both = first + f'2 sha256:{_digest(two)} sdm-contract-2.json\n'

    assert lock.check(both, [('a' * 40, first)], schemas.get) == []

    # Version 1's schema changed in place, its entry rewritten to match.
    changed = b'{"v": 1, "x": 0}\n'
    rewritten = f'1 sha256:{_digest(changed)} sdm-contract-1.json\n'
    problems = lock.check(
        rewritten, [('b' * 40, first)], {'sdm-contract-1.json': changed}.get
    )
    assert any('was rewritten' in p for p in problems), problems

    # An entry an earlier commit held is gone.
    problems = lock.check(
        f'2 sha256:{_digest(two)} sdm-contract-2.json\n',
        [('c' * 40, both)],
        schemas.get,
    )
    assert any('was removed' in p for p in problems), problems


def test_the_lock_check_fails_a_schema_changed_without_a_bump() -> None:
    lock = _lock_module()
    one = b'{"v": 1}\n'
    entry = f'1 sha256:{_digest(one)} sdm-contract-1.json\n'
    problems = lock.check(
        entry, [], {'sdm-contract-1.json': b'{"v": 1, "y": 0}\n'}.get
    )
    assert any('needs a new version' in p for p in problems)
    assert lock.check(entry, [], {}.get) == [
        'version 1: sdm-contract-1.json is missing'
    ]
