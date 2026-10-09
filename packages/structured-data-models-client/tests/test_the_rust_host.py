# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Python host of the Rust core: its codec, its facts and its error table.

The codec reads every contract scalar exactly and a message the core sent
whole as an attachment (decisions E4, H7, E23). The host facts name the
zone lookup roots in ``zoneinfo``'s order with their versions, and the
platform's year padding (E23). Every code the core has maps to one
``StructuredDataError`` class, keyed on the code alone and reading no
message text (H6, E22); warnings keep their order and their class.
"""

from __future__ import annotations

import ast
import json
import warnings
import zoneinfo
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from sdm_client import _native
from sdm_client._rust import codec, errors, facts
from sdm_client._rust.run import run
from sdm_client.core.warnings import SDMWarning
from sdm_client.errors import StructuredDataError

ROOT = Path(__file__).resolve().parents[3]
EDGES = ROOT / 'rust/crates/sdm-contract/tests/scalar_edges.json'
STUB = (
    ROOT / 'packages/structured-data-models-client/src/sdm_client/_native.pyi'
)
CANARY = 'CANARY-77c1'


# --- codec -----------------------------------------------------------------


def test_every_scalar_edge_decodes_exactly() -> None:
    edges = json.loads(EDGES.read_text(encoding='utf-8'))['edges']
    read = {name: codec.decode_scalar(edge) for name, edge in edges.items()}
    assert read['i64_min'] == -(2**63)
    assert read['u64_max'] == 2**64 - 1
    for name, edge in edges.items():
        value = read[name]
        if edge['t'] in ('d128', 'd256'):
            assert isinstance(value, Decimal)
            assert value.as_tuple().exponent == -edge['s']
            digits = edge['v'].lstrip('-')
            assert ''.join(map(str, value.as_tuple().digits)) == digits
        elif edge['t'] in ('f32', 'f64') and edge['v'] not in ('NaN',):
            assert value == float(edge['v'])
        elif edge['t'] in ('ts', 'duration', 'time64', 'date32'):
            assert isinstance(value, codec.Temporal)
            assert value.value == int(edge['v'])


@pytest.mark.parametrize(
    'scalar',
    [
        {'t': 'i8', 'v': '128'},
        {'t': 'u64', 'v': '-1'},
        {'t': 'i64', 'v': '007'},
        {'t': 'i64', 'v': CANARY},
        {'t': 'f64', 'v': CANARY},
        {'t': 'd128', 'p': 2, 's': 0, 'v': '100'},
        {'t': 'd128', 'p': 39, 's': 0, 'v': '1'},
        {'t': 'ts', 'unit': 'ms', 'zone': None, 'v': '1'},
        {'t': 'ts', 'unit': 'us', 'zone': '', 'v': '1'},
        {'t': 'binary', 'v': f'{CANARY}*'},
        {'t': CANARY, 'v': '1'},
    ],
)
def test_a_scalar_the_contract_does_not_hold_is_refused_without_its_text(
    scalar: dict[str, Any],
) -> None:
    with pytest.raises(codec.CodecError) as refused:
        codec.decode_scalar(scalar)
    assert CANARY not in str(refused.value)


def test_the_encoder_writes_only_what_the_decoder_reads() -> None:
    written = codec.timestamp(-(2**63), 'us', 'UTC')
    assert codec.decode_scalar(written) == codec.Temporal(
        'ts', -(2**63), 'us', 'UTC'
    )
    for refused in [
        lambda: codec.timestamp(2**63, 'us', None),
        lambda: codec.timestamp(0, 'ms', None),
        lambda: codec.timestamp(0, 'us', ''),
        lambda: codec.integer('u8', 256),
        lambda: codec.integer('i64', True),
    ]:
        with pytest.raises(codec.CodecError):
            refused()


def test_an_attached_message_is_read_from_its_attachment() -> None:
    whole = {'message': 'warnings', 'warnings': []}
    control = json.dumps({'message': 'attached', 'attachment': 3}).encode()
    assert codec.read(control, {3: json.dumps(whole).encode()}) == whole
    plain = json.dumps(whole).encode()
    assert codec.read(plain, {}) == whole
    with pytest.raises(codec.CodecError):
        codec.read(control, {})
    with pytest.raises(codec.CodecError):
        codec.read(control, {3: control})
    with pytest.raises(codec.CodecError) as unreadable:
        codec.read(f'{{"message": "{CANARY}'.encode(), {})
    assert CANARY not in str(unreadable.value)


def test_a_message_past_the_cap_is_refused_and_input_travels_attached() -> None:
    big = 'x' * codec.CONTROL_CAP
    with pytest.raises(codec.CodecError):
        codec.encode({'message': 'start', 'text': big})
    assert json.loads(codec.attachment(big)) == big


# --- facts -------------------------------------------------------------------


def test_zone_roots_follow_zoneinfo_s_order_with_their_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mac, debian, empty = tmp_path / 'mac', tmp_path / 'debian', tmp_path / 'e'
    for root in (mac, debian, empty):
        root.mkdir()
    (mac / '+VERSION').write_text('2026c\n')
    (debian / 'tzdata.zi').write_text('# version 2025b\n# more\n')
    missing = tmp_path / 'missing'
    monkeypatch.setattr(
        zoneinfo, 'TZPATH', (str(debian), str(missing), str(mac), str(empty))
    )
    roots = facts.zone_roots()
    system = [root for root in roots if not root['path'].endswith('zoneinfo')]
    assert system == [
        {'path': str(debian), 'version': '2025b'},
        {'path': str(mac), 'version': '2026c'},
        {'path': str(empty), 'version': None},
    ]
    # The tzdata package comes last, when it is installed.
    assert roots[: len(system)] == system


def test_the_local_zone_is_named_or_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / 'Europe').mkdir()
    (tmp_path / 'Europe' / 'Berlin').write_bytes(b'TZif')
    roots = [{'path': str(tmp_path), 'version': None}]
    for setting, expected in [
        ('Europe/Berlin', {'iana': 'Europe/Berlin'}),
        (':Europe/Berlin', {'iana': 'Europe/Berlin'}),
        ('UTC', {'iana': 'UTC'}),
        ('EST5EDT,M3.2.0,M11.1.0', {'posix': 'EST5EDT,M3.2.0,M11.1.0'}),
        ('JST-9', {'posix': 'JST-9'}),
        ('Mars/Olympus', None),
        ('../../etc/passwd', None),
        ('', None),
    ]:
        monkeypatch.setenv('TZ', setting)
        assert facts.local_zone(roots) == expected, setting


def test_a_posix_local_zone_the_host_names_is_one_the_core_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for setting in ('EST5EDT,M3.2.0,M11.1.0', 'JST-9', '<+0330>-3:30'):
        monkeypatch.setenv('TZ', setting)
        assert facts.local_zone([]) == {'posix': setting}
        finished = run(
            {
                'op': 'validate_query',
                'query': 1,
                'schema': 2,
                'profile': {'name': 'RELATIONAL', 'model': None},
            },
            {
                1: codec.attachment('PREDICT t.c FOR EACH t.id'),
                2: codec.attachment({'tables': [], 'links': []}),
            },
        )
        assert finished.result['op'] == 'validate_query'


def test_the_platform_s_year_padding_is_read_from_strftime() -> None:
    import datetime

    platform = facts.platform()
    assert platform['strftime_pads_year'] == (
        datetime.date(1, 1, 1).strftime('%Y') == '0001'
    )
    assert platform['os'] in ('linux', 'darwin', 'windows')
    now = facts.host_facts()['now']
    assert (now['t'], now['unit'], now['zone']) == ('ts', 'us', 'UTC')


# --- errors ------------------------------------------------------------------


def _stub_codes() -> list[str]:
    tree = ast.parse(STUB.read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == '_Code'
        ):
            assert isinstance(node.value, ast.Subscript)
            assert isinstance(node.value.slice, ast.Tuple)
            return [
                element.value
                for element in node.value.slice.elts
                if isinstance(element, ast.Constant)
            ]
    raise AssertionError('the stub lists no codes')


def test_every_code_maps_to_one_structured_data_error_class() -> None:
    codes = _native.ERROR_CODES
    assert isinstance(codes, tuple)
    assert list(codes) == _stub_codes()
    assert set(errors.ERRORS) == set(codes)
    for code in codes:
        cls, sentence = errors.ERRORS[code]
        assert issubclass(cls, StructuredDataError), code
        assert sentence, code


def _error(code: str, message: str, **more: Any) -> bytes:
    error = {'kind': 'internal', 'code': code, 'message': message, **more}
    return json.dumps({'message': 'error', 'error': error}).encode()


def test_the_mapping_keys_on_the_code_and_reads_no_message_text() -> None:
    for code in _native.ERROR_CODES:
        made = [
            errors.error_from_core(_error(code, message, kind=kind), {})
            for message, kind in [
                ('one', 'internal'),
                (CANARY, 'protocol'),
                (f'{CANARY} INVALID_REQUEST', 'data_source'),
            ]
        ]
        assert {type(error) for error in made} == {errors.ERRORS[code][0]}
        assert {error.code for error in made} == {code}
        assert len({str(error) for error in made}) == 1
        assert all(CANARY not in str(error) for error in made)
    # Nothing that maps an error reads a message: only a warning's text is
    # shown, by warn_from_core.
    tree = ast.parse(Path(errors.__file__).read_text(encoding='utf-8'))
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        reads = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Constant) and node.value == 'message'
        ]
        if function.name == 'warn_from_core':
            assert reads
        else:
            assert not reads, function.name


def test_details_carry_the_fields_kind_and_retryability() -> None:
    fields = {
        'sqlstate': {'t': 'utf8', 'v': '40001'},
        'line': {'t': 'u64', 'v': '18446744073709551615'},
        'names': [{'t': 'utf8', 'v': 'a'}, {'t': 'null'}],
    }
    error = errors.error_from_core(
        _error(
            'PROVIDER_ERROR',
            'm',
            kind='data_source',
            fields=fields,
            retryable=True,
        ),
        {},
        driver_class='driver.OperationalError',
    )
    assert error.details == {
        'sqlstate': '40001',
        'line': 2**64 - 1,
        'names': ['a', None],
        'kind': 'data_source',
        'retryable': True,
        'driver_class': 'driver.OperationalError',
    }
    whole = _error('INTERNAL_ERROR', 'x' * 70_000)
    attached = json.dumps({'message': 'attached', 'attachment': 1}).encode()
    assert errors.error_from_core(attached, {1: whole}).code == 'INTERNAL_ERROR'


def test_an_error_is_raised_from_none() -> None:
    try:
        raise ValueError(CANARY)
    except ValueError:
        with pytest.raises(StructuredDataError) as raised:
            errors.raise_from_core(_error('PROVIDER_ERROR', 'm'), {})
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


class _Synthetic(SDMWarning):
    pass


def test_warnings_keep_their_order_and_map_to_their_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(errors.WARNINGS, 'TEST_SYNTHETIC', _Synthetic)
    sent = [
        {'code': 'TEST_SYNTHETIC', 'message': 'first'},
        {'code': 'TEST_OTHER', 'message': 'second'},
        {'code': 'TEST_SYNTHETIC', 'message': 'third'},
    ]
    control = json.dumps({'message': 'warnings', 'warnings': sent}).encode()
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        errors.warn_from_core(control, {})
    assert [(w.category, str(w.message)) for w in shown] == [
        (_Synthetic, '[TEST_SYNTHETIC] first'),
        (SDMWarning, '[TEST_OTHER] second'),
        (_Synthetic, '[TEST_SYNTHETIC] third'),
    ]
