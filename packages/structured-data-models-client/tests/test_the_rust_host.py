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
import decimal
import json
import math
import struct
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
FLOATS = ROOT / 'rust/crates/sdm-contract/tests/float_text.json'
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
        elif edge['t'] == 'f64' and edge['v'] != 'NaN':
            assert value == float(edge['v'])
        elif edge['t'] == 'f32' and edge['v'] != 'NaN':
            width = struct.unpack('>f', struct.pack('>f', float(edge['v'])))
            assert value == width[0]
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


@pytest.mark.parametrize('trapping', [False, True])
def test_float_text_reads_at_its_width_as_the_core_reads_it(
    trapping: bool,
) -> None:
    # The vectors sdm-contract's float_text.rs checks the core against, read
    # also under a caller's context that traps mixing floats and decimals
    # and every inexact result: the codec keeps a context of its own.
    if trapping:
        with decimal.localcontext() as caller:
            caller.traps[decimal.FloatOperation] = True
            caller.traps[decimal.Inexact] = True
            caller.prec = 3
            _read_float_vectors()
            edges = json.loads(EDGES.read_text(encoding='utf-8'))['edges']
            for edge in edges.values():
                codec.decode_scalar(edge)
    else:
        _read_float_vectors()


def _read_float_vectors() -> None:
    document = json.loads(FLOATS.read_text(encoding='utf-8'))
    assert document['format'] == 'sdm-float-text/1'
    for vector in document['vectors']:
        scalar = {'t': vector['t'], 'v': vector['v']}
        if vector.get('refused'):
            with pytest.raises(codec.CodecError):
                codec.decode_scalar(scalar)
            continue
        value = codec.decode_scalar(scalar)
        if vector.get('nan'):
            assert math.isnan(value), scalar
            continue
        width = '>f' if vector['t'] == 'f32' else '>d'
        assert struct.pack(width, value).hex() == vector['bits'], scalar
    # A float32 widens exactly: 0.1 read at its width is not the double 0.1.
    assert codec.decode_scalar({'t': 'f32', 'v': '0.1'}) == 0.10000000149011612


def test_the_encoder_writes_only_what_the_decoder_reads() -> None:
    written = codec.timestamp(-(2**63), 'us', 'UTC')
    assert codec.decode_scalar(written) == codec.Temporal(
        'ts', -(2**63), 'us', 'UTC'
    )
    for refused in [
        lambda: codec.timestamp(2**63, 'us', None),
        lambda: codec.timestamp(0, 'ms', None),
        lambda: codec.timestamp(0, 'us', ''),
        lambda: codec.timestamp(0, 'us', 42),  # type: ignore[arg-type]
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


#: ``TZ`` settings and the local zone each gives over no zone roots: a POSIX
#: string the host's parser takes keeps the zone; anything else is no zone.
TZ_SETTINGS: list[tuple[str, dict[str, str] | None]] = [
    ('<+03>-3', {'posix': '<+03>-3'}),
    ('JST-9', {'posix': 'JST-9'}),
    ('UTC0', {'posix': 'UTC0'}),
    ('EST5EDT,M3.2.0,M11.1.0', {'posix': 'EST5EDT,M3.2.0,M11.1.0'}),
    ('CET-1CEST,M3.5.0,M10.5.0/3', {'posix': 'CET-1CEST,M3.5.0,M10.5.0/3'}),
    ('GMT0BST,M3.5.0/1,M10.5.0', {'posix': 'GMT0BST,M3.5.0/1,M10.5.0'}),
    ('UTC', {'iana': 'UTC'}),
    # A daylight name with no rule.
    ('CET-1CEST', None),
    ('GMT0BST', None),
    ('EST5EDT', None),
    # Malformed offsets and rules, and hours out of range.
    ('ABC99', None),
    ('XYZ25', None),
    ('EST5EDT,garbage,garbage', None),
    ('<+03', None),
    ('EST', None),
    ('Mars/Olympus', None),
    ('../../etc/passwd', None),
    ('', None),
]


def test_the_local_zone_is_named_only_as_the_host_reads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for setting, expected in TZ_SETTINGS:
        monkeypatch.setenv('TZ', setting)
        assert facts.local_zone([], _native.is_posix_zone) == expected, setting
    # An IANA name a root holds, with or without the leading colon, wins
    # over the POSIX reading (EST5EDT is both).
    (tmp_path / 'Europe').mkdir()
    (tmp_path / 'Europe' / 'Berlin').write_bytes(b'TZif')
    (tmp_path / 'EST5EDT').write_bytes(b'TZif')
    roots = [{'path': str(tmp_path), 'version': None}]
    for setting, name in [
        ('Europe/Berlin', 'Europe/Berlin'),
        (':Europe/Berlin', 'Europe/Berlin'),
        ('EST5EDT', 'EST5EDT'),
    ]:
        monkeypatch.setenv('TZ', setting)
        assert facts.local_zone(roots, _native.is_posix_zone) == {'iana': name}


@pytest.mark.parametrize(('setting', 'expected'), TZ_SETTINGS)
def test_no_tz_setting_refuses_a_run_and_a_valid_one_keeps_its_zone(
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    expected: dict[str, str] | None,
) -> None:
    monkeypatch.setenv('TZ', setting)
    # No root holds a name, so a run's facts are exactly the table's.
    monkeypatch.setattr(facts, 'zone_roots', list)
    sent: list[Any] = []
    real = facts.host_facts

    def spy(is_posix: Any) -> dict[str, Any]:
        built = real(is_posix)
        sent.append(built['local_zone'])
        return built

    monkeypatch.setattr(facts, 'host_facts', spy)
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
    assert sent == [expected]


def test_the_platform_s_year_padding_is_read_from_strftime() -> None:
    import datetime

    platform = facts.platform()
    assert platform['strftime_pads_year'] == (
        datetime.date(1, 1, 1).strftime('%Y') == '0001'
    )
    assert platform['os'] in ('linux', 'darwin', 'windows')
    now = facts.host_facts(_native.is_posix_zone)['now']
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
    # shown, by said, for a code the warning table does not word (T1.23).
    tree = ast.parse(Path(errors.__file__).read_text(encoding='utf-8'))
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        reads = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Constant) and node.value == 'message'
        ]
        if function.name == 'said':
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


def _warning(code: str, **fields: Any) -> dict[str, Any]:
    r"""A core warning of ``code`` whose fields are ``fields``, each a text,
    a list of texts, a count or a scalar already encoded."""

    def encoded(value: Any) -> Any:
        if isinstance(value, list):
            return [encoded(item) for item in value]
        if isinstance(value, dict):
            return value
        if isinstance(value, int):
            return {'t': 'u64', 'v': str(value)}
        return {'t': 'utf8', 'v': value}

    return {
        'code': code,
        'message': 'the core says this naming nothing',
        'fields': {name: encoded(value) for name, value in fields.items()},
    }


def _shown(*given: dict[str, Any]) -> list[tuple[str, str]]:
    control = json.dumps({'message': 'warnings', 'warnings': list(given)})
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        errors.warn_from_core(control.encode(), {})
    # Not a ResourceWarning the collector raises meanwhile for another
    # test's unclosed connection.
    return [
        (type(w.message).__name__, str(w.message))
        for w in shown
        if not issubclass(w.category, ResourceWarning)
    ]


def test_the_schema_s_warnings_are_worded_here_as_the_reference_words_them() -> (
    None
):
    # T1.23 (E59): the core gives codes and fields, and this host words
    # them, the reference's Python remedies included. The texts the core
    # wrote itself before T1.23, word for word.
    assert _shown(
        _warning(
            'LINK_CANNOT_JOIN',
            source_table='orders',
            source_column='user_id',
            target_table='users',
            target_column='user_id',
            reason='types',
            variant='types',
            source_dtype='string',
            target_dtype='int64',
        )
    ) == [
        (
            'SDMWarning',
            "[LINK_CANNOT_JOIN] orders.user_id names the key of 'users' and "
            "holds 'string' where that key holds 'int64', which cannot match, "
            'so no relationship was made from it. Store the two as one type '
            'if they are the same identifier.',
        )
    ]
    assert _shown(
        _warning(
            'LINK_CANNOT_JOIN',
            source_table='lines',
            source_column='sku',
            target_table='items',
            target_column='sku',
            reason='not_the_key',
            variant='not_the_key',
            key=['item_id'],
        )
    )[0][1] == (
        '[LINK_CANNOT_JOIN] lines.sku refers to items.sku, but the primary '
        "key of 'items' is 'item_id'. A relationship joins to a primary key, "
        "so none was made from it. Pass primary_keys={'items': 'sku'} if "
        "'sku' identifies a row of 'items'."
    )
    two_ways = _shown(
        _warning(
            'TABLES_NAME_EACH_OTHER',
            source_table='accounts',
            target_table='refunds',
        ),
        _warning(
            'TWO_WAYS_TO_A_TABLE',
            source_table='refunds',
            target_table='accounts',
            columns=['account_id', 'payer'],
            kept='account_id',
            others=['payer'],
        ),
    )
    assert [text for _, text in two_ways] == [
        "[TABLES_NAME_EACH_OTHER] 'accounts' and 'refunds' name each other, "
        'and a relationship is crossed in both directions, so this is two '
        'ways over one pair and an aggregation between them is refused for '
        'not having a single one. Declare whichever of the two a question is '
        'about and leave the other out.',
        "[TWO_WAYS_TO_A_TABLE] 'refunds' reaches 'accounts' through "
        "'account_id' and 'payer', and an aggregation across the two tables "
        'has to be resolved to one of them, so a query that crosses this pair '
        'is refused until it is. They are all kept because they are all real, '
        'and which one a question means is not something this can decide. To '
        "keep 'account_id', drop the others: schema.relationships = [link for "
        'link in schema.relationships if not (link.source_table == '
        "'refunds' and link.target_table == 'accounts' and "
        "link.source_column in ['payer'])].",
    ]
    # One warning a line, shown as the reference's one warning of them all.
    stamp = {'t': 'ts', 'unit': 'us', 'zone': None, 'v': '1757585707263000'}
    guessed = _shown(
        _warning(
            'TIME_COLUMNS_GUESSED',
            table='plans',
            column='modified_at',
            reason='edit_stamp',
            variant='refused_edit_stamp',
        ),
        _warning(
            'TIME_COLUMNS_GUESSED',
            table='plans',
            column='stamped',
            reason='constant',
            missing=0,
            rows=9,
            value=stamp,
            variant='refused_constant',
        ),
        _warning(
            'TIME_COLUMNS_GUESSED',
            table='visits',
            column='seen',
            variant='taken_earliest',
            among=['seen', 'left', 'third'],
            other='left',
        ),
    )
    assert guessed == [
        (
            'SDMWarning',
            '[TIME_COLUMNS_GUESSED] Some time columns were inferred, or '
            'refused, on a guess about what a column holds:\n'
            '- plans.modified_at was not taken as the time column because it '
            'looks like when a row was last changed, not when it began. If it '
            'does say when a row became true, name it: '
            "time_columns={'plans': 'modified_at'}.\n"
            '- plans.stamped was not taken as the time column because every '
            'row holds 2025-09-11 10:15:07.263000. If it does say when a row '
            "became true, name it: time_columns={'plans': 'stamped'}.\n"
            '- visits.seen was taken as the time column because its values '
            "start earliest of 'seen', 'left' and 'third'. If another says "
            'when a row became true, name it: '
            "time_columns={'visits': 'left'}. If none does, clear it: "
            "time_columns={'visits': None}.",
        )
    ]
    # Columns left out, said once a table.
    assert _shown(
        _warning(
            'COLUMNS_LEFT_OUT', table='users', column='f', dtype='intlist'
        ),
        _warning(
            'COLUMNS_LEFT_OUT', table='users', column='g', dtype='stringlist'
        ),
        _warning(
            'COLUMNS_LEFT_OUT', table='orders', column='h', dtype='floatlist'
        ),
    ) == [
        (
            'SDMWarning',
            '[COLUMNS_LEFT_OUT] users leaves out f (intlist), g (stringlist): '
            'a request carries one scalar value per cell, and these hold none',
        ),
        (
            'SDMWarning',
            '[COLUMNS_LEFT_OUT] orders leaves out h (floatlist): a request '
            'carries one scalar value per cell, and these hold none',
        ),
    ]
    time = [
        {'t': 'time64', 'unit': 'us', 'v': str(v)}
        for v in (1_800_000_000, 5_400_123_456)
    ]
    assert _shown(
        _warning(
            'TIME_VALUES_UNREAD',
            table='users',
            column='f',
            unread=1_234,
            rows=2_000,
            examples=time,
        )
    ) == [
        (
            'UserWarning',
            '[TIME_VALUES_UNREAD] Could not parse 1,234 of 2,000 value(s) of '
            "time column 'f' (for example [datetime.time(0, 30), "
            "datetime.time(1, 30, 0, 123456)]); they were dropped as 'NaT'",
        )
    ]


def test_a_remedy_names_a_table_as_python_writes_its_name() -> None:
    # The vectors CPython 3.12's repr wrote (tests/quoted.json, from
    # T1.21d): each name a remedy holds is the literal Python reads back.
    vectors = json.loads(
        (Path(__file__).parent / 'quoted.json').read_text(encoding='utf-8')
    )['vectors']
    assert len(vectors) >= 20
    for vector in vectors:
        [(_, text)] = _shown(
            _warning(
                'TIME_COLUMNS_GUESSED',
                table=vector['name'],
                column='c',
                variant='refused_load_stamp',
            )
        )
        assert text.endswith(f"time_columns={{{vector['repr']}: 'c'}}."), text
        assert ast.literal_eval(vector['repr']) == vector['name']


def test_a_warning_the_table_cannot_word_is_shown_in_the_core_s_sentence() -> (
    None
):
    # An unknown code, an unknown variant, and a field the words need left
    # out: each shown, in the core's own sentence, never raised.
    for given in (
        _warning('NOT_A_CODE_HERE'),
        _warning('TIME_COLUMNS_GUESSED', table='t', column='c', variant='new'),
        _warning('TWO_WAYS_TO_A_TABLE', source_table='a'),
    ):
        [(_, text)] = _shown(given)
        assert text.endswith('the core says this naming nothing'), text


#: The core's warnings this host does not word, each with why: none. Every
#: code and variant the core gives (`_native.WARNING_CODES`) is worded from
#: `errors.SAID` (T1.23 fix 1, subagent P2-1).
CORE_WORDED: dict[tuple[str, str | None], str] = {}


def test_every_warning_the_core_gives_is_worded_here() -> None:
    # The binding lists every code the core may give with its variants, as
    # it lists its error codes; each is worded by SAID exactly once, or
    # named here as the core's to word, with the reason. The fallback to the
    # core's sentence stays for robustness; this test is the guard.
    given = {
        (code, variant)
        for code, variants in _native.WARNING_CODES
        for variant in (variants or (None,))
    }
    assert len(given) >= 22
    worded = {
        (code, variant)
        for code, templates in errors.SAID.items()
        for variant in templates
    }
    assert given - set(CORE_WORDED) == worded
    assert not set(CORE_WORDED) & worded
    for code, variants in _native.WARNING_CODES:
        assert code == code.upper() and ' ' not in code, code
        assert len(set(variants)) == len(variants), code
