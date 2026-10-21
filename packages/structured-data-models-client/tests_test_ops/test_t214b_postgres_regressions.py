# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regressions for T2.14b's fix round 1 in the Postgres provider's path
(the package's test author's, from review round 2's probes), on the core
route beside the pinned Postgres service.

    SDM_POSTGRES_TEST_DSN=... python -m pytest \
        tests_test_ops/test_t214b_postgres_regressions.py -q

- A declared numeric's refusal names its column, never the cell's text (fix
  1, r1 review P2-1; H17 as KO2-86 keeps it: a value read from data stays
  out of an error). The provider's answer for a ``numeric(100,0)`` column is
  replaced by text no number reads, a sentinel, and the core's refusal is
  held to name the column and hold the sentinel nowhere.
- A ``json[]`` column's elements reach the core as JSON text (fix 1, r1
  review P3-6): the provider marks the list's elements as JSON
  (``arrow.json``) and hands each over as the JSON text of the value the
  driver read, an element that is itself an array kept one value. (What
  the core then makes of the column is its own: the reference fails such a
  column unnamed, ``unsupported data type``.)

Gating: ``contract/tests/t214b-pending.json``'s ``pending`` names, as
``pg-regress:<name>``, the tests that failed when written; each must still
fail (strict). Where ``SDM_POSTGRES_TEST_DSN`` names no server and
``SDM_REQUIRE_POSTGRES`` is not set, there is nothing to hold.
"""

from __future__ import annotations

import json
import os
import traceback
import uuid
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from sdm_client import Schema, StructuredDataError, _routing

ROOT = Path(__file__).resolve().parents[3]
PENDING = ROOT / 'contract/tests/t214b-pending.json'
PREFIX = 'pg-regress:'
NAMES = ('numeric-names-the-column', 'json-array-elements')
#: Text no number reads, which an error must never show.
SENTINEL = 'SENTINEL-numeric-cell-41c9'
#: The json[] column's rows, as the driver reads them: objects, an
#: element that is itself an array, text, a number, a null element, and a
#: null array.
JSON_ROWS = [
    ['{"a": 1}', '[1, [2, 3]]', '"x"'],
    ['2.5', 'null', '{"b": [true, false]}'],
    None,
]


def held(name: str, check: Any) -> None:
    r"""Runs ``check``: it must pass, or, while ``name`` is pending, fail."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert name in gating['authored'], f'{name} is not authored'
    if name not in gating['pending']:
        check()
        return
    try:
        check()
    except Exception:  # noqa: BLE001 - pending: failing, as when written
        return
    raise AssertionError(
        f'{name} passes now: take it out of '
        'contract/tests/t214b-pending.json pending'
    )


def test_the_pending_list_only_shrinks() -> None:
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert set(gating['pending']) <= set(gating['authored'])
    mine = {k for k in gating['authored'] if k.startswith(PREFIX)}
    assert mine == {f'{PREFIX}{name}' for name in NAMES}


@pytest.fixture
def connection(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
            'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
        )
        yield None
        return
    import psycopg

    monkeypatch.setenv(
        _routing.ENVIRONMENT,
        ','.join(
            _routing.FROM_SOURCE_POSTGRES_TABLES + _routing.MATERIALIZE_POSTGRES
        ),
    )
    schema = f't214b_{uuid.uuid4().hex[:12]}'
    with psycopg.connect(dsn) as made:
        made.execute(f'CREATE SCHEMA {schema}')
        made.execute(f'SET search_path TO {schema}')
        made.commit()
        try:
            yield made
        finally:
            made.rollback()
            made.execute(f'DROP SCHEMA {schema} CASCADE')
            made.commit()


def _answers(monkeypatch: pytest.MonkeyPatch, edit: Any = None) -> list[Any]:
    r"""The Postgres provider, each answer it hands the core kept (after
    ``edit``, where one is given)."""
    from sdm_client._rust.providers.postgres import PostgresProvider

    kept: list[Any] = []
    original = PostgresProvider.__call__

    def call(self: Any, *arguments: Any, **named: Any) -> Any:
        answer = original(self, *arguments, **named)
        table = pa.RecordBatchReader._import_from_c_capsule(
            answer.__arrow_c_stream__()
        ).read_all()
        if edit is not None:
            table = edit(table)
        kept.append(table)
        return table

    monkeypatch.setattr(PostgresProvider, '__call__', call)
    return kept


def _marked_json(field: Any) -> bool:
    r"""Whether ``field`` is marked JSON: Arrow's ``arrow.json`` extension,
    as a registered type (a newer PyArrow) or as the field's metadata."""
    kind = field.type
    if isinstance(kind, pa.BaseExtensionType):
        return kind.extension_name == 'arrow.json'
    return (field.metadata or {}).get(b'ARROW:extension:name') == b'arrow.json'


def _from_source(connection: Any, table: str) -> Any:
    from sdm_connectors import PostgresSource

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_source(
            PostgresSource(connection), tables={table: table}
        )


def test_a_declared_numeric_s_refusal_names_its_column_never_its_cell(
    connection: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""Fix 1 (r1 review P2-1): a ``numeric(100,0)`` column whose cells the
    core cannot read is refused naming the column (``nf``), and the cell's
    text is in no part of the error: its message, details, ``repr`` or
    traceback."""
    if connection is None:
        return
    connection.execute(
        'CREATE TABLE numbers (id BIGINT PRIMARY KEY, nf NUMERIC(100, 0))'
    )
    connection.execute('INSERT INTO numbers VALUES (1, 1), (2, 2)')
    connection.commit()

    def unreadable(table: Any) -> Any:
        if 'nf' not in table.column_names:
            return table
        at = table.schema.get_field_index('nf')
        return table.set_column(
            at,
            table.schema.field(at),
            pa.array([SENTINEL] * table.num_rows, pa.string()),
        )

    def check() -> None:
        _answers(monkeypatch, unreadable)
        with pytest.raises(StructuredDataError) as refused:
            _from_source(connection, 'numbers')
        error = refused.value
        written = ' '.join(
            [
                str(error),
                repr(error),
                repr(error.details),
                ''.join(traceback.format_exception(error)),
            ]
        )
        assert SENTINEL not in written, written
        assert 'nf' in str(error) or (error.details or {}).get('column') == (
            'nf'
        ), (str(error), error.details)

    held(f'{PREFIX}numeric-names-the-column', check)


def test_a_json_array_s_elements_reach_the_core_as_json_text(
    connection: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""Fix 1 (r1 review P3-6): a ``json[]`` column is handed over as a list
    whose elements are marked JSON, each the JSON text of the value the
    driver read (an array element kept one value, a null element null, a
    null array null)."""
    if connection is None:
        return
    connection.execute(
        'CREATE TABLE docs (id BIGINT PRIMARY KEY, d JSON[], age INTEGER)'
    )
    for at, row in enumerate(JSON_ROWS):
        connection.execute(
            'INSERT INTO docs VALUES (%s, %s::json[], %s)',
            (at, None if row is None else row, 20 + at),
        )
    connection.commit()
    read = [
        value
        for (value,) in connection.execute(
            'SELECT d FROM docs ORDER BY id'
        ).fetchall()
    ]

    def check() -> None:
        kept = _answers(monkeypatch)
        _from_source(connection, 'docs')
        answers = [t for t in kept if 'd' in t.column_names]
        assert answers, 'no answer held the column'
        for table in answers:
            field = table.schema.field('d')
            assert pa.types.is_list(field.type), field
            element = field.type.value_field
            assert _marked_json(element), element
            ids = table.column('id').to_pylist()
            for key, cells in zip(
                ids, table.column('d').to_pylist(), strict=True
            ):
                driver = read[key]
                if driver is None:
                    assert cells is None, (key, cells)
                    continue
                assert cells is not None and len(cells) == len(driver), cells
                for text, value in zip(cells, driver, strict=True):
                    if value is None:
                        assert text in (None, 'null'), (key, text)
                    else:
                        assert isinstance(text, str), (key, text)
                        assert json.loads(text) == value, (key, text, value)

    held(f'{PREFIX}json-array-elements', check)
