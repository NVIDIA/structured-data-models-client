# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The capture writer and reader (Phase 3 design, choice 2; T3.3).

A capture is what the old client did and got on one warehouse leg for one
case, recorded live in C-1 (or E-0) by ``run_python.py``'s recording mode
(T3.2): the old client's answers and the statements it sent, which the
core's SQL is compared with (the owner's B-lite). As the writer takes it
and :func:`read` gives it back, a capture is plain data:

- ``case``, ``leg``: the case's id and its warehouse leg (:data:`LEGS`);
- ``recording_commit``, ``reference_commit``: the commit whose harness
  recorded it, and the reference's, each a full SHA;
- ``versions``: the leg's runtime, Python, pandas and driver versions;
- ``input_hash``: the hash of the case's input;
- ``session``: the session's ``time_zone`` and ``null_order``
  (``nulls_first`` or ``nulls_last`` for an ascending order), the session
  the recording ran in;
- ``reference``: the old client's reference line, the observation
  Phase 2's lines use;
- ``tables``: each table of the case as the warehouse returned it through
  the leg's driver, a ``pyarrow.Table`` by name;
- ``derived``: each relation the old client derived from a view, evaluated
  by the warehouse: ``view``, ``table``, ``columns``, ``filter``, ``data``
  (a table, or ``None`` with the ``refused`` columns);
- ``draws``: each random draw: ``table``, ``seed`` (``None`` where the
  engine takes none), ``fraction`` (the sampling clause's argument as the
  old client wrote it: ``12.5``, ``1e-05``, ``5 ROWS``) and ``rows``, the
  rows picked, a table;
- ``statements``: each statement sent, in order: ``text``, ``parameters``
  (JSON values, as passed), and its ``result`` (a table) or ``error``;
- ``loader``: the loader's notes, each constraint the engine did not
  store (optional).

On disk a capture is a directory: ``capture.json``, the capture with each
table replaced by the name of its file, and ``tables/<sha256>.arrows``, each
distinct table once, an uncompressed Arrow IPC stream. C-1's captures are
``contract/captures/<leg>/<case>/``, and each leg's ``manifest.json``
(:func:`write_manifest`) lists its cases, each one's input hash and its
directory, with the run's commits and the leg's versions, which
``capture_check.py`` (T3.2) reads.

Private names never reach a file. Each value of a variable of
:data:`PLACEHOLDERS` in the environment is replaced by its placeholder
(``{catalog}``, ``{schema}``, ``{host}``, ``{user}``, ...) in every text of
the capture, the tables' cells, field names and metadata included, in each
case form a warehouse folds it to (``MY_DB`` becomes ``{CATALOG}``), and a
URL's host by ``{host}``; then the whole capture is searched,
case-insensitively, for the value of every variable of :data:`PRIVATE` and
every ``SDM_PRIVATE_*``, ``SDM_LIVE_*`` or ``SDM_CAPTURE_*`` variable, and
the write fails by
the variable's name, never its value (:class:`PrivateName`), before any
file is written.

The format is versioned. An older version is read through
:data:`CONVERTERS`, one function per version, so an old capture is
converted, never re-recorded (choice 2): ``captures.py convert DIR``
rewrites every capture under DIR in the current version.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

#: The capture format this writer writes; :func:`upgrade` reads every
#: version in :data:`VERSIONS`.
FORMAT = 'sdm-capture/1'
VERSIONS = ('sdm-capture/1',)

#: The legs a capture is for (design, choice 1).
LEGS = ('databricks-sql', 'databricks-spark', 'snowflake', 'lakebase')

#: Each leg's NULL order in an ascending sort, as its engine orders by
#: default (Spark puts NULLs first; Snowflake and Postgres last).
NULL_ORDER = {
    'databricks-sql': 'nulls_first',
    'databricks-spark': 'nulls_first',
    'snowflake': 'nulls_last',
    'lakebase': 'nulls_last',
}

#: A capture directory's document.
DOCUMENT = 'capture.json'

#: Where C-1's captures live, from the repository's root.
ROOT = Path(__file__).resolve().parents[2]
CAPTURES = ROOT / 'contract' / 'captures'

#: Each environment variable whose value is a private name, and the
#: placeholder written in its place. A case names a private resource by a
#: placeholder, which a live run resolves from ``SDM_CAPTURE_<NAME>``
#: (``run_python.py``'s ``resolve_placeholders``, T3.2): the capture
#: writes the placeholder back for that variable's value, so the case and
#: its capture name it alike. Every ``SDM_CAPTURE_*`` variable is one
#: (:func:`placeholder_variables`); those listed here are the ones the
#: cases and live runs use: the test schema the write guard allows
#: (KO3-51), the existing views read (KO3-52, KO3-21), the hosts and the
#: users.
PLACEHOLDERS = {
    'SDM_CAPTURE_CATALOG': '{catalog}',
    'SDM_CAPTURE_SCHEMA': '{schema}',
    'SDM_CAPTURE_ADVENTUREWORKS_METRIC_VIEW': '{adventureworks_metric_view}',
    'SDM_CAPTURE_SHOP_METRIC_VIEW': '{shop_metric_view}',
    'SDM_CAPTURE_SHOP_SEMANTIC_VIEW': '{shop_semantic_view}',
    'SDM_CAPTURE_HOST': '{host}',
    'SDM_CAPTURE_USER': '{user}',
    'DATABRICKS_HOST': '{host}',
    'DATABRICKS_SERVER_HOSTNAME': '{host}',
    'SNOWFLAKE_URL': '{host}',
    'PGHOST': '{host}',
    'DATABRICKS_USER': '{user}',
    'SNOWFLAKE_USERNAME': '{user}',
    'PGUSER': '{user}',
}
#: The prefix of a placeholder's variable: ``SDM_CAPTURE_<NAME>`` stands
#: for ``{<name>}``.
PLACEHOLDER_PREFIX = 'SDM_CAPTURE_'


def placeholder_variables(environ: Mapping[str, str]) -> dict[str, str]:
    r"""Each variable of the environment with a placeholder, and the
    placeholder: :data:`PLACEHOLDERS`', and ``{<name>}`` for every
    ``SDM_CAPTURE_<NAME>``."""
    found = dict(PLACEHOLDERS)
    for name in environ:
        rest = name[len(PLACEHOLDER_PREFIX) :]
        if name.startswith(PLACEHOLDER_PREFIX) and re.fullmatch(
            r'[A-Z][A-Z0-9_]*', rest
        ):
            found.setdefault(name, '{' + rest.lower() + '}')
    return found


#: Every environment variable whose value must never be written: the
#: placeholders' and every credential or resource name a live run reads
#: (the owner's Snowflake env file among them, KO3-24). Any variable named
#: ``SDM_PRIVATE_*``, ``SDM_LIVE_*`` or ``SDM_CAPTURE_*`` is searched for
#: too, so a live run names a resource of its own without changing this
#: list.
PRIVATE = tuple(
    sorted(
        {
            *PLACEHOLDERS,
            'DATABRICKS_HTTP_PATH',
            'DATABRICKS_WAREHOUSE_ID',
            'DATABRICKS_CLIENT_ID',
            'DATABRICKS_CLIENT_SECRET',
            'DATABRICKS_TOKEN',
            'DATABRICKS_CONFIG_PROFILE',
            'SNOWFLAKE_ACCOUNT',
            'SNOWFLAKE_USER',
            'SNOWFLAKE_ACCESS_TOKEN',
            'SNOWFLAKE_PASSWORD',
            'SNOWFLAKE_WAREHOUSE',
            'SNOWFLAKE_DATABASE',
            'SNOWFLAKE_SCHEMA',
            'SNOWFLAKE_ROLE',
            'PGPASSWORD',
            'PGDATABASE',
            'SDM_DBX_PROFILE',
            'SDM_DBX_WORKSPACE_FOLDER',
            'SDM_DBX_POLICY_ID',
            'SDM_DBX_USER',
            'SDM_DBX_VOLUME_DIR',
        }
    )
)
PRIVATE_PREFIXES = ('SDM_PRIVATE_', 'SDM_LIVE_', PLACEHOLDER_PREFIX)

#: The shortest value searched for or replaced when found by a prefix or
#: as a URL's part: a shorter one would match ordinary words. A listed
#: variable's whole value is searched for at any length.
SHORTEST = 4

#: The keys of a capture, the required ones and those with a default.
REQUIRED = (
    'case',
    'leg',
    'recording_commit',
    'reference_commit',
    'versions',
    'input_hash',
    'session',
    'reference',
    'tables',
)
OPTIONAL = {'derived': [], 'draws': [], 'statements': [], 'loader': []}


class CaptureError(Exception):
    r"""A capture that cannot be written or read as this format says."""


class PrivateName(CaptureError):
    r"""A private name left in a capture: named by where it is and by the
    variable it is the value of, never by the value."""


# --- private names -------------------------------------------------------------------


def _url_parts(value: str) -> list[str]:
    r"""A URL's host and the host's first label, when ``value`` is one."""
    host = re.match(r'^[a-z][a-z0-9+.-]*://([^/:?#@]+)', value, re.I)
    if not host:
        return []
    return [host.group(1), host.group(1).split('.')[0]]


def private_values(
    environ: Mapping[str, str] | None = None,
) -> dict[str, list[str]]:
    r"""Each private variable set in the environment and the values it
    makes private: its value, and a URL's host and first label."""
    environ = os.environ if environ is None else environ
    found: dict[str, list[str]] = {}
    for name, value in sorted(environ.items()):
        listed = name in PRIVATE
        if not listed and not name.startswith(PRIVATE_PREFIXES):
            continue
        value = value.strip()
        if not value or (not listed and len(value) < SHORTEST):
            continue
        values = [value]
        values += [part for part in _url_parts(value) if len(part) >= SHORTEST]
        found[name] = values
    return found


class Placeholders:
    r"""Each private value with a placeholder, replaced by it where it is a
    whole name (no letter, digit or underscore either side, as T3.2's
    recording mode reads a name): the value as set, and its upper- and
    lower-case forms by the placeholder's upper- and lower-case forms, so a
    name a warehouse folds keeps its fold offline. A value inside other
    text (``main`` in ``domain``) is left for the search, which fails the
    write: data is never changed to hide a name."""

    def __init__(self, names: Mapping[str, str]) -> None:
        pairs: dict[str, str] = {}
        for value, token in names.items():
            for found, written in (
                (value, token),
                (value.upper(), token.upper()),
                (value.lower(), token.lower()),
            ):
                pairs.setdefault(found, written)
        # Longest first, so a name holding another is replaced whole.
        ordered = sorted(pairs.items(), key=lambda p: (-len(p[0]), p[0]))
        self._tokens = dict(ordered)
        self._pattern = (
            re.compile(
                r'(?<![A-Za-z0-9_])(?:'
                + '|'.join(re.escape(value) for value, _ in ordered)
                + r')(?![A-Za-z0-9_])'
            )
            if ordered
            else None
        )

    def __call__(self, text: str) -> str:
        if self._pattern is None:
            return text
        return self._pattern.sub(lambda m: self._tokens[m.group(0)], text)


def placeholders(environ: Mapping[str, str] | None = None) -> Placeholders:
    r"""The placeholders of the environment's private names
    (:data:`PLACEHOLDERS`), a URL's host and first label as ``{host}``."""
    environ = os.environ if environ is None else environ
    values = private_values(environ)
    names: dict[str, str] = {}
    for variable, token in placeholder_variables(environ).items():
        for value in values.get(variable, []):
            names.setdefault(value, token)
    return Placeholders(names)


def _searcher(private: Mapping[str, str]) -> Callable[[str], str | None]:
    r"""A function naming the variable of the first private value a text
    holds, case-insensitively."""
    labels = {value.casefold(): label for value, label in private.items()}
    if not labels:
        return lambda text: None
    pattern = re.compile(
        '|'.join(re.escape(v) for v in sorted(labels, key=len, reverse=True))
    )

    def search(text: str) -> str | None:
        found = pattern.search(text.casefold())
        return labels[found.group(0)] if found else None

    return search


def searcher(
    environ: Mapping[str, str] | None = None,
) -> Callable[[str], str | None]:
    r"""The search for every private value of the environment, naming the
    variable holding it."""
    return _searcher(
        {
            value: variable
            for variable, values in private_values(environ).items()
            for value in values
        }
    )


# --- Arrow ---------------------------------------------------------------------------


def _metadata(
    metadata: Mapping[bytes, bytes] | None, change: Callable[[str], str]
) -> dict[bytes, bytes] | None:
    if not metadata:
        return metadata  # type: ignore[return-value]
    return {
        change(k.decode('utf-8', 'replace')).encode(): change(
            v.decode('utf-8', 'replace')
        ).encode()
        for k, v in metadata.items()
    }


def _field(field: Any, kind: Any, change: Callable[[str], str]) -> Any:
    r"""``field`` with its name and metadata through ``change`` and its type
    ``kind``, its nullability kept."""
    import pyarrow as pa

    return pa.field(
        change(field.name),
        kind,
        field.nullable,
        _metadata(field.metadata, change),
    )


def _mask(array: Any) -> Any:
    import pyarrow as pa

    return pa.array(
        [not valid for valid in array.is_valid().to_pylist()], pa.bool_()
    )


def _texts(array: Any, change: Callable[[str], str]) -> Any:
    r"""``array`` with every text in it, at any depth, through ``change``:
    cells, and each nested field's name and metadata; every other part of
    each type (a child's nullability, a list's element name, a map's
    sortedness) kept as it was."""
    import pyarrow as pa

    kind = array.type
    t = pa.types
    if t.is_string(kind) or t.is_large_string(kind):
        return pa.array(
            [None if v is None else change(v) for v in array.to_pylist()],
            type=kind,
        )
    if t.is_dictionary(kind):
        return pa.DictionaryArray.from_arrays(
            array.indices, _texts(array.dictionary, change)
        )
    if t.is_struct(kind):
        children = [
            _texts(array.field(i), change) for i in range(kind.num_fields)
        ]
        fields = [
            _field(kind.field(i), children[i].type, change)
            for i in range(kind.num_fields)
        ]
        return pa.StructArray.from_arrays(
            children, fields=fields, mask=_mask(array)
        )
    if t.is_map(kind):
        keys = _texts(array.keys, change)
        items = _texts(array.items, change)
        mapped = pa.map_(
            _field(kind.key_field, keys.type, change),
            _field(kind.item_field, items.type, change),
            kind.keys_sorted,
        )
        return pa.MapArray.from_arrays(
            array.offsets, keys, items, type=mapped, mask=_mask(array)
        )
    if t.is_list(kind) or t.is_large_list(kind):
        values = _texts(array.values, change)
        element = _field(kind.value_field, values.type, change)
        if t.is_list(kind):
            return pa.ListArray.from_arrays(
                array.offsets, values, type=pa.list_(element), mask=_mask(array)
            )
        return pa.LargeListArray.from_arrays(
            array.offsets,
            values,
            type=pa.large_list(element),
            mask=_mask(array),
        )
    if t.is_fixed_size_list(kind):
        values = _texts(array.values, change)
        element = _field(kind.value_field, values.type, change)
        return pa.FixedSizeListArray.from_arrays(
            values, type=pa.list_(element, kind.list_size), mask=_mask(array)
        )
    return array


def _type_texts(kind: Any) -> Iterable[str]:
    r"""Every nested field's name and metadata a type holds, at any depth."""
    import pyarrow as pa

    t = pa.types
    if t.is_dictionary(kind):
        yield from _type_texts(kind.value_type)
        return
    if t.is_map(kind):
        fields = [kind.key_field, kind.item_field]
    elif t.is_list(kind) or t.is_large_list(kind) or t.is_fixed_size_list(kind):
        fields = [kind.value_field]
    elif t.is_struct(kind) or t.is_union(kind):
        fields = [kind.field(i) for i in range(kind.num_fields)]
    else:
        return
    for field in fields:
        yield field.name
        for k, v in (field.metadata or {}).items():
            yield k.decode('utf-8', 'replace')
            yield v.decode('utf-8', 'replace')
        yield from _type_texts(field.type)


def _changes(table: Any, change: Callable[[str], str]) -> bool:
    r"""Whether ``change`` alters any text ``table`` holds."""
    return any(change(text) != text for text in _table_texts(table))


def scrubbed_table(table: Any, change: Callable[[str], str]) -> Any:
    r"""``table`` with every text through ``change``: its cells at any
    depth, every field's name and metadata, nested ones included, and the
    schema's metadata. A table ``change`` leaves alone comes back as it is,
    every type exactly the driver's."""
    import pyarrow as pa

    if not _changes(table, change):
        return table
    table = table.combine_chunks()
    columns = []
    fields = []
    for i, field in enumerate(table.schema):
        column = table.column(i)
        chunk = (
            column.chunk(0) if column.num_chunks else pa.array([], field.type)
        )
        array = _texts(chunk, change)
        columns.append(array)
        fields.append(_field(field, array.type, change))
    schema = pa.schema(
        fields, metadata=_metadata(table.schema.metadata, change)
    )
    return pa.Table.from_arrays(columns, schema=schema)


def _table_texts(table: Any) -> Iterable[str]:
    r"""Every text a table holds: every field's name and metadata, nested
    fields' included, the schema's metadata, and each text or byte cell
    (bytes read as UTF-8, invalid bytes replaced)."""
    import pyarrow as pa

    for field in table.schema:
        yield field.name
        for k, v in (field.metadata or {}).items():
            yield k.decode('utf-8', 'replace')
            yield v.decode('utf-8', 'replace')
        yield from _type_texts(field.type)
    for k, v in (table.schema.metadata or {}).items():
        yield k.decode('utf-8', 'replace')
        yield v.decode('utf-8', 'replace')

    def walk(value: Any) -> Iterable[str]:
        if isinstance(value, str):
            yield value
        elif isinstance(value, (bytes, bytearray)):
            yield bytes(value).decode('utf-8', 'replace')
        elif isinstance(value, dict):
            for k, v in value.items():
                yield from walk(k)
                yield from walk(v)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from walk(item)

    for column in table.columns:
        if pa.types.is_primitive(column.type) and not pa.types.is_binary(
            column.type
        ):
            continue
        for value in column.to_pylist():
            yield from walk(value)


def table_bytes(table: Any) -> bytes:
    r"""A table as one Arrow IPC stream, uncompressed."""
    import pyarrow as pa

    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def read_table(data: bytes) -> Any:
    import pyarrow as pa

    return pa.ipc.open_stream(io.BytesIO(data)).read_all()


# --- the writer ----------------------------------------------------------------------


class _Scrub:
    r"""Every text of a capture through the placeholders and the search,
    and each table into its file's bytes."""

    def __init__(self, environ: Mapping[str, str] | None) -> None:
        self.substitute = placeholders(environ)
        self.search = searcher(environ)
        self.files: dict[str, bytes] = {}

    def check(self, text: str, where: str) -> None:
        variable = self.search(text)
        if variable is not None:
            raise PrivateName(f'{where} holds the value of {variable}')

    def json(self, value: Any, where: str) -> Any:
        if isinstance(value, str):
            text = self.substitute(value)
            self.check(text, where)
            return text
        if isinstance(value, Mapping):
            out = {}
            for k, v in value.items():
                key = self.json(str(k), f'{where} (a key)')
                out[key] = self.json(v, f'{where}.{k}')
            return out
        if isinstance(value, (list, tuple)):
            return [self.json(v, f'{where}[{i}]') for i, v in enumerate(value)]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        raise CaptureError(f'{where}: a {type(value).__name__} is not JSON')

    def table(self, table: Any, where: str) -> str:
        if not hasattr(table, 'schema') or not hasattr(table, 'num_rows'):
            raise CaptureError(f'{where}: not an Arrow table')
        table = scrubbed_table(table, self.substitute)
        for text in _table_texts(table):
            self.check(text, where)
        data = table_bytes(table)
        name = f'tables/{hashlib.sha256(data).hexdigest()}.arrows'
        self.files[name] = data
        return name

    def optional_table(self, table: Any, where: str) -> str | None:
        return None if table is None else self.table(table, where)


def _integer(value: Any, where: str, *, null: bool = True) -> Any:
    if value is None and null:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise CaptureError(f'{where}: an integer, as recorded')
    return value


def document(
    capture: Mapping[str, Any], environ: Mapping[str, str] | None = None
) -> tuple[dict[str, Any], dict[str, bytes]]:
    r"""A capture as its ``capture.json`` document and the table files it
    names, in memory, every private name replaced and searched for;
    nothing written. A value nested past what Python's recursion reads is
    refused by name."""
    try:
        return _document(capture, environ)
    except RecursionError:
        raise CaptureError('a value nested too deep to write') from None


def _document(
    capture: Mapping[str, Any], environ: Mapping[str, str] | None
) -> tuple[dict[str, Any], dict[str, bytes]]:
    missing = [key for key in REQUIRED if key not in capture]
    unknown = sorted(set(capture) - set(REQUIRED) - set(OPTIONAL) - {'format'})
    if missing or unknown:
        raise CaptureError(
            f'a capture holds {", ".join(REQUIRED)} and may hold '
            f'{", ".join(OPTIONAL)}: missing {missing}, unknown {unknown}'
        )
    if capture['leg'] not in LEGS:
        raise CaptureError(f'no leg {capture["leg"]!r}; it is one of {LEGS}')
    for key in ('recording_commit', 'reference_commit'):
        if not re.fullmatch(r'[0-9a-f]{40}', str(capture[key])):
            raise CaptureError(f'{key}: not a full SHA')
    session = capture['session']
    if not isinstance(session, Mapping) or not isinstance(
        session.get('time_zone'), str
    ):
        raise CaptureError("a capture's session names its time zone")
    if session.get('null_order') not in ('nulls_first', 'nulls_last'):
        raise CaptureError(
            "a session's null_order is nulls_first or nulls_last"
        )
    scrub = _Scrub(environ)
    out: dict[str, Any] = {'format': FORMAT}
    for key in REQUIRED:
        if key != 'tables':
            out[key] = scrub.json(capture[key], key)
    tables = capture['tables']
    if not isinstance(tables, Mapping):
        raise CaptureError('tables: a table by name')
    out['tables'] = [
        {
            'name': scrub.json(name, f'tables[{name!r}]'),
            'data': scrub.table(table, f'tables[{name!r}]'),
        }
        for name, table in tables.items()
    ]
    out['derived'] = []
    for i, entry in enumerate(capture.get('derived', [])):
        where = f'derived[{i}]'
        out['derived'].append(
            {
                'view': scrub.json(entry['view'], f'{where}.view'),
                'table': scrub.json(entry['table'], f'{where}.table'),
                'columns': scrub.json(entry['columns'], f'{where}.columns'),
                'filter': scrub.json(entry.get('filter'), f'{where}.filter'),
                'refused': scrub.json(
                    list(entry.get('refused', [])), f'{where}.refused'
                ),
                'data': scrub.optional_table(entry.get('data'), where),
                **{
                    key: scrub.json(entry[key], f'{where}.{key}')
                    for key in ('statement', 'sql', 'error')
                    if entry.get(key) is not None
                },
            }
        )
    out['draws'] = []
    for i, draw in enumerate(capture.get('draws', [])):
        where = f'draws[{i}]'
        fraction = draw.get('fraction')
        if not isinstance(fraction, str):
            raise CaptureError(f'{where}: the fraction is text, as written')
        entry = {
            'table': scrub.json(draw['table'], f'{where}.table'),
            'seed': _integer(draw.get('seed'), f'{where}.seed'),
            'fraction': scrub.json(fraction, f'{where}.fraction'),
            'rows': scrub.table(draw['rows'], f'{where}.rows'),
        }
        if draw.get('statement') is not None:
            entry['statement'] = scrub.json(draw['statement'], where)
        out['draws'].append(entry)
    out['statements'] = []
    for i, statement in enumerate(capture.get('statements', [])):
        where = f'statements[{i}]'
        entry = {
            'text': scrub.json(statement['text'], f'{where}.text'),
            'parameters': scrub.json(
                statement.get('parameters'), f'{where}.parameters'
            ),
            'result': scrub.optional_table(
                statement.get('result'), f'{where}.result'
            ),
        }
        if statement.get('error') is not None:
            entry['error'] = scrub.json(statement['error'], f'{where}.error')
        out['statements'].append(entry)
    out['loader'] = scrub.json(list(capture.get('loader', [])), 'loader')
    return out, scrub.files


def dumps(found: Mapping[str, Any]) -> str:
    return json.dumps(found, indent=1, ensure_ascii=False) + '\n'


def write(
    path: Path,
    capture: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    r"""Writes ``capture`` as the directory ``path`` (which must not exist):
    first every text through the placeholders and the whole capture
    searched for private names (:class:`PrivateName`), then the files, into
    a directory beside ``path`` renamed into place, so a failed write
    leaves nothing at ``path``."""
    path = Path(path)
    found, files = document(capture, environ)
    if path.exists():
        raise CaptureError(f'{path}: a capture is written once')
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.capture-', dir=path.parent))
    try:
        for name, data in files.items():
            (staging / name).parent.mkdir(parents=True, exist_ok=True)
            (staging / name).write_bytes(data)
        (staging / DOCUMENT).write_text(dumps(found), encoding='utf-8')
        staging.rename(path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return path


# --- the reader and converter --------------------------------------------------------

#: One function per older version, each returning the document in the
#: next version: a format change adds one, and every older capture is read
#: through the chain (choice 2: converted, never re-recorded).
CONVERTERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def upgrade(found: dict[str, Any]) -> dict[str, Any]:
    r"""A capture document in the current version, converted from any
    version in :data:`VERSIONS`; an unknown version is refused by name."""
    version = found.get('format')
    seen = set()
    while version != FORMAT:
        if version not in VERSIONS or version not in CONVERTERS:
            raise CaptureError(f'the capture format {version!r} is unknown')
        if version in seen:
            raise CaptureError(f'the converters loop at {version!r}')
        seen.add(version)
        found = CONVERTERS[version](copy.deepcopy(found))
        version = found.get('format')
    return found


def read_document(path: Path) -> dict[str, Any]:
    r"""The ``capture.json`` of the capture at ``path``, in the current
    version, its tables still named by file."""
    path = Path(path)
    source = path / DOCUMENT if path.is_dir() else path
    try:
        found = json.loads(source.read_text(encoding='utf-8'))
    except (OSError, ValueError, RecursionError) as error:
        raise CaptureError(
            f'{source}: not a capture ({type(error).__name__})'
        ) from None
    if not isinstance(found, dict):
        raise CaptureError(f'{source}: not a capture')
    return upgrade(found)


def table(directory: Path, name: str) -> Any:
    r"""A table a capture in ``directory`` names, its content checked
    against its name."""
    if not re.fullmatch(r'tables/[0-9a-f]{64}\.arrows', str(name)):
        raise CaptureError(f'{name!r} is not a capture table')
    data = (Path(directory) / name).read_bytes()
    if f'tables/{hashlib.sha256(data).hexdigest()}.arrows' != name:
        raise CaptureError(f'{name}: its content is not its name')
    return read_table(data)


def read(path: Path) -> dict[str, Any]:
    r"""The capture at ``path`` (its directory), in the current version and
    the shape :func:`write` takes, ``format`` added: each table read."""
    path = Path(path)
    found = read_document(path)
    directory = path if path.is_dir() else path.parent

    def load(name: str | None) -> Any:
        return None if name is None else table(directory, name)

    back = {key: found[key] for key in REQUIRED if key != 'tables'}
    back['format'] = found['format']
    back['tables'] = {
        entry['name']: load(entry['data']) for entry in found['tables']
    }
    back['derived'] = [
        {**entry, 'data': load(entry['data'])} for entry in found['derived']
    ]
    back['draws'] = [
        {**draw, 'rows': load(draw['rows'])} for draw in found['draws']
    ]
    back['statements'] = [
        {**statement, 'result': load(statement['result'])}
        for statement in found['statements']
    ]
    back['loader'] = found['loader']
    return back


def convert(directory: Path) -> list[Path]:
    r"""Rewrites each capture under ``directory`` in an older version in
    the current one; returns the documents rewritten."""
    changed = []
    for path in sorted(Path(directory).rglob(DOCUMENT)):
        found = json.loads(path.read_text(encoding='utf-8'))
        if found.get('format') == FORMAT:
            continue
        path.write_text(dumps(upgrade(found)), encoding='utf-8')
        changed.append(path)
    return changed


#: A leg directory's list of its captures.
MANIFEST = 'manifest.json'


def write_manifest(directory: Path) -> Path:
    r"""Writes ``directory/manifest.json``: every capture under it (each
    ``*/capture.json``), all of one recording and one reference commit,
    the versions of each leg, and each case's input hash and directory,
    keyed ``<case>@<leg>``. A capture of another commit, two of one key, or
    two versions of one leg are refused by name."""
    directory = Path(directory)
    commits: set[tuple[str, str]] = set()
    legs: dict[str, Any] = {}
    cases: dict[str, Any] = {}
    for path in sorted(directory.rglob(DOCUMENT)):
        found = read_document(path.parent)
        commits.add((found['recording_commit'], found['reference_commit']))
        leg = found['leg']
        if legs.setdefault(leg, found['versions']) != found['versions']:
            raise CaptureError(f'{path}: {leg} recorded with two versions')
        key = f'{found["case"]}@{leg}'
        if key in cases:
            raise CaptureError(f'{path}: {key} captured twice')
        cases[key] = {
            'input_hash': found['input_hash'],
            'capture': path.parent.relative_to(directory).as_posix(),
        }
    if len(commits) != 1:
        raise CaptureError(
            f'{directory}: captures of {len(commits)} recordings, where a '
            'manifest lists one'
        )
    ((recording, reference),) = commits
    path = directory / MANIFEST
    path.write_text(
        dumps(
            {
                'format': FORMAT,
                'recording_commit': recording,
                'reference_commit': reference,
                'legs': legs,
                'cases': cases,
            }
        ),
        encoding='utf-8',
    )
    return path


# --- the recording mode's lines ------------------------------------------------------


def _parameter(value: Any) -> Any:
    r"""A parameter as the recording mode logged it (``{"type", "value"}``,
    its type and ``repr``), as the JSON value the old client passed: a
    built-in value read back from its ``repr``, anything else kept as
    logged."""
    import ast

    if not isinstance(value, Mapping) or set(value) - {'type', 'value', 'name'}:
        return value
    read: Any = dict(value)
    if str(value.get('type', '')).startswith('builtins.'):
        try:
            found = ast.literal_eval(str(value['value']))
        except (ValueError, SyntaxError):
            found = None
            literal = False
        else:
            literal = found is None or isinstance(
                found, (bool, int, float, str)
            )
        if literal:
            read = found
    # A named binding keeps its name (``%(x)s``, ``:x``): it is no
    # positional one.
    if 'name' in value:
        return {'name': str(value['name']), 'value': read}
    return read


def _ipc_text(text: str) -> Any:
    r"""A table the recording mode wrote as the base64 text of an IPC
    stream."""
    import base64

    import pyarrow as pa

    return pa.ipc.open_stream(io.BytesIO(base64.b64decode(text))).read_all()


#: The recording mode's line format (``run_python.py --record``, T3.2),
#: which :func:`from_recording` reads; its first lines carried the
#: capture's own label before the recording took a name of its own.
RECORDING_FORMATS = ('sdm-record/1', 'sdm-capture/1')


def from_recording(
    line: Mapping[str, Any], null_order: str | None = None
) -> dict[str, Any]:
    r"""One line of ``run_python.py --record`` (the recording mode, T3.2:
    ``capture_line``, its private names already placeheld and every table
    the base64 text of an IPC stream) as a capture :func:`write` takes:
    each statement's first result its result, each draw its statement's
    rows, each view-derived relation's warehouse-evaluated table (or the
    statement's error, its relation then served by exact replay alone),
    the session's zone, and the leg's NULL order (:data:`NULL_ORDER`)."""
    if line.get('format') not in RECORDING_FORMATS:
        raise CaptureError(
            f'a recording line of format {line.get("format")!r} is not one of '
            f'{RECORDING_FORMATS}'
        )
    import pyarrow as pa

    leg = line['leg']
    case = str(line['case']).rpartition('@')[0] or str(line['case'])
    recording = line.get('recording') or {}

    def table_of(value: Any) -> Any:
        return None if value is None else _ipc_text(value)

    statements = []
    for position, statement in enumerate(line.get('statements', [])):
        reads = statement.get('results') or []
        if any(read.get('arrow') is None for read in reads):
            raise CaptureError(
                f'{case}@{leg}: statement {position} has a read whose rows '
                'the recording kept no table of; nothing is written'
            )
        result = None
        if reads:
            # Each read of the statement's cursor, in order (a fetchmany,
            # then a fetchall), is part of its one result: every row kept.
            parts = [table_of(read['arrow']) for read in reads]
            if any(not part.schema.equals(parts[0].schema) for part in parts):
                raise CaptureError(
                    f'{case}@{leg}: statement {position} read with two '
                    'schemas; nothing is written'
                )
            result = pa.concat_tables(parts) if len(parts) > 1 else parts[0]
        entry: dict[str, Any] = {
            'text': statement.get('sql') or '',
            'parameters': [
                _parameter(value) for value in statement.get('parameters') or []
            ],
            'result': result,
        }
        if statement.get('error') is not None:
            entry['error'] = statement['error']
        statements.append(entry)
    draws = []
    for draw in line.get('draws', []):
        rows = statements[draw['statement']]['result']
        if rows is None:
            raise CaptureError(
                f'{case}@{leg}: a draw whose statement kept no rows'
            )
        draws.append(
            {
                'table': draw['table'],
                'seed': None if draw.get('seed') is None else int(draw['seed']),
                'fraction': draw['fraction']
                + (' ROWS' if draw.get('kind') == 'sample_rows' else ''),
                'rows': rows,
                'statement': statements[draw['statement']]['text'],
            }
        )
    derived = []
    for i, relation in enumerate(line.get('relations', [])):
        # What the recording mode keeps of a view-derived relation: the
        # statement that read it, its SQL whole, and the warehouse's table
        # of it, or the error the warehouse refused it with (served by
        # exact replay alone, B2). Its columns and filter are the SQL's.
        derived.append(
            {
                'view': '',
                'table': f'derived_{i}',
                'columns': [],
                'filter': None,
                'statement': relation.get('statement'),
                'sql': relation.get('sql'),
                'error': relation.get('error'),
                'refused': [] if relation.get('table') else ['(all)'],
                'data': table_of(relation.get('table')),
            }
        )
    zone = line.get('zone')
    if not zone:
        raise CaptureError(
            f'{case}@{leg}: the recording names no session time zone, which '
            'the offline run reads (B5)'
        )
    return {
        'case': case,
        'leg': leg,
        'recording_commit': recording.get('commit'),
        'reference_commit': recording.get('reference'),
        'versions': line.get('versions') or {},
        'input_hash': line.get('input'),
        'session': {
            'time_zone': zone,
            'null_order': null_order or NULL_ORDER[leg],
        },
        'reference': line.get('observation') or {},
        'tables': {
            entry['name']: table_of(entry['arrow'])
            for entry in line.get('tables', [])
        },
        'derived': derived,
        'draws': draws,
        'statements': statements,
    }


def import_recording(
    path: Path, out: Path, environ: Mapping[str, str] | None = None
) -> list[Path]:
    r"""Every line of the recording mode's file ``path`` written as a
    capture under ``out/<leg>/<case>/`` (:func:`write`: placeholders and
    the private-name search again, so a name the recording missed still
    fails here), then each leg's manifest; returns the manifests."""
    legs: set[Path] = set()
    for text in Path(path).read_text(encoding='utf-8').splitlines():
        if not text.strip():
            continue
        capture = from_recording(json.loads(text))
        directory = Path(out) / capture['leg']
        write(directory / capture['case'], capture, environ=environ)
        legs.add(directory)
    return [write_manifest(directory) for directory in sorted(legs)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    converting = commands.add_parser(
        'convert', help='rewrite older captures in the current version'
    )
    converting.add_argument('directory', type=Path)
    scanning = commands.add_parser(
        'scan', help='search captures for private names of the environment'
    )
    scanning.add_argument('directory', type=Path)
    listing = commands.add_parser(
        'manifest', help="write a leg directory's manifest.json"
    )
    listing.add_argument('directory', type=Path)
    importing = commands.add_parser(
        'import',
        help="write each line of run_python.py --record's file as a capture",
    )
    importing.add_argument('recording', type=Path)
    importing.add_argument('out', type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == 'import':
        for path in import_recording(arguments.recording, arguments.out):
            print(f'wrote {path}')
        return 0
    if arguments.command == 'manifest':
        print(f'wrote {write_manifest(arguments.directory)}')
        return 0
    if arguments.command == 'convert':
        for path in convert(arguments.directory):
            print(f'converted {path}')
        return 0
    search = searcher()
    problems = 0
    for path in sorted(arguments.directory.rglob('*')):
        if path.suffix == '.json':
            texts: Iterable[str] = [path.read_text(encoding='utf-8')]
        elif path.suffix == '.arrows':
            texts = _table_texts(read_table(path.read_bytes()))
        else:
            continue
        for text in texts:
            variable = search(text)
            if variable is not None:
                print(f'{path}: holds the value of {variable}')
                problems += 1
                break
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
