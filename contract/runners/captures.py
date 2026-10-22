# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The capture writer and reader (Phase 3 design, choice 2; T3.3).

A capture is what the old client did and got on one warehouse leg for one
case, recorded live in C-1 (or E-0) by ``run_python.py``'s recording mode
(T3.2), and what the stand-in (``sdm-testkit``'s ``standin.rs``) answers
the core's requests from offline. One leg's captures are a directory,
``contract/captures/<leg>/``:

    <case id>.json          one case's capture (:data:`FORMAT`)
    tables/<sha256>.arrows  each distinct Arrow table once, an IPC stream

A case's file holds, beside its format, case, leg and the hash of the
case's input:

- ``recording_commit`` and ``reference_commit``: the commit whose harness
  recorded it and the reference's commit;
- ``runtime``: the leg's runtime, Python, pandas, pyarrow and driver
  versions, as the recording read them;
- ``session``: the session's time zone and NULL order, which the stand-in
  sets on its local engine (B5);
- ``reference``: the old client's reference line, the observation
  Phase 2's lines use;
- ``loader``: the loader's notes, each constraint the engine did not store;
- ``tables``: every table of the case as the warehouse returned it through
  the leg's driver, by name;
- ``views``: one entry per relation the old client derived from a view:
  its columns and filter, the warehouse-evaluated table, or ``null`` and
  the columns the warehouse refused;
- ``draws``: every random draw: the table, the seed, the fraction as
  written, and the rows picked;
- ``statements``: every statement sent, in order, with its parameters and
  its result (a table) or error.

Private names never reach a file. Every text in the capture, the tables'
cells, field names and metadata included, has each private name the
recording knows (the catalog, schema, host, user, and the view names)
replaced by its placeholder (``{catalog}``, ``{schema}``, ...) before
anything is written, in each case form a warehouse folds it to
(``MY_DB`` becomes ``{CATALOG}``); then the whole capture is searched,
case-insensitively, for every private name, both those placeholders
stand for and every value of :data:`PRIVATE_VARIABLES` in the
environment, and the write fails by the variable's name, never its
value (:class:`PrivateName`), with no file written.

The format is versioned. A later version is read through
:data:`CONVERTERS`, one function per older version, so an old capture is
converted, never re-recorded (choice 2): ``captures.py convert DIR``
rewrites every file of DIR in the current version.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
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
#: default (Spark puts NULLs first; Snowflake and Postgres last), written
#: in a capture's session unless the recording read the session's own.
NULL_ORDER = {
    'databricks-sql': 'nulls_first',
    'databricks-spark': 'nulls_first',
    'snowflake': 'nulls_last',
    'lakebase': 'nulls_last',
}

#: Where the captures live, from the repository's root.
ROOT = Path(__file__).resolve().parents[2]
CAPTURES = ROOT / 'contract' / 'captures'

#: The environment variables naming a private resource or credential: each
#: value present (and each part of a URL's host) is searched for in a
#: capture before it is written. Any variable named ``SDM_PRIVATE_*`` or
#: ``SDM_LIVE_*`` is searched for too, so a live run names a resource of
#: its own without changing this list.
PRIVATE_VARIABLES = (
    'DATABRICKS_HOST',
    'DATABRICKS_SERVER_HOSTNAME',
    'DATABRICKS_HTTP_PATH',
    'DATABRICKS_WAREHOUSE_ID',
    'DATABRICKS_CLIENT_ID',
    'DATABRICKS_TOKEN',
    'DATABRICKS_CONFIG_PROFILE',
    'DATABRICKS_USER',
    'SNOWFLAKE_URL',
    'SNOWFLAKE_ACCOUNT',
    'SNOWFLAKE_USERNAME',
    'SNOWFLAKE_USER',
    'SNOWFLAKE_ACCESS_TOKEN',
    'SNOWFLAKE_PASSWORD',
    'SNOWFLAKE_WAREHOUSE',
    'SNOWFLAKE_DATABASE',
    'SNOWFLAKE_SCHEMA',
    'SNOWFLAKE_ROLE',
    'PGHOST',
    'PGUSER',
    'PGPASSWORD',
    'PGDATABASE',
)
PRIVATE_PREFIXES = ('SDM_PRIVATE_', 'SDM_LIVE_')

#: The shortest value searched for: a shorter one (a role ``dev``) would
#: match ordinary words; a placeholder's own value is searched for at any
#: length.
SHORTEST = 4

#: A placeholder's name: lower-case letters, digits and underscores.
_PLACEHOLDER = re.compile(r'[a-z][a-z0-9_]*')


class CaptureError(Exception):
    r"""A capture that cannot be written or read as this format says."""


class PrivateName(CaptureError):
    r"""A private name left in a capture: named by where it is and by the
    placeholder or variable it is the value of, never by the value."""


# --- private names -------------------------------------------------------------------


def private_names(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    r"""Every private value of the environment, by the variable holding
    it: each of :data:`PRIVATE_VARIABLES` and every ``SDM_PRIVATE_*`` or
    ``SDM_LIVE_*`` variable that is set, and a URL's host and each part of
    it before the domain, at least :data:`SHORTEST` characters long."""
    environ = os.environ if environ is None else environ
    found: dict[str, str] = {}
    for name, value in sorted(environ.items()):
        if name not in PRIVATE_VARIABLES and not name.startswith(
            PRIVATE_PREFIXES
        ):
            continue
        value = value.strip()
        if not value:
            continue
        candidates = [value]
        host = re.match(r'^[a-z][a-z0-9+.-]*://([^/:?#]+)', value, re.I)
        if host:
            candidates.append(host.group(1))
            candidates.append(host.group(1).split('.')[0])
        for i, candidate in enumerate(candidates):
            if len(candidate) >= SHORTEST:
                found[name if i == 0 else f'{name} (host part {i})'] = candidate
    return found


class Placeholders:
    r"""Private names and the placeholders written for them: ``{name}``
    for the value as given, and its upper- and lower-case forms for the
    value's upper- and lower-case forms, so a name a warehouse folds keeps
    its fold offline."""

    def __init__(self, names: Mapping[str, str]) -> None:
        for name, value in names.items():
            if not _PLACEHOLDER.fullmatch(name):
                raise CaptureError(f'the placeholder {name!r} is not a name')
            if not value:
                raise CaptureError(f'the placeholder {{{name}}} has no value')
        self.names = dict(names)
        pairs: dict[str, str] = {}
        for name, value in names.items():
            token = '{' + name + '}'
            pairs.setdefault(value, token)
            pairs.setdefault(value.upper(), token.upper())
            pairs.setdefault(value.lower(), token.lower())
        # Longest first, so a name holding another is replaced whole.
        self._pairs = sorted(pairs.items(), key=lambda p: (-len(p[0]), p[0]))
        self._pattern = (
            re.compile('|'.join(re.escape(value) for value, _ in self._pairs))
            if self._pairs
            else None
        )
        self._tokens = dict(self._pairs)

    def __call__(self, text: str) -> str:
        if self._pattern is None:
            return text
        return self._pattern.sub(lambda m: self._tokens[m.group(0)], text)


def _searcher(private: Mapping[str, str]) -> Callable[[str], str | None]:
    r"""A function naming the first private value a text holds, by its
    label, case-insensitively."""
    labels = {value.casefold(): label for label, value in private.items()}
    if not labels:
        return lambda text: None
    pattern = re.compile(
        '|'.join(re.escape(v) for v in sorted(labels, key=len, reverse=True))
    )

    def search(text: str) -> str | None:
        found = pattern.search(text.casefold())
        return labels[found.group(0)] if found else None

    return search


# --- Arrow ---------------------------------------------------------------------------


def _texts(array: Any, change: Callable[[str], str]) -> Any:
    r"""``array`` with every text in it, at any depth, through ``change``."""
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
            kind.field(i).with_name(change(kind.field(i).name))
            for i in range(kind.num_fields)
        ]
        return pa.StructArray.from_arrays(
            children, fields=fields, mask=array.is_null()
        )
    if t.is_map(kind):
        return _map_texts(array, change)
    if t.is_list(kind) or t.is_large_list(kind):
        values = _texts(array.values, change)
        cls = pa.ListArray if t.is_list(kind) else pa.LargeListArray
        out = cls.from_arrays(array.offsets, values, mask=array.is_null())
        return out
    if t.is_fixed_size_list(kind):
        values = _texts(array.values, change)
        return pa.FixedSizeListArray.from_arrays(
            values, kind.list_size, mask=array.is_null()
        )
    return array


def _map_texts(array: Any, change: Callable[[str], str]) -> Any:
    import pyarrow as pa

    keys = _texts(array.keys, change)
    items = _texts(array.items, change)
    return pa.MapArray.from_arrays(
        array.offsets, keys, items, mask=array.is_null()
    )


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


def scrubbed_table(table: Any, change: Callable[[str], str]) -> Any:
    r"""``table`` with every text through ``change``: its cells at any
    depth, its field names and its metadata."""
    import pyarrow as pa

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
        fields.append(
            pa.field(
                change(field.name),
                array.type,
                field.nullable,
                _metadata(field.metadata, change),
            )
        )
    schema = pa.schema(
        fields, metadata=_metadata(table.schema.metadata, change)
    )
    return pa.Table.from_arrays(columns, schema=schema)


def _table_texts(table: Any) -> Iterable[str]:
    r"""Every text a table holds: field names, metadata, and each text or
    byte cell (bytes read as UTF-8, invalid bytes replaced)."""
    import pyarrow as pa

    for field in table.schema:
        yield field.name
        for k, v in (field.metadata or {}).items():
            yield k.decode('utf-8', 'replace')
            yield v.decode('utf-8', 'replace')
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


def rows_table(columns: list[str], rows: list[Any]) -> Any:
    r"""A statement's rows read through ``fetchall`` (a catalog read), as
    an Arrow table: each column's values as pyarrow infers their type."""
    import pyarrow as pa

    return pa.Table.from_arrays(
        [pa.array([row[i] for row in rows]) for i in range(len(columns))],
        names=list(columns),
    )


# --- the writer ----------------------------------------------------------------------


class CaptureWriter:
    r"""Writes one leg's captures under ``root/<leg>/``: every text through
    the placeholders, then the whole capture searched for each private
    name, and only then the files (:class:`PrivateName` otherwise)."""

    def __init__(
        self,
        root: Path,
        leg: str,
        *,
        recording_commit: str,
        reference_commit: str,
        runtime: Mapping[str, Any],
        session: Mapping[str, Any],
        placeholders: Mapping[str, str],
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if leg not in LEGS:
            raise CaptureError(f'no leg {leg!r}; it is one of {LEGS}')
        for name, commit in (
            ('recording', recording_commit),
            ('reference', reference_commit),
        ):
            if not re.fullmatch(r'[0-9a-f]{40}', commit):
                raise CaptureError(f'the {name} commit is not a full SHA')
        session = {'null_order': NULL_ORDER[leg], **session}
        if session['null_order'] not in ('nulls_first', 'nulls_last'):
            raise CaptureError('a NULL order is nulls_first or nulls_last')
        if not isinstance(session.get('time_zone'), str):
            raise CaptureError("a capture's session names its time zone")
        self.directory = Path(root) / leg
        self.leg = leg
        self.substitute = Placeholders(placeholders)
        private = {
            f'the value of {{{name}}}': value
            for name, value in placeholders.items()
        }
        private.update(
            (f'the value of {name}', value)
            for name, value in private_names(environ).items()
        )
        self.search = _searcher(private)
        self.head = {
            'format': FORMAT,
            'leg': leg,
            'recording_commit': recording_commit,
            'reference_commit': reference_commit,
            'runtime': self._json(dict(runtime), 'runtime'),
            'session': self._json(dict(session), 'session'),
        }

    # Every JSON value through the placeholders, then searched.
    def _json(self, value: Any, where: str) -> Any:
        if isinstance(value, str):
            text = self.substitute(value)
            self._check(text, where)
            return text
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                key = self._json(str(k), f'{where} (a key)')
                out[key] = self._json(v, f'{where}.{k}')
            return out
        if isinstance(value, (list, tuple)):
            return [self._json(v, f'{where}[{i}]') for i, v in enumerate(value)]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        raise CaptureError(f'{where}: a {type(value).__name__} is not JSON')

    def _check(self, text: str, where: str) -> None:
        label = self.search(text)
        if label is not None:
            raise PrivateName(f'{where} holds {label}')

    def _table(self, table: Any, where: str, files: dict[str, bytes]) -> str:
        table = scrubbed_table(table, self.substitute)
        for text in _table_texts(table):
            self._check(text, where)
        data = table_bytes(table)
        name = f'tables/{hashlib.sha256(data).hexdigest()}.arrows'
        files[name] = data
        return name

    def document(
        self, case: str, **parts: Any
    ) -> tuple[dict[str, Any], dict[str, bytes]]:
        r"""One case's capture and the table files it names, in memory,
        every private name replaced and searched for; nothing written. A
        value nested past what Python's recursion reads is refused by
        name."""
        try:
            return self._document(case, **parts)
        except RecursionError:
            raise CaptureError(
                f'{case}: a value nested too deep to write'
            ) from None

    def _document(
        self,
        case: str,
        *,
        input_hash: str,
        reference: Mapping[str, Any],
        tables: Mapping[str, Any],
        views: Iterable[Mapping[str, Any]] = (),
        draws: Iterable[Mapping[str, Any]] = (),
        statements: Iterable[Mapping[str, Any]] = (),
        loader: Iterable[str] = (),
    ) -> tuple[dict[str, Any], dict[str, bytes]]:
        files: dict[str, bytes] = {}
        document = dict(self.head)
        document['case'] = self._json(case, 'case')
        document['input_hash'] = self._json(input_hash, 'input_hash')
        document['reference'] = self._json(dict(reference), 'reference')
        document['loader'] = self._json(list(loader), 'loader')
        document['tables'] = [
            {
                'name': self._json(name, f'tables[{name!r}]'),
                'data': self._table(table, f'tables[{name!r}]', files),
            }
            for name, table in tables.items()
        ]
        document['views'] = []
        for i, view in enumerate(views):
            where = f'views[{i}]'
            entry = {
                'view': self._json(view['view'], f'{where}.view'),
                'relation': self._json(view['relation'], f'{where}.relation'),
                'columns': self._json(
                    list(view['columns']), f'{where}.columns'
                ),
                'filter': self._json(view.get('filter'), f'{where}.filter'),
                'refused': self._json(list(view.get('refused', ())), where),
                'data': None,
            }
            if view.get('table') is not None:
                entry['data'] = self._table(view['table'], where, files)
            document['views'].append(entry)
        document['draws'] = []
        for i, draw in enumerate(draws):
            where = f'draws[{i}]'
            seed = draw.get('seed')
            if seed is not None and (
                not isinstance(seed, int) or isinstance(seed, bool)
            ):
                raise CaptureError(f'{where}: a seed is an integer or null')
            fraction = draw.get('fraction')
            if fraction is not None and not isinstance(fraction, str):
                raise CaptureError(f'{where}: a fraction is kept as written')
            rows = draw.get('rows')
            if rows is not None and (
                not isinstance(rows, int) or isinstance(rows, bool)
            ):
                raise CaptureError(f'{where}: a row count is an integer')
            document['draws'].append(
                {
                    'table': self._json(draw['table'], f'{where}.table'),
                    'seed': seed,
                    'fraction': fraction,
                    'rows': rows,
                    'statement': self._json(
                        draw.get('statement'), f'{where}.statement'
                    ),
                    'data': self._table(draw['picked'], where, files),
                }
            )
        document['statements'] = []
        for i, statement in enumerate(statements):
            where = f'statements[{i}]'
            entry = {
                'sql': self._json(statement['sql'], f'{where}.sql'),
                'parameters': self._json(
                    statement.get('parameters'), f'{where}.parameters'
                ),
                'data': None,
                'error': self._json(statement.get('error'), f'{where}.error'),
            }
            if statement.get('result') is not None:
                entry['data'] = self._table(statement['result'], where, files)
            document['statements'].append(entry)
        return document, files

    def write(self, case: str, **parts: Any) -> Path:
        r"""Writes one case's capture (:meth:`document`'s parts), its table
        files first, only once all of it is free of private names."""
        document, files = self.document(case, **parts)
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', case):
            raise CaptureError(f'{case!r} is not a case id a file can be named')
        for name, data in files.items():
            path = self.directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(data)
        path = self.directory / f'{case}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(document))
        return path


def dumps(document: Mapping[str, Any]) -> str:
    return json.dumps(document, indent=1, ensure_ascii=False) + '\n'


# --- the reader and converter --------------------------------------------------------

#: One function per older version, each returning the document in the
#: next version: a format change adds one, and every older capture is read
#: through the chain (choice 2: converted, never re-recorded).
CONVERTERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def upgrade(document: dict[str, Any]) -> dict[str, Any]:
    r"""A capture in the current version, converted from any version in
    :data:`VERSIONS`; an unknown version is refused by name."""
    version = document.get('format')
    seen = set()
    while version != FORMAT:
        if version not in VERSIONS or version not in CONVERTERS:
            raise CaptureError(f'the capture format {version!r} is unknown')
        if version in seen:
            raise CaptureError(f'the converters loop at {version!r}')
        seen.add(version)
        document = CONVERTERS[version](document)
        version = document.get('format')
    return document


def read(path: Path) -> dict[str, Any]:
    r"""One case's capture, in the current version."""
    try:
        document = json.loads(Path(path).read_text())
    except (OSError, ValueError, RecursionError) as error:
        raise CaptureError(
            f'{path}: not a capture ({type(error).__name__})'
        ) from None
    if not isinstance(document, dict):
        raise CaptureError(f'{path}: not a capture')
    return upgrade(document)


def table(directory: Path, name: str) -> Any:
    r"""A table a capture in ``directory`` names, its content checked
    against its name."""
    if not re.fullmatch(r'tables/[0-9a-f]{64}\.arrows', name):
        raise CaptureError(f'{name!r} is not a capture table')
    data = (Path(directory) / name).read_bytes()
    if f'tables/{hashlib.sha256(data).hexdigest()}.arrows' != name:
        raise CaptureError(f'{name}: its content is not its name')
    return read_table(data)


def convert(directory: Path) -> list[Path]:
    r"""Rewrites each capture under ``directory`` in an older version in
    the current one; returns the files rewritten."""
    changed = []
    for path in sorted(Path(directory).rglob('*.json')):
        document = json.loads(path.read_text())
        if document.get('format') == FORMAT:
            continue
        path.write_text(dumps(upgrade(document)))
        changed.append(path)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    converting = commands.add_parser(
        'convert', help='rewrite older captures in the current version'
    )
    converting.add_argument('directory', type=Path)
    checking = commands.add_parser(
        'scan', help='search captures for private names of the environment'
    )
    checking.add_argument('directory', type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == 'convert':
        for path in convert(arguments.directory):
            print(f'converted {path}')
        return 0
    search = _searcher(
        {f'the value of {k}': v for k, v in private_names().items()}
    )
    problems = 0
    for path in sorted(arguments.directory.rglob('*')):
        if path.suffix == '.json':
            texts: Iterable[str] = [path.read_text()]
        elif path.suffix == '.arrows':
            texts = _table_texts(read_table(path.read_bytes()))
        else:
            continue
        for text in texts:
            label = search(text)
            if label is not None:
                print(f'{path}: holds {label}')
                problems += 1
                break
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
