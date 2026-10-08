# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Removing credentials and the names of private places from recordings.

A live recording can carry a token in an error message, a workspace's host
in a URL, or a password in a connection string. Before one is stored, every
string in it, JSON and Arrow alike, is scrubbed of:

* the values of the recording environment's credential variables
  (:data:`SECRET_VARIABLES`), wherever they appear;
* anything shaped like a secret, a person or a private place
  (:data:`PATTERNS`): tokens, keys, passwords in URLs, warehouse and storage
  hosts, bucket and container URLs, email addresses;
* any value kept under a key that names a secret (:data:`SECRET_KEYS`): a
  JSON key, an Arrow column's name, or the first cell of a key-value row
  such as ``DESCRIBE TABLE EXTENDED`` returns (``Owner``, ``Location``).

Arrow strings are scrubbed wherever they are: in columns of any nesting
(lists, structs, maps), in dictionary-encoded columns, in field names and in
schema and field metadata.

Each is replaced with ``<redacted:WHAT>``, so a reader can see that something
was there and what it was. ``scan`` reports what ``scrub`` would remove.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

#: Environment variables whose values are secret wherever they appear.
SECRET_VARIABLES = (
    'DATABRICKS_SERVER_HOSTNAME',
    'DATABRICKS_TOKEN',
    'DATABRICKS_HTTP_PATH',
    'DATABRICKS_WAREHOUSE_ID',
    'DATABRICKS_HOST',
    'DATABRICKS_CLIENT_SECRET',
    'SNOWFLAKE_ACCOUNT',
    'SNOWFLAKE_USER',
    'SNOWFLAKE_PASSWORD',
    'SNOWFLAKE_TOKEN',
    # A personal access token's file names them so; the recorder's
    # SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER and SNOWFLAKE_TOKEN are mapped from them.
    'SNOWFLAKE_URL',
    'SNOWFLAKE_USERNAME',
    'SNOWFLAKE_ACCESS_TOKEN',
    'PGHOST',
    'PGUSER',
    'PGPASSWORD',
)

#: Shapes that are secret, or name a private place, on sight.
PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        'jwt',
        re.compile(
            r'eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}'
        ),
        '',
    ),
    ('databricks-token', re.compile(r'\bdapi[0-9a-f]{32}(?:-\d+)?\b'), ''),
    ('aws-access-key', re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'), ''),
    (
        'bearer-token',
        re.compile(r'(?i)\b(bearer\s+)(?!<redacted)[A-Za-z0-9._~+/=-]{8,}'),
        r'\1',
    ),
    (
        'password',
        re.compile(
            r'(\b[a-z][a-z0-9+.-]*://(?:<redacted:[a-z0-9_-]+>|[^\s:/@<>]+):)'
            r'(?!<redacted)[^\s@/]+(@)'
        ),
        r'\1|\2',
    ),
    (
        'host',
        re.compile(
            r'\b[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?'
            r'(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*'
            r'\.(?:azuredatabricks\.net|databricks\.com|databricksapps\.com|'
            r'snowflakecomputing\.com|privatelink\.snowflakecomputing\.com)\b'
        ),
        '',
    ),
    (
        'storage-host',
        re.compile(
            r'\b[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?'
            r'(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*'
            r'\.(?:(?:blob|dfs|file|queue|table)\.core\.windows\.net|'
            r'amazonaws\.com|storage\.googleapis\.com)\b'
        ),
        '',
    ),
    (
        'storage-url',
        re.compile(
            r'(?i)\b((?:s3|s3a|s3n|gs|gcs|wasbs?|abfss?|r2)://)'
            r'(?!<redacted)[^\s\'"`<>]+'
        ),
        r'\1',
    ),
    ('workspace-id', re.compile(r'([?&]o=)\d{6,}'), r'\1'),
    ('warehouse', re.compile(r'(/sql/1\.0/warehouses/)[0-9a-f]{8,}'), r'\1'),
    (
        'email',
        re.compile(
            r'(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@'
            r'[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b'
        ),
        '',
    ),
)

#: Keys whose values are secret, whatever they look like. Compared lowercased.
SECRET_KEYS = frozenset(
    {
        'password',
        'passwd',
        'pwd',
        'secret',
        'client_secret',
        'token',
        'access_token',
        'refresh_token',
        'id_token',
        'api_key',
        'apikey',
        'x-api-key',
        'authorization',
        'private_key',
        'private_key_passphrase',
        'aws_access_key_id',
        'aws_secret_access_key',
        'aws_session_token',
        'workspace_id',
        'account',
        'account_id',
        'account_name',
        'host',
        'hostname',
        'server_hostname',
        'http_path',
        # Who owns or made a table, and where it is stored: what
        # `DESCRIBE TABLE EXTENDED` and `SHOW TABLES` answer with.
        'owner',
        'created by',
        'created_by',
        'last modified by',
        'principal',
        'location',
        'storage_location',
        'storage location',
    }
)

_MARK = re.compile(r'^<redacted:[a-z0-9_-]+>$')


def mark(what: str) -> str:
    r"""The mark for ``what``, in the letters :data:`_MARK` recognizes, so a
    scrubbed value is not found again: ``Created By`` is ``created-by``."""
    slug = re.sub(r'[^a-z0-9_-]+', '-', what.strip().lower()).strip('-')
    return f'<redacted:{slug or "secret"}>'


@dataclass(frozen=True)
class Finding:
    where: str
    what: str


def _environment(
    environ: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    r"""The credential values set here, longest first, so a value inside
    another is not replaced before the one holding it."""
    environ = os.environ if environ is None else environ
    found = [
        (name.lower(), environ[name])
        for name in SECRET_VARIABLES
        if len(environ.get(name, '').strip()) >= 4
    ]
    return sorted(found, key=lambda pair: len(pair[1]), reverse=True)


class Scrubber:
    def __init__(self, environ: dict[str, str] | None = None) -> None:
        self.secrets = _environment(environ)

    def text(self, value: str) -> tuple[str, list[str]]:
        r"""``value`` scrubbed, and what was found in it."""
        found = []
        for what, secret in self.secrets:
            if secret in value:
                value = value.replace(secret, mark(what))
                found.append(what)
        for what, pattern, keep in PATTERNS:
            if keep == r'\1|\2':
                replacement = r'\1' + mark(what) + r'\2'
            else:
                replacement = keep + mark(what)
            value, count = pattern.subn(replacement, value)
            if count:
                found.append(what)
        return value, found

    def json(self, value: Any, path: str = '') -> tuple[Any, list[Finding]]:
        r"""A JSON value scrubbed, keys included, and what was found where."""
        if isinstance(value, str):
            cleaned, found = self.text(value)
            return cleaned, [Finding(path or '<root>', w) for w in found]
        if isinstance(value, list):
            out, findings = [], []
            for index, item in enumerate(value):
                cleaned, found = self.json(item, f'{path}[{index}]')
                out.append(cleaned)
                findings += found
            return out, findings
        if isinstance(value, dict):
            out_dict: dict[str, Any] = {}
            findings = []
            for key, item in value.items():
                cleaned_key, found = self.text(key)
                where = f'{path}.{cleaned_key}' if path else cleaned_key
                findings += [Finding(where, f'key {w}') for w in found]
                if self._secret_value(key, item):
                    out_dict[cleaned_key] = mark(key.strip().lower())
                    findings.append(Finding(where, f'value under {key!r}'))
                    continue
                cleaned, found = self.json(item, where)
                out_dict[cleaned_key] = cleaned
                findings += found
            return out_dict, findings
        return value, []

    def _secret_value(self, key: str, value: Any) -> bool:
        return (
            key.strip().lower() in SECRET_KEYS
            and isinstance(value, (str, int, float))
            and not isinstance(value, bool)
            and not (isinstance(value, str) and _MARK.match(value))
            and value != ''
        )

    def _python(self, value: Any, where: str) -> tuple[Any, list[Finding]]:
        r"""A value read out of Arrow, every string in it scrubbed.

        Struct members and map keys are scrubbed as names, and a member or
        map entry named as a secret has its value redacted.
        """
        if isinstance(value, str):
            cleaned, found = self.text(value)
            return cleaned, [Finding(where, w) for w in found]
        if isinstance(value, list):
            out, findings = [], []
            for index, item in enumerate(value):
                cleaned, found = self._python(item, f'{where}[{index}]')
                out.append(cleaned)
                findings += found
            return out, findings
        if isinstance(value, dict):
            out_dict, findings = {}, []
            for key, item in value.items():
                name, found = self.text(key)
                findings += [Finding(f'{where}.{name}', w) for w in found]
                if self._secret_value(key, item):
                    out_dict[name] = mark(key.strip().lower())
                    findings.append(
                        Finding(f'{where}.{name}', f'value under {key!r}')
                    )
                    continue
                cleaned, found = self._python(item, f'{where}.{name}')
                out_dict[name] = cleaned
                findings += found
            return out_dict, findings
        if isinstance(value, tuple) and len(value) == 2:
            # A map entry: (key, value).
            key, item = value
            cleaned_key, found = self._python(key, f'{where}.key')
            if isinstance(key, str) and self._secret_value(key, item):
                return (cleaned_key, mark(key.strip().lower())), [
                    *found,
                    Finding(f'{where}.value', f'value under {key!r}'),
                ]
            cleaned, more = self._python(item, f'{where}.value')
            return (cleaned_key, cleaned), found + more
        return value, []

    def _type(
        self, data_type: pa.DataType, where: str
    ) -> tuple[pa.DataType, list[Finding]]:
        r"""A type with the names of its nested fields scrubbed."""
        findings: list[Finding] = []
        if pa.types.is_struct(data_type):
            fields = []
            for member in data_type:
                cleaned, found = self._field(member, where)
                fields.append(cleaned)
                findings += found
            return pa.struct(fields), findings
        if pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
            item, found = self._field(data_type.value_field, where)
            maker = pa.list_ if pa.types.is_list(data_type) else pa.large_list
            return maker(item), found
        if pa.types.is_fixed_size_list(data_type):
            item, found = self._field(data_type.value_field, where)
            return pa.list_(item, data_type.list_size), found
        if pa.types.is_map(data_type):
            key, found = self._field(data_type.key_field, where)
            item, more = self._field(data_type.item_field, where)
            return pa.map_(key, item, data_type.keys_sorted), found + more
        return data_type, findings

    def _metadata(
        self, metadata: dict[bytes, bytes] | None, where: str
    ) -> tuple[dict[bytes, bytes] | None, list[Finding]]:
        if not metadata:
            return metadata, []
        out, findings = {}, []
        for key, value in metadata.items():
            texts = []
            for raw in (key, value):
                try:
                    text = raw.decode('utf-8')
                except UnicodeDecodeError:
                    texts.append(raw)
                    continue
                cleaned, found = self.text(text)
                findings += [Finding(f'{where} metadata', w) for w in found]
                texts.append(cleaned.encode('utf-8'))
            out[texts[0]] = texts[1]
        return out, findings

    def _field(
        self, member: pa.Field, where: str
    ) -> tuple[pa.Field, list[Finding]]:
        name, found = self.text(member.name)
        at = f'{where}.{name}' if where else f'column {name}'
        findings = [Finding(at, w) for w in found]
        data_type, more = self._type(member.type, at)
        metadata, most = self._metadata(member.metadata, at)
        cleaned = pa.field(name, data_type, member.nullable, metadata)
        return cleaned, findings + more + most

    def table(self, table: pa.Table) -> tuple[pa.Table, list[Finding]]:
        r"""An Arrow table with every string scrubbed, wherever it is."""
        findings: list[Finding] = []
        fields, columns = [], []
        for member, column in zip(table.schema, table.columns, strict=True):
            cleaned_field, found = self._field(member, '')
            findings += found
            name = cleaned_field.name
            values = column.to_pylist()
            secret_column = any(
                self._secret_value(member.name, value) for value in values
            )
            if secret_column:
                findings.append(
                    Finding(f'column {name}', f'values under {member.name!r}')
                )
                values = [
                    mark(member.name.strip().lower())
                    if self._secret_value(member.name, value)
                    else value
                    for value in values
                ]
                if not _is_text(cleaned_field.type):
                    # A number under a secret name: the mark is text.
                    cleaned_field = cleaned_field.with_type(pa.string())
                    values = [
                        None if value is None else str(value)
                        for value in values
                    ]
            cleaned_values, found = self._python(values, name)
            findings += found
            changed = secret_column or found or not cleaned_field.equals(member)
            if changed:
                column = pa.chunked_array(
                    [pa.array(cleaned_values, type=cleaned_field.type)],
                    type=cleaned_field.type,
                )
            fields.append(cleaned_field)
            columns.append(column)
        metadata, found = self._metadata(table.schema.metadata, 'schema')
        findings += found
        table = pa.Table.from_arrays(
            columns, schema=pa.schema(fields, metadata=metadata)
        )
        table, found = self._key_value_rows(table)
        return table, findings + found

    def _key_value_rows(
        self, table: pa.Table
    ) -> tuple[pa.Table, list[Finding]]:
        r"""A result of key-value rows, such as ``DESCRIBE TABLE EXTENDED``'s,
        with the rest of each row a secret-named first cell starts redacted.
        """
        if table.num_columns < 2 or not _is_text(table.schema[0].type):
            return table, []
        keys = table.column(0).to_pylist()
        rows = [
            index
            for index, key in enumerate(keys)
            if isinstance(key, str) and key.strip().lower() in SECRET_KEYS
        ]
        if not rows:
            return table, []
        findings = []
        columns = [table.column(0)]
        for member, column in zip(
            list(table.schema)[1:], table.columns[1:], strict=True
        ):
            values = column.to_pylist()
            hit = False
            for index in rows:
                if self._secret_value(keys[index], values[index]):
                    values[index] = mark(keys[index].strip().lower())
                    findings.append(
                        Finding(
                            f'{member.name}[{index}]',
                            f'value under {keys[index]!r}',
                        )
                    )
                    hit = True
            if hit:
                column = pa.chunked_array(
                    [pa.array(values, type=member.type)], type=member.type
                )
            columns.append(column)
        return pa.Table.from_arrays(columns, schema=table.schema), findings


def _is_text(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_string(data_type)
        or pa.types.is_large_string(data_type)
        or (
            pa.types.is_dictionary(data_type) and _is_text(data_type.value_type)
        )
    )


def _files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.arrow'):
            yield path


def _read_arrow(path: Path) -> tuple[pa.Table, str]:
    data = path.read_bytes()
    try:
        return pa.ipc.open_file(pa.BufferReader(data)).read_all(), 'file'
    except pa.ArrowInvalid:
        return pa.ipc.open_stream(pa.BufferReader(data)).read_all(), 'stream'


def _write_arrow(path: Path, table: pa.Table, kind: str) -> None:
    with pa.OSFile(str(path), 'wb') as sink:
        opener = pa.ipc.new_file if kind == 'file' else pa.ipc.new_stream
        with opener(sink, table.schema) as writer:
            writer.write_table(table)


def scrub_tree(
    source: Path, target: Path, scrubber: Scrubber | None = None
) -> list[Finding]:
    r"""Copy the JSON and Arrow files under ``source`` to ``target``, scrubbed."""
    from _parity.transcript import _pretty

    scrubber = scrubber or Scrubber()
    findings: list[Finding] = []
    target.mkdir(parents=True, exist_ok=True)
    for path in _files(source):
        out = target / path.relative_to(source)
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.json':
            cleaned, found = scrubber.json(json.loads(path.read_text('utf-8')))
            out.write_text(_pretty(cleaned, 0) + '\n', encoding='utf-8')
        else:
            table, kind = _read_arrow(path)
            cleaned_table, found = scrubber.table(table)
            if found:
                _write_arrow(out, cleaned_table, kind)
            else:
                shutil.copyfile(path, out)
        findings += [
            Finding(f'{out.relative_to(target)}: {f.where}', f.what)
            for f in found
        ]
    return findings


def scan(roots: list[Path], scrubber: Scrubber | None = None) -> list[Finding]:
    r"""What scrubbing the files under ``roots`` would remove."""
    scrubber = scrubber or Scrubber()
    findings: list[Finding] = []
    for root in roots:
        for path in _files(root):
            if path.suffix == '.json':
                try:
                    value = json.loads(path.read_text('utf-8'))
                except ValueError:
                    findings.append(Finding(str(path), 'unreadable JSON'))
                    continue
                _, found = scrubber.json(value)
            else:
                _, found = scrubber.table(_read_arrow(path)[0])
            findings += [Finding(f'{path}: {f.where}', f.what) for f in found]
    return findings
