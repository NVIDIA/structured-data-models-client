# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The live write guard: every live write, DDL included, goes only to
the one test schema of its warehouse; anything else is refused before the
driver sees it (KO3-51, the owner's rule).

    uninstall = install(from_environment())   # every driver, guarded
    ...
    uninstall()

One guard per dialect (``databricks``, ``snowflake``, and ``postgres`` for
Lakebase), enforced in one place: :func:`install` wraps every entry point a
statement reaches a server through, on the driver classes themselves
(Snowflake's cursor base and connection, sync and async, and Snowpark's
session, Databricks SQL's cursor, Spark's session, classic and Connect,
Spark Connect's execution boundary, psycopg's cursors and
connections, and every subclass, now or defined later), so every
connection, every cursor a driver hands back and a session's own
connection are guarded, with no wrapper object to step around. Spark SQL
given arguments to format into its text is refused. Spark's and Delta's
writer classes, whole, and Spark's DataFrame and catalog writes, which
send no statement the guard could read, and Spark's global temporary
views, are refused outright, as is Snowflake's ``execute_stream``; on
Spark Connect a plan passes only as a read or a SQL command, every SQL
text in it checked. A dialect with no test schema configured refuses every statement.

The test schema comes from the constructor or the environment:
``SDM_LIVE_<DIALECT>_TEST_DATABASE`` and ``SDM_LIVE_<DIALECT>_TEST_SCHEMA``,
else ``SDM_LIVE_TEST_DATABASE`` and ``SDM_LIVE_TEST_SCHEMA`` (Postgres
needs no database: its names are ``<schema>.<object>``). No resource name
is written here.

An allowlist. A statement is admitted only in one of the forms below, read
whole; everything else is refused, so a gap fails closed. One statement
per text: any comment, semicolon, backslash, character outside ASCII
(outside a string), flow or pipe operator (Snowflake's ``->>``,
Databricks' ``|>``), raw or escaped string (``r'...'``, ``E'...'``,
``U&'...'``), or quoted name holding its own quote or a control character
is refused, so the guard and the engine never read the text differently.

Reads (the forms the old client and the runner issue), anywhere:

- ``SELECT`` and ``WITH`` queries holding no write verb, no ``INTO`` and
  no sequence (``NEXTVAL`` and the like), calling only the pure functions
  of :data:`FUNCTIONS`, by plain name (Postgres' also as ``pg_catalog.``),
  a keyword taking a ``(`` only where it cannot be a call, and casting
  (``::`` or ``CAST``) only to a built-in type of :data:`CAST_TYPES`;
- ``DESCRIBE TABLE [EXTENDED] <t>``, ``SHOW TABLES [LIKE <p>] [IN [SCHEMA]
  <s>]``, ``SHOW IMPORTED KEYS IN TABLE <t>``, ``SHOW PARAMETERS LIKE <p>
  IN SESSION``, ``USE CATALOG | SCHEMA | DATABASE <s>`` and ``ALTER
  SESSION SET <p> = <value>`` of :data:`SESSION_PARAMETERS`.

Writes (the forms the contract loader issues), each target ``<t>`` fully
qualified in the test schema (``<database>.<schema>.<object>``; Postgres
``<schema>.<object>``):

- ``CREATE [OR REPLACE] TABLE <t> (<columns>)`` (Databricks: then
  ``TBLPROPERTIES ('...' = '...')``): each column ``<name> <type> [NOT
  NULL]`` of a built-in type the loader writes (:data:`TYPES`; on Postgres
  also a composite type in the test schema), or a key constraint whose
  ``REFERENCES`` is in the test schema; nothing else (no ``LIKE``,
  ``CHECK``, ``DEFAULT``, generated column, domain or other schema's type);
- ``INSERT INTO <t> [(<names>)] VALUES | SELECT ...``, holding nothing a
  read may not;
- ``DROP TABLE | VIEW [IF EXISTS] <t>``, Snowflake's ``DROP SEMANTIC VIEW``
  and Postgres' ``DROP TYPE`` too: one target, never ``CASCADE``;
- Databricks: ``CREATE [OR REPLACE] VIEW <t> WITH METRICS LANGUAGE YAML AS
  $$...$$``; Postgres: ``CREATE TYPE <t> AS (...)`` and ``COPY <t> FROM
  STDIN``;
- Snowflake: ``CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('<the test
  schema>', $$<yaml>$$)``, exactly two literal arguments, the YAML's
  ``name`` one plain name and each table's ``base_table`` in the test
  schema, no key given twice.

Snowflake reads an unquoted name upper-cased and a quoted one exactly;
Postgres an unquoted one lower-cased and a quoted one exactly; Unity
Catalog case-insensitively (a double-quoted text is a string there).

Not covered: statements sent below the drivers' public entry points (a
Spark session's JVM, libpq called directly), and functions a statement
calls without parentheses (Postgres' attribute notation, a user-defined
operator), which need a writing function or operator to exist already;
neither the loader nor the old client uses them.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

DIALECTS = ('databricks', 'snowflake', 'postgres')


class WriteRefused(PermissionError):
    r"""A statement the guard refuses; the message names the rule and the
    statement's first words, never a bound value."""


# Tokens.


@dataclass(frozen=True)
class Token:
    kind: str  # 'word', 'name' (quoted), 'string', 'number', 'punct'
    text: str


_WORD = re.compile(r'[A-Za-z_][A-Za-z0-9_$]*')
_NUMBER = re.compile(r'[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?')
_PLAIN = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
#: Text never admitted outside a string: comments, statement separators,
#: and Snowflake's flow and Databricks' pipe operators (Postgres' ``->>``
#: reads a JSON field, which the old client's SQL does).
_NEVER = {
    'databricks': ('--', '/*', '*/', ';', '|>'),
    'snowflake': ('--', '/*', '*/', ';', '//', '->>'),
    'postgres': ('--', '/*', '*/', ';'),
}


def tokens(sql: str, dialect: str) -> list[Token]:
    r"""``sql``'s tokens in ``dialect``, refusing what the guard and the
    engine could read differently (see the module's text)."""
    if '\\' in sql:
        raise WriteRefused('a backslash')
    out: list[Token] = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if not c.isascii():
            raise WriteRefused('a character outside ASCII, outside a string')
        never = next((t for t in _NEVER[dialect] if sql.startswith(t, i)), None)
        if never:
            raise WriteRefused(f'{never} outside a string')
        if c.isspace():
            i += 1
        elif c == '$' and dialect != 'postgres' and sql.startswith('$$', i):
            end = sql.find('$$', i + 2)
            if end < 0:
                raise WriteRefused('an unterminated $$ body')
            out.append(Token('string', sql[i + 2 : end]))
            i = end + 2
        elif c == '$' and dialect == 'postgres':
            raise WriteRefused('a $ on Postgres')
        elif c == "'" or (c == '"' and dialect == 'databricks'):
            # Databricks' raw r'...', Postgres' E'...' and U&'...' read
            # escapes their own way.
            if sql[i - 1 : i] == '&' or (
                sql[i - 1 : i].isalpha()
                and out
                and out[-1].kind == 'word'
                and out[-1].text.upper() in ('R', 'E', 'U')
            ):
                raise WriteRefused('a string with escapes of its own')
            text, i = _quoted(sql, i, c)
            out.append(Token('string', text))
        elif c == '"' or (c == '`' and dialect == 'databricks'):
            text, i = _quoted(sql, i, c)
            if (
                c in text
                or not text.isascii()
                or any(ord(ch) < 32 for ch in text)
            ):
                raise WriteRefused(
                    'a quoted name holding its quote or a control character'
                )
            out.append(Token('name', text))
        elif _WORD.match(sql, i):
            word = _WORD.match(sql, i)
            assert word is not None
            out.append(Token('word', word.group()))
            i = word.end()
        elif c.isdigit():
            number = _NUMBER.match(sql, i)
            assert number is not None
            out.append(Token('number', number.group()))
            i = number.end()
        else:
            out.append(Token('punct', c))
            i += 1
    return out


def _quoted(sql: str, start: int, quote: str) -> tuple[str, int]:
    r"""The text quoted by ``quote`` from ``start``, a doubled quote read as
    one."""
    i = start + 1
    while True:
        end = sql.find(quote, i)
        if end < 0:
            raise WriteRefused('an unterminated quoted text')
        if sql.startswith(quote * 2, end):
            i = end + 2
            continue
        return sql[start + 1 : end].replace(quote * 2, quote), end + 1


_OPEN = Token('punct', '(')
_CLOSE = Token('punct', ')')
_COMMA = Token('punct', ',')
_DOT = Token('punct', '.')
_EQUALS = Token('punct', '=')

#: Words a read may not hold, nor what a write fills its target with (a
#: word followed by ``(`` is a function's name, not one).
WRITE_WORDS = frozenset(
    {
        'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'CREATE', 'REPLACE', 'DROP',
        'ALTER', 'TRUNCATE', 'GRANT', 'REVOKE', 'COPY', 'PUT', 'REMOVE',
        'CALL', 'EXECUTE', 'UNDROP', 'VACUUM', 'OPTIMIZE', 'RESTORE',
        'INTO', 'SET',
    }
)  # fmt: skip
#: Words refused, besides those, among a table's columns.
COLUMN_WORDS = WRITE_WORDS | {
    'AS', 'CASCADE', 'CHECK', 'DEFAULT', 'GENERATED', 'LOCATION', 'OPTIONS',
    'USING', 'PURGE',
}  # fmt: skip
#: The functions a read, or what an ``INSERT`` fills its target with, may
#: call: a closed list of pure functions, the ones the old client's
#: generated SQL and the loader's literals use (every Postgres statement
#: the reference recorded, and the loader's statements for every case, are
#: held to it by the tests). Anything else called, by any name, quoted or
#: qualified, is refused.
FUNCTIONS = frozenset(
    {
        # Aggregates, windows and the session's identity.
        'COUNT', 'MIN', 'MAX', 'SUM', 'AVG', 'MEDIAN', 'ROW_NUMBER',
        'CURRENT_SCHEMA', 'CURRENT_DATABASE', 'CURRENT_CATALOG',
        'CURRENT_TIMEZONE',
        # Casts, conversions and literals.
        'CAST', 'TRY_CAST', 'PARSE_JSON', 'HEX_DECODE_STRING', 'TO_BINARY',
        'FROM_JSON', 'COALESCE',
        # Text and time.
        'LOWER', 'UPPER', 'TRIM', 'REPLACE', 'CHR', 'DATE_TRUNC', 'DATEADD',
        'EXTRACT',
        # Nested values and their rows.
        'STRUCT', 'NAMED_STRUCT', 'OBJECT', 'ARRAY', 'MAP', 'FLATTEN',
        'TABLE', 'POSEXPLODE', 'JSONB_ARRAY_ELEMENTS',
        # Postgres' catalog and its sampling's seed (the session's own).
        'FORMAT_TYPE', 'RANDOM', 'SETSEED',
        # The recorder's read of the session's time zone (Postgres).
        'CURRENT_SETTING',
        # The semantic view's definition, read.
        'SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW',
    }
)  # fmt: skip
#: Keywords no dialect lets name a function, so a ``(`` after one, never
#: qualified, calls nothing.
RESERVED = frozenset(
    {
        'ALL', 'AND', 'ANY', 'AS', 'BETWEEN', 'CASE', 'DISTINCT', 'ELSE',
        'END', 'EXCEPT', 'EXISTS', 'FROM', 'GROUP', 'HAVING', 'IN',
        'INTERSECT', 'INTERVAL', 'LATERAL', 'NOT', 'ON', 'OR', 'ROW',
        'SELECT', 'SOME', 'THEN', 'UNION', 'USING', 'VALUES', 'WHEN',
        'WHERE', 'WITH',
    }
)  # fmt: skip
#: Keywords a function could be named, taking a ``(`` only where the
#: grammar puts them: after a closing ``)`` (a window, a filter, a
#: sample's seed), after ORDER, GROUP or PARTITION, or after a table.
AFTER_CLOSE = frozenset({'OVER', 'FILTER', 'REPEATABLE', 'WITHIN'})
AFTER_BY = frozenset({'ORDER', 'GROUP', 'PARTITION'})
AFTER_TABLE = frozenset({'TABLESAMPLE', 'SAMPLE'})
#: The types a cast (``::`` or ``CAST(... AS ...)``) may name, by dialect:
#: built-in types only (a domain's checks may call anything).
CAST_TYPES = {
    'snowflake': frozenset(
        {
            'ARRAY', 'BINARY', 'BOOLEAN', 'DATE', 'DECIMAL', 'DOUBLE',
            'FLOAT', 'INT', 'INTEGER', 'BIGINT', 'NUMBER', 'NUMERIC',
            'OBJECT', 'STRING', 'TEXT', 'TIME', 'TIMESTAMP',
            'TIMESTAMP_LTZ', 'TIMESTAMP_NTZ', 'TIMESTAMP_TZ', 'VARCHAR',
            'VARIANT',
        }
    ),
    'databricks': frozenset(
        {
            'ARRAY', 'BIGINT', 'BINARY', 'BOOLEAN', 'DATE', 'DECIMAL',
            'DOUBLE', 'FLOAT', 'INT', 'INTEGER', 'MAP', 'SMALLINT', 'STRING',
            'STRUCT', 'TIMESTAMP', 'TIMESTAMP_NTZ', 'TINYINT',
        }
    ),
    'postgres': frozenset(
        {
            'BIGINT', 'BOOLEAN', 'BYTEA', 'DATE', 'DOUBLE', 'FLOAT',
            'INT', 'INTEGER', 'INTERVAL', 'JSON', 'JSONB', 'NUMERIC',
            'PRECISION', 'REAL', 'SMALLINT', 'TEXT', 'TIME', 'TIMESTAMP',
            'TIMESTAMPTZ', 'VARCHAR', 'WITH', 'WITHOUT', 'ZONE',
        }
    ),
}  # fmt: skip
#: Words that advance or read a sequence, refused wherever they stand.
SEQUENCE_WORDS = frozenset(
    {'NEXTVAL', 'GETNEXTVAL', 'CURRVAL', 'SETVAL', 'LASTVAL'}
)
#: The session parameters ``ALTER SESSION SET`` may set.
SESSION_PARAMETERS = frozenset(
    {'STATEMENT_TIMEOUT_IN_SECONDS', 'TIMEZONE', 'QUERY_TAG'}
)


#: The column types the loader writes, by dialect: a closed list of
#: built-in types (a qualified or unknown name may be a domain, whose
#: checks can call anything).
TYPES = {
    'snowflake': frozenset(
        {
            'ARRAY', 'BINARY', 'BOOLEAN', 'DATE', 'FLOAT', 'NUMBER', 'OBJECT',
            'TIME', 'TIMESTAMP_LTZ', 'TIMESTAMP_NTZ', 'VARCHAR', 'VARIANT',
            'NOT', 'NULL',
        }
    ),
    'databricks': frozenset(
        {
            'ARRAY', 'BIGINT', 'BINARY', 'BOOLEAN', 'DATE', 'DECIMAL',
            'DOUBLE', 'FLOAT', 'INT', 'MAP', 'SMALLINT', 'STRING', 'STRUCT',
            'TIMESTAMP', 'TIMESTAMP_NTZ', 'TINYINT', 'NOT', 'NULL',
        }
    ),
    'postgres': frozenset(
        {
            'BIGINT', 'BOOLEAN', 'BYTEA', 'DATE', 'DOUBLE', 'INTEGER',
            'INTERVAL', 'JSONB', 'NUMERIC', 'PRECISION', 'REAL', 'SMALLINT',
            'TEXT', 'TIME', 'TIMESTAMP', 'WITH', 'WITHOUT', 'ZONE', 'NOT',
            'NULL',
        }
    ),
}  # fmt: skip
#: The words a table's key constraint starts with.
KEY_WORDS = frozenset({'CONSTRAINT', 'PRIMARY', 'UNIQUE', 'FOREIGN'})


def _split(items: list[Token]) -> list[list[Token]]:
    r"""``items`` split at their top-level commas."""
    parts: list[list[Token]] = [[]]
    depth = 0
    for token in items:
        if token in (_OPEN, Token('punct', '<')):
            depth += 1
        elif token in (_CLOSE, Token('punct', '>')):
            depth -= 1
        if token == _COMMA and depth == 0:
            parts.append([])
        else:
            parts[-1].append(token)
    return parts


def _names(items: list[Token], what: str) -> None:
    if not items or any(
        t.kind not in ('word', 'name') and t != _COMMA for t in items
    ):
        raise WriteRefused(f'{what}: a column list that is not names')


def _names_only(items: list[Token], at: int) -> bool:
    r"""Whether the parenthesised group opening at ``at`` holds names
    only, as an alias's column list does (nothing in it can call)."""
    depth = 0
    for token in items[at:]:
        if token == _OPEN:
            depth += 1
            if depth > 1:
                return False
        elif token == _CLOSE:
            return True
        elif token.kind not in ('word', 'name') and token != _COMMA:
            return False
    return False


def _upper(token: Token | None) -> str:
    return (
        token.text.upper() if token is not None and token.kind == 'word' else ''
    )


class _Statement:
    r"""One statement's tokens, read left to right."""

    def __init__(self, items: list[Token], what: str) -> None:
        self.items = items
        self.at = 0
        self.what = what

    def peek(self, ahead: int = 0) -> Token | None:
        at = self.at + ahead
        return self.items[at] if at < len(self.items) else None

    def accept(self, *words: str) -> bool:
        if all(_upper(self.peek(k)) == word for k, word in enumerate(words)):
            self.at += len(words)
            return True
        return False

    def expect(self, *words: str) -> None:
        if not self.accept(*words):
            self.refuse(f'not {" ".join(words)} here')

    def take(self, token: Token) -> bool:
        if self.peek() == token:
            self.at += 1
            return True
        return False

    def literal(self) -> None:
        r"""Moves past a string, a number or a bound parameter."""
        token = self.peek()
        if token is not None and token.kind in ('string', 'number'):
            self.at += 1
        elif not (
            self.take(Token('punct', '?'))
            or (self.take(Token('punct', '%')) and self.accept('S'))
        ):
            self.refuse('not a literal here')

    def end(self) -> None:
        if self.at != len(self.items):
            self.refuse('more than the form the guard admits')

    def group(self) -> list[Token]:
        r"""The tokens inside the parenthesised group at the cursor, which
        it moves past."""
        if not self.take(_OPEN):
            self.refuse('no ( here')
        start, depth = self.at, 1
        while self.at < len(self.items):
            token = self.items[self.at]
            self.at += 1
            depth += (token == _OPEN) - (token == _CLOSE)
            if depth == 0:
                return self.items[start : self.at - 1]
        self.refuse('an unclosed (')
        raise AssertionError

    def refuse(self, why: str) -> None:
        raise WriteRefused(f'{self.what}: {why}')


class WriteGuard:
    r"""Checks statements against the one test schema of ``dialect``'s
    warehouse (see the module's text)."""

    def __init__(
        self,
        dialect: str,
        database: str | None = None,
        schema: str | None = None,
    ) -> None:
        if dialect not in DIALECTS:
            raise ValueError(
                f'a write guard for {dialect!r}; one of {DIALECTS}'
            )
        prefix = f'SDM_LIVE_{dialect.upper()}_TEST_'
        database = (
            database
            or os.environ.get(prefix + 'DATABASE')
            or os.environ.get('SDM_LIVE_TEST_DATABASE')
        )
        schema = (
            schema
            or os.environ.get(prefix + 'SCHEMA')
            or os.environ.get('SDM_LIVE_TEST_SCHEMA')
        )
        if not schema or (dialect != 'postgres' and not database):
            raise WriteRefused(
                f'the {dialect} write guard names no test schema: set '
                'SDM_LIVE_TEST_DATABASE and SDM_LIVE_TEST_SCHEMA'
            )
        self.dialect = dialect
        self.database = database if dialect != 'postgres' else None
        self.schema = schema

    # Names.

    def _same(self, part: tuple[str, bool], stored: str) -> bool:
        text, quoted = part
        if self.dialect == 'databricks':
            return text.lower() == stored.lower()
        if quoted:
            return text == stored
        return (
            text.upper() if self.dialect == 'snowflake' else text.lower()
        ) == stored

    def _container(self, parts: list[tuple[str, bool]]) -> bool:
        r"""Whether ``parts`` name the test schema itself."""
        if self.database is None:
            return len(parts) == 1 and self._same(parts[0], self.schema)
        return (
            len(parts) == 2
            and self._same(parts[0], self.database)
            and self._same(parts[1], self.schema)
        )

    def _name(self, s: _Statement) -> list[tuple[str, bool]]:
        r"""The dotted name at the cursor, which it moves past."""
        parts: list[tuple[str, bool]] = []
        while True:
            token = s.peek()
            if token is None or token.kind not in ('word', 'name'):
                break
            if token.kind == 'word' and token.text.upper() == 'IDENTIFIER':
                s.refuse('IDENTIFIER(...) as a name')
            parts.append((token.text, token.kind == 'name'))
            s.at += 1
            if not s.take(_DOT):
                break
        if not parts:
            s.refuse('no name here')
        return parts

    def _target(self, s: _Statement) -> None:
        r"""Moves past a name in the test schema, refusing any other."""
        parts = self._name(s)
        if not self._container(parts[:-1]):
            s.refuse(
                'a write target outside the test schema, or not fully '
                'qualified in it'
            )

    def _held(
        self, items: list[Token], words: frozenset[str], what: str
    ) -> None:
        r"""Refuses ``items`` holding one of ``words`` or a sequence's word,
        casting to a type not built in, or calling anything but
        :data:`FUNCTIONS` by a plain name; a keyword takes a ``(`` only
        where it cannot be a call (:data:`RESERVED` and the words after it),
        and an alias after ``AS`` its list of names."""
        # The word each open parenthesis follows, innermost last: inside a
        # cast, AS names its type.
        openers: list[str] = []
        types: set[int] = set()
        for i, token in enumerate(items):
            word = _upper(token)
            after = items[i + 1 : i + 2]
            before = items[i - 1] if i else None
            if token == _OPEN:
                openers.append(_upper(before))
                continue
            if token == _CLOSE:
                if openers:
                    openers.pop()
                continue
            if word in words and after != [_OPEN]:
                raise WriteRefused(f'{what}: {word} where no write may be')
            if word in SEQUENCE_WORDS:
                raise WriteRefused(f'{what}: {word}, a sequence')
            if token == Token('punct', ':') and after == [Token('punct', ':')]:
                types |= self._cast_type(items, i + 2, what)
                continue
            if word == 'AS' and openers[-1:] in (['CAST'], ['TRY_CAST']):
                types |= self._cast_type(items, i + 1, what)
                continue
            if after != [_OPEN] or token.kind not in ('word', 'name'):
                continue
            if i in types:
                continue  # a cast's type, its parameters after it
            if token.kind == 'name':
                raise WriteRefused(f'{what}: a quoted name called')
            if before == _DOT:
                if not (
                    self.dialect == 'postgres'
                    and word in FUNCTIONS
                    and _upper(items[i - 2]) == 'PG_CATALOG'
                    and items[i - 3 : i - 2] != [_DOT]
                ):
                    raise WriteRefused(f'{what}: a qualified function')
                continue
            if before == Token('punct', '!'):
                raise WriteRefused(f'{what}: a method called')
            if (
                word in FUNCTIONS
                or word in RESERVED
                or (word in AFTER_CLOSE and before == _CLOSE)
                or (word == 'BY' and _upper(before) in AFTER_BY)
                or (
                    word in AFTER_TABLE
                    and before is not None
                    and before.kind in ('word', 'name')
                    and _upper(before) not in RESERVED
                )
                or (
                    _upper(before) == 'AS'
                    and openers[-1:] not in (['CAST'], ['TRY_CAST'])
                    and _names_only(items, i + 1)
                )
            ):
                continue
            raise WriteRefused(f'{what}: {word}(), a function on no list')

    def _cast_type(self, items: list[Token], at: int, what: str) -> set[int]:
        r"""The built-in type a cast names at ``at`` (one word or several,
        never qualified or quoted); refuses any other. Returns the index
        of its words."""
        found: set[int] = set()
        i = at
        while (
            i < len(items)
            and items[i].kind == 'word'
            and _upper(items[i]) in CAST_TYPES[self.dialect]
        ):
            found.add(i)
            i += 1
        if not found or items[i : i + 1] == [_DOT]:
            raise WriteRefused(f'{what}: a cast to a type that is not built in')
        return found

    # Statements.

    def check(self, sql: str) -> None:
        r"""Refuses ``sql`` (:class:`WriteRefused`) unless it is one
        statement of a form the module's text admits."""
        if not isinstance(sql, str):
            raise WriteRefused('a statement that is not text')
        items = tokens(sql, self.dialect)
        what = ' '.join(t.text for t in items[:2] if t.kind == 'word').upper()
        s = _Statement(items, what or 'a statement')
        verb = _upper(s.peek())
        s.at = 1
        if verb in ('SELECT', 'WITH'):
            self._held(items, WRITE_WORDS, s.what)
        elif verb in ('DESCRIBE', 'DESC'):
            s.expect('TABLE')
            s.accept('EXTENDED')
            self._name(s)
            s.end()
        elif verb == 'SHOW':
            self._show(s)
        elif verb == 'USE':
            if not (
                s.accept('CATALOG')
                or s.accept('SCHEMA')
                or s.accept('DATABASE')
            ):
                s.refuse('not USE CATALOG, SCHEMA or DATABASE')
            self._name(s)
            s.end()
        elif verb == 'ALTER' and self.dialect == 'snowflake':
            s.expect('SESSION', 'SET')
            if _upper(s.peek()) not in SESSION_PARAMETERS:
                s.refuse('a session parameter the guard does not set')
            s.at += 1
            if not s.take(_EQUALS):
                s.refuse('no = here')
            s.literal()
            s.end()
        elif verb == 'CREATE':
            self._create(s)
        elif verb == 'DROP':
            self._drop(s)
        elif verb == 'INSERT':
            s.expect('INTO')
            self._target(s)
            if s.peek() == _OPEN:
                columns = s.group()
                if any(
                    t.kind not in ('word', 'name') and t != _COMMA
                    for t in columns
                ):
                    s.refuse('an INSERT column list that is not names')
            if _upper(s.peek()) not in ('VALUES', 'SELECT'):
                s.refuse('an INSERT of neither VALUES nor SELECT')
            self._held(s.items[s.at :], WRITE_WORDS, s.what)
        elif verb == 'COPY' and self.dialect == 'postgres':
            self._target(s)
            s.expect('FROM', 'STDIN')
            s.end()
        elif verb == 'CALL' and self.dialect == 'snowflake':
            self._call(s)
        else:
            s.refuse('a statement the guard does not admit')

    def _show(self, s: _Statement) -> None:
        if s.accept('TABLES'):
            if s.accept('LIKE'):
                s.literal()
            if s.accept('IN'):
                s.accept('SCHEMA')
                self._name(s)
        elif s.accept('IMPORTED', 'KEYS', 'IN', 'TABLE'):
            self._name(s)
        elif s.accept('PARAMETERS', 'LIKE'):
            s.literal()
            s.expect('IN', 'SESSION')
        else:
            s.refuse('a SHOW the guard does not admit')
        s.end()

    def _create(self, s: _Statement) -> None:
        replace = s.accept('OR', 'REPLACE')
        if s.accept('TABLE'):
            self._target(s)
            self._columns(s.group(), s.what, constraints=True)
            if self.dialect == 'databricks' and s.accept('TBLPROPERTIES'):
                properties = s.group()
                if any(t.kind not in ('string', 'punct') for t in properties):
                    s.refuse('a table property that is not a literal')
            s.end()
        elif self.dialect == 'databricks' and s.accept('VIEW'):
            self._target(s)
            s.expect('WITH', 'METRICS', 'LANGUAGE', 'YAML', 'AS')
            body = s.peek()
            if body is None or body.kind != 'string':
                s.refuse('a metric view with no YAML body')
            s.at += 1
            s.end()
        elif self.dialect == 'postgres' and not replace and s.accept('TYPE'):
            self._target(s)
            s.expect('AS')
            self._columns(s.group(), s.what, constraints=False)
            s.end()
        else:
            s.refuse('an object the guard does not create')

    def _columns(
        self, items: list[Token], what: str, constraints: bool
    ) -> None:
        r"""A table's (or composite type's) columns, each ``<name> <type>
        [NOT NULL]`` with a built-in type of :data:`TYPES` (on Postgres
        also a composite type in the test schema), or, in a table, a key
        constraint (``[CONSTRAINT <name>] PRIMARY KEY | UNIQUE (<names>)``,
        ``FOREIGN KEY (<names>) REFERENCES <t> (<names>)``, ``<t>`` in the
        test schema): nothing taken from another object (``LIKE``), and
        nothing a write could call (a domain's or user type's checks)."""
        for column in _split(items):
            if not column:
                raise WriteRefused(f'{what}: an empty column')
            if constraints and _upper(column[0]) in KEY_WORDS:
                self._constraint(column, what)
                continue
            if column[0].kind not in ('word', 'name'):
                raise WriteRefused(f'{what}: a column with no name')
            self._type(column[1:], what)

    def _type(self, items: list[Token], what: str) -> None:
        i = 0
        while i < len(items):
            token = items[i]
            word = _upper(token)
            if token.kind == 'word' and word in TYPES[self.dialect]:
                i += 1
            elif token.kind == 'number' or (
                token.kind == 'punct' and token.text in '()<>[],'
            ):
                i += 1
            elif (
                token.kind == 'name'
                and self.dialect == 'databricks'
                and items[i + 1 : i + 2] == [Token('punct', ':')]
            ):
                i += 2  # a struct's field
            elif self.dialect == 'postgres' and token.kind in ('word', 'name'):
                inner = _Statement(items[i:], what)
                self._target(inner)  # a composite type in the test schema
                i += inner.at
            else:
                raise WriteRefused(
                    f'{what}: a column type the guard does not admit'
                )

    def _constraint(self, items: list[Token], what: str) -> None:
        s = _Statement(items, what)
        if s.accept('CONSTRAINT'):
            if s.peek() is None or s.peek().kind not in ('word', 'name'):  # type: ignore[union-attr]
                s.refuse('a constraint with no name')
            s.at += 1
        if s.accept('PRIMARY', 'KEY') or s.accept('UNIQUE'):
            _names(s.group(), what)
        elif s.accept('FOREIGN', 'KEY'):
            _names(s.group(), what)
            s.expect('REFERENCES')
            self._target(s)
            _names(s.group(), what)
        else:
            s.refuse('a constraint the guard does not admit')
        s.end()

    def _drop(self, s: _Statement) -> None:
        if not (
            s.accept('TABLE')
            or s.accept('VIEW')
            or (self.dialect == 'postgres' and s.accept('TYPE'))
            or (self.dialect == 'snowflake' and s.accept('SEMANTIC', 'VIEW'))
        ):
            s.refuse('an object the guard does not drop')
        s.accept('IF', 'EXISTS')
        self._target(s)
        s.end()

    def _call(self, s: _Statement) -> None:
        s.expect('SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML')
        arguments = s.group()
        s.end()
        if not (
            len(arguments) == 3
            and arguments[0].kind == 'string'
            and arguments[1] == _COMMA
            and arguments[2].kind == 'string'
        ):
            s.refuse('a semantic view not created from two literal arguments')
        schema = _Statement(tokens(arguments[0].text, self.dialect), s.what)
        if not self._container(self._name(schema)):
            s.refuse('a semantic view outside the test schema')
        schema.end()
        try:
            import yaml

            document = yaml.load(arguments[2].text, Loader=_unique_keys(yaml))
        except Exception:  # noqa: BLE001 - YAML it cannot read is refused
            s.refuse('a semantic view YAML the guard cannot read')
        if not isinstance(document, dict):
            s.refuse('a semantic view YAML that is no mapping')
        name = document.get('name')
        if not (isinstance(name, str) and _PLAIN.fullmatch(name)):
            s.refuse('a semantic view whose name is not one plain name')
        for table in document.get('tables') or []:
            base = table.get('base_table') if isinstance(table, dict) else None
            if not (
                isinstance(base, dict)
                and set(base) == {'database', 'schema', 'table'}
            ):
                s.refuse(
                    'a semantic view table with no database, schema and table'
                )
            parts = [
                _yaml_name(base[key]) for key in ('database', 'schema', 'table')
            ]
            if None in parts or not self._container(parts[:2]):  # type: ignore[arg-type]
                s.refuse('a semantic view table outside the test schema')


def _unique_keys(yaml: Any) -> Any:
    r"""A safe YAML loader refusing a mapping that gives a key twice,
    which an engine may read otherwise than the guard does."""

    class Loader(yaml.SafeLoader):
        def construct_mapping(self, node: Any, deep: bool = False) -> Any:
            keys = [self.construct_object(k, deep=deep) for k, _ in node.value]
            if len(keys) != len(set(map(repr, keys))):
                raise ValueError('a YAML key given twice')
            return super().construct_mapping(node, deep=deep)

    return Loader


def _yaml_name(value: Any) -> tuple[str, bool] | None:
    r"""A semantic view YAML's name: plain, or quoted as ``"..."`` holding
    no quote or control character."""
    if not isinstance(value, str):
        return None
    if _PLAIN.fullmatch(value):
        return value, False
    inner = value[1:-1]
    if (
        len(value) > 2
        and value[0] == value[-1] == '"'
        and '"' not in inner
        and all(ord(c) >= 32 for c in inner)
    ):
        return inner, True
    return None


# The drivers.


#: Each driver entry point a statement reaches a server through, by
#: module, class, attributes and dialect: each is guarded on the class and
#: on every subclass, now or defined later, that defines or inherits
#: another one.
ENTRY_POINTS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    # SnowflakeCursorBase holds every cursor's entry points (SnowflakeCursor
    # and its sibling DictCursor); older connectors have SnowflakeCursor only.
    ('snowflake.connector.cursor', 'SnowflakeCursorBase', ('execute', 'executemany', 'execute_async'), 'snowflake'),
    ('snowflake.connector.cursor', 'SnowflakeCursor', ('execute', 'executemany', 'execute_async'), 'snowflake'),
    ('snowflake.connector.cursor', 'DictCursor', ('execute', 'executemany', 'execute_async'), 'snowflake'),
    ('snowflake.connector.connection', 'SnowflakeConnection', ('execute_string', 'cmd_query'), 'snowflake'),
    ('snowflake.connector.aio._cursor', 'SnowflakeCursorBase', ('execute', 'executemany', 'execute_async'), 'snowflake'),
    ('snowflake.connector.aio._connection', 'SnowflakeConnection', ('execute_string', 'cmd_query'), 'snowflake'),
    ('snowflake.snowpark.session', 'Session', ('sql',), 'snowflake'),
    ('databricks.sql.client', 'Cursor', ('execute', 'executemany', 'execute_async'), 'databricks'),
    ('pyspark.sql.session', 'SparkSession', ('sql',), 'databricks'),
    ('pyspark.sql.connect.session', 'SparkSession', ('sql',), 'databricks'),
    ('psycopg', 'Cursor', ('execute', 'executemany', 'copy', 'stream'), 'postgres'),
    ('psycopg', 'ServerCursor', ('execute',), 'postgres'),
    ('psycopg', 'AsyncCursor', ('execute', 'executemany', 'copy', 'stream'), 'postgres'),
    ('psycopg', 'AsyncServerCursor', ('execute',), 'postgres'),
    ('psycopg', 'Connection', ('execute',), 'postgres'),
    ('psycopg', 'AsyncConnection', ('execute',), 'postgres'),
)  # fmt: skip
#: Driver interfaces that write with no statement the guard could read,
#: refused outright.
REFUSED: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ('snowflake.connector.connection', 'SnowflakeConnection', ('execute_stream',)),
    ('snowflake.connector.aio._connection', 'SnowflakeConnection', ('execute_stream',)),
    ('pyspark.sql.dataframe', 'DataFrame', ('write', 'writeTo', 'writeStream', 'mergeInto', 'createGlobalTempView', 'createOrReplaceGlobalTempView')),
    ('pyspark.sql.connect.dataframe', 'DataFrame', ('write', 'writeTo', 'writeStream', 'mergeInto', 'createGlobalTempView', 'createOrReplaceGlobalTempView')),
    ('pyspark.sql.catalog', 'Catalog', ('createTable', 'createExternalTable', 'recoverPartitions', 'dropGlobalTempView')),
    ('pyspark.sql.connect.catalog', 'Catalog', ('createTable', 'createExternalTable', 'recoverPartitions', 'dropGlobalTempView')),
)  # fmt: skip
#: Writer classes refused whole, every public method and property, so a
#: writer kept from before the install, or made directly, writes nothing:
#: Spark's (classic and Connect) and Delta's. Absent from a version, a
#: class is skipped.
REFUSED_CLASSES: tuple[tuple[str, str], ...] = (
    ('pyspark.sql.readwriter', 'DataFrameWriter'),
    ('pyspark.sql.readwriter', 'DataFrameWriterV2'),
    ('pyspark.sql.streaming.readwriter', 'DataStreamWriter'),
    ('pyspark.sql.merge', 'MergeIntoWriter'),
    ('pyspark.sql.connect.readwriter', 'DataFrameWriter'),
    ('pyspark.sql.connect.readwriter', 'DataFrameWriterV2'),
    ('pyspark.sql.connect.streaming.readwriter', 'DataStreamWriter'),
    ('pyspark.sql.connect.merge', 'MergeIntoWriter'),
    ('delta.tables', 'DeltaTable'),
    ('delta.tables', 'DeltaMergeBuilder'),
    ('delta.tables', 'DeltaTableBuilder'),
    ('delta.tables', 'DeltaOptimizeBuilder'),
    ('delta.connect.tables', 'DeltaTable'),
    ('delta.connect.tables', 'DeltaMergeBuilder'),
    ('delta.connect.tables', 'DeltaTableBuilder'),
    ('delta.connect.tables', 'DeltaOptimizeBuilder'),
)
#: Entry point classes a driver version may not have (each covered by
#: another entry when absent).
_OPTIONAL = frozenset(
    {
        ('snowflake.connector.cursor', 'SnowflakeCursorBase'),
        ('snowflake.connector.cursor', 'DictCursor'),
    }
)
#: Spark Connect's execution boundary: every plan a Connect session sends
#: (``to_table``, ``to_pandas``, a command, a function's registration)
#: passes one of these with its whole request. A plan is admitted only as a
#: read (a relation) or a SQL command, every SQL text in it checked; each
#: method must exist, or the install stops.
SPARK_CONNECT = (
    'pyspark.sql.connect.client.core',
    'SparkConnectClient',
    ('_execute_and_fetch_as_iterator', '_execute'),
)
#: The keyword each driver names its statement by, when it is not given
#: first.
_STATEMENT_KEYWORDS = (
    'command', 'operation', 'query', 'sqlQuery', 'statement', 'sql',
)  # fmt: skip
#: The keywords Spark's ``sql`` may be given: its bound ``args``; any other
#: is formatted into the text after the guard reads it.
_SPARK_KEYWORDS = frozenset({'sqlQuery', 'args'})
_MARK = '__sdm_write_guard__'
_INSTALLED: list[tuple[type, str, Any]] = []
#: Every refusal an installed entry point made since :func:`install`, its
#: message: a caller that catches one (the old client's call makes it the
#: case's outcome) cannot hide it from the run, which fails the case.
REFUSALS: list[str] = []


def _keeping(function: Any) -> Any:
    r"""``function``, each refusal it raises kept in :data:`REFUSALS`."""

    @functools.wraps(function)
    def kept(*arguments: Any, **keywords: Any) -> Any:
        try:
            return function(*arguments, **keywords)
        except WriteRefused as refused:
            REFUSALS.append(str(refused))
            raise

    setattr(kept, _MARK, True)
    return kept


#: What :func:`install` applies, by base class, for a subclass defined later.
_RULES: list[tuple[type, str, Callable[[Any], Any]]] = []
_ABSENT = object()


def _text(owner: Any, statement: Any) -> str:
    r"""``statement`` as the text the server runs: a string, UTF-8 bytes,
    or a psycopg composed statement rendered for ``owner``."""
    if isinstance(statement, str):
        return statement
    if isinstance(statement, bytes):
        try:
            return statement.decode('utf-8')
        except UnicodeDecodeError:
            raise WriteRefused('a statement that is not UTF-8') from None
    as_string = getattr(statement, 'as_string', None)
    if callable(as_string):
        for context in (owner, None):
            try:
                text = as_string(context)
            except Exception:  # noqa: BLE001, S112 - try the next context
                continue
            if isinstance(text, str):
                return text
    raise WriteRefused(
        f'a statement the guard cannot read ({type(statement).__name__})'
    )


def _guarded(
    original: Any, check: Callable[[str], None], spark: bool = False
) -> Any:
    @functools.wraps(original)
    def guarded(self: Any, *arguments: Any, **keywords: Any) -> Any:
        if spark and (len(arguments) > 2 or set(keywords) - _SPARK_KEYWORDS):
            raise WriteRefused(
                'Spark SQL with arguments formatted into its text'
            )
        if arguments:
            statement = arguments[0]
        else:
            given = [k for k in _STATEMENT_KEYWORDS if k in keywords]
            if len(given) != 1:
                raise WriteRefused('a statement the guard cannot find')
            statement = keywords[given[0]]
        check(_text(self, statement))
        return original(self, *arguments, **keywords)

    return _keeping(guarded)


def _sql_texts(message: Any) -> list[str]:
    r"""Every SQL text in a protobuf ``message``: each field named ``sql``,
    a text or a SQL relation's ``query``, at any depth."""
    found: list[str] = []
    for field, value in message.ListFields():
        if hasattr(value, 'ListFields') or isinstance(
            value, (str, bytes, int, float, bool)
        ):
            values = [value]
        elif callable(getattr(value, 'values', None)):
            values = list(value.values())  # a map's values
        else:
            values = list(value)  # a repeated field
        for each in values:
            if field.name == 'sql' and isinstance(each, str):
                found.append(each)
            elif hasattr(each, 'ListFields'):
                query = getattr(each, 'query', None)
                if field.name == 'sql' and isinstance(query, str):
                    found.append(query)
                found.extend(_sql_texts(each))
    return found


def _plan_request(original: Any, check: Callable[[str], None]) -> Any:
    r"""Spark Connect's execution boundary, admitting a request's plan only
    as a read (a relation) or a SQL command, every SQL text in it checked."""

    @functools.wraps(original)
    def guarded(
        self: Any, request: Any, *arguments: Any, **keywords: Any
    ) -> Any:
        try:
            plan = request.plan
            kind = plan.WhichOneof('op_type')
            command = (
                plan.command.WhichOneof('command_type')
                if kind == 'command'
                else None
            )
            texts = _sql_texts(plan)
        except Exception:  # noqa: BLE001 - a plan it cannot read is refused
            raise WriteRefused(
                'a Spark Connect plan the guard cannot read'
            ) from None
        if kind not in ('root', 'command') or (
            kind == 'command' and (command != 'sql_command' or not texts)
        ):
            raise WriteRefused(
                f'a Spark Connect {command or kind}, which is no read or SQL command'
            )
        for text in texts:
            check(text)
        return original(self, request, *arguments, **keywords)

    return _keeping(guarded)


def _refused(name: str) -> Any:
    def refused(*arguments: Any, **keywords: Any) -> Any:
        raise WriteRefused(f'{name}: writes no statement the guard can read')

    return _keeping(refused)


def _refusing(where: str) -> Callable[[Any], Any]:
    r"""What replaces a refused attribute, of the same kind."""

    def make(current: Any) -> Any:
        if isinstance(current, property):
            return property(_refused(where))
        if isinstance(current, classmethod):
            return classmethod(_refused(where))
        if isinstance(current, staticmethod):
            return staticmethod(_refused(where))
        return _refused(where)

    return make


def _marked(current: Any) -> bool:
    for each in (
        current,
        getattr(current, 'fget', None),
        getattr(current, '__func__', None),
    ):
        if getattr(each, _MARK, False):
            return True
    return False


def _wrap_one(cls: type, attribute: str, make: Callable[[Any], Any]) -> None:
    try:
        current = inspect.getattr_static(cls, attribute)
    except AttributeError:
        return
    if _marked(current):
        return
    _INSTALLED.append((cls, attribute, cls.__dict__.get(attribute, _ABSENT)))
    setattr(cls, attribute, make(current))


def _wrap(cls: type, attribute: str, make: Callable[[Any], Any]) -> None:
    r"""``attribute`` of ``cls`` and of each subclass resolving it to
    something else, replaced by ``make`` of what it resolved to; a
    subclass defined later gets the same (:func:`_hook`)."""
    seen: set[type] = set()
    pending = [cls]
    while pending:
        each = pending.pop()
        if each in seen:
            continue
        seen.add(each)
        pending.extend(each.__subclasses__())
        _wrap_one(each, attribute, make)
    _RULES.append((cls, attribute, make))
    _hook(cls)


def _hook(cls: type) -> None:
    r"""``cls``'s subclasses, when defined, wrapped by every rule of
    :data:`_RULES` that covers them."""
    previous = cls.__dict__.get('__init_subclass__', _ABSENT)
    if previous is not _ABSENT and _marked(previous):
        return

    def __init_subclass__(sub: type, **keywords: Any) -> None:
        if previous is _ABSENT:
            super(cls, sub).__init_subclass__(**keywords)
        else:
            previous.__get__(None, sub)(**keywords)
        for base, attribute, make in list(_RULES):
            if issubclass(sub, base):
                _wrap_one(sub, attribute, make)

    setattr(__init_subclass__, _MARK, True)
    _INSTALLED.append((cls, '__init_subclass__', previous))
    cls.__init_subclass__ = classmethod(__init_subclass__)  # type: ignore[assignment]


def _closed(dialect: str) -> Callable[[str], None]:
    def check(sql: str) -> None:
        raise WriteRefused(
            f'no {dialect} test schema: every {dialect} statement refused'
        )

    return check


def install(guards: Mapping[str, WriteGuard | None]) -> Callable[[], None]:
    r"""Every driver entry point (:data:`ENTRY_POINTS`) checked by its
    dialect's guard in ``guards`` (none, or one missing: every statement
    of that dialect refused), every :data:`REFUSED` interface and every
    :data:`REFUSED_CLASSES` writer refused, for every driver importable
    here; returns what undoes it. A driver module that imports but lacks
    an entry point's class stops the install, as its entry points may
    have moved."""
    if _INSTALLED:
        raise RuntimeError('the live write guard is installed already')
    REFUSALS.clear()
    try:
        for module_name, class_name, attributes, dialect in ENTRY_POINTS:
            cls = _class(
                module_name,
                class_name,
                required=(module_name, class_name) not in _OPTIONAL,
            )
            if cls is None:
                continue
            guard = guards.get(dialect)
            check = guard.check if guard is not None else _closed(dialect)
            spark = module_name.startswith('pyspark.')
            for attribute in attributes:
                _wrap(
                    cls,
                    attribute,
                    lambda f, c=check, k=spark: _guarded(f, c, k),
                )
        module_name, class_name, attributes = SPARK_CONNECT
        cls = _class(module_name, class_name)
        for attribute in attributes if cls is not None else ():
            if not hasattr(cls, attribute):
                raise RuntimeError(
                    f'{class_name} has no {attribute}: the write guard cannot '
                    'cover this Spark Connect version'
                )
            guard = guards.get('databricks')
            check = guard.check if guard is not None else _closed('databricks')
            _wrap(cls, attribute, lambda f, c=check: _plan_request(f, c))
        for module_name, class_name, attributes in REFUSED:
            cls = _class(module_name, class_name)
            for attribute in attributes if cls is not None else ():
                _wrap(cls, attribute, _refusing(f'{class_name}.{attribute}'))
        for module_name, class_name in REFUSED_CLASSES:
            cls = _class(module_name, class_name, required=False)
            if cls is None:
                continue
            for attribute in dir(cls):
                if attribute.startswith('_'):
                    continue
                current = inspect.getattr_static(cls, attribute)
                if callable(current) or isinstance(
                    current, (property, classmethod, staticmethod)
                ):
                    _wrap(
                        cls, attribute, _refusing(f'{class_name}.{attribute}')
                    )
    except BaseException:
        uninstall()
        raise
    return uninstall


def _class(
    module_name: str, class_name: str, required: bool = True
) -> type | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - a driver that cannot import runs nothing
        return None
    cls = getattr(module, class_name, None)
    if isinstance(cls, type):
        return cls
    if not required:
        return None
    raise RuntimeError(
        f'{module_name} has no class {class_name}: the write guard cannot '
        'cover this driver version'
    )


def uninstall() -> None:
    r"""Undoes :func:`install`, the last change first."""
    _RULES.clear()
    while _INSTALLED:
        cls, attribute, previous = _INSTALLED.pop()
        if previous is _ABSENT:
            delattr(cls, attribute)
        else:
            setattr(cls, attribute, previous)


def from_environment() -> dict[str, WriteGuard | None]:
    r"""Each dialect's guard from the environment, or ``None`` where it
    names no test schema (that dialect then refuses every statement)."""
    guards: dict[str, WriteGuard | None] = {}
    for dialect in DIALECTS:
        try:
            guards[dialect] = WriteGuard(dialect)
        except WriteRefused:
            guards[dialect] = None
    return guards
