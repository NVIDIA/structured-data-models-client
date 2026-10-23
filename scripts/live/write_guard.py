# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The live write guard: every live write, DDL included, goes only to
the one test schema of its warehouse; anything else is refused before the
driver sees it (KO3-51, the owner's rule).

    guard = WriteGuard('snowflake')   # or 'databricks'
    connection = guard_connection(connection, guard)

The test schema comes from the constructor or from the environment:
``SDM_LIVE_<DIALECT>_TEST_DATABASE`` and ``SDM_LIVE_<DIALECT>_TEST_SCHEMA``
(``<DIALECT>`` is ``SNOWFLAKE`` or ``DATABRICKS``), else
``SDM_LIVE_TEST_DATABASE`` and ``SDM_LIVE_TEST_SCHEMA``. No resource name is
written here.

The rules, statement by statement (multi-statement text is split on its
semicolons; comments, string literals and ``$$`` bodies are skipped):

- Reads run anywhere: SELECT, WITH and VALUES (with no write verb outside
  parentheses), SHOW, DESCRIBE, EXPLAIN, LIST, GET, USE, SET, UNSET, ALTER
  SESSION and transaction control.
- Every other statement must name each target fully qualified as
  ``<database>.<schema>.<object>`` in the test schema: CREATE and REPLACE,
  INSERT (every INTO target), DROP, ALTER (with its RENAME TO and SWAP WITH
  targets), DELETE, UPDATE, MERGE, TRUNCATE, COPY INTO (a table or a
  ``@stage``; a string location is refused), PUT and REMOVE (a
  ``@database.schema.stage`` only), GRANT, REVOKE and COMMENT ON,
  OPTIMIZE, VACUUM, ANALYZE, REFRESH, RESTORE, UNDROP and MSCK.
- ``CREATE SCHEMA IF NOT EXISTS <the test schema>`` is allowed; every other
  statement on a schema, database or catalog as a whole is refused (so
  ``DROP SCHEMA <the test schema>`` is).
- On Snowflake, ``CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('<the test
  schema>', ...)`` is allowed: it creates one semantic view in the schema
  it names (the contract loader's semantic views).
- Anything else (another CALL, EXECUTE IMMEDIATE, scripting blocks,
  IDENTIFIER(...) as a target, a Databricks LOCATION clause, a statement
  not recognised) is refused, so a parsing gap fails closed.

Snowflake reads an unquoted name upper-cased and a quoted one exactly;
Unity Catalog reads names case-insensitively, in backticks or bare
(a double-quoted text is a string there, so never a name).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

DIALECTS = ('databricks', 'snowflake')


class WriteRefused(PermissionError):
    r"""A statement the guard refuses; the message names the rule, and the
    statement's first words, never a bound value."""


@dataclass(frozen=True)
class Token:
    kind: str  # 'word', 'name' (quoted), 'string', 'number', 'punct'
    text: str


_WORD = re.compile(r'[A-Za-z_][A-Za-z0-9_$]*')
_NUMBER = re.compile(r'[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?')


def tokens(sql: str, dialect: str) -> list[Token]:
    r"""``sql``'s tokens in ``dialect``: comments dropped; a quoted
    identifier's text unquoted; strings and ``$$`` bodies kept as one
    ``string`` token each. Unterminated quotes or comments are refused."""
    out: list[Token] = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c.isspace():
            i += 1
        elif sql.startswith('--', i) or (
            # Snowflake's // comment; file:// in a PUT is a path.
            dialect == 'snowflake'
            and sql.startswith('//', i)
            and sql[i - 1 : i] not in (':', '/')
        ):
            end = sql.find('\n', i)
            i = n if end < 0 else end + 1
        elif sql.startswith('/*', i):
            end = sql.find('*/', i + 2)
            if end < 0:
                raise WriteRefused('an unterminated comment')
            i = end + 2
        elif sql.startswith('$$', i):
            end = sql.find('$$', i + 2)
            if end < 0:
                raise WriteRefused('an unterminated $$ body')
            out.append(Token('string', sql[i + 2 : end]))
            i = end + 2
        elif c == "'" or (c == '"' and dialect == 'databricks'):
            text, i = _quoted(sql, i, c, backslash=True)
            out.append(Token('string', text))
        elif c == '"' or (c == '`' and dialect == 'databricks'):
            text, i = _quoted(sql, i, c, backslash=False)
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


def _quoted(
    sql: str, start: int, quote: str, backslash: bool
) -> tuple[str, int]:
    r"""The text quoted by ``quote`` from ``start``, a doubled quote read as
    one (and, in a string, a backslash escaping the next character)."""
    text: list[str] = []
    i = start + 1
    while i < len(sql):
        c = sql[i]
        if backslash and c == '\\' and i + 1 < len(sql):
            text.append(sql[i + 1])
            i += 2
        elif c == quote:
            if sql.startswith(quote * 2, i):
                text.append(quote)
                i += 2
            else:
                return ''.join(text), i + 1
        else:
            text.append(c)
            i += 1
    raise WriteRefused('an unterminated quoted text')


def statements(sql: str, dialect: str) -> list[list[Token]]:
    r"""``sql`` split into its statements' tokens, empty ones dropped."""
    found: list[list[Token]] = [[]]
    for token in tokens(sql, dialect):
        if token == Token('punct', ';'):
            found.append([])
        else:
            found[-1].append(token)
    return [statement for statement in found if statement]


def _words(statement: list[Token], count: int) -> list[str]:
    return [
        token.text.upper() if token.kind == 'word' else ''
        for token in statement[:count]
    ]


#: Statements that read, by their first word (ALTER SESSION and the
#: transaction words with their own checks).
READS = {
    'SELECT', 'WITH', 'VALUES', 'SHOW', 'DESCRIBE', 'DESC', 'EXPLAIN',
    'LIST', 'LS', 'GET', 'USE', 'SET', 'UNSET',
}  # fmt: skip
TRANSACTION = {'COMMIT', 'ROLLBACK', 'BEGIN', 'START'}
#: Verbs that write, refused in a read outside parentheses.
WRITE_VERBS = {
    'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'CREATE', 'REPLACE', 'DROP',
    'ALTER', 'TRUNCATE', 'GRANT', 'REVOKE', 'COPY', 'PUT', 'REMOVE',
    'CALL', 'EXECUTE', 'UNDROP', 'OPTIMIZE', 'VACUUM', 'RESTORE',
}  # fmt: skip
#: Words between a verb and its target that name the object's kind or
#: modify it; any other word there is refused.
KIND_WORDS = {
    'OR', 'REPLACE', 'TEMPORARY', 'TEMP', 'TRANSIENT', 'VOLATILE', 'LOCAL',
    'GLOBAL', 'SECURE', 'RECURSIVE', 'MATERIALIZED', 'STREAMING', 'HYBRID',
    'DYNAMIC', 'ICEBERG', 'EXTERNAL', 'MANAGED', 'TABLE', 'VIEW', 'SEMANTIC',
    'METRIC', 'STAGE', 'VOLUME', 'FUNCTION', 'PROCEDURE', 'SEQUENCE',
    'FILE', 'FORMAT', 'STREAM', 'TASK', 'PIPE', 'TAG', 'MASKING',
    'ROW', 'ACCESS', 'POLICY', 'ALERT', 'AGGREGATION', 'PROJECTION',
}  # fmt: skip
#: Kinds naming a schema, database or catalog as a whole.
CONTAINERS = {'SCHEMA', 'DATABASE', 'CATALOG', 'NAMESPACE'}
#: Verbs whose target follows their kind words, and which are refused
#: when no kind words lead to a qualified name.
SIMPLE_TARGET = {
    'DROP', 'ALTER', 'TRUNCATE', 'OPTIMIZE', 'VACUUM', 'ANALYZE', 'REFRESH',
    'RESTORE', 'UNDROP', 'UPDATE',
}  # fmt: skip


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
        if not database or not schema:
            raise WriteRefused(
                'the write guard names no test schema: set '
                'SDM_LIVE_TEST_DATABASE and SDM_LIVE_TEST_SCHEMA'
            )
        self.dialect = dialect
        self.database = database
        self.schema = schema

    # Names.

    def _same(self, read: tuple[str, bool], stored: str) -> bool:
        text, quoted = read
        if self.dialect == 'databricks':
            return text.lower() == stored.lower()
        return (text if quoted else text.upper()) == stored

    def _in_schema(self, parts: list[tuple[str, bool]]) -> bool:
        return (
            len(parts) == 3
            and self._same(parts[0], self.database)
            and self._same(parts[1], self.schema)
        )

    def _is_schema(self, parts: list[tuple[str, bool]]) -> bool:
        return (
            len(parts) == 2
            and self._same(parts[0], self.database)
            and self._same(parts[1], self.schema)
        )

    @staticmethod
    def _name(
        statement: list[Token], at: int
    ) -> tuple[list[tuple[str, bool]], int]:
        r"""The dotted name at ``at``: its parts as (text, quoted), and the
        index after it; no parts when no name starts there."""
        parts: list[tuple[str, bool]] = []
        i = at
        while i < len(statement):
            token = statement[i]
            if token.kind not in ('word', 'name'):
                break
            if token.kind == 'word' and token.text.upper() == 'IDENTIFIER':
                raise WriteRefused('IDENTIFIER(...) as a write target')
            parts.append((token.text, token.kind == 'name'))
            i += 1
            if i < len(statement) and statement[i] == Token('punct', '.'):
                i += 1
            else:
                break
        return parts, i

    def _target(self, statement: list[Token], at: int, what: str) -> int:
        r"""Refuses unless a name in the test schema starts at ``at``;
        returns the index after it."""
        parts, after = self._name(statement, at)
        if not self._in_schema(parts):
            raise WriteRefused(
                f'{what}: a write target outside the test schema, or not '
                'written <database>.<schema>.<object>'
            )
        return after

    def _stage(self, statement: list[Token], at: int, what: str) -> None:
        if at >= len(statement) or statement[at] != Token('punct', '@'):
            raise WriteRefused(f'{what}: no @<database>.<schema>.<stage>')
        self._target(statement, at + 1, what)

    def _after_kind(self, statement: list[Token], at: int, what: str) -> int:
        r"""The index of the target after ``at``'s kind words and IF [NOT]
        EXISTS; a schema, database or catalog as a whole is refused."""
        i = at
        while i < len(statement) and statement[i].kind == 'word':
            word = statement[i].text.upper()
            if word in CONTAINERS:
                raise WriteRefused(
                    f'{what}: a schema, database or catalog as a whole'
                )
            if word == 'IF':
                i += 1
                if i < len(statement) and _words(statement[i:], 1) == ['NOT']:
                    i += 1
                if _words(statement[i:], 1) != ['EXISTS']:
                    raise WriteRefused(
                        f'{what}: an IF that is not IF [NOT] EXISTS'
                    )
                i += 1
                break
            if word not in KIND_WORDS:
                break
            i += 1
        return i

    # Statements.

    def check(self, sql: str) -> None:
        r"""Refuses ``sql`` (:class:`WriteRefused`) unless each of its
        statements reads or writes only in the test schema."""
        if not isinstance(sql, str):
            raise WriteRefused('a statement that is not text')
        for statement in statements(sql, self.dialect):
            self._statement(statement)

    def _statement(self, statement: list[Token]) -> None:
        first = statement[0]
        if first.kind != 'word':
            raise WriteRefused('a statement not recognised')
        verb = first.text.upper()
        what = ' '.join(
            t.text for t in statement[:2] if t.kind == 'word'
        ).upper()
        if verb in READS:
            self._read(statement, what)
            return
        if verb in TRANSACTION:
            self._transaction(statement, what)
            return
        if verb == 'ALTER' and _words(statement, 2)[1:] == ['SESSION']:
            return
        if verb in ('CREATE', 'REPLACE'):
            self._create(statement, what)
        elif verb == 'INSERT':
            self._insert(statement, what)
        elif verb in ('DELETE', 'MERGE'):
            expected = 'FROM' if verb == 'DELETE' else 'INTO'
            if _words(statement, 2)[1:] != [expected]:
                raise WriteRefused(f'{what}: not {verb} {expected} <target>')
            self._target(statement, 2, what)
        elif verb == 'MSCK':
            if _words(statement, 3)[1:] != ['REPAIR', 'TABLE']:
                raise WriteRefused(f'{what}: not MSCK REPAIR TABLE')
            self._target(statement, 3, what)
        elif verb in SIMPLE_TARGET:
            after = self._target(
                statement, self._after_kind(statement, 1, what), what
            )
            if verb == 'ALTER':
                self._alter_targets(statement, after, what)
        elif verb == 'COPY':
            self._copy(statement, what)
        elif verb in ('PUT', 'REMOVE', 'RM'):
            at = (
                1
                if verb != 'PUT'
                else next(
                    (
                        i
                        for i, t in enumerate(statement)
                        if t == Token('punct', '@')
                    ),
                    len(statement),
                )
            )
            self._stage(statement, at, what)
        elif verb in ('GRANT', 'REVOKE', 'COMMENT'):
            self._on(statement, what)
        elif verb == 'CALL' and self.dialect == 'snowflake':
            self._call(statement, what)
        else:
            raise WriteRefused(f'{what}: a statement not recognised')
        if self.dialect == 'databricks' and any(
            t.kind == 'word' and t.text.upper() == 'LOCATION' for t in statement
        ):
            raise WriteRefused(
                f'{what}: a LOCATION writes outside the test schema'
            )

    def _read(self, statement: list[Token], what: str) -> None:
        r"""Refuses a query (SELECT, WITH, VALUES) holding a write verb
        outside parentheses (``WITH ... INSERT``); a verb's name called as
        a function (``REPLACE(...)``) is not one."""
        if statement[0].text.upper() not in ('SELECT', 'WITH', 'VALUES'):
            return
        depth = 0
        for i, token in enumerate(statement):
            if token == Token('punct', '('):
                depth += 1
            elif token == Token('punct', ')'):
                depth -= 1
            elif (
                depth == 0
                and token.kind == 'word'
                and token.text.upper() in WRITE_VERBS
                and statement[i + 1 : i + 2] != [Token('punct', '(')]
            ):
                raise WriteRefused(f'{what}: a query that writes')

    def _transaction(self, statement: list[Token], what: str) -> None:
        if len(statement) == 1 or (
            len(statement) == 2
            and _words(statement, 2)[1] in ('WORK', 'TRANSACTION')
        ):
            return
        raise WriteRefused(
            f'{what}: a scripting block or transaction form not recognised'
        )

    def _create(self, statement: list[Token], what: str) -> None:
        words = _words(statement, 6)
        kinds = [w for w in words[1:] if w not in ('OR', 'REPLACE')]
        if kinds and kinds[0] in CONTAINERS:
            # CREATE SCHEMA IF NOT EXISTS <the test schema>, and nothing else.
            at = words.index(kinds[0])
            if kinds[0] == 'SCHEMA' and _words(statement[at + 1 :], 3) == [
                'IF',
                'NOT',
                'EXISTS',
            ]:
                parts, after = self._name(statement, at + 4)
                if self._is_schema(parts) and after == len(statement):
                    return
            raise WriteRefused(
                f'{what}: a schema, database or catalog as a whole'
            )
        self._target(statement, self._after_kind(statement, 1, what), what)

    def _insert(self, statement: list[Token], what: str) -> None:
        targets = 0
        depth = 0
        words = _words(statement, 3)
        if words[1] == 'OVERWRITE' and words[2] not in ('INTO',):
            if words[2] in ('DIRECTORY', 'LOCAL'):
                raise WriteRefused(f'{what}: an INSERT OVERWRITE DIRECTORY')
            at = 3 if words[2] == 'TABLE' else 2
            self._target(statement, at, what)
            targets += 1
        for i, token in enumerate(statement):
            if token == Token('punct', '('):
                depth += 1
            elif token == Token('punct', ')'):
                depth -= 1
            elif (
                depth == 0
                and token.kind == 'word'
                and token.text.upper() == 'INTO'
            ):
                at = i + 1
                if _words(statement[at:], 1) == ['TABLE']:
                    at += 1
                self._target(statement, at, what)
                targets += 1
        if not targets:
            raise WriteRefused(f'{what}: an INSERT with no target')

    def _alter_targets(
        self, statement: list[Token], at: int, what: str
    ) -> None:
        for i in range(at, len(statement) - 1):
            pair = _words(statement[i : i + 2], 2)
            if pair in (['RENAME', 'TO'], ['SWAP', 'WITH']):
                self._target(statement, i + 2, what)

    def _copy(self, statement: list[Token], what: str) -> None:
        if _words(statement, 2)[1:] != ['INTO']:
            raise WriteRefused(f'{what}: not COPY INTO <target>')
        if len(statement) > 2 and statement[2] == Token('punct', '@'):
            self._stage(statement, 2, what)
        else:
            self._target(statement, 2, what)

    def _on(self, statement: list[Token], what: str) -> None:
        on = next(
            (
                i
                for i, t in enumerate(statement)
                if t.kind == 'word' and t.text.upper() == 'ON'
            ),
            None,
        )
        if on is None or (statement[0].text.upper() == 'COMMENT' and on != 1):
            raise WriteRefused(f'{what}: no ON <object>')
        if _words(statement[on + 1 :], 1)[0:1] in (['ALL'], ['FUTURE']):
            raise WriteRefused(f'{what}: ON ALL or FUTURE objects')
        self._target(statement, self._after_kind(statement, on + 1, what), what)

    def _call(self, statement: list[Token], what: str) -> None:
        if not (
            len(statement) >= 4
            and statement[1].kind == 'word'
            and statement[1].text.upper()
            == 'SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML'
            and statement[2] == Token('punct', '(')
            and statement[3].kind == 'string'
        ):
            raise WriteRefused(f'{what}: a CALL')
        named = statements(statement[3].text, self.dialect)
        parts, after = self._name(named[0], 0) if len(named) == 1 else ([], 0)
        if not self._is_schema(parts) or after != len(named[0]):
            raise WriteRefused(
                f'{what}: a semantic view outside the test schema'
            )


class _Cursor:
    def __init__(self, cursor: Any, guard: WriteGuard) -> None:
        self._cursor = cursor
        self._guard = guard

    def execute(self, sql: str, *arguments: Any, **keywords: Any) -> Any:
        self._guard.check(sql)
        result = self._cursor.execute(sql, *arguments, **keywords)
        return self if result is self._cursor else result

    def executemany(self, sql: str, *arguments: Any, **keywords: Any) -> Any:
        self._guard.check(sql)
        result = self._cursor.executemany(sql, *arguments, **keywords)
        return self if result is self._cursor else result

    def execute_async(self, sql: str, *arguments: Any, **keywords: Any) -> Any:
        self._guard.check(sql)
        return self._cursor.execute_async(sql, *arguments, **keywords)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._cursor)

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: Any) -> None:
        self._cursor.close()


class _Connection:
    def __init__(self, connection: Any, guard: WriteGuard) -> None:
        self._connection = connection
        self._guard = guard

    def cursor(self, *arguments: Any, **keywords: Any) -> _Cursor:
        return _Cursor(
            self._connection.cursor(*arguments, **keywords), self._guard
        )

    def execute(self, sql: str, *arguments: Any, **keywords: Any) -> Any:
        self._guard.check(sql)
        return self._connection.execute(sql, *arguments, **keywords)

    def execute_string(self, sql: str, *arguments: Any, **keywords: Any) -> Any:
        self._guard.check(sql)
        return self._connection.execute_string(sql, *arguments, **keywords)

    def execute_stream(self, *arguments: Any, **keywords: Any) -> Any:
        raise WriteRefused(
            'execute_stream is refused: its text is not read ahead'
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    def __enter__(self) -> _Connection:
        return self

    def __exit__(self, *exc: Any) -> None:
        self._connection.close()


def guard_connection(connection: Any, guard: WriteGuard) -> Any:
    r"""``connection`` (a Snowflake or Databricks SQL DB-API connection),
    each statement its cursors' ``execute``, ``executemany`` and
    ``execute_async`` and its ``execute`` and ``execute_string`` run
    checked by ``guard`` first; ``execute_stream`` refused."""
    return _Connection(connection, guard)
