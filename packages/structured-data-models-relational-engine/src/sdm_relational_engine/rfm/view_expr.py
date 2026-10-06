# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Checks on SQL expressions taken from a metric or semantic view.

A view is authored by a third party, yet its expressions are spliced verbatim
into sampler queries and run under the caller's warehouse credentials.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

_UNSAFE_EXPR_TOKENS = (';', '--', '/*', '*/', '|>', '->>')
_EXPR_QUOTE_DELIMITERS = ('$$', "'", '"', '`')


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    start: int
    end: int

    @property
    def folded(self) -> str:
        return self.text.lower() if self.kind == 'word' else ''


# How each dialect splits an expression into tokens, in the order tried. Every
# check reads quoted text through these, so no two can disagree about where a
# literal ends. A `string` is data, and a `quoted` token is a name.
_ESCAPED_STRING = r"'(?:''|\\[\s\S]|[^'\\])*'"
_ESCAPED_DOUBLE = r'"(?:""|\\[\s\S]|[^"\\])*"'
_BACKTICKED = r'`(?:``|[^`])*`'
_DOLLARS = r'\$\$[\s\S]*?\$\$'
_WORD = r'[A-Za-z_][\w$]*'
# Databricks reads a run of letters, digits and `_` as a name unless the
# whole run is a number, as `1e5`, `2d` and `10L` are: `2024_total` names a
# column on a DBSQL 2026.38 warehouse.
_WHOLE_NUMBER = r'(?i:\d+(?:e\d+)?(?:bd|d|f)?|\d+[lsy])(?![A-Za-z0-9_])'
BARE_NAME = rf'(?!{_WHOLE_NUMBER})[A-Za-z0-9_]+'
# A signed exponent outruns any name, and a decimal point ends a number
# only where no name character follows, as in Spark's lexer.
_DATABRICKS_NUMBER = (
    r'(?i:\d+e[+-]\d+(?:bd|d|f)?'
    r'|(?:\d+\.\d*|\.\d+)(?:e[+-]?\d+)?(?:bd|d|f)?(?![A-Za-z0-9_]))'
    rf'|{_WHOLE_NUMBER}'
)
# Words and numbers elsewhere. A word may hold `$`, as a Snowflake name
# does, so `a$$` opens no literal.
_LOOSE_WORDS = (
    ('open', r'/\*'),
    ('number', r'(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[A-Za-z]*'),
    ('word', _WORD),
)
_LEXEMES = {
    # A raw string, `r'...'`, takes no escapes: `r'x\'` ends at its second
    # quote. Double quotes delimit a string, not a name. A line comment goes
    # on past a backslash ending its line, as Databricks reads it.
    'databricks': (
        ('comment', r'--(?:\\\n|[^\r\n])*'),
        (
            'string',
            rf"[rR]'[^']*'|[rR]\"[^\"]*\"|{_ESCAPED_STRING}|{_ESCAPED_DOUBLE}",
        ),
        ('quoted', _BACKTICKED),
        ('open', r'/\*'),
        ('number', _DATABRICKS_NUMBER),
        ('word', BARE_NAME),
    ),
    # Double quotes delimit a name, which takes no escapes. Snowflake
    # refuses a backtick, so reading one as a quote hides nothing from it.
    'snowflake': (
        ('comment', r'(?:--|//)[^\r\n]*'),
        ('string', rf'{_DOLLARS}|{_ESCAPED_STRING}'),
        ('quoted', rf'"(?:""|[^"])*"|{_BACKTICKED}'),
        *_LOOSE_WORDS,
    ),
    # What a check that knows no dialect reads as quoted.
    None: (
        ('comment', r'--[^\r\n]*'),
        ('string', rf'{_DOLLARS}|{_ESCAPED_STRING}|{_ESCAPED_DOUBLE}'),
        ('quoted', _BACKTICKED),
        *_LOOSE_WORDS,
    ),
}
_LEXEMES_EVERYWHERE = (
    ('punct', r'<=>|<=|>=|<>|::|->|=>|[.:(),\[\]<>]'),
    ('space', r'\s+'),
    ('other', r'[\s\S]'),
)


def _lexer(
    lexemes: tuple[tuple[str, str], ...],
) -> tuple[re.Pattern[str], list[str]]:
    r"""One pattern trying each of ``lexemes`` in turn, and their kinds by
    group.
    """
    pattern = '|'.join(f'({lexeme})' for _, lexeme in lexemes)
    return re.compile(pattern), [kind for kind, _ in lexemes]


_LEXERS = {
    dialect: _lexer((*own, *_LEXEMES_EVERYWHERE))
    for dialect, own in _LEXEMES.items()
}


def _comment_end(expr: str, start: int, nested: bool) -> int | None:
    r"""Where the ``/*`` comment at ``start`` ends, or ``None`` if it never
    does. Databricks nests one such comment in another.
    """
    depth, at = 1, start + 2
    while depth:
        close = expr.find('*/', at)
        if close < 0:
            return None
        open_ = expr.find('/*', at, close) if nested else -1
        depth += 1 if open_ >= 0 else -1
        at = open_ + 2 if open_ >= 0 else close + 2
    return at


def _lex(expr: str, dialect: str | None = 'databricks') -> list[_Token]:
    r"""The tokens of ``expr`` as ``dialect`` reads them, spaces left out.

    A quote or ``/*`` that is never closed is a token of kind ``other``.
    """
    pattern, kinds = _LEXERS[dialect]
    tokens: list[_Token] = []
    at = 0
    while at < len(expr):
        # Some lexeme matches any character, so a match always exists.
        match = pattern.match(expr, at)
        assert match is not None
        kind, end = kinds[(match.lastindex or 1) - 1], match.end()
        if kind == 'open':
            close = _comment_end(expr, at, nested=dialect == 'databricks')
            kind, end = ('comment', close) if close else ('other', at + 1)
        if kind != 'space':
            tokens.append(_Token(kind, expr[at:end], at, end))
        at = end
    return tokens


def strip_comments(expr: str) -> str:
    r"""A Databricks expression with each comment replaced by a space.

    A comment hides nothing from a check of what is left, and what is left
    is what runs, so a call written inside one is never executed.
    """
    out: list[str] = []
    last = 0
    for token in _lex(expr):
        if token.kind == 'comment':
            out += [expr[last : token.start], ' ']
            last = token.end
    out.append(expr[last:])
    return ''.join(out).strip()


def _masked(expr: str, dialect: str | None, mask: str) -> str:
    r"""``expr`` with each string literal and quoted name replaced by
    ``mask``.
    """
    out: list[str] = []
    last = 0
    for token in _lex(expr, dialect):
        if token.kind in ('string', 'quoted'):
            out += [expr[last : token.start], mask]
            last = token.end
    out.append(expr[last:])
    return ''.join(out)


def string_literals(expr: str) -> list[tuple[int, int]]:
    r"""Where each string literal of a Databricks expression starts and
    ends.
    """
    return [
        (token.start, token.end)
        for token in _lex(expr)
        if token.kind == 'string'
    ]


_UNSAFE_EXPR_KEYWORDS = re.compile(
    r'(?<![\w.])'
    r'(?:(?:select|table|update|delete|merge|create|alter|drop|grant|revoke'
    r'|execute)\b'
    r'|(?:insert|truncate)\b(?!\s*\())',
    flags=re.IGNORECASE,
)
# Databricks runs `(FROM t |> ...)` and `(TABLE t)` as sub-queries.
_QUERY_START = re.compile(r'(?:^|\()\s*from\b', flags=re.IGNORECASE)


def _unsafe_expr_reason(expr: str, dialect: str | None = None) -> str | None:
    r"""Return why a remote view expression is unsafe, or ``None``.

    A scalar expression never needs a statement separator, a comment or a
    sub-query, so those are refused while arithmetic, casts, ``CASE`` and
    function calls are left alone. Which functions may be called is decided
    separately by :func:`_untrusted_call_reason`.

    Keywords are matched only outside string literals and quoted identifiers,
    since a keyword inside one is data rather than SQL and rejecting it would
    drop a legitimate column. The token check deliberately stays on the raw
    expression: quoting rules vary by dialect, so a separator or comment must
    not become invisible just because this function believes it is quoted.

    Deciding what counts as quoted is itself dialect-specific, and getting it
    wrong defeats the guard. Snowflake also has ``$$...$$``, both warehouses
    honour backslash escapes inside a literal, and a Databricks raw string,
    ``r'...'``, takes none. A payload using ``$$``, ``\'`` or ``r'\'`` shifts
    where a reader missing that form sees the literal end, and can smuggle a
    sub-query through the gap, so each dialect is read as it lexes. A
    remaining delimiter means this function and the warehouse disagree about
    string boundaries, so it fails closed.

    ``INSERT`` and ``TRUNCATE`` are also ordinary Snowflake scalar functions,
    and as statements are always followed by ``INTO``/``TABLE`` rather than by
    ``(``, so a call cannot be one. A keyword directly behind a ``.`` is the
    trailing part of a qualified name (``ORDERS.MERGE``), which also cannot
    begin a statement. ``SELECT`` may legally be followed by ``(``, so it is
    always refused. Databricks also reads a table without ``SELECT``: through
    ``TABLE t``, a query opening with ``FROM``, and the ``|>`` pipe, and
    Snowflake chains statements with ``->>``, so those are refused too.

    Parentheses must balance outside quoted text. An expression is spliced
    into a select list or a filter's ``WHERE (...)``, and a stray ``)`` would
    close that and let the rest append clauses.

    Args:
        expr: The expression as written in the view definition.
        dialect: ``'databricks'`` or ``'snowflake'`` to also refuse quoting
            that dialect does not have: ``$$`` on Databricks, and a backslash
            inside a Snowflake double-quoted name. ``None`` checks only what
            both share.
    """
    for token in _UNSAFE_EXPR_TOKENS:
        if token in expr:
            return f'it contains {token!r}'
    # Databricks has no `$$` literal, and quotes no name with a backslash.
    if dialect == 'databricks' and '$$' in expr:
        return "it contains '$$', which Databricks SQL does not quote with"
    # Snowflake also comments out the rest of a line with `//`.
    if dialect == 'snowflake' and '//' in expr:
        return "it contains '//'"
    if dialect == 'snowflake':
        for quoted in _lex(expr, dialect):
            if quoted.text.startswith('"') and '\\' in quoted.text:
                return 'it has a backslash inside a double-quoted name'
    unquoted = _masked(expr, dialect, ' ')
    for delimiter in _EXPR_QUOTE_DELIMITERS:
        if delimiter in unquoted:
            return (
                f'it leaves {delimiter!r} unpaired, so which part of it '
                f'is quoted text cannot be established'
            )
    depth = 0
    for char in unquoted:
        depth += {'(': 1, ')': -1}.get(char, 0)
        if depth < 0:
            return "it closes a '(' it never opened"
    if depth != 0:
        return "it leaves a '(' unclosed"
    if match := _UNSAFE_EXPR_KEYWORDS.search(unquoted):
        return f"it contains the SQL keyword '{match.group()}'"
    if _QUERY_START.search(unquoted):
        return 'it contains a sub-query'
    return None


# Pure, deterministic, built-in scalar functions, after the Databricks SQL
# built-in function reference. Checked against `SHOW SYSTEM FUNCTIONS`.
# Some can be costly (`repeat`, `sequence`) or read the session time zone
# (`from_unixtime`); docs/about/architecture.md says so, and they stay allowed.
_DATABRICKS_FUNCTIONS = frozenset(
    {
        # Conditional and null handling:
        'coalesce', 'decode', 'equal_null', 'greatest', 'if', 'iff',
        'ifnull', 'isnan', 'isnotnull', 'isnull', 'least', 'nanvl', 'nullif',
        'nullifzero', 'nvl', 'nvl2', 'zeroifnull',
        # String:
        'ascii', 'base64', 'bit_length', 'btrim', 'char', 'char_length',
        'character_length', 'charindex', 'chr', 'concat', 'concat_ws',
        'contains', 'crc32', 'elt', 'encode', 'endswith', 'find_in_set',
        'format_number', 'format_string', 'hash', 'hex', 'initcap', 'instr',
        'jaro_winkler_similarity', 'lcase', 'left', 'len', 'length',
        'levenshtein', 'locate', 'lower', 'lpad', 'ltrim', 'luhn_check',
        'mask', 'md5', 'octet_length', 'overlay', 'parse_url', 'position',
        'printf', 'quote', 'regexp_count', 'regexp_extract',
        'regexp_extract_all', 'regexp_instr', 'regexp_like', 'regexp_replace',
        'regexp_substr', 'repeat', 'replace', 'reverse', 'right', 'rlike',
        'rpad', 'rtrim', 'sentences', 'sha', 'sha1', 'sha2', 'soundex',
        'space', 'split', 'split_part', 'startswith', 'str_to_map', 'substr',
        'substring', 'substring_index', 'translate', 'trim', 'try_parse_url',
        'try_url_decode', 'ucase', 'unbase64', 'unhex', 'upper', 'url_decode',
        'url_encode', 'xxhash64',
        # Math:
        'abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh',
        'bin', 'bit_count', 'bit_get', 'bround', 'cbrt', 'ceil', 'ceiling',
        'conv', 'cos', 'cosh', 'cot', 'csc', 'degrees', 'div', 'e', 'exp',
        'expm1', 'factorial', 'floor', 'getbit', 'hypot', 'ln', 'log',
        'log10', 'log1p', 'log2', 'mod', 'negative', 'pi', 'pmod', 'positive',
        'pow', 'power', 'radians', 'rint', 'round', 'sec', 'shiftleft',
        'shiftright', 'shiftrightunsigned', 'sign', 'signum', 'sin', 'sinh',
        'sqrt', 'tan', 'tanh', 'try_add', 'try_divide', 'try_mod',
        'try_multiply', 'try_remainder', 'try_subtract', 'width_bucket',
        # Date and time:
        'add_months', 'convert_timezone', 'date_add', 'date_diff',
        'date_format', 'date_from_unix_date', 'date_part', 'date_sub',
        'date_trunc', 'dateadd', 'datediff', 'datepart', 'day', 'dayname',
        'dayofmonth', 'dayofweek', 'dayofyear', 'extract', 'from_unixtime',
        'from_utc_timestamp', 'hour', 'last_day', 'make_date',
        'make_dt_interval', 'make_interval', 'make_timestamp',
        'make_timestamp_ntz', 'make_ym_interval', 'minute', 'month',
        'monthname', 'months_between', 'next_day', 'quarter', 'second',
        'timestamp_micros', 'timestamp_millis', 'timestamp_seconds',
        'timestampadd', 'timestampdiff', 'to_date', 'to_timestamp',
        'to_timestamp_ltz', 'to_timestamp_ntz', 'to_unix_timestamp',
        'to_utc_timestamp', 'trunc', 'try_make_timestamp', 'try_to_date',
        'try_to_timestamp', 'unix_date', 'unix_micros', 'unix_millis',
        'unix_seconds', 'weekday', 'weekofyear', 'year',
        # Cast and conversion:
        'bigint', 'binary', 'boolean', 'cast', 'date', 'decimal', 'double',
        'float', 'int', 'smallint', 'string', 'timestamp', 'tinyint',
        'to_binary', 'to_char', 'to_number', 'to_varchar', 'try_cast',
        'try_to_binary', 'try_to_number', 'typeof',
        # Array, map and struct, with their lambda functions:
        'aggregate', 'array', 'array_append', 'array_compact',
        'array_contains', 'array_distinct', 'array_except', 'array_insert',
        'array_intersect', 'array_join', 'array_max', 'array_min',
        'array_position', 'array_prepend', 'array_remove', 'array_repeat',
        'array_size', 'array_sort', 'array_union', 'arrays_overlap',
        'arrays_zip', 'cardinality', 'element_at', 'exists', 'filter',
        'flatten', 'forall', 'get', 'map', 'map_concat', 'map_contains_key',
        'map_entries', 'map_filter', 'map_from_arrays', 'map_from_entries',
        'map_keys', 'map_values', 'map_zip_with', 'named_struct', 'reduce',
        'sequence', 'size', 'slice', 'sort_array', 'struct', 'transform',
        'transform_keys', 'transform_values', 'trim_array', 'try_element_at',
        'zip_with',
        # JSON and variant paths:
        'from_json', 'get_json_object', 'is_variant_null',
        'json_array_length', 'json_object_keys', 'parse_json',
        'schema_of_json', 'schema_of_variant', 'to_json', 'try_parse_json',
        'try_variant_get', 'variant_get',
    }
)  # fmt: skip

# The same, after the Snowflake built-in function reference.
_SNOWFLAKE_FUNCTIONS = frozenset(
    {
        # Conditional and null handling:
        'booland', 'boolnot', 'boolor', 'boolxor', 'coalesce', 'decode',
        'div0', 'div0null', 'equal_null', 'greatest', 'greatest_ignore_nulls',
        'iff', 'ifnull', 'is_null_value', 'least', 'least_ignore_nulls',
        'nullif', 'nullifzero', 'nvl', 'nvl2', 'regr_valx', 'regr_valy',
        'zeroifnull',
        # String:
        'ascii', 'base64_decode_string', 'base64_encode', 'bit_length',
        'char', 'charindex', 'chr', 'concat', 'concat_ws', 'contains',
        'editdistance', 'endswith', 'hash', 'hex_decode_string', 'hex_encode',
        'ilike', 'initcap', 'insert', 'jarowinkler_similarity', 'left', 'len',
        'length', 'like', 'lower', 'lpad', 'ltrim', 'md5', 'md5_hex',
        'md5_number_lower64', 'md5_number_upper64', 'octet_length',
        'parse_url', 'position', 'regexp_count', 'regexp_instr',
        'regexp_like', 'regexp_replace', 'regexp_substr', 'regexp_substr_all',
        'repeat', 'replace', 'reverse', 'right', 'rlike', 'rpad', 'rtrim',
        'sha1', 'sha1_hex', 'sha2', 'sha2_hex', 'soundex', 'soundex_p123',
        'space', 'split', 'split_part', 'startswith', 'strtok',
        'strtok_to_array', 'substr', 'substring', 'translate', 'trim',
        'try_base64_decode_string', 'try_hex_decode_string', 'unicode',
        'upper',
        # Math:
        'abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh',
        'bitand', 'bitnot', 'bitor', 'bitshiftleft', 'bitshiftright',
        'bitxor', 'cbrt', 'ceil', 'cos', 'cosh', 'cot', 'degrees', 'exp',
        'factorial', 'floor', 'haversine', 'ln', 'log', 'mod', 'pi', 'pow',
        'power', 'radians', 'round', 'sign', 'sin', 'sinh', 'sqrt', 'square',
        'tan', 'tanh', 'trunc', 'truncate', 'width_bucket',
        # Date and time:
        'add_months', 'convert_timezone', 'date_from_parts', 'date_part',
        'date_trunc', 'dateadd', 'datediff', 'datefromparts', 'day',
        'dayname', 'dayofmonth', 'dayofweek', 'dayofweekiso', 'dayofyear',
        'extract', 'hour', 'last_day', 'minute', 'month', 'monthname',
        'months_between', 'next_day', 'previous_day', 'quarter', 'second',
        'time_from_parts', 'time_slice', 'timeadd', 'timediff',
        'timefromparts', 'timestamp_from_parts', 'timestamp_ltz_from_parts',
        'timestamp_ntz_from_parts', 'timestamp_tz_from_parts',
        'timestampadd', 'timestampdiff', 'timestampfromparts', 'week',
        'weekiso', 'weekofyear', 'year', 'yearofweek', 'yearofweekiso',
        # Cast and conversion:
        'as_array', 'as_binary', 'as_boolean', 'as_char', 'as_date',
        'as_decimal', 'as_double', 'as_integer', 'as_number', 'as_object',
        'as_real', 'as_time', 'as_timestamp_ltz', 'as_timestamp_ntz',
        'as_timestamp_tz', 'as_varchar', 'cast', 'is_array', 'is_boolean',
        'is_date', 'is_decimal', 'is_double', 'is_integer', 'is_object',
        'is_real', 'is_time', 'is_timestamp_ltz', 'is_timestamp_ntz',
        'is_timestamp_tz', 'is_varchar', 'to_array', 'to_binary',
        'to_boolean', 'to_char', 'to_date', 'to_decimal', 'to_double',
        'to_number', 'to_numeric', 'to_object', 'to_time', 'to_timestamp',
        'to_timestamp_ltz', 'to_timestamp_ntz', 'to_timestamp_tz',
        'to_varchar', 'to_variant', 'try_cast', 'try_to_binary',
        'try_to_boolean', 'try_to_date', 'try_to_decimal', 'try_to_double',
        'try_to_number', 'try_to_numeric', 'try_to_time', 'try_to_timestamp',
        'try_to_timestamp_ltz', 'try_to_timestamp_ntz',
        'try_to_timestamp_tz', 'typeof',
        # Array and object, with their lambda functions:
        'array_append', 'array_cat', 'array_compact', 'array_construct',
        'array_construct_compact', 'array_contains', 'array_distinct',
        'array_except', 'array_flatten', 'array_generate_range',
        'array_insert', 'array_intersection', 'array_max', 'array_min',
        'array_position', 'array_prepend', 'array_remove', 'array_remove_at',
        'array_reverse', 'array_size', 'array_slice', 'array_sort',
        'array_to_string', 'arrays_overlap', 'arrays_to_object', 'arrays_zip',
        'filter', 'object_construct', 'object_construct_keep_null',
        'object_delete', 'object_insert', 'object_keys', 'object_pick',
        'reduce', 'transform',
        # JSON and variant paths:
        'check_json', 'get', 'get_ignore_case', 'get_path',
        'json_extract_path_text', 'parse_json', 'strip_null_value',
        'try_parse_json',
    }
)  # fmt: skip

_TRUSTED_FUNCTIONS = {
    'databricks': _DATABRICKS_FUNCTIONS,
    'snowflake': _SNOWFLAKE_FUNCTIONS,
}

# Keywords that an expression may follow with `(` without calling anything.
_KEYWORDS = frozenset(
    {
        'all', 'and', 'any', 'as', 'between', 'both', 'by', 'case', 'distinct',
        'else', 'end', 'escape', 'exists', 'for', 'from', 'ilike', 'in',
        'interval', 'is', 'leading', 'like', 'not', 'or', 'over', 'placing',
        'regexp', 'rlike', 'some', 'then', 'trailing', 'when',
    }
)  # fmt: skip

# Type names take `(` for their parameters after `::`, or after the `AS` of a
# cast. Anywhere else `AS` starts an alias and the name is a call.
_TYPE_NAMES = frozenset(
    {
        'array', 'binary', 'char', 'character', 'dec', 'decimal', 'double',
        'float', 'map', 'nchar', 'number', 'numeric', 'nvarchar', 'object',
        'string', 'struct', 'text', 'time', 'timestamp', 'timestamp_ltz',
        'timestamp_ntz', 'timestamp_tz', 'varbinary', 'varchar', 'vector',
    }
)  # fmt: skip

# Read without parentheses, these return the session's identity or clock.
_SESSION_WORDS = frozenset(
    {
        'current_catalog', 'current_database', 'current_date',
        'current_path', 'current_role', 'current_schema', 'current_time',
        'current_timestamp', 'current_user', 'localtime', 'localtimestamp',
        'session_user', 'user',
    }
)  # fmt: skip
_SESSION_VALUES = re.compile(
    r'(?<![\w$.\ue000])'
    rf'({"|".join(sorted(_SESSION_WORDS))})\b(?!\s*\()',
    flags=re.IGNORECASE,
)
# Quoting the word does not keep it from the session: on a DBSQL 2026.38
# warehouse `user` and `CURRENT_DATE`, in backticks and with no column of
# that name, answer the session user and today. A column of the name wins
# for every session word quoted, `current_path` included. How Snowflake
# reads a quoted session word is not documented, so none is excused there.
_COLUMN_OVER_QUOTED_SESSION = {
    'databricks': _SESSION_WORDS,
    'snowflake': frozenset(),
}

# The session words a column of the same name overrides. On a DBSQL 2026.38
# warehouse each reads the column, bare or nested, in a comparison, a CASE,
# an aggregate or a metric view, except `current_path`, which always returns
# the search path. Snowflake reserves `current_date`, `current_time`,
# `current_timestamp`, `current_user`, `localtime` and `localtimestamp`, so no
# bare column takes those names, and how it reads the others is not
# documented, so every session word stays refused there.
_COLUMN_OVER_SESSION = {
    'databricks': frozenset(
        {
            'current_catalog', 'current_database', 'current_date',
            'current_role', 'current_schema', 'current_time',
            'current_timestamp', 'current_user', 'localtime',
            'localtimestamp', 'session_user', 'user',
        }
    ),
    'snowflake': frozenset(),
}  # fmt: skip

# A private-use character stands in for quoted text, so a call to a quoted
# name stays visible as a call.
_QUOTED = '\ue000'

#: Operators written as words. They end no operand, so a keyword behind
#: one is being called: `1 DIV escape(x)` calls `escape`.
_WORD_OPERATORS = frozenset({'div', 'collate'})

# Every Databricks SQL keyword, from `SELECT keyword FROM sql_keywords()` on a
# DBSQL 2026.38 warehouse. None is reserved, so each may also name a routine.
_DATABRICKS_KEYWORDS = frozenset(
    {
        'access', 'action', 'add', 'after', 'aggregate', 'align',
        'all', 'alter', 'always', 'analyze', 'and', 'anonymous',
        'anti', 'any', 'any_value', 'apply', 'approx', 'archive',
        'archived', 'array', 'as', 'asc', 'asensitive', 'asof',
        'async', 'at', 'atomic', 'authorization', 'auto',
        'autoclusteringprofile', 'autoclusteringupdate', 'autocommit',
        'begin', 'bernoulli', 'between', 'bigint', 'bin',
        'bin_distribute_ratio', 'bin_end', 'bin_start', 'binary',
        'binding', 'bitemporal', 'bloomfilter', 'boolean', 'both',
        'bucket', 'buckets', 'by', 'byte', 'cache', 'call', 'called',
        'cascade', 'case', 'cast', 'catalog', 'catalogs', 'cdc',
        'change', 'changes', 'char', 'character', 'check',
        'checkpoint', 'cleanroom', 'cleanrooms', 'clear', 'clone',
        'close', 'cloud', 'cluster', 'clustered', 'code', 'codegen',
        'collaborators', 'collate', 'collation', 'collations',
        'collect', 'collection', 'column', 'columns', 'comment',
        'comments', 'commit', 'compact', 'compactions', 'compensation',
        'compute', 'concatenate', 'condition', 'condition_identifier',
        'conf', 'constraint', 'constraints', 'contains', 'continue',
        'convert', 'copy', 'copy_options', 'cost', 'create',
        'credential', 'credentials', 'cron', 'cross', 'cube',
        'current', 'current_database', 'current_date', 'current_path',
        'current_recipient', 'current_schema', 'current_time',
        'current_timestamp', 'current_user', 'cursor', 'data',
        'database', 'databases', 'date', 'date_add', 'date_diff',
        'dateadd', 'datediff', 'day', 'dayofyear', 'days', 'dbfs',
        'dbproperties', 'dec', 'decfloat', 'decfloat_rounding_mode',
        'decimal', 'declare', 'deep', 'default', 'default_path',
        'deferrable', 'deferred', 'define', 'defined', 'definer',
        'delay', 'delete', 'deletion', 'delimited', 'delta', 'deny',
        'depends', 'desc', 'describe', 'description', 'detail',
        'deterministic', 'dfs', 'diagnostics', 'directories',
        'directory', 'disable', 'disk', 'distance', 'distinct',
        'distribute', 'div', 'do', 'double', 'drop', 'dropped', 'dry',
        'effective', 'else', 'elseif', 'empty', 'enable', 'encryption',
        'end', 'enforced', 'environment', 'error', 'escape', 'escaped',
        'every', 'evolution', 'exact', 'except', 'exchange', 'exclude',
        'exclusive', 'execute', 'exists', 'exit', 'expect',
        'expiration', 'explain', 'export', 'extend', 'extended',
        'external', 'extract', 'fail', 'false', 'feature', 'feed',
        'fetch', 'fields', 'file', 'fileformat', 'files', 'filter',
        'first', 'flamegraph', 'float', 'flow', 'flows', 'fn',
        'following', 'for', 'force', 'foreign', 'format',
        'format_options', 'format_version', 'formatted', 'found',
        'frame', 'freshness', 'from', 'fsck', 'full', 'function',
        'functions', 'generate', 'generated', 'geography', 'geometry',
        'get', 'get_column_tag_value', 'get_tag_value', 'global',
        'governed', 'grant', 'grants', 'group', 'grouping', 'groups',
        'handler', 'having', 'history', 'hour', 'hours',
        'iceberg_compat_version', 'id', 'identified', 'identifier',
        'identity', 'if', 'ignore', 'ilike', 'immediate', 'import',
        'in', 'include', 'inclusive', 'increment', 'incremental',
        'index', 'indexes', 'inherit', 'initially', 'inner', 'inout',
        'inpath', 'input', 'inputformat', 'insensitive', 'insert',
        'int', 'integer', 'intersect', 'interval', 'into', 'inventory',
        'invoker', 'is', 'isolation', 'items', 'iterate', 'jfr',
        'join', 'json', 'json_table', 'key', 'keys', 'language',
        'last', 'lateral', 'lazy', 'leading', 'leave', 'left', 'level',
        'like', 'limit', 'line_number', 'lines', 'list', 'lite',
        'live', 'load', 'local', 'localtime', 'location', 'locations',
        'lock', 'locks', 'logical', 'long', 'loop', 'macro', 'managed',
        'manual', 'many', 'map', 'mask', 'masks', 'match',
        'match_condition', 'match_recognize', 'matched', 'matches',
        'materialize', 'materialized', 'max', 'mcp', 'measure',
        'measures', 'memory', 'merge', 'message_arguments',
        'message_text', 'metadata', 'metadata_path', 'metastore',
        'metastores', 'metric', 'metrics', 'microsecond',
        'microseconds', 'millisecond', 'milliseconds', 'minus',
        'minute', 'minutes', 'model', 'models', 'modified', 'modifies',
        'month', 'months', 'most', 'move', 'msck', 'name', 'namespace',
        'namespaces', 'nanosecond', 'nanoseconds', 'natural',
        'nearest', 'next', 'no', 'nocopy', 'none', 'norely', 'not',
        'novalidate', 'null', 'nulls', 'numeric', 'of', 'offset',
        'omit', 'on', 'once', 'one', 'only', 'open', 'optimization',
        'optimize', 'option', 'options', 'or', 'order', 'ordinality',
        'out', 'outer', 'outputformat', 'over', 'overlaps', 'overlay',
        'overwrite', 'owner', 'parameter', 'parquet', 'partial',
        'partition', 'partitioned', 'partitions', 'past', 'path',
        'pattern', 'per', 'percent', 'permute', 'pivot', 'place',
        'placing', 'point', 'policies', 'policy', 'position',
        'preceding', 'predictive', 'preserve', 'primary', 'principals',
        'private', 'privileges', 'procedure', 'procedures', 'profile',
        'profiler', 'properties', 'provider', 'providers', 'purge',
        'put', 'qualify', 'quarantine', 'quarter', 'query',
        'query_tags', 'range', 'read', 'reads', 'real', 'recipient',
        'recipients', 'recordreader', 'recordwriter', 'recover',
        'recursion', 'recursive', 'reduce', 'references', 'refresh',
        'region', 'relationship', 'rely', 'remove', 'rename', 'reorg',
        'repair', 'repeat', 'repeatable', 'replace', 'replicas',
        'reset', 'resignal', 'resolve', 'respect', 'restore',
        'restrict', 'retain', 'return', 'returned_sqlstate', 'returns',
        'revoke', 'right', 'role', 'roles', 'rollback', 'rollup',
        'row', 'row_count', 'rows', 'run', 'sample', 'scd', 'schedule',
        'schema', 'schemas', 'search', 'second', 'seconds', 'secret',
        'secrets', 'security', 'select', 'semi', 'separated',
        'sequence', 'serde', 'serdeproperties', 'service',
        'session_user', 'set', 'sets', 'shallow', 'share', 'shares',
        'short', 'show', 'shred', 'signal', 'similarity', 'simple',
        'single', 'skewed', 'skip', 'smallint', 'snapshot', 'some',
        'sort', 'sorted', 'source', 'specific', 'sql', 'sqlexception',
        'sqlstate', 'start', 'static', 'statistics', 'storage',
        'stored', 'stratify', 'stream', 'streaming', 'strict',
        'string', 'struct', 'style', 'subset', 'substr', 'substring',
        'sync', 'system', 'system_path', 'system_time',
        'system_version', 'table', 'tableref', 'tables', 'tablesample',
        'tag', 'tags', 'target', 'tblproperties', 'tblproperty',
        'terminated', 'then', 'time', 'timediff', 'timeseries',
        'timestamp', 'timestamp_ltz', 'timestamp_ntz', 'timestampadd',
        'timestampdiff', 'tinyint', 'to', 'touch', 'track', 'trailing',
        'transaction', 'transaction_active', 'transactions',
        'transform', 'trigger', 'trim', 'true', 'truncate', 'try_cast',
        'type', 'unarchive', 'unbounded', 'uncache', 'undrop',
        'uniform', 'union', 'unique', 'unknown', 'unlock', 'unmatched',
        'unmaterialize', 'unnest', 'unpivot', 'unset', 'unshred',
        'until', 'update', 'updates', 'upgrade', 'url', 'use', 'user',
        'users', 'using', 'vacuum', 'validate', 'value', 'values',
        'var', 'varchar', 'variable', 'variant', 'vector', 'verify',
        'version', 'view', 'views', 'violation', 'void', 'volume',
        'volumes', 'watermark', 'week', 'weeks', 'when', 'where',
        'while', 'width', 'window', 'with', 'within', 'without',
        'work', 'x', 'year', 'years', 'zone', 'zorder',
    }
)  # fmt: skip

# Snowflake's reserved keywords, after its "Reserved & limited keywords" page.
_SNOWFLAKE_RESERVED = frozenset(
    {
        'account', 'all', 'alter', 'and', 'any', 'as', 'between', 'by',
        'case', 'cast', 'check', 'column', 'connect', 'connection',
        'constraint', 'create', 'cross', 'current', 'current_date',
        'current_time', 'current_timestamp', 'current_user',
        'database', 'delete', 'distinct', 'drop', 'else', 'exists',
        'false', 'following', 'for', 'from', 'full', 'grant', 'group',
        'gscluster', 'having', 'ilike', 'in', 'increment', 'inner',
        'insert', 'intersect', 'into', 'is', 'issue', 'join',
        'lateral', 'left', 'like', 'localtime', 'localtimestamp',
        'minus', 'natural', 'not', 'null', 'of', 'on', 'or', 'order',
        'organization', 'qualify', 'regexp', 'revoke', 'right',
        'rlike', 'row', 'rows', 'sample', 'schema', 'select', 'set',
        'some', 'start', 'table', 'tablesample', 'then', 'to',
        'trigger', 'true', 'try_cast', 'union', 'unique', 'update',
        'using', 'values', 'view', 'when', 'whenever', 'where',
        'window', 'with',
    }
)  # fmt: skip

#: Keywords that never precede an expression, checked by hand against the
#: Spark grammar and the Databricks and Snowflake references. Read as grammar,
#: each is followed by a literal, a name, `(`, `,` or another keyword, so a
#: column of that name before `IN (` is a value. A keyword missing here stays
#: grammar, which refuses rather than lifts.
_DATETIME_UNITS = frozenset(
    {
        'day', 'dayofyear', 'days', 'hour', 'hours', 'microsecond',
        'microseconds', 'millisecond', 'milliseconds', 'minute', 'minutes',
        'month', 'months', 'nanosecond', 'nanoseconds', 'quarter', 'second',
        'seconds', 'week', 'weeks', 'year', 'years',
    }
)  # fmt: skip
_NEVER_BEFORE_AN_EXPRESSION = _DATETIME_UNITS | frozenset(
    {
        # Each datetime unit is followed by a literal, `,`, `FROM` or `TO`.
        # Typed literals and type names, followed by a literal, `AS` or `(`.
        'date', 'number', 'text', 'time', 'timestamp', 'x',
        # Words that only commands, clauses or `name(` forms use.
        'access', 'action', 'code', 'comment', 'cost', 'data', 'description',
        'file', 'format', 'history', 'id', 'index', 'key', 'language',
        'level', 'location', 'measure', 'metric', 'model', 'name', 'owner',
        'path', 'percent', 'position', 'region', 'role', 'schedule',
        'sequence', 'source', 'start', 'tag', 'tags', 'target', 'type', 'url',
        'value', 'version',
    }
)  # fmt: skip

#: Words the parser may read as grammar, so a column named like one is still
#: not a value: `WHERE from(y)` calls `from` whatever the table holds. `zone`
#: stays too, out of caution: neither warehouse parses standard SQL's `AT TIME
#: ZONE <expr>` today, but Databricks keeps `zone` as a keyword.
_GRAMMAR_WORDS = (
    _DATABRICKS_KEYWORDS
    | _SNOWFLAKE_RESERVED
    | _KEYWORDS
    | _WORD_OPERATORS
    | _TYPE_NAMES
    | {'cast', 'date_part', 'extract', 'identifier', 'try_cast'}
) - _NEVER_BEFORE_AN_EXPRESSION

# What may stand right before a `(` that opens a group rather than a call.
_OPENS_A_GROUP = frozenset('+-*/%=<>!|&^~,([?:')
_LAST_TOKEN = re.compile(r'([\w$\ue000]+)\s*$')
_QUALIFIED = re.compile(r'\.\s*$')
_AFTER_DOUBLE_COLON = re.compile(r'::\s*$')
_AFTER_AS = re.compile(r'(?<![\w$])as\s*$', flags=re.IGNORECASE)
_CASTS = frozenset({'cast', 'try_cast'})
# The field of `EXTRACT(YEAR FROM d)` is a value, whatever its name.
_DATETIME_FIELD = re.compile(
    r'(?<![\w$.])(?:extract|date_part)\s*\(\s*$', flags=re.IGNORECASE
)
_LAMBDA_PARAMETERS = re.compile(
    r'(?:\(\s*([\w$]+(?:\s*,\s*[\w$]+)*)\s*\)|([\w$]+))\s*->'
)
_WORD_OR_BRACKET = re.compile(r'[\w$\ue000]+|[()\[\]]')
_WHEN = re.compile(r'\s*when\b', flags=re.IGNORECASE)


#: The keyword pairs SQL actually writes, as in ``x NOT LIKE ('a')`` and
#: ``CASE WHEN (...)``. Any other keyword needs a value in front of it
#: before a word that could name a function counts as one.
_FOLLOWS_A_KEYWORD = {
    'not': frozenset(
        {
            'between',
            'distinct',
            'exists',
            'ilike',
            'in',
            'like',
            'regexp',
            'rlike',
        }
    ),
    # Only LIKE and ILIKE take a quantifier: `x RLIKE any(y)` calls `any`.
    'like': frozenset({'all', 'any', 'some'}),
    'ilike': frozenset({'all', 'any', 'some'}),
    'is': frozenset({'distinct', 'not'}),
    'distinct': frozenset({'from'}),
    # WHEN, THEN and ELSE each take an operand, so a keyword right after one
    # is a call: Databricks runs `CASE WHEN then(x) THEN 1 END` as one.
    'case': frozenset({'when'}),
}


def _closes_a_case(lead: str) -> bool:
    r"""Whether a ``CASE`` is open at the end of ``lead``, in the same group.

    Args:
        lead: The expression text before an ``END``, quoted text replaced.
    """
    open_cases = [0]
    for token in _WORD_OR_BRACKET.finditer(lead):
        text = token.group()
        if text in '([':
            open_cases.append(0)
        elif text in ')]':
            if len(open_cases) > 1:
                open_cases.pop()
        elif _QUALIFIED.search(lead[: token.start()]):
            continue
        elif text.lower() == 'case':
            open_cases[-1] += 1
        elif text.lower() == 'end' and open_cases[-1] > 0:
            open_cases[-1] -= 1
    return open_cases[-1] > 0


def _follows_a_value(lead: str, columns: frozenset[str]) -> bool:
    r"""Whether ``lead`` ends with something a comparison can be about.

    A value is recognised by what it is rather than by what it is not, so a
    word nobody thought of fails closed. It is a closing bracket, quoted
    text, a literal, a qualified name, a type after ``::``, the field of
    ``EXTRACT``, the ``END`` of an open ``CASE``, or a bare word naming one
    of ``columns`` that the parser cannot read as grammar instead. A keyword
    is none of these: after ``NOT`` or ``AND`` the next thing is an
    expression.

    Args:
        lead: The expression text before a keyword, quoted text replaced.
        columns: The folded names an expression may use bare.
    """
    lead = lead.rstrip()
    if not lead:
        return False
    if lead[-1] in ')]' + _QUOTED:
        return True
    last = _LAST_TOKEN.search(lead)
    if last is None:
        return False
    before = lead[: last.start()]
    if (
        _QUALIFIED.search(before)
        or _AFTER_DOUBLE_COLON.search(before)
        or _DATETIME_FIELD.search(before)
    ):
        return True
    word = last.group(1).lower()
    if word[0].isdigit() or word in ('true', 'false', 'null'):
        return True
    if word == 'end':
        return _closes_a_case(before) and _follows_a_value(before, columns)
    # A table may name a column `where`, which the parser still reads as WHERE.
    return word in columns and word not in _GRAMMAR_WORDS


def _opens_a_simple_case(lead: str, group: str) -> bool:
    r"""Whether ``CASE`` before ``group`` starts ``CASE (x) WHEN ...``.

    ``CASE`` names no function in either dialect, yet the same text calls one
    wherever a ``CASE`` cannot start or no ``WHEN`` follows, so both are
    required.

    Args:
        lead: The expression text before ``CASE``, quoted text replaced.
        group: The expression text from the ``(`` after ``CASE`` on.
    """
    lead = lead.rstrip()
    if lead and lead[-1] not in '(,':
        return False
    depth = 0
    for index, char in enumerate(group):
        depth += {'(': 1, ')': -1}.get(char, 0)
        if depth == 0:
            return _WHEN.match(group, index + 1) is not None
    return False


def _acts_as_keyword(
    word: str, lead: str, group: str, columns: frozenset[str]
) -> bool:
    r"""Whether ``word``, followed by ``(``, is a keyword rather than a call.

    Databricks lets most of these words name a function too, so ``escape(x)``
    calls one. As a keyword each follows a *value*: ``x IN (...)``,
    ``TRIM(BOTH ' ' FROM x)``.

    What stands in front matters, not merely that something does. After a
    value, ``escape(`` is the ``ESCAPE`` of a ``LIKE``. After ``NOT``, which
    takes an expression next, the same text calls a function, and reading it
    as a keyword let ``NOT escape(secret())`` through.

    A few pairs are real: ``x NOT LIKE (...)`` negates a comparison. Those
    need the value the comparison is about, so ``NOT regexp(x)`` with nothing
    in front of it is a call.

    ``NOT`` and ``CASE`` are their own cases. Neither names a function in
    either dialect, and ``CASE (x) WHEN ...`` compares a value, so ``CASE``
    is keyword syntax only where a ``CASE`` may start.

    Args:
        word: The folded word before ``(``.
        lead: The expression text before ``word``.
        group: The expression text from that ``(`` on.
        columns: The folded names an expression may use bare.
    """
    if word == 'not':
        return True
    if word == 'case':
        return _opens_a_simple_case(lead, group)
    return _in_keyword_position(word, lead, columns)


def _in_keyword_position(word: str, lead: str, columns: frozenset[str]) -> bool:
    r"""Whether ``word`` sits where a keyword may, given what precedes it.

    After a value, yes. After another keyword, only where SQL writes that
    pair, and only if that keyword is itself in keyword position: ``IS
    DISTINCT FROM`` reaches a value through ``IS``, while a bare ``DISTINCT
    FROM`` does not.
    """
    if _follows_a_value(lead, columns):
        return True
    lead = lead.rstrip()
    previous = _LAST_TOKEN.search(lead)
    if previous is None:
        return False
    before = previous.group(1).lower()
    if word not in _FOLLOWS_A_KEYWORD.get(before, frozenset()):
        return False
    if before == 'case':
        return True
    return _in_keyword_position(before, lead[: previous.start()], columns)


def _names_a_type(lead: str) -> bool:
    r"""Whether the word after ``lead`` stands where only a type can.

    That is after ``::``, or after an ``AS`` directly inside ``CAST(`` or
    ``TRY_CAST(``. Elsewhere ``AS`` gives an alias, so ``x AS object(1)``
    calls ``object``.

    Args:
        lead: The expression text before the word, quoted text replaced.
    """
    if _AFTER_DOUBLE_COLON.search(lead):
        return True
    after_as = _AFTER_AS.search(lead)
    if after_as is None:
        return False
    depth = 0
    for index in range(after_as.start() - 1, -1, -1):
        depth += {')': 1, '(': -1}.get(lead[index], 0)
        if depth < 0:
            opener = lead[:index].rstrip()
            name = _LAST_TOKEN.search(opener)
            return (
                name is not None
                and name.group(1).lower() in _CASTS
                and not _QUALIFIED.search(opener[: name.start()])
            )
    return False


def _lambda_parameters(text: str) -> set[str]:
    r"""The folded names ``text`` binds in lambdas, as in ``x -> x + 1``."""
    names: set[str] = set()
    for group, single in _LAMBDA_PARAMETERS.findall(text):
        names.update(
            name.strip().lower() for name in (group or single).split(',')
        )
    return names


def _untrusted_call_reason(
    expr: str, dialect: str, *, columns: Collection[str]
) -> str | None:
    r"""Return why an expression calls a function that is not trusted, or
    ``None``.

    Only the dialect's pure, deterministic, built-in scalar functions are
    trusted, called by their plain name. A user-defined function, or a
    built-in like ``secret()``, ``http_request()`` or ``SYSTEM$WAIT()``, would
    otherwise run with the caller's credentials every time a graph samples.
    Qualified and quoted names, and ``IDENTIFIER()``, are refused since they
    can name anything. Session values (``current_user``), bare or quoted,
    are refused too: a column must not change with who reads it. A session
    word that one of ``columns`` also names reads that column where the
    dialect lets the column win, and is accepted there.

    Every ``(`` is checked, and fails closed: it must open a group, follow
    a keyword, give a type its parameters, or call a trusted function. So a
    name this does not recognise as a name is refused rather than skipped.

    A keyword before ``(`` is grammar only after a value, as in ``x IN (1)``
    or ``x NOT LIKE ('a')``, and a call anywhere else: ``NOT escape(x)``. A
    bare word counts as a value only if it names one of ``columns`` or a
    lambda parameter, and the parser cannot read it as grammar instead.

    Apply :func:`_unsafe_expr_reason` first, which establishes that every
    quote in ``expr`` is paired.

    Args:
        expr: The expression as written in the view definition.
        dialect: ``'databricks'`` or ``'snowflake'``.
        columns: The columns of the table the expression is evaluated on.
            Pass ``()`` where none are known, and no bare word is a value.
    """
    trusted = _TRUSTED_FUNCTIONS[dialect]
    text = _masked(expr, dialect, _QUOTED)
    named = frozenset(column.lower() for column in columns)
    visible = named | _lambda_parameters(text)
    for index, char in enumerate(text):
        if char != '(':
            continue
        before = text[:index].rstrip()
        if not before or before[-1] in _OPENS_A_GROUP:
            continue
        if before[-1] in ')]':
            return 'it calls the result of an expression'
        token = _LAST_TOKEN.search(before)
        if token is None:
            return f"it calls what precedes '(' after {before[-10:]!r}"
        name = token.group(1)
        if _QUOTED in name:
            return 'it calls a function by a quoted name'
        if _QUALIFIED.search(before[: token.start()]):
            return f"it calls the qualified function '{name}'"
        folded = name.lower()
        if folded == 'identifier':
            return 'it names a function through IDENTIFIER()'
        if folded in _KEYWORDS and _acts_as_keyword(
            folded, before[: token.start()], text[index:], visible
        ):
            continue
        if folded in _TYPE_NAMES and _names_a_type(before[: token.start()]):
            continue
        if folded not in trusted:
            return (
                f"it calls '{name}', which is not a pure built-in SQL function"
            )
    # A lambda parameter is in scope only inside its lambda, so only the
    # table's columns can stand in for a session word.
    overridden = named & _COLUMN_OVER_SESSION[dialect]
    for session in _SESSION_VALUES.finditer(text):
        if session.group(1).lower() not in overridden:
            return f"it reads the session value '{session.group(1)}'"
    overridden = named & _COLUMN_OVER_QUOTED_SESSION[dialect]
    for quoted in _lex(expr, dialect):
        if quoted.kind != 'quoted' or _QUALIFIED.search(expr[: quoted.start]):
            continue
        word = _unquoted(quoted).lower()
        if word in _SESSION_WORDS and word not in overridden:
            return f'it reads the session value {quoted.text}'
    return None


# Words of a Databricks expression that may be grammar. On a DBSQL 2026.38
# warehouse a column may take each of these names but `true`, `false` and
# `null`, and is read where an operand stands: `SUM(filter)`, `case + 1`. A
# datetime unit is grammar only where a unit stands, so is not among them.
_EXPRESSION_KEYWORDS = frozenset(
    {
        'all', 'and', 'any', 'as', 'asc', 'between', 'by', 'case', 'current',
        'desc', 'distinct', 'else', 'end', 'escape', 'exists', 'false',
        'filter', 'first', 'following', 'from', 'ilike', 'in', 'interval',
        'is', 'last', 'like', 'not', 'null', 'nulls', 'or', 'order', 'over',
        'partition', 'preceding', 'range', 'rlike', 'row', 'rows', 'some',
        'then', 'true', 'unbounded', 'when', 'where',
    }
)  # fmt: skip
_LITERAL_WORDS = frozenset({'true', 'false', 'null'})
# Keywords an operand follows, as in `NOT x` and `COUNT(DISTINCT x)`.
_TAKE_AN_OPERAND = frozenset(
    {
        'all', 'and', 'between', 'by', 'case', 'distinct', 'else', 'from',
        'ilike', 'interval', 'like', 'not', 'or', 'rlike', 'then', 'when',
        'where',
    }
)  # fmt: skip
# Keywords that may follow an operand, as in `x IS NULL` and `ORDER BY x ASC`.
_FOLLOW_AN_OPERAND = frozenset(
    {
        'and', 'as', 'asc', 'between', 'collate', 'desc', 'div', 'else', 'end',
        'escape', 'for', 'from', 'ilike', 'in', 'is', 'like', 'nulls', 'or',
        'range', 'regexp', 'rlike', 'rows', 'then', 'when',
    }
)  # fmt: skip
# `NOT` follows an operand only before these, as in `x NOT IN (1)`.
_AFTER_A_NEGATED_OPERAND = frozenset(
    {'between', 'ilike', 'in', 'like', 'regexp', 'rlike'}
)
# Databricks reads these as grammar before a sign: `SELECT all + 1` is 1.
_BEFORE_A_SIGN = frozenset({'all', 'distinct', 'not'})
_OPERATORS = frozenset({'<=>', '<=', '>=', '<>', '<', '>', '->', '=>'})
# What a window's frame says, grammar anywhere inside `OVER (...)`.
_FRAME_WORDS = frozenset(
    {'current', 'following', 'preceding', 'range', 'row', 'rows', 'unbounded'}
)
# Functions whose first argument may be a datetime unit, as a bare word.
_UNIT_FIRST = frozenset({'extract', 'timestampadd', 'timestampdiff'})
# These take a unit first only in their three-argument form: `datediff(a,
# b)` reads two values.
_UNIT_FIRST_OF_THREE = frozenset({'dateadd', 'datediff'})


@dataclass(frozen=True)
class NameReference:
    r"""A name an expression reads as a value, possibly qualified.

    Args:
        parts: Its parts, unquoted.
        span: Where it is in the expression as written.
        bare: Whether it is one unquoted part.
    """

    parts: tuple[str, ...]
    span: tuple[int, int]
    bare: bool


def _closing(tokens: list[_Token], at: int) -> int:
    r"""The index of the bracket closing the one at ``at``, or the last."""
    pairs = {'(': ')', '[': ']', '<': '>'}
    opening, depth = tokens[at].text, 0
    for index in range(at, len(tokens)):
        if tokens[index].text == opening:
            depth += 1
        elif tokens[index].text == pairs[opening]:
            depth -= 1
            if depth == 0:
                return index
    return len(tokens) - 1


def _arguments(tokens: list[_Token], at: int) -> int:
    r"""How many arguments the call whose ``(`` is at ``at`` is given."""
    commas, depth = 0, 0
    for token in tokens[at : _closing(tokens, at)]:
        depth += {'(': 1, ')': -1, '[': 1, ']': -1}.get(token.text, 0)
        commas += token.text == ',' and depth == 1
    return commas + 1


def _lambda_scopes(tokens: list[_Token]) -> list[tuple[int, int, set[str]]]:
    r"""Each lambda's parameters, folded, with the tokens they are bound in:
    from the parameters to the end of the group holding the lambda.
    """
    scopes: list[tuple[int, int, set[str]]] = []
    for index, token in enumerate(tokens):
        if token.text != '->' or index == 0:
            continue
        before = tokens[index - 1]
        if before.text == ')':
            depth, start = 0, index - 1
            while start > 0:
                depth += {')': 1, '(': -1}.get(tokens[start].text, 0)
                if depth == 0:
                    break
                start -= 1
            names = {t.folded for t in tokens[start:index] if t.folded}
        else:
            start, names = index - 1, {before.folded}
        depth, end = 0, len(tokens) - 1
        for at in range(start, len(tokens)):
            depth += {'(': 1, ')': -1}.get(tokens[at].text, 0)
            if depth < 0:
                end = at
                break
        scopes.append((start, end, names))
    return scopes


def _grammar(tokens: list[_Token]) -> set[int]:
    r"""Where in ``tokens`` a keyword is grammar rather than a column.

    A keyword names a column only where a lone operand stands: after an
    operator, an opening bracket, a comma or a keyword taking an operand,
    and before an operator, a closing bracket, a comma or a keyword that
    may follow an operand.
    """
    grammar: set[int] = set()
    openers: list[str] = []
    for index, token in enumerate(tokens):
        previous = tokens[index - 1] if index > 0 else None
        after = tokens[index + 1] if index + 1 < len(tokens) else None
        before, beyond = (
            tokens[at].folded if 0 <= at < len(tokens) else ''
            for at in (index - 2, index + 2)
        )
        if token.text in '([':
            openers.append(previous.folded if previous else '')
        elif token.text in ')]' and openers:
            openers.pop()
        word = token.folded
        if word not in _EXPRESSION_KEYWORDS:
            continue
        opens = (
            previous is None
            or previous.kind == 'other'
            or previous.text in ('(', '[', ',', *_OPERATORS)
            or (
                index - 1 in grammar
                and previous.folded in _TAKE_AN_OPERAND
                # `IS NOT` takes `NULL`, `TRUE` or `DISTINCT FROM`.
                and not (previous.folded == 'not' and before == 'is')
            )
        )
        closes = (
            after is None
            or (after.kind == 'other' and word not in _BEFORE_A_SIGN)
            or after.text in (')', ']', ',', '[', '::', *_OPERATORS)
            or after.folded in _FOLLOW_AN_OPERAND
            or (after.folded == 'not' and beyond in _AFTER_A_NEGATED_OPERAND)
        )
        # `CASE WHEN` opens a searched case, unless a `CASE` comes first.
        if word == 'case' and after is not None and after.folded == 'when':
            closes = previous is not None and previous.folded == 'case'
        in_frame = bool(openers) and openers[-1] == 'over'
        if (
            word in _LITERAL_WORDS
            or not (opens and closes)
            or (in_frame and word in _FRAME_WORDS)
        ):
            grammar.add(index)
    return grammar


def name_references(expr: str) -> list[NameReference]:
    r"""The names a Databricks expression reads as values.

    That leaves out a function's name, a type, a keyword where it is
    grammar, a datetime unit, a named argument, a lambda's parameter and a
    JSON path step, so a word of the grammar is never read as a column or a
    parameter.
    """
    tokens = [token for token in _lex(expr) if token.kind != 'comment']
    scopes = _lambda_scopes(tokens)
    grammar = _grammar(tokens)
    # Who opened each group still open, by name where a call did.
    openers: list[str] = []
    # Where a datetime unit may stand: after an INTERVAL's operand, a
    # literal, or TO after a unit.
    unit_after: set[int] = set()
    skip_to = -1
    references: list[NameReference] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        previous = tokens[index - 1] if index > 0 else None
        if token.text in '([':
            called = previous is not None and previous.kind in (
                'word',
                'quoted',
            )
            openers.append(previous.folded if called and previous else '')
        elif token.text in ')]' and openers:
            openers.pop()
        if token.kind in ('number', 'string'):
            unit_after.add(index)
        if token.folded == 'interval' and index + 1 < len(tokens):
            operand = index + 1
            if tokens[operand].text == '(':
                operand = _closing(tokens, operand)
            unit_after.add(operand)
        if token.kind not in ('word', 'quoted') or (
            previous is not None and previous.text == '.'
        ):
            index += 1
            continue
        last = index
        while (
            last + 2 < len(tokens)
            and tokens[last + 1].text == '.'
            and tokens[last + 2].kind in ('word', 'quoted')
        ):
            last += 2
        chain = tokens[index : last + 1 : 2]
        after = tokens[last + 1] if last + 1 < len(tokens) else None
        bare = len(chain) == 1 and token.kind == 'word'
        # After AS stands a type in a cast, and a name given elsewhere, as
        # in `struct(a AS f)`.
        after_as = previous is not None and previous.folded == 'as'
        is_type = (previous is not None and previous.text == '::') or (
            after_as and bool(openers) and openers[-1] in ('cast', 'try_cast')
        )
        is_unit = (
            bare
            and token.folded in _DATETIME_UNITS
            and (
                index - 1 in unit_after
                or (
                    previous is not None
                    and previous.text == '('
                    and len(openers) > 0
                    and (
                        openers[-1] in _UNIT_FIRST
                        or (
                            openers[-1] in _UNIT_FIRST_OF_THREE
                            and _arguments(tokens, index - 1) == 3
                        )
                    )
                )
            )
        )
        if is_unit:
            unit_after.add(index)
            if after is not None and after.folded == 'to':
                unit_after.add(last + 1)
        if is_type and after is not None and after.text == '<':
            skip_to = _closing(tokens, last + 1)
        if not (
            is_type
            or after_as
            or is_unit
            or index <= skip_to
            or (after is not None and after.text in ('(', '=>'))
            or (previous is not None and previous.text == ':')
            or (bare and index in grammar)
            or (bare and token.folded == 'to' and index - 1 in unit_after)
            or (
                bare
                and any(
                    start <= index <= end and token.folded in names
                    for start, end, names in scopes
                )
            )
        ):
            references.append(
                NameReference(
                    tuple(_unquoted(t) for t in chain),
                    (token.start, tokens[last].end),
                    bare,
                )
            )
        index = last + 1
    return references


def _unquoted(token: _Token) -> str:
    if token.kind == 'quoted':
        quote = token.text[0]
        return token.text[1:-1].replace(quote * 2, quote)
    return token.text
