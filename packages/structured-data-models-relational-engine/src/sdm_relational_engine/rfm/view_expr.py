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
from dataclasses import dataclass, field, replace

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
# `?::` is Databricks' try-cast, `x ?:: INT`, read wherever `::` is.
_LEXEMES_EVERYWHERE = (
    ('punct', r'<=>|<=|>=|<>|\?::|::|->|=>|[.:(),\[\]<>]'),
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
# Databricks runs each of these as a sub-query where it opens the expression
# or a `(`: `(FROM t |> ...)`, `(VALUES 1)`, `(WITH a AS (...) ...)` and the
# script transforms `(MAP x USING 'cmd')` and `(REDUCE ...)`, which need a
# USING before their bracket closes. A filter is spliced into `WHERE (...)`,
# so `values + 1 > 0` opens `(VALUES +1 > 0)`. `TABLE` is refused anywhere.
_QUERY_START = re.compile(
    r'(?:^|\()\s*(?:'
    r'(?:from|values)\b'
    r'|with\s+(?:recursive\s+)?(?:[\w$]+\s*)?(?:\([^()]*\)\s*)?as\s*\('
    r'|(?P<script>(?:map|reduce)\b(?!\s*\())'
    r')',
    flags=re.IGNORECASE,
)
_USING = re.compile(r'\busing\b', flags=re.IGNORECASE)


def _starts_a_query(unquoted: str) -> bool:
    r"""Whether ``unquoted``, its quoted text masked, opens a sub-query."""
    for match in _QUERY_START.finditer(unquoted):
        if not match.group('script'):
            return True
        depth, end = 0, len(unquoted)
        for at in range(match.end(), len(unquoted)):
            depth += {'(': 1, ')': -1}.get(unquoted[at], 0)
            if depth < 0:
                end = at
                break
        if _USING.search(unquoted, match.end(), end):
            return True
    return False


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
    always refused. Databricks also runs a query without ``SELECT``: through
    ``TABLE t``, a query opening with ``FROM``, ``VALUES``, ``WITH``, ``MAP``
    or ``REDUCE``, and the ``|>`` pipe, and Snowflake chains statements with
    ``->>``, so those are refused too.

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
    # Snowflake also comments out the rest of a line with `//`, which a
    # literal such as a URL may hold.
    if dialect == 'snowflake' and any(
        token.kind == 'comment' for token in _lex(expr, dialect)
    ):
        return "it contains '//'"
    if dialect == 'snowflake':
        for quoted in _lex(expr, dialect):
            if quoted.text.startswith('"') and '\\' in quoted.text:
                return 'it has a backslash inside a double-quoted name'
    unquoted = _masked(expr, dialect, ' ')
    # Which characters may join a name, and so keep a quote after them from
    # opening a literal, is documented for ASCII alone: Snowflake might read
    # `é$$` as a name or as `é` and a literal. Refusing anything else outside
    # quotes holds whatever the warehouse decides.
    if other := re.search(r'[^\x00-\x7f]', unquoted):
        return f'it has {other.group()!r} outside quotes, which is not ASCII'
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
    if _starts_a_query(unquoted):
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


# A column may take any keyword's name but TRUE, FALSE and NULL, so where a
# word stands decides whether it names one. Where both readings parse,
# Databricks takes the form, as in `NOT and(x)`.
_LITERAL_WORDS = frozenset({'true', 'false', 'null'})
# Words that, after an operand, take another: `x REGEXP y`, `s FOR 2`, and
# a type, field, collation or window name after AS, COLLATE or OVER.
_TAKE_AN_OPERAND = frozenset(
    {
        'and', 'as', 'between', 'collate', 'div', 'else', 'escape', 'filter',
        'for', 'from', 'ilike', 'in', 'is', 'like', 'or', 'over', 'placing',
        'range', 'regexp', 'rlike', 'rows', 'then', 'when', 'within',
    }
)  # fmt: skip
_TAKE_A_NAME = frozenset({'as', 'collate', 'over'})
# Words before `(` that open no call, or a form with its own grammar.
_NO_CALL_OPENERS = _TAKE_AN_OPERAND | {
    '', 'cast', 'extract', 'group', 'overlay', 'position', 'substring',
    'struct', 'trim', 'try_cast',
}  # fmt: skip
# What a CASE may have read last before each of its words.
_CASE_STEPS = {
    'when': ('case', 'then'),
    'then': ('when',),
    'else': ('then',),
    'end': ('then', 'else'),
}
# How many steps trial readings may take per token, and how deep they may
# nest, which only keeps the call stack in bounds: each level is a bracket.
_TRIAL_STEPS = 16
_DEEPEST_TRIAL = 64
# Words that follow an operand only before these: `x NOT IN (1)`,
# `IGNORE NULLS`.
_FOLLOWED_BY = {
    'not': frozenset({'between', 'ilike', 'in', 'like', 'regexp', 'rlike'}),
    'ignore': frozenset({'nulls'}),
    'respect': frozenset({'nulls'}),
}
_NAME_KINDS = ('word', 'quoted')
_OPERATOR_PAIRS = frozenset({'!=', '==', '||', '<<', '>>'})
# What may stand after IS: `IS [NOT] NULL`, `IS UNKNOWN`, `IS DISTINCT FROM`.
_AFTER_IS = frozenset(
    {'distinct', 'false', 'from', 'not', 'null', 'true', 'unknown'}
)
# What a window's frame says between ROWS or RANGE and its `)`, and what
# may start it.
_FRAME_WORDS = frozenset(
    {'between', 'current', 'following', 'preceding', 'row', 'unbounded'}
)
_FRAME_STARTS = frozenset({'between', 'current', 'interval', 'unbounded'})
# Words before a string that type it: `DATE '2024-01-01'`, and `X'1F'`, a
# binary literal in hex.
_TYPED_LITERALS = frozenset(
    {'date', 'timestamp', 'timestamp_ltz', 'timestamp_ntz', 'x'}
)
_TRIM_SIDES = frozenset({'both', 'leading', 'trailing'})
# The words before BY that only a window's clauses open with, as in
# `OVER (DISTRIBUTE BY y SORT BY n)`. ORDER BY also sorts a WITHIN GROUP.
_WINDOW_BY = frozenset({'cluster', 'distribute', 'partition', 'sort'})
# The types that take fields in `<...>`; after any other, `<` compares.
_TYPES_WITH_FIELDS = frozenset({'array', 'map', 'struct'})
# The fields an interval type names, as in `INTERVAL DAY TO SECOND`.
_INTERVAL_FIELDS = frozenset(
    {'year', 'month', 'day', 'hour', 'minute', 'second'}
)
# The time zones a timestamp type may name.
_TIME_ZONES = (('with', 'local', 'time', 'zone'), ('without', 'time', 'zone'))
# Functions whose first argument may be a datetime unit, as a bare word.
_UNIT_FIRST = frozenset({'timediff', 'timestampadd', 'timestampdiff'})
# These take a unit first only in their three-argument form: `datediff(a,
# b)` reads two values.
_UNIT_FIRST_OF_THREE = frozenset(
    {'date_add', 'date_diff', 'dateadd', 'datediff'}
)


class UnreadableExpressionError(ValueError):
    r"""An expression whose names could not be settled within the bounds
    the scan keeps, so no reading of it is safe.
    """


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


def _type_end(tokens: list[_Token], at: int) -> int:
    r"""Where the type named at ``at`` ends: past its parameters, as in
    ``DECIMAL(38, 2)``, its fields, as in ``ARRAY<INT>``, ``STRUCT<>`` and
    ``INTERVAL DAY TO SECOND``, or its time zone. An alias, collation or
    window name ends where it starts.
    """
    word, after = tokens[at].folded, tokens[at + 1].text
    if after == '(' or (after == '<' and word in _TYPES_WITH_FIELDS):
        return _closing(tokens, at + 1)
    if after == '<>' and word == 'struct':
        return at + 1
    if word == 'interval' and tokens[at + 1].folded in _INTERVAL_FIELDS:
        return at + 3 if tokens[at + 2].folded == 'to' else at + 1
    if word == 'timestamp':
        words = tuple(token.folded for token in tokens[at + 1 : at + 5])
        for zone in _TIME_ZONES:
            if words[: len(zone)] == zone:
                return at + len(zone)
    return at


def _next(tokens: list[_Token], at: int) -> int:
    r"""Where the token after the one at ``at`` is, past the type a ``::``,
    ``?::`` or ``AS`` there introduces, read as the scan reads it, so a comma
    in ``MAP<STRING, INT>`` separates nothing.
    """
    casts = tokens[at].text in ('::', '?::') or tokens[at].folded == 'as'
    if casts and tokens[at + 1].kind == 'word':
        return _type_end(tokens, at + 1) + 1
    return at + 1


def _arguments(tokens: list[_Token], at: int) -> int:
    r"""How many arguments the call whose ``(`` is at ``at`` is given.

    A type that runs past the call's ``)`` leaves the call unclosed, and the
    scan refuses the expression whatever this counts.
    """
    commas, depth, end = 0, 0, _closing(tokens, at)
    while at < end:
        text = tokens[at].text
        depth += {'(': 1, ')': -1, '[': 1, ']': -1}.get(text, 0)
        commas += text == ',' and depth == 1
        at = _next(tokens, at)
    return commas + 1


def _lambda_scopes(tokens: list[_Token]) -> list[tuple[int, int, set[str]]]:
    r"""Each lambda's parameters, folded, with the tokens they are bound in:
    from the parameters to the end of the lambda's argument, at the comma or
    bracket after it. The next argument may be another lambda, as the
    finish of ``aggregate`` is, which binds names of its own.
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
            names = {
                _folded_name(t) for t in tokens[start:index] if _folded_name(t)
            }
        else:
            start, names = index - 1, {_folded_name(before)}
        depth, end, at = 0, len(tokens) - 1, start
        while at < len(tokens):
            text = tokens[at].text
            depth += {'(': 1, ')': -1, '[': 1, ']': -1}.get(text, 0)
            if depth < 0 or (depth == 0 and text == ','):
                end = at
                break
            at = _next(tokens, at)
        scopes.append((start, end, names))
    return scopes


@dataclass
class _Group:
    r"""A bracket open while scanning, and what is open inside it."""

    opener: str  # The word before it, folded, if any.
    # Each CASE open, by what it last read: CASE, WHEN, THEN or ELSE.
    cases: tuple[str, ...] = ()
    frame: bool = False  # Whether a window's frame is being read.


@dataclass
class _State:
    r"""What a scan carries from one token to the next."""

    groups: list[_Group] = field(default_factory=lambda: [_Group('')])
    operand: bool = True  # An operand is expected next.
    name_next: bool = False  # A type, alias, collation or window comes.
    after_is: bool = False  # Within `IS [NOT] ...`.
    case_at: int = -1  # Where a CASE's subject or first WHEN stands.
    by_at: int = -1  # Where the BY of `PARTITION BY` stands.
    skip_to: int = -1  # The end of a type.

    def taken(self) -> tuple[bool, bool, bool]:
        r"""What the last token left expected, cleared for the next."""
        left = self.operand, self.name_next, self.after_is
        self.operand = self.name_next = self.after_is = False
        return left

    def copy(self) -> _State:
        return replace(self, groups=[replace(g) for g in self.groups])

    def trial(self) -> _State:
        r"""A copy holding only the innermost group, which is all a trial to
        the bracket's end reads, under an outermost one.
        """
        return replace(self, groups=[_Group(''), replace(self.groups[-1])])

    def settled(self) -> bool:
        r"""Whether the innermost group may end here, nothing left owed."""
        return not (self.operand or self.name_next or self.groups[-1].cases)

    def key(self) -> tuple[object, ...]:
        r"""What a reading to the end of the current bracket depends on."""
        flags = (v for k, v in vars(self).items() if k != 'groups')
        return (*flags, *vars(self.groups[-1]).values())


class _NameScan:
    r"""Reads an expression's tokens in order, keeping whether an operand is
    expected next, which decides whether a word names a value.

    Where a word may open a form or be a column, as NOT, CASE and a CASE's
    first WHEN may, Databricks' parser takes the form if the rest parses,
    and so does the scan. A trial reads on as if the word opened the form,
    to the end of its bracket, checking that each token may stand where it
    does. Meeting another such word in that bracket, it takes the form and
    comes back to read a column if the form fails; one in a deeper bracket
    is a trial of its own. What a trial finds is kept by where it starts
    and the state it starts in.
    """

    def __init__(self, tokens: list[_Token]) -> None:
        self.end = len(tokens)
        # Past the end, and so at index -1, stand tokens of no kind.
        self.tokens = [*tokens, *[_Token('end', '', -1, -1)] * 4]
        self.scopes = _lambda_scopes(self.tokens)
        self.parsed: dict[tuple[object, ...], bool] = {}
        # For each trial under way, the readings as a column left to try.
        self.trials: list[list[tuple[int, _State, tuple[object, ...]]]] = []
        # Past this many steps in all, or the deepest trial, the scan stops
        # trying and keeps nothing more, and the expression is refused as
        # unreadable. So a long expression is read in time linear in its
        # length, and none is read wrong.
        self.budget = _TRIAL_STEPS * (self.end + 1)
        self.cut = False
        self.parses = True  # Whether the whole expression parses.
        # Where the bracket around each token closes, or the end.
        around, closing, open_at = [], {-1: self.end}, [-1]
        for index, token in enumerate(tokens):
            if token.text in ')]' and len(open_at) > 1:
                closing[open_at.pop()] = index
            around.append(open_at[-1])
            if token.text in '([':
                open_at.append(index)
        self.closes = [closing.get(at, self.end) for at in around]

    def references(self) -> list[NameReference]:
        state, found = _State(), list[NameReference]()
        index = 0
        while index < self.end:
            index, parses = self._step(index, state, found)
            self.parses = self.parses and parses
        self.parses = self.parses and len(state.groups) == 1 and state.settled()
        return found

    def _parses(self, index: int, state: _State) -> bool:
        r"""Whether the tokens from ``index`` to the end of the current
        bracket parse, read from ``state``.
        """
        key = (index, *state.key())
        if key in self.parsed:
            return self.parsed[key]
        if self.budget <= 0 or len(self.trials) >= _DEEPEST_TRIAL:
            self.cut = True
            return True  # Undecided: the expression will be refused.
        columns: list[tuple[int, _State, tuple[object, ...]]] = []
        self.trials.append(columns)
        state = state.trial()
        while not (parses := self._reads_on(index, state)) and columns:
            index, state, form = columns.pop()
            self._keep(form, False)
        self.trials.pop()
        for _, _, form in columns:
            self._keep(form, True)  # Each form taken on the way parsed.
        self._keep(key, parses)
        return parses

    def _keep(self, key: tuple[object, ...], parses: bool) -> None:
        if not self.cut:
            self.parsed[key] = parses

    def _reads_on(self, index: int, state: _State) -> bool:
        r"""Whether every token from ``index`` to the end of the current
        bracket may stand where it does, read from ``state``.
        """
        while index < self.end and len(state.groups) > 1:
            known = self.parsed.get((index, *state.key()))
            if known is not None and (not known or len(state.groups) == 2):
                return known  # Read from here before, from this state.
            if known:
                # The rest of this deeper bracket parses: go on after it.
                index = self.closes[index] + 1
                state.groups.pop()
                state.operand = state.name_next = state.after_is = False
                continue
            if self.budget <= 0:
                self.cut = True
                return True
            self.budget -= 1
            index, parses = self._step(index, state, None)
            if not parses:
                return False
        # A `)` checks its own group as it closes it.
        return len(state.groups) == 1 or state.settled()

    def _step(
        self, index: int, state: _State, found: list[NameReference] | None
    ) -> tuple[int, bool]:
        r"""Reads the token or name at ``index``, adding a name read as a
        value to ``found``. Returns where the next starts, and whether what
        was read may stand where it does.
        """
        tokens = self.tokens
        if index <= state.skip_to:
            return index + 1, True
        if tokens[index].kind not in _NAME_KINDS:
            if found is not None and self._star(index, state):
                found.append(
                    NameReference(
                        ('*',), (tokens[index].start, tokens[index].end), False
                    )
                )
            return index + 1, self._punctuation(index, state)
        last = index
        while (
            tokens[last + 1].text == '.'
            and tokens[last + 2].kind in _NAME_KINDS
        ):
            last += 2
        if tokens[last + 1].text == '.' and tokens[last + 2].text == '*':
            last += 2  # `t.*`, every column of `t`.
        value, parses = self._read_name(index, last, state)
        if value and found is not None:
            found.append(
                NameReference(
                    tuple(_unquoted(t) for t in tokens[index : last + 1 : 2]),
                    (tokens[index].start, tokens[last].end),
                    last == index and tokens[index].kind == 'word',
                )
            )
        return last + 1, parses

    def _star(self, index: int, state: _State) -> bool:
        r"""Whether the token at ``index`` is a ``*`` reading every column,
        as in ``struct(*)``. ``COUNT(*)`` counts rows instead.
        """
        tokens = self.tokens
        if tokens[index].text != '*' or not state.operand:
            return False
        return not (
            state.groups[-1].opener == 'count'
            and tokens[index - 1].text == '('
            and tokens[index + 1].text == ')'
        )

    def _punctuation(self, index: int, state: _State) -> bool:
        r"""Reads the token at ``index``, no name, and says whether it may
        stand where it does.
        """
        token, previous = self.tokens[index], self.tokens[index - 1]
        text, operand = token.text, state.operand
        state.name_next, state.after_is = text in ('::', '?::'), False
        if text in '([':
            state.groups.append(_Group(previous.folded))
            state.operand = True
            return operand or text == '['
        if text in ')]':
            closed = len(state.groups) > 1 and not state.groups.pop().cases
            state.operand = False
            return closed and (not operand or previous.text == '(')
        if token.kind in ('number', 'string'):
            state.operand = False
            # `INTERVAL 1 YEAR 2 MONTHS` gives a value after each unit.
            return operand or previous.folded in _DATETIME_UNITS
        if text in '+-~' or (text == '*' and operand):
            state.operand = text != '*'  # A sign, or COUNT(*).
            return True
        if (
            previous.end == token.start
            and previous.text + text in _OPERATOR_PAIRS
        ):
            return True  # The second half of an operator.
        if text == '!' and self.tokens[index + 1].text != '=':
            return operand  # Not, as in `!x`.
        state.operand = True
        return not operand  # An operator after an operand, or a comma.

    def _follows(self, word: str, state: _State, at: int) -> bool:
        r"""Whether ``word`` at ``at`` may stand right after an operand."""
        group, before = state.groups[-1], self.tokens[at - 1]
        after = self.tokens[at + 1].folded
        if word in _CASE_STEPS:
            return bool(group.cases) and group.cases[-1] in _CASE_STEPS[word]
        if word in ('preceding', 'following', 'row'):
            return group.frame
        if word in _FOLLOWED_BY:
            return after in _FOLLOWED_BY[word]
        if word == 'nulls':
            return after in ('first', 'last') or before.folded in (
                'ignore',
                'respect',
            )
        if word in ('first', 'last'):
            return before.folded == 'nulls'
        if word == 'to':
            return before.folded in _DATETIME_UNITS
        if word in _DATETIME_UNITS:
            # `INTERVAL 1 DAY`, `'1' DAY TO HOUR`, `INTERVAL (n) DAYS`.
            return (
                before.kind in ('number', 'string')
                or before.text == ')'
                or before.folded == 'to'
                or self.tokens[at - 2].folded == 'interval'
            )
        if word in _TAKE_AN_OPERAND or word in ('asc', 'desc'):
            return True
        return self._implicit_alias(at, state)

    def _implicit_alias(self, at: int, state: _State) -> bool:
        r"""Whether the name at ``at``, right after an operand, is an alias
        STRUCT takes without AS: ``struct(x a, y `b c`)``.
        """
        alias = self.tokens[at + 1].text in (',', ')')
        return alias and state.groups[-1].opener == 'struct'

    def _from_follows(self, at: int) -> bool:
        r"""Whether a FROM follows the token at ``at`` in its own bracket."""
        depth = 0
        for token in self.tokens[at + 1 : self.closes[at]]:
            depth += {'(': 1, ')': -1, '[': 1, ']': -1}.get(token.text, 0)
            if depth == 0 and token.folded == 'from':
                return True
        return False

    def _names_a_field(self, at: int, state: _State) -> bool:
        r"""Whether the name at ``at``, first in its call, is the field of
        ``EXTRACT(DOW FROM d)`` or the unit of ``dateadd(DAY, 1, d)``. Either
        may be quoted, and as Databricks' parser, which refuses a unit it
        does not know only later, any lone name there is one.

        A qualified name calls a function someone wrote, never the builtin,
        so nothing in its call is a field or a unit, as for a cast.
        """
        opener = state.groups[-1].opener
        if self.tokens[at - 1].text != '(':
            return False
        if self.tokens[at - 3].text == '.':
            return False
        if opener == 'extract':
            return True
        three = opener in _UNIT_FIRST_OF_THREE
        unit_first = opener in _UNIT_FIRST or (
            three and _arguments(self.tokens, at - 1) == 3
        )
        return unit_first and self.tokens[at + 1].text == ','

    def _read_name(
        self, index: int, last: int, state: _State
    ) -> tuple[bool, bool]:
        r"""Reads the name at ``index`` to ``last``, and says whether it is
        read as a value and whether it may stand where it does.
        """
        tokens, group = self.tokens, state.groups[-1]
        previous, after = tokens[index - 1], tokens[last + 1]
        word = tokens[index].folded if index == last else ''
        name = _folded_name(tokens[index]) if index == last else ''
        operand, name_next, after_is = state.taken()
        if previous.text in ('.', ':'):
            return False, True  # A field or a JSON path's step.
        if name_next:
            if (end := _type_end(tokens, last)) > last:
                state.skip_to = end
            return False, True
        if after.text == '(' and operand:
            # `NOT (x)` and `CASE (x) WHEN` open forms, and others call.
            self._opens_a_form(index, word, state, after_is)
            state.operand = True
            return False, True
        if after.text == '=>':
            return False, operand  # A named argument.
        if name and any(
            s <= index <= e and name in n for s, e, n in self.scopes
        ):
            return False, operand  # A lambda's parameter.
        if not word:
            # Quoted or qualified: a value, but for a quoted field, unit or
            # alias, as in ``EXTRACT(`DOW` FROM d)``.
            if (
                tokens[index].kind == 'quoted'
                and index == last
                and (
                    self._names_a_field(index, state)
                    if operand
                    else self._implicit_alias(last, state)
                )
            ):
                return False, True
            return True, operand
        if index == state.by_at:
            state.operand = True
            return False, True
        if after.folded == 'by' and (
            word == 'order' or (word in _WINDOW_BY and group.opener == 'over')
        ):
            # After an operand, or first in `OVER (`.
            state.by_at = last + 1
            return False, not operand or previous.text == '('
        if operand:
            return not self._opens_a_form(index, word, state, after_is), True
        if word == 'except' and previous.text == '*' and after.text == '(':
            # The columns `* EXCEPT (...)` leaves out are read as no value.
            state.skip_to = _closing(tokens, last + 1)
            return False, True
        # Right after an operand: an operator or a clause's word.
        parses = self._follows(word, state, index)
        state.operand = word in _TAKE_AN_OPERAND
        state.name_next = word in _TAKE_A_NAME
        state.after_is = word == 'is'
        group.frame = group.frame or word in ('rows', 'range')
        if word == 'end':
            group.cases = group.cases[:-1]
        elif word in _CASE_STEPS and group.cases:
            group.cases = (*group.cases[:-1], word)
        return False, parses

    def _opens_a_form(
        self, index: int, word: str, state: _State, after_is: bool
    ) -> bool:
        r"""Whether ``word``, where an operand is expected, is grammar,
        moving ``state`` on past it.
        """
        form = self._fixed_form(index, word, state, after_is)
        return self._tried_form(index, word, state) if form is None else form

    def _fixed_form(
        self, index: int, word: str, state: _State, after_is: bool
    ) -> bool | None:
        r"""Whether ``word`` at ``index`` is grammar by where it stands, or
        ``None`` if a trial decides.
        """
        group, tokens = state.groups[-1], self.tokens
        after, opens = tokens[index + 1], tokens[index - 1].text == '('
        if word in _LITERAL_WORDS:
            return True
        if after_is and word in _AFTER_IS:
            state.after_is = word in ('not', 'distinct')
            state.operand = word in ('not', 'distinct', 'from')
            return True
        if group.frame and word in _FRAME_WORDS:
            state.operand = word == 'between'
            return True
        if self._names_a_field(index, state):
            return True
        if opens and group.opener == 'trim' and word in _TRIM_SIDES:
            # `TRIM(BOTH FROM s)`, `TRIM(BOTH from FROM s)`, and `TRIM(both)`
            # and `TRIM(both < x)`, which have no FROM. In `TRIM(BOTH FROM
            # from)` no operand follows the second `from`.
            form = self._from_follows(index)
            third = tokens[index + 3]
            column = tokens[index + 2].folded == 'from' and (
                third.kind in (*_NAME_KINDS, 'number', 'string')
                or third.text in ('(', '+', '-', '~')
            )
            state.operand = form and (after.folded != 'from' or column)
            return form
        if opens and group.opener == 'over' and word in ('rows', 'range'):
            if after.folded in _FRAME_STARTS or after.kind == 'number':
                group.frame = state.operand = True
                return True
        if word == 'interval':
            # `INTERVAL 1 DAY` and `INTERVAL -1 DAY`, where `interval - 1`
            # and `interval ASC` read a column.
            signed = after.text in ('+', '-')
            value = signed or (
                after.kind == 'word' and after.folded not in _TAKE_AN_OPERAND
            )
            unit = tokens[index + 2 + signed].folded in _DATETIME_UNITS
            state.operand = (
                after.kind in ('number', 'string')
                or after.text == '('
                or (value and unit)
            )
            return state.operand
        if word == 'where':
            # FILTER takes a WHERE, and nothing else does.
            state.operand = opens and group.opener == 'filter'
            return state.operand
        if word in _TYPED_LITERALS and after.kind == 'string':
            state.operand = True
            return True
        return None

    def _tried_form(self, index: int, word: str, state: _State) -> bool:
        r"""Whether ``word`` at ``index``, which may open a form or be a
        column, opens the form: whether what follows then parses.
        """
        form = state.trial()
        inner = form.groups[-1]
        if word == 'when' and index == state.case_at:
            # A searched CASE's first WHEN, or a subject named `when`.
            inner.cases = (*inner.cases[:-1], 'when')
        elif word == 'case':
            inner.cases, form.case_at = (*inner.cases, 'case'), index + 1
        elif word != 'not' and not (
            word in ('all', 'distinct')
            and self.tokens[index - 1].text == '('
            and inner.opener not in _NO_CALL_OPENERS
        ):
            return False
        form.operand = True
        if self.trials and len(state.groups) == 2:
            # In the bracket this trial reads to the end of: take the form,
            # and keep the column to come back to.
            column = (index + 1, state.copy(), (index + 1, *form.key()))
            self.trials[-1].append(column)
        elif not self._parses(index + 1, form):
            return False
        state.groups[-1].cases, state.case_at = inner.cases, form.case_at
        state.operand = True
        return True


def name_references(expr: str) -> list[NameReference]:
    r"""The names a Databricks expression reads as values.

    That leaves out a function's name, a type, a keyword where it is
    grammar, a datetime unit, a named argument, a lambda's parameter and a
    JSON path step, so a word of the grammar is never read as a column or a
    parameter. Where a word stands decides which it is, as it does for
    Databricks: in ``SUM(x) OVER (PARTITION BY range)`` the ``range`` is a
    column, and in ``col REGEXP 'x'`` the ``REGEXP`` is an operator.

    An expression that does not parse as the scan reads SQL is refused,
    since a failed reading of a form makes its word a column: an unknown
    clause late in the expression would turn an earlier ``NOT`` into one.

    Raises:
        UnreadableExpressionError: If the expression does not parse as the
            scan reads SQL, or brackets nest too deeply, or it is too long,
            to settle which words are names.
    """
    tokens = [token for token in _lex(expr) if token.kind != 'comment']
    scan = _NameScan(tokens)
    references = scan.references()
    if scan.cut:
        raise UnreadableExpressionError(
            'it is too deeply nested or too long to read safely'
        )
    if not scan.parses:
        raise UnreadableExpressionError(
            'it has syntax that is not understood, so which of its words '
            'name columns cannot be settled'
        )
    return references


def _folded_name(token: _Token) -> str:
    r"""The folded name ``token`` gives, quoted or not, or ``''``."""
    return _unquoted(token).lower() if token.kind in ('word', 'quoted') else ''


def _unquoted(token: _Token) -> str:
    if token.kind == 'quoted':
        quote = token.text[0]
        return token.text[1:-1].replace(quote * 2, quote)
    return token.text
