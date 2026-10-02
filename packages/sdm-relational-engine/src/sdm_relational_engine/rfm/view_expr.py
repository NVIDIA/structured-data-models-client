# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Checks on SQL expressions taken from a metric or semantic view.

A view is authored by a third party, yet its expressions are spliced verbatim
into sampler queries and run under the caller's warehouse credentials.
"""

from __future__ import annotations

import re

_UNSAFE_EXPR_TOKENS = (';', '--', '/*', '*/', '|>', '->>')
_EXPR_QUOTE_DELIMITERS = ('$$', "'", '"', '`')
_QUOTED_EXPR_TEXT = re.compile(
    '|'.join(
        [
            r'\$\$[\s\S]*?\$\$',
            r"'(?:''|\\[\s\S]|[^'\\])*'",
            r'"(?:""|\\[\s\S]|[^"\\])*"',
            r'`(?:``|[^`])*`',
        ]
    )
)
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
    wrong defeats the guard. Snowflake also has ``$$...$$``, and both
    Snowflake and Databricks honour backslash escapes inside a literal. A
    payload using ``$$`` or ``\'`` shifts where a doubling-only reader sees the
    literal end and can smuggle a sub-query through the gap, so both forms are
    recognised. A remaining delimiter means this function and the warehouse
    disagree about string boundaries, so it fails closed. This also refuses a
    quoted name ending in a lone backslash, which is invalid on Databricks and
    exotic on Snowflake.

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
    if dialect == 'snowflake':
        for quoted in _QUOTED_EXPR_TEXT.findall(expr):
            if quoted.startswith('"') and '\\' in quoted:
                return 'it has a backslash inside a double-quoted name'
    unquoted = _QUOTED_EXPR_TEXT.sub(' ', expr)
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

# Type names take `(` for their parameters behind `AS` or `::`.
_TYPE_NAMES = frozenset(
    {
        'array', 'binary', 'char', 'character', 'dec', 'decimal', 'double',
        'float', 'map', 'nchar', 'number', 'numeric', 'nvarchar', 'object',
        'string', 'struct', 'text', 'time', 'timestamp', 'timestamp_ltz',
        'timestamp_ntz', 'timestamp_tz', 'varbinary', 'varchar', 'vector',
    }
)  # fmt: skip

# Read without parentheses, these return the session's identity or clock.
_SESSION_VALUES = re.compile(
    r'(?<![\w$.])'
    r'(current_(?:catalog|database|date|role|schema|time|timestamp|user)'
    r'|localtime|localtimestamp|session_user)\b(?!\s*\()',
    flags=re.IGNORECASE,
)

# A private-use character stands in for quoted text, so a call to a quoted
# name stays visible as a call.
_QUOTED = '\ue000'
# What may stand right before a `(` that opens a group rather than a call.
_OPENS_A_GROUP = frozenset('+-*/%=<>!|&^~,([?:')
_LAST_TOKEN = re.compile(r'([\w$\ue000]+)\s*$')
_QUALIFIED = re.compile(r'\.\s*$')
_TYPE_CONTEXT = re.compile(r'(?:(?<![\w$])as|::)\s*$', flags=re.IGNORECASE)


def _acts_as_keyword(word: str, lead: str) -> bool:
    r"""Whether ``word``, followed by ``(``, is a keyword rather than a call.

    Databricks lets most of these words name a function too, so ``escape(x)``
    calls one. As a keyword each follows a value, a ``)`` or another keyword,
    as in ``x IN (...)`` and ``TRIM(BOTH ' ' FROM x)``. ``NOT`` and ``CASE``
    can also begin an expression.

    Args:
        word: The folded word before ``(``.
        lead: The expression text before ``word``.
    """
    if word in ('not', 'case'):
        return True
    lead = lead.rstrip()
    return bool(lead) and (
        lead[-1] in ')]' + _QUOTED or bool(re.match(r'[\w$]', lead[-1]))
    )


def _untrusted_call_reason(expr: str, dialect: str) -> str | None:
    r"""Return why an expression calls a function that is not trusted, or
    ``None``.

    Only the dialect's pure, deterministic, built-in scalar functions are
    trusted, called by their plain name. A user-defined function, or a
    built-in like ``secret()``, ``http_request()`` or ``SYSTEM$WAIT()``, would
    otherwise run with the caller's credentials every time a graph samples.
    Qualified and quoted names, and ``IDENTIFIER()``, are refused since they
    can name anything. Session values (``current_user``) are refused too: a
    column must not change with who reads it.

    Every ``(`` is checked, and fails closed: it must open a group, follow
    a keyword, give a type its parameters, or call a trusted function. So a
    name this does not recognise as a name is refused rather than skipped.

    Apply :func:`_unsafe_expr_reason` first, which establishes that every
    quote in ``expr`` is paired.

    Args:
        expr: The expression as written in the view definition.
        dialect: ``'databricks'`` or ``'snowflake'``.
    """
    trusted = _TRUSTED_FUNCTIONS[dialect]
    text = _QUOTED_EXPR_TEXT.sub(_QUOTED, expr)
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
            folded, before[: token.start()]
        ):
            continue
        if folded in _TYPE_NAMES and _TYPE_CONTEXT.search(
            before[: token.start()]
        ):
            continue
        if folded not in trusted:
            return (
                f"it calls '{name}', which is not a pure built-in SQL function"
            )
    if session := _SESSION_VALUES.search(text):
        return f"it reads the session value '{session.group(1)}'"
    return None
