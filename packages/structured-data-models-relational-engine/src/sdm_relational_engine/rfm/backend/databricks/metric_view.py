# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

from sdm_relational_engine.exceptions import KumoRelationalError
from sdm_relational_engine.rfm.view_expr import (
    _unsafe_expr_reason,
    _untrusted_call_reason,
)

if TYPE_CHECKING:
    from sdm_relational_engine.rfm.backend.databricks import AnyConnection

SOURCE_ALIAS = 'source'

_CARDINALITIES = {'many_to_one', 'one_to_many'}
_MEASURE_SUFFIX = ' measure'

_BARE_IDENT = r'[A-Za-z_][A-Za-z0-9_]*'
_IDENT = rf'`(?:[^`]|``)+`|{_BARE_IDENT}'

_TABLE_REFERENCE_RE = re.compile(
    rf'\s*({_IDENT})(?:\.({_IDENT}))?(?:\.({_IDENT}))?\s*$'
)
_CHAIN = rf'(?:{_IDENT})(?:\s*\.\s*(?:{_IDENT}))*'
_JOIN_CONDITION_RE = re.compile(rf'\s*({_CHAIN})\s*=\s*({_CHAIN})\s*$')
_IDENT_RE = re.compile(_IDENT)
# Double quotes delimit a string, not a name, in Databricks SQL.
_STRING_LITERAL_RE = re.compile(
    r"'(?:''|\\[\s\S]|[^'\\])*'|\"(?:\"\"|\\[\s\S]|[^\"\\])*\""
)


class UnsupportedJoinError(KumoRelationalError, ValueError):
    r"""Raised when a metric view join cannot be converted into a graph
    edge.
    """


class UnsupportedFilterError(KumoRelationalError, ValueError):
    r"""Raised when a metric view's filter cannot be applied to its source
    table.
    """


@dataclass(frozen=True)
class JoinKeyRef:
    r"""One side of a metric view join condition.

    Args:
        qualifier: The (dot-joined) table alias or join path qualifying the
            column, if any.
        column: The referenced column name.
    """

    qualifier: str | None
    column: str


@dataclass(frozen=True)
class MetricViewJoin:
    r"""A join of a metric view definition.

    Args:
        alias: The join name, used as the graph table name.
        table: The parsed name parts of the joined table.
        parent_alias: The alias of the table this join is declared on, either
            :obj:`SOURCE_ALIAS` or the alias of an enclosing join.
        path: The aliases of all enclosing joins followed by ``alias``.
            Dimensions reference nested joins by this full path, *e.g.*,
            ``customer.nation.n_name``.
        condition: The two sides of the single-column equality join condition.
        cardinality: Either ``'many_to_one'`` or ``'one_to_many'``.
    """

    alias: str
    table: tuple[str, ...]
    parent_alias: str
    path: tuple[str, ...]
    condition: tuple[JoinKeyRef, JoinKeyRef]
    cardinality: str


@dataclass(frozen=True)
class MetricViewColumn:
    r"""A metric view dimension resolved onto a single table.

    Args:
        name: The dimension name, used as the column name.
        alias: The alias of the table the dimension refers to, either
            :obj:`SOURCE_ALIAS` or a join alias.
        expr: The dimension expression with the alias qualifier stripped.
    """

    name: str
    alias: str
    expr: str


@dataclass
class MetricViewSpec:
    r"""The graph-relevant parts of a metric view definition.

    Args:
        source: The parsed name parts of the source (fact) table.
        joins: All convertible joins, parents before children.
        columns: All dimensions resolved onto a single table.
        messages: Human-readable messages for every dropped element.
        filter: The view's ``filter``, a condition on the source table's
            columns, if it has one.
    """

    source: tuple[str, ...]
    joins: list[MetricViewJoin]
    columns: list[MetricViewColumn]
    messages: list[str]
    filter: MetricViewFilter | None = None


@dataclass(frozen=True)
class MetricViewFilter:
    r"""A metric view's ``filter``, held back until what it calls is checked.

    Every read of the source table runs the filter under the caller's
    credentials, and which of its words are columns is known only once the
    source table's columns are read. So the SQL is handed out only by
    :meth:`checked`, given those columns, or by :meth:`trusted`.

    Args:
        _sql: The condition, which already references no joined table and
            holds no sub-query, comment or statement separator.
    """

    _sql: str

    def checked(self, columns: Collection[str]) -> str:
        r"""Returns the filter if it calls only pure built-in functions.

        Args:
            columns: The columns of the source table.

        Raises:
            UnsupportedFilterError: If the filter calls a function that is
                not trusted, or reads a session value.
        """
        reason = _untrusted_call_reason(
            self._sql, 'databricks', columns=columns
        )
        if reason is not None:
            _refuse_filter(self._sql, reason)
        return self._sql

    def trusted(self) -> str:
        r"""Returns the filter unchecked, for a caller who trusts whoever
        authored the view.
        """
        return self._sql


def unquote_ident(name: str) -> str:
    r"""Removes backtick quoting from an identifier, if present."""
    name = name.strip()
    if len(name) >= 2 and name.startswith('`') and name.endswith('`'):
        return name[1:-1].replace('``', '`')
    return name


def parse_table_reference(name: Any) -> tuple[str, ...] | None:
    r"""Parses a table reference into its unquoted name parts.

    Returns ``None`` if the reference is not a plain (optionally qualified)
    table name, *e.g.*, a SQL query.
    """
    if not isinstance(name, str):
        return None
    match = _TABLE_REFERENCE_RE.match(name)
    if match is None:
        return None
    return tuple(
        unquote_ident(part) for part in match.groups() if part is not None
    )


def parse_join_condition(
    condition: Any,
) -> tuple[JoinKeyRef, JoinKeyRef] | None:
    r"""Parses a single-column equality join condition.

    Column references may be qualified by an alias or a (dotted) join path.
    Returns ``None`` for any other condition, *e.g.*, composite or
    non-equality joins.
    """
    if not isinstance(condition, str):
        return None
    match = _JOIN_CONDITION_RE.match(condition)
    if match is None:
        return None

    def _ref(chain: str) -> JoinKeyRef:
        parts = [unquote_ident(part) for part in _IDENT_RE.findall(chain)]
        return JoinKeyRef(
            qualifier='.'.join(parts[:-1]) or None,
            column=parts[-1],
        )

    left, right = match.groups()
    return _ref(left), _ref(right)


def _quoted_variants(name: str) -> str:
    quoted = re.escape('`' + name.replace('`', '``') + '`')
    if re.fullmatch(_BARE_IDENT, name):
        return f'{quoted}|{re.escape(name)}'
    return quoted


def _to_path(alias: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(alias, str):
        return (alias,)
    return tuple(alias)


def _path_pattern(path: Sequence[str]) -> re.Pattern[str]:
    chain = r'\s*\.\s*'.join(f'(?:{_quoted_variants(part)})' for part in path)
    return re.compile(
        rf'(?<![A-Za-z0-9_`$.]){chain}\s*\.',
        flags=re.IGNORECASE,
    )


def _field_pattern(name: str) -> re.Pattern[str]:
    return re.compile(
        rf'(?<![A-Za-z0-9_`$.])(?:{_quoted_variants(name)})(?![A-Za-z0-9_`])',
        flags=re.IGNORECASE,
    )


def _sub_outside_strings(
    pattern: re.Pattern[str],
    repl: str | Callable[[re.Match[str]], str],
    expr: str,
) -> str:
    r"""``pattern.sub(repl, expr)`` with every string literal left as is.

    A literal holding `source.` or a field's name is data, and rewriting it
    changes the values the view declares.
    """
    out: list[str] = []
    last = 0
    for match in _STRING_LITERAL_RE.finditer(expr):
        out.append(pattern.sub(repl, expr[last : match.start()]))
        out.append(match.group())
        last = match.end()
    out.append(pattern.sub(repl, expr[last:]))
    return ''.join(out)


def references_alias(expr: str, alias: str | Sequence[str]) -> bool:
    r"""Returns ``True`` if the expression references a column of the given
    table alias or join path outside its string literals.
    """
    masked = _STRING_LITERAL_RE.sub("''", expr)
    return _path_pattern(_to_path(alias)).search(masked) is not None


def strip_alias_qualifier(expr: str, alias: str | Sequence[str]) -> str:
    r"""Removes all qualified references to the given table alias or join
    path from the expression, leaving its string literals as they are.
    """
    pattern = _path_pattern(_to_path(alias))
    return _sub_outside_strings(pattern, '', expr).strip()


def parse_metric_view(definition: str) -> MetricViewSpec:
    r"""Parses a metric view YAML definition into its graph-relevant parts.

    Measures define aggregations rather than row-level values, and are
    therefore never part of the returned specification. Elements that cannot
    be represented in a graph (SQL query join sources, composite or
    non-equality join conditions, dimensions referencing multiple tables,
    parameters, and materializations) are dropped and reported in
    :attr:`MetricViewSpec.messages`.

    The ``filter`` decides which rows the view holds, so it is never dropped:
    it is returned to be applied to the source table, or the view is refused.
    Which functions it calls depends on the source table's columns, so it
    comes back as a :class:`MetricViewFilter` that gives out its SQL only
    once checked against them.

    Args:
        definition: The YAML definition.

    Raises:
        ValueError: If the definition is not a YAML mapping, misses a source,
            or defines a SQL query as its source.
        UnsupportedFilterError: If the filter references a joined table, or
            is not a safe scalar condition.
    """
    import yaml

    cfg = yaml.safe_load(definition)
    if not isinstance(cfg, dict):
        raise ValueError('The metric view definition is not a YAML mapping')

    source = cfg.get('source')
    if source is None:
        raise ValueError("The metric view definition misses a 'source'")
    source_ref = parse_table_reference(source)
    if source_ref is None:
        raise ValueError(
            f'Unsupported metric view source {source!r}. Only plain table '
            f'names are supported as the source of a metric view when '
            f'converting it into a graph'
        )

    messages: list[str] = []
    for key in ('parameters', 'materialization'):
        if key in cfg:
            messages.append(
                f"Ignored the '{key}' section since it cannot "
                f'be represented in a graph'
            )

    joins: list[MetricViewJoin] = []
    aliases: set[str] = set()
    kept: set[str] = set()

    def _collect_aliases(join_cfgs: Any) -> None:
        for join_cfg in join_cfgs or []:
            if isinstance(join_cfg, dict):
                if isinstance(join_cfg.get('name'), str):
                    aliases.add(join_cfg['name'])
                _collect_aliases(join_cfg.get('joins'))

    def _add_joins(join_cfgs: Any, parent_path: tuple[str, ...]) -> None:
        parent_alias = parent_path[-1] if parent_path else SOURCE_ALIAS
        for join_cfg in join_cfgs or []:
            if not isinstance(join_cfg, dict):
                messages.append(
                    f'Failed to add join {join_cfg!r} since it is not a mapping'
                )
                continue

            alias = join_cfg.get('name')
            if not isinstance(alias, str):
                messages.append(
                    f'Failed to add join {join_cfg!r} since it misses a name'
                )
                continue

            def _skip(msg: str) -> None:
                messages.append(f"Failed to add join '{alias}' since {msg}")
                aliases.add(alias)
                _collect_aliases(join_cfg.get('joins'))

            if alias == SOURCE_ALIAS or alias in kept:
                _skip(f"the name '{alias}' is already in use")
                continue

            table = parse_table_reference(join_cfg.get('source'))
            if table is None:
                _skip('only plain table names are supported as join sources')
                continue

            cardinality = join_cfg.get('cardinality', 'many_to_one')
            if cardinality not in _CARDINALITIES:
                _skip(f"of its unsupported cardinality '{cardinality}'")
                continue

            using = join_cfg.get('using')
            if isinstance(using, str):
                using = [using]
            if using is not None:
                if not isinstance(using, list) or len(using) != 1:
                    _skip('composite key references are not yet supported')
                    continue
                column = unquote_ident(str(using[0]))
                condition = (
                    JoinKeyRef(qualifier=parent_alias, column=column),
                    JoinKeyRef(qualifier=alias, column=column),
                )
            else:
                on_cfg = join_cfg.get('on', join_cfg.get(True))
                parsed = parse_join_condition(on_cfg)
                if parsed is None:
                    _skip(
                        'only single-column equality join conditions are '
                        'supported'
                    )
                    continue
                condition = parsed

            joins.append(
                MetricViewJoin(
                    alias=alias,
                    table=table,
                    parent_alias=parent_alias,
                    path=parent_path + (alias,),
                    condition=condition,
                    cardinality=cardinality,
                )
            )
            aliases.add(alias)
            kept.add(alias)
            _add_joins(
                join_cfg.get('joins'), parent_path=parent_path + (alias,)
            )

    _add_joins(cfg.get('joins'), parent_path=())

    candidates: list[tuple[tuple[str, ...], str]] = []
    for join in joins:
        candidates.append((join.path, join.alias))
        if len(join.path) > 1:
            candidates.append(((join.alias,), join.alias))
    for alias in aliases - kept:
        candidates.append(((alias,), alias))
    candidates.append(((SOURCE_ALIAS,), SOURCE_ALIAS))
    candidates.sort(key=lambda candidate: len(candidate[0]), reverse=True)

    row_filter = _check_filter(cfg.get('filter'), candidates)

    columns: list[MetricViewColumn] = []
    defined: dict[str, str] = {}
    dim_cfgs = list(cfg.get('dimensions') or []) + list(cfg.get('fields') or [])
    for dim_cfg in dim_cfgs:
        if not isinstance(dim_cfg, dict):
            messages.append(
                f'Failed to add dimension {dim_cfg!r} since it is not a mapping'
            )
            continue

        name = dim_cfg.get('name')
        expr = dim_cfg.get('expr')
        if not isinstance(name, str) or not isinstance(expr, str):
            messages.append(
                f'Failed to add dimension {dim_cfg!r} since it '
                f'misses a name or expression'
            )
            continue

        for field_name, field_expr in defined.items():
            expr = _sub_outside_strings(
                _field_pattern(field_name),
                lambda match: f'({field_expr})',
                expr,
            )
        defined[name] = expr

        refs: set[str] = set()
        for path, path_alias in candidates:
            if references_alias(expr, path):
                refs.add(path_alias)
                expr = strip_alias_qualifier(expr, path)
        if len(refs) > 1:
            messages.append(
                f"Failed to add dimension '{name}' since its "
                f'expression references multiple tables'
            )
            continue

        alias = refs.pop() if refs else SOURCE_ALIAS
        if alias != SOURCE_ALIAS and alias not in kept:
            messages.append(
                f"Failed to add dimension '{name}' since it "
                f"references the skipped join '{alias}'"
            )
            continue

        columns.append(
            MetricViewColumn(
                name=name,
                alias=alias,
                expr=expr,
            )
        )

    return MetricViewSpec(
        source=source_ref,
        joins=joins,
        columns=columns,
        messages=messages,
        filter=row_filter,
    )


def _check_filter(
    row_filter: Any,
    candidates: Sequence[tuple[tuple[str, ...], str]],
) -> MetricViewFilter | None:
    r"""Returns the filter if it can be applied to the source table alone.

    The graph keeps each join as a table of its own, so a condition on a
    joined table has no single table to apply to. Dropping it instead would
    let predictions read rows the view leaves out.

    Args:
        row_filter: The ``filter`` section, if any.
        candidates: Every join path or alias a column can be qualified by,
            with the alias it resolves to.
    """
    if row_filter is None:
        return None

    reason: str | None = None
    if not isinstance(row_filter, str) or not row_filter.strip():
        reason = 'it is not a SQL condition'
    else:
        reason = _unsafe_expr_reason(row_filter, 'databricks')
    if reason is None:
        assert isinstance(row_filter, str)
        masked = _STRING_LITERAL_RE.sub("''", row_filter)
        for path, alias in candidates:
            if alias != SOURCE_ALIAS and references_alias(masked, path):
                reason = f"it references the joined table '{alias}'"
                break
    if reason is not None:
        _refuse_filter(row_filter, reason)
    return MetricViewFilter(row_filter.strip())


def _refuse_filter(row_filter: Any, reason: str) -> NoReturn:
    raise UnsupportedFilterError(
        f'Cannot apply the metric view filter {row_filter!r} since '
        f'{reason}. The graph would otherwise read rows the view leaves '
        f'out. Create a regular view with this WHERE clause and point the '
        f"metric view's source at it"
    )


def resolve_join_keys(
    join: MetricViewJoin,
    child_columns: Collection[str],
    parent_columns: Collection[str],
) -> tuple[str, str, str]:
    r"""Resolves a join condition into ``(other_alias, other_key,
    child_key)``.

    ``other_alias`` is the alias of the non-child side of the join, either
    :attr:`MetricViewJoin.parent_alias` or :obj:`SOURCE_ALIAS`. Unqualified
    column references are resolved by their (unambiguous) membership in the
    child or parent table.

    Raises:
        UnsupportedJoinError: If the condition cannot be resolved.
    """
    child_names = {join.alias, '.'.join(join.path)}
    parent_names = {join.parent_alias}
    if len(join.path) > 1:
        parent_names.add('.'.join(join.path[:-1]))

    def _side(ref: JoinKeyRef) -> str | None:
        if ref.qualifier is not None:
            if ref.qualifier in child_names:
                return join.alias
            if ref.qualifier in parent_names:
                return join.parent_alias
            if ref.qualifier == SOURCE_ALIAS:
                return SOURCE_ALIAS
            return None
        in_child = ref.column in child_columns
        in_parent = ref.column in parent_columns
        if in_child and not in_parent:
            return join.alias
        if in_parent and not in_child:
            return join.parent_alias
        return None

    first, second = join.condition
    first_side, second_side = _side(first), _side(second)

    if first_side == join.alias and second_side not in (None, join.alias):
        assert second_side is not None
        return second_side, second.column, first.column
    if second_side == join.alias and first_side not in (None, join.alias):
        assert first_side is not None
        return first_side, first.column, second.column

    raise UnsupportedJoinError(
        f"Failed to add join '{join.alias}' since its join condition could "
        f"not be resolved into a key reference between '{join.parent_alias}' "
        f"and '{join.alias}'"
    )


@dataclass(frozen=True)
class MetricViewDescription:
    r"""What ``DESCRIBE TABLE EXTENDED`` says about a metric view.

    Args:
        definition: The YAML definition.
        dtypes: The data types of all dimensions, by name.
        parameters: The data types of all parameters, by name. Their
            defaults are in the definition only.
        catalog: The catalog of the metric view, if given.
        schema: The schema of the metric view, if given.
    """

    definition: str
    dtypes: dict[str, str]
    parameters: dict[str, str]
    catalog: str | None
    schema: str | None


# The section of `DESCRIBE TABLE EXTENDED` that comes before any header.
_COLUMNS_SECTION = ''


def _describe_sections(rows: Sequence[Any]) -> dict[str, list[tuple[str, str]]]:
    r"""Splits the rows of ``DESCRIBE TABLE EXTENDED`` into their sections,
    keyed by the lower-case header without its ``#``.

    The columns come first, then sections such as ``# Parameters`` and
    ``# Detailed Table Information``. A header names no type, which tells it
    from a column named ``#...``. Row width differs between a SQL warehouse
    (four) and a Spark session (three), and so does whether an empty cell is
    ``''`` or ``None``, so only the first two cells are read.
    """
    sections: dict[str, list[tuple[str, str]]] = {_COLUMNS_SECTION: []}
    current: str | None = _COLUMNS_SECTION
    for row in rows:
        name = str(row[0]).strip() if row[0] is not None else ''
        value = str(row[1]) if len(row) > 1 and row[1] is not None else ''
        if name.startswith('#') and not value.strip():
            header = name.lstrip('#').strip().lower()
            # Spark heads the rows of some sections with `# col_name`.
            if header != 'col_name':
                current = header
                sections.setdefault(current, [])
            continue
        if not name:
            # A blank row ends the columns; rows up to the next header belong
            # to no section.
            if current == _COLUMNS_SECTION:
                current = None
            continue
        if current is not None:
            sections[current].append((name, value))
    return sections


def read_metric_view_definition(
    connection: AnyConnection,
    quoted_name: str,
) -> MetricViewDescription:
    r"""Reads a metric view definition from a Databricks SQL warehouse or
    Spark session.

    Each section of the description is read on its own, so a parameter named
    like a detail, such as ``Type`` or ``View Text``, cannot stand in for it.

    Raises:
        ValueError: If the relation is not a metric view or exposes no
            definition.
    """
    with connection.cursor() as cursor:
        cursor.execute(f'DESCRIBE TABLE EXTENDED {quoted_name}')
        rows = cursor.fetchall()

    sections = _describe_sections(rows)
    dtypes = {
        name: data_type
        for name, data_type in sections[_COLUMNS_SECTION]
        if not data_type.endswith(_MEASURE_SUFFIX)
    }
    parameters = dict(sections.get('parameters', []))
    details: dict[str, str] = {}
    for name, value in sections.get('detailed table information', []):
        details.setdefault(name, value)

    if details.get('Type') != 'METRIC_VIEW':
        raise ValueError(
            f"'{quoted_name}' is not a metric view "
            f"(got type '{details.get('Type', 'UNKNOWN')}')"
        )

    definition = details.get('View Text')
    if not definition:
        raise ValueError(
            f"Could not read the definition of metric view '{quoted_name}'"
        )

    return MetricViewDescription(
        definition=definition,
        dtypes=dtypes,
        parameters=parameters,
        catalog=details.get('Catalog'),
        schema=details.get('Database'),
    )
