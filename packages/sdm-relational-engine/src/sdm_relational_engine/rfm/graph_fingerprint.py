# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A name for the shape of a graph, stable enough to key a cache on.

A caller that reuses a built graph needs to know when the one it holds is still
the graph the question would produce. ``hash()`` cannot answer that: Python salts
string hashing per process, so the same graph names itself differently in two
workers, and the built-in covers only table names and edges, so a column that
changed type, a key that moved and a time column that appeared all leave the
name untouched.

This walks the whole shape a prediction depends on, writes it in a fixed order,
and hashes the bytes. Two graphs that would behave identically get one name;
anything a prediction can see changes it.

What is *not* part of it is the engine's package version. Versioning on that
would throw away every downstream cache on a release that had nothing to do
with graphs, so a name says the shape is the same, not that the engine reading
it is. `FINGERPRINT_VERSION` below is the version of this encoding, and it is
the only version here: bump it when the bytes written change, and every name
changes with it.

Format
------
Every field is written as its length in characters, a colon, then its text, and
the fields are concatenated. Nothing separates them, because the length says
where each one ends, so no value can be mistaken for a delimiter.

The order is: the format version, then each table sorted by name, then the
edges sorted, then the edges inference guessed. A table is its name, its column
count, then each column in the order the table holds it as name, semantic type
and storage type; then its declared key columns in order preceded by their
count; then its primary key, time column and end time column.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

# The encoding below, bumped only when THAT changes. Versioning on the package
# version instead would throw away every downstream cache on a release that had
# nothing to do with fingerprinting.
FINGERPRINT_VERSION = 1


class _Column(Protocol):
    r"""What this reads off a column. Named so a rename fails a type check.

    Read through getattr the fields would go missing silently, and the
    fingerprint would quietly stop telling apart graphs that differ.
    """

    name: str
    stype: Any
    dtype: Any


class _Table(Protocol):
    r"""What this reads off a table."""

    name: str
    columns: Any
    primary_key: Any
    primary_key_columns: Any
    time_column: Any
    end_time_column: Any


class _Graph(Protocol):
    r"""What this reads off a graph.

    A protocol rather than the concrete class, for the reason the two above
    are: a rename fails a type check here instead of going quiet. It also
    keeps the annotation resolvable at runtime, which an import guarded by
    ``TYPE_CHECKING`` is not.
    """

    tables: Any
    edges: Any
    inferred_edges: Any


def _term(value: object) -> str:
    r"""One field, written so no value can be mistaken for a delimiter.

    Each is its length then its text, so a name containing whatever character
    was chosen as a separator cannot split into two fields, and two different
    shapes cannot write the same bytes.
    """
    text = '' if value is None else str(value)
    return f'{len(text)}:{text}'


def _column_terms(column: _Column) -> list[str]:
    r"""A column, including the expression that defines it.

    A logical column is projected into the sampler's SQL as
    ``<expr> AS <name>``, so two graphs whose columns share a name and type but
    compute it differently read different data. `Column` already counts the
    expression as part of its identity; a fingerprint that dropped it would
    call those two graphs the same.
    """
    return [
        _term(column.name),
        _term(column.stype),
        _term(column.dtype),
        _term(getattr(column, 'expr', None)),
    ]


def _table_terms(table: _Table) -> list[str]:
    r"""Everything about a table that a prediction can see.

    Columns are named in their own order rather than sorted, because that order
    is the one the sampler reads them in.

    The source name is carried as well as the name. A warehouse-backed table
    can be called `orders` while reading `orders_2025`, and a graph reading
    `orders_2026` under the same name is a different graph. For a table with no
    separate source the two are equal, which costs a repeated term and keeps
    the rule one line.
    """
    terms = [
        _term(table.name),
        _term(getattr(table, 'source_name', None)),
        _term(len(table.columns)),
    ]
    for column in table.columns:
        terms.extend(_column_terms(column))
    key = table.primary_key_columns or ()
    terms.append(_term(len(key)))
    terms.extend(_term(c) for c in key)
    for attribute in ('time_column', 'end_time_column'):
        column = getattr(table, attribute, None)
        terms.append(_term(column.name if column is not None else None))
    return terms


def graph_fingerprint(graph: _Graph) -> str:
    r"""A hex name for *graph*, equal across processes for equal shapes.

    Tables are visited in sorted order so that two graphs built from the same
    source in a different order share a name. Edges are sorted for the same
    reason: they are a set, and the order they were added in says nothing about
    what the graph does.
    """
    names = sorted(graph.tables)
    terms: list[str] = [
        _term('sdm-relational-engine'),
        _term(FINGERPRINT_VERSION),
        _term(len(names)),
    ]

    for name in names:
        terms.extend(_table_terms(graph.tables[name]))

    edges = sorted(
        graph.edges, key=lambda e: (e.src_table, e.fkey, e.dst_table)
    )
    terms.append(_term(len(edges)))
    for edge in edges:
        terms.extend(
            [_term(edge.src_table), _term(edge.fkey), _term(edge.dst_table)]
        )

    inferred = tuple(sorted(graph.inferred_edges))
    terms.append(_term(len(inferred)))
    for src, fkey, dst in inferred:
        terms.extend([_term(src), _term(fkey), _term(dst)])

    return hashlib.sha256(''.join(terms).encode('utf-8')).hexdigest()
