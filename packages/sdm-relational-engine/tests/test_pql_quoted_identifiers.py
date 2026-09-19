# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pandas as pd
import pytest
from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener
from predictive_query.grammar.PQLGrammarLexer import PQLGrammarLexer
from predictive_query.grammar.PQLGrammarParser import PQLGrammarParser
from predictive_query.names import (
    canonical_fqn,
    fqn,
    quote_name,
    split_fqn,
)
from predictive_query.schema.typing import Stype
from sdm_relational_engine.rfm.query.parser import parse_query_locally


class _CollectErrors(ErrorListener):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def syntaxError(self, recognizer, offending, line, column, msg, e) -> None:
        self.errors.append(msg)


def parse_errors(query: str) -> list[str]:
    r"""Parses *query* and returns the syntax errors, if any."""
    listener = _CollectErrors()
    lexer = PQLGrammarLexer(InputStream(query))
    lexer.removeErrorListeners()
    lexer.addErrorListener(listener)
    parser = PQLGrammarParser(CommonTokenStream(lexer))
    parser.removeErrorListeners()
    parser.addErrorListener(listener)
    parser.prog()
    return listener.errors


@pytest.fixture()
def spaced_graph():
    r"""A graph whose names a bare identifier cannot spell."""
    import sdm_relational_engine.rfm as rfm

    users = pd.DataFrame(
        {
            'Customer ID': [0, 1, 2, 3],
            'AGE': [20, 30, 40, 50],
        }
    )
    orders = pd.DataFrame(
        {
            'ORDER_ID': list(range(8)),
            'Customer ID': [0, 0, 1, 1, 2, 2, 3, 3],
            'Line Total': [10.0, 15.0, 20.0, 25.0, 30.0, 10.0, 15.0, 20.0],
            'TIME': [f'2025-01-0{i % 8 + 1}' for i in range(8)],
        }
    )
    with warnings.catch_warnings():
        # 'Customer ID' is not a name primary-key inference recognises, which
        # is the point of the fixture; it is declared right below.
        warnings.filterwarnings('ignore', message='No primary key')
        graph = rfm.Graph.from_data(
            {'My People': users, 'ORDERS': orders}, verbose=False
        )
    graph['My People'].primary_key = 'Customer ID'
    graph['ORDERS'].primary_key = 'ORDER_ID'
    graph['ORDERS'].time_column = 'TIME'
    graph.link(src_table='ORDERS', fkey='Customer ID', dst_table='My People')
    graph['ORDERS']['Line Total'].stype = Stype.numerical
    return graph


@pytest.mark.parametrize(
    'query',
    [
        'PREDICT COUNT(ORDERS.*, 0, 30, days) = 0 FOR EACH PEOPLE.CUSTOMER_ID',
        'PREDICT SUM(ORDERS.TOTAL, 0, 30, days) FOR EACH PEOPLE.CUSTOMER_ID',
    ],
)
def test_bare_identifiers_still_parse(query: str) -> None:
    assert parse_errors(query) == []


@pytest.mark.parametrize(
    'query',
    [
        # The column a warehouse spells with a space.
        'PREDICT COUNT(ORDERS.*, 0, 30, days) = 0 FOR EACH PEOPLE.`Customer ID`',
        # Both sides quoted.
        'PREDICT COUNT(`Order Lines`.*, 0, 30, days) = 0 '
        'FOR EACH `My People`.`Customer ID`',
        # Quoting is not limited to the entity: an aggregation target may need it.
        'PREDICT SUM(ORDERS.`Line Total`, 0, 30, days) FOR EACH PEOPLE.`Customer ID`',
        # A filter over a quoted column.
        "PREDICT COUNT(ORDERS.* WHERE ORDERS.`Order Status` = 'open', 0, 30, days) "
        '> 0 FOR EACH PEOPLE.`Customer ID`',
    ],
)
def test_quoted_identifiers_parse(query: str) -> None:
    assert parse_errors(query) == []


def test_unterminated_quote_is_rejected() -> None:
    query = (
        'PREDICT COUNT(ORDERS.*, 0, 30, days) = 0 FOR EACH PEOPLE.`Customer ID'
    )
    assert parse_errors(query) != []


@pytest.mark.parametrize(
    'full_name,expected',
    [
        ('PEOPLE.CUSTOMER_ID', ['PEOPLE', 'CUSTOMER_ID']),
        ('PEOPLE.`Customer ID`', ['PEOPLE', 'Customer ID']),
        ('`My Table`.`Customer ID`', ['My Table', 'Customer ID']),
        ('PEOPLE.*', ['PEOPLE', '*']),
        ('PEOPLE.`First.Last`', ['PEOPLE', 'First.Last']),
    ],
)
def test_split_fqn_keeps_quoted_dots_and_strips_quotes(
    full_name: str,
    expected: list[str],
) -> None:
    assert split_fqn(full_name) == expected


@pytest.mark.parametrize(
    'full_name,expected',
    [
        ('PEOPLE.CUSTOMER_ID', 'PEOPLE.CUSTOMER_ID'),
        ('PEOPLE.`Customer ID`', 'PEOPLE.Customer ID'),
        ('`My People`.`Customer ID`', 'My People.Customer ID'),
        ('ORDERS.*', 'ORDERS.*'),
    ],
)
def test_canonical_fqn_drops_the_quoting(full_name: str, expected: str) -> None:
    assert canonical_fqn(full_name) == expected


def test_canonical_fqn_rejects_a_dot_inside_a_quoted_name() -> None:
    # Rejoined with dots, such a name is indistinguishable from a
    # table/column pair at every consumer that splits on one.
    with pytest.raises(ValueError, match='contains a dot'):
        canonical_fqn('PEOPLE.`First.Last`')


@pytest.mark.parametrize(
    'name,expected',
    [
        ('CUSTOMER_ID', 'CUSTOMER_ID'),
        ('_private', '_private'),
        ('col2', 'col2'),
        ('*', '*'),
        ('Customer ID', '`Customer ID`'),
        ('2_leading_digit', '`2_leading_digit`'),
    ],
)
def test_quote_name_quotes_only_what_the_grammar_cannot_spell(
    name: str,
    expected: str,
) -> None:
    assert quote_name(name) == expected


@pytest.mark.parametrize('name', ['a`b', 'a\nb', 'a\rb'])
def test_quote_name_refuses_a_name_no_quoting_can_carry(name: str) -> None:
    # Wrapping these would render a query that cannot be parsed back.
    with pytest.raises(ValueError, match='cannot express'):
        quote_name(name)


def test_fqn_leaves_spellable_names_unquoted() -> None:
    # Keeps every query that parsed before quoting existed byte-identical.
    assert fqn('PEOPLE', 'CUSTOMER_ID') == 'PEOPLE.CUSTOMER_ID'
    assert fqn('PEOPLE', 'Customer ID') == 'PEOPLE.`Customer ID`'


def test_the_ast_holds_the_name_the_data_uses(spaced_graph) -> None:
    r"""The regression that made quoting useless.

    The parser sees ```Customer ID``` but every consumer -- the sampler, a
    dataframe lookup, the graph -- knows the column as ``Customer ID``. If the
    quotes reach the AST the query validates and then fails with a
    ``KeyError`` once it is executed.
    """
    query = (
        'PREDICT SUM(ORDERS.`Line Total`, 0, 30, days) '
        'FOR EACH `My People`.`Customer ID`'
    )
    validated = parse_query_locally(
        query, spaced_graph._to_api_graph_definition()
    )

    assert validated.entity_column == 'My People.Customer ID'
    for column in validated.all_query_columns:
        assert '`' not in column


def test_a_rendered_query_can_be_parsed_back(spaced_graph) -> None:
    r"""``to_string`` has to put the quoting back that the AST dropped."""
    query = (
        'PREDICT SUM(ORDERS.`Line Total`, 0, 30, days) '
        'FOR EACH `My People`.`Customer ID`'
    )
    graph_definition = spaced_graph._to_api_graph_definition()

    rendered = parse_query_locally(query, graph_definition).to_string()

    assert '`Customer ID`' in rendered
    assert parse_query_locally(rendered, graph_definition).to_string() == (
        rendered
    )


def test_quoting_a_spellable_name_is_accepted(spaced_graph) -> None:
    r"""Quoting every identifier is a habit a SQL user brings with them.

    ``ORDER_ID`` needs no quoting, so quoting it has to be a no-op rather than
    a new way to fail.
    """
    graph_definition = spaced_graph._to_api_graph_definition()
    bare = parse_query_locally(
        'PREDICT COUNT(ORDERS.ORDER_ID, 0, 30, days) '
        'FOR EACH `My People`.`Customer ID`',
        graph_definition,
    )
    quoted = parse_query_locally(
        'PREDICT COUNT(`ORDERS`.`ORDER_ID`, 0, 30, days) '
        'FOR EACH `My People`.`Customer ID`',
        graph_definition,
    )

    assert quoted.to_string() == bare.to_string()
