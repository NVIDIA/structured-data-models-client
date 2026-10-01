# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

import io
import logging
import re
from collections.abc import Iterator
from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from predictive_query.query import ValidatedPredictiveQuery
from predictive_query.schema.typing import ProblemType
from sdm_relational_engine.rfm import Graph, KumoRelational
from sdm_relational_engine.rfm.query.parser import parse_query_locally


@dataclass(frozen=True)
class ExpectedQuery:
    canonical: str
    entity_ids: list[int | str]
    target_ast: str
    entity_ast: str = 'USERS.USER_ID'
    whatif_ast: str | None = None
    problem_type: ProblemType | None = None
    num_forecasts: int = 1
    top_k: int | None = None


def test_kumo_rfm_parse_query_accepts_validated_query(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)

    assert model._parse_query(ltv) is ltv


def test_kumo_rfm_parse_query_delegates_strings_to_local_parser(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    query = 'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) FOR USERS.USER_ID=0'

    with patch(
        'sdm_relational_engine.rfm.rfm.parse_query_locally',
        return_value=ltv,
    ) as mock_parse:
        assert model._parse_query(query) is ltv

    mock_parse.assert_called_once_with(query, model._graph_def)


def test_parse_query_locally_returns_validated_query(
    user_store_graph: Graph,
) -> None:
    query = parse_query_locally(
        'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) FOR USERS.USER_ID=0',
        user_store_graph._to_api_graph_definition(),
    )

    assert isinstance(query, ValidatedPredictiveQuery)
    assert query.to_string() == (
        'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) FOR USERS.USER_ID = 0'
    )
    assert query.get_rfm_entity_id_list() == [0]


def test_parse_query_names_allowed_aggregation_time_units() -> None:
    from predictive_query import RELATIONAL
    from predictive_query.parser.parser import PQLParser

    query = 'PREDICT COUNT(ORDERS.*, 0, 30, fortnights) FOR EACH USERS.USER_ID'

    with pytest.raises(
        ValueError,
        match=(
            r"Line 1, col \d+; Invalid aggregation time unit 'fortnights'. "
            r"Expected one of: 'seconds', 'minutes', 'hours', 'days', 'weeks', or 'months'\."
        ),
    ):
        PQLParser(mode=RELATIONAL.mode).to_parsed_predictive_query(query)


def test_demo_prefix_offsets_assuming_location(
    user_store_graph: Graph,
) -> None:
    from predictive_query import RELATIONAL
    from predictive_query.parser.parser import PQLParser
    from predictive_query.validator.model_validator import ModelValidator

    prefix = 'EXPLAIN '
    parsed = PQLParser(mode=RELATIONAL.mode).to_parsed_predictive_query(
        prefix + 'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) '
        'FOR USERS.USER_ID = 0 '
        'ASSUMING SUM(ORDERS.AMOUNT, 0, 7, days) >= 10'
    )
    assert parsed.whatif_ast is not None
    original_start_col = parsed.whatif_ast.location.start_col

    ModelValidator(
        user_store_graph._to_api_graph_definition(),
        mode=RELATIONAL.mode,
        can=RELATIONAL.capabilities,
    ).update_location_interval(parsed)

    assert parsed.whatif_ast.location.start_col == (
        original_start_col + len(prefix)
    )


@pytest.mark.parametrize(
    ('query', 'expected'),
    [
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) FOR USERS.USER_ID=0',
            ExpectedQuery(
                canonical=(
                    'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) '
                    'FOR USERS.USER_ID = 0'
                ),
                entity_ids=[0],
                target_ast='SUM(ORDERS.AMOUNT, 0, 7, days)',
            ),
            id='single-numeric-entity-id',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 0, 7) FOR USERS.USER_ID IN (0, 1, 3)',
            ExpectedQuery(
                canonical=(
                    'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) '
                    'FOR USERS.USER_ID IN (0, 1, 3)'
                ),
                entity_ids=[0, 1, 3],
                target_ast='SUM(ORDERS.AMOUNT, 0, 7, days)',
            ),
            id='default-time-unit',
        ),
        pytest.param(
            'predict sum(ORDERS.AMOUNT, 0, 7, days) # target\n'
            'for USERS.USER_ID in (0, 1, 3)',
            ExpectedQuery(
                canonical=(
                    'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) '
                    'FOR USERS.USER_ID IN (0, 1, 3)'
                ),
                entity_ids=[0, 1, 3],
                target_ast='SUM(ORDERS.AMOUNT, 0, 7, days)',
            ),
            id='comments-and-keyword-case',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.*, 0, 7, days) '
            'FOR USERS.USER_ID IN (0, 1, 3)',
            ExpectedQuery(
                canonical=(
                    'PREDICT COUNT(ORDERS.*, 0, 7, days) '
                    'FOR USERS.USER_ID IN (0, 1, 3)'
                ),
                entity_ids=[0, 1, 3],
                target_ast='COUNT(ORDERS.*, 0, 7, days)',
            ),
            id='count-wildcard',
        ),
        pytest.param(
            'PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 7, days) '
            'RANK TOP 2 FOR USERS.USER_ID IN (0, 1, 3)',
            ExpectedQuery(
                canonical=(
                    'PREDICT LIST_DISTINCT(ORDERS.STORE_ID, '
                    '0, 7, days) RANK TOP 2 FOR USERS.USER_ID '
                    'IN (0, 1, 3)'
                ),
                entity_ids=[0, 1, 3],
                target_ast='LIST_DISTINCT(ORDERS.STORE_ID, 0, 7, days)',
                problem_type=ProblemType.RANK,
                top_k=2,
            ),
            id='rank-link-prediction-top-k',
        ),
        pytest.param(
            'PREDICT AVG(ORDERS.AMOUNT, 0, 7, days) FOR USERS.USER_ID = 0',
            ExpectedQuery(
                canonical=(
                    'PREDICT AVG(ORDERS.AMOUNT, 0, 7, days) '
                    'FOR USERS.USER_ID = 0'
                ),
                entity_ids=[0],
                target_ast='AVG(ORDERS.AMOUNT, 0, 7, days)',
            ),
            id='avg-aggregation',
        ),
        pytest.param(
            'PREDICT MIN(ORDERS.AMOUNT, 0, 7, days) FOR USERS.USER_ID = 0',
            ExpectedQuery(
                canonical=(
                    'PREDICT MIN(ORDERS.AMOUNT, 0, 7, days) '
                    'FOR USERS.USER_ID = 0'
                ),
                entity_ids=[0],
                target_ast='MIN(ORDERS.AMOUNT, 0, 7, days)',
            ),
            id='min-aggregation',
        ),
        pytest.param(
            'PREDICT MAX(ORDERS.AMOUNT, 0, 7, days) FOR USERS.USER_ID = 0',
            ExpectedQuery(
                canonical=(
                    'PREDICT MAX(ORDERS.AMOUNT, 0, 7, days) '
                    'FOR USERS.USER_ID = 0'
                ),
                entity_ids=[0],
                target_ast='MAX(ORDERS.AMOUNT, 0, 7, days)',
            ),
            id='max-aggregation',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.* WHERE ORDERS.AMOUNT > 10.5, '
            '0, 7, days) FOR USERS.USER_ID IN (0, 1, 3)',
            ExpectedQuery(
                canonical=(
                    'PREDICT COUNT(ORDERS.* WHERE ORDERS.AMOUNT > '
                    '10.5, 0, 7, days) FOR USERS.USER_ID IN '
                    '(0, 1, 3)'
                ),
                entity_ids=[0, 1, 3],
                target_ast=(
                    'COUNT(ORDERS.* WHERE ORDERS.AMOUNT > 10.5, 0, 7, days)'
                ),
            ),
            id='target-filter-decimal',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.* WHERE ORDERS.AMOUNT IN (10, 15), '
            '0, 7, days) FOR USERS.USER_ID IN (0, 1, 3)',
            ExpectedQuery(
                canonical=(
                    'PREDICT COUNT(ORDERS.* WHERE ORDERS.AMOUNT IN '
                    '(10, 15), 0, 7, days) FOR USERS.USER_ID IN '
                    '(0, 1, 3)'
                ),
                entity_ids=[0, 1, 3],
                target_ast=(
                    'COUNT(ORDERS.* WHERE ORDERS.AMOUNT IN '
                    '(10, 15), 0, 7, days)'
                ),
            ),
            id='target-filter-membership',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.* WHERE NOT ORDERS.AMOUNT IS NULL, '
            '0, 7, days) FOR USERS.USER_ID IN (0, 1, 3)',
            ExpectedQuery(
                canonical=(
                    'PREDICT COUNT(ORDERS.* WHERE NOT '
                    '(ORDERS.AMOUNT IS NULL), 0, 7, days) '
                    'FOR USERS.USER_ID IN (0, 1, 3)'
                ),
                entity_ids=[0, 1, 3],
                target_ast=(
                    'COUNT(ORDERS.* WHERE NOT '
                    '(ORDERS.AMOUNT IS NULL), 0, 7, days)'
                ),
            ),
            id='target-filter-null-negation',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.* WHERE ORDERS.AMOUNT >= 10 '
            'AND ORDERS.AMOUNT <= 25, 0, 7, days) '
            'FOR USERS.USER_ID IN (0, 1, 3)',
            ExpectedQuery(
                canonical=(
                    'PREDICT COUNT(ORDERS.* WHERE '
                    '(ORDERS.AMOUNT >= 10) AND '
                    '(ORDERS.AMOUNT <= 25), 0, 7, days) '
                    'FOR USERS.USER_ID IN (0, 1, 3)'
                ),
                entity_ids=[0, 1, 3],
                target_ast=(
                    'COUNT(ORDERS.* WHERE (ORDERS.AMOUNT >= 10) '
                    'AND (ORDERS.AMOUNT <= 25), 0, 7, days)'
                ),
            ),
            id='target-filter-logical-and',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) > 25 '
            'FOR USERS.USER_ID IN (0, 1, 3)',
            ExpectedQuery(
                canonical=(
                    'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) > 25 '
                    'FOR USERS.USER_ID IN (0, 1, 3)'
                ),
                entity_ids=[0, 1, 3],
                target_ast='SUM(ORDERS.AMOUNT, 0, 7, days) > 25',
            ),
            id='target-threshold-condition',
        ),
        pytest.param(
            'PREDICT USERS.AGE FOR USERS.USER_ID IN (2, 3)',
            ExpectedQuery(
                canonical='PREDICT USERS.AGE FOR USERS.USER_ID IN (2, 3)',
                entity_ids=[2, 3],
                target_ast='USERS.AGE',
            ),
            id='static-numeric-target',
        ),
        pytest.param(
            'PREDICT USERS.STATUS = "A" FOR USERS.USER_ID IN (0, 1, 3)',
            ExpectedQuery(
                canonical=(
                    'PREDICT USERS.STATUS = "A" FOR USERS.USER_ID IN (0, 1, 3)'
                ),
                entity_ids=[0, 1, 3],
                target_ast='USERS.STATUS = "A"',
            ),
            id='static-categorical-condition',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.*, 0, 7, days) '
            'FOR USERS.USER_ID IN (0, 1, 3) '
            'WHERE USERS.STATUS = "A" OR USERS.AGE >= 30',
            ExpectedQuery(
                canonical=(
                    'PREDICT COUNT(ORDERS.*, 0, 7, days) '
                    'FOR USERS.USER_ID IN (0, 1, 3) WHERE '
                    '(USERS.STATUS = "A") OR (USERS.AGE >= 30)'
                ),
                entity_ids=[0, 1, 3],
                target_ast='COUNT(ORDERS.*, 0, 7, days)',
                entity_ast=(
                    'USERS.USER_ID WHERE (USERS.STATUS = "A") '
                    'OR (USERS.AGE >= 30)'
                ),
            ),
            id='entity-filter-static-or',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.*, 0, 7, days) '
            'FOR USERS.USER_ID IN (0, 1, 3) '
            'WHERE COUNT(ORDERS.*, -7, 0, days) > 0',
            ExpectedQuery(
                canonical=(
                    'PREDICT COUNT(ORDERS.*, 0, 7, days) '
                    'FOR USERS.USER_ID IN (0, 1, 3) WHERE '
                    'COUNT(ORDERS.*, -7, 0, days) > 0'
                ),
                entity_ids=[0, 1, 3],
                target_ast='COUNT(ORDERS.*, 0, 7, days)',
                entity_ast=(
                    'USERS.USER_ID WHERE COUNT(ORDERS.*, -7, 0, days) > 0'
                ),
            ),
            id='entity-filter-temporal-aggregation',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) '
            'FOR USERS.USER_ID IN (0, 1, 3) '
            'ASSUMING SUM(ORDERS.AMOUNT, 0, 7, days) >= 10',
            ExpectedQuery(
                canonical=(
                    'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) '
                    'FOR USERS.USER_ID IN (0, 1, 3) ASSUMING '
                    'SUM(ORDERS.AMOUNT, 0, 7, days) >= 10'
                ),
                entity_ids=[0, 1, 3],
                target_ast='SUM(ORDERS.AMOUNT, 0, 7, days)',
                whatif_ast='SUM(ORDERS.AMOUNT, 0, 7, days) >= 10',
            ),
            id='assuming-whatif-condition',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 0, 1, days) '
            'FORECAST 4 TIMEFRAMES FOR USERS.USER_ID = 0',
            ExpectedQuery(
                canonical=(
                    'PREDICT SUM(ORDERS.AMOUNT, 0, 1, days) '
                    'FORECAST 4 TIMEFRAMES FOR USERS.USER_ID = 0'
                ),
                entity_ids=[0],
                target_ast='SUM(ORDERS.AMOUNT, 0, 1, days)',
                problem_type=ProblemType.FORECAST,
                num_forecasts=4,
            ),
            id='forecast',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) FOR EACH USERS.USER_ID',
            ExpectedQuery(
                canonical=(
                    'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) '
                    'FOR EACH USERS.USER_ID'
                ),
                entity_ids=[],
                target_ast='SUM(ORDERS.AMOUNT, 0, 7, days)',
            ),
            id='for-each-entity',
        ),
    ],
)
def test_parse_valid_queries_against_rfm_graph(
    user_store_graph: Graph,
    query: str,
    expected: ExpectedQuery,
) -> None:
    parsed = KumoRelational(user_store_graph, verbose=False)._parse_query(query)

    assert isinstance(parsed, ValidatedPredictiveQuery)
    assert parsed.to_string() == expected.canonical
    assert parsed.get_rfm_entity_id_list() == expected.entity_ids
    assert str(parsed.target_ast) == expected.target_ast
    assert str(parsed.entity_ast) == expected.entity_ast
    assert parsed.problem_type == expected.problem_type
    assert parsed.num_forecasts == expected.num_forecasts
    assert parsed.top_k == expected.top_k
    if expected.whatif_ast is None:
        assert parsed.whatif_ast is None
    else:
        assert str(parsed.whatif_ast) == expected.whatif_ast


@pytest.mark.parametrize(
    ('query', 'expected_ids', 'expected_target'),
    [
        pytest.param(
            'PREDICT COUNT(ORDERS.*, 0, 365, days) '
            'FOR USERS.USER_ID = "user_a"',
            ['user_a'],
            'COUNT(ORDERS.*, 0, 365, days)',
            id='single-double-quoted-string-id',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.*, 0, 365, days) '
            'FOR USERS.USER_ID IN ("user_a", "user_c")',
            ['user_a', 'user_c'],
            'COUNT(ORDERS.*, 0, 365, days)',
            id='double-quoted-string-id-list',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.*, 0, 365, days) '
            "FOR USERS.USER_ID IN ('user_a', 'user_c')",
            ['user_a', 'user_c'],
            'COUNT(ORDERS.*, 0, 365, days)',
            id='single-quoted-string-id-list',
        ),
        pytest.param(
            'PREDICT USERS.NAME = "Alice" '
            'FOR USERS.USER_ID IN ("user_a", "user_c")',
            ['user_a', 'user_c'],
            'USERS.NAME = "Alice"',
            id='string-condition-target',
        ),
    ],
)
def test_parse_string_entity_ids(
    string_user_graph: Graph,
    query: str,
    expected_ids: list[str],
    expected_target: str,
) -> None:
    parsed = KumoRelational(string_user_graph, verbose=False)._parse_query(
        query
    )

    assert parsed.get_rfm_entity_id_list() == expected_ids
    assert str(parsed.target_ast) == expected_target


@pytest.mark.parametrize(
    ('query', 'expected_message'),
    [
        pytest.param(
            'SUM(ORDERS.AMOUNT, 0, 7, days) FOR USERS.USER_ID = 0',
            'Query does not start with the PREDICT keyword',
            id='missing-predict',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.MISSING, 0, 7, days) FOR USERS.USER_ID = 0',
            "Column 'MISSING' does not exist in table 'ORDERS'",
            id='unknown-column',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) FOR USERSS.USER_ID = 0',
            "Table 'USERSS' does not exist in the schema",
            id='unknown-table',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) FOR USERS.AGE = 20',
            "Entity 'USERS.AGE' is not a primary key",
            id='entity-not-primary-key',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.*, 0, 7, days) FOR USERS.USER_ID = 0',
            'Wildcard column ORDERS.* appeared in an invalid position',
            id='wildcard-outside-count',
        ),
        pytest.param(
            'PREDICT ORDERS.AMOUNT FOR USERS.USER_ID = 0',
            "Couldn't pick a unique ORDERS.AMOUNT for each USERS",
            id='many-to-one-target-without-aggregation',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 7, 0, days) FOR USERS.USER_ID = 0',
            "start date offset '7' needs to be smaller",
            id='invalid-time-range',
        ),
        pytest.param(
            'PREDICT FIRST(ORDERS.AMOUNT, 0, 7, days) FOR USERS.USER_ID = 0',
            'Aggregation FIRST type is not supported',
            id='unsupported-aggregation',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.* WHERE STORES.CAT = "pizza", '
            '0, 7, days) FOR USERS.USER_ID IN (0, 1, 3)',
            'Unsupported implicit join',
            id='unsupported-implicit-join',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.*, 0, 7, days) RANK TOP 2 '
            'FOR USERS.USER_ID = 0',
            "Problem type 'RANK' is only supported",
            id='unsupported-rank-problem-type',
        ),
        pytest.param(
            'PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 7, days) '
            'TOP 100 FOR USERS.USER_ID IN (0, 1, 3)',
            'TOP K requires RANK',
            id='top-k-without-rank',
        ),
        pytest.param(
            'PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 7, days) '
            'RANK TOP 0 FOR USERS.USER_ID IN (0, 1, 3)',
            'TOP K value must be at least 1',
            id='top-k-zero',
        ),
        pytest.param(
            'PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 7, days) '
            'RANK TOP -1 FOR USERS.USER_ID IN (0, 1, 3)',
            'TOP K value must be at least 1',
            id='top-k-negative',
        ),
        pytest.param(
            'PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 7, days) '
            'RANK TOP 21 FOR USERS.USER_ID IN (0, 1, 3)',
            'Top k 21 exceeds maximum',
            id='top-k-exceeds-rfm-limit',
        ),
        pytest.param(
            'PREDICT USERS.AGE FORECAST 4 TIMEFRAMES FOR USERS.USER_ID = 0',
            'FORECAST requires a temporal target',
            id='forecast-static-target',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) FOR USERS.USER_ID IN ("0")',
            'Array with single element is not supported',
            id='single-element-in-list',
        ),
    ],
)
def test_parse_invalid_queries_report_useful_errors(
    user_store_graph: Graph,
    query: str,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        KumoRelational(user_store_graph, verbose=False)._parse_query(query)

    message = str(exc_info.value)
    assert query in message
    assert expected_message in message


class _RecordCollector(logging.Handler):
    r"""Collects records straight off the ``sdm_relational_engine`` logger.

    Not ``caplog``: that handler lives on the root logger, and the package sets
    ``propagate = False``, so whether it observes anything at all depends on
    the environment. It also collects records from every other library, which
    would make "no warnings were emitted" assertions fail on unrelated output.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def _capture_sdm_relational_engine_logs() -> Iterator[_RecordCollector]:
    # The parser is its own distribution now, so a record it emits arrives
    # under its own logger rather than the engine's. Both are captured, because
    # what this asserts is that no raw query reaches a WARNING anywhere.
    loggers = [
        logging.getLogger('sdm_relational_engine'),
        logging.getLogger('predictive_query'),
    ]
    collector = _RecordCollector()
    previous = [logger.level for logger in loggers]
    for logger in loggers:
        logger.addHandler(collector)
        logger.setLevel(logging.DEBUG)
    try:
        yield collector
    finally:
        for logger, level in zip(loggers, previous, strict=True):
            logger.removeHandler(collector)
            logger.setLevel(level)


@contextmanager
def _restored_logging() -> Iterator[None]:
    r"""Restore every logger ``initialize_logging`` touches."""
    names = ['sdm_relational_engine', 'matplotlib', 'urllib3', 'snowflake']
    loggers = [logging.getLogger(name) for name in names]
    saved = [
        (logger, list(logger.handlers), logger.propagate, logger.level)
        for logger in loggers
    ]
    root = logging.getLogger()
    saved_root = (list(root.handlers), root.propagate, root.level)
    try:
        yield
    finally:
        for logger, handlers, propagate, level in saved:
            logger.handlers[:] = handlers
            logger.propagate = propagate
            logger.setLevel(level)
        root.handlers[:], root.propagate, root.level = saved_root


def test_parse_failure_does_not_log_the_raw_query_at_warning(
    user_store_graph: Graph,
) -> None:
    # PQL carries literal filter values, and WARNING-and-above records are
    # typically shipped to a central log store; the raised ValueError already
    # quotes the query, so the log record must not repeat it.
    secret = 'ssn-123-45-6789'
    query = (
        f'PREDICT SUM(ORDERS.AMOUNT, 0, 30, dayz) FOR EACH USERS.USER_ID '
        f"WHERE USERS.STATUS = '{secret}'"
    )

    with _capture_sdm_relational_engine_logs() as captured:
        with pytest.raises(ValueError, match=secret):
            KumoRelational(user_store_graph, verbose=False)._parse_query(query)

    # Guard against the assertions below passing on an empty capture.
    assert captured.records, 'the parser must still log the translated error'
    assert [r for r in captured.records if r.levelno >= logging.WARNING] == []
    assert not any(secret in r.getMessage() for r in captured.records)
    assert any(
        'was translated to' in r.getMessage() and r.levelno == logging.DEBUG
        for r in captured.records
    )


def test_importing_and_using_the_driver_leaves_the_root_logger_alone() -> None:
    # A library must never call logging.basicConfig(): it attaches a handler to
    # the root logger and silently reconfigures the host application.
    import sdm_relational_engine.utils.logging

    with _restored_logging():
        root = logging.getLogger()
        before = list(root.handlers)
        sdm_relational_engine.utils.logging.initialize_logging()
        assert list(root.handlers) == before


def test_the_driver_handles_its_own_records_only_when_nothing_else_does() -> (
    None
):
    """Both halves of the rule, so neither depends on how pytest is invoked.

    A host that has already configured logging owns the output format, and a
    library installing its own handler there would make its records both
    unformattable and uncapturable. A host that has configured nothing gets the
    driver's own console output. Asserting only the second half passed or
    failed depending on whether the runner had installed a root handler yet.
    """
    import sdm_relational_engine.utils.logging

    logger = logging.getLogger('sdm_relational_engine')

    with _restored_logging():
        logging.getLogger().handlers[:] = []
        logger.handlers[:] = []
        sdm_relational_engine.utils.logging.initialize_logging()
        assert logger.handlers, (
            'with nothing else configured, the driver handles its own records'
        )
        assert logger.propagate is False

    with _restored_logging():
        logging.getLogger().handlers[:] = [logging.NullHandler()]
        logger.handlers[:] = []
        sdm_relational_engine.utils.logging.initialize_logging()
        assert not logger.handlers, (
            'the host had configured logging, so it owns the output format'
        )


@pytest.mark.parametrize(
    'query',
    [
        'PREDICT USERS.STATUS FOR USERS.USER_ID IN (1)',
        'PREDICT USERS.STATUS FOR USERS.USER_ID IN (1,)',
        'SELECT * FROM USERS',
        'PREDICT COUNT(ORDERS.* 0, 7, days) FOR EACH USERS.USER_ID',
        'PREDICT ((( USERS.STATUS FOR EACH USERS.USER_ID',
    ],
)
def test_a_rejected_query_writes_nothing_to_stdout(
    user_store_graph: Graph,
    query: str,
) -> None:
    """Two grammar actions raise a bare ``RecognitionException``, which ANTLR's
    default error strategy does not recognise and announces with a ``print``
    before notifying the listeners. Removing the error *listeners*, as the
    parser does, leaves that line in place, so ``IN (1)``, an easy mistake,
    wrote ``unknown recognition error type: RecognitionException`` to stdout
    even under ``verbose=False``.
    """
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        with pytest.raises(ValueError):
            KumoRelational(user_store_graph, verbose=False)._parse_query(query)

    assert stdout.getvalue() == ''
    assert stderr.getvalue() == ''


def test_the_rejection_still_explains_itself(user_store_graph: Graph) -> None:
    r"""Silencing the strategy must not silence the diagnosis: the exception
    carries the whole message, including the suggestion.
    """
    with pytest.raises(ValueError) as excinfo:
        KumoRelational(user_store_graph, verbose=False)._parse_query(
            'PREDICT USERS.STATUS FOR USERS.USER_ID IN (1)'
        )

    message = str(excinfo.value)
    assert 'Array with single element is not supported' in message
    assert "'IS IN (const)'" in message


def test_a_valid_query_still_parses(user_store_graph: Graph) -> None:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        parsed = KumoRelational(user_store_graph, verbose=False)._parse_query(
            'PREDICT COUNT(ORDERS.*, 0, 30, days) FOR EACH USERS.USER_ID'
        )
    assert parsed is not None
    assert stdout.getvalue() == ''


@pytest.mark.parametrize('unit', ['day', 'year'])
def test_an_unsupported_time_unit_is_named(
    unit: str, user_store_graph: Graph
) -> None:
    r"""The rejected unit is named, not the clause that happens to contain it."""
    with pytest.raises(ValueError) as excinfo:
        KumoRelational(user_store_graph, verbose=False)._parse_query(
            f'PREDICT COUNT(ORDERS.*, 0, 4, {unit}) FOR EACH USERS.USER_ID'
        )

    message = str(excinfo.value)
    assert f"Invalid aggregation time unit '{unit}'." in message
    assert (
        "Expected one of: 'seconds', 'minutes', 'hours', 'days', "
        "'weeks', or 'months'." in message
    )
    assert re.search(r'Line 1, col \d+;', message)
    assert 'target (PREDICT) clause' not in message
