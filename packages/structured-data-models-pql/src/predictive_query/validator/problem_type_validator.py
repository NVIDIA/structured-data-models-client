# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from predictive_query.ast import Column
from predictive_query.profiles import Mode
from predictive_query.query import ParsedPredictiveQuery
from predictive_query.schema import codes
from predictive_query.schema.common import (
    Span,
    ValidationError,
    ValidationResponse,
)
from predictive_query.schema.typing import ProblemType, Stype, symbol

VALID_PROBLEM_TYPES = ['CLASSIFY', 'RANK', 'FORECAST', 'ANOMALY']
MAX_TIMEFRAMES = 10_000


class ProblemTypeValidator:
    r"""All validation related to `problem_type` field
    in `PredictiveQuery` object.
    """

    def validate(
        self,
        query: ParsedPredictiveQuery,
        mode: Mode,
    ) -> ValidationResponse:
        r"""Validates the `problem_type` field in `PredictiveQuery` object.
        Currently the only supported problem types are `CLASSIFY` and `RANK`.
        And the target should have stype `multicategorical`.

        Args:
            query: `ParsedPredictiveQuery` with stypes inferred.
            mode: Whether the query defines a task or asks a question.

        Note:
            This validator requires the stypes to be resolved before being
            called as it uses the stypes to validate the problem type.

        """
        problem_type = query.problem_type
        validation_response = ValidationResponse()
        target_ast = query.target_ast

        if problem_type == ProblemType.ANOMALY:
            return self.validate_anomaly(query)

        if query.top_k is not None:
            if query.top_k < 1:
                return ValidationResponse(
                    errors=[
                        ValidationError(
                            'Invalid TOP K value',
                            f'TOP K value must be at least 1, got {query.top_k}.',
                            code=codes.TASK_TOP_K_NOT_POSITIVE,
                        )
                    ]
                )
            if problem_type != ProblemType.RANK:
                return ValidationResponse(
                    errors=[
                        ValidationError(
                            'TOP K requires RANK.',
                            'TOP K can only be specified with the RANK problem '
                            "type. Please rewrite the query as 'RANK TOP k'.",
                            code=codes.TASK_TOP_K_REQUIRES_RANK,
                        )
                    ]
                )

        if (
            target_ast.stype == Stype.multicategorical
            and problem_type is None
            and mode is Mode.TRAINING
        ):
            return ValidationResponse(
                errors=[
                    ValidationError(
                        'Problem type is required for multicategorical targets.',
                        f'{query.entity_ast.get_location().message_start}: '
                        f'Since your target({target_ast}) is multicategorical, '
                        f'you must specify a problem type (CLASSIFY or RANK). '
                        f'Please rewrite the query as: PREDICT {target_ast} '
                        f'CLASSIFY/RANK TOP k',
                        code=codes.TASK_PROBLEM_TYPE_REQUIRED,
                        span=Span.at(query.entity_ast.get_location()),
                    )
                ]
            )
        if problem_type is None:
            return validation_response
        if problem_type == ProblemType.FORECAST:
            return self.validate_forecast(query)

        if symbol(problem_type).upper() not in VALID_PROBLEM_TYPES:
            return ValidationResponse(
                errors=[
                    ValidationError(
                        f"Problem type '{problem_type}' is not supported. ",
                        f'Only {VALID_PROBLEM_TYPES} are supported problem types '
                        f'In PREDICT.',
                        code=codes.TASK_PROBLEM_TYPE_UNKNOWN,
                    )
                ]
            )

        assert target_ast.stype is not None
        if target_ast.stype != Stype.multicategorical:
            error_msg = (
                f'{query.target_ast.get_location().message_start}: '
                f"Target '{target_ast}' is not multicategorical. "
                f'Either drop the problem type {problem_type} OR change the '
                f"target to a multicategorical column, 'LAST'/'FIRST' "
                f'aggregation of a multicategorical column, '
                f"or a 'LIST_DISTINCT' aggregation."
            )
            return ValidationResponse(
                errors=[
                    ValidationError(
                        f"Problem type '{problem_type}' is only supported for "
                        f'multicategorical targets. ',
                        error_msg,
                        code=codes.TASK_PROBLEM_TYPE_NEEDS_MULTICATEGORICAL,
                    )
                ]
            )

        if problem_type == ProblemType.RANK and query.top_k is None:
            return ValidationResponse(
                errors=[
                    ValidationError(
                        f"Problem type '{symbol(problem_type)}' "
                        f'requires TOP K to be specified. ',
                        f'{query.entity_ast.get_location().message_start}: '
                        f"Please rewrite the query with 'TOP' as 'RANK TOP k'.",
                        code=codes.TASK_RANK_REQUIRES_TOP_K,
                        span=Span.at(query.entity_ast.get_location()),
                    )
                ]
            )
        return validation_response

    def validate_anomaly(
        self, parsed_query: ParsedPredictiveQuery
    ) -> ValidationResponse:
        r"""Validates the ``DETECT ANOMALY`` clause constraints.

        An anomaly score is produced by the model rather than computed from
        the data, so the clauses that shape a computed label have nothing to
        act on. The score is joint across the scored channels, so the target
        names the row rather than one column of it.

        The grammar already excludes RANK, CLASSIFY, TOP and FORECAST, which
        are alternatives of the same clause.

        Rules:
            1. No ASSUMING clause.
            2. No time range on the target.
            3. The target is a wildcard column.

        Args:
            parsed_query: the input parsed query.

        Returns:
            ValidationResponse: Errors and warnings encountered.
        """
        response = ValidationResponse()
        target_ast = parsed_query.target_ast

        carried = []
        if parsed_query.whatif_ast is not None:
            carried.append('ASSUMING')
        if target_ast.date_offset_range is not None:
            carried.append('a time range')
        if carried:
            response.errors.append(
                ValidationError(
                    title='Clause not supported for an anomaly score',
                    message=f'DETECT ANOMALY returns a score the model '
                    f'produces, so {", ".join(carried)} has nothing to act '
                    f'on. Remove it.',
                    code=codes.TASK_ANOMALY_INCOMPATIBLE_CLAUSE,
                )
            )

        is_wildcard = isinstance(
            target_ast, Column
        ) and target_ast.fqn.endswith('.*')
        if not is_wildcard:
            response.errors.append(
                ValidationError(
                    title='DETECT ANOMALY requires a wildcard target',
                    message='The score is joint across the scored channels, '
                    'so naming one column would promise a score specific to '
                    'it. Write the table wildcard, e.g. '
                    'PREDICT readings.* DETECT ANOMALY.',
                    code=codes.TASK_ANOMALY_REQUIRES_WILDCARD_TARGET,
                )
            )
        return response

    def validate_forecast(
        self, parsed_query: ParsedPredictiveQuery
    ) -> ValidationResponse:
        r"""Validates the FORECAST clause constraints.

        Rules:
            1. N must be >= 1 (N=1 is a single-horizon forecast).
            2. N must be <= MAX_TIMEFRAMES.
            3. Target must have a forward-looking time range.
            4. Target must be numerical.

        Args:
            parsed_query: the input parsed query.

        Returns:
            ValidationResponse: Errors and warnings encountered.
        """
        response = ValidationResponse()
        num_forecasts = parsed_query.num_forecasts
        if num_forecasts < 1:
            response.errors.append(
                ValidationError(
                    title='Invalid FORECAST value',
                    message=f'FORECAST value must be at least 1, '
                    f'got {num_forecasts}.',
                    code=codes.TASK_FORECAST_NOT_POSITIVE,
                )
            )
            return response

        if num_forecasts > MAX_TIMEFRAMES:
            response.errors.append(
                ValidationError(
                    title='FORECAST value too large',
                    message=f'FORECAST {num_forecasts} TIMEFRAMES exceeds '
                    f'the maximum of {MAX_TIMEFRAMES}.',
                    code=codes.TASK_FORECAST_TOO_LARGE,
                )
            )
            return response

        target_ast = parsed_query.target_ast
        has_time_range = target_ast.date_offset_range is not None
        is_bare_column = isinstance(target_ast, Column)
        if not has_time_range and not is_bare_column:
            response.errors.append(
                ValidationError(
                    title='FORECAST requires a column or a time range',
                    message='FORECAST takes either a column of the table, '
                    'whose own next values are forecast, or a target with a '
                    'forward-looking time range (e.g., '
                    'SUM(table.col, 0, 7, days)), whose aggregate is. A '
                    'derived target with neither cannot be forecast.',
                    code=codes.TASK_FORECAST_REQUIRES_TEMPORAL_TARGET,
                )
            )
            return response

        if target_ast.stype != Stype.numerical:
            response.errors.append(
                ValidationError(
                    title='FORECAST requires a numerical target',
                    message='FORECAST is only supported for numerical targets.',
                    code=codes.TASK_FORECAST_REQUIRES_NUMERICAL_TARGET,
                )
            )
            return response

        return response
