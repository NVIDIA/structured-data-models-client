# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from predictive_query.ast import (
    Aggregation,
    ASTNode,
    Column,
    Condition,
    Constant,
    Filter,
    Join,
)
from predictive_query.ast.location_interval import (
    ASTQueryLocationInterval,
)
from predictive_query.profiles import Capabilities, Mode
from predictive_query.query import ParsedPredictiveQuery
from predictive_query.schema import codes
from predictive_query.schema.common import (
    Span,
    ValidationError,
    ValidationResponse,
)
from predictive_query.schema.definition import ColumnKey, SchemaDefinition
from predictive_query.schema.typing import (
    AggregationType,
    Dtype,
    MemberOp,
    ProblemType,
    RelOp,
    StrOp,
    Stype,
    symbol,
)
from predictive_query.validator.utils import col_name, merge, table_name

FOR = 'FOR'
FOR_EACH = 'FOR EACH'
MAX_ENTITY_LIST_LENGTH = 1000
MAX_TOP_K = 20


class ModelValidator:
    r"""What a query must satisfy to be asked of a pretrained model.

    A query under :attr:`~predictive_query.profiles.Mode.TRAINING` defines a task
    to learn, and one under :attr:`~predictive_query.profiles.Mode.INFERENCE`
    asks a question of a model that already exists. The second is the more
    constrained: it cannot name a task the model was never given, and the
    entity it asks about has to be one it can be asked about.

    Args:
        schema: The schema that query is written for.
        mode: Whether the query defines a task or asks a question.
        can: What the model and its deployment support.
    """

    def __init__(
        self,
        schema: SchemaDefinition,
        mode: Mode = Mode.TRAINING,
        can: Capabilities | None = None,
    ):
        self.schema = schema
        self.mode = mode
        self.can = can if can is not None else Capabilities()

    def validate(
        self, parsed_query: ParsedPredictiveQuery
    ) -> ValidationResponse:
        r"""Checks the structure a pretrained model requires.

        Args:
            parsed_query: the input parsed query.

        Returns:
            List of encountered errors/warnings.

        A `LIST_DISTINCT` over a column that is not a foreign key is a
        multilabel target whether or not the query ranks it, and it is refused
        outside the RANK branch for that reason. Asking only about the ranked
        form let the plain one reach the engine, which does not implement it,
        so a caller was told the SDK had broken with `INTERNAL_ERROR` when the
        answer is that this model does not serve that kind of task. On a
        foreign key both forms resolve to link prediction and are left alone.

        Every reason a query is refused is collected before any is raised. A
        query can be static *and* ask for too many candidates, and reporting
        only the first sends the caller to fix it and be refused again for the
        second, which is the shape of defect this exists to remove.
        """
        response = ValidationResponse()
        if parsed_query.rfm_query:
            # TODO: deprecate the `rfm_query` field. It does not need to be
            # user-facing
            assert self.mode is Mode.INFERENCE
        if self.mode is Mode.TRAINING:
            # A query that defines a task cannot also name the entities
            # it is asked about; that is what asking a question means.
            if parsed_query.for_each == FOR:
                response.errors.append(
                    ValidationError(
                        title='Invalid query structure',
                        message=(
                            '"FOR" clause is only supported for a pretrained '
                            'model, use FOR EACH when training your '
                            'own model.'
                        ),
                        code=codes.MODEL_FOR_REQUIRES_PRETRAINED,
                    )
                )
            if parsed_query.rfm_entity_ids is not None:
                response.errors.append(
                    ValidationError(
                        title='Invalid query structure',
                        message=(
                            'Specifying entities is only supported for a '
                            'pretrained model, not when training '
                            'your own model.'
                        ),
                        code=codes.MODEL_ENTITY_IDS_REQUIRE_PRETRAINED,
                    )
                )
            return response

        assert self.mode is Mode.INFERENCE

        # Disable multilabel everything
        target_ast = parsed_query.target_ast
        if isinstance(target_ast, Join):
            # Handles static link prediction
            target_ast = target_ast.rhs_target

        if (
            isinstance(target_ast, Column)
            and parsed_query.problem_type == ProblemType.CLASSIFY
        ):
            response.errors.append(
                ValidationError(
                    title='Unsupported query structure',
                    message=(
                        'Foundation Model queries do not support '
                        'static multilabel classification. To make '
                        'predictions with this query, train a new model.'
                    ),
                    code=codes.MODEL_NO_STATIC_MULTILABEL,
                )
            )

        if (
            isinstance(target_ast, Aggregation)
            and target_ast.aggr == AggregationType.LIST_DISTINCT
        ):
            aggregated_col_name = target_ast.get_target_column_name()
            _, target_is_fkey = self._is_key_col(aggregated_col_name)

            if parsed_query.problem_type == ProblemType.CLASSIFY or (
                target_ast.stype == Stype.multicategorical
                and not target_is_fkey
            ):
                response.errors.append(
                    ValidationError(
                        title='Unsupported query structure',
                        message=(
                            'Foundation Model queries do not support '
                            'multilabel tasks. Use `LIST_DISTINCT` on a '
                            'foreign key column to make a link prediction '
                            'query. To make predictions with this query, '
                            'train a new model.'
                        ),
                        code=codes.MODEL_NO_MULTILABEL,
                    )
                )
            else:
                if target_ast.aggr_time_range is None:
                    response.errors.append(
                        ValidationError(
                            title='Static Link Prediction not supported',
                            message=(
                                'LIST_DISTINCT without a time range is not '
                                'supported by the foundation model. Give it a '
                                'time range such as (0, 30, days), which needs '
                                'a time column on the aggregated table. To '
                                'make predictions with this query as written, '
                                'train a new model.'
                            ),
                            code=codes.MODEL_NO_STATIC_LINK_PREDICTION,
                        )
                    )
                top_k = parsed_query.top_k
                if (
                    parsed_query.problem_type == ProblemType.RANK
                    and top_k is not None
                    and top_k > MAX_TOP_K
                ):
                    response.errors.append(
                        ValidationError(
                            title='Top k is too large',
                            message=(
                                f'Top k {top_k} exceeds maximum '
                                f'foundation model supported value of '
                                f'{MAX_TOP_K}. To list more candidates, '
                                f'train a new model.'
                            ),
                            code=codes.MODEL_TOP_K_TOO_LARGE,
                        )
                    )

        # Disable one-hop away filters/targets
        response = merge(
            response, self._find_disabled_joins(parsed_query.entity_ast)
        )
        response = merge(
            response, self._find_disabled_joins(parsed_query.target_ast)
        )
        if parsed_query.whatif_ast is not None:
            response = merge(
                response, self._find_disabled_joins(parsed_query.whatif_ast)
            )
        # Disable unsupported string operations
        response = merge(
            response, self._find_disabled_ops(parsed_query.entity_ast)
        )
        response = merge(
            response, self._find_disabled_ops(parsed_query.target_ast)
        )
        if parsed_query.whatif_ast is not None:
            response = merge(
                response, self._find_disabled_ops(parsed_query.whatif_ast)
            )
        # Disable unsupported aggregations
        response = merge(
            response,
            self._find_disabled_aggrs(
                parsed_query.entity_ast,
                permit_list_distinct=False,
                nested_aggr=False,
            ),
        )
        response = merge(
            response,
            self._find_disabled_aggrs(
                parsed_query.target_ast,
                permit_list_distinct=True,
                nested_aggr=False,
            ),
        )
        if parsed_query.whatif_ast is not None:
            response = merge(
                response,
                self._find_disabled_aggrs(
                    parsed_query.whatif_ast,
                    permit_list_distinct=False,
                    nested_aggr=False,
                ),
            )
        # Disable unsupported key references
        response = merge(
            response,
            self._find_key_cols(
                parsed_query.entity_ast,
                permit_list_distinct=False,
                permit_fkey=False,
                permit_pkey=True,
            ),
        )
        response = merge(
            response,
            self._find_key_cols(
                parsed_query.target_ast,
                permit_list_distinct=True,
                permit_fkey=False,
                permit_pkey=False,
            ),
        )
        if parsed_query.whatif_ast is not None:
            response = merge(
                response,
                self._find_key_cols(
                    parsed_query.whatif_ast,
                    permit_list_distinct=False,
                    permit_fkey=False,
                    permit_pkey=False,
                ),
            )

        if parsed_query.for_each == FOR:
            if parsed_query.rfm_entity_ids is None:
                response.errors.append(
                    ValidationError(
                        title='Invalid query structure',
                        message=(
                            'When using "FOR" in foundation model '
                            'queries, you must specify one or multiple '
                            'entities to make the prediction for '
                            '(e.g., "FOR table.column = <ID>")'
                        ),
                        code=codes.MODEL_FOR_REQUIRES_ENTITY_IDS,
                    )
                )
                return response

            # Check that entity IDs are defined correctly
            assert isinstance(parsed_query.rfm_entity_ids, Condition)
            message = parsed_query.rfm_entity_ids.get_location().message_start
            op = parsed_query.rfm_entity_ids.op
            assert isinstance(parsed_query.rfm_entity_ids.value, Constant)
            entity_ids = parsed_query.rfm_entity_ids.value.typed_value()
            if op == RelOp.EQ:
                if isinstance(entity_ids, list):
                    response.errors.append(
                        ValidationError(
                            title='Invalid query structure',
                            message=(
                                f'{message}: Expected a single entity ID, '
                                f'but got a list: '
                                + str(entity_ids)
                                .replace('[', '(')
                                .replace(']', ')')
                            ),
                            code=codes.MODEL_ENTITY_ID_LIST_NOT_ALLOWED,
                        )
                    )
                    return response
            elif op == MemberOp.IN:
                if not isinstance(entity_ids, list):
                    response.errors.append(
                        ValidationError(
                            title='Invalid query structure',
                            message=(
                                f'{message}: Expected a list of entity IDs, '
                                f'but got {entity_ids}.'
                            ),
                            code=codes.MODEL_ENTITY_ID_LIST_REQUIRED,
                        )
                    )
                else:
                    if len(entity_ids) > MAX_ENTITY_LIST_LENGTH:
                        entity_ids_repr = str(
                            entity_ids[:MAX_ENTITY_LIST_LENGTH]
                        )
                        entity_ids_repr = entity_ids_repr[:-1] + ', ... ]'
                        response.errors.append(
                            ValidationError(
                                title='The entity id list is too long',
                                message=(
                                    f'{message}: List of entity IDs '
                                    f'{entity_ids_repr} has length '
                                    f'{len(entity_ids)} which exceeds '
                                    f'maximum allowed length of '
                                    f'{MAX_ENTITY_LIST_LENGTH}.'
                                ),
                                code=codes.MODEL_ENTITY_ID_LIST_TOO_LONG,
                            )
                        )
            else:
                response.errors.append(
                    ValidationError(
                        title='Invalid query structure',
                        message=(
                            f'{message}: Expected entity IDs, with operators '
                            f'IN or =, but got {symbol(op)}. Valid syntax for '
                            f'specifying entity IDs is '
                            f'"FOR table.column = <ID>" or '
                            f'"FOR table.column IN (<ID>, <ID>, ...)".'
                        ),
                        code=codes.MODEL_ENTITY_ID_OPERATOR,
                    )
                )
        else:
            if parsed_query.rfm_entity_ids is not None:
                message = (
                    parsed_query.rfm_entity_ids.get_location().message_start
                )
                response.errors.append(
                    ValidationError(
                        title='Invalid query structure',
                        message=(
                            f'{message}: Entity IDs are not expected '
                            f'when `FOR EACH` syntax is used. Got '
                            f'{parsed_query.rfm_entity_ids}.'
                        ),
                        code=codes.MODEL_ENTITY_IDS_WITH_FOR_EACH,
                    )
                )
        return response

    def _find_disabled_joins(self, node: ASTNode) -> ValidationResponse:
        response = ValidationResponse()
        if isinstance(node, Join):
            if isinstance(node.rhs_target, Column):
                response.errors.append(
                    ValidationError(
                        title='Unsupported implicit join',
                        message=(
                            f'{node.get_location().message_start}: '
                            f'Static references to columns from other '
                            f'tables that implicitly contain a '
                            f'foreign key to primary key connection '
                            f'are not supported in foundation model queries. '
                            f'Your query implicitly requires a join '
                            f'{node.lhs_key} -> {node.rhs_key}. '
                            f'Please remove the reference to table '
                            f'{table_name(node.rhs_key)} and retry, or '
                            f'train a new model instead of the '
                            f'foundation model.'
                        ),
                        code=codes.MODEL_NO_IMPLICIT_JOIN,
                        span=Span.at(node.get_location()),
                    )
                )
        for child in node.children:
            response = merge(response, self._find_disabled_joins(child))
        return response

    def _is_key_col(self, fqn: str) -> tuple[bool, bool]:
        r"""Returns (is_pkey, is_fkey) for the given column FQN."""
        col_key = ColumnKey(table_name(fqn), col_name(fqn))
        is_pkey = self.schema.tables[table_name(fqn)].pkey == col_name(fqn)
        all_keys = []
        for col_group in self.schema.col_groups:
            all_keys.extend(list(col_group.columns))
        is_fkey = col_key in all_keys and not is_pkey
        return is_pkey, is_fkey

    def _find_disabled_ops(self, node: ASTNode) -> ValidationResponse:
        response = ValidationResponse()
        unsupported_str_ops = (
            StrOp.STARTS_WITH,
            StrOp.ENDS_WITH,
            StrOp.CONTAINS,
            StrOp.NOT_CONTAINS,
            RelOp.LEQ,
            RelOp.GEQ,
            RelOp.LT,
            RelOp.GT,
        )
        if isinstance(node, Condition):
            assert isinstance(node.value, Constant)
            if (
                node.value.dtype_maybe == Dtype.string
                and node.op in unsupported_str_ops
            ):
                op_rep = (
                    symbol(node.op) if node.input_op is None else node.input_op
                )
                response.errors.append(
                    ValidationError(
                        title='Unsupported operation',
                        message=(
                            f'{node.get_location().message_start}: '
                            f'Operation {op_rep} between strings is '
                            f'not supported in the foundation model queries.'
                        ),
                        code=codes.MODEL_NO_STRING_COMPARISON,
                        span=Span.at(node.get_location()),
                    )
                )
            if (
                isinstance(node.target, Column)
                and node.target.stype == Stype.ID
                and any(self._is_key_col(node.target.fqn))
                and not self.can.key_columns_in_filters
            ):
                response.errors.append(
                    ValidationError(
                        title='Unsupported operation',
                        message=(
                            f'{node.get_location().message_start}: '
                            f'Operations on primary key and foreign key '
                            f'columns are not supported in the foundation '
                            f'model queries.'
                        ),
                        code=codes.MODEL_NO_KEY_COLUMN_FILTER,
                        span=Span.at(node.get_location()),
                    )
                )
            elif not (
                node.target.stype in [Stype.numerical, Stype.categorical]
                or (
                    node.target.stype == Stype.timestamp
                    and self.can.timestamp_columns_in_filters
                )
                or node.target.stype == Stype.ID
            ):
                response.errors.append(
                    ValidationError(
                        title='Unsupported operation',
                        message=(
                            f'{node.get_location().message_start}: '
                            f'Operations on a '
                            f'{node.target.stype.value} column are '
                            f'not supported in the foundation model queries '
                            f'yet.'
                        ),
                        code=codes.MODEL_UNSUPPORTED_COLUMN_OPERATION,
                        span=Span.at(node.get_location()),
                    )
                )
        for child in node.children:
            response = merge(response, self._find_disabled_ops(child))
        return response

    def _find_disabled_aggrs(
        self,
        node: ASTNode,
        permit_list_distinct: bool = False,
        nested_aggr: bool = False,
    ) -> ValidationResponse:
        response = ValidationResponse()
        unsupported_aggrs = (
            AggregationType.FIRST,
            AggregationType.LAST,
            AggregationType.COUNT_DISTINCT,
        )
        if isinstance(node, Aggregation):
            if node.aggr in unsupported_aggrs:
                response.errors.append(
                    ValidationError(
                        title='Unsupported aggregation',
                        message=(
                            f'{node.get_location().message_start}: '
                            f'Aggregation {symbol(node.aggr)} type is not '
                            f'supported in the foundation model queries.'
                        ),
                        code=codes.MODEL_UNSUPPORTED_AGGREGATION,
                        span=Span.at(node.get_location()),
                    )
                )
            if (
                not node.target.dtype.is_numerical()
                and node.aggr != AggregationType.LIST_DISTINCT
            ):
                response.errors.append(
                    ValidationError(
                        title='Unsupported aggregation',
                        message=(
                            f'{node.get_location().message_start}: '
                            f'Aggregations of {node.target.stype.value} '
                            f'columns are not '
                            f'supported in the foundation model queries.'
                        ),
                        code=codes.MODEL_UNSUPPORTED_AGGREGATION_COLUMN,
                        span=Span.at(node.get_location()),
                    )
                )
            if (
                node.aggr == AggregationType.LIST_DISTINCT
                and not permit_list_distinct
            ):
                response.errors.append(
                    ValidationError(
                        title='Unsupported aggregation',
                        message=(
                            f'{node.get_location().message_start}: '
                            f'Aggregation {symbol(node.aggr)} is not '
                            f'supported in the foundation model queries '
                            f'except for foreign key targets.'
                        ),
                        code=codes.MODEL_AGGREGATION_NEEDS_FOREIGN_KEY,
                        span=Span.at(node.get_location()),
                    )
                )
            if nested_aggr:
                response.errors.append(
                    ValidationError(
                        title='Unsupported nested aggregation',
                        message=(
                            f'{node.get_location().message_start}: '
                            f'Nesting aggregations is not '
                            f'supported in the foundation model queries.'
                        ),
                        code=codes.MODEL_NO_NESTED_AGGREGATION,
                        span=Span.at(node.get_location()),
                    )
                )

            # NOTE: Set nested_aggr=True after processing current aggregation
            # to prevent further aggregations from being nested inside this
            # one. This ensures we catch cases where aggregations are used
            # inside other aggregations, which is not supported by the
            # relational foundation model.
            nested_aggr = True
        if isinstance(node, Condition):
            permit_list_distinct = False
        for child in node.children:
            response = merge(
                response,
                self._find_disabled_aggrs(
                    child,
                    permit_list_distinct=permit_list_distinct,
                    nested_aggr=nested_aggr,
                ),
            )
        return response

    def _find_key_cols(
        self,
        node: ASTNode,
        permit_list_distinct: bool = False,
        permit_fkey: bool = False,
        permit_pkey: bool = False,
    ) -> ValidationResponse:
        response = ValidationResponse()
        # The only two situations where we permit key references:
        # Entity definition and within a target LIST_DISTINCT
        if (
            isinstance(node, Aggregation)
            and node.aggr == AggregationType.LIST_DISTINCT
            and permit_list_distinct
        ):
            permit_fkey = True
        if isinstance(node, Aggregation) and node.aggr == AggregationType.COUNT:
            permit_fkey = True
            permit_pkey = True
        if isinstance(node, Condition):
            permit_list_distinct = False
            # A deployment whose serving stack cannot handle key columns
            # says so; it is not a property of the language or of a model.
            permit_fkey = self.can.key_columns_in_filters
            permit_pkey = self.can.key_columns_in_filters
        if isinstance(node, Column) and col_name(node.fqn) != '*':
            target_is_pkey, target_is_fkey = self._is_key_col(node.fqn)
            if (target_is_pkey and not permit_pkey) or (
                target_is_fkey and not permit_fkey
            ):
                response.errors.append(
                    ValidationError(
                        title='Unsupported column reference',
                        message=(
                            f'{node.get_location().message_start}: '
                            f'References to primary keys and foreign keys are '
                            f'not supported in the foundation model queries '
                            f'except for foreign key targets in COUNT and '
                            f'LIST_DISTINCT targets.'
                        ),
                        code=codes.MODEL_NO_KEY_COLUMN_REFERENCE,
                        span=Span.at(node.get_location()),
                    )
                )
        # Filter requires special arg care
        if isinstance(node, Filter):
            response = merge(
                response,
                self._find_key_cols(
                    node.condition,
                    permit_list_distinct=permit_list_distinct,
                    permit_fkey=False,
                    permit_pkey=False,
                ),
            )
            response = merge(
                response,
                self._find_key_cols(
                    node.target,
                    permit_list_distinct=permit_list_distinct,
                    permit_fkey=permit_fkey,
                    permit_pkey=permit_pkey,
                ),
            )
            return response
        for child in node.children:
            response = merge(
                response,
                self._find_key_cols(
                    child,
                    permit_list_distinct=permit_list_distinct,
                    permit_fkey=permit_fkey,
                    permit_pkey=permit_pkey,
                ),
            )
        return response

    def update_location_interval(
        self, parsed_query: ParsedPredictiveQuery
    ) -> None:
        r"""If `parsed_query` has `self.evaluate` or `self.explain` set to
        :obj:`True`, updates the interval values across the entire PQuery
        to adjust for the removed prefix.

        Args:
            parsed_query (ParsedPredictiveQuery): Query to adjust locations to.
        """
        offset = 0
        if parsed_query.explain:
            offset += len('EXPLAIN ')
        if parsed_query.evaluate:
            offset += len('EVALUATE ')
        if offset == 0:
            return
        self._update_location_interval(parsed_query.target_ast, offset)
        self._update_location_interval(parsed_query.entity_ast, offset)
        if parsed_query.rfm_entity_ids is not None:
            self._update_location_interval(parsed_query.rfm_entity_ids, offset)
        if parsed_query.whatif_ast is not None:
            self._update_location_interval(parsed_query.whatif_ast, offset)

    def _update_location_interval(self, node: ASTNode, offset: int) -> None:
        location = node.location
        if location is not None and location.data_available:
            # The second branch reads what the first one wrote, so rebind
            # rather than reading the field again.
            if location.start_row == 1:
                location = ASTQueryLocationInterval(
                    start_row=location.start_row,
                    start_col=location.start_col + offset,
                    end_row=location.end_row,
                    end_col=location.end_col,
                )
                node.location = location
            if location.end_row == 1:
                location = ASTQueryLocationInterval(
                    start_row=location.start_row,
                    start_col=location.start_col,
                    end_row=location.end_row,
                    end_col=location.end_col + offset,
                )
                node.location = location
        for child in node.children:
            self._update_location_interval(child, offset)
