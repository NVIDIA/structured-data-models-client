# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import deque

from predictive_query.ast import (
    Aggregation,
    ASTNode,
    Column,
    Condition,
    Filter,
    Join,
    LogicalOperation,
)
from predictive_query.query import ParsedPredictiveQuery
from predictive_query.schema import codes
from predictive_query.schema.common import (
    Span,
    ValidationError,
    ValidationResponse,
)
from predictive_query.schema.definition import ColumnKey, SchemaDefinition
from predictive_query.validator.utils import col_name, merge, table_name


class JoinValidator:
    r"""Class with all the logic around :class:`ASTNode` join inferral and
    validation.

    Args:
        schema: Schema on which the query is defined.
    """

    def __init__(self, schema: SchemaDefinition) -> None:
        self.schema = schema
        self.target_filter_example: str | None = None
        self.entity_filter_tables: set[str] = set()
        self.entity_filter_is_row_level = False
        self.allow_temporal_entity_filter = False

    def infer_and_validate_joins(
        self,
        parsed_query: ParsedPredictiveQuery,
    ) -> ValidationResponse:
        r"""Modifies the AST in place, inserting :class:`Join` nodes where
        appropriate. Returns errors/warnings if there are issues resolving
        fkey to pkey relations, like ambiguous references, or missing keys.

        Args:
            parsed_query: the input parsed query.

        Returns:
            ValidationResponse: List of encountered errors/warnings.
        """
        entity_column = parsed_query.entity_column
        entity_table = table_name(entity_column)
        entity_filter_condition = None
        self.entity_filter_tables = set()
        self.entity_filter_is_row_level = False
        if isinstance(parsed_query.entity_ast, Filter):
            condition = parsed_query.entity_ast.condition
            entity_filter_condition = repr(condition)
            self.entity_filter_tables = {
                table_name(column) for column in condition.all_query_columns
            }
            self.entity_filter_is_row_level = not self._contains_aggregation(
                condition
            )
        self.allow_temporal_entity_filter = (
            parsed_query.target_ast.date_offset_range is not None
        )
        self.target_filter_example = self._get_target_filter_example(
            parsed_query
        )
        response = ValidationResponse()
        entity_ast, entity_response = self._infer_joins(
            parsed_query.entity_ast,
            entity_table,
            in_entity_filter=isinstance(parsed_query.entity_ast, Filter),
            entity_filter_condition=entity_filter_condition,
        )

        assert isinstance(entity_ast, Column | Filter)
        parsed_query.entity_ast = entity_ast
        target_ast, target_response = self._infer_joins(
            parsed_query.target_ast, entity_table
        )
        assert isinstance(
            target_ast,
            LogicalOperation | Join | Condition | Column | Aggregation,
        )
        parsed_query.target_ast = target_ast
        response = merge(response, entity_response)
        response = merge(response, target_response)
        if parsed_query.whatif_ast is not None:
            whatif_ast, whatif_response = self._infer_joins(
                parsed_query.whatif_ast, entity_table
            )
            assert isinstance(whatif_ast, (Condition, LogicalOperation))
            parsed_query.whatif_ast = whatif_ast
            response = merge(response, whatif_response)
        return response

    def _get_target_filter_example(
        self, parsed_query: ParsedPredictiveQuery
    ) -> str | None:
        r"""The same filter, rewritten onto the target, when that is equivalent.

        Only when the entity filter is row-level and reads the very table the
        target aggregates: then moving it inside the aggregation says what the
        caller meant, so the message can quote the exact query to write.
        """
        entity_ast = parsed_query.entity_ast
        target_ast = parsed_query.target_ast
        if (
            not isinstance(entity_ast, Filter)
            or not isinstance(target_ast, Aggregation)
            or not isinstance(target_ast.target, Column)
            or self._contains_aggregation(entity_ast.condition)
        ):
            return None

        condition_tables = {
            table_name(column)
            for column in entity_ast.condition.all_query_columns
        }
        target_column = target_ast.target.fqn
        if condition_tables != {table_name(target_column)}:
            return None

        filtered_column = f'{target_column} WHERE {entity_ast.condition}'
        return repr(target_ast).replace(target_column, filtered_column, 1)

    def _contains_aggregation(self, node: ASTNode) -> bool:
        r"""Returns whether the AST subtree contains an aggregation."""
        if isinstance(node, Aggregation):
            return True
        return any(self._contains_aggregation(child) for child in node.children)

    def _route_between(self, start: str, goal: str) -> list[str]:
        r"""The shortest route of foreign keys from ``start`` to ``goal``.

        Only for explaining a refusal. An aggregation joins on a single key, so
        a target two tables away cannot be expressed however short the route
        is -- but "there is no foreign key from C to A" is false when C reaches
        A through B, and it sends the caller to add a key rather than to ask
        the question one level down.

        Directed, child to parent, because an undirected walk invents routes
        that cannot be travelled. `A -> B` is the reverse of `B -> A` and
        finding it would advise aggregating the parent for each child, which is
        the query that was just refused; two children of one parent are joined
        to it and to nothing else, and finding `C -> A -> D` would advise a
        join between siblings that does not exist. The parent of a group is the
        table whose own primary key is in it.

        Two tables joined on each other's key, or neither on its own, name no
        parent between them and so have no route through here.
        """
        points_at: dict[str, set[str]] = {}
        for col_group in self.schema.col_groups:
            tables = {column.table_name for column in col_group.columns}
            parents = {
                table
                for table in tables
                if self.schema.tables[table].pkey is not None
                and ColumnKey(table, self.schema.tables[table].pkey)
                in col_group.columns
            }
            if len(parents) != 1:
                continue
            parent = next(iter(parents))
            for child in tables - {parent}:
                points_at.setdefault(child, set()).add(parent)

        seen = {start}
        queue: deque[list[str]] = deque([[start]])
        while queue:
            route = queue.popleft()
            for parent in sorted(points_at.get(route[-1], ())):
                if parent == goal:
                    return [*route, parent]
                if parent not in seen:
                    seen.add(parent)
                    queue.append([*route, parent])
        return []

    def _find_keys(
        self,
        node: Column | Aggregation,
        fkey_table: str,
        pkey_table: str,
        is_target_entity: bool = False,
        in_entity_filter: bool = False,
        entity_filter_condition: str | None = None,
    ) -> tuple[list[str], ValidationResponse]:
        r"""Find the unique foreign key from ``fkey_table`` to ``pkey_table``.

        Returns the matching key names and a validation response. The response
        contains an error unless exactly one key is valid, because guessing the
        join would change the query. ``node`` only makes those errors specific;
        it takes no part in choosing the key.

        Keys are accumulated across groups rather than taken from the last.
        A group qualifies by holding the parent key, which every child of that
        parent does, so a sibling's group arrives holding no column of the
        table being asked about and overwrote the answer an earlier group had
        found. That left only the last-declared child of a parent aggregatable,
        and reversing the declaration order swapped which one worked.

        Accumulating is also what makes the ambiguity check true: two foreign
        keys from one table into the same parent arrive as two groups, and
        taking the last would see one key and call the join unique when it is
        not. Repeats are dropped so that a group seen twice cannot report an
        ambiguity that is not there.

        A table that is reachable but further than a join can go is named as
        such. Saying there is no foreign key would send the caller to add one
        that already exists a hop away, which is the wrong repair for the
        wrong problem; and reading a column rather than aggregating rows is
        the same distance asked a different way, so its repair is a column one
        hop away rather than an aggregation one level down.
        """
        response = ValidationResponse()
        keys = []
        reverse_keys = []
        for col_group in self.schema.col_groups:
            is_keys_group = False
            is_rev_keys_group = False
            table_of_interest = None
            pkey_of_pkey_table = self.schema.tables[pkey_table].pkey
            pkey_of_fkey_table = self.schema.tables[fkey_table].pkey
            if (
                pkey_of_pkey_table is not None
                and ColumnKey(pkey_table, pkey_of_pkey_table)
                in col_group.columns
            ):
                is_keys_group = True
                table_of_interest = fkey_table
            elif (
                pkey_of_fkey_table is not None
                and ColumnKey(fkey_table, pkey_of_fkey_table)
                in col_group.columns
            ):
                is_rev_keys_group = True
                table_of_interest = pkey_table
            if not is_keys_group and not is_rev_keys_group:
                continue
            assert isinstance(table_of_interest, str)
            # When fkey_table == pkey_table (self-referential), exclude the
            # pkey itself from the results - we only want actual foreign keys.
            pkey_col = self.schema.tables[pkey_table].pkey
            curr_keys = [
                f'{col.table_name}.{col.col_name}'
                for col in col_group.columns
                if col.table_name == table_of_interest
                and not (fkey_table == pkey_table and col.col_name == pkey_col)
            ]
            if is_keys_group:
                keys.extend(k for k in curr_keys if k not in keys)
            else:
                assert is_rev_keys_group
                reverse_keys.extend(
                    k for k in curr_keys if k not in reverse_keys
                )

        if isinstance(node, Aggregation):
            pkey_name = self.schema.tables[pkey_table].pkey
            additional_explanation = (
                f"To unambiguously link each row in '{fkey_table}' to a "
                f"'{pkey_table}.{pkey_name}', exactly one foreign key should "
                f"point from '{fkey_table}' to '{pkey_table}.{pkey_name}'."
            )
        else:
            additional_explanation = (
                f"To unambiguously link each row in '{fkey_table}' to a "
                f"'{node}', exactly one foreign key should "
                f"point from '{fkey_table}' to '{pkey_table}'."
            )

            # Add an additional error message to the response
            # which captures the case that the target column is not
            # reachable from the source table through a foreign key.
            # Here is an example where target column
            # TRANSACTIONS.AMOUNT is not connected to the source table
            # USERS through a foreign key in the USERS table:
            # PREDICT TRANSACTIONS.AMOUNT FOR EACH USERS.USER_ID
            # Here source table is the fkey_table
            if is_target_entity:
                additional_explanation += (
                    f' This may also mean there is an incorrect '
                    f'target column specification. Column {node.fqn} is '
                    f'not reachable through a foreign key in table '
                    f'{fkey_table}.'
                )
        if len(keys) == 0:
            # This is likely the case where the target column is not reachable
            # unambiguously from the entity table. For example, for the query:
            # PREDICT transactions.age FOR EACH customers.customer_id
            # where there is a foreign key from transactions to customers (
            # containing the primary key). We do not support this if there is
            # no aggregation.
            is_aggregation = False
            if len(reverse_keys) > 0:
                if not isinstance(node, Aggregation):
                    column_name = col_name(node.fqn)
                else:
                    is_aggregation = True
                if not isinstance(node, Aggregation) and in_entity_filter:
                    condition = entity_filter_condition or repr(node)
                    fixes = []
                    if self.target_filter_example is not None:
                        fixes.append(
                            'To filter events used by the prediction target, '
                            f"use '{self.target_filter_example}'."
                        )
                    if (
                        self.allow_temporal_entity_filter
                        and entity_filter_condition is not None
                        and self.entity_filter_is_row_level
                        and self.entity_filter_tables == {pkey_table}
                        and self.schema.tables[pkey_table].time_col is not None
                    ):
                        fixes.append(
                            'To select entities that have matching events, '
                            f"use 'COUNT({pkey_table}.* WHERE {condition}, "
                            "-INF, 0) > 0'."
                        )
                    message = (
                        f'{node.get_location().message_start}: '
                        f"The entity filter condition '{condition}' "
                        f"references event column '{node.fqn}' directly. "
                        f"Table '{pkey_table}' can contain multiple rows for "
                        f"each '{fkey_table}', so there is no single value to "
                        f'compare for each entity. Move this condition inside '
                        f'an aggregation.'
                    )
                    if fixes:
                        message += f' {" ".join(fixes)}'
                    response.errors.append(
                        ValidationError(
                            title='Invalid entity filter',
                            message=message,
                            code=codes.JOIN_EVENT_COLUMN_IN_ENTITY_FILTER,
                        )
                    )
                    return keys, response
                if not isinstance(node, Aggregation):
                    column_name = col_name(node.fqn)
                    response.errors.append(
                        ValidationError(
                            title=f"Couldn't pick a unique {node.fqn} for "
                            f'each {fkey_table}',
                            message=f'{node.get_location().message_start}: '
                            'Encountered an error, we have found multiple '
                            f"'{column_name}' values in the '{pkey_table}' "
                            f"table for a single '{fkey_table}'. To fix "
                            f'this, you have two options:\n'
                            '1. Aggregate the values:'
                            f" If your '{pkey_table}' table has a "
                            f"'created at' column, use an aggregation "
                            f'function to reduce them to one value. For '
                            f'example: PREDICT LAST({node.fqn}, 0, 30, '
                            f'days).\n'
                            '2. Rewrite your query:'
                            ' If you are writing a static node '
                            'prediction predictive query, make sure that '
                            'you either use the same table to define your '
                            'target and entity, or you have a foreign key '
                            'from the entity pointing to the target table, '
                            'not the other way around.',
                            code=codes.JOIN_AMBIGUOUS_TARGET_VALUE,
                            span=Span.at(node.get_location()),
                        )
                    )
            if is_aggregation or len(reverse_keys) == 0:
                through = self._route_between(fkey_table, pkey_table)
                if through and isinstance(node, Aggregation):
                    explanation = (
                        f'{fkey_table} reaches {pkey_table} through '
                        f'{" -> ".join(through)}, and an aggregation joins on '
                        f'one key rather than following a path, so its target '
                        f'has to be a table that points at {pkey_table} '
                        f'directly. Aggregate {fkey_table} for each '
                        f'{through[1]} instead, or carry '
                        f"{pkey_table}'s key on {fkey_table} as well."
                    )
                elif through:
                    explanation = (
                        f'{node.fqn} is on {pkey_table}, which {fkey_table} '
                        f'reaches through {" -> ".join(through)}, and a query '
                        f'follows one foreign key rather than a path. Ask for '
                        f'a column of {through[1]} instead, or carry '
                        f"{pkey_table}'s key on {fkey_table} as well."
                    )
                else:
                    explanation = (
                        f'There is no foreign key from table {fkey_table} to '
                        f'table {pkey_table}. {additional_explanation}'
                    )
                response.errors.append(
                    ValidationError(
                        title='Missing foreign key',
                        message=f'{node.get_location().message_start}: '
                        f"Encountered an error when processing '{node}'. "
                        f'{explanation}',
                        code=codes.JOIN_MISSING_FOREIGN_KEY,
                        span=Span.at(node.get_location()),
                    )
                )
        elif len(keys) > 1:
            # Naming the call is the difference between a diagnosis and a
            # repair: the caller has to drop every foreign key but one, and
            # which one to keep is theirs to decide, not ours to guess.
            # `keys` are qualified as '<table>.<column>' but `Schema.unlink`
            # takes the bare column, so a suggestion built from them verbatim
            # would not run.
            drops = '; '.join(
                f"schema.unlink('{fkey_table}', "
                f"'{key.split('.')[-1]}', '{pkey_table}')"
                for key in keys[1:]
            )
            response.errors.append(
                ValidationError(
                    title='Ambiguous link between tables',
                    message=f'{node.get_location().message_start}: '
                    f"Encountered an error when processing '{node}'. "
                    f'The foreign key from table {fkey_table} '
                    f'to table {pkey_table} is not unique (got {keys}). '
                    f'{additional_explanation} Keep the one this query '
                    f'means and drop the rest, for example: {drops}',
                    code=codes.JOIN_AMBIGUOUS_FOREIGN_KEY,
                    span=Span.at(node.get_location()),
                )
            )
        return keys, response

    def _infer_joins(
        self,
        node: ASTNode,
        source_table: str,
        in_entity_filter: bool = False,
        entity_filter_condition: str | None = None,
    ) -> tuple[ASTNode, ValidationResponse]:
        r"""Make ``node`` reachable from ``source_table`` by adding joins.

        The existing AST is mutated in place except when a new
        :class:`Join` wrapper is needed. The validation response carries any
        lookup errors or ambiguity found while choosing the join.
        """
        # We handle five cases:
        # 1. Column might get replaced with Join->Column if
        #    it's one-hop away
        # 2. Aggregation always gets replaced with Join->Aggregation
        # 3. Condition is the only node where children might get replaced and
        #    not just modified in-place and return nodes need to be handled
        # 4. Filter has special handling of source table - source table for
        #    condition is the target table.
        # 5. LogicalOperation does not need any extra action
        response = ValidationResponse()
        if isinstance(node, Column):
            # fkey -> pkey hop (if necessary)
            col_table = table_name(node.fqn)
            if source_table != col_table:
                # There is a link to a separate table, ensure that it's unique
                keys, response = self._find_keys(
                    node,
                    source_table,
                    col_table,
                    is_target_entity=True,
                    in_entity_filter=in_entity_filter,
                    entity_filter_condition=entity_filter_condition,
                )
                if not response.ok:
                    return node, response
                rhs_pkey = self.schema.tables[col_table].pkey
                assert rhs_pkey is not None
                return Join(
                    rhs_target=node,
                    lhs_key=keys[0],
                    rhs_key=col_table + '.' + rhs_pkey,
                ), response
        elif isinstance(node, Aggregation):
            # pkey -> fkey hop
            target_table = table_name(node.get_target_column_name())
            aggregation_target, target_response = self._infer_joins(
                node.target, target_table
            )
            assert isinstance(aggregation_target, Filter | Column)
            node.target = aggregation_target
            response = merge(response, target_response)
            keys, join_response = self._find_keys(
                node, target_table, source_table
            )
            response = merge(response, join_response)
            if not response.ok:
                return node, response
            node.group_by = keys[0]
            lhs_pkey = self.schema.tables[source_table].pkey
            assert lhs_pkey is not None
            return Join(
                rhs_target=node,
                lhs_key=source_table + '.' + lhs_pkey,
                rhs_key=keys[0],
            ), response
        elif isinstance(node, Condition):
            # replace the target
            filter_condition = entity_filter_condition
            if in_entity_filter and filter_condition is None:
                filter_condition = repr(node)
            condition_target, target_response = self._infer_joins(
                node.target,
                source_table,
                in_entity_filter=in_entity_filter,
                entity_filter_condition=filter_condition,
            )
            assert isinstance(condition_target, Aggregation | Column | Join)
            node.target = condition_target
            response = merge(response, target_response)
        elif isinstance(node, Filter):
            # special handling of condition source table
            target_table = table_name(node.target.fqn)
            target, target_response = self._infer_joins(
                node.target, source_table
            )
            assert isinstance(target, Column)
            node.target = target
            response = merge(response, target_response)
            condition, condition_response = self._infer_joins(
                node.condition,
                target_table,
                in_entity_filter=in_entity_filter,
                entity_filter_condition=entity_filter_condition,
            )
            assert isinstance(condition, (Condition, LogicalOperation))
            node.condition = condition
            response = merge(response, condition_response)
        else:
            for child in node.children:
                modified_child, child_response = self._infer_joins(
                    child,
                    source_table,
                    in_entity_filter=in_entity_filter,
                    entity_filter_condition=entity_filter_condition,
                )
                # assert that no re-initialisation is necessary
                assert child is modified_child
                response = merge(response, child_response)
        return node, response
