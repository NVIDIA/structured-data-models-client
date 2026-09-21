# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any

from predictive_query import RELATIONAL, Profile
from predictive_query.parser.parser import PQLParser
from predictive_query.query import ValidatedPredictiveQuery
from predictive_query.schema.definition import SchemaDefinition
from predictive_query.validator.capability import check_capabilities
from predictive_query.validator.predictive_query_validator import (
    PredictiveQueryValidator,
)

from sdm_relational_engine.rfm.base import composite_key


def _name_the_identity(
    parsed_query: Any,
    graph_definition: SchemaDefinition,
) -> None:
    r"""Lets a query name a table whose identity spans several columns.

    Such a table is keyed on a column this client derives, and asking a caller to
    spell that is asking them to know an implementation detail. Any column of
    the identity therefore names the identity, and the derived column keeps
    working for a query already written against it.

    The identity is read back out of the derived column's own name, so a
    caller validating a query holds nothing but the graph definition the
    service is given -- the rule here and the rule there cannot drift apart.
    """
    entity = getattr(parsed_query, 'entity_column_obj', None)
    if entity is None or not isinstance(getattr(entity, 'fqn', None), str):
        return
    table_name, _, column = entity.fqn.partition('.')
    table = graph_definition.tables.get(table_name)
    if table is None or not table.pkey:
        return
    identity = composite_key.decode_identity(table.pkey)
    if identity is None or column not in identity:
        return
    entity.fqn = f'{table_name}.{table.pkey}'


def parse_query_locally(
    query: str,
    graph_definition: SchemaDefinition,
    profile: Profile = RELATIONAL,
) -> ValidatedPredictiveQuery:
    r"""Parse and validate ``query`` against ``graph_definition``.

    The profile is what the model may be asked, and it may only reject: the
    language accepts a fixed set of queries and every model accepts a subset.
    Passing it here is what makes a query legal for one model and refused for
    another, without the parser knowing which models exist. The previous
    parser took an enum with one value per *caller*, which is not the same
    thing as a capability.

    The three steps are spelled out rather than deferred to
    ``predictive_query.validate`` because a composite identity has to be named
    between the second and the third: the query says
    ``FOR EACH people.\`Customer ID\``` and the graph knows that column only as
    part of a folded key, so the reference is rewritten once the tree exists
    and before anything validates it against the schema.
    """
    try:
        parsed_query = PQLParser(mode=profile.mode).to_parsed_predictive_query(
            query
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Failed to parse query '{query}'. {exc}") from None

    _name_the_identity(parsed_query, graph_definition)

    try:
        validated_query, response = PredictiveQueryValidator(
            graph_definition,
            mode=profile.mode,
            can=profile.capabilities,
        ).validate_predictive_query(parsed_query)
    except Exception as exc:
        raise ValueError(f"Failed to parse query '{query}'. {exc}") from None

    if validated_query is None or not response.ok:
        raise ValueError(
            f"Failed to parse query '{query}'. {_response_message(response)}"
        )

    capability = check_capabilities(validated_query, graph_definition, profile)
    response.errors.extend(capability.errors)
    response.warnings.extend(capability.warnings)
    if not response.ok:
        raise ValueError(
            f"Failed to parse query '{query}'. {_response_message(response)}"
        )

    if len(response.warnings) > 0:
        msg = '\n'.join(
            [
                f'{i + 1}. {warning.title}: {warning.message}'
                for i, warning in enumerate(response.warnings)
            ]
        )
        warnings.warn(
            f'Encountered the following warnings during parsing:\n{msg}'
        )

    return validated_query


def _response_message(response: Any) -> str:
    message = getattr(response, 'message', None)
    if callable(message):
        return str(message())
    return str(response)
