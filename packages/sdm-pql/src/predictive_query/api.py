# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

r"""The one call: text and a schema in, a resolved query or diagnostics out."""

from predictive_query.parser.parser import PQLParser
from predictive_query.profiles import Profile
from predictive_query.query import ValidatedPredictiveQuery
from predictive_query.schema import codes
from predictive_query.schema.common import ValidationError, ValidationResponse
from predictive_query.schema.definition import SchemaDefinition
from predictive_query.validator.capability import (
    check_capabilities,
    check_parsed_capabilities,
)
from predictive_query.validator.predictive_query_validator import (
    PredictiveQueryValidator,
)


def validate(
    query: str,
    schema: SchemaDefinition,
    profile: Profile,
) -> tuple[ValidatedPredictiveQuery | None, ValidationResponse]:
    r"""Resolve *query* against *schema*, then ask whether *profile* can run it.

    The two are separate on purpose. Resolution decides whether this is a
    predictive query at all, and its answer does not depend on the model.
    Capability decides whether this particular model will take it. A query that
    fails resolution fails for every model; a query that fails only capability
    is good PQL that something else can answer.

    Capability is asked first. A model that cannot answer this shape of
    question cannot answer it against any schema, and resolving first put a
    complaint about the schema in front of a capability problem.

    Args:
        query: The predictive query, as text.
        schema: The schema to resolve names and types against. No rows.
        profile: What the query is for, and what the model can do.

    Returns:
        The resolved query and an empty response, or :obj:`None` and the
        diagnostics explaining why not.
    """
    try:
        parsed = PQLParser(mode=profile.mode).to_parsed_predictive_query(query)
    except ValueError as error:
        return None, ValidationResponse(
            errors=[
                ValidationError(
                    title='Invalid Syntax',
                    message=str(error),
                    code=codes.PARSE_SYNTAX,
                )
            ]
        )
    except TypeError as error:
        # A literal the grammar accepts but nothing can be made of, such as a
        # date of 30 February. The parser raises for it; a caller handing us
        # text should get a diagnostic rather than a TypeError.
        return None, ValidationResponse(
            errors=[
                ValidationError(
                    title='Invalid constant',
                    message=str(error),
                    code=codes.VALUE_INVALID_LITERAL,
                )
            ]
        )

    parsed_capability = check_parsed_capabilities(parsed, profile)
    if not parsed_capability.ok:
        return None, parsed_capability

    resolved, response = PredictiveQueryValidator(
        schema, mode=profile.mode, can=profile.capabilities
    ).validate_predictive_query(parsed)
    if resolved is None or not response.ok:
        return None, response

    capability = check_capabilities(resolved, schema, profile)
    response.errors.extend(capability.errors)
    response.warnings.extend(capability.warnings)
    if not response.ok:
        return None, response
    return resolved, response
