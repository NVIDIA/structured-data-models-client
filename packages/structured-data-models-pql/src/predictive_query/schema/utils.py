# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from predictive_query.schema import codes
from predictive_query.schema.common import (
    StrEnum,
    ValidationError,
    ValidationResponse,
)


class QueryType(StrEnum):
    r"""Defines the type of a predictive query."""

    STATIC = 'static'
    TEMPORAL = 'temporal'


def validate_int(
    const: int, min_int: int | None = None, max_int: int | None = None
) -> ValidationResponse:
    r"""Validate that size of :obj:`const` is within the supported
    limits.

    Args:
        const (int): Evaluated const.
        min_int (int, optional): Minimum permitted int. If :obj:`None`,
            minimum of int64 is used. (default: :obj:`None`)
        max_int (int, optional): Maximum permitted int. If :obj:`None`,
            maximum of int64 is used. (default: :obj:`None`)

    Returns:
        ValidationResponse: List of encountered errors.
    """
    if min_int is None:
        min_int = np.iinfo(np.int64).min
    if max_int is None:
        max_int = np.iinfo(np.int64).max
    response = ValidationResponse()
    if const > max_int or const < min_int:
        response.errors.append(
            ValidationError(
                title='Unsupported constant',
                message=f'Constant {const} is outside of the range of '
                f'supported integers ({min_int}, {max_int}).',
                code=codes.VALUE_INT_OUT_OF_RANGE,
            )
        )
    return response


def maybe_bold(s: str, rich: bool) -> str:
    return f'[bold]{s}[/bold]' if rich else s
