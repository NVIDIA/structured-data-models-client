# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from sdm_relational_engine.client.client import capped_body
from sdm_relational_engine.client.transport import TransportResponse
from sdm_relational_engine.exceptions import (
    HTTPException,
    InvalidResponseError,
)


def raise_on_error(response: TransportResponse) -> None:
    r"""Raise an :class:`~sdm_relational_engine.exceptions.HTTPException` for
    a non-OK response.

    The body is capped: a NIM behind a proxy can answer an error with a large
    HTML page, and the whole of it used to become the exception message. The
    response is already bounded by the read cap, which is measured in
    megabytes, so the cap here is what keeps the message readable.

    A missing status code is reported rather than asserted. This is the path
    every failed request takes, and under ``python -O`` an assertion here is
    removed entirely: the status went on to ``http.HTTPStatus(None)``, which
    raised an unrelated ``TypeError`` and hid the real failure.
    """
    if not response.ok:
        if response.status_code is None:
            raise InvalidResponseError(
                'The NIM returned a failed response carrying no HTTP status '
                f'code. Body: {capped_body(response.text)}'
            )
        raise HTTPException(response.status_code, capped_body(response.text))
