# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An endpoint that does not serve sessions raises what `fit` documents.

`fit` promised `UnsupportedRequestError` and raised the base
`StructuredDataError`. The promised class is a subclass of the raised one, so
the `except` written against the docstring never fired. Raising the subclass
instead would have replaced the error's ``SESSIONS_UNSUPPORTED`` code with
that class's fixed ``UNSUPPORTED_REQUEST``, and the code is what callers are
told to branch on, so the docstring was corrected to name what is raised,
and this pins both halves of it.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from sdm_client import KumoTabular, Request, SDMClient
from sdm_client.errors import StructuredDataError, UnsupportedRequestError


def _frame(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'id': [float(index) for index in range(rows)],
            'x': [float(index) for index in range(rows)],
            'y': [index % 2 for index in range(rows)],
        }
    )


class _NoSessions:
    def __init__(self, status: int) -> None:
        self.status = status

    def create_session(self, payload: dict) -> dict:
        error = StructuredDataError('no such route', code='HTTP_ERROR')
        error.status_code = self.status
        raise error

    def close(self) -> None:
        return None


@pytest.fixture
def request_() -> Request:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Request(
            context=_frame(),
            predict=_frame(3)[['id', 'x']],
            target='y',
            task='classification',
        )


@pytest.mark.parametrize('status', [404, 405, 501])
def test_fit_raises_the_error_its_docstring_names(
    request_: Request, status: int
) -> None:
    client = SDMClient('http://127.0.0.1:1', 'none', timeout=1.0, max_retries=0)
    client._transport = _NoSessions(status)

    with pytest.raises(StructuredDataError) as caught:
        client.fit(request_, model=KumoTabular())

    assert caught.value.code == 'SESSIONS_UNSUPPORTED'
    assert not isinstance(caught.value, UnsupportedRequestError)
