# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import Enum


class HTTPMethod(Enum):
    r"""HTTP methods supported by the API."""

    GET = 'GET'
    POST = 'POST'
    DELETE = 'DELETE'


@dataclass(frozen=True)
class Endpoint:
    r"""Represents an API endpoint with its path and HTTP method."""

    path: str | None = field(default=None)
    method: HTTPMethod = HTTPMethod.GET

    def get_path(self) -> str:
        if self.path is None:
            raise ValueError('Endpoint requires a path')
        return self.path
