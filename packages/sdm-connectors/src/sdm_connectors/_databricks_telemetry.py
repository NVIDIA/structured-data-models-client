# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from __future__ import annotations

import re

#: Databricks requires a partner product to identify every connection it
#: opens. The product is this SDK, named as the distribution a Databricks
#: administrator would see installed, never a model or endpoint.
DATABRICKS_PARTNER = 'nvidia'
DATABRICKS_PRODUCT = 'sdm-client'

_PACKAGE_VERSION_RE = re.compile(
    r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)'
    r'(?:(a|b|rc)(0|[1-9]\d*))?'
    r'(?:(?:\.dev(0|[1-9]\d*))|(?:\.post(0|[1-9]\d*)))?$'
)
_PRERELEASE_NAMES = {'a': 'alpha', 'b': 'beta', 'rc': 'rc'}


class DatabricksTelemetryVersionError(ValueError):
    r"""The installed release cannot be represented as a Databricks version."""


class DatabricksUserAgentEntryError(ValueError):
    r"""A caller-provided User-Agent entry is not safe to send as a header."""


def databricks_product_version(package_version: str) -> str:
    r"""Return the SDK release in the SemVer form Databricks accepts.

    Stable SDK releases already use SemVer. Python pre-releases use PEP 440's
    compact spelling, so translate the pre-release, development and post-release
    forms supported by this project. Other PEP 440 forms are rejected
    deliberately: an invented fallback would claim the wrong SDK release.
    """
    match = _PACKAGE_VERSION_RE.fullmatch(package_version)
    if match is None:
        raise DatabricksTelemetryVersionError(
            f'Package version {package_version!r} cannot be represented in the '
            'Databricks User-Agent; expected a supported PEP 440 release '
            'based on X.Y.Z'
        )
    major, minor, patch, prerelease, number, dev, post = match.groups()
    version = f'{major}.{minor}.{patch}'
    if prerelease is not None:
        version = f'{version}-{_PRERELEASE_NAMES[prerelease]}.{number}'
    if dev is not None:
        separator = '.' if prerelease is not None else '-'
        version = f'{version}{separator}dev.{dev}'
    if post is not None:
        version = f'{version}+post.{post}'
    return version


def databricks_user_agent(
    package_version: str,
    additional_entry: str | None = None,
) -> str:
    r"""The partner identifier plus an optional caller-provided entry."""
    version = databricks_product_version(package_version)
    required = f'{DATABRICKS_PARTNER}_{DATABRICKS_PRODUCT}/{version}'
    if additional_entry is None:
        return required
    if not isinstance(additional_entry, str):
        raise TypeError('user_agent_entry must be a string')
    if not additional_entry.strip():
        raise DatabricksUserAgentEntryError(
            'user_agent_entry must contain a visible character'
        )
    if any(
        ord(character) < 32 or ord(character) == 127
        for character in additional_entry
    ):
        raise DatabricksUserAgentEntryError(
            'user_agent_entry must not contain control characters'
        )
    # A header is latin-1 on the wire. A character outside it raises a
    # UnicodeEncodeError from inside the HTTP layer at send time, which reads
    # as a connection problem and sends the caller to look at the network for
    # something they typed.
    try:
        additional_entry.encode('latin-1')
    except UnicodeEncodeError as error:
        raise DatabricksUserAgentEntryError(
            'user_agent_entry must be latin-1 encodable, since it is sent as '
            f'an HTTP header: {error}'
        ) from error
    return f'{required} {additional_entry}'
