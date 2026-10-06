# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""How a prediction request names this SDK to Databricks.

Databricks attributes partner traffic by a ``<partner>_<product>/<version>``
entry in the User-Agent. structured-data-models-connectors keeps its own copy for the
connections it opens, so the serving extra need not bring it and its pandas;
a test holds the two to the same identifier.
"""

from __future__ import annotations

import re

from sdm_client._version import __version__

#: The product half of the partner identifier: a name an administrator can
#: tie to this client, never a model or endpoint. It is a wire value, so it
#: outlived the rename of the distribution that sends it.
DATABRICKS_PARTNER = 'nvidia'
DATABRICKS_PRODUCT = 'sdm-client'

_PACKAGE_VERSION_RE = re.compile(
    r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)'
    r'(?:(a|b|rc)(0|[1-9]\d*))?'
    r'(?:(?:\.dev(0|[1-9]\d*))|(?:\.post(0|[1-9]\d*)))?$'
)
_PRERELEASE_NAMES = {'a': 'alpha', 'b': 'beta', 'rc': 'rc'}


def databricks_product_version(package_version: str) -> str:
    r"""Return the SDK release in the SemVer form Databricks accepts.

    Stable SDK releases already use SemVer. Python pre-releases use PEP 440's
    compact spelling, so translate the pre-release, development and post-release
    forms supported by this project. Other PEP 440 forms are rejected
    deliberately: an invented fallback would claim the wrong SDK release.

    Raises:
        ValueError: If ``package_version`` has no such form.
    """
    match = _PACKAGE_VERSION_RE.fullmatch(package_version)
    if match is None:
        raise ValueError(
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


def sdm_client_user_agent() -> tuple[str, str]:
    r"""The ``(name, version)`` entry naming this SDK, for
    ``Config.with_user_agent_extra``.

    An extra rather than the ``product`` slot, which names the application
    and may be the caller's own.
    """
    return (
        f'{DATABRICKS_PARTNER}_{DATABRICKS_PRODUCT}',
        databricks_product_version(__version__),
    )
