# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""How sdm-client names itself to Databricks.

Databricks requires a partner product to identify every connection, SDK call
and API request it sends, by a ``<partner>_<product>/<version>`` entry. Here
rather than in sdm-client so the connectors can send it too, and imported by
sdm-client's serving transport so there is one identifier. The five
distributions release on one version, so this package's is sdm-client's.
"""

from __future__ import annotations

import re

from sdm_connectors._version import __version__

#: Named as the distribution a Databricks administrator would see installed,
#: never a model or endpoint.
DATABRICKS_PARTNER = 'nvidia'
DATABRICKS_PRODUCT = 'sdm-client'

#: The query tag on every warehouse session sdm-client opens, which says what
#: its statements are for without changing the identifier above.
SAMPLING_QUERY_TAG = ('sdm_client_purpose', 'sampling')

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


def partner_user_agent() -> tuple[str, str]:
    r"""The ``(name, version)`` naming sdm-client, for
    ``Config.with_user_agent_extra``.

    An extra rather than the SDK's ``product`` slot, which names the
    application and may be the caller's own.
    """
    return (
        f'{DATABRICKS_PARTNER}_{DATABRICKS_PRODUCT}',
        databricks_product_version(__version__),
    )


def partner_identifier() -> str:
    r"""``nvidia_sdm-client/<version>``, as a driver's User-Agent entry or a
    Postgres ``application_name`` carries it."""
    return '/'.join(partner_user_agent())
