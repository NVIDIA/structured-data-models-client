# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""How structured-data-models-client names itself to Databricks.

Databricks requires a partner product to identify every connection, SDK call
and API request it sends, by a ``<partner>_<product>/<version>`` entry. A
private copy of structured-data-models-client's, since the connectors never
import the client; a test holds the two to the same identifier. The five
distributions release on one version, so this package's is
structured-data-models-client's.
"""

from __future__ import annotations

import re

from sdm_connectors._version import __version__

#: The product half of the partner identifier: a name an administrator can
#: tie to this client, never a model or endpoint. It is the distribution's
#: name: Databricks wants an underscore between partner and product and
#: its SDK takes ``[a-zA-Z0-9_.+-]``, so the dashes stay and that underscore
#: is the only one. It is a wire value that partner telemetry keys on, so it
#: changes only with the distribution's name.
DATABRICKS_PARTNER = 'nvidia'
DATABRICKS_PRODUCT = 'structured-data-models-client'

#: The query tag on every warehouse session this client opens, which says what
#: its statements are for without changing the identifier above. Its key is
#: the distribution's name with underscores for the dashes, since Databricks
#: refuses a query tag key holding ``-``: a session's tag with one is dropped
#: and reported as ``tags_invalid``.
SAMPLING_QUERY_TAG = ('structured_data_models_client_purpose', 'sampling')

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
    r"""The ``(name, version)`` naming structured-data-models-client, for
    ``Config.with_user_agent_extra``.

    An extra rather than the SDK's ``product`` slot, which names the
    application and may be the caller's own.
    """
    return (
        f'{DATABRICKS_PARTNER}_{DATABRICKS_PRODUCT}',
        databricks_product_version(__version__),
    )


def partner_identifier() -> str:
    r"""``nvidia_structured-data-models-client/<version>``, as a driver's
    User-Agent entry or a Postgres ``application_name`` carries it."""
    return '/'.join(partner_user_agent())
