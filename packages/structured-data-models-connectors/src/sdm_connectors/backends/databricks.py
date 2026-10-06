# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeAlias

from sdm_connectors._databricks_partner import (
    SAMPLING_QUERY_TAG,
    partner_identifier,
)
from sdm_connectors.sql import (
    ConnectorError,
    check_connect_args,
    merge_driver_options,
    require_driver,
)

_ABSENT = ('databricks', 'databricks.sql', 'databricks.sql.client')

# A type checker reads the driver's own types; at runtime its absence is
# reported as MissingBackendError rather than an ImportError.
if TYPE_CHECKING:
    import databricks.sql as databricks_sql
    import databricks.sql.client as _client
else:
    databricks_sql = require_driver(
        'databricks',
        'databricks-sql-connector',
        'databricks.sql',
        absent=_ABSENT,
    )
    _client = require_driver(
        'databricks',
        'databricks-sql-connector',
        'databricks.sql.client',
        absent=_ABSENT,
    )

if not hasattr(_client, 'Connection'):
    raise ImportError(
        "cannot import name 'Connection' from 'databricks.sql.client'"
    )

Connection: TypeAlias = _client.Connection

_ENV_BY_ARG = {
    'server_hostname': 'DATABRICKS_SERVER_HOSTNAME',
    'http_path': 'DATABRICKS_HTTP_PATH',
    'access_token': 'DATABRICKS_TOKEN',
    'catalog': 'DATABRICKS_CATALOG',
    'schema': 'DATABRICKS_SCHEMA',
}


_DOCUMENTED_KWARGS = frozenset(
    {
        'auth_type',
        'azure_client_id',
        'azure_client_secret',
        'azure_tenant_id',
        'azure_workspace_resource_id',
        'credentials_provider',
        'experimental_oauth_persistence',
        'oauth_client_id',
        'oauth_client_secret',
        'identity_federation_client_id',
        'oauth_jwt_algorithm',
        'oauth_jwt_key_file',
        'oauth_jwt_kid',
        'oauth_jwt_passphrase',
        'oauth_redirect_port',
        'oauth_redirect_port_range',
        'oauth_scopes',
        'oauth_token_cache_enabled',
        'password',
        'token_url',
        'session_configuration',
        'tls_verify',
        'use_cloud_fetch',
        'use_kernel',
        'user_agent_entry',
        'username',
        '_tls_client_cert_file',
        '_tls_client_cert_key_file',
        '_tls_client_cert_key_password',
        '_use_cert_as_auth',
    }
)


# What this guard is for: the driver's thrift route, handed nothing it
# recognises, falls through to an OAuth flow that opens a browser and waits for
# a human, binding a listening socket while it does. A script that simply
# forgot its token never returns. Everything else the driver refuses itself,
# immediately and with a better message than this one could give.
#
# So the question is narrow: would this call reach that fallthrough? The names
# below are the branches the installed driver dispatches on before it, each
# with the names it reads together. Anything that matches one of them, and
# anything on the kernel route (which raises rather than waiting) is left
# to the driver.
_WAYS_THE_THRIFT_ROUTE_AUTHENTICATES = (
    frozenset({'access_token'}),
    frozenset({'credentials_provider'}),
    frozenset({'auth_type'}),
    frozenset({'_use_cert_as_auth', '_tls_client_cert_file'}),
    # Refused by the driver before it reaches any flow. Its sentence says that
    # this is no longer supported and to use OAuth or an access token, which is
    # more than this one could say.
    frozenset({'username'}),
    frozenset({'password'}),
)


def _an_auth_type_the_driver_knows() -> frozenset[str]:
    r"""The flows the installed driver dispatches ``auth_type`` on.

    Read from the driver. Any other value falls through to the browser, so
    accepting `auth_type` on sight let a typo produce exactly the wait this
    guard exists to prevent.
    """
    try:
        from databricks.sql.auth.auth import AuthType

        return frozenset(member.value for member in AuthType)
    except Exception:  # noqa: BLE001 - an unknown driver is not a credential
        return frozenset()


_KNOWN_AUTH_TYPES = _an_auth_type_the_driver_knows()


#: Naming an OAuth application is choosing the browser flow on purpose. The
#: driver resolves to it for anything it does not recognise, including an
#: empty call. The driver cannot tell a deliberate U2M setup from a
#: forgotten token, and this is the only thing that can: one of them says the
#: word `oauth`. Not beside a machine-to-machine secret, which asks for
#: something else and would be quietly downgraded to a browser.
_AN_OAUTH_APP = frozenset(
    {'oauth_client_id', 'oauth_redirect_port', 'oauth_redirect_port_range'}
)
_MACHINE_TO_MACHINE = frozenset(
    {'oauth_client_secret', 'oauth_jwt_key_file', 'azure_client_secret'}
)


def _says_how_to_authenticate(given: Mapping[str, Any]) -> bool:
    r"""Check whether the call reaches a supported authentication route.

    False is the answer that matters: whatever the call names, the driver
    would fall through to its browser flow. A machine-to-machine secret
    alone returns False for that reason. It is a credential, but this route
    does not use it.
    """
    named = {name for name, value in given.items() if value}
    if 'use_kernel' in named:
        return True
    if 'auth_type' in named and given['auth_type'] not in _KNOWN_AUTH_TYPES:
        named = named - {'auth_type'}
    if _AN_OAUTH_APP & named and not _MACHINE_TO_MACHINE & named:
        return True
    return any(way <= named for way in _WAYS_THE_THRIFT_ROUTE_AUTHENTICATES)


def _declared_connect_args() -> frozenset[str]:
    r"""Return the Databricks driver's accepted connection keywords.

    The driver accepts ``**kwargs`` and silently ignores unknown names, so this
    guard rejects typos before they fall back to ambient defaults. OAuth
    options are added explicitly because the driver routes them through
    ``**kwargs``, where ``inspect.signature`` cannot see them.
    """
    parameters = inspect.signature(Connection.__init__).parameters.values()
    return frozenset(
        parameter.name
        for parameter in parameters
        if parameter.kind is not parameter.VAR_KEYWORD
        and parameter.name != 'self'
    )


_CONNECT_ARGS = (
    _declared_connect_args() | frozenset(_ENV_BY_ARG) | _DOCUMENTED_KWARGS
)


def connect(
    *,
    driver_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Connection:
    r"""Connect to a Databricks SQL warehouse.

    ``DATABRICKS_*`` environment variables are used when their corresponding
    keyword arguments are absent, including ``DATABRICKS_CATALOG`` and
    ``DATABRICKS_SCHEMA``, which decide which table a bare name resolves to.
    Passing one explicitly as ``None`` stops this function reading the variable
    for it, and the key is then not sent at all; for ``catalog`` and ``schema``
    that leaves the choice to the workspace, and for a credential it leaves no
    credential, which is refused rather than sent.

    A call that would reach the browser flow without asking for it is
    refused here. The driver's own answer is to open a browser and wait for
    a human to finish an OAuth flow, binding a listening socket while it
    does, so a script that simply forgot the token never returned. Asking
    for that flow deliberately is allowed through. Such a request names an
    OAuth application or an ``auth_type`` the driver dispatches on.

    The connection names sdm-client in its User-Agent, ahead of any
    ``user_agent_entry`` given, and tags its session's statements
    ``sdm_client_purpose: sampling``.
    """
    check_connect_args('databricks', kwargs, _CONNECT_ARGS)
    given = set(kwargs)
    kwargs = merge_driver_options('databricks', kwargs, driver_options)
    for arg, env in _ENV_BY_ARG.items():
        if arg not in given and arg not in kwargs:
            kwargs[arg] = os.getenv(env)
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    if not _says_how_to_authenticate(kwargs):
        raise ConnectorError(
            'connecting to Databricks needs a credential and none was given: '
            'pass access_token=..., set DATABRICKS_TOKEN, or name '
            'credentials_provider, an auth_type, an oauth_client_id, or '
            '_use_cert_as_auth with _tls_client_cert_file. Half of a pair is '
            'not a credential, and '
            'neither is a machine-to-machine one on this route: the driver '
            'resolves what it does not recognise to its browser flow, which '
            'waits for a human and never returns',
            code='INVALID_CONNECTOR_ARGS',
        )
    return databricks_sql.connect(**_as_sdm_client(kwargs))


def _as_sdm_client(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    r"""Return ``databricks.sql.connect`` arguments for a connection
    structured-data-models-client opens.

    Databricks requires a partner product to identify every connection it
    opens, so ``user_agent_entry`` starts with ``nvidia_sdm-client/<version>``
    and keeps any entry the caller gave after it. The session's statements
    are tagged ``sdm_client_purpose: sampling``, beside any query tags the
    caller set, so a query history can tell sampling apart without a
    second identifier. A caller's own ``sdm_client_purpose`` is kept instead.

    Raises:
        ConnectorError: ``INVALID_CONNECTOR_ARGS`` for a ``user_agent_entry``
            no HTTP header can carry.
    """
    connect_args = dict(kwargs)
    entry = connect_args.get('user_agent_entry')
    # None is omitted, as connect() drops any None it is given and the driver
    # reads one as its default.
    if entry is None:
        connect_args['user_agent_entry'] = partner_identifier()
    else:
        _check_user_agent_entry(entry)
        connect_args['user_agent_entry'] = f'{partner_identifier()} {entry}'
    key, value = SAMPLING_QUERY_TAG
    tags = connect_args.get('query_tags')
    if isinstance(tags, Mapping):
        # The driver lets these replace the session's QUERY_TAGS outright. A
        # caller's own sdm_client_purpose is theirs to set, so it wins.
        connect_args['query_tags'] = {key: value, **tags}
        return connect_args
    # The session's own setting, which every driver from 4.1.3 sends on both
    # its Thrift and SEA routes; `query_tags=` would need 4.2.6.
    session = dict(connect_args.get('session_configuration') or {})
    name = next((k for k in session if k.upper() == 'QUERY_TAGS'), 'QUERY_TAGS')
    theirs = session.get(name)
    if not theirs:
        session[name] = f'{key}:{value}'
    elif key not in _query_tag_keys(str(theirs)):
        session[name] = f'{theirs},{key}:{value}'
    connect_args['session_configuration'] = session
    return connect_args


def _query_tag_keys(serialized: str) -> set[str]:
    r"""The keys in a ``QUERY_TAGS`` string, written as the driver's
    ``serialize_query_tags`` writes one: ``key:value`` pairs joined by ``,``,
    with a backslash escaping the character after it.

    Raises:
        ConnectorError: ``INVALID_CONNECTOR_ARGS`` for a string that ends in
            an unpaired backslash, which would escape the separator after it.
    """
    keys: set[str] = set()
    key: list[str] = []
    in_key = True
    characters = iter(serialized)
    for character in characters:
        if character == '\\':
            escaped = next(characters, None)
            if escaped is None:
                raise ConnectorError(
                    'invalid Databricks QUERY_TAGS session setting: it ends '
                    'in a backslash that escapes nothing; write a literal '
                    'backslash as two',
                    code='INVALID_CONNECTOR_ARGS',
                )
            if in_key:
                key.append(escaped)
        elif character == ',':
            keys.add(''.join(key))
            key, in_key = [], True
        elif character == ':' and in_key:
            in_key = False
        elif in_key:
            key.append(character)
    keys.add(''.join(key))
    return keys


def _check_user_agent_entry(entry: object) -> None:
    r"""Refuse a ``user_agent_entry`` that no HTTP header can carry.

    The driver sends it as is, and a bad one fails inside the HTTP layer at
    send time, where it reads as a network problem rather than a typo.
    """
    if not isinstance(entry, str):
        refusal = 'must be a string'
    elif not entry.strip():
        refusal = 'must contain a visible character'
    elif any(
        ord(character) < 32 or ord(character) == 127 for character in entry
    ):
        refusal = 'must not contain control characters'
    elif not _latin_1(entry):
        refusal = (
            'must be latin-1 encodable, since it is sent as an HTTP header'
        )
    else:
        return
    raise ConnectorError(
        f'invalid Databricks user_agent_entry: it {refusal}',
        code='INVALID_CONNECTOR_ARGS',
    )


def _latin_1(text: str) -> bool:
    try:
        text.encode('latin-1')
    except UnicodeEncodeError:
        return False
    return True
