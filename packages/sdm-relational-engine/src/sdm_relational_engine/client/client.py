# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from typing import Any, NoReturn
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from sdm_api.endpoints import Endpoint, HTTPMethod

# `typing.assert_never` is 3.11+, and this package supports 3.10. Sourced from
# `typing_extensions`, as `Self` already is throughout the package.
from typing_extensions import assert_never
from urllib3.util import Retry

from sdm_relational_engine.exceptions import (
    AuthenticationError,
    ClientInitializationError,
    InvalidResponseError,
    NimTimeoutError,
    NimUnreachableError,
)

logger = logging.getLogger('sdm_relational_engine')

_AUTH_STATUS_CODES = frozenset({401, 403})
_LOCAL_HOSTS = frozenset({'localhost', '127.0.0.1', '::1'})
_MAX_BODY_SNIPPET = 512
# Mirrors ``sdm_client.core.transport``: well above any real prediction
# response (a full-precision float64 column for a million rows is ~25 MB of
# JSON), but far below what a compressed hostile body can inflate to.
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_RESPONSE_CHUNK_BYTES = 1024 * 1024

_SESSIONS_PATH = '/v1/sessions'
_RETRY_STATUS = [408, 429, 500, 502, 503, 504]
_RETRY_BACKOFF_FACTOR = 2.0
# Caps how long a server-chosen `Retry-After` may park the caller; urllib3's
# own default is six hours.
_RETRY_AFTER_MAX_SECONDS = 60
# The readiness/model probes run at init, before the caller is waiting on a
# result, so they use their own short bound rather than the request timeout:
# an unreachable endpoint should fail fast rather than hold up construction.
_PROBE_TIMEOUT_SECONDS = 10
# The `user:secret@` of a URL authority, wherever one appears in free text.
_USERINFO_RE = re.compile(r'(?<=//)[^/@\s]+@')


def _build_retry(max_retries: int, retry_post: bool = True) -> Retry:
    r"""The retry policy for one Kumo Relational client.

    ``retry_post=False`` drops ``POST`` from the retryable methods, switching
    off the status retries for a route whose side effect the client cannot
    reconcile. Connection failures stay retryable either way: urllib3 gates
    status retries on the method but not connect ones, and a request that never
    reached the server cannot have had an effect.

    ``read=False`` switches off read retries entirely. A read timeout on a
    prediction means the NIM may already be running it, and resending
    duplicates the most expensive work in the system on a GPU that is by
    hypothesis struggling. The request-level ``predict(num_retries=...)``
    knows how to pace that and treats a timeout as transient. Passing
    ``False`` rather than ``0`` also re-raises the underlying ``ReadTimeout``
    instead of burying it in a ``MaxRetryError``, which is what lets the caller
    be told to raise their timeout.

    ``retry_after_max`` needs urllib3 >= 2.3; older versions fall back to
    urllib3's own (six hour) cap.
    """
    methods = {'GET', 'POST'} if retry_post else {'GET'}
    options: dict[str, Any] = {
        'total': max_retries,
        'connect': max_retries,
        'read': False,
        'status': max_retries,
        'status_forcelist': _RETRY_STATUS,
        'allowed_methods': frozenset(methods),
        'backoff_factor': _RETRY_BACKOFF_FACTOR,
        'respect_retry_after_header': True,
        'raise_on_status': False,
    }
    try:
        return Retry(retry_after_max=_RETRY_AFTER_MAX_SECONDS, **options)
    except TypeError:
        return Retry(**options)


def _validate_url(url: str, api_key: str | None) -> None:
    r"""Mirrors ``sdm_client.core.transport._validate_url``.

    This client carries every Kumo Relational prediction, so the documented
    guard must hold here too, not only on the path that constructs a
    ``Transport`` first.
    """
    if not isinstance(url, str):
        raise ValueError(f'url must be a string, got {type(url).__name__}')
    parsed = urlparse(url)
    shown = redact_url(url)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError(
            f'url must start with http:// or https://, got {shown!r}'
        )
    if not parsed.hostname:
        raise ValueError(f'url is missing a host, got {shown!r}')
    # A base URL is a scheme, a host and at most a path prefix. Routes are
    # appended to this string, so a query or fragment produces a malformed
    # request target, and a credential placed there travels in the request
    # line, where redaction never sees it.
    if parsed.query or parsed.fragment:
        raise ValueError(
            'url must be a base endpoint without a query string or fragment; '
            'pass a credential as api_key rather than in the URL'
        )
    # Userinfo is a credential too: `requests` turns it into an
    # `Authorization: Basic` header. Checking only api_key let
    # `http://user:secret@host` send it in the clear.
    carries_credential = bool(api_key or parsed.username or parsed.password)
    if (
        carries_credential
        and parsed.scheme == 'http'
        and parsed.hostname.lower() not in _LOCAL_HOSTS
    ):
        what = 'an API key' if api_key else 'the credential in this URL'
        raise ValueError(
            f'refusing to send {what} over plaintext HTTP; use an '
            f'https:// URL (or a localhost endpoint), got {shown!r}'
        )


def redact_url(url: str | None) -> str | None:
    r"""Strip any userinfo from ``url`` so it is safe to log or display.

    ``https://user:token@host/path`` is a legal way to point the client at a
    deployment, and the credential is in the URL itself. Anywhere a URL reaches
    a log line, an error message or a repr, it goes through here first.
    """
    if not url:
        return url
    parsed = urlparse(url)
    if not parsed.username and not parsed.password:
        return url
    host = parsed.hostname or ''
    if ':' in host:
        host = f'[{host}]'
    if parsed.port:
        host = f'{host}:{parsed.port}'
    return parsed._replace(netloc=host).geturl()


def scrub_userinfo(text: str) -> str:
    r"""Remove any ``user:secret@`` credential from free text.

    This complements :func:`redact_url` for strings that contain a URL, mainly
    driver exception messages in initialization errors that echo the requested
    URL.
    """
    return _USERINFO_RE.sub('', text)


class _Session(requests.Session):
    r"""Mirrors ``sdm_client.core.transport._Session``.

    ``requests`` strips only the standard ``Authorization`` header when a
    redirect changes origin; a custom header set on the session is re-sent
    verbatim. Without this, the configured endpoint could hand the API key and,
    on a 307/308, the request body to any host it names, defeating
    ``_validate_url``.

    The two clients keep their own copy rather than sharing one: the only
    package both depend on is ``sdm-connectors``, which is a SQL-connector
    package with no HTTP surface and no ``requests`` dependency, and
    ``sdm_relational_engine`` cannot import ``sdm_client`` because the
    dependency runs the other way. This is the same arrangement as
    ``_validate_url`` above.
    """

    @staticmethod
    def _release_unread(resp: object) -> None:
        r"""Give up a redirect hop's socket without reading its body.

        ``_read_capped`` bounds the response this client finally parses, but it
        only runs once ``requests`` has returned. Redirect hops are consumed
        before that, in full and uncapped, to release their sockets. Closing
        each one instead keeps the cap meaningful for the whole exchange rather
        than just its last leg.
        """
        if not isinstance(resp, requests.Response):
            return
        raw = getattr(resp, 'raw', None)
        if raw is not None:
            raw.close()
        resp._content = b''
        resp._content_consumed = True  # type: ignore[attr-defined]

    def resolve_redirects(  # type: ignore[override]
        self,
        resp: requests.Response,
        req: requests.PreparedRequest,
        **kwargs: Any,
    ) -> Any:
        r"""Follow redirects without reading the bodies they carry.

        Every hop is released, not just the first: ``requests`` resolves the
        rest inside the generator below and reads each as it loops, so each is
        released as it is yielded.
        """
        # ``Session.send`` calls this for every response, redirect or not, so
        # the check matters: releasing unconditionally would close the body of
        # a successful response before the caller ever saw it.
        if self.get_redirect_target(resp) is not None:
            self._release_unread(resp)
        for hop in super().resolve_redirects(resp, req, **kwargs):
            if isinstance(hop, requests.Response) and (
                self.get_redirect_target(hop) is not None
            ):
                self._release_unread(hop)
            yield hop

    def rebuild_auth(
        self,
        prepared_request: requests.PreparedRequest,
        response: requests.Response,
    ) -> None:
        super().rebuild_auth(prepared_request, response)
        previous_url = response.request.url
        if (
            previous_url
            and prepared_request.url
            and self.should_strip_auth(previous_url, prepared_request.url)
        ):
            prepared_request.headers.pop('X-API-Key', None)


def _read_capped(response: requests.Response, url: str) -> requests.Response:
    r"""Read a streamed response body under a cap, then reattach it.

    The client package no longer reads a response body, so this cap lives here
    alone. This side alone parses what a NIM returns. ``requests`` inflates
    ``Content-Encoding: gzip`` without a ratio limit, so an unbounded read lets
    a small compressed body use hundreds of megabytes of client memory before
    anything is parsed.

    The bytes are put back on the response rather than returned, because this
    client hands the ``requests.Response`` itself to its callers, who read it
    with ``.json()``/``.text``/``.content``. Filling ``_content`` is exactly
    what ``Response.content`` does on first access, so a capped response is
    indistinguishable from an eagerly read one, including for a second pass
    through ``iter_content``, which replays the stored bytes.

    Args:
        response: The streamed response.
        url: The requested URL, named in the error.
    """
    body = bytearray()
    try:
        for chunk in response.iter_content(_RESPONSE_CHUNK_BYTES):
            if len(body) + len(chunk) > _MAX_RESPONSE_BYTES:
                raise InvalidResponseError(
                    f'Response body from {redact_url(url)!r} exceeds the '
                    f'{_MAX_RESPONSE_BYTES} byte limit'
                )
            body.extend(chunk)
    finally:
        response.close()
    response._content = bytes(body)
    response._content_consumed = True  # type: ignore[attr-defined]
    return response


def _json_or_none(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def capped_body(text: str | None) -> str:
    r"""A one-line, length-capped view of a response body, so a multi-megabyte
    error page cannot become a multi-megabyte exception message.
    """
    if not text:
        return ''
    text = ' '.join(text.split())
    if len(text) <= _MAX_BODY_SNIPPET:
        return text
    return f'{text[:_MAX_BODY_SNIPPET]}... [truncated, {len(text)} chars total]'


def _snippet(response: requests.Response | None) -> str:
    return capped_body(response.text if response is not None else None)


def _raise_init_error(url: str, exc: BaseException) -> NoReturn:
    r"""Translate an init-time requests exception into a user-facing error."""
    # The driver echoes the URL it was given back into its own message, so the
    # quoted cause is scrubbed as well as the URL this function was handed.
    shown = redact_url(url)
    cause = scrub_userinfo(str(exc))
    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response
        status_code = response.status_code if response is not None else None
        body = _snippet(response)
        if status_code in _AUTH_STATUS_CODES:
            raise AuthenticationError(
                f'Client authentication failed for {shown!r}. If the NIM is '
                'behind an authenticating gateway, pass a matching '
                f"'api_key'. HTTP {status_code}: {body}"
            ) from exc
        raise ClientInitializationError(
            f'NIM at {shown!r} returned HTTP {status_code} while '
            f'initializing the client. Response: {body}'
        ) from exc
    if isinstance(exc, requests.exceptions.Timeout):
        raise NimTimeoutError(
            f'Timed out connecting to the NIM at {shown!r}. '
            f'Original error: {cause}'
        ) from exc
    if isinstance(exc, requests.exceptions.ConnectionError):
        raise NimUnreachableError(
            f'Could not connect to the NIM at {shown!r}. Verify the '
            f'server is running and the URL is correct. Original error: '
            f'{cause}'
        ) from exc
    raise ClientInitializationError(
        f'Failed to initialize Kumo Relational client against {shown!r}. '
        f'Exception: {cause}'
    ) from exc


class NimClient:
    r"""Speaks the prediction contract to one endpoint.

    The transport under the relational engine: it holds the pooled
    session, sends requests with the API key attached when there is one,
    and retries what is worth retrying. A contract request comes back
    whatever its status, because deciding that a status is an error
    belongs to the layer above; the probes this makes while connecting
    raise for themselves.
    """

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        verify_ssl: bool = True,
        timeout: float | None = None,
        max_retries: int = 3,
    ) -> None:
        r"""Create a client for requests to a Universal TFM NIM.

        NIMs are unauthenticated by contract. ``api_key`` is optional and only
        needed when a deployment adds its own authenticating gateway; when
        given it is sent as the ``X-API-Key`` header, otherwise no auth header
        is set.

        ``timeout`` bounds each individual request attempt, in seconds.
        ``None`` (the default) leaves requests unbounded, so a hung NIM blocks
        the caller until the connection drops.

        ``max_retries`` bounds the transport-level retries of a transient
        failure (408/429/5xx, or a connection that never established); ``0``
        disables them. It is what ``SDMClient(max_retries=...)`` forwards, so
        the knob reaches this path rather than being dropped for a fixed policy
        of its own. It is separate from ``predict(num_retries=...)``, which
        retries the prediction call itself at the application level and is what
        covers a read timeout.

        ``POST /v1/sessions`` is held out of the retries: creating a session
        pins the context in the worker's memory before the response is written,
        so a re-sent create leaves one orphaned pinned context per attempt that
        no ``session_id`` can release. ``requests`` resolves an adapter by
        longest matching URL prefix, so the ordinary policy is re-mounted on
        the replayable routes below it.
        """
        _validate_url(url, api_key)
        self._url = url
        self._api_key = api_key
        self._verify_ssl = verify_ssl
        self._timeout = timeout
        self._max_retries = max_retries

        http_adapter = HTTPAdapter(max_retries=_build_retry(max_retries))
        session = _Session()
        session.mount('http://', http_adapter)
        session.mount('https://', http_adapter)
        session.mount(
            url + _SESSIONS_PATH,
            HTTPAdapter(
                max_retries=_build_retry(max_retries, retry_post=False)
            ),
        )
        session.mount(url + _SESSIONS_PATH + '/', http_adapter)
        self._session = session
        if self._api_key:
            self._session.headers.update({'X-API-Key': self._api_key})

    def close(self) -> None:
        r"""Closes the underlying HTTP session and its pooled connections."""
        self._session.close()

    def authenticate(self) -> None:
        r"""Verify the endpoint is a reachable Universal TFM NIM.

        NIMs are unauthenticated by contract. When a deployment fronts the NIM
        with an authenticating gateway, a 401/403 from either probe surfaces as
        an authentication error rather than being masked as a missing model.
        """
        ready = self._probe('/v1/health/ready')
        ready_data = _json_or_none(ready)
        ready_status = (
            str(ready_data.get('status', '')).lower()
            if isinstance(ready_data, dict)
            else ''
        )
        ready_check = (
            str(ready_data.get('check', '')).lower()
            if isinstance(ready_data, dict)
            else ''
        )
        if not (
            ready_status == 'ready'
            or (ready_status == 'healthy' and ready_check == 'ready')
        ):
            raise ValueError(
                f'NIM at {redact_url(self._url)!r} is not ready. '
                f"'/v1/health/ready' "
                f'returned: {ready_data!r}'
            )

        models = self._probe('/v1/models')
        data = _json_or_none(models)
        advertised = isinstance(data, dict) and any(
            isinstance(model, dict) and model.get('id') == 'kumo-relational'
            for model in data.get('data', [])
        )
        if not advertised:
            raise ValueError(
                f'Endpoint {redact_url(self._url)!r} did not advertise '
                "the 'kumo-relational' "
                "model at '/v1/models'. Point the client at a Universal TFM NIM "
                'serving KumoRelational.'
            )

    def _probe(self, path: str) -> requests.Response:
        try:
            response = self._send(
                'GET', self._url + path, timeout=_PROBE_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            _raise_init_error(self._url, e)

    def _request(self, endpoint: Endpoint, **kwargs: Any) -> requests.Response:
        r"""Send an HTTP request to the specified endpoint."""
        endpoint_str = endpoint.get_path()
        if endpoint.method == HTTPMethod.GET:
            return self._get(endpoint_str, **kwargs)
        if endpoint.method == HTTPMethod.POST:
            return self._post(endpoint_str, **kwargs)
        if endpoint.method == HTTPMethod.DELETE:
            return self._delete(endpoint_str, **kwargs)
        # HTTPMethod is closed and every member is handled above, so this is
        # unreachable today. Stated as an exhaustiveness check so that adding a
        # member without extending the dispatch fails type checking rather than
        # falling through and returning None.
        assert_never(endpoint.method)

    def _send(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        r"""Issues one request and reads its body under a size cap.

        Every request this client makes goes through here, so the cap cannot be
        missed by a route that was added later. ``stream=True`` is an internal
        detail: :func:`_read_capped` re-attaches the body, so the response
        handed back behaves exactly like an eagerly read one.
        """
        kwargs.setdefault('timeout', self._timeout)
        response = self._session.request(
            method, url, verify=self._verify_ssl, stream=True, **kwargs
        )
        return _read_capped(response, url)

    def _get(self, endpoint: str, **kwargs: Any) -> requests.Response:
        return self._send('GET', self._format_endpoint_url(endpoint), **kwargs)

    def _post(self, endpoint: str, **kwargs: Any) -> requests.Response:
        return self._send('POST', self._format_endpoint_url(endpoint), **kwargs)

    def _delete(self, endpoint: str, **kwargs: Any) -> requests.Response:
        return self._send(
            'DELETE', self._format_endpoint_url(endpoint), **kwargs
        )

    def _format_endpoint_url(self, endpoint: str) -> str:
        if not endpoint.startswith('/'):
            raise ValueError("Endpoint path must start with '/'")
        return f'{self._url}{endpoint}'
