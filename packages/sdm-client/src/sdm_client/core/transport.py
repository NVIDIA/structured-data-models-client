# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, urlparse

import requests
import urllib3.exceptions
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sdm_client.errors import (
    NimRequestError,
    StructuredDataError,
    format_invalid_params,
)

_PREDICTIONS_PATH = '/v1/predictions'
_SESSIONS_PATH = '/v1/sessions'
_HEALTH_READY_PATH = '/v1/health/ready'
# Supplied by the NIM runtime rather than the Universal TFM contract, so an
# endpoint is not obliged to serve it.
_MODELS_PATH = '/v1/models'
_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_FACTOR = 0.5
#: The transient ones. A 500 is not among them: the payload and the model are
#: the same on every attempt, so a server fault reproduces, and the caller
#: mistakes this endpoint reports as a 500 are reclassified from the body as
#: the caller's after three more uploads of the whole gathered context.
_RETRY_STATUS = (429, 502, 503, 504)
_RETRY_AFTER_MAX_SECONDS = 60
_MAX_BODY_SNIPPET = 512
_LOCAL_HOSTS = frozenset({'localhost', '127.0.0.1', '::1'})
# Well above any real prediction response (a full-precision float64 column
# for a million rows is ~25 MB of JSON), but far below what a compressed
# hostile body can inflate to.
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_RESPONSE_CHUNK_BYTES = 1024 * 1024
# Deleting a session is a bookkeeping call the server answers immediately, and
# it runs on cleanup paths where the caller is no longer waiting on a result.
# The full request timeout would let an unreachable endpoint stall a teardown.
_DELETE_TIMEOUT_SECONDS = 10.0
_USERINFO_RE = re.compile(r'(?<=//)[^/@\s]+@')


def redact_url(url: str | None) -> str | None:
    r"""Strip any userinfo from ``url`` so it is safe to log or display.

    ``https://user:token@host/path`` is a legal way to point this client at a
    deployment, and the credential is then in the URL itself rather than in
    ``api_key``. Anywhere a URL reaches an error message or a repr it goes
    through here first, so the credential does not travel with the diagnosis
    into a log aggregator or a bug report.

    Mirrors ``sdm_relational_engine.client.client.redact_url``, which guards
    the same credential on the Kumo Relational path. The two cannot share one
    implementation:
    this package must not import the optional driver, and the package both
    depend on is a SQL-connector package with no URL handling. Keep them in
    step.
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

    The counterpart to :func:`redact_url` for strings that merely *contain* a
    URL rather than being one, chiefly the ``requests`` exception messages
    quoted into transport errors, which echo back whatever URL was requested.
    """
    return _USERINFO_RE.sub('', text)


def _validate_tls_verification(url: str, verify_ssl: bool) -> None:
    """Allow disabled TLS verification only for local development endpoints."""
    if verify_ssl:
        return
    host = (urlparse(url).hostname or '').lower()
    if host in _LOCAL_HOSTS:
        return
    raise StructuredDataError(
        f'TLS certificate verification may be disabled only for a loopback '
        f'endpoint, got {redact_url(url)!r}. Set verify_ssl=True and trust the '
        'endpoint certificate instead.',
        code='INVALID_CONFIGURATION',
    )


def _validate_url(url: str, api_key: str | None) -> None:
    if not isinstance(url, str):
        raise StructuredDataError(
            f'url must be a string, got {type(url).__name__}',
            code='INVALID_CONFIGURATION',
        )
    parsed = urlparse(url)
    shown = redact_url(url)
    if parsed.scheme not in ('http', 'https'):
        raise StructuredDataError(
            f'url must start with http:// or https://, got {shown!r}',
            code='INVALID_CONFIGURATION',
        )
    if not parsed.hostname:
        raise StructuredDataError(
            f'url is missing a host, got {shown!r}',
            code='INVALID_CONFIGURATION',
        )
    # A base URL is a scheme, a host and at most a path prefix. A query or
    # fragment is not a meaningful part of one, and every route is appended to
    # this string. Thus, `http://host?api_key=x` becomes
    # `http://host/?api_key=x/v1/predictions`, which is both a broken URL and a
    # secret in a plaintext request line that redaction never sees.
    if parsed.query or parsed.fragment:
        raise StructuredDataError(
            'url must be a base endpoint without a query string or fragment; '
            'pass a credential as api_key rather than in the URL',
            code='INVALID_CONFIGURATION',
        )

    host = parsed.hostname.lower()
    # Userinfo in the URL is a credential too, and ``requests`` turns it into
    # an ``Authorization: Basic`` header. Checking only ``api_key`` let
    # ``http://user:secret@host`` send that in the clear.
    carries_credential = bool(api_key or parsed.username or parsed.password)
    if (
        carries_credential
        and parsed.scheme == 'http'
        and host not in _LOCAL_HOSTS
    ):
        what = 'an API key' if api_key else 'the credential in this URL'
        raise StructuredDataError(
            f'refusing to send {what} over plaintext HTTP; use an https:// '
            f'URL (or a localhost endpoint), got {shown!r}',
            code='INVALID_CONFIGURATION',
        )


class _Session(requests.Session):
    r"""A ``Session`` that enforces redirect security policy.

    ``requests`` strips only the standard ``Authorization`` header when a
    redirect changes origin; a custom header set on the session is re-sent
    verbatim, so without this the configured endpoint could hand the API key
    to any host it names, defeating ``_validate_url``.

    Redirects are still followed, and this covers the credential only. A
    ``307``/``308`` therefore re-sends the request *body* to the redirect
    target,
    including the caller's whole labelled context table for Kumo Tabular and
    across an ``https`` to ``http`` downgrade, which ``_validate_url`` never
    sees because it runs at construction against the configured URL. That is
    an accepted trade when certificate verification is enabled. When it is
    disabled for a loopback development endpoint, redirect targets must remain
    loopback too, so the request-level ``verify=False`` cannot reach a remote
    host.
    """

    def __init__(self, *, verify_ssl: bool) -> None:
        super().__init__()
        self._verify_ssl = verify_ssl

    @staticmethod
    def _release_unread(resp: object) -> None:
        r"""Give up a redirect hop's socket without reading its body.

        Close rather than drain: the connection is surrendered instead of being
        read to the end so it can be reused. Marking the body consumed is what
        stops ``requests`` reading it afterwards. ``resp.content`` returns the
        empty bytes set here instead of pulling from the socket.
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

        ``requests`` consumes each hop's body (``resp.content``) to release the
        socket, with no cap. This layer never needs those bytes because its
        only
        request reports a status. They are attacker-controlled on
        a misconfigured endpoint.

        Every hop is released, not just the first. ``requests`` resolves the
        rest inside the generator below, reading each one as it loops, so each
        is released as it is yielded, before control returns to that loop.
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
        if prepared_request.url:
            _validate_tls_verification(prepared_request.url, self._verify_ssl)


def _validate_limits(timeout: Any, max_retries: Any) -> None:
    r"""Reject unusable transport limits at construction.

    Without this the offending value first surfaces as a raw ``ValueError``
    from urllib3, on the first request rather than at the call that set it.

    ``inf`` and ``nan`` are rejected explicitly: both compare ``False`` against
    ``<= 0`` and so pass an ordinary positivity test, then fail on the first
    request with ``OverflowError``/``ValueError`` from the socket layer.
    ``inf``
    is the natural way to reach for "no timeout" here, since this constructor
    does not accept ``None``.
    """
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise StructuredDataError(
            f'timeout must be a positive, finite number of seconds, got '
            f'{timeout!r}',
            code='INVALID_CONFIGURATION',
        )
    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or max_retries < 0
    ):
        raise StructuredDataError(
            f'max_retries must be a non-negative integer, got {max_retries!r}',
            code='INVALID_CONFIGURATION',
        )


def _build_retry(
    max_retries: int,
    backoff_factor: float,
    *,
    retry_post: bool = True,
) -> Retry:
    r"""The retry policy this transport's requests run under.

    ``retry_after_max`` caps how long a server-chosen ``Retry-After`` header
    may park the caller; it needs urllib3 >= 2.3, and older versions fall back
    to urllib3's own (6 hour) cap.

    ``retry_post=False`` drops ``POST`` from the retryable methods, which
    switches off the read and status retries for a route whose side effect the
    client cannot reconcile. Connection failures stay retryable either way:
    urllib3 gates read and status retries on the method but not connect ones,
    and a request that never reached the server cannot have had an effect.

    ``DELETE`` is always retryable. Releasing a session is idempotent by
    contract. The NIM answers 204 whether or not the session is still there,
    and the status it returns when it cannot release one right now is a
    `429` whose message asks the caller to retry. Leaving it out meant the
    client raised the one error it documents as worth retrying and then did
    not retry it.
    """
    methods = {'GET', 'DELETE'}
    if retry_post:
        methods.add('POST')
    options: dict[str, Any] = {
        'total': max_retries,
        'connect': max_retries,
        'read': max_retries,
        'status': max_retries,
        'backoff_factor': backoff_factor,
        'status_forcelist': _RETRY_STATUS,
        'allowed_methods': frozenset(methods),
        'respect_retry_after_header': True,
        'raise_on_status': False,
    }
    try:
        return Retry(retry_after_max=_RETRY_AFTER_MAX_SECONDS, **options)
    except TypeError:
        return Retry(**options)


def _timed_out(error: BaseException | None) -> bool:
    r"""Whether a request failure is a timeout, however it reached us.

    urllib3 wraps a read timeout in ``MaxRetryError``, which ``requests``
    surfaces as ``ConnectionError`` rather than ``Timeout``.
    """
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(
            error, (requests.Timeout, urllib3.exceptions.TimeoutError)
        ):
            return True
        error = error.__cause__ or error.__context__
    return False


def _failed_to_reach(
    error: Exception, url: str, timeout: float | None
) -> StructuredDataError:
    r"""One account of a failed request, for every route that can fail.

    A timeout names the knob and the value rather than urllib3's "Max retries
    exceeded", which says the opposite of the truth when none were asked for.
    """
    if _timed_out(error) and timeout is not None:
        return StructuredDataError(
            f'Request to {redact_url(url)} timed out after {timeout:g} '
            f'seconds, which is the timeout this client was given',
            code='TRANSPORT_ERROR',
            details={'timeout': timeout, 'url': redact_url(url)},
        )
    return StructuredDataError(
        f'Request to {redact_url(url)} failed: {scrub_userinfo(str(error))}',
        code='TRANSPORT_ERROR',
        details={'url': redact_url(url)},
    )


def _decoded_object(content: bytes, url: str) -> dict[str, Any]:
    r"""``content`` as the JSON object it is supposed to be.

    In one place because it was in two, and one checked only that it was JSON:
    a 200 carrying an array escaped `models()` as a bare ``AttributeError``.
    A deeply nested body raises `RecursionError`, a `RuntimeError`, which would
    otherwise escape the contract these errors state.
    """
    try:
        body = json.loads(content or b'{}')
    except (ValueError, RecursionError) as error:
        raise StructuredDataError(
            f'{redact_url(url)} did not return JSON',
            code='TRANSPORT_ERROR',
            details={'url': redact_url(url)},
        ) from error
    if not isinstance(body, dict):
        raise StructuredDataError(
            f'Expected a JSON object from {redact_url(url)}, got '
            f'{type(body).__name__}',
            code='TRANSPORT_ERROR',
            details={'url': redact_url(url)},
        )
    return body


def _snippet(text: str) -> str:
    r"""A one-line, length-capped view of a response body, so a multi-megabyte
    HTML error page cannot become a multi-megabyte exception message.
    """
    text = ' '.join(text.split())
    if len(text) <= _MAX_BODY_SNIPPET:
        return text
    return f'{text[:_MAX_BODY_SNIPPET]}... [truncated, {len(text)} chars total]'


def _read_capped(response: requests.Response, url: str) -> bytes:
    r"""Reads a streamed response body, refusing anything past the cap.

    ``requests`` inflates ``Content-Encoding: gzip`` with no ratio limit, so
    an unbounded read lets a small compressed body expand into hundreds of
    megabytes of client memory before anything is parsed.
    """
    body = bytearray()
    for chunk in response.iter_content(_RESPONSE_CHUNK_BYTES):
        if len(body) + len(chunk) > _MAX_RESPONSE_BYTES:
            raise StructuredDataError(
                f'Response body from {redact_url(url)} exceeds the '
                f'{_MAX_RESPONSE_BYTES} byte limit',
                code='TRANSPORT_ERROR',
            )
        body.extend(chunk)
    return bytes(body)


def _path_segment(value: str) -> str:
    r"""Escapes a server-chosen id before it is spliced into a URL path, so a
    hostile or malformed ``session_id`` cannot redirect the call to another
    route.

    ``quote`` leaves a dot alone, because it is unreserved. A resolver then
    removes the only two segments that mean something to it, ``.`` and ``..``.
    Thus, ``..`` turned a scoring call against a pinned
    context into a one-shot prediction elsewhere. Those name no session and
    are refused; ``...`` is an ordinary segment and is escaped like any
    other. An empty id is refused for its own reason: it contributes no
    segment at all, so the path closes up around it.
    """
    if value in {'', '.', '..'}:
        raise StructuredDataError(
            f'{value!r} is not a session id: it does not survive as a path '
            f'segment when the URL is resolved, so the call would be made '
            f'against a different route than the one it names',
            code='INVALID_RESPONSE',
        )
    return quote(value, safe='')


def _build_session(
    url: str,
    api_key: str | None,
    verify_ssl: bool,
    max_retries: int,
    backoff_factor: float,
) -> requests.Session:
    r"""The pooled session, with ``POST /v1/sessions`` held out of the retries.

    Creating a session fits and pins the context in the worker's memory before
    the response is written, so a re-sent ``POST`` leaves one orphaned pinned
    context per attempt: the client keeps only the last ``session_id`` and can
    never ``DELETE`` the earlier ones. That is a side effect the client cannot
    reconcile, unlike ``/v1/predictions``, which the same policy may safely
    replay.

    ``requests`` resolves an adapter by longest matching URL prefix, so the
    stricter policy is mounted on the exact create-session URL and the ordinary
    one re-mounted on the routes below it: scoring against a pinned context and
    releasing one are both replayable and keep the full policy.
    """
    session = _Session(verify_ssl=verify_ssl)
    if api_key:
        session.headers['X-API-Key'] = api_key
    adapter = HTTPAdapter(max_retries=_build_retry(max_retries, backoff_factor))
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.mount(
        url + _SESSIONS_PATH,
        HTTPAdapter(
            max_retries=_build_retry(
                max_retries, backoff_factor, retry_post=False
            )
        ),
    )
    session.mount(url + _SESSIONS_PATH + '/', adapter)
    return session


class Transport:
    r"""The HTTP wire layer to one Universal TFM NIM.

    Owns a pooled ``requests.Session`` with retry/backoff on the transient
    failures :data:`_RETRY_STATUS` names. Not part of the public API surface;
    a ``SDMClient`` holds one of these.

    ``timeout`` bounds each individual attempt, not the call as a whole: a
    request that exhausts ``max_retries`` can take up to
    ``(max_retries + 1) * timeout`` plus backoff.
    """

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        *,
        verify_ssl: bool = True,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
    ) -> None:
        _validate_url(url, api_key)
        _validate_limits(timeout, max_retries)
        _validate_tls_verification(url, verify_ssl)
        self._url = url.rstrip('/')
        self._api_key = api_key
        self._verify_ssl = verify_ssl
        self._timeout = timeout
        self._closed = False
        self._max_retries = max_retries
        self._session = _build_session(
            self._url, api_key, verify_ssl, max_retries, backoff_factor
        )

    @property
    def url(self) -> str:
        return self._url

    @property
    def api_key(self) -> str | None:
        return self._api_key

    @property
    def verify_ssl(self) -> bool:
        return self._verify_ssl

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def close(self) -> None:
        self._closed = True
        self._session.close()

    def _require_open(self) -> None:
        if self._closed:
            raise StructuredDataError(
                'this client is closed; construct a new SDMClient',
                code='INVALID_CONFIGURATION',
            )

    def health_ready(
        self,
        *,
        timeout: float | None = None,
        retry: bool = True,
        raise_on_not_ready: bool = False,
    ) -> bool:
        self._require_open()
        request_timeout = self._timeout if timeout is None else timeout
        _validate_limits(request_timeout, 0)
        url = self._url + _HEALTH_READY_PATH
        session = (
            self._session
            if retry
            else _build_session(
                self._url,
                self._api_key,
                self._verify_ssl,
                max_retries=0,
                backoff_factor=_DEFAULT_BACKOFF_FACTOR,
            )
        )
        try:
            # Streamed and closed unread: only the status matters, and the
            # body is attacker-controlled on a misconfigured endpoint.
            with session.get(
                url,
                timeout=request_timeout,
                verify=self._verify_ssl,
                stream=True,
            ) as response:
                if response.status_code != 200 and raise_on_not_ready:
                    raise StructuredDataError(
                        f'Readiness check at {redact_url(url)} returned '
                        f'HTTP {response.status_code}',
                        code='TRANSPORT_ERROR',
                        details={
                            'status_code': response.status_code,
                            'url': redact_url(url),
                        },
                    )
                return response.status_code == 200
        except requests.RequestException as error:
            if _timed_out(error):
                raise StructuredDataError(
                    f'Readiness check at {redact_url(url)} timed out after '
                    f'{request_timeout:g} seconds',
                    code='TRANSPORT_ERROR',
                    details={
                        'timeout': request_timeout,
                        'url': redact_url(url),
                    },
                ) from error
            raise _failed_to_reach(error, url, None) from error
        finally:
            if session is not self._session:
                session.close()

    def list_models(self, *, timeout: float | None = None) -> list[str]:
        r"""The model ids the endpoint advertises.

        ``/v1/models`` is supplied by the NIM runtime and is **not** one of the
        contract's four paths, so an endpoint may not serve it at all. That is
        reported as a transport error naming the path rather than being
        answered from this client's own list of models, which is what the
        previous generation did. That meant it claimed a model the
        endpoint did not serve.
        """
        self._require_open()
        request_timeout = self._timeout if timeout is None else timeout
        _validate_limits(request_timeout, 0)
        url = self._url + _MODELS_PATH
        try:
            with self._session.get(
                url,
                timeout=request_timeout,
                verify=self._verify_ssl,
                stream=True,
            ) as response:
                content = _read_capped(response, url)
                if response.status_code != 200:
                    raise _to_nim_error(response.status_code, content)
        except requests.RequestException as error:
            raise _failed_to_reach(error, url, request_timeout) from error

        entries = _decoded_object(content, url).get('data')
        if entries is None:
            entries = []
        if not isinstance(entries, list):
            raise StructuredDataError(
                f'Expected a list of models under "data" from '
                f'{redact_url(url)}, got {type(entries).__name__}',
                code='TRANSPORT_ERROR',
                details={'url': redact_url(url)},
            )
        return [
            str(entry['id'])
            for entry in entries
            if isinstance(entry, dict) and entry.get('id')
        ]

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(_PREDICTIONS_PATH, payload)

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        r"""Pins a context server-side; ``payload`` carries the context-only
        sections of a prediction request.
        """
        return self._post(_SESSIONS_PATH, payload)

    def session_predict(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        r"""Scores rows against a pinned context; ``payload`` carries only the
        per-call sections.
        """
        return self._post(
            f'{_SESSIONS_PATH}/{_path_segment(session_id)}/predictions', payload
        )

    def delete_session(self, session_id: str) -> None:
        r"""Releases a pinned context. Idempotent server-side: the NIM answers
        204 whether or not the session is still there.
        """
        self._require_open()
        url = f'{self._url}{_SESSIONS_PATH}/{_path_segment(session_id)}'
        request_timeout = min(self._timeout, _DELETE_TIMEOUT_SECONDS)
        try:
            with self._session.delete(
                url,
                timeout=request_timeout,
                verify=self._verify_ssl,
                stream=True,
            ) as response:
                status_code = response.status_code
                content = _read_capped(response, url)
        except requests.RequestException as error:
            raise _failed_to_reach(error, url, request_timeout) from error
        if status_code >= 400:
            raise _to_nim_error(status_code, content)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_open()
        url = self._url + path
        # Encode before sending, so a value with no JSON form is named here
        # rather than raising a TypeError from inside `requests` that mentions
        # neither this client nor the offending key.
        _check_encodable(payload)
        try:
            body = json.dumps(payload, allow_nan=False)
        except RecursionError as error:  # pragma: no cover - see the check
            raise StructuredDataError(
                'this request is nested too deeply to encode as JSON',
                code='INVALID_REQUEST',
            ) from error
        except (TypeError, ValueError) as error:
            raise StructuredDataError(
                f'this request carries a value that cannot be sent as JSON: '
                f'{error}',
                code='INVALID_REQUEST',
            ) from error
        try:
            with self._session.post(
                url,
                # Encoded here rather than handed over as `str`: requests
                # gives a str body to httplib, which encodes latin-1, so this
                # is only safe today because `json.dumps` defaults to
                # `ensure_ascii=True`. Not a default worth depending on.
                data=body.encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                timeout=self._timeout,
                verify=self._verify_ssl,
                stream=True,
            ) as response:
                status_code = response.status_code
                content = _read_capped(response, url)
        except requests.RequestException as error:
            raise _failed_to_reach(error, url, self._timeout) from error
        if status_code >= 400:
            raise _to_nim_error(status_code, content)
        return _decoded_object(content, url)


#: Deeper than any real request and far below the interpreter's own limit,
#: so the walk below reports rather than exhausting the stack.
_MAX_PAYLOAD_DEPTH = 200


def _check_encodable(payload: Any) -> None:
    r"""Refuse a payload that nests too deeply or refers back to itself.

    Not left to ``json.dumps`` raising ``RecursionError``. Whether it does is
    an implementation detail of the encoder: CPython 3.12 raises on a 30,000
    level mapping and 3.14 encodes the same value without complaint, so a
    guard written around the exception protects one interpreter and not the
    next. This package supports both.

    Iterative, because a recursive check would hit the same limit it exists to
    report.
    """
    stack: list[tuple[Any, int, tuple[int, ...]]] = [(payload, 0, ())]
    while stack:
        value, depth, parents = stack.pop()
        if not isinstance(value, (Mapping, list, tuple)):
            continue
        if id(value) in parents:
            raise StructuredDataError(
                'this request contains a value that refers back to itself, '
                'so it cannot be sent as JSON',
                code='INVALID_REQUEST',
            )
        if depth > _MAX_PAYLOAD_DEPTH:
            raise StructuredDataError(
                f'this request nests more than {_MAX_PAYLOAD_DEPTH} levels '
                f'deep, so it cannot be sent as JSON',
                code='INVALID_REQUEST',
            )
        nested = (*parents, id(value))
        items = value.values() if isinstance(value, Mapping) else value
        for item in items:
            stack.append((item, depth + 1, nested))


_THE_CALLERS_TO_FIX = (
    ('failed to collect any context examples', 'EMPTY_CONTEXT'),
)

_LEAKED_EXCEPTION = re.compile(
    r'\b(?:failed unexpectedly with\s+)?'
    r'[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\s*:\s*'
)


def _diagnosis_in(reported: Any) -> tuple[str, str] | None:
    r"""The wording and stable code of a failure the caller can act on.

    ``_THE_CALLERS_TO_FIX`` holds the failures the endpoint reports as its own
    that describe the request instead. Each is the caller's to fix, and an
    orchestrator told a server faulted will retry one that can only fail the
    same way.

    They are matched on the endpoint's wording, which is the only signal the
    response gives: the status says server, the code says internal, and only
    the sentence says otherwise. A build that rewords one reverts to reporting
    it the way the endpoint did, rather than misreporting anything.

    Only a string is read. A body whose ``detail`` is an object is not the
    shape the contract describes, and rendering one to text could match this
    on a debug field the endpoint never meant as its diagnosis.
    """
    if not isinstance(reported, str):
        return None
    said = reported.casefold()
    for wording, code in _THE_CALLERS_TO_FIX:
        if wording in said:
            return wording, code
    return None


def _the_diagnosis_alone(message: str, wording: str) -> str:
    r"""The part of ``message`` that says what the caller got wrong.

    What comes before it is the endpoint describing its own failure: which
    prediction, at which address, and the Python type that reached its
    response, matched by ``_LEAKED_EXCEPTION``. None of that is the caller's
    to act on, and a type from inside the server reads as a crash, which is
    the impression being corrected.
    """
    found = message.casefold().find(wording)
    kept = message[found:].strip() if found >= 0 else message
    return _LEAKED_EXCEPTION.sub('', kept).strip() or message


def _to_nim_error(status_code: int, content: bytes) -> NimRequestError:
    r"""The error an endpoint's failed response means.

    A failure the caller can act on is reported as theirs whatever status it
    arrived under, with what the endpoint called it kept in ``details``: that
    is worth knowing even where it was wrong.
    """
    try:
        body = json.loads(content)
    except (ValueError, RecursionError):
        body = {}
    if not isinstance(body, dict):
        body = {}
    code = body.get('code')
    reported = (
        body.get('detail')
        or body.get('title')
        or content.decode('utf-8', 'replace')
    )
    message = (
        _snippet(str(reported))
        if reported
        else (f'NIM request failed with status {status_code}')
    )
    # A validation failure's top-level detail is often only "Request validation
    # failed."; the per-field diagnosis is in invalid_params, so render it into
    # the message rather than leaving it for the caller to dig out of details.
    message += format_invalid_params(body.get('invalid_params'))
    details = {
        key: value
        for key, value in body.items()
        if key not in ('code', 'detail', 'title', 'type', 'status', 'instance')
    }
    found = _diagnosis_in(reported) if status_code >= 500 else None
    if found is not None:
        wording, diagnosis = found
        details = {**details, 'endpoint_code': code, 'diagnosis': diagnosis}
        code = 'INVALID_REQUEST'
        message = _the_diagnosis_alone(message, wording)
    return NimRequestError(
        status_code,
        code=code,
        message=message,
        details=details,
    )
