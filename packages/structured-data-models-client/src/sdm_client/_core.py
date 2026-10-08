# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The host of the Rust core: how the public classes ask it to do the work.

The core decides; this module only talks to it. It turns what a caller passed
into plain data (frames through ``sdm_core.describe.describe_frames``,
everything else as tagged JSON), drives the core's runs, answering each
statement a run asks for through the source's ``_run`` and each model call
through the client's transport, gives the warnings the core gave on the
caller's line, and raises what the core refused with as this package's
errors, and a panic in it as one with the internal code. Nothing the binding
raises reaches a caller.
"""

from __future__ import annotations

import ast
import importlib
import json
import warnings
from collections.abc import Mapping
from typing import Any

from sdm_core import _native
from sdm_core.answers import HeldValues, decode
from sdm_core.describe import describe_frames

from sdm_client.core.warnings import caller_stacklevel

#: The packages of the old implementation, which the core never imports.
_PORTED = ('predictive_query', 'sdm_relational_engine')


# --- Warnings ---------------------------------------------------------------------


def _category(qualified: str) -> type[Warning]:
    r"""The warning class a core warning names."""
    if qualified == 'sdm_relational_engine.rfm.graph.ViewConversionWarning':
        from sdm_client.core.warnings import ViewConversionWarning

        return ViewConversionWarning
    module, _, name = qualified.rpartition('.')
    if module.split('.')[0] in _PORTED:
        return UserWarning
    try:
        found = getattr(importlib.import_module(module or 'builtins'), name)
    except (ImportError, AttributeError):
        return UserWarning
    if isinstance(found, type) and issubclass(found, Warning):
        return found
    return UserWarning


def give(given: Any) -> None:
    r"""Give the warnings the core gave, in its order, on the caller's line."""
    for warning in given or ():
        if isinstance(warning, str):
            continue
        category = _category(warning['category'])
        if (
            category is RuntimeWarning
            and warning['message'] == 'overflow encountered in cast'
        ):
            import numpy as np

            with warnings.catch_warnings(record=True) as caught:
                if warning.get('masked_float32_cast') is True:
                    import pandas as pd

                    pd.Series(
                        [np.finfo(np.float64).max], dtype='Float64'
                    ).astype('float32')
                else:
                    np.array([np.finfo(np.float64).max]).astype('float32')
            for emitted in caught:
                warnings.warn(
                    str(emitted.message),
                    emitted.category,
                    stacklevel=caller_stacklevel(),
                )
            continue
        warnings.warn(
            warning['message'], category, stacklevel=caller_stacklevel()
        )


def give_json(text: str) -> None:
    r"""Give the warnings of a JSON list the core returned."""
    give(json.loads(text))


# --- Errors -----------------------------------------------------------------------


class _Refused(Exception):
    r"""A refusal read back from what the core gave as a value: the
    attributes a ``CoreError`` carries."""

    def __init__(
        self,
        error_class: str | None,
        code: str | None,
        message: str,
        details: Any = None,
        status_code: Any = None,
        warnings: Any = None,
        chain: Any = None,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.code = code
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        self.warnings = warnings or []
        self.chain = chain or []


def refusal(found: Mapping[str, Any]) -> _Refused:
    r"""A refusal the core wrote as JSON (``refusal_json``)."""
    return _Refused(
        found.get('class'),
        found.get('code'),
        found.get('message', ''),
        found.get('details'),
        found.get('status'),
        None,
        found.get('chain'),
    )


def _class_named(qualified: str) -> type[BaseException] | None:
    module, _, name = qualified.rpartition('.')
    if module.split('.')[0] in _PORTED:
        return None
    try:
        found = getattr(importlib.import_module(module or 'builtins'), name)
    except (ImportError, AttributeError):
        return None
    if isinstance(found, type) and issubclass(found, BaseException):
        return found
    return None


def _built(
    qualified: str,
    message: str,
    code: str | None = None,
    details: Mapping[str, Any] | None = None,
    status: Any = None,
) -> BaseException:
    r"""An instance of the class ``qualified`` names, as today's code raises
    it."""
    from sdm_client import errors

    cls = _class_named(qualified)
    details = dict(details or {})
    if cls is None:
        return errors.StructuredDataError(message, code=code, details=details)
    if issubclass(cls, errors.StructuredDataError):
        if cls is errors.UnsupportedRequestError:
            return cls(message, details=details)
        if cls is errors.NimRequestError:
            return cls(
                status if isinstance(status, int) else 0,
                code=code,
                message=message,
                details=details,
            )
        if cls is errors.UnknownModelError:
            return cls(
                str(details.get('model', '')),
                list(details.get('known_models', ())),
            )
        if cls is errors.MissingExtraError:
            return cls(
                str(details.get('extra', '')), str(details.get('package', ''))
            )
        made = cls.__new__(cls)
        errors.StructuredDataError.__init__(
            made, message, code=code, details=details
        )
        if isinstance(status, int) and cls is not errors.StructuredDataError:
            made.status_code = status  # type: ignore[attr-defined]
        return made
    if cls is KeyError:
        # The core's message is today's `str(error)`, which for a `KeyError`
        # is the repr of what it was raised with.
        try:
            raised = ast.literal_eval(message)
        except (ValueError, SyntaxError):
            raised = message
        return KeyError(raised)
    try:
        return cls(message)
    except Exception:  # noqa: BLE001 - a class that will not take a message
        return errors.StructuredDataError(message, code=code, details=details)


def error_of(
    error: Any,
    caused: Mapping[str, BaseException] | None = None,
    references: HeldValues | None = None,
) -> BaseException:
    r"""What the core refused with, as the error today's code raises, with
    the errors today's raises beneath it.

    ``caused`` holds the errors this host met on the core's behalf: the last
    statement a source refused (``query``) and the last model call the
    transport refused (``model``). The core names one of them where today's
    error is raised from it.
    """
    qualified = getattr(error, 'error_class', None)
    message = getattr(error, 'message', None)
    if not isinstance(message, str):
        message = str(error)
    if not isinstance(qualified, str):
        from sdm_client.errors import StructuredDataError

        return StructuredDataError(message, code=getattr(error, 'code', None))
    made = _built(
        qualified,
        message,
        getattr(error, 'code', None),
        decode(getattr(error, 'details', None), references)
        if references is not None
        else getattr(error, 'details', None),
        getattr(error, 'status_code', None),
    )
    beneath = _chain(
        getattr(error, 'chain', None) or [], caused or {}, references
    )
    if beneath is not None:
        made.__cause__ = beneath
    made.__suppress_context__ = True
    return made


def _chain(
    chain: list[Any],
    caused: Mapping[str, BaseException],
    references: HeldValues | None = None,
) -> BaseException | None:
    r"""The errors beneath a refusal, as the core described them, outermost
    first: each a class and message, or the error this host met (``from``)."""
    built: list[BaseException] = []
    for link in chain:
        if not isinstance(link, Mapping):
            continue
        if link.get('_sdm_internal_cause') is not None:
            continue
        source = link.get('from')
        if source is not None:
            met = caused.get(source)
            if met is not None:
                built.append(met)
                # What the host met carries its own chain.
                break
            continue
        qualified = link.get('class') or 'builtins.ValueError'
        if qualified.split('.')[0] in _PORTED:
            # A class of the old implementation, which only it raises.
            continue
        built.append(
            _built(
                qualified,
                link.get('message') or '',
                link.get('code'),
                decode(link.get('details'), references)
                if references is not None
                else link.get('details'),
                link.get('status'),
            )
        )
    for upper, lower in zip(built, built[1:], strict=False):
        if upper.__cause__ is None:
            upper.__cause__ = lower
        upper.__suppress_context__ = True
    return built[0] if built else None


def panicked(error: BaseException) -> BaseException:
    r"""A panic in the core, which is a bug in it and never a refusal of
    what the caller gave, as this package's error with the internal code;
    nothing of the binding is kept beneath it."""
    from sdm_client.errors import StructuredDataError

    made = StructuredDataError(
        f'{error}; sdm-core failed unexpectedly, please report it',
        code='INTERNAL_ERROR',
    )
    made.__suppress_context__ = True
    return made


def call(
    function: Any, *args: Any, references: HeldValues | None = None
) -> Any:
    r"""A binding function's value, its refusal raised as today's error."""
    try:
        return function(*args)
    except _native.CoreError as error:
        found = error
    except _native.CorePanic as error:
        raise panicked(error) from None
    if references is not None and references.error is not None:
        give(getattr(found, 'warnings', None))
        raise references.error from references.error.__cause__
    _raise_from_core(found, references=references)
    raise AssertionError('unreachable')  # pragma: no cover


def _raise_from_core(
    error: Any,
    caused: Mapping[str, BaseException] | None = None,
    references: HeldValues | None = None,
) -> None:
    give(getattr(error, 'warnings', None))
    made = error_of(error, caused, references)
    raise made from made.__cause__


# --- Runs ---------------------------------------------------------------------------


class Driver:
    r"""What a run's steps are answered through: a source's ``_run`` for
    statements, a transport for model calls; and the errors each raised."""

    def __init__(
        self,
        source: Any = None,
        transport: Any = None,
        references: HeldValues | None = None,
    ) -> None:
        self.source = source
        self.transport = transport
        self.references = references
        self.caused: dict[str, BaseException] = {}
        #: How the core was told of the last model call's error.
        self._told: dict[str, Any] | None = None

    def drive(self, run: Any) -> Any:
        r"""The run, driven to its end: its ``done`` value."""
        try:
            while True:
                step = run.next()
                give(step.warnings)
                if step.kind == 'done':
                    return step.value
                if step.kind == 'run_queries':
                    for query in step.queries:
                        run.submit(query.id, self._ran(query))
                    continue
                run.submit(step.id, self._called(step))
        except _native.CoreError as error:
            found = error
        except _native.CorePanic as error:
            raise panicked(error) from None
        if self.references is not None and self.references.error is not None:
            give(getattr(found, 'warnings', None))
            raise self.references.error from self.references.error.__cause__
        met = self.caused.get('model')
        if met is not None and self._passed_through(found):
            # The core let the call's own error through, as today's code
            # does with anything but a refusal it can explain: the same
            # object, raised again.
            give(getattr(found, 'warnings', None))
            raise met
        _raise_from_core(found, self.caused, self.references)
        raise AssertionError('unreachable')  # pragma: no cover

    def _passed_through(self, error: Any) -> bool:
        told = self._told
        if told is None or getattr(error, 'chain', None):
            return False
        return bool(
            getattr(error, 'error_class', None) == told['class']
            and getattr(error, 'message', None) == told['message']
            and getattr(error, 'code', None) == told['code']
            and getattr(error, 'status_code', None) == told['status']
        )

    def _ran(self, query: Any) -> Any:
        try:
            table = self.source._run(query.sql, query.params or None)
        except Exception as error:  # noqa: BLE001 - handed to the core
            self.caused['query'] = error
            sqlstate = getattr(error, 'sqlstate', None)
            errno = getattr(error, 'errno', None)
            return _native.Outcome.error(
                sqlstate if isinstance(sqlstate, str) else None,
                errno if isinstance(errno, int) else None,
                str(error),
            )
        return _native.Outcome.data(table)

    def _called(self, step: Any) -> Any:
        transport = self.transport
        try:
            if step.endpoint == 'predict':
                answer = transport.predict(self._body(step.body))
            elif step.endpoint == 'session_create':
                answer = transport.create_session(self._body(step.body))
            elif step.endpoint == 'session_predict':
                answer = transport.session_predict(
                    step.session_id, self._body(step.body)
                )
            else:
                answer = transport.delete_session(step.session_id)
        except Exception as error:  # noqa: BLE001 - handed to the core
            self.caused['model'] = error
            status = getattr(error, 'status_code', None)
            code = getattr(error, 'code', None)
            message = getattr(error, 'message', None)
            details = getattr(error, 'details', None)
            cls = type(error)
            told = {
                'class': f'{cls.__module__}.{cls.__qualname__}',
                'status': status if isinstance(status, int) else None,
                'code': code if isinstance(code, str) and code else None,
                'message': message if isinstance(message, str) else str(error),
                'details': plain(dict(details))
                if isinstance(details, Mapping)
                else {},
            }
            self._told = told
            return _native.Outcome.model_error(json.dumps(told))
        return _native.Outcome.response(json.dumps(answer))

    def _body(self, body: str) -> Any:
        value = json.loads(body)
        return (
            value if self.references is None else decode(value, self.references)
        )


# --- Values -------------------------------------------------------------------------


class SequenceValue:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.error: BaseException | None = None

    def __call__(self, operation: str) -> str:
        found: Any
        try:
            if operation == 'iter':
                found = [plain(item, max_depth=100) for item in self.value]
            elif operation == 'list':
                found = [
                    plain(item, max_depth=100) for item in list(self.value)
                ]
            elif operation == 'len':
                found = len(self.value)
            elif operation == 'bool':
                found = bool(self.value)
            else:
                raise ValueError(f'unknown sequence protocol: {operation}')
            return json.dumps(found)
        except BaseException as error:
            self.error = error
            raise


def plain(
    value: Any,
    *,
    max_depth: int | None = None,
    references: HeldValues | None = None,
) -> Any:
    r"""A value a caller passed, as tagged JSON; what has no plain form is
    described by its type, so the core can refuse it as today does."""
    from sdm_core.answers import encode

    try:
        return encode(value, max_depth=max_depth, references=references)
    except TypeError:
        config = _inference_config(value)
        if config is not None:
            return config
        return opaque(value)


def _inference_config(value: Any) -> Any:
    r"""The relational engine's inference-config dataclass, which a caller
    may hand over in place of a mapping, as its class and fields; the core
    reads it as the dataclass does. Recognised by its name and its module's,
    without importing the engine."""
    import dataclasses

    cls = type(value)
    if not (
        dataclasses.is_dataclass(value)
        and cls.__module__.startswith('sdm_relational_engine.')
        and cls.__name__.endswith('InferenceConfig')
    ):
        return None
    return {
        '$inference_config': {
            'class': cls.__name__,
            'fields': {
                f.name: plain(getattr(value, f.name))
                for f in dataclasses.fields(value)
            },
        }
    }


def opaque(value: Any) -> Any:
    r"""A value with no plain form: its type and ``repr``, or its shape."""
    import pandas as pd

    if isinstance(value, pd.Series):
        return {'$object': {'type': 'Series', 'repr': ''}}
    if isinstance(value, Mapping):
        return {'$mapping': type(value).__name__}
    if hasattr(value, '__iter__') and not isinstance(value, str):
        return {'$container': type(value).__name__}
    try:
        shown = repr(value)
    except Exception:  # noqa: BLE001 - a repr that raises shows as its type
        shown = f'<{type(value).__name__}>'
    return {'$object': {'type': type(value).__name__, 'repr': shown}}


def model_arg(model: Any) -> Any:
    r"""A model, as its id, or ``{"$model": CLASS}`` for a handle: the class
    of the registered handle its spec is, or what names an unregistered
    one."""
    from sdm_client.models import MODELS, Model

    if not isinstance(model, Model):
        return plain(model)
    spec = getattr(model, '_spec', None)
    model_id = getattr(spec, 'model_id', None)
    if model_id is not None and MODELS.get(model_id) is spec:
        return {'$model': _HANDLES[model_id]}
    return {'$model': model_id or type(model).__name__}


#: The handle class of each model id.
_HANDLES = {'kumo-tabular': 'KumoTabular', 'kumo-relational': 'KumoRelational'}


class Frames:
    r"""Arrow tables by name, as the core reads them, and their tags."""

    def __init__(
        self, tables: Mapping[str, Any] | None = None, tags: Any = None
    ) -> None:
        self.tables: dict[str, Any] = dict(tables or {})
        self.tags: dict[str, Any] = dict(tags or {})

    def add(self, other: Frames) -> Frames:
        return Frames(
            {**self.tables, **other.tables}, {**self.tags, **other.tags}
        )

    @staticmethod
    def described(frames: Mapping[str, Any]) -> Frames:
        r"""pandas frames, described for the core under their names."""
        if not frames:
            return Frames()
        tables, tags = describe_frames(frames)
        return Frames(tables, json.loads(tags))
