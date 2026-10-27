# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local native column/error owners for fixed frame domain values.

Core-selected source bindings name
native dtype owners retained at fresh conversion; caller Schema metadata is
never reread here. Graph policy, sampling, normalization and error policy
remain in Rust. No exception is displayed while capturing native facts.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType, TracebackType
from typing import Any, Iterator, NoReturn, cast
import unicodedata

import pandas as pd

from sdm_client._rust.codec import relation_name
from sdm_client.errors import StructuredDataError


@dataclass(frozen=True)
class NativeClass:
    """Raw native ancestry metadata, without importing an engine runtime.

    Qualified names describe classes; they are not proof of canonical class
    identity and must not authorize arbitrary constructors or coercion.
    """

    module: str
    name: str


@dataclass(frozen=True)
class NativeFailureFacts:
    ancestry: tuple[str, ...]
    classes: tuple[NativeClass, ...]
    deepest_module: str | None
    relational_error: bool
    python_unicode_database: str

    def as_wire(self) -> dict[str, Any]:
        return {
            'builtin': {
                'ancestry': [
                    name
                    for name in self.ancestry
                    if name not in ('key_error', 'structured_data_error')
                ]
            },
            'key_error': 'key_error' in self.ancestry,
            'structured_data_error': 'structured_data_error' in self.ancestry,
            'relational_error': self.relational_error,
            'deepest_module': self.deepest_module,
            'python_unicode_database': self.python_unicode_database,
        }


@dataclass(frozen=True, repr=False, eq=False)
class NativeFrameFailure:
    """The exact local exception, independent of an encoded failure value.

    Retaining it does not call str/repr, read args, choose an SDK error code,
    inspect message phrases or append sanitization advice. The eventual core
    guard must select any native text observation at its actual boundary.
    """

    error: Exception
    facts: NativeFailureFacts
    traceback: TracebackType | None
    cause: BaseException | None
    context: BaseException | None
    suppress_context: bool

    @classmethod
    def capture(
        cls, error: Exception, relational_type: type[Exception] | None = None
    ) -> NativeFrameFailure:
        traceback = BaseException.__getattribute__(error, '__traceback__')
        deepest = traceback
        while deepest is not None and deepest.tb_next is not None:
            deepest = deepest.tb_next
        ancestry = tuple(
            name
            for kind, name in (
                (SyntaxError, 'syntax_error'),
                (ValueError, 'value_error'),
                (TypeError, 'type_error'),
                (MemoryError, 'memory_error'),
                (RecursionError, 'recursion_error'),
                (KeyError, 'key_error'),
                (StructuredDataError, 'structured_data_error'),
            )
            if isinstance(error, kind)
        )
        classes = tuple(
            NativeClass(
                type.__getattribute__(kind, '__module__'),
                type.__getattribute__(kind, '__qualname__'),
            )
            for kind in type.__getattribute__(type(error), '__mro__')
        )
        return cls(
            error,
            NativeFailureFacts(
                ancestry,
                classes,
                None
                if deepest is None
                else deepest.tb_frame.f_globals.get('__name__'),
                relational_type is not None
                and isinstance(error, relational_type),
                unicodedata.unidata_version,
            ),
            traceback,
            BaseException.__getattribute__(error, '__cause__'),
            BaseException.__getattribute__(error, '__context__'),
            BaseException.__getattribute__(error, '__suppress_context__'),
        )

    def read_message(self) -> str:
        """One reached native observation, deliberately neither cached nor caught."""
        with self.active():
            return str(self.error)

    def read_type_name(self) -> str:
        """Observe the native class name only when selected for presentation."""
        with self.active():
            return type(self.error).__name__

    @contextmanager
    def active(self) -> Iterator[None]:
        """Observe within the actual preserved native exception context.

        Raising a stored error under an incidental core decoding exception
        would replace its context. Restore the original native traceback and
        chain before the observation, including when that observation fails.
        This chooses no guard policy and catches no observation exception.
        """
        chain = {
            name: BaseException.__getattribute__(self.error, name)
            for name in (
                '__traceback__',
                '__cause__',
                '__context__',
                '__suppress_context__',
            )
        }
        try:
            raise self.error
        except Exception:
            for name, value in chain.items():
                BaseException.__setattr__(self.error, name, value)
            yield


@dataclass(frozen=True, repr=False, eq=False)
class NativeColumn:
    """An actual successful whole-column native result, not an Arrow remake."""

    values: pd.Series

    def arrow(self) -> tuple[Any, str]:
        """Export the actual strings, retaining ordinary Arrow buffers.

        A successful native Python string result can contain surrogate
        points that Arrow UTF8 cannot hold. Only that export case selects
        the explicitly described lossless binary representation; a native
        constructor/astype failure has already stopped before this method.
        """
        import pyarrow as pa

        try:
            array = pa.array(self.values)
        except UnicodeEncodeError:
            array = pa.array(
                [
                    value.encode('utf-8', 'surrogatepass')
                    if isinstance(value, str)
                    else None
                    for value in self.values.array
                ],
                type=pa.large_binary(),
            )
            return pa.table({'values': array}), 'utf8_surrogatepass'
        if pa.types.is_null(array.type):
            array = array.cast(pa.string())
        return pa.table({'values': array}), 'utf8'

    def text_facts(self, identity_input: str) -> dict[str, Any]:
        """Observe this actual result's native class/storage and fixed kernel.

        The core supplies the fixed probe input and selects the casing
        profile. No user text is lowered or clipped by this observer.
        """
        dtype = self.values.dtype
        if type(dtype) is pd.StringDtype:
            target = (
                'pandas_string_python'
                if dtype.storage == 'python'
                else 'pandas_string_pyarrow'
            )
        else:
            target = 'numpy_unicode'
        identity = None
        if target == 'pandas_string_pyarrow':
            import pyarrow as pa
            import pyarrow.compute as pc

            identity = pc.utf8_lower(
                pa.array([identity_input], type=pa.string())
            )[0].as_py()
        return {
            'target': target,
            'python_unicode_database': unicodedata.unidata_version,
            'arrow_simple_identity': identity,
        }


class NativeFrameColumns:
    """Native dtype lifetime companion to one materializer's core graph owner.

    This owns no caller Schema, frame copy, mutable neighbor stream or task
    continuation. The exact frozen core graph class remains binding-owned.
    Construct this companion from the fresh converter's ordered native types
    and retain it with that graph even if a later warning/conversion fails.
    """

    def __init__(self, columns: Mapping[str, Mapping[str, Any]]) -> None:
        self._columns = MappingProxyType(
            {
                source: MappingProxyType(dict(types))
                for source, types in columns.items()
            }
        )

    def dtype(self, binding: Mapping[str, Any]) -> Any:
        """Read exactly the source Relation/column resolved by the core graph."""
        source = relation_name(binding['source'])
        return self._columns[source][binding['column']]

    def prediction(
        self, binding: Mapping[str, Any], indices: list[Any]
    ) -> NativeColumn | NativeFrameFailure:
        """Execute the reached whole constructor with the retained dtype owner.

        Failed construction does not start diagnostics or display anything.
        Those positions require their core-selected constructor outcome and
        are distinct from complete-task and per-batch reference conversion.
        """
        dtype = self.dtype(binding)
        try:
            return NativeColumn(pd.Series(indices, dtype=dtype))
        except Exception as error:
            return NativeFrameFailure.capture(error)

    def prediction_diagnostic(
        self,
        binding: Mapping[str, Any],
        indices: list[Any],
        failure: NativeFrameFailure,
    ) -> tuple[dict[str, Any], NativeFrameFailure]:
        """Observe the actual failed constructor's stop-at-first diagnostic.

        The native constructor catches TypeError/ValueError before this
        diagnostic. Neither other failures nor an unvisited suffix are
        converted. Rust selects the resulting public wrapper and wording.
        """
        if not isinstance(failure.error, (TypeError, ValueError)):
            return {'outcome': 'not_reached'}, failure
        dtype = self.dtype(binding)
        with failure.active():
            try:
                first: dict[str, Any] | None = None
                for ordinal, value in enumerate(indices):
                    try:
                        pd.Series([value], dtype=dtype)
                    except (TypeError, ValueError):
                        first = {'ordinal': ordinal, 'native_repr': repr(value)}
                        break
                diagnostic = {
                    'outcome': 'completed',
                    'first': first,
                    'dtype_display': str(dtype),
                }
            except Exception as error:
                reached = NativeFrameFailure.capture(error)
                return {
                    'outcome': 'failed',
                    'failure': reached.facts.as_wire(),
                }, reached
        return diagnostic, failure


def convert_text(selected: pd.Series) -> NativeColumn | NativeFrameFailure:
    """Convert one actual selected native column before Rust normalization.

    Selection and original dtype reconstruction belong to the fixed sampled
    context boundary. There is no row loop, cap, lower, split, casing profile,
    fabricated result after failure or scan of unselected source values.
    """
    try:
        return NativeColumn(selected.fillna('').astype(str))
    except Exception as error:
        return NativeFrameFailure.capture(error)


def _diagnostic(
    value: dict[str, Any], *, part: dict[str, Any] | None = None
) -> dict[str, Any]:
    """One fixed frame diagnostic value, decoded through the ordinary host."""
    import json

    from sdm_client._rust import codec, results
    from sdm_client._rust.run import run

    operation: dict[str, Any] = {'op': 'prepare_frames', 'input': 1}
    if part is not None:
        operation['part'] = part
    finished = run(operation, {1: codec.attachment(value)})
    answer = results.expect(finished.result, results.FramePrepared)
    return json.loads(finished.take(answer.value))


def _assessment(
    native: NativeFrameFailure,
    origin: str,
    signature: list[int] | None,
    *,
    part: dict[str, Any] | None = None,
    window_outran: bool | None = None,
) -> dict[str, Any]:
    from sdm_client._rust.codec import CodecError

    value = _diagnostic(
        {
            'stage': 'assess_failure',
            'origin': origin,
            'failure': native.facts.as_wire(),
            'signature': signature,
            'window_outran': window_outran,
        },
        part=part,
    )
    if value.get('value') != 'failure_assessment':
        raise CodecError(
            'frame diagnostic result is not a native failure assessment'
        )
    return value['data']


def guard_failure(
    native: NativeFrameFailure,
    *,
    origin: str,
    dropped_summary: str | None = None,
    rows_per_call: int | None = None,
    part: dict[str, Any] | None = None,
) -> NoReturn:
    """Present the closed diagnostic classification selected by the core.

    The host observes exact native values and retains their object identity;
    Rust owns every signature phrase, caller/internal decision, refusal flag,
    graph-detail rule, wrapper and size-remedy selection. No warning or
    exception message is read before the corresponding classified position.
    """
    from sdm_client._rust import errors
    from sdm_client._rust.codec import CodecError

    assessment = _assessment(native, origin, None, part=part)
    category = assessment['category']
    if category == 'pass_through':
        if (
            part is not None
            and 'structured_data_error' in native.facts.ancestry
        ):
            if assessment['part'] is None:
                # The first classification establishes that the original
                # structured exception reaches its forecast wrapper. Ranking
                # already has a complete core-selected sentence and never
                # observes this native detail or its truth conversion.
                with native.active():
                    window_outran = bool(
                        (getattr(native.error, 'details', None) or {}).get(
                            'window_outran_the_data'
                        )
                    )
                assessment = _assessment(
                    native,
                    origin,
                    None,
                    part=part,
                    window_outran=window_outran,
                )
            selected_part = assessment['part']
            if selected_part is None:
                raise CodecError(
                    'native part classification lacks its presentation'
                )
            with native.active():
                first = native.error.args[0].rstrip()
                if first and first[-1] not in '.!?':
                    first += '.'
                suffix = errors._said(
                    selected_part['sentence'],
                    errors.decoded(selected_part['fields']),
                )
                said = f'{first} {suffix}'
                structured = cast(StructuredDataError, native.error)
                carried = structured.__class__.__new__(structured.__class__)
                carried.__dict__.update(native.error.__dict__)
                carried.args = (said, *native.error.args[1:])
                carried.message = said
                carried.__cause__ = native.error
            errors.raise_error(carried)
        errors.raise_native(native.error)
    if category == 'type_error':
        signature = [ord(point) for point in native.read_message()]
        assessment = _assessment(native, origin, signature, part=part)
        category = assessment['category']
    name = None
    refusal = None
    if category in ('internal', 'call_signature'):
        name = native.read_type_name()
    elif category in ('caller_type', 'caller_data'):
        refusal = [ord(point) for point in native.read_message()]
    else:
        raise CodecError(
            'frame diagnostic assessment has no final native category'
        )
    said = [ord(point) for point in native.read_message()]
    selected = None
    with errors.holding(
        frame_native_message=lambda fields: ''.join(
            map(chr, fields['message_points'])
        )
    ):
        try:
            _diagnostic(
                {
                    'stage': 'report_failure',
                    'report': {
                        'assessment': assessment,
                        'native_type': name,
                        'refusal': refusal,
                        'said': said,
                        'dropped_summary': dropped_summary,
                        'rows_per_call': rows_per_call,
                    },
                },
                part=part,
            )
        except StructuredDataError as error:
            selected = error
    if selected is None:
        raise CodecError(
            'frame native error report unexpectedly returned a value'
        )
    selected.__cause__ = native.error
    errors.raise_error(selected)
