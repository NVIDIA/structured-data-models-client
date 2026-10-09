# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Rust core's binding, built from rust/bindings/sdm-py.

Control messages cross as ``bytes`` both ways, each with the attachments it
names in a ``dict`` by id. An attachment is ``bytes`` or Arrow data: in, any
object with ``__arrow_c_stream__``; out, an :class:`ArrowStream`. Only
``sdm_client._rust`` imports this module, and only on a core route.
"""

from collections.abc import Mapping
from typing import Literal, Protocol, final

#: The codes an error of the core carries, the contract's ``ErrorCode``, in
#: its order; each kind a code may carry is fixed in ``sdm-contract``
#: (``ErrorCode::kinds``).
_Code = Literal[
    'INVALID_REQUEST',
    'UNSUPPORTED_REQUEST',
    'INVALID_RESPONSE',
    'INVALID_CONFIGURATION',
    'TRANSPORT_ERROR',
    'INTERNAL_ERROR',
    'UNKNOWN_MODEL',
    'MISSING_EXTRA',
    'INVALID_SCHEMA',
    'SESSIONS_UNSUPPORTED',
    'UNSUPPORTED_TYPE',
    'VALUE_OUT_OF_RANGE',
    'PROVIDER_SCHEMA_MISMATCH',
    'PROVIDER_ERROR',
    'UNKNOWN_REPLY',
    'REPLY_MISSING',
    'WRONG_REPLY',
    'RUN_FINISHED',
    'RUN_CANCELLED',
    'RUN_POISONED',
    'RUN_BUSY',
    'INVALID_MESSAGE',
]
#: The kinds of an error of the core, the contract's ``ErrorKind``.
_Kind = Literal[
    'invalid_request',
    'unsupported',
    'not_found',
    'data_source',
    'model_endpoint',
    'cancelled',
    'protocol',
    'internal',
]

#: The contract version this build speaks.
CONTRACT_VERSION: int
#: Whether this build has the test-only operations; the shipped wheel has not.
TEST_OPS: bool
#: Every code an error of the core may carry, in the contract's order.
ERROR_CODES: tuple[_Code, ...]

class _ArrowStreamExportable(Protocol):
    def __arrow_c_stream__(
        self, requested_schema: object | None = None
    ) -> object: ...

@final
class ArrowStream:
    """Arrow data the core gave, exported through the PyCapsule interface."""

    def __arrow_c_stream__(
        self, requested_schema: object | None = None
    ) -> object: ...
    def __arrow_c_schema__(self) -> object: ...

#: An attachment a caller sends.
_AttachmentIn = bytes | _ArrowStreamExportable
#: An attachment the core gives.
_AttachmentOut = bytes | ArrowStream
#: A control message the core wrote, with the attachments it names.
_Outgoing = tuple[bytes, dict[int, _AttachmentOut]]

class CoreError(Exception):
    """An error the core returned, as its ``error`` control message."""

    kind: _Kind
    code: _Code
    #: Whether the same call may succeed if tried again; carried from
    #: provider and transport errors, false otherwise.
    retryable: bool
    control: bytes
    attachments: dict[int, _AttachmentOut]

@final
class Run:
    """One core run. A call while another is in the run raises ``RUN_BUSY``."""

    @staticmethod
    def start(
        message: bytes, attachments: Mapping[int, _AttachmentIn]
    ) -> Run: ...
    def next(self) -> _Outgoing: ...
    def reply(
        self, message: bytes, attachments: Mapping[int, _AttachmentIn]
    ) -> None: ...
    def cancel(self) -> None: ...
    def take_attachment(self, id: int) -> _AttachmentOut | None: ...
    def take_warnings(self) -> _Outgoing | None: ...

def reset_log_cache() -> None:
    """Forget the logger levels the logging bridge cached."""

def is_posix_zone(text: str) -> bool:
    """Whether a run takes ``text`` as a POSIX ``TZ`` local zone."""
