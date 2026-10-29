# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Rust core's binding, built from rust/bindings/sdm-py.

Control messages cross as ``bytes`` both ways, each with the attachments it
names in a ``dict`` by id. Attachments carry ``bytes``, Arrow data or exact
core-created frame owners. Arrow data enters as any object with
``__arrow_c_stream__`` and leaves as an :class:`ArrowStream`.
Only ``sdm_client._rust`` imports this module, and only on a core route.
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
    'ENDPOINT_REFUSED',
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
#: The databases this build's core provides for, by name: the staging
#: rule's private, plain-data marker (contract/runners/run_python.py's
#: check_legs; T2.1a), never public API.
PROVIDERS: tuple[str, ...]
#: Every code an error of the core may carry, in the contract's order.
ERROR_CODES: tuple[_Code, ...]
#: Each ``(stype, dtype)`` the language lets a column mean, by the
#: contract's names (T3.16): the client's check of a meaning, read in place
#: of the old PQL package's.
MEANINGS: tuple[tuple[str, str], ...]
#: Each data type name a host may give a column, and the contract's name
#: of the type it names (T3.16): the one table of them.
DTYPE_NAMES: tuple[tuple[str, str], ...]

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

#: The builtin values presented from the core's logical JSON data.
_LogicalValue = (
    bool
    | int
    | float
    | str
    | None
    | list['_LogicalValue']
    | dict[str, '_LogicalValue']
)

@final
class FrameGraph:
    """An immutable graph owner created by the core."""

    def same_owner(self, other: FrameGraph) -> bool: ...

@final
class FrameQuestion:
    """An opaque question owner created by the core."""

@final
class FramePopulation:
    """An immutable population owner created by the core."""

    def keys(self) -> ArrowStream: ...

@final
class FrameLabelGroups:
    """Actual immutable pre-mask native label groups created by the core."""

    def rows(self) -> ArrowStream: ...

@final
class FrameLabelBatch:
    """An immutable label batch owner created by the core."""

    def rows(self) -> ArrowStream: ...

@final
class LabelInputs:
    """Reached source rows of an actual target evaluation."""

    def rows(self) -> ArrowStream: ...

@final
class TargetColumn:
    """Actual completed native target values and mathematical row selection."""

    def rows(self) -> ArrowStream: ...
    def selection(self) -> ArrowStream: ...

@final
class FrameTask:
    """An immutable task owner created by the core."""

    def context(self) -> ArrowStream: ...
    def prediction(self) -> ArrowStream: ...

@final
class FrameReferences:
    """An immutable reference owner created by the core."""

    def nodes(self) -> ArrowStream: ...

@final
class FramePayload:
    """A complete immutable payload created by the core."""

    def wire(self) -> bytes: ...
    def value(self) -> _LogicalValue: ...

@final
class LookupRows:
    """Actual counted task lookup data created by the core."""

@final
class MaterializeSource:
    """Actual immutable provider-backed materialization data."""

@final
class MaterializePopulation:
    """Actual immutable provider-backed materialization data."""

@final
class MaterializeGroups:
    """Actual immutable provider-backed materialization data."""

    def rows(self) -> ArrowStream: ...

@final
class MaterializeBatch:
    """Actual immutable provider-backed materialization data."""

_FrameOwner = (
    FrameGraph
    | FrameQuestion
    | FramePopulation
    | FrameLabelGroups
    | FrameLabelBatch
    | LabelInputs
    | TargetColumn
    | FrameTask
    | FrameReferences
    | FramePayload
)
#: An attachment a caller sends.
_MaterializeOwner = (
    LookupRows
    | MaterializeSource
    | MaterializePopulation
    | MaterializeGroups
    | MaterializeBatch
)
_AttachmentIn = bytes | _ArrowStreamExportable | _FrameOwner | _MaterializeOwner
#: An attachment the core gives.
_AttachmentOut = bytes | ArrowStream | _FrameOwner | _MaterializeOwner
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
class FrameKeyCandidates:
    """The core's ordered native cardinality and nominee observations."""

    def __init__(self, meanings: bytes, rows: int) -> None: ...
    def column(self) -> str | None: ...
    def observe(self, distinct: int, rows: int) -> None: ...
    def selected(self) -> str | None: ...
    def finish(
        self, missing: tuple[bool, int] | None
    ) -> tuple[bytes, bytes]: ...

@final
class FrameObjectConstructor:
    def __init__(self, rows: int) -> None: ...
    def content(self, content: str) -> None: ...
    def next_read(self) -> tuple[str, int]: ...
    def observe(self, value: bool) -> None: ...
    def distinct(self, distinct: int) -> None: ...
    def meaning(self) -> bytes: ...

@final
class FrameKeyAdoption:
    def __init__(self, product: bytes) -> None: ...
    def validation_columns(self) -> list[str] | None: ...
    def validation_empty(self, empty: bool) -> None: ...
    def validation_quality(self, quality: bytes) -> None: ...
    def missing_columns(self) -> list[str]: ...
    def validation_needs_rows(self) -> bool: ...
    def validate(self, missing: list[str], rows: int | None) -> None: ...
    def adoption_columns(self) -> list[str] | None: ...
    def adopt(self, quality: bytes | None) -> None: ...
    def discriminators(self) -> list[str]: ...
    def discriminator(self, name: str, quality: bytes) -> None: ...
    def usable_columns(self) -> list[str]: ...
    def usable(self, name: str, missing: bool | None) -> None: ...
    def needs_rows(self) -> bool: ...
    def finish(self, rows: int | None) -> tuple[bytes, bytes]: ...

@final
class FrameKeyAdvice:
    def __init__(self, plan: bytes, dates: dict[str, str]) -> None: ...
    def declined_tables(self) -> list[str]: ...
    def begin_declined(self, name: str, empty: bool) -> None: ...
    def declined_column(self) -> str | None: ...
    def declined_distinct(self, distinct: int, rows: int) -> bool: ...
    def declined_quality(self, quality: bytes) -> bool: ...
    def declined_notice(self, rows: int) -> bytes: ...
    def keyless_tables(self) -> list[str]: ...
    def keyless_columns(self, name: str, empty: bool) -> list[str]: ...
    def keyless_missing(self, missing: bool) -> bool: ...
    def keyless_quality(
        self, table: str, column: str, quality: bytes
    ) -> None: ...
    def keyless_notice(self, table: str) -> bytes: ...
    def ambiguous_notices(self) -> bytes: ...

def check_frame_cell(described: str, kind: str, native_type: str) -> None: ...
def frame_sample_facts(sample: bytes) -> bytes: ...

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
    def take_sampler_state(self) -> _Outgoing | None: ...

def reset_log_cache() -> None:
    """Forget the logger levels the logging bridge cached."""

def templates() -> str:
    """The table of what every error's and warning's sentence says, as
    JSON (T3.17)."""

def is_posix_zone(text: str) -> bool:
    """Whether a run takes ``text`` as a POSIX ``TZ`` local zone."""

def local_zone(setting: str | None, roots: list[str]) -> tuple[str, str] | None:
    """The local zone by the core's rule (T3.12): ``('iana', name)``,
    ``('posix', text)`` or ``None``, from ``TZ`` and the lookup roots."""

def capabilities() -> str:
    """Each model's capabilities, as the client describes them, as JSON
    (T3.12)."""

@final
class FrameNativeConstructor:
    def dtype(self) -> bytes: ...
    def __init__(
        self, table: str, name: str, values: bytes, declared: bytes
    ) -> None: ...
    def timestamp_sample(self) -> bool: ...
    def separated_sample(self) -> bool: ...
    def element_sample(self) -> bool: ...
    def element_count(self, parts: int) -> None: ...
    def distinct_sample(self) -> bool: ...
    def timestamp_counts(self, parsed: int, present: int) -> None: ...
    def separator(self, points: list[int]) -> str | None: ...
    def separated_counts(self, whole: int, parts: int) -> None: ...
    def distinct_count(self, distinct: int) -> bool: ...
    def population_count(self, present: int) -> None: ...
    def meaning(self) -> bytes: ...
