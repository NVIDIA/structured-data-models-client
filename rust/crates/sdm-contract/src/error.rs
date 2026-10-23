// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Errors and warnings.
//!
//! An error is `{kind, code, message, fields, cause, retryable,
//! drop_held}` with a closed list of kinds and a closed list of codes; a warning travels on its
//! own channel, in order. No string here names a Python or Java class: each
//! host maps every code to its own classes through one reviewed table,
//! never reading message text. Codes are promised, message text is not.
//!
//! Each code may carry only the kinds [`ErrorCode::kinds`] lists, so a
//! host's mapping keys on the code alone (decision E22). Every construction
//! is held to that table: [`CoreError::new`] takes its code's kind;
//! [`CoreError::provider`] and [`CoreError::transport`], whose kind comes
//! from the host, give `INTERNAL_ERROR` for a kind outside the code's set;
//! and since the fields are public and the type deserializes, `sdm-host`
//! checks every pair where an error crosses the boundary
//! ([`CoreError::holds_to_table`]).
//!
//! Every error and every warning says one [`Sentence`] of a closed list
//! (invariant 8 of the Phase 3 structure design, T3.17): the core picks it,
//! and each host fills its one shared template ([`crate::wording`]) with the
//! values it names, as that host writes them. A sentence gives the public
//! code ([`Sentence::code`]) and, where it has one, the rule a caller
//! branches on (`fields.rule`, [`Sentence::rule`]); no host chooses one.

use std::collections::BTreeMap;
use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::ProviderError;
use crate::scalar::Scalar;

/// What kind of problem an error is.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// The caller's request is not one the core accepts.
    InvalidRequest,
    /// A valid request the core does not support, such as a type outside
    /// the closed set.
    Unsupported,
    /// Something the request names does not exist.
    NotFound,
    /// The data source or its driver failed.
    DataSource,
    /// The model endpoint or its transport failed.
    ModelEndpoint,
    /// The run was cancelled.
    Cancelled,
    /// The host broke the run protocol: a call in the wrong state, a reply
    /// the run did not ask for, a malformed control message.
    Protocol,
    /// A defect in the core, a contained panic included.
    Internal,
}

impl ErrorKind {
    /// Every kind, in the order the contract lists them.
    pub const ALL: [ErrorKind; 8] = [
        ErrorKind::InvalidRequest,
        ErrorKind::Unsupported,
        ErrorKind::NotFound,
        ErrorKind::DataSource,
        ErrorKind::ModelEndpoint,
        ErrorKind::Cancelled,
        ErrorKind::Protocol,
        ErrorKind::Internal,
    ];

    /// The kind's name, as the contract spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorKind::InvalidRequest => "invalid_request",
            ErrorKind::Unsupported => "unsupported",
            ErrorKind::NotFound => "not_found",
            ErrorKind::DataSource => "data_source",
            ErrorKind::ModelEndpoint => "model_endpoint",
            ErrorKind::Cancelled => "cancelled",
            ErrorKind::Protocol => "protocol",
            ErrorKind::Internal => "internal",
        }
    }
}

/// The codes the core gives: one closed list, so each host's mapping
/// matches every code with no catch-all arm (decision H6). Host and core
/// ship together, so the list is exhaustive; a new code is a contract
/// version bump. PQL's own diagnostic codes stay strings inside
/// `PqlDiagnostic`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    /// The caller's request is not one the core accepts.
    InvalidRequest,
    /// A valid request this core does not support.
    UnsupportedRequest,
    /// The model endpoint answered with something the core cannot read.
    InvalidResponse,
    /// The host's configuration of a run is not one the core accepts.
    InvalidConfiguration,
    /// The transport to the model endpoint failed: it could not reach the
    /// endpoint, or could not read its answer.
    TransportError,
    /// A defect in the core, a contained panic included.
    InternalError,
    /// The model a request names does not exist.
    UnknownModel,
    /// An optional dependency of the host is not installed.
    MissingExtra,
    /// A schema the core cannot work with.
    InvalidSchema,
    /// The endpoint does not support sessions.
    SessionsUnsupported,
    /// An Arrow type outside the closed set.
    UnsupportedType,
    /// A value the admitted type of its column cannot hold, such as a `timestamp[s]` past the microseconds an int64 counts.
    ValueOutOfRange,
    /// A provider answered with a schema other than the one asked for.
    ProviderSchemaMismatch,
    /// A provider's query failed.
    ProviderError,
    /// A reply for an id the run is not waiting on: unknown, stale or already answered.
    UnknownReply,
    /// `next` was called before every reply the run asked for arrived.
    ReplyMissing,
    /// A reply of the wrong kind: a model answer for a fetch, or the other way round.
    WrongReply,
    /// The run was used after it finished.
    RunFinished,
    /// The run was used after it was cancelled.
    RunCancelled,
    /// The run was used after a panic poisoned it.
    RunPoisoned,
    /// A second call reached a run while one was in progress.
    RunBusy,
    /// A control message the contract cannot read, larger than the control cap, of the wrong direction, or whose attachments do not match the ids it names.
    InvalidMessage,
}

impl ErrorCode {
    /// Every code, in the order the contract lists them.
    pub const ALL: [ErrorCode; 22] = [
        ErrorCode::InvalidRequest,
        ErrorCode::UnsupportedRequest,
        ErrorCode::InvalidResponse,
        ErrorCode::InvalidConfiguration,
        ErrorCode::TransportError,
        ErrorCode::InternalError,
        ErrorCode::UnknownModel,
        ErrorCode::MissingExtra,
        ErrorCode::InvalidSchema,
        ErrorCode::SessionsUnsupported,
        ErrorCode::UnsupportedType,
        ErrorCode::ValueOutOfRange,
        ErrorCode::ProviderSchemaMismatch,
        ErrorCode::ProviderError,
        ErrorCode::UnknownReply,
        ErrorCode::ReplyMissing,
        ErrorCode::WrongReply,
        ErrorCode::RunFinished,
        ErrorCode::RunCancelled,
        ErrorCode::RunPoisoned,
        ErrorCode::RunBusy,
        ErrorCode::InvalidMessage,
    ];

    /// The kinds an error of this code may carry (decision E22): one for
    /// every code but `PROVIDER_ERROR`, whose kind the host's provider
    /// chooses from a closed set.
    ///
    /// `VALUE_OUT_OF_RANGE` stays `unsupported` (E13): the value is the
    /// source's own, valid there; what fails is that the type the core
    /// admits it as cannot hold it, which is the core's limit, not the
    /// source's failure.
    pub fn kinds(self) -> &'static [ErrorKind] {
        use ErrorKind as K;
        match self {
            ErrorCode::InvalidRequest
            | ErrorCode::InvalidConfiguration
            | ErrorCode::InvalidSchema => &[K::InvalidRequest],
            ErrorCode::UnsupportedRequest
            | ErrorCode::MissingExtra
            | ErrorCode::SessionsUnsupported
            | ErrorCode::UnsupportedType
            | ErrorCode::ValueOutOfRange => &[K::Unsupported],
            ErrorCode::UnknownModel => &[K::NotFound],
            ErrorCode::InvalidResponse | ErrorCode::TransportError => &[K::ModelEndpoint],
            ErrorCode::InternalError => &[K::Internal],
            ErrorCode::ProviderSchemaMismatch => &[K::DataSource],
            ErrorCode::ProviderError => &[K::DataSource, K::NotFound],
            ErrorCode::RunCancelled => &[K::Cancelled],
            ErrorCode::UnknownReply
            | ErrorCode::ReplyMissing
            | ErrorCode::WrongReply
            | ErrorCode::RunFinished
            | ErrorCode::RunPoisoned
            | ErrorCode::RunBusy
            | ErrorCode::InvalidMessage => &[K::Protocol],
        }
    }

    /// Whether an error of this code may carry `kind`.
    pub fn allows(self, kind: ErrorKind) -> bool {
        self.kinds().contains(&kind)
    }

    /// The code's name, as the contract spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::InvalidRequest => "INVALID_REQUEST",
            ErrorCode::UnsupportedRequest => "UNSUPPORTED_REQUEST",
            ErrorCode::InvalidResponse => "INVALID_RESPONSE",
            ErrorCode::InvalidConfiguration => "INVALID_CONFIGURATION",
            ErrorCode::TransportError => "TRANSPORT_ERROR",
            ErrorCode::InternalError => "INTERNAL_ERROR",
            ErrorCode::UnknownModel => "UNKNOWN_MODEL",
            ErrorCode::MissingExtra => "MISSING_EXTRA",
            ErrorCode::InvalidSchema => "INVALID_SCHEMA",
            ErrorCode::SessionsUnsupported => "SESSIONS_UNSUPPORTED",
            ErrorCode::UnsupportedType => "UNSUPPORTED_TYPE",
            ErrorCode::ValueOutOfRange => "VALUE_OUT_OF_RANGE",
            ErrorCode::ProviderSchemaMismatch => "PROVIDER_SCHEMA_MISMATCH",
            ErrorCode::ProviderError => "PROVIDER_ERROR",
            ErrorCode::UnknownReply => "UNKNOWN_REPLY",
            ErrorCode::ReplyMissing => "REPLY_MISSING",
            ErrorCode::WrongReply => "WRONG_REPLY",
            ErrorCode::RunFinished => "RUN_FINISHED",
            ErrorCode::RunCancelled => "RUN_CANCELLED",
            ErrorCode::RunPoisoned => "RUN_POISONED",
            ErrorCode::RunBusy => "RUN_BUSY",
            ErrorCode::InvalidMessage => "INVALID_MESSAGE",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A value under an error's or a warning's `fields`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Detail {
    Scalar(Scalar),
    List(Vec<Scalar>),
}

impl Detail {
    pub fn text(value: impl Into<String>) -> Self {
        Detail::Scalar(Scalar::Utf8(value.into()))
    }
}

/// A state a host holds between runs and hands back to the core with a
/// later run's start, opaque to it.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum HeldState {
    /// The sampler state a materializer's last run left
    /// (`Materialize.sampler_state`, decision E52).
    SamplerState,
}

/// An error the core reports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CoreError {
    pub kind: ErrorKind,
    /// Stable across releases and languages.
    pub code: ErrorCode,
    /// What the error says, of [`ErrorCode`]'s sentences: the template each
    /// host fills, read only for wording (T3.17).
    pub sentence: Sentence,
    /// Where the error happened in a question asked in parts (a forecast's
    /// step, a ranking's candidate), a sentence of [`Code::Part`] said after
    /// the error's own; none for any other error (T3.17 fix 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<Sentence>,
    /// For a person to read; not promised, and never holding an input value.
    pub message: String,
    /// What callers branch on, by name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fields: BTreeMap<String, Detail>,
    /// The error this one was raised from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cause: Option<Box<CoreError>>,
    /// Whether the same call may succeed if tried again; carried from
    /// provider and transport errors, false otherwise.
    #[serde(default)]
    pub retryable: bool,
    /// The states the host handed this run that it must drop rather than
    /// hand back again: the core refused them, so a later run given them
    /// would be refused the same way (decision E74). The host follows this
    /// alone, never reading which field an error names; empty for any
    /// other error.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drop_held: Vec<HeldState>,
    /// What a request asked in several model calls had been answered when
    /// a later call failed (design choice 1, `main`'s answered parts,
    /// T2.8c): the error keeps what was paid for, so the rest can be asked
    /// again. Absent for any other error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answered: Option<AnsweredParts>,
}

/// The answers a failed request had received ([`CoreError::answered`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnsweredParts {
    /// An Arrow attachment of the answers received, laid out as
    /// `Predicted.result`, travelling with the error.
    pub result: crate::data::AttachmentId,
    /// The rows those answers score (`main`'s `rows_answered`).
    pub rows: u64,
}

impl CoreError {
    /// An error saying `sentence`, of its code and that code's kind; the
    /// sentence's rule, where it has one, is the error's `rule` field. A
    /// warning's sentence is no error's: a debug build asserts it, and a
    /// release build gives `INTERNAL_ERROR`.
    pub fn new(sentence: Sentence, message: impl Into<String>) -> Self {
        let Code::Error(code) = sentence.code() else {
            debug_assert!(false, "{} is no error's sentence", sentence.as_str());
            return Self::new(
                Sentence::Internal,
                "an error was given another kind's sentence",
            );
        };
        let mut out = Self {
            kind: code.kinds()[0],
            code,
            sentence,
            part: None,
            message: message.into(),
            fields: BTreeMap::new(),
            cause: None,
            retryable: false,
            drop_held: Vec::new(),
            answered: None,
        };
        if let Some(rule) = sentence.rule() {
            out = out.with_field("rule", Detail::text(rule));
        }
        out
    }

    /// A protocol error: the host broke the run protocol.
    pub fn protocol(sentence: Sentence, message: impl Into<String>) -> Self {
        let out = Self::new(sentence, message);
        debug_assert_eq!(out.kind, ErrorKind::Protocol, "{}", sentence.as_str());
        out
    }

    /// The error a provider's failure becomes: its kind and retryability,
    /// with the driver's SQLSTATE and vendor code as fields. The driver's
    /// message is not part of the contract, since drivers quote values in
    /// it.
    ///
    /// A kind `PROVIDER_ERROR` does not carry gives `INTERNAL_ERROR`
    /// instead (decision E22).
    pub fn provider(error: &ProviderError) -> Self {
        if !ErrorCode::ProviderError.allows(error.kind) {
            return Self::outside_table(error.kind, ErrorCode::ProviderError);
        }
        let mut out = Self::new(Sentence::ProviderFailed, "the data provider failed");
        out.kind = error.kind;
        out.retryable = error.retryable;
        if let Some(sqlstate) = &error.sqlstate {
            out = out.with_field("sqlstate", Detail::text(sqlstate.clone()));
        }
        if let Some(vendor_code) = &error.vendor_code {
            out = out.with_field("vendor_code", Detail::text(vendor_code.clone()));
        }
        out
    }

    /// The error a model transport's failure becomes: its kind and
    /// retryability. A kind `TRANSPORT_ERROR` does not carry gives
    /// `INTERNAL_ERROR` instead (decision E22).
    pub fn transport(kind: ErrorKind, retryable: bool) -> Self {
        if !ErrorCode::TransportError.allows(kind) {
            return Self::outside_table(kind, ErrorCode::TransportError);
        }
        let mut out = Self::new(
            Sentence::TransportFailed,
            "the transport to the model endpoint failed",
        );
        out.kind = kind;
        out.retryable = retryable;
        out
    }

    /// The `INTERNAL_ERROR` that replaces an error whose kind its code does
    /// not carry, naming the pair as `pair_kind` and `pair_code`.
    pub fn outside_table(kind: ErrorKind, code: ErrorCode) -> Self {
        Self::new(
            Sentence::Internal,
            "an error carried a kind its code does not allow",
        )
        .with_field("pair_kind", Detail::text(kind.as_str()))
        .with_field("pair_code", Detail::text(code.as_str()))
    }

    /// Whether this error, and every error in its cause chain, carries a
    /// kind its code allows (decision E22) and a sentence of its code
    /// (T3.17).
    pub fn holds_to_table(&self) -> bool {
        self.code.allows(self.kind)
            && self.sentence.code() == Code::Error(self.code)
            && self.part.is_none_or(|part| part.code() == Code::Part)
            && self.cause.as_deref().is_none_or(CoreError::holds_to_table)
    }

    #[must_use]
    pub fn with_field(mut self, name: &str, value: Detail) -> Self {
        self.fields.insert(name.to_owned(), value);
        self
    }

    /// This error saying `sentence` in place of its own, a sentence of the
    /// same code: the core's choice between two ways of saying one refusal
    /// once it knows which applies. The rule follows the sentence.
    #[must_use]
    pub fn saying(mut self, sentence: Sentence) -> Self {
        debug_assert_eq!(
            sentence.code(),
            Code::Error(self.code),
            "{}",
            sentence.as_str()
        );
        self.sentence = sentence;
        match sentence.rule() {
            Some(rule) => self.with_field("rule", Detail::text(rule)),
            None => self,
        }
    }

    /// This error, said of the part of a question asked in parts it
    /// happened in: `part`, a sentence of [`Code::Part`], said after its own.
    #[must_use]
    pub fn in_part(mut self, part: Sentence) -> Self {
        debug_assert_eq!(part.code(), Code::Part, "{}", part.as_str());
        self.part = Some(part);
        self
    }

    /// This error, telling the host to drop the held state `state` it
    /// handed the run (decision E74).
    #[must_use]
    pub fn dropping(mut self, state: HeldState) -> Self {
        if !self.drop_held.contains(&state) {
            self.drop_held.push(state);
        }
        self
    }

    /// This error, carrying the answers its request had received before it
    /// ([`CoreError::answered`]).
    #[must_use]
    pub fn answering(mut self, answered: AnsweredParts) -> Self {
        self.answered = Some(answered);
        self
    }

    #[must_use]
    pub fn caused_by(mut self, cause: CoreError) -> Self {
        self.cause = Some(Box::new(cause));
        self
    }
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for CoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.cause
            .as_deref()
            .map(|c| c as &(dyn std::error::Error + 'static))
    }
}

/// A warning a run gives, kept in the order given.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CoreWarning {
    /// The sentence's [`WarningCode`], as the contract spells it.
    #[schemars(regex(pattern = r"^[A-Z][A-Z0-9_]*$"))]
    pub code: String,
    /// What the warning says (T3.17).
    pub sentence: Sentence,
    pub message: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fields: BTreeMap<String, Detail>,
}

impl CoreWarning {
    /// A warning saying `sentence`, of its code. An error's sentence is no
    /// warning's: a debug build asserts it.
    pub fn new(sentence: Sentence, message: impl Into<String>) -> Self {
        let code = match sentence.code() {
            Code::Warning(code) => code.as_str(),
            other => {
                debug_assert!(false, "{} is no warning's sentence", sentence.as_str());
                other.as_str().unwrap_or_default()
            }
        };
        Self {
            code: code.to_owned(),
            sentence,
            message: message.into(),
            fields: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn with_field(mut self, name: &str, value: Detail) -> Self {
        self.fields.insert(name.to_owned(), value);
        self
    }

    /// This warning with each of `fields`.
    #[must_use]
    pub fn with_fields<'a>(mut self, fields: impl IntoIterator<Item = (&'a str, Detail)>) -> Self {
        for (name, value) in fields {
            self.fields.insert(name.to_owned(), value);
        }
        self
    }

    /// Whether the warning's code is its sentence's (T3.17).
    pub fn holds_to_table(&self) -> bool {
        matches!(self.sentence.code(), Code::Warning(code) if code.as_str() == self.code)
    }
}

/// The code a sentence gives: an error's or a warning's; or none, for the
/// sentence of the part of a question asked in parts an error happened in,
/// said after the error's own of any code ([`CoreError::part`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Code {
    Error(ErrorCode),
    Warning(WarningCode),
    Part,
}

impl Code {
    /// The code's name, as the contract spells it; none for a part's.
    pub fn as_str(self) -> Option<&'static str> {
        match self {
            Code::Error(code) => Some(code.as_str()),
            Code::Warning(code) => Some(code.as_str()),
            Code::Part => None,
        }
    }
}

macro_rules! warning_codes {
    ($($variant:ident = $name:literal,)*) => {
        /// The codes of the warnings the core gives: one closed list, each
        /// spelled as it always has been on the wire (T3.17).
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
            JsonSchema,
        )]
        pub enum WarningCode {
            $(
                #[serde(rename = $name)]
                $variant,
            )*
        }

        impl WarningCode {
            /// Every warning code, in the order the contract lists them.
            pub const ALL: &'static [WarningCode] = &[$(WarningCode::$variant,)*];

            /// The code's name, as the contract spells it.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(WarningCode::$variant => $name,)*
                }
            }
        }
    };
}

macro_rules! rule_of {
    () => {
        None
    };
    ($rule:literal) => {
        Some($rule)
    };
}

macro_rules! sentences {
    ($($variant:ident = $name:literal, $code:path $(, rule $rule:literal)?;)*) => {
        /// What an error or a warning says: one variant per distinct
        /// sentence, each with the one template every host fills
        /// ([`crate::wording::template`]). The core picks it; a host only
        /// fills it.
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
            JsonSchema,
        )]
        pub enum Sentence {
            $(
                #[serde(rename = $name)]
                $variant,
            )*
        }

        impl Sentence {
            /// Every sentence, in the order the contract lists them.
            pub const ALL: &'static [Sentence] = &[$(Sentence::$variant,)*];

            /// The sentence's name, as the contract spells it.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Sentence::$variant => $name,)*
                }
            }

            /// The public code of an error or warning that says it.
            pub fn code(self) -> Code {
                use ErrorCode as E;
                use WarningCode as W;
                match self {
                    $(Sentence::$variant => Code::from($code),)*
                }
            }

            /// The rule an error that says it names in its `rule` field,
            /// the public detail a caller branches on; none where the
            /// error's site names its own, or none.
            pub fn rule(self) -> Option<&'static str> {
                match self {
                    $(Sentence::$variant => rule_of!($($rule)?),)*
                }
            }
        }
    };
}

impl From<ErrorCode> for Code {
    fn from(code: ErrorCode) -> Self {
        Code::Error(code)
    }
}

impl From<WarningCode> for Code {
    fn from(code: WarningCode) -> Self {
        Code::Warning(code)
    }
}

warning_codes! {
    KeyNoValue = "KEY_NO_VALUE",
    KeyRepeats = "KEY_REPEATS",
    KeyDeclined = "KEY_DECLINED",
    TimeColumnsGuessed = "TIME_COLUMNS_GUESSED",
    ColumnsLeftOut = "COLUMNS_LEFT_OUT",
    TimeValuesUnread = "TIME_VALUES_UNREAD",
    DatesReadDayFirst = "DATES_READ_DAY_FIRST",
    KeyCandidatesTied = "KEY_CANDIDATES_TIED",
    LinkCannotJoin = "LINK_CANNOT_JOIN",
    TablesNameEachOther = "TABLES_NAME_EACH_OTHER",
    TwoWaysToATable = "TWO_WAYS_TO_A_TABLE",
    CandidatesUnscored = "CANDIDATES_UNSCORED",
    ContextConstant = "CONTEXT_CONSTANT",
    OutputsAbsent = "OUTPUTS_ABSENT",
    StepsUnexplained = "STEPS_UNEXPLAINED",
    MissingIndices = "MISSING_INDICES",
    ContextBeforeData = "CONTEXT_BEFORE_DATA",
    AnchorAfterData = "ANCHOR_AFTER_DATA",
    Float32Overflow = "FLOAT32_OVERFLOW",
    ContextAfterAnchor = "CONTEXT_AFTER_ANCHOR",
    ContextLeaks = "CONTEXT_LEAKS",
    HopsIgnored = "HOPS_IGNORED",
    ContextCoverage = "CONTEXT_COVERAGE",
    ContextCapped = "CONTEXT_CAPPED",
    TablesUnlinked = "TABLES_UNLINKED",
    TablesUnreached = "TABLES_UNREACHED",
    ColumnsUnread = "COLUMNS_UNREAD",
    FilterChoseExamples = "FILTER_CHOSE_EXAMPLES",
    ContextAnswers = "CONTEXT_ANSWERS",
    NullKeysLeftOut = "NULL_KEYS_LEFT_OUT",
    DateLiteralDayFirst = "DATE_LITERAL_DAY_FIRST",
}

sentences! {
    Internal = "internal", E::InternalError;
    ProviderFailed = "provider_failed", E::ProviderError;
    TransportFailed = "transport_failed", E::TransportError;
    UnknownReply = "unknown_reply", E::UnknownReply;
    ReplyMissing = "reply_missing", E::ReplyMissing;
    WrongReply = "wrong_reply", E::WrongReply;
    RunFinished = "run_finished", E::RunFinished;
    RunCancelled = "run_cancelled", E::RunCancelled;
    RunPoisoned = "run_poisoned", E::RunPoisoned;
    RunBusy = "run_busy", E::RunBusy;
    InvalidMessage = "invalid_message", E::InvalidMessage;
    UnknownModel = "unknown_model", E::UnknownModel;
    MissingExtra = "missing_extra", E::MissingExtra;
    SessionsUnsupported = "sessions_unsupported", E::SessionsUnsupported;
    RelationNoPart = "relation_no_part", E::InvalidRequest, rule "relation";
    RelationTooManyParts = "relation_too_many_parts", E::UnsupportedRequest, rule "relation";
    DerivedRelationUnread = "derived_relation_unread", E::UnsupportedRequest, rule "derived_relation";
    DerivedRelationMalformed = "derived_relation_malformed", E::InvalidRequest, rule "derived_relation";
    DataRequestRefused = "data_request_refused", E::InvalidRequest;
    UnsupportedType = "unsupported_type", E::UnsupportedType;
    NotYetDescribe = "not_yet_describe", E::UnsupportedRequest;
    EmptyTable = "empty_table", E::InvalidRequest;
    TableNameEmpty = "table_name_empty", E::InvalidRequest;
    TableNameCharacters = "table_name_characters", E::InvalidRequest;
    TimeOutsideYears = "time_outside_years", E::ValueOutOfRange;
    NoTable = "no_table", E::InvalidRequest;
    KeyNotNames = "key_not_names", E::InvalidRequest, rule "key_not_names";
    KeyNotNamesGiven = "key_not_names_given", E::InvalidRequest, rule "key_not_names";
    TimeColumnNotAName = "time_column_not_a_name", E::InvalidRequest, rule "time_column_not_a_name";
    TimeColumnNotANameGiven = "time_column_not_a_name_given", E::InvalidRequest, rule "time_column_not_a_name";
    ColumnNotAName = "column_not_a_name", E::InvalidRequest;
    ColumnNotANameGiven = "column_not_a_name_given", E::InvalidRequest;
    EmptyKey = "empty_key", E::InvalidRequest;
    NoColumn = "no_column", E::InvalidRequest;
    KeyColumnTwice = "key_column_twice", E::InvalidRequest, rule "key_column_twice";
    KeyCannotIdentify = "key_cannot_identify", E::InvalidRequest;
    KeyIsTimeColumn = "key_is_time_column", E::InvalidRequest;
    TimeIsKey = "time_is_key", E::InvalidRequest;
    MeaningNotCarried = "meaning_not_carried", E::InvalidRequest;
    NotASemanticType = "not_a_semantic_type", E::InvalidRequest;
    NotASemanticTypeGiven = "not_a_semantic_type_given", E::InvalidRequest;
    NotASemanticTypeUnread = "not_a_semantic_type_unread", E::InvalidRequest;
    LeftOutKey = "left_out_key", E::InvalidRequest;
    LeftOutTimeColumn = "left_out_time_column", E::InvalidRequest;
    NotARelationship = "not_a_relationship", E::InvalidRequest;
    NotARelationshipGiven = "not_a_relationship_given", E::InvalidRequest;
    LinkUnheld = "link_unheld", E::InvalidRequest;
    LinkNotTheKey = "link_not_the_key", E::InvalidRequest;
    LinkNoKey = "link_no_key", E::InvalidRequest;
    LinkCompositeKey = "link_composite_key", E::InvalidRequest;
    LinkTypes = "link_types", E::InvalidRequest;
    PrimaryKeyAsForeignKey = "primary_key_as_foreign_key", E::InvalidRequest;
    SchemaTableTwice = "schema_table_twice", E::InvalidRequest;
    SchemaColumnTwice = "schema_column_twice", E::InvalidRequest;
    IncompatibleSemanticType = "incompatible_semantic_type", E::InvalidRequest;
    KeyNotUniqueColumn = "key_not_unique_column", E::InvalidRequest, rule "key_not_unique";
    KeyNotUniqueColumns = "key_not_unique_columns", E::InvalidRequest, rule "key_not_unique";
    KeyNotUniqueUnenforced = "key_not_unique_unenforced", E::InvalidRequest, rule "key_not_unique";
    KeyMissingValuesColumn = "key_missing_values_column", E::InvalidRequest, rule "key_missing_values";
    KeyMissingValuesColumns = "key_missing_values_columns", E::InvalidRequest, rule "key_missing_values";
    KeyNoValueColumn = "key_no_value_column", W::KeyNoValue;
    KeyNoValueColumns = "key_no_value_columns", W::KeyNoValue;
    KeyRepeatsColumn = "key_repeats_column", W::KeyRepeats;
    KeyRepeatsColumns = "key_repeats_columns", W::KeyRepeats;
    KeyDeclined = "key_declined", W::KeyDeclined;
    TimeColumnsTakenEarliest = "time_columns_taken_earliest", W::TimeColumnsGuessed;
    TimeColumnsTakenUncheckedEmpty = "time_columns_taken_unchecked_empty", W::TimeColumnsGuessed;
    TimeColumnsTakenUncheckedSparse = "time_columns_taken_unchecked_sparse", W::TimeColumnsGuessed;
    TimeColumnsTakenUncheckedConstantWhereHeld = "time_columns_taken_unchecked_constant_where_held", W::TimeColumnsGuessed;
    TimeColumnsTakenUncheckedConstant = "time_columns_taken_unchecked_constant", W::TimeColumnsGuessed;
    TimeColumnsTakenEditStamp = "time_columns_taken_edit_stamp", W::TimeColumnsGuessed;
    TimeColumnsRefusedEmpty = "time_columns_refused_empty", W::TimeColumnsGuessed;
    TimeColumnsRefusedSparse = "time_columns_refused_sparse", W::TimeColumnsGuessed;
    TimeColumnsRefusedConstantWhereHeld = "time_columns_refused_constant_where_held", W::TimeColumnsGuessed;
    TimeColumnsRefusedConstant = "time_columns_refused_constant", W::TimeColumnsGuessed;
    TimeColumnsRefusedLoadStamp = "time_columns_refused_load_stamp", W::TimeColumnsGuessed;
    TimeColumnsRefusedEditStamp = "time_columns_refused_edit_stamp", W::TimeColumnsGuessed;
    TimeColumnsRefusedBirthDate = "time_columns_refused_birth_date", W::TimeColumnsGuessed;
    ColumnsLeftOut = "columns_left_out", W::ColumnsLeftOut;
    TimeValuesUnread = "time_values_unread", W::TimeValuesUnread;
    DatesReadDayFirst = "dates_read_day_first", W::DatesReadDayFirst;
    KeyCandidatesTied = "key_candidates_tied", W::KeyCandidatesTied;
    LinkCannotJoinNotTheKey = "link_cannot_join_not_the_key", W::LinkCannotJoin;
    LinkCannotJoinNoKey = "link_cannot_join_no_key", W::LinkCannotJoin;
    LinkCannotJoinTypes = "link_cannot_join_types", W::LinkCannotJoin;
    TablesNameEachOther = "tables_name_each_other", W::TablesNameEachOther;
    TwoWaysToATable = "two_ways_to_a_table", W::TwoWaysToATable;
    NoTables = "no_tables", E::InvalidRequest, rule "no_tables";
    SourceNoPart = "source_no_part", E::InvalidRequest, rule "relation";
    SourceNameEmpty = "source_name_empty", E::InvalidRequest;
    TableNamedTwice = "table_named_twice", E::InvalidRequest;
    NoSuchTable = "no_such_table", E::InvalidRequest, rule "no_such_table";
    NoSuchTableInSource = "no_such_table_in_source", E::InvalidRequest, rule "no_such_table";
    ProviderSchemaMismatch = "provider_schema_mismatch", E::ProviderSchemaMismatch;
    ValueNotANumber = "value_not_a_number", E::ProviderSchemaMismatch;
    Undeclared = "undeclared", E::UnsupportedRequest;
    SeedPastRange = "seed_past_range", E::InvalidRequest;
    BigIntegerPastType = "big_integer_past_type", E::ValueOutOfRange;
    LagPastPayload = "lag_past_payload", E::InvalidRequest;
    DateLiteralPacked = "date_literal_packed", E::InvalidRequest, rule "date_literal_packed";
    LocalZoneUnread = "local_zone_unread", E::InvalidRequest;
    ClockPastYears = "clock_past_years", E::InvalidRequest;
    UnknownZone = "unknown_zone", E::InvalidRequest;
    TestOperationRefused = "test_operation_refused", E::InvalidRequest;
    FallbackLimit = "fallback_limit", E::UnsupportedRequest, rule "fallback_limit";
    FallbackUnordered = "fallback_unordered", E::UnsupportedRequest, rule "fallback";
    FallbackTimePast = "fallback_time_past", E::UnsupportedRequest, rule "fallback";
    FallbackTimeType = "fallback_time_type", E::UnsupportedRequest, rule "fallback";
    FallbackNoColumn = "fallback_no_column", E::UnsupportedRequest, rule "fallback";
    FallbackSeedTypes = "fallback_seed_types", E::UnsupportedRequest, rule "fallback";
    HostClockNotUtc = "host_clock_not_utc", E::InvalidRequest;
    StateVersion = "state_version", E::InvalidRequest;
    StateUnread = "state_unread", E::InvalidRequest;
    SeedNegative = "seed_negative", E::InvalidRequest;
    SamplerStateVersion = "sampler_state_version", E::InvalidRequest;
    OptionValuesUnread = "option_values_unread", E::InvalidRequest;
    CandidatesNotRanked = "candidates_not_ranked", E::InvalidRequest, rule "candidates_not_ranked";
    RankingStatistic = "ranking_statistic", E::InvalidRequest, rule "ranking_statistic";
    RankingColumn = "ranking_column", E::InvalidRequest, rule "ranking_column";
    RankingFilter = "ranking_filter", E::InvalidRequest, rule "ranking_filter";
    CandidatesMissing = "candidates_missing", E::InvalidRequest, rule "candidates_missing";
    CandidatesEmpty = "candidates_empty", E::InvalidRequest, rule "candidates_empty";
    TooManyCandidates = "too_many_candidates", E::InvalidRequest, rule "too_many_parts";
    TooManySteps = "too_many_steps", E::InvalidRequest, rule "too_many_parts";
    RankingOutputsNone = "ranking_outputs_none", E::InvalidRequest, rule "ranking_outputs";
    RankingOutputUnknown = "ranking_output_unknown", E::InvalidRequest, rule "ranking_outputs";
    RankingOutputsUnknown = "ranking_outputs_unknown", E::InvalidRequest, rule "ranking_outputs";
    RankingUnscoredOne = "ranking_unscored_one", E::UnsupportedRequest, rule "ranking_unscored";
    RankingUnscoredSeveral = "ranking_unscored_several", E::UnsupportedRequest, rule "ranking_unscored";
    CandidatesUnscoredOne = "candidates_unscored_one", W::CandidatesUnscored;
    CandidatesUnscoredSeveral = "candidates_unscored_several", W::CandidatesUnscored;
    BatchedPartRanking = "batched_part_ranking", E::UnsupportedRequest, rule "batched_part";
    BatchedPartForecast = "batched_part_forecast", E::UnsupportedRequest, rule "batched_part";
    CandidateMissing = "candidate_missing", E::InvalidRequest, rule "candidate_missing";
    CandidateBool = "candidate_bool", E::InvalidRequest, rule "candidate_bool";
    CandidateUnreadable = "candidate_unreadable", E::InvalidRequest, rule "candidate_unreadable";
    CandidateNotExact = "candidate_not_exact", E::InvalidRequest, rule "candidate_not_exact";
    CandidateNotWhole = "candidate_not_whole", E::InvalidRequest, rule "candidate_not_whole";
    CandidateUnwritable = "candidate_unwritable", E::InvalidRequest, rule "candidate_unwritable";
    CandidateType = "candidate_type", E::UnsupportedRequest, rule "candidate_type";
    ProbabilitiesUnread = "probabilities_unread", E::InternalError;
    ProbabilitiesNone = "probabilities_none", E::InternalError;
    NotYetPredict = "not_yet_predict", E::UnsupportedRequest;
    ResponseUnread = "response_unread", E::InvalidResponse;
    ExplanationNotAMapping = "explanation_not_a_mapping", E::InvalidResponse;
    MetadataNotAMapping = "metadata_not_a_mapping", E::InvalidResponse;
    ModelLacksShape = "model_lacks_shape", E::UnsupportedRequest;
    ModelLacksTask = "model_lacks_task", E::UnsupportedRequest;
    ModelLacksOutputs = "model_lacks_outputs", E::UnsupportedRequest;
    ContextPastModel = "context_past_model", E::UnsupportedRequest;
    NoContextRows = "no_context_rows", E::UnsupportedRequest;
    TooManyClasses = "too_many_classes", E::UnsupportedRequest;
    OneClassColumn = "one_class_column", E::UnsupportedRequest;
    OneClassPredicted = "one_class_predicted", E::UnsupportedRequest;
    OneClassQuestion = "one_class_question", E::UnsupportedRequest;
    OneClassAnswer = "one_class_answer", E::UnsupportedRequest;
    ContextConstantColumn = "context_constant_column", W::ContextConstant;
    ContextConstantPredicted = "context_constant_predicted", W::ContextConstant;
    ContextConstantQuestion = "context_constant_question", W::ContextConstant;
    ContextConstantAnswer = "context_constant_answer", W::ContextConstant;
    OutputsAbsent = "outputs_absent", W::OutputsAbsent;
    StepsUnexplained = "steps_unexplained", W::StepsUnexplained;
    ExplanationOneEntity = "explanation_one_entity", E::UnsupportedRequest;
    ExplanationFastMode = "explanation_fast_mode", E::UnsupportedRequest;
    OptionsUnread = "options_unread", E::UnsupportedRequest;
    OptionsUnreadStatistic = "options_unread_statistic", E::UnsupportedRequest;
    PayloadSmallerCalls = "payload_smaller_calls", E::InvalidRequest;
    PayloadNoSmallerCall = "payload_no_smaller_call", E::InvalidRequest;
    CallPastEndpoint = "call_past_endpoint", E::InvalidRequest;
    OutputsNone = "outputs_none", E::InvalidRequest;
    NotFinite = "not_finite", E::InvalidRequest;
    PastInt64 = "past_int64", E::InvalidRequest;
    ColumnTimePastNanoseconds = "column_time_past_nanoseconds", E::ValueOutOfRange;
    LagNegative = "lag_negative", E::InvalidRequest;
    LagNotWhole = "lag_not_whole", E::InvalidRequest;
    LagNotANumber = "lag_not_a_number", E::InvalidRequest;
    LagPastTimes = "lag_past_times", E::ValueOutOfRange;
    NumContextSamplesNotWhole = "num_context_samples_not_whole", E::InvalidRequest;
    NumContextSamplesBelowOne = "num_context_samples_below_one", E::InvalidRequest;
    NumContextSamplesPastMost = "num_context_samples_past_most", E::UnsupportedRequest;
    NumHopsNotWhole = "num_hops_not_whole", E::InvalidRequest;
    NumHopsNone = "num_hops_none", E::InvalidRequest;
    RandomSeedNotWhole = "random_seed_not_whole", E::InvalidRequest;
    TopKNotWhole = "top_k_not_whole", E::InvalidRequest;
    TopKOutOfRange = "top_k_out_of_range", E::InvalidRequest;
    RowsPerCall = "rows_per_call", E::InvalidRequest;
    MaxPqIterationsNotAWholeNumber = "max_pq_iterations_not_a_whole_number", E::InvalidRequest;
    IterationsBelowOne = "iterations_below_one", E::InvalidRequest;
    OptionNotWhole = "option_not_whole", E::InvalidRequest;
    OptionUnknown = "option_unknown", E::InvalidRequest;
    OptionNotABoolean = "option_not_a_boolean", E::InvalidRequest;
    NeighborsNotASequence = "neighbors_not_a_sequence", E::InvalidRequest;
    HopNotWhole = "hop_not_whole", E::InvalidRequest;
    HopNegative = "hop_negative", E::InvalidRequest;
    ContextAnchorNotATime = "context_anchor_not_a_time", E::InvalidRequest;
    ExcludedNotAMapping = "excluded_not_a_mapping", E::InvalidRequest;
    ExcludedNotASequence = "excluded_not_a_sequence", E::InvalidRequest;
    StatisticUnknown = "statistic_unknown", E::InvalidRequest;
    StatisticNotARegression = "statistic_not_a_regression", E::InvalidRequest;
    NoHopNumHops = "no_hop_num_hops", E::InvalidRequest;
    NoHopNumNeighbors = "no_hop_num_neighbors", E::InvalidRequest;
    HopsPastEndpoint = "hops_past_endpoint", E::UnsupportedRequest;
    HopsPastEndpointUncounted = "hops_past_endpoint_uncounted", E::UnsupportedRequest;
    InferenceConfigNotAnObject = "inference_config_not_an_object", E::InvalidRequest;
    InferenceOptionUnknown = "inference_option_unknown", E::InvalidRequest;
    InferenceOptionNotABoolean = "inference_option_not_a_boolean", E::InvalidRequest;
    InferenceOptionNotOneToFour = "inference_option_not_one_to_four", E::InvalidRequest;
    InferenceOptionNotTransforms = "inference_option_not_transforms", E::InvalidRequest;
    InferenceOptionNotAnOutputType = "inference_option_not_an_output_type", E::InvalidRequest;
    SamplerStateUnread = "sampler_state_unread", E::InvalidRequest;
    TimeConstantOffset = "time_constant_offset", E::InvalidRequest;
    TimeConstantPast = "time_constant_past", E::InvalidRequest;
    TimeColumnNotATime = "time_column_not_a_time", E::InvalidRequest;
    OptionsWithoutValues = "options_without_values", E::InvalidRequest;
    CompositeKeyFloat = "composite_key_float", E::InvalidRequest;
    CompositeKeyNameLong = "composite_key_name_long", E::InvalidRequest;
    CompositeKeyNameTaken = "composite_key_name_taken", E::InvalidRequest;
    ReservedName = "reserved_name", E::InvalidRequest;
    ReservedNames = "reserved_names", E::InvalidRequest;
    SchemaLinkUnread = "schema_link_unread", E::InvalidRequest;
    WireReservedOne = "wire_reserved_one", E::InvalidSchema;
    WireReservedSeveral = "wire_reserved_several", E::InvalidSchema;
    IndicesAbsent = "indices_absent", E::InvalidRequest;
    IndicesNotOneColumn = "indices_not_one_column", E::InvalidRequest;
    IndicesEmpty = "indices_empty", E::InvalidRequest;
    IndexMissing = "index_missing", E::InvalidRequest;
    AnchorEntityNotStatic = "anchor_entity_not_static", E::InvalidRequest;
    AnchorEntityNoTimeColumn = "anchor_entity_no_time_column", E::InvalidRequest;
    AnchorEntityNotShared = "anchor_entity_not_shared", E::InvalidRequest;
    IterationsWholeForWindow = "iterations_whole_for_window", E::InvalidRequest;
    EntityTableEmpty = "entity_table_empty", E::InvalidRequest;
    EntityKeyMissing = "entity_key_missing", E::InvalidRequest;
    TargetNotAFeature = "target_not_a_feature", E::InvalidRequest;
    ContextPastRows = "context_past_rows", E::InvalidRequest;
    HeldBySeveralRowsOne = "held_by_several_rows_one", E::InvalidRequest;
    HeldBySeveralRows = "held_by_several_rows", E::InvalidRequest;
    HeldBySeveralRowsMany = "held_by_several_rows_many", E::InvalidRequest;
    NotInTable = "not_in_table", E::InvalidRequest;
    SubgraphPastTables = "subgraph_past_tables", E::InvalidRequest;
    Unresolved = "unresolved", E::InvalidRequest;
    AnchorNotATime = "anchor_not_a_time", E::InvalidRequest;
    AnchorBeforeData = "anchor_before_data", E::InvalidRequest;
    ContextBeforeDataRefused = "context_before_data_refused", E::InvalidRequest;
    ExcludedUnknownTable = "excluded_unknown_table", E::InvalidRequest;
    ExcludedNotAFeature = "excluded_not_a_feature", E::InvalidRequest;
    NullComparedNowhere = "null_compared_nowhere", E::InvalidRequest;
    NoUniqueForeignKey = "no_unique_foreign_key", E::InvalidRequest;
    ContextTooRestrictive = "context_too_restrictive", E::InvalidRequest;
    NoEntityAtAnchor = "no_entity_at_anchor", E::InvalidRequest;
    NoEntityAtAnchorLater = "no_entity_at_anchor_later", E::InvalidRequest;
    NoEntityPassed = "no_entity_passed", E::InvalidRequest;
    AnchorTooLate = "anchor_too_late", E::InvalidRequest;
    NoEntityPassedSeveral = "no_entity_passed_several", E::InvalidRequest;
    AnchorTooLateSeveral = "anchor_too_late_several", E::InvalidRequest;
    ForecastTopK = "forecast_top_k", E::InvalidRequest, rule "forecast_top_k";
    MostEntities = "most_entities", E::UnsupportedRequest, rule "most_entities";
    MostEntitiesParts = "most_entities_parts", E::UnsupportedRequest, rule "most_entities";
    MostCalls = "most_calls", E::UnsupportedRequest, rule "most_calls";
    NotYetMaterialize = "not_yet_materialize", E::UnsupportedRequest;
    MissingIndexTable = "missing_index_table", W::MissingIndices;
    MissingIndexTables = "missing_index_tables", W::MissingIndices;
    MissingIndicesTable = "missing_indices_table", W::MissingIndices;
    MissingIndicesTables = "missing_indices_tables", W::MissingIndices;
    ContextBeforeData = "context_before_data", W::ContextBeforeData;
    ContextBeforeDataUntimed = "context_before_data_untimed", W::ContextBeforeData;
    AnchorAfterData = "anchor_after_data", W::AnchorAfterData;
    Float32Overflow = "float32_overflow", W::Float32Overflow;
    ContextAfterAnchor = "context_after_anchor", W::ContextAfterAnchor;
    ContextLeaks = "context_leaks", W::ContextLeaks;
    HopsIgnored = "hops_ignored", W::HopsIgnored;
    ContextCoverage = "context_coverage", W::ContextCoverage;
    ContextCapped = "context_capped", W::ContextCapped;
    TablesUnlinkedOne = "tables_unlinked_one", W::TablesUnlinked;
    TablesUnreachedOne = "tables_unreached_one", W::TablesUnreached;
    ColumnsUnreadOne = "columns_unread_one", W::ColumnsUnread;
    TablesUnlinkedSeveral = "tables_unlinked_several", W::TablesUnlinked;
    TablesUnreachedSeveral = "tables_unreached_several", W::TablesUnreached;
    ColumnsUnreadSeveral = "columns_unread_several", W::ColumnsUnread;
    FilterChoseExamples = "filter_chose_examples", W::FilterChoseExamples;
    ContextAnswersWindowed = "context_answers_windowed", W::ContextAnswers;
    ContextAnswersStatic = "context_answers_static", W::ContextAnswers;
    NullKeysLeftOut = "null_keys_left_out", W::NullKeysLeftOut;
    DateLiteralDayFirst = "date_literal_day_first", W::DateLiteralDayFirst;
    EntityUncast = "entity_uncast", E::InvalidRequest;
    TimePastNanoseconds = "time_past_nanoseconds", E::ValueOutOfRange;
    TimePastUnit = "time_past_unit", E::ValueOutOfRange;
    InListFraction = "in_list_fraction", E::InvalidRequest;
    RenderingRefused = "rendering_refused", E::UnsupportedRequest;
    NoDialect = "no_dialect", E::UnsupportedRequest;
    NoDialectNamed = "no_dialect_named", E::UnsupportedRequest;
    ValueNotJson = "value_not_json", E::ProviderSchemaMismatch;
    ValueNotAtScale = "value_not_at_scale", E::ProviderSchemaMismatch;
    TextNotUtf8 = "text_not_utf8", E::ValueOutOfRange, rule "text_not_utf8";
    UnsupportedTypeUnread = "unsupported_type_unread", E::UnsupportedType;
    AdmittedPastType = "admitted_past_type", E::ValueOutOfRange;
    InvalidConfiguration = "invalid_configuration", E::InvalidConfiguration;
    PartForecastStep = "part_forecast_step", Code::Part;
    PartForecastStepOutran = "part_forecast_step_outran", Code::Part;
    PartRankingCandidate = "part_ranking_candidate", Code::Part;
}
