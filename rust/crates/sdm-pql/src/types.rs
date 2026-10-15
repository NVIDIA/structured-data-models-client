// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The enumerations of `predictive_query.schema.typing`, `schema.task` and
//! `schema.utils`: types, units, problem types, aggregations and operators.
//! Each is written as Python writes its value.

// Only documentation passes through: an attribute taken whole (`$doc:meta`)
// could be an `allow` of a clippy ban, which l0_scan.py refuses.
macro_rules! str_enum {
    ($(#[doc = $doc:literal])* $name:ident { $($variant:ident => $text:literal),+ $(,)? }) => {
        $(#[doc = $doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum $name {
            $($variant),+
        }

        impl $name {
            /// Every member, in definition order.
            pub const ALL: &'static [$name] = &[$($name::$variant),+];

            /// The member's value, as Python's `str()` writes it.
            pub fn value(self) -> &'static str {
                match self {
                    $($name::$variant => $text),+
                }
            }

            /// The member with this value, as `Enum(value)` finds it.
            pub fn parse(text: &str) -> Option<Self> {
                match text {
                    $($text => Some($name::$variant),)+
                    _ => None,
                }
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.value())
            }
        }
    };
}

str_enum!(
    /// The semantic type of a column.
    Stype {
        Numerical => "numerical",
        Categorical => "categorical",
        Multicategorical => "multicategorical",
        Id => "ID",
        Text => "text",
        Timestamp => "timestamp",
        Sequence => "sequence",
        Image => "image",
        Unsupported => "unsupported",
    }
);

str_enum!(
    /// The data type of a column.
    Dtype {
        Bool => "bool",
        Int => "int",
        Byte => "byte",
        Int16 => "int16",
        Int32 => "int32",
        Int64 => "int64",
        Float => "float",
        Float32 => "float32",
        Float64 => "float64",
        String => "string",
        Binary => "binary",
        Date => "date",
        Time => "time",
        Timedelta => "timedelta",
        Floatlist => "floatlist",
        Intlist => "intlist",
        Stringlist => "stringlist",
        Unsupported => "unsupported",
    }
);

str_enum!(
    /// The unit of a time window.
    TimeUnit {
        Seconds => "seconds",
        Minutes => "minutes",
        Hours => "hours",
        Days => "days",
        Weeks => "weeks",
        Months => "months",
    }
);

str_enum!(
    /// `RANK`, `CLASSIFY` or `FORECAST`.
    ProblemType {
        Rank => "RANK",
        Classify => "CLASSIFY",
        Forecast => "FORECAST",
    }
);

str_enum!(
    /// The aggregations PQL knows.
    AggregationType {
        Sum => "SUM",
        Avg => "AVG",
        Min => "MIN",
        Max => "MAX",
        Count => "COUNT",
        CountDistinct => "COUNT_DISTINCT",
        First => "FIRST",
        Last => "LAST",
        ListDistinct => "LIST_DISTINCT",
    }
);

str_enum!(
    /// Relational operators.
    RelOp {
        Neq => "!=",
        Leq => "<=",
        Geq => ">=",
        Eq => "=",
        Lt => "<",
        Gt => ">",
    }
);

str_enum!(
    /// Membership operators.
    MemberOp {
        IsIn => "IS IN",
        In => "IN",
    }
);

str_enum!(
    /// String operators.
    StrOp {
        StartsWith => "STARTS WITH",
        EndsWith => "ENDS WITH",
        Contains => "CONTAINS",
        NotContains => "NOT CONTAINS",
    }
);

str_enum!(
    /// Boolean operators.
    BoolOp {
        And => "AND",
        Or => "OR",
        Not => "NOT",
    }
);

str_enum!(
    /// What a predictive query asks the model to do.
    TaskType {
        BinaryClassification => "binary_classification",
        MulticlassClassification => "multiclass_classification",
        MultilabelClassification => "multilabel_classification",
        MultilabelRanking => "multilabel_ranking",
        Regression => "regression",
        TemporalLinkPrediction => "temporal_link_prediction",
        StaticLinkPrediction => "static_link_prediction",
        Forecasting => "forecasting",
        LinkPrediction => "link_prediction",
    }
);

str_enum!(
    /// Whether a query's target carries a time window.
    QueryType {
        Static => "static",
        Temporal => "temporal",
    }
);

str_enum!(
    /// The unit of a timestamp column.
    TimestampUnit {
        Second => "s",
        Millisecond => "ms",
        Microsecond => "us",
        Nanosecond => "ns",
    }
);

impl Dtype {
    pub fn is_bool(self) -> bool {
        self == Dtype::Bool
    }

    pub fn is_int(self) -> bool {
        matches!(
            self,
            Dtype::Int | Dtype::Byte | Dtype::Int16 | Dtype::Int32 | Dtype::Int64
        )
    }

    pub fn is_float(self) -> bool {
        matches!(self, Dtype::Float | Dtype::Float32 | Dtype::Float64)
    }

    pub fn is_numerical(self) -> bool {
        self.is_int() || self.is_float() || self == Dtype::Timedelta
    }

    pub fn is_string(self) -> bool {
        matches!(self, Dtype::String | Dtype::Binary)
    }

    pub fn is_timestamp(self) -> bool {
        matches!(self, Dtype::Date | Dtype::Time)
    }

    pub fn is_maybe_timestamp(self) -> bool {
        self.is_timestamp() || self == Dtype::String
    }

    pub fn is_list(self) -> bool {
        matches!(self, Dtype::Floatlist | Dtype::Intlist | Dtype::Stringlist)
    }

    pub fn is_unsupported(self) -> bool {
        self == Dtype::Unsupported
    }

    /// The semantic type a column of this data type gets by default.
    pub fn default_stype(self) -> Stype {
        if self.is_bool() {
            Stype::Categorical
        } else if self.is_numerical() {
            Stype::Numerical
        } else if self == Dtype::Binary {
            Stype::Categorical
        } else if self == Dtype::String {
            Stype::Text
        } else if self.is_timestamp() {
            Stype::Timestamp
        } else if self == Dtype::Stringlist {
            Stype::Multicategorical
        } else if matches!(self, Dtype::Floatlist | Dtype::Intlist) {
            Stype::Sequence
        } else {
            Stype::Unsupported
        }
    }
}

impl Stype {
    /// Whether a column of this semantic type may hold `dtype`.
    pub fn supports_dtype(self, dtype: Dtype) -> bool {
        match self {
            Stype::Numerical => dtype.is_numerical(),
            Stype::Categorical => dtype.is_bool() || dtype.is_numerical() || dtype.is_string(),
            Stype::Multicategorical => dtype.is_string() || dtype.is_list(),
            Stype::Id => dtype.is_int() || dtype.is_string() || dtype.is_float(),
            Stype::Text | Stype::Image => dtype == Dtype::String,
            Stype::Timestamp => dtype.is_maybe_timestamp(),
            Stype::Sequence => matches!(dtype, Dtype::Floatlist | Dtype::Intlist | Dtype::String),
            Stype::Unsupported => true,
        }
    }
}

impl TaskType {
    pub fn node_pred_tasks() -> &'static [TaskType] {
        &[
            TaskType::BinaryClassification,
            TaskType::MulticlassClassification,
            TaskType::MultilabelClassification,
            TaskType::MultilabelRanking,
            TaskType::Regression,
            TaskType::Forecasting,
        ]
    }

    pub fn link_pred_tasks() -> &'static [TaskType] {
        &[
            TaskType::TemporalLinkPrediction,
            TaskType::StaticLinkPrediction,
            TaskType::LinkPrediction,
        ]
    }

    pub fn classification_tasks() -> &'static [TaskType] {
        &[
            TaskType::BinaryClassification,
            TaskType::MulticlassClassification,
            TaskType::MultilabelClassification,
        ]
    }

    pub fn multilabel_tasks() -> &'static [TaskType] {
        &[
            TaskType::MultilabelClassification,
            TaskType::MultilabelRanking,
        ]
    }

    pub fn ranking_tasks() -> &'static [TaskType] {
        &[
            TaskType::TemporalLinkPrediction,
            TaskType::StaticLinkPrediction,
            TaskType::LinkPrediction,
            TaskType::MultilabelRanking,
        ]
    }

    pub fn is_node_pred(self) -> bool {
        Self::node_pred_tasks().contains(&self)
    }

    pub fn is_link_pred(self) -> bool {
        Self::link_pred_tasks().contains(&self)
    }

    pub fn is_classification(self) -> bool {
        Self::classification_tasks().contains(&self)
    }

    pub fn is_multilabel(self) -> bool {
        Self::multilabel_tasks().contains(&self)
    }

    pub fn is_ranking(self) -> bool {
        Self::ranking_tasks().contains(&self)
    }
}

/// An operator a condition compares with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Rel(RelOp),
    Member(MemberOp),
    Str(StrOp),
}

impl Op {
    pub fn value(self) -> &'static str {
        match self {
            Op::Rel(op) => op.value(),
            Op::Member(op) => op.value(),
            Op::Str(op) => op.value(),
        }
    }

    /// The operator spelled `text`: relational, then string, then membership,
    /// as `Condition` looks them up.
    pub fn parse(text: &str) -> Option<Self> {
        RelOp::parse(text)
            .map(Op::Rel)
            .or_else(|| StrOp::parse(text).map(Op::Str))
            .or_else(|| MemberOp::parse(text).map(Op::Member))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_round_trip() {
        for d in Dtype::ALL {
            assert_eq!(Dtype::parse(d.value()), Some(*d));
        }
        assert_eq!(Dtype::Floatlist.default_stype(), Stype::Sequence);
        assert!(Stype::Id.supports_dtype(Dtype::Float));
        assert!(TaskType::MultilabelRanking.is_ranking());
        let families: Vec<(&str, [bool; 5])> = TaskType::ALL
            .iter()
            .map(|t| {
                (
                    t.value(),
                    [
                        t.is_node_pred(),
                        t.is_link_pred(),
                        t.is_classification(),
                        t.is_multilabel(),
                        t.is_ranking(),
                    ],
                )
            })
            .collect();
        // What today's Python's TaskType properties give.
        assert_eq!(
            families,
            [
                ("binary_classification", [true, false, true, false, false]),
                (
                    "multiclass_classification",
                    [true, false, true, false, false]
                ),
                (
                    "multilabel_classification",
                    [true, false, true, true, false]
                ),
                ("multilabel_ranking", [true, false, false, true, true]),
                ("regression", [true, false, false, false, false]),
                (
                    "temporal_link_prediction",
                    [false, true, false, false, true]
                ),
                ("static_link_prediction", [false, true, false, false, true]),
                ("forecasting", [true, false, false, false, false]),
                ("link_prediction", [false, true, false, false, true]),
            ]
        );
        assert_eq!(Op::parse("IN"), Some(Op::Member(MemberOp::In)));
        assert_eq!(Op::parse("LIKE"), None);
    }
}
