// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The parts of a query that choose its context examples by the entity's
//! own row, as value filters ([`sdm_contract::data::FilterKind`]) on the
//! entity sample, so a provider draws from the rows that may qualify
//! rather than from every row.
//!
//! A value filter narrows the read and never decides an example: the
//! labels check every example by the query's own rules afterwards. So each
//! filter written here holds for every row the query's condition holds
//! for, and may hold for more. Where that is not certain for a column's
//! type, the part is not written:
//!
//! - an integer or a boolean column compares exactly;
//! - a float or decimal column, which the labels compare at float32, is
//!   compared over a range widened past float32's rounding;
//! - a text column is compared for equality and membership only, since a
//!   source's collation may order text otherwise and may not tell case
//!   apart;
//! - a time, or any other type, is not compared.
//!
//! `NOT` is written as its parts negated, a null meeting the negation, as
//! a null meets no comparison; a negated float comparison, which a NaN may
//! meet in the labels and not in SQL, or a negated membership, is not
//! written.

use sdm_contract::data::{Bound, CompareOp, Filter};

use super::*;

/// The value filters choosing the context examples by the entity's own
/// row: each of the query's choosing conditions (its entity filter and its
/// assumption, [`Asked::choosing`]) that can be written, and, for a target
/// of the entity's own column, that column holding a value, since an
/// example without one has no label.
pub(super) fn entity_filters(asked: &Asked, entity: &Table) -> Vec<Filter> {
    let mut filters: Vec<Filter> = asked
        .choosing
        .iter()
        .filter_map(|shape| written(shape, entity, false))
        .collect();
    if let Shape::Column(column) = &asked.target {
        filters.push(Filter::NotNull {
            column: column.clone(),
        });
    }
    filters
}

/// `shape` as a value filter holding wherever it does (or, `negated`,
/// wherever it does not), or none where that is not certain.
fn written(shape: &Shape, entity: &Table, negated: bool) -> Option<Filter> {
    match shape {
        Shape::Condition { target, test } => {
            let Shape::Column(column) = target.as_ref() else {
                // A condition of an aggregation reads other tables' rows.
                return None;
            };
            let kind = Kind::of(&entity.column(column)?.received)?;
            match test {
                Test::Compare(op, Some(constant)) => {
                    compared(column, kind, rel(*op), constant, negated)
                }
                // `= NULL` meets no row in the labels; not written.
                Test::Compare(_, None) => None,
                Test::In(values) if !negated => member(column, kind, values),
                Test::In(_) => None,
            }
        }
        Shape::Not(inner) => written(inner, entity, !negated),
        Shape::Both { op, left, right } => {
            // `NOT (a AND b)` is `NOT a OR NOT b`, and `NOT (a OR b)` is
            // `NOT a AND NOT b`.
            let conjunction = matches!(op, labels::BoolOp::And) != negated;
            let left = written(left, entity, negated);
            let right = written(right, entity, negated);
            if conjunction {
                // Each part kept narrows it; one not written leaves the
                // other.
                let parts: Vec<Filter> = left.into_iter().chain(right).collect();
                match parts.len() {
                    0 => None,
                    1 => parts.into_iter().next(),
                    _ => Some(Filter::AllOf { filters: parts }),
                }
            } else {
                // A disjunction holds only where one of its parts is
                // written for certain.
                Some(Filter::AnyOf {
                    filters: vec![left?, right?],
                })
            }
        }
        Shape::Column(_) | Shape::Aggregated(_) | Shape::Links(_) => None,
    }
}

/// A window of rows of a table referring to the entity's that an example
/// needs at least `at_least` of to qualify: a choosing `COUNT` condition's
/// floor, or the one row with a value an `AVG`, `MIN` or `MAX` needs to
/// have an answer.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct WindowCheck {
    pub table: String,
    pub start: Offset,
    pub end: Offset,
    /// Not-null and value filters on the table's rows.
    pub filters: Vec<Filter>,
    pub at_least: u64,
}

/// The window checks an example must pass: each choosing condition's,
/// among the parts of a conjunction, and the target's, where an
/// aggregation of no rows gives it no answer. A part that is not certain
/// to fail with fewer rows (`SUM`, `COUNT <`, `NOT`, `OR`) gives none.
pub(super) fn window_checks(asked: &Asked, graph: &Graph) -> Vec<WindowCheck> {
    let mut checks = Vec::new();
    for shape in &asked.choosing {
        choosing_checks(shape, graph, &mut checks);
    }
    let target = match &asked.target {
        Shape::Condition { target, .. } => target.as_ref(),
        other => other,
    };
    if let Shape::Aggregated(windowed) = target
        && let Some(check) = answered(windowed, graph)
    {
        checks.push(check);
    }
    checks.dedup();
    checks
}

fn choosing_checks(shape: &Shape, graph: &Graph, into: &mut Vec<WindowCheck>) {
    match shape {
        Shape::Both {
            op: labels::BoolOp::And,
            left,
            right,
        } => {
            choosing_checks(left, graph, into);
            choosing_checks(right, graph, into);
        }
        Shape::Condition { target, test } => {
            let Shape::Aggregated(windowed) = target.as_ref() else {
                return;
            };
            let check = match windowed.aggregation {
                Some(labels::Aggregation::Count) => match test {
                    Test::Compare(op, Some(constant)) => {
                        // `COUNT(table.column)` counts the rows holding a
                        // value, as `answered_by` asks.
                        count_floor(*op, constant)
                            .and_then(|at_least| answered_by(windowed, graph, at_least))
                    }
                    _ => None,
                },
                Some(
                    labels::Aggregation::Avg | labels::Aggregation::Min | labels::Aggregation::Max,
                ) => answered(windowed, graph),
                _ => None,
            };
            into.extend(check);
        }
        _ => {}
    }
}

/// The fewest rows a `COUNT op constant` holds with: `> c` one past
/// `c`'s floor, `>= c` its ceiling, `= c` itself; none where no rows may
/// meet it (a bound under one) or more rows do not keep it (`<`, `<=`,
/// `<>`).
fn count_floor(op: labels::RelOp, constant: &labels::Constant) -> Option<u64> {
    let c = match constant {
        labels::Constant::Int(i) => *i as f64,
        labels::Constant::Float(f) if f.is_finite() => *f,
        _ => return None,
    };
    let at_least = match op {
        labels::RelOp::Gt => c.floor() + 1.0,
        labels::RelOp::Ge => c.ceil(),
        labels::RelOp::Eq if c.fract() == 0.0 => c,
        _ => return None,
    };
    (at_least >= 1.0 && at_least < u64::MAX as f64).then_some(at_least as u64)
}

/// The check that `windowed`, an `AVG`, `MIN` or `MAX`, has an answer:
/// one row in its window with a value.
fn answered(windowed: &Windowed, graph: &Graph) -> Option<WindowCheck> {
    match windowed.aggregation {
        Some(labels::Aggregation::Avg | labels::Aggregation::Min | labels::Aggregation::Max) => {
            answered_by(windowed, graph, 1)
        }
        _ => None,
    }
}

/// `at_least` rows of `windowed`'s window: its column holding a value, and
/// its filter's parts written for the aggregated table.
fn answered_by(windowed: &Windowed, graph: &Graph, at_least: u64) -> Option<WindowCheck> {
    let table = graph.table(&windowed.table)?;
    let mut filters = Vec::new();
    if let Some(column) = &windowed.column {
        filters.push(Filter::NotNull {
            column: column.clone(),
        });
    }
    if let Some(filter) = &windowed.filter
        && let Some(written) = written(filter, table, false)
    {
        filters.push(written);
    }
    Some(WindowCheck {
        table: windowed.table.clone(),
        start: windowed.start,
        end: windowed.end,
        filters,
        at_least,
    })
}

/// A day, in nanoseconds: how far each bound of a window check is moved
/// out, past any zone a source compares a time in.
const DAY: i64 = 86_400_000_000_000;

/// `check` at `anchor`, as a filter on the entity table keyed by `key`:
/// none where the referring table's time is not a time the sources
/// compare with a literal (text a source may store a time as), the link is
/// not one foreign key, or a bound is past what nanoseconds hold. The
/// window, rows after `anchor + start` and not after `anchor + end` in the
/// labels, is widened by a day each way.
pub(super) fn has_rows(
    check: &WindowCheck,
    graph: &Graph,
    entity: &Table,
    key: &str,
    anchor: i64,
) -> Option<Filter> {
    let table = graph.table(&check.table)?;
    let time = table.time_column.as_ref()?;
    if !matches!(
        table.column(time)?.received,
        ColumnType::Timestamp { .. } | ColumnType::Date32
    ) {
        return None;
    }
    let edges: Vec<_> = graph
        .edges
        .iter()
        .filter(|e| e.src == table.name && e.dst == entity.name)
        .collect();
    let [edge] = edges.as_slice() else {
        return None;
    };
    let bound = |at: i64| Bound {
        value: Scalar::Timestamp {
            unit: Unit::Ns,
            zone: None,
            value: at,
        },
        inclusive: true,
    };
    let lower = check
        .start
        .add_to(anchor, "anchor_time")
        .ok()?
        .checked_sub(DAY)?;
    let upper = check
        .end
        .add_to(anchor, "anchor_time")
        .ok()?
        .checked_add(DAY)?;
    let mut filters = vec![Filter::TimeRange {
        column: time.clone(),
        lower: Some(bound(lower)),
        upper: Some(bound(upper)),
    }];
    filters.extend(check.filters.iter().cloned());
    Some(Filter::HasRows {
        column: key.to_owned(),
        table: table.source.clone(),
        key: edge.fkey.clone(),
        filters,
        at_least: check.at_least,
    })
}

/// What a column's values compare as, by the type the source hands over.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Integer,
    /// Floats and decimals, which the labels compare at float32.
    Float,
    Bool,
    Text,
}

impl Kind {
    fn of(received: &ColumnType) -> Option<Kind> {
        Some(match received {
            ColumnType::Int8
            | ColumnType::Int16
            | ColumnType::Int32
            | ColumnType::Int64
            | ColumnType::Uint8
            | ColumnType::Uint16
            | ColumnType::Uint32
            | ColumnType::Uint64 => Kind::Integer,
            ColumnType::Float32
            | ColumnType::Float64
            | ColumnType::Decimal128 { .. }
            | ColumnType::Decimal256 { .. } => Kind::Float,
            ColumnType::Bool => Kind::Bool,
            ColumnType::Utf8 | ColumnType::LargeUtf8 => Kind::Text,
            _ => return None,
        })
    }
}

fn rel(op: labels::RelOp) -> CompareOp {
    match op {
        labels::RelOp::Eq => CompareOp::Eq,
        labels::RelOp::Ne => CompareOp::Ne,
        labels::RelOp::Lt => CompareOp::Lt,
        labels::RelOp::Le => CompareOp::Le,
        labels::RelOp::Gt => CompareOp::Gt,
        labels::RelOp::Ge => CompareOp::Ge,
    }
}

fn negation(op: CompareOp) -> CompareOp {
    match op {
        CompareOp::Eq => CompareOp::Ne,
        CompareOp::Ne => CompareOp::Eq,
        CompareOp::Lt => CompareOp::Ge,
        CompareOp::Le => CompareOp::Gt,
        CompareOp::Gt => CompareOp::Le,
        CompareOp::Ge => CompareOp::Lt,
    }
}

/// `column op constant`, or its negation with a null meeting it.
fn compared(
    column: &str,
    kind: Kind,
    op: CompareOp,
    constant: &labels::Constant,
    negated: bool,
) -> Option<Filter> {
    let op = if negated { negation(op) } else { op };
    let filter = match kind {
        Kind::Integer => Filter::Compare {
            column: column.to_owned(),
            op,
            value: number(constant)?,
        },
        Kind::Float => {
            if negated {
                return None;
            }
            float_range(column, op, float(constant)?)?
        }
        Kind::Bool => match (op, constant) {
            (CompareOp::Eq | CompareOp::Ne, labels::Constant::Bool(b)) => Filter::Compare {
                column: column.to_owned(),
                op,
                value: Scalar::Bool(*b),
            },
            _ => return None,
        },
        Kind::Text => match (op, constant) {
            (CompareOp::Eq, labels::Constant::Text(t)) => Filter::Compare {
                column: column.to_owned(),
                op,
                value: Scalar::Utf8(t.clone()),
            },
            _ => return None,
        },
    };
    Some(if negated {
        Filter::AnyOf {
            filters: vec![
                filter,
                Filter::IsNull {
                    column: column.to_owned(),
                },
            ],
        }
    } else {
        filter
    })
}

/// `column IN (values)`, or none where one value is not written.
fn member(column: &str, kind: Kind, values: &[labels::Constant]) -> Option<Filter> {
    match kind {
        Kind::Integer => Some(Filter::In {
            column: column.to_owned(),
            values: values.iter().map(number).collect::<Option<Vec<_>>>()?,
        }),
        Kind::Text => Some(Filter::In {
            column: column.to_owned(),
            values: values
                .iter()
                .map(|v| match v {
                    labels::Constant::Text(t) => Some(Scalar::Utf8(t.clone())),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?,
        }),
        Kind::Bool => Some(Filter::In {
            column: column.to_owned(),
            values: values
                .iter()
                .map(|v| match v {
                    labels::Constant::Bool(b) => Some(Scalar::Bool(*b)),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?,
        }),
        Kind::Float => {
            let ranges = values
                .iter()
                .map(|v| float_range(column, CompareOp::Eq, float(v)?))
                .collect::<Option<Vec<_>>>()?;
            Some(Filter::AnyOf { filters: ranges })
        }
    }
}

/// An integer or float constant as a number an integer column compares
/// with as the labels do: exactly, an integer as itself and a float as a
/// double.
fn number(constant: &labels::Constant) -> Option<Scalar> {
    match constant {
        labels::Constant::Int(i) => i64::try_from(*i).ok().map(Scalar::Int64),
        labels::Constant::Float(f) if f.is_finite() => Some(Scalar::Float64(*f)),
        _ => None,
    }
}

fn float(constant: &labels::Constant) -> Option<f64> {
    match constant {
        labels::Constant::Int(i) => Some(*i as f64),
        labels::Constant::Float(f) => Some(*f),
        _ => None,
    }
    .filter(|f| f.is_finite() && f.abs() <= f64::from(f32::MAX))
}

/// `column op value` for a float or decimal column the labels compare at
/// float32: every value that rounds to the same float32 as `value` may
/// compare equal there, so each bound is moved out by more than float32's
/// rounding, a millionth of the value or of one; `<>` needs no widening
/// (only `value` itself fails it in both).
fn float_range(column: &str, op: CompareOp, value: f64) -> Option<Filter> {
    let slack = value.abs().max(1.0) * 1e-6;
    let bound = |op: CompareOp, value: f64| Filter::Compare {
        column: column.to_owned(),
        op,
        value: Scalar::Float64(value),
    };
    Some(match op {
        CompareOp::Ne => bound(CompareOp::Ne, value),
        CompareOp::Gt | CompareOp::Ge => bound(CompareOp::Ge, value - slack),
        CompareOp::Lt | CompareOp::Le => bound(CompareOp::Le, value + slack),
        CompareOp::Eq => Filter::AllOf {
            filters: vec![
                bound(CompareOp::Ge, value - slack),
                bound(CompareOp::Le, value + slack),
            ],
        },
    })
}

#[cfg(test)]
mod tests {
    use sdm_contract::ops::DescribedColumn;

    use super::*;

    fn table(columns: &[(&str, ColumnType)]) -> Table {
        Table {
            name: "users".into(),
            source: sdm_contract::data::Relation {
                columns: vec![],
                filter: None,
                found: false,
                source: vec!["users".into()],
            },
            columns: columns
                .iter()
                .map(|(name, received)| DescribedColumn {
                    name: (*name).into(),
                    received: received.clone(),
                    dtype: Dtype::Unsupported,
                    stype: Stype::Numerical,
                    stype_before_key: None,
                    nullable: true,
                    unique: false,
                    declared_type: None,
                    catalog: None,
                    exact_number: None,
                })
                .collect(),
            primary_key: Some("id".into()),
            time_column: None,
            end_time_column: None,
        }
    }

    fn users() -> Table {
        table(&[
            ("id", ColumnType::Int64),
            ("age", ColumnType::Int64),
            ("score", ColumnType::Float64),
            ("name", ColumnType::Utf8),
            ("active", ColumnType::Bool),
            ("seen", ColumnType::Date32),
        ])
    }

    fn condition(column: &str, op: labels::RelOp, constant: labels::Constant) -> Shape {
        Shape::Condition {
            target: Box::new(Shape::Column(column.into())),
            test: Test::Compare(op, Some(constant)),
        }
    }

    fn compare(column: &str, op: CompareOp, value: Scalar) -> Filter {
        Filter::Compare {
            column: column.into(),
            op,
            value,
        }
    }

    #[test]
    fn an_integer_comparison_is_written_exactly() {
        let shape = condition("age", labels::RelOp::Gt, labels::Constant::Int(76));
        assert_eq!(
            written(&shape, &users(), false),
            Some(compare("age", CompareOp::Gt, Scalar::Int64(76)))
        );
    }

    #[test]
    fn a_negation_holds_for_nulls_too() {
        let shape = Shape::Not(Box::new(condition(
            "age",
            labels::RelOp::Gt,
            labels::Constant::Int(76),
        )));
        assert_eq!(
            written(&shape, &users(), false),
            Some(Filter::AnyOf {
                filters: vec![
                    compare("age", CompareOp::Le, Scalar::Int64(76)),
                    Filter::IsNull {
                        column: "age".into()
                    },
                ]
            })
        );
    }

    #[test]
    fn a_float_bound_is_widened_past_float32_rounding() {
        let shape = condition("score", labels::RelOp::Gt, labels::Constant::Float(2.0));
        let Some(Filter::Compare {
            op: CompareOp::Ge,
            value: Scalar::Float64(bound),
            ..
        }) = written(&shape, &users(), false)
        else {
            panic!()
        };
        // A double just under 2 that float32 rounds to 2 is kept.
        let under = f64::from(2.0f32) - 1e-9;
        assert!(under as f32 == 2.0f32 && under >= bound);
        // A negated float comparison is not written.
        assert_eq!(written(&Shape::Not(Box::new(shape)), &users(), false), None);
    }

    #[test]
    fn text_is_written_for_equality_and_membership_only() {
        let text = || labels::Constant::Text("ann".into());
        assert!(
            written(
                &condition("name", labels::RelOp::Eq, text()),
                &users(),
                false
            )
            .is_some()
        );
        for op in [labels::RelOp::Ne, labels::RelOp::Lt, labels::RelOp::Ge] {
            assert_eq!(
                written(&condition("name", op, text()), &users(), false),
                None
            );
        }
        let member = Shape::Condition {
            target: Box::new(Shape::Column("name".into())),
            test: Test::In(vec![text()]),
        };
        assert_eq!(
            written(&member, &users(), false),
            Some(Filter::In {
                column: "name".into(),
                values: vec![Scalar::Utf8("ann".into())]
            })
        );
        assert_eq!(
            written(&Shape::Not(Box::new(member)), &users(), false),
            None
        );
    }

    #[test]
    fn a_conjunction_keeps_the_parts_written_and_a_disjunction_needs_both() {
        let age = || condition("age", labels::RelOp::Gt, labels::Constant::Int(76));
        let seen = || condition("seen", labels::RelOp::Gt, labels::Constant::Time(0));
        let both = |op| Shape::Both {
            op,
            left: Box::new(age()),
            right: Box::new(seen()),
        };
        assert_eq!(
            written(&both(labels::BoolOp::And), &users(), false),
            Some(compare("age", CompareOp::Gt, Scalar::Int64(76)))
        );
        assert_eq!(written(&both(labels::BoolOp::Or), &users(), false), None);
        // `NOT (a OR b)` is `NOT a AND NOT b`: the part written stays.
        assert_eq!(
            written(
                &Shape::Not(Box::new(both(labels::BoolOp::Or))),
                &users(),
                false
            ),
            Some(Filter::AnyOf {
                filters: vec![
                    compare("age", CompareOp::Le, Scalar::Int64(76)),
                    Filter::IsNull {
                        column: "age".into()
                    },
                ]
            })
        );
    }

    #[test]
    fn a_condition_of_an_unknown_column_or_a_null_is_not_written() {
        let missing = condition("nope", labels::RelOp::Gt, labels::Constant::Int(1));
        assert_eq!(written(&missing, &users(), false), None);
        let null = Shape::Condition {
            target: Box::new(Shape::Column("age".into())),
            test: Test::Compare(labels::RelOp::Eq, None),
        };
        assert_eq!(written(&null, &users(), false), None);
    }

    #[test]
    fn a_count_condition_needs_as_many_rows_as_its_floor() {
        use labels::{Constant, RelOp};
        assert_eq!(count_floor(RelOp::Gt, &Constant::Int(1)), Some(2));
        assert_eq!(count_floor(RelOp::Gt, &Constant::Float(1.5)), Some(2));
        assert_eq!(count_floor(RelOp::Ge, &Constant::Float(1.5)), Some(2));
        assert_eq!(count_floor(RelOp::Ge, &Constant::Int(3)), Some(3));
        assert_eq!(count_floor(RelOp::Eq, &Constant::Int(2)), Some(2));
        // No rows may meet these, or more rows do not keep them.
        assert_eq!(count_floor(RelOp::Gt, &Constant::Int(-1)), None);
        assert_eq!(count_floor(RelOp::Ge, &Constant::Int(0)), None);
        assert_eq!(count_floor(RelOp::Eq, &Constant::Int(0)), None);
        assert_eq!(count_floor(RelOp::Eq, &Constant::Float(1.5)), None);
        for op in [RelOp::Lt, RelOp::Le, RelOp::Ne] {
            assert_eq!(count_floor(op, &Constant::Int(5)), None);
        }
    }
}
