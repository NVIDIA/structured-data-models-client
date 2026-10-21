// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The syntax tree of a predictive query (`predictive_query.ast`).
//!
//! Nodes are shared and changed in place as Python's are: the validators
//! write types, join keys and time columns into them, wrap them in joins, and
//! the entity column of `FOR t.c = 1` is one node in two places. A node is
//! therefore an `Rc<RefCell<Node>>`, and every constructor runs the checks
//! and the derived state of the Python class's `__post_init__`.

use std::cell::RefCell;
use std::rc::Rc;

use crate::dates::{DayFirst, Reader, Reading, Stamp};
use crate::dor::{DateOffsetRange, DorRef};
use crate::escapes::unescape;
use crate::literal::{Literal, read_float};
use crate::names;
use crate::pyint::PyInt;
use crate::schema::ColumnKey;
use crate::types::{AggregationType, BoolOp, Dtype, Op, RelOp, StrOp, Stype};

pub type NodeRef = Rc<RefCell<Node>>;

/// Where a node is in the query: rows from 1, columns from 0, and the end
/// is the first character of the node's last token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Location {
    pub start_row: usize,
    pub start_col: usize,
    pub end_row: usize,
    pub end_col: usize,
    pub data_available: bool,
}

impl Location {
    pub fn new(start_row: usize, start_col: usize, end_row: usize, end_col: usize) -> Self {
        Location {
            start_row,
            start_col,
            end_row,
            end_col,
            data_available: true,
        }
    }

    /// The earlier start and the later end of two locations.
    pub fn merge(a: Location, b: Location) -> Location {
        if !a.data_available {
            return b;
        }
        if !b.data_available {
            return a;
        }
        let start_row = a.start_row.min(b.start_row);
        let start_col = if a.start_row == b.start_row {
            a.start_col.min(b.start_col)
        } else if a.start_row == start_row {
            a.start_col
        } else {
            b.start_col
        };
        let end_row = a.end_row.max(b.end_row);
        let end_col = if a.end_row == b.end_row {
            a.end_col.max(b.end_col)
        } else if a.end_row == end_row {
            a.end_col
        } else {
            b.end_col
        };
        Location::new(start_row, start_col, end_row, end_col)
    }

    pub fn message_start(&self) -> String {
        if !self.data_available {
            return String::new();
        }
        format!("row {}, column {}", self.start_row, self.start_col)
    }

    pub fn message_end(&self) -> String {
        if !self.data_available {
            return String::new();
        }
        format!("row {}, column {}", self.end_row, self.end_col)
    }

    pub fn message(&self) -> String {
        if !self.data_available {
            return String::new();
        }
        format!("({}: {})", self.message_start(), self.message_end())
    }
}

/// A node's data type: a `Dtype`, or an array of one (`ArrayDtype`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeDtype {
    Plain(Dtype),
    Array(Dtype),
}

impl NodeDtype {
    /// `ArrayDtype(nested)`, refused for a list type as nested.
    pub fn array(nested: Dtype) -> Result<Self, String> {
        if nested.is_list() {
            return Err(format!("ArrayDtype not supported for Dtype {nested}."));
        }
        Ok(NodeDtype::Array(nested))
    }

    pub fn plain(self) -> Option<Dtype> {
        match self {
            NodeDtype::Plain(d) => Some(d),
            NodeDtype::Array(_) => None,
        }
    }

    pub fn is_bool(self) -> bool {
        self.plain().is_some_and(Dtype::is_bool)
    }

    pub fn is_int(self) -> bool {
        self.plain().is_some_and(Dtype::is_int)
    }

    pub fn is_float(self) -> bool {
        self.plain().is_some_and(Dtype::is_float)
    }

    pub fn is_numerical(self) -> bool {
        self.plain().is_some_and(Dtype::is_numerical)
    }

    pub fn is_string(self) -> bool {
        self.plain().is_some_and(Dtype::is_string)
    }

    pub fn is_timestamp(self) -> bool {
        self.plain().is_some_and(Dtype::is_timestamp)
    }

    pub fn is_list(self) -> bool {
        match self {
            NodeDtype::Plain(d) => d.is_list(),
            NodeDtype::Array(_) => true,
        }
    }

    pub fn is_unsupported(self) -> bool {
        match self {
            NodeDtype::Plain(d) | NodeDtype::Array(d) => d.is_unsupported(),
        }
    }

    /// `.value`: `int`, or `List[int]` for an array.
    pub fn value(self) -> String {
        match self {
            NodeDtype::Plain(d) => d.value().to_string(),
            NodeDtype::Array(d) => format!("List[{}]", d.value()),
        }
    }

    /// `ArrayDtype.to_dtype`: the list type of an array's elements.
    pub fn to_dtype(self) -> Result<Dtype, String> {
        match self {
            NodeDtype::Plain(d) => Ok(d),
            NodeDtype::Array(d) if d.is_float() => Ok(Dtype::Floatlist),
            NodeDtype::Array(d) if d.is_int() => Ok(Dtype::Intlist),
            NodeDtype::Array(Dtype::String) => Ok(Dtype::Stringlist),
            NodeDtype::Array(d) => Err(format!(
                "ArrayDtype.to_dtype() only supported for numerical nested types, got {d}."
            )),
        }
    }
}

impl std::fmt::Display for NodeDtype {
    /// As an f-string writes it: `int`, or `ArrayDtype(nested_dtype=int)`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeDtype::Plain(d) => f.write_str(d.value()),
            NodeDtype::Array(d) => write!(f, "ArrayDtype(nested_dtype={d})"),
        }
    }
}

/// `None` as an f-string writes it, else the type.
pub fn dtype_text(dtype: Option<NodeDtype>) -> String {
    dtype.map_or_else(|| "None".to_string(), |d| d.to_string())
}

/// What every node has (`ASTNode`'s fields).
#[derive(Clone, Debug, Default)]
pub struct Base {
    pub date_offset_range: Option<DorRef>,
    pub dtype: Option<NodeDtype>,
    pub stype: Option<Stype>,
    pub location: Option<Location>,
}

#[derive(Clone, Debug)]
pub enum ConstValue {
    /// The constant as the query wrote it.
    Text(String),
    /// An array's elements.
    List(Vec<NodeRef>),
}

#[derive(Debug)]
pub enum Kind {
    Column {
        /// The table and the column, apart (KO2-36).
        name: ColumnKey,
        /// Whether a `COUNT` aggregates this column (`_count_col`).
        count_col: bool,
    },
    Constant {
        value: ConstValue,
        /// A date literal's reading, made when the constant is built.
        time: Option<Stamp>,
        /// D4's warning, when the reading is day first and names its format.
        day_first: Option<DayFirst>,
    },
    Condition {
        target: NodeRef,
        op: Op,
        value: NodeRef,
        input_op: Option<String>,
    },
    Logical {
        left: NodeRef,
        op: BoolOp,
        right: Option<NodeRef>,
    },
    Filter {
        target: NodeRef,
        condition: NodeRef,
    },
    Aggregation {
        target: NodeRef,
        aggr: AggregationType,
        aggr_time_range: Option<DorRef>,
        group_by: Option<ColumnKey>,
        time_col: Option<ColumnKey>,
    },
    Join {
        rhs_target: NodeRef,
        lhs_key: ColumnKey,
        rhs_key: ColumnKey,
    },
}

#[derive(Debug)]
pub struct Node {
    pub base: Base,
    pub kind: Kind,
}

/// Why a node cannot be built, as the reference's parser tells the reasons
/// apart: a literal that cannot be read is `PQL_VALUE_INVALID_LITERAL`, and
/// any other refusal of the node is `PQL_PARSE_SYNTAX`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuildError {
    Syntax(String),
    Literal(String),
    /// A date literal with a two-digit year, and no current year to place
    /// it (`Reading::Unplaced`): no diagnostic, and never the literal.
    Unplaced,
    /// A date literal with a run of five digits or more (D8,
    /// `Reading::Packed`), refused by name where it is: no diagnostic; the
    /// literal as the caller wrote it, which the refusal may name (KO2-86,
    /// narrowing decision E79).
    Packed(Option<Location>, String),
}

impl BuildError {
    pub fn message(&self) -> &str {
        match self {
            BuildError::Syntax(m) | BuildError::Literal(m) => m,
            BuildError::Unplaced => "a two-digit year cannot be placed without the current year",
            BuildError::Packed(..) => "a date literal with a run of five digits or more is refused",
        }
    }
}

type Built = Result<NodeRef, BuildError>;

fn wrap(base: Base, kind: Kind) -> NodeRef {
    Rc::new(RefCell::new(Node { base, kind }))
}

/// Runs `ASTNode.__post_init__` on a new node: its date offset range
/// becomes the merge of its own and all its children's.
fn finish(node: NodeRef) -> Built {
    let combined = combined_time(&node).map_err(BuildError::Syntax)?;
    node.borrow_mut().base.date_offset_range = combined;
    Ok(node)
}

/// A node's own fields and the fields every node has.
fn base_at(location: Option<Location>) -> Base {
    Base {
        location,
        ..Base::default()
    }
}

/// `Column(fqn=..., location=...)`: the table and the column held apart,
/// so neither is split on a dot (KO2-36), where the reference splits
/// `fqn` and refuses one that is not two parts.
pub fn column(name: ColumnKey, location: Option<Location>) -> Built {
    finish(wrap(
        base_at(location),
        Kind::Column {
            name,
            count_col: false,
        },
    ))
}

/// `Constant(value=..., dtype_maybe=..., location=...)`, which reads its
/// value to check it, refusing one it cannot read with a `TypeError`. A
/// date literal is read once, here, with `reader`.
pub fn constant(
    value: ConstValue,
    dtype: Option<NodeDtype>,
    location: Option<Location>,
    reader: Reader,
) -> Built {
    let base = Base {
        dtype,
        location,
        ..Base::default()
    };
    let (time, day_first) = match (&value, dtype) {
        (ConstValue::Text(text), Some(NodeDtype::Plain(d))) if d.is_timestamp() => {
            match reader.read(text) {
                Reading::Read { stamp, day_first } => (Some(stamp), day_first),
                Reading::Refused => (None, None),
                Reading::Unplaced => return Err(BuildError::Unplaced),
                Reading::Packed => return Err(BuildError::Packed(location, text.clone())),
            }
        }
        _ => (None, None),
    };
    let node = finish(wrap(
        base,
        Kind::Constant {
            value,
            time,
            day_first,
        },
    ))?;
    if typed_value(&node).is_err() {
        let text = match &node.borrow().kind {
            Kind::Constant {
                value: ConstValue::Text(text),
                ..
            } => text.clone(),
            Kind::Constant {
                value: ConstValue::List(items),
                ..
            } => format!(
                "[{}]",
                items.iter().map(to_string).collect::<Vec<_>>().join(", ")
            ),
            _ => String::new(),
        };
        return Err(BuildError::Literal(format!(
            "{text} is not a valid instance of {}",
            dtype_text(dtype)
        )));
    }
    Ok(node)
}

/// `Condition(target=..., op=..., value=..., location=...)`. `op` is the
/// operator's text; `LIKE` and `NOT LIKE` become string operators on a value
/// without its `%`.
pub fn condition(target: NodeRef, op: &str, value: NodeRef, location: Option<Location>) -> Built {
    let (op, input_op) = match Op::parse(op) {
        Some(op) => (op, None),
        None if op == "LIKE" || op == "NOT LIKE" => {
            let (str_op, new_value) = like_to_str_op(&value, op == "NOT LIKE")?;
            if let Kind::Constant { value: v, .. } = &mut value.borrow_mut().kind {
                *v = ConstValue::Text(new_value);
            }
            (Op::Str(str_op), Some(op.to_string()))
        }
        None => {
            return Err(BuildError::Syntax(format!(
                "{op} is not found in the relational nor the string operators."
            )));
        }
    };
    finish(wrap(
        base_at(location),
        Kind::Condition {
            target,
            op,
            value,
            input_op,
        },
    ))
}

/// `Condition.like_to_str_op`: `%x%` is `CONTAINS`, `x%` `STARTS WITH` and
/// `%x` `ENDS WITH`; only the first can be negated.
fn like_to_str_op(value: &NodeRef, negate: bool) -> Result<(StrOp, String), BuildError> {
    let typed = typed_value(value).map_err(BuildError::Syntax)?;
    let Literal::Str(text) = typed else {
        return Err(BuildError::Syntax(
            "A LIKE pattern must be a string, such as '%text%'.".to_string(),
        ));
    };
    let op = if text.ends_with('%') && text.starts_with('%') {
        StrOp::Contains
    } else if text.ends_with('%') {
        StrOp::StartsWith
    } else if text.starts_with('%') {
        StrOp::EndsWith
    } else {
        return Err(BuildError::Syntax(
            "Condition must contain a % at the start, end, or both ends of the value being \
             compared when using the LIKE operator."
                .to_string(),
        ));
    };
    let raw = match &value.borrow().kind {
        Kind::Constant {
            value: ConstValue::Text(raw),
            ..
        } => raw.replace('%', ""),
        _ => String::new(),
    };
    if negate {
        if op != StrOp::Contains {
            return Err(BuildError::Syntax(
                "'NOT LIKE' only works for values that start with '%' and end with '%'."
                    .to_string(),
            ));
        }
        return Ok((StrOp::NotContains, raw));
    }
    Ok((op, raw))
}

/// `LogicalOperation(left=..., bool_op=..., right=..., location=...)`.
pub fn logical(
    left: NodeRef,
    op: BoolOp,
    right: Option<NodeRef>,
    location: Option<Location>,
) -> Built {
    if op != BoolOp::Not && right.is_none() {
        return Err(BuildError::Syntax(
            "Nested 'LogicalOperation' is missing a right-hand side 'LogicalOperation' since \
             the boolean operator is 'AND' or 'OR'."
                .to_string(),
        ));
    }
    if op == BoolOp::Not && right.is_some() {
        return Err(BuildError::Syntax(
            "Nested 'LogicalOperation' shouldn't have a right-hand side 'LogicalOperation' \
             since boolean operator is 'NOT'."
                .to_string(),
        ));
    }
    finish(wrap(base_at(location), Kind::Logical { left, op, right }))
}

/// `Filter(target=..., condition=..., location=...)`.
pub fn filter(target: NodeRef, condition: NodeRef, location: Option<Location>) -> Built {
    finish(wrap(base_at(location), Kind::Filter { target, condition }))
}

/// `Aggregation(target=..., aggr=..., aggr_time_range=..., location=...)`:
/// its range starts as the window itself, the same object.
pub fn aggregation(
    target: NodeRef,
    aggr: AggregationType,
    aggr_time_range: Option<DorRef>,
    location: Option<Location>,
) -> Built {
    let base = Base {
        date_offset_range: aggr_time_range.clone(),
        location,
        ..Base::default()
    };
    finish(wrap(
        base,
        Kind::Aggregation {
            target,
            aggr,
            aggr_time_range,
            group_by: None,
            time_col: None,
        },
    ))
}

/// `Join(rhs_target=..., lhs_key=..., rhs_key=...)`, which takes its types
/// and location from what it joins.
pub fn join(rhs_target: NodeRef, lhs_key: ColumnKey, rhs_key: ColumnKey) -> Built {
    let node = finish(wrap(
        Base::default(),
        Kind::Join {
            rhs_target: rhs_target.clone(),
            lhs_key,
            rhs_key,
        },
    ))?;
    {
        let rhs = rhs_target.borrow();
        let mut n = node.borrow_mut();
        n.base.dtype = rhs.base.dtype;
        n.base.stype = rhs.base.stype;
        n.base.location = rhs.base.location;
    }
    Ok(node)
}

/// `Condition.__and__` and friends, for building queries in code.
pub fn and(left: NodeRef, right: NodeRef) -> Built {
    logical(left, BoolOp::And, Some(right), None)
}

pub fn or(left: NodeRef, right: NodeRef) -> Built {
    logical(left, BoolOp::Or, Some(right), None)
}

pub fn not(node: NodeRef) -> Built {
    logical(node, BoolOp::Not, None, None)
}

// --- Reading nodes -----------------------------------------------------------

/// `node.children`, in Python's order.
pub fn children(node: &NodeRef) -> Vec<NodeRef> {
    match &node.borrow().kind {
        Kind::Column { .. } | Kind::Constant { .. } => Vec::new(),
        Kind::Condition { target, value, .. } => vec![target.clone(), value.clone()],
        Kind::Logical { left, right, .. } => {
            let mut out = vec![left.clone()];
            out.extend(right.clone());
            out
        }
        Kind::Filter { target, condition } => vec![target.clone(), condition.clone()],
        Kind::Aggregation { target, .. } => vec![target.clone()],
        Kind::Join { rhs_target, .. } => vec![rhs_target.clone()],
    }
}

/// `ASTNode.get_combined_time`: the node's own range merged with every
/// child's, recomputed down the tree. Merging may rewrite the ranges it
/// merges, as `merge_ranges` does.
pub fn combined_time(node: &NodeRef) -> Result<Option<DorRef>, String> {
    let mut range = node.borrow().base.date_offset_range.clone();
    for child in children(node) {
        let Some(child_range) = combined_time(&child)? else {
            continue;
        };
        range = Some(match range {
            None => child_range,
            Some(own) => DateOffsetRange::merge_ranges(&own, &child_range)?,
        });
    }
    Ok(range)
}

/// `get_location`: the node's location, or its children's merged, or an
/// empty one.
pub fn get_location(node: &NodeRef) -> Location {
    find_location(node).unwrap_or(Location {
        start_row: 0,
        start_col: 0,
        end_row: 0,
        end_col: 0,
        data_available: false,
    })
}

fn find_location(node: &NodeRef) -> Option<Location> {
    let mut location = node.borrow().base.location;
    for child in children(node) {
        let child_location = find_location(&child);
        location = match (location, child_location) {
            (None, c) => c,
            (Some(l), Some(c)) => Some(Location::merge(l, c)),
            (l, None) => l,
        };
    }
    location
}

pub fn is_column(node: &NodeRef) -> bool {
    matches!(node.borrow().kind, Kind::Column { .. })
}

pub fn is_aggregation(node: &NodeRef) -> bool {
    matches!(node.borrow().kind, Kind::Aggregation { .. })
}

pub fn is_filter(node: &NodeRef) -> bool {
    matches!(node.borrow().kind, Kind::Filter { .. })
}

pub fn is_join(node: &NodeRef) -> bool {
    matches!(node.borrow().kind, Kind::Join { .. })
}

pub fn is_condition(node: &NodeRef) -> bool {
    matches!(node.borrow().kind, Kind::Condition { .. })
}

pub fn is_logical(node: &NodeRef) -> bool {
    matches!(node.borrow().kind, Kind::Logical { .. })
}

/// A column's table and column; `None` for any other node.
pub fn column_name(node: &NodeRef) -> Option<ColumnKey> {
    match &node.borrow().kind {
        Kind::Column { name, .. } => Some(name.clone()),
        _ => None,
    }
}

/// A column's fully qualified name, `table.column` unquoted, as a message
/// gives it; empty for any other node.
pub fn fqn(node: &NodeRef) -> String {
    column_name(node).map(|n| n.to_string()).unwrap_or_default()
}

/// The node a filter, aggregation, condition or join is about.
pub fn target(node: &NodeRef) -> Option<NodeRef> {
    match &node.borrow().kind {
        Kind::Condition { target, .. }
        | Kind::Filter { target, .. }
        | Kind::Aggregation { target, .. } => Some(target.clone()),
        Kind::Join { rhs_target, .. } => Some(rhs_target.clone()),
        _ => None,
    }
}

pub fn dtype(node: &NodeRef) -> Option<NodeDtype> {
    node.borrow().base.dtype
}

pub fn stype(node: &NodeRef) -> Option<Stype> {
    node.borrow().base.stype
}

pub fn date_offset_range(node: &NodeRef) -> Option<DorRef> {
    node.borrow().base.date_offset_range.clone()
}

/// An aggregation's window.
pub fn aggr_time_range(node: &NodeRef) -> Option<DorRef> {
    match &node.borrow().kind {
        Kind::Aggregation {
            aggr_time_range, ..
        } => aggr_time_range.clone(),
        _ => None,
    }
}

pub fn aggr(node: &NodeRef) -> Option<AggregationType> {
    match &node.borrow().kind {
        Kind::Aggregation { aggr, .. } => Some(*aggr),
        _ => None,
    }
}

/// `Aggregation.is_static`: no window.
pub fn is_static(node: &NodeRef) -> bool {
    aggr_time_range(node).is_none()
}

/// `Aggregation.get_target_column_name`: the aggregated column, through a
/// filter if there is one.
pub fn get_target_column_name(node: &NodeRef) -> Option<ColumnKey> {
    let target = target(node)?;
    if is_column(&target) {
        return column_name(&target);
    }
    self::target(&target).and_then(|t| column_name(&t))
}

/// `str(node)`, the query text that would build it.
pub fn to_string(node: &NodeRef) -> String {
    render(node, false)
}

/// `maybe_bold`: `[bold]text[/bold]` when writing for a rich console.
pub fn maybe_bold(text: &str, rich: bool) -> String {
    if rich {
        format!("[bold]{text}[/bold]")
    } else {
        text.to_string()
    }
}

/// `node.to_string(rich=rich)`: the query text that would build the node,
/// its keywords in bold for a rich console when `rich`.
pub fn render(node: &NodeRef, rich: bool) -> String {
    let n = node.borrow();
    match &n.kind {
        Kind::Column { name, .. } => names::fqn(name),
        Kind::Constant { value, .. } => match value {
            ConstValue::Text(text) => text.clone(),
            ConstValue::List(items) => format!(
                "({})",
                items.iter().map(to_string).collect::<Vec<_>>().join(", ")
            ),
        },
        Kind::Condition {
            target, op, value, ..
        } => {
            let mut op_text = op.value().to_string();
            let value_text = if value.borrow().base.dtype.is_none() {
                if *op == Op::Rel(RelOp::Eq) {
                    op_text = "IS".to_string();
                }
                if *op == Op::Rel(RelOp::Neq) {
                    op_text = "IS NOT".to_string();
                }
                op_text = maybe_bold(&op_text, rich);
                maybe_bold("NULL", rich)
            } else {
                render(value, rich)
            };
            format!("{} {op_text} {value_text}", render(target, rich))
        }
        Kind::Logical { left, op, right } => match right {
            None => format!("{} ({})", maybe_bold("NOT", rich), render(left, rich)),
            Some(right) => format!(
                "({}) {} ({})",
                render(left, rich),
                maybe_bold(op.value(), rich),
                render(right, rich)
            ),
        },
        Kind::Filter { target, condition } => format!(
            "{} {} {}",
            render(target, rich),
            maybe_bold("WHERE", rich),
            render(condition, rich)
        ),
        Kind::Aggregation {
            target,
            aggr,
            aggr_time_range,
            ..
        } => {
            let name = maybe_bold(aggr.value(), rich);
            let target = render(target, rich);
            match aggr_time_range {
                None => format!("{name}({target})"),
                Some(range) => {
                    let range = range.borrow();
                    let start = range
                        .start
                        .map_or_else(|| "-INF".to_string(), |s| s.to_string());
                    format!(
                        "{name}({target}, {start}, {}, {})",
                        range.end,
                        range.unit.value()
                    )
                }
            }
        }
        Kind::Join { rhs_target, .. } => render(rhs_target, rich),
    }
}

/// `Constant.typed_value`: what the constant stands for.
pub fn typed_value(node: &NodeRef) -> Result<Literal, String> {
    let n = node.borrow();
    let Kind::Constant { value, time, .. } = &n.kind else {
        return Ok(Literal::Null);
    };
    let Some(dtype) = n.base.dtype else {
        return Ok(Literal::Null);
    };
    if dtype.is_list() {
        let ConstValue::List(items) = value else {
            return Err("a list type holds a list".to_string());
        };
        return items
            .iter()
            .map(typed_value)
            .collect::<Result<Vec<_>, _>>()
            .map(Literal::List);
    }
    let text = match value {
        ConstValue::Text(text) => text.clone(),
        ConstValue::List(items) => format!(
            "[{}]",
            items.iter().map(to_string).collect::<Vec<_>>().join(", ")
        ),
    };
    let NodeDtype::Plain(dtype) = dtype else {
        unreachable!("an array type is a list type");
    };
    if dtype.is_bool() {
        return Ok(Literal::Bool(text.to_lowercase() == "true"));
    }
    if dtype.is_int() {
        return PyInt::parse(&text).map(Literal::Int);
    }
    if dtype.is_float() {
        return read_float(&text).map(Literal::Float);
    }
    if dtype.is_string() {
        let chars: Vec<char> = text.chars().collect();
        let inner: String = if chars.len() >= 2 {
            chars[1..chars.len() - 1].iter().collect()
        } else {
            String::new()
        };
        return unescape(&inner).map(Literal::Str);
    }
    if dtype.is_timestamp() {
        return time
            .map(Literal::Time)
            .ok_or_else(|| format!("{text} is not a date"));
    }
    Ok(Literal::Str(text))
}

/// `all_query_columns_with_locations`: every column the subtree names, with
/// where. Python collects them into a set, whose order follows string
/// hashing and changes from run to run; here they come in the order the
/// tree holds them, each once.
pub fn all_query_columns_with_locations(node: &NodeRef) -> Vec<(ColumnKey, Location)> {
    let mut out = Vec::new();
    collect_columns(node, &mut out);
    let mut seen = Vec::new();
    out.retain(|item| {
        if seen.contains(item) {
            false
        } else {
            seen.push(item.clone());
            true
        }
    });
    out
}

fn collect_columns(node: &NodeRef, out: &mut Vec<(ColumnKey, Location)>) {
    if let Kind::Column { name, .. } = &node.borrow().kind {
        let location = node.borrow().base.location.unwrap_or(Location {
            start_row: 0,
            start_col: 0,
            end_row: 0,
            end_col: 0,
            data_available: false,
        });
        out.push((name.clone(), location));
        return;
    }
    for child in children(node) {
        collect_columns(&child, out);
    }
}

/// `all_query_columns`: every column the subtree names, each once, in the
/// order the tree holds them.
pub fn all_query_columns(node: &NodeRef) -> Vec<ColumnKey> {
    let mut out: Vec<ColumnKey> = Vec::new();
    for (name, _) in all_query_columns_with_locations(node) {
        if !out.contains(&name) {
            out.push(name);
        }
    }
    out
}

/// `all_join_columns`: the keys of every join in the subtree, each once.
pub fn all_join_columns(node: &NodeRef) -> Vec<ColumnKey> {
    let mut out: Vec<ColumnKey> = Vec::new();
    let mut push = |key: ColumnKey| {
        if !out.contains(&key) {
            out.push(key);
        }
    };
    if let Kind::Join {
        rhs_target,
        lhs_key,
        rhs_key,
    } = &node.borrow().kind
    {
        for key in all_join_columns(rhs_target) {
            push(key);
        }
        push(lhs_key.clone());
        push(rhs_key.clone());
        return out;
    }
    for child in children(node) {
        for key in all_join_columns(&child) {
            push(key);
        }
    }
    out
}

/// `all_time_columns`: the time column of every windowed aggregation in the
/// subtree, each once.
pub fn all_time_columns(node: &NodeRef) -> Vec<ColumnKey> {
    let mut out: Vec<ColumnKey> = Vec::new();
    for child in children(node) {
        for column in all_time_columns(&child) {
            if !out.contains(&column) {
                out.push(column);
            }
        }
    }
    if let Kind::Aggregation {
        time_col: Some(time_col),
        ..
    } = &node.borrow().kind
        && !out.contains(time_col)
    {
        out.push(time_col.clone());
    }
    out
}

/// `non_inf_date_offset_range`: the merged windows of the subtree, leaving
/// out those that start at `-INF`.
pub fn non_inf_date_offset_range(node: &NodeRef) -> Result<Option<DorRef>, String> {
    let mut result: Option<DorRef> = None;
    for child in children(node) {
        let Some(range) = non_inf_date_offset_range(&child)? else {
            continue;
        };
        if range.borrow().start.is_none() {
            continue;
        }
        result = Some(match result {
            None => range,
            Some(r) => DateOffsetRange::merge_ranges(&r, &range)?,
        });
    }
    if let Some(window) = aggr_time_range(node) {
        if window.borrow().is_open() {
            return Ok(result);
        }
        return Ok(Some(match result {
            None => window,
            Some(r) => DateOffsetRange::merge_ranges(&r, &window)?,
        }));
    }
    Ok(result)
}

/// `end_date_offset`: the end of the subtree's range.
pub fn end_date_offset(node: &NodeRef) -> Option<i64> {
    date_offset_range(node).map(|r| r.borrow().end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TimeUnit;

    const READER: Reader = Reader {
        year: Some(2026),
        pads_year: false,
    };

    fn col_condition() -> NodeRef {
        let two = constant(
            ConstValue::Text("2".into()),
            Some(NodeDtype::Plain(Dtype::Int)),
            None,
            READER,
        )
        .unwrap();
        condition(col("t.c"), "!=", two, None).unwrap()
    }

    fn col(name: &str) -> NodeRef {
        let (table, column_name) = name.split_once('.').unwrap();
        column(
            ColumnKey::new(table, column_name),
            Some(Location::new(1, 0, 1, 0)),
        )
        .unwrap()
    }

    #[test]
    fn nodes_write_back_as_queries() {
        let one = constant(
            ConstValue::Text("1".into()),
            Some(NodeDtype::Plain(Dtype::Int)),
            None,
            READER,
        )
        .unwrap();
        let cond = condition(col("t.a"), ">", one, None).unwrap();
        let null = constant(ConstValue::Text("NULL".into()), None, None, READER).unwrap();
        let is_null = condition(col("t.b"), "=", null, None).unwrap();
        let both = and(cond, not(is_null).unwrap()).unwrap();
        assert_eq!(to_string(&both), "(t.a > 1) AND (NOT (t.b IS NULL))");
        let either = or(both, col_condition()).unwrap();
        assert_eq!(
            to_string(&either),
            "((t.a > 1) AND (NOT (t.b IS NULL))) OR (t.c != 2)"
        );
        let window = DateOffsetRange::from_i64(None, 3, TimeUnit::Days).unwrap();
        let count = aggregation(col("t.*"), AggregationType::Count, Some(window), None).unwrap();
        assert_eq!(to_string(&count), "COUNT(t.*, -INF, 3, days)");
        assert_eq!(
            get_target_column_name(&count),
            Some(ColumnKey::new("t", "*"))
        );
        let dotted = column(ColumnKey::new("a", "b.c"), None).unwrap();
        assert_eq!(to_string(&dotted), "a.`b.c`");
        assert_eq!(fqn(&dotted), "a.b.c");
    }

    #[test]
    fn ranges_merge_down_the_tree_as_python_merges_them() {
        let query = crate::visitor::parse_query(
            "PREDICT COUNT(orders.* WHERE COUNT(reviews.*, -INF, 0, days) > 0 AND \
             SUM(items.x, -5, 2, hours) > 1, 0, 3, days) FOR EACH users.user_id",
            crate::profiles::Mode::Training,
            READER,
        )
        .0
        .unwrap();
        let target = &query.target_ast;
        let range = |r: Option<DorRef>| r.map(|r| r.borrow().clone());
        let hours = |start, end| {
            Some(crate::dor::DateOffsetRange {
                start,
                end,
                unit: TimeUnit::Hours,
            })
        };
        // The outer window was rewritten in hours by the merge, as in Python.
        assert_eq!(range(aggr_time_range(target)), hours(Some(0), 72));
        assert_eq!(range(date_offset_range(target)), hours(None, 72));
        assert_eq!(
            range(non_inf_date_offset_range(target).unwrap()),
            hours(Some(-5), 72)
        );
        assert_eq!(end_date_offset(target), Some(72));
        assert!(all_time_columns(target).is_empty());
    }

    #[test]
    fn joins_and_locations() {
        let joined = join(
            col("b.x"),
            ColumnKey::new("a", "id"),
            ColumnKey::new("b", "a_id"),
        )
        .unwrap();
        let mut keys = all_join_columns(&joined);
        keys.sort();
        assert_eq!(
            keys,
            [ColumnKey::new("a", "id"), ColumnKey::new("b", "a_id")]
        );
        let a = Location::new(1, 5, 2, 3);
        let b = Location::new(2, 1, 1, 9);
        assert_eq!(Location::merge(a, b), a);
        assert_eq!(a.message(), "(row 1, column 5: row 2, column 3)");
        assert_eq!(a.message_end(), "row 2, column 3");
        let none = Location {
            data_available: false,
            ..a
        };
        assert_eq!(Location::merge(none, b), b);
        assert_eq!(none.message(), "");
    }

    #[test]
    fn like_becomes_a_string_operator() {
        let pattern = constant(
            ConstValue::Text("'%a%'".into()),
            Some(NodeDtype::Plain(Dtype::String)),
            None,
            READER,
        )
        .unwrap();
        let like = condition(col("t.s"), "LIKE", pattern, None).unwrap();
        assert_eq!(to_string(&like), "t.s CONTAINS 'a'");
        let bad = constant(
            ConstValue::Text("'a%'".into()),
            Some(NodeDtype::Plain(Dtype::String)),
            None,
            READER,
        )
        .unwrap();
        assert!(condition(col("t.s"), "NOT LIKE", bad, None).is_err());
    }

    #[test]
    fn constants_refuse_what_they_cannot_read() {
        let date = constant(
            ConstValue::Text("30.02.2024".into()),
            Some(NodeDtype::Plain(Dtype::Time)),
            None,
            READER,
        );
        assert_eq!(
            date.unwrap_err(),
            BuildError::Literal("30.02.2024 is not a valid instance of time".into())
        );
    }
}
