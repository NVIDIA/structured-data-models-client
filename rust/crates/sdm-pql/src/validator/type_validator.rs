// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Data and semantic types: inferred for every node, then checked
//! (`validator/type_validator.py`).

use crate::ast::{self, ConstValue, Kind, NodeDtype, NodeRef, dtype_text};
use crate::codes;
use crate::literal::Literal;
use crate::pyint::PyInt;
use crate::query::ParsedQuery;
use crate::response::{Span, ValidationError, ValidationResponse, ValidationWarning};
use crate::schema::{ColumnKey, SchemaDefinition};
use crate::types::{AggregationType, Dtype, MemberOp, Op, RelOp, Stype};

/// A constant's value; one that cannot be read reads as nothing, as it
/// never happens to a constant the parser built.
pub fn typed(node: &NodeRef) -> Literal {
    ast::typed_value(node).unwrap_or(Literal::Null)
}

/// `validate_int`: the integer within `[min, max]`, int64 by default.
pub fn validate_int(value: &PyInt, min: i64, max: i64) -> ValidationResponse {
    if value.gt(max) || value.lt(min) {
        return ValidationResponse::error(ValidationError::new(
            "Unsupported constant",
            format!(
                "Constant {value} is outside of the range of supported integers ({min}, {max})."
            ),
            codes::VALUE_INT_OUT_OF_RANGE,
            None,
        ));
    }
    ValidationResponse::default()
}

pub fn validate_int64(value: &PyInt) -> ValidationResponse {
    validate_int(value, i64::MIN, i64::MAX)
}

/// The inference and checks, against one schema.
pub struct TypeValidator<'a> {
    pub schema: &'a SchemaDefinition,
    pub allow_array_targets: bool,
    pub allow_timestamp_arrays: bool,
}

fn loc_start(node: &NodeRef) -> String {
    ast::get_location(node).message_start()
}

fn span(node: &NodeRef) -> Option<Span> {
    Span::at(ast::get_location(node))
}

fn target_of(node: &NodeRef) -> NodeRef {
    ast::target(node).expect("the node has a target")
}

/// A node's data type, which inference has set.
fn dt(node: &NodeRef) -> NodeDtype {
    ast::dtype(node).unwrap_or(NodeDtype::Plain(Dtype::Unsupported))
}

fn st(node: &NodeRef) -> Stype {
    ast::stype(node).unwrap_or(Stype::Unsupported)
}

impl TypeValidator<'_> {
    /// The schema's column a column node names.
    fn schema_column(&self, name: &ColumnKey) -> Option<crate::schema::Column> {
        let table = self.schema.table(&name.table_name)?;
        table.cols.iter().find(|c| c.name == name.col_name).cloned()
    }

    fn infer_dtype(&self, node: &NodeRef) {
        let dtype = {
            let n = node.borrow();
            match &n.kind {
                Kind::Aggregation { target, aggr, .. } => {
                    let target_dtype = ast::dtype(target);
                    Some(match aggr {
                        AggregationType::Count | AggregationType::CountDistinct => {
                            Some(NodeDtype::Plain(Dtype::Int))
                        }
                        AggregationType::Avg => Some(NodeDtype::Plain(Dtype::Float)),
                        AggregationType::ListDistinct => {
                            if self.allow_array_targets {
                                match target_dtype {
                                    Some(NodeDtype::Array(_)) => target_dtype,
                                    Some(NodeDtype::Plain(d)) => NodeDtype::array(d).ok(),
                                    None => None,
                                }
                            } else {
                                Some(NodeDtype::Plain(Dtype::String))
                            }
                        }
                        AggregationType::Sum
                            if target_dtype == Some(NodeDtype::Plain(Dtype::Bool)) =>
                        {
                            Some(NodeDtype::Plain(Dtype::Int))
                        }
                        _ => target_dtype,
                    })
                }
                Kind::Filter { target, .. } => Some(ast::dtype(target)),
                Kind::Column { name, .. } => Some(if name.is_wildcard() {
                    Some(NodeDtype::Plain(Dtype::Int))
                } else {
                    self.schema_column(name).map(|c| {
                        if self.allow_array_targets {
                            match c.dtype {
                                Dtype::Intlist => NodeDtype::Array(Dtype::Int),
                                Dtype::Floatlist => NodeDtype::Array(Dtype::Float),
                                Dtype::String if c.stype == Stype::Multicategorical => {
                                    NodeDtype::Array(Dtype::String)
                                }
                                d => NodeDtype::Plain(d),
                            }
                        } else {
                            NodeDtype::Plain(c.dtype)
                        }
                    })
                }),
                Kind::Constant {
                    value: ConstValue::List(items),
                    ..
                } => {
                    let nested: Vec<NodeDtype> = items.iter().filter_map(ast::dtype).collect();
                    if nested.iter().all(|t| t.is_int()) {
                        Some(Some(NodeDtype::Plain(Dtype::Intlist)))
                    } else if nested.iter().all(|t| t.is_numerical()) {
                        Some(Some(NodeDtype::Plain(Dtype::Floatlist)))
                    } else if nested.iter().all(|t| t.is_string()) {
                        Some(Some(NodeDtype::Plain(Dtype::Stringlist)))
                    } else {
                        None
                    }
                }
                Kind::Constant { .. } | Kind::Join { .. } => None,
                Kind::Logical { .. } | Kind::Condition { .. } => {
                    Some(Some(NodeDtype::Plain(Dtype::Bool)))
                }
            }
        };
        if let Some(dtype) = dtype {
            node.borrow_mut().base.dtype = dtype;
        }
    }

    fn infer_stype(&self, node: &NodeRef) {
        let stype = {
            let n = node.borrow();
            match &n.kind {
                Kind::Aggregation { target, aggr, .. } => Some(match aggr {
                    AggregationType::Avg
                    | AggregationType::Count
                    | AggregationType::CountDistinct
                    | AggregationType::Sum => Some(Stype::Numerical),
                    AggregationType::ListDistinct => Some(Stype::Multicategorical),
                    _ => ast::stype(target),
                }),
                Kind::Filter { target, .. } => Some(ast::stype(target)),
                Kind::Column { name, .. } => Some(if name.is_wildcard() {
                    Some(Stype::Categorical)
                } else {
                    self.schema_column(name).map(|c| c.stype)
                }),
                Kind::Logical { .. } | Kind::Condition { .. } => Some(Some(Stype::Categorical)),
                Kind::Constant { .. } | Kind::Join { .. } => None,
            }
        };
        if let Some(stype) = stype {
            node.borrow_mut().base.stype = stype;
        }
    }

    fn infer_dtypes_of(&self, node: &NodeRef) {
        for child in ast::children(node) {
            self.infer_dtypes_of(&child);
        }
        self.infer_dtype(node);
    }

    fn infer_stypes_of(&self, node: &NodeRef) {
        for child in ast::children(node) {
            self.infer_stypes_of(&child);
        }
        self.infer_stype(node);
    }

    /// Every node's data type, the entity IDs' included.
    pub fn infer_dtypes(&self, query: &ParsedQuery) {
        self.infer_dtypes_of(&query.entity_ast);
        self.infer_dtypes_of(&query.target_ast);
        if let Some(whatif) = &query.whatif_ast {
            self.infer_dtypes_of(whatif);
        }
        if let Some(ids) = &query.rfm_entity_ids {
            self.infer_dtypes_of(ids);
        }
    }

    /// Every node's semantic type; not the entity IDs', as in Python.
    pub fn infer_stypes(&self, query: &ParsedQuery) {
        self.infer_stypes_of(&query.entity_ast);
        self.infer_stypes_of(&query.target_ast);
        if let Some(whatif) = &query.whatif_ast {
            self.infer_stypes_of(whatif);
        }
    }

    fn validate_dtype(&self, node: &NodeRef) -> ValidationResponse {
        let is_aggr = ast::is_aggregation(node);
        if is_aggr {
            return self.validate_dtype_aggr(node);
        }
        if ast::is_condition(node) {
            return self.validate_dtype_condition(node);
        }
        ValidationResponse::default()
    }

    fn validate_dtype_aggr(&self, node: &NodeRef) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        let aggr = ast::aggr(node).unwrap_or(AggregationType::Count);
        let target = target_of(node);
        let target_dtype = dt(&target);
        let target_text = ast::to_string(&target);
        if matches!(aggr, AggregationType::Avg | AggregationType::Sum)
            && !target_dtype.is_numerical()
        {
            response.errors.push(ValidationError::new(
                "Type Mismatch",
                format!(
                    "{}: Aggregation {aggr} can only operate on integers and floats, but \
                     {target_text} has data type {}.",
                    loc_start(node),
                    target_dtype.value()
                ),
                codes::TYPE_AGGREGATION_NEEDS_NUMERIC,
                span(node),
            ));
        }
        if matches!(aggr, AggregationType::Min | AggregationType::Max)
            && target_dtype == NodeDtype::Plain(Dtype::Bool)
        {
            response.warnings.push(ValidationWarning::new(
                "Type Mismatch",
                format!(
                    "{}: Aggregation {aggr} is not intended to operate on bool columns. During \
                     this label computation, True will be considered larger than False. To avoid \
                     this warning, save column {target_text} as an integer column instead.",
                    loc_start(node)
                ),
                codes::TYPE_AGGREGATION_ON_BOOL,
                span(node),
            ));
        }
        if self.allow_array_targets && !self.allow_timestamp_arrays && target_dtype.is_timestamp() {
            response.errors.push(ValidationError::new(
                "Unsupported type",
                format!(
                    "{}: Aggregation {aggr} on columns of type timestamp is not supported on \
                     Snowflake.",
                    loc_start(node)
                ),
                codes::TYPE_AGGREGATION_ON_TIMESTAMP_UNSUPPORTED,
                span(node),
            ));
        }
        response
    }

    fn validate_dtype_condition(&self, node: &NodeRef) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        let (op, value) = match &node.borrow().kind {
            Kind::Condition { op, value, .. } => (*op, value.clone()),
            _ => return response,
        };
        response = match op {
            Op::Member(_) => response.merge(self.validate_array_dtype(node, op, &value)),
            Op::Str(_) => response.merge(self.validate_strop_dtype(node, op, &value)),
            Op::Rel(_) => response.merge(self.validate_relop_dtype(node, op, &value)),
        };
        if let Some(d) = ast::dtype(&value)
            && d.is_int()
            && let Literal::Int(int) = typed(&value)
        {
            response = response.merge(validate_int64(&int));
        }
        response
    }

    /// `_value_dtype_mismatch`: whether a value cannot stand for `dtype`.
    fn mismatch(dtype: NodeDtype, value: &Literal, allow_null: bool) -> bool {
        match value {
            Literal::Str(_) => !dtype.is_string(),
            Literal::Bool(_) => !dtype.is_bool(),
            Literal::Int(_) | Literal::Float(_) => !dtype.is_numerical(),
            Literal::Time(_) => !dtype.is_timestamp(),
            Literal::Null => !allow_null,
            Literal::List(_) => true,
        }
    }

    fn validate_relop_dtype(&self, node: &NodeRef, op: Op, value: &NodeRef) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        let typed_val = typed(value);
        if matches!(typed_val, Literal::List(_)) {
            response.errors.push(ValidationError::new(
                "Type Mismatch",
                format!(
                    "{}: Operator {} does not support operations on arrays.",
                    loc_start(node),
                    op.value()
                ),
                codes::TYPE_OPERATOR_ON_ARRAY,
                span(node),
            ));
            return response;
        }
        let target = target_of(node);
        let target_dtype = dt(&target);
        let target_text = ast::to_string(&target);
        if Self::mismatch(target_dtype, &typed_val, true) {
            let quote_helper = if target_dtype.is_string() {
                " Did you forget to add quotes?"
            } else {
                ""
            };
            response.errors.push(ValidationError::new(
                "Type Mismatch",
                format!(
                    "{}: Expression {target_text} has type {target_dtype} but constant {} has \
                     type {}.{quote_helper}",
                    loc_start(node),
                    ast::to_string(value),
                    dtype_text(ast::dtype(value))
                ),
                codes::TYPE_CONSTANT_MISMATCH,
                span(node),
            ));
        }
        let equality = matches!(op, Op::Rel(RelOp::Eq | RelOp::Neq));
        if typed_val == Literal::Null && !equality {
            response.errors.push(ValidationError::new(
                "Type Mismatch",
                format!(
                    "{}: Expression {target_text} has type {target_dtype} but operator {} does \
                     not support comparison with NULL. Please use IS NULL or IS NOT NULL.",
                    loc_start(node),
                    op.value()
                ),
                codes::TYPE_NULL_COMPARISON,
                span(node),
            ));
        }
        if matches!(typed_val, Literal::Bool(_)) && !equality {
            response.errors.push(ValidationError::new(
                "Type Mismatch",
                format!(
                    "{}: Expression {target_text} has type {target_dtype} but operator {} does \
                     not support comparisons of booleans.",
                    loc_start(node),
                    op.value()
                ),
                codes::TYPE_BOOL_COMPARISON,
                span(node),
            ));
        }
        response
    }

    fn validate_strop_dtype(&self, node: &NodeRef, op: Op, value: &NodeRef) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        let typed_val = typed(value);
        let value_str = ast::to_string(value);
        let target = target_of(node);
        let target_dtype = dt(&target);
        if self.allow_array_targets
            && let NodeDtype::Array(nested) = target_dtype
            && matches!(
                op,
                Op::Str(crate::types::StrOp::Contains | crate::types::StrOp::NotContains)
            )
        {
            if Self::mismatch(NodeDtype::Plain(nested), &typed_val, false) {
                response.errors.push(ValidationError::new(
                    "Type Mismatch",
                    format!(
                        "{}: Expression {} has type {} but constant {value_str} has type {}.",
                        loc_start(node),
                        ast::to_string(&target),
                        target_dtype.value(),
                        dtype_text(ast::dtype(value))
                    ),
                    codes::TYPE_CONSTANT_MISMATCH,
                    span(node),
                ));
            }
            return response;
        }
        if !target_dtype.is_string() {
            response.errors.push(ValidationError::new(
                "Type Mismatch",
                format!(
                    "{}: Operator {} operates on strings but {} has data type {}.",
                    loc_start(node),
                    op.value(),
                    ast::to_string(&target),
                    target_dtype.value()
                ),
                codes::TYPE_OPERATOR_NEEDS_STRING,
                span(node),
            ));
        }
        if !matches!(typed_val, Literal::Str(_)) {
            response.errors.push(ValidationError::new(
                "Type Mismatch",
                format!(
                    "{}: Operator {} operates on strings but {value_str} has data type {}. Did \
                     you forget to add quotes?",
                    loc_start(node),
                    op.value(),
                    dtype_text(ast::dtype(value))
                ),
                codes::TYPE_OPERATOR_NEEDS_STRING,
                span(node),
            ));
        }
        response
    }

    fn validate_array_dtype(&self, node: &NodeRef, op: Op, value: &NodeRef) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        let typed_val = typed(value);
        let value_str = ast::to_string(value);
        if op == Op::Member(MemberOp::IsIn) {
            response.errors.push(ValidationError::new(
                "Operation Deprecated",
                format!(
                    "{}: Operator {} is deprecated. Please use operator `IN` instead.",
                    loc_start(node),
                    op.value()
                ),
                codes::TYPE_OPERATOR_DEPRECATED,
                span(node),
            ));
        }
        if !matches!(typed_val, Literal::List(_)) {
            response.errors.push(ValidationError::new(
                "Type Mismatch",
                format!(
                    "{}: Operator {} expects an array constant but {value_str} has data type {}.",
                    loc_start(node),
                    op.value(),
                    dtype_text(ast::dtype(value))
                ),
                codes::TYPE_OPERATOR_NEEDS_ARRAY,
                span(node),
            ));
            return response;
        }
        let target = target_of(node);
        let target_dtype = dt(&target);
        let items = match &value.borrow().kind {
            Kind::Constant {
                value: ConstValue::List(items),
                ..
            } => items.clone(),
            _ => Vec::new(),
        };
        for element in items {
            if Self::mismatch(target_dtype, &typed(&element), false) {
                response.errors.push(ValidationError::new(
                    "Type Mismatch",
                    format!(
                        "{}: Expression {} has type {target_dtype} but array contains element {} \
                         of type {}.",
                        loc_start(node),
                        ast::to_string(&target),
                        ast::to_string(&element),
                        dtype_text(ast::dtype(&element))
                    ),
                    codes::TYPE_ARRAY_ELEMENT_MISMATCH,
                    span(node),
                ));
                break;
            }
        }
        response
    }

    fn validate_stype(&self, node: &NodeRef) -> ValidationResponse {
        if ast::is_aggregation(node) {
            return self.validate_stype_aggr(node);
        }
        if let Kind::Condition { op, target, .. } = &node.borrow().kind {
            return Self::validate_stype_condition(node, *op, target);
        }
        ValidationResponse::default()
    }

    fn validate_stype_aggr(&self, node: &NodeRef) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        let aggr = ast::aggr(node).unwrap_or(AggregationType::Count);
        let target = target_of(node);
        let target_stype = st(&target);
        let target_text = ast::to_string(&target);
        let name = ast::get_target_column_name(node).unwrap_or_else(|| ColumnKey::new("", ""));
        let mut target_is_key = self
            .schema
            .table(&name.table_name)
            .is_some_and(|t| t.pkey.as_ref() == Some(&name.col_name));
        if self.schema.all_keys().contains(&&name) {
            target_is_key = true;
        }
        use AggregationType as A;
        if !matches!(
            aggr,
            A::ListDistinct | A::Last | A::First | A::Count | A::Min | A::Max
        ) && (target_stype == Stype::Id || target_is_key)
        {
            response.errors.push(ValidationError::new(
                "Semantic Type Mismatch",
                format!(
                    "{}: Aggregation {aggr} is not intended to operate on foreign keys and \
                     ID-type columns.",
                    loc_start(node)
                ),
                codes::TYPE_AGGREGATION_ON_KEY_COLUMN,
                span(node),
            ));
        }
        if matches!(aggr, A::Min | A::Max)
            && !matches!(
                target_stype,
                Stype::Numerical | Stype::Text | Stype::Timestamp
            )
        {
            response.warnings.push(ValidationWarning::new(
                "Semantic Type Mismatch",
                format!(
                    "Aggregation {aggr} is intended to operate on numerical, text, and \
                     timestamp columns, but {target_text} has semantic type {target_stype}. \
                     Consider changing the underlying semantic type to fit the task better."
                ),
                codes::TYPE_AGGREGATION_WANTS_NUMERIC_TEXT_OR_TIME,
                None,
            ));
        }
        if matches!(aggr, A::Avg | A::Sum) && target_stype != Stype::Numerical {
            response.warnings.push(ValidationWarning::new(
                "Semantic Type Mismatch",
                format!(
                    "{}: Aggregation {aggr} is intended to operate on numerical columns, but \
                     {target_text} has semantic type {target_stype}. Consider changing the \
                     underlying semantic type to fit the task better.",
                    loc_start(node)
                ),
                codes::TYPE_AGGREGATION_WANTS_NUMERIC,
                span(node),
            ));
        }
        if aggr == A::ListDistinct
            && !matches!(
                target_stype,
                Stype::Categorical | Stype::Multicategorical | Stype::Id
            )
        {
            response.warnings.push(ValidationWarning::new(
                "Semantic Type Mismatch",
                format!(
                    "{}: Aggregation {aggr} is intended to operate on categorical and \
                     multicategorical columns, but {target_text} has semantic type \
                     {target_stype}. Consider changing the underlying semantic type to fit the \
                     task better.",
                    loc_start(node)
                ),
                codes::TYPE_AGGREGATION_WANTS_CATEGORICAL,
                span(node),
            ));
        }
        response
    }

    fn validate_stype_condition(node: &NodeRef, op: Op, target: &NodeRef) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        let target_stype = st(target);
        let target_text = ast::to_string(target);
        if matches!(
            target_stype,
            Stype::Categorical | Stype::Multicategorical | Stype::Id
        ) && matches!(op, Op::Rel(RelOp::Leq | RelOp::Geq | RelOp::Lt | RelOp::Gt))
        {
            response.warnings.push(ValidationWarning::new(
                "Semantic Type Mismatch",
                format!(
                    "{}: Expression {target_text} has semantic type {target_stype}. But you are \
                     comparing it by value with operator {}. Consider changing the underlying \
                     semantic type to fit the task better.",
                    loc_start(node),
                    op.value()
                ),
                codes::TYPE_COMPARED_BY_VALUE,
                span(node),
            ));
        }
        if target_stype == Stype::Id && matches!(op, Op::Str(_)) {
            response.warnings.push(ValidationWarning::new(
                "Semantic Type Mismatch",
                format!(
                    "{}: Expression {target_text} has semantic type {target_stype}. But you are \
                     comparing it by value with operator {}. Consider changing the underlying \
                     semantic type to fit the task better.",
                    loc_start(node),
                    op.value()
                ),
                codes::TYPE_COMPARED_BY_VALUE,
                span(node),
            ));
        }
        response
    }

    fn validate_dtypes_of(&self, node: &NodeRef) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        for child in ast::children(node) {
            response = response.merge(self.validate_dtypes_of(&child));
        }
        response.merge(self.validate_dtype(node))
    }

    fn validate_stypes_of(&self, node: &NodeRef) -> ValidationResponse {
        let mut response = ValidationResponse::default();
        for child in ast::children(node) {
            response = response.merge(self.validate_stypes_of(&child));
        }
        response.merge(self.validate_stype(node))
    }

    pub fn validate_dtypes(&self, query: &ParsedQuery) -> ValidationResponse {
        let mut response = self.validate_dtypes_of(&query.entity_ast);
        response = response.merge(self.validate_dtypes_of(&query.target_ast));
        if let Some(NodeDtype::Array(nested)) = ast::dtype(&query.target_ast)
            && !(nested == Dtype::String || nested.is_numerical())
        {
            response.errors.push(ValidationError::new(
                "Invalid target type",
                format!(
                    "{}: Target label has data type {}, which cannot be predicted.",
                    loc_start(&query.target_ast),
                    NodeDtype::Array(nested)
                ),
                codes::TYPE_TARGET_DTYPE_NOT_PREDICTABLE,
                None,
            ));
        }
        if let Some(whatif) = &query.whatif_ast {
            response = response.merge(self.validate_dtypes_of(whatif));
        }
        if let Some(ids) = &query.rfm_entity_ids {
            response = response.merge(self.validate_dtypes_of(ids));
        }
        response
    }

    pub fn validate_stypes(&self, query: &ParsedQuery) -> ValidationResponse {
        let mut response = self.validate_stypes_of(&query.entity_ast);
        response = response.merge(self.validate_stypes_of(&query.target_ast));
        let target_stype = st(&query.target_ast);
        if matches!(
            target_stype,
            Stype::Text | Stype::Timestamp | Stype::Sequence | Stype::Unsupported
        ) {
            response.errors.push(ValidationError::new(
                "Invalid target type",
                format!(
                    "{}: Target label has semantic type {target_stype}, which cannot be \
                     predicted.",
                    loc_start(&query.target_ast)
                ),
                codes::TYPE_TARGET_STYPE_NOT_PREDICTABLE,
                None,
            ));
        }
        if let Some(whatif) = &query.whatif_ast {
            response = response.merge(self.validate_stypes_of(whatif));
        }
        response
    }
}
