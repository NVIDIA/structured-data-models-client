// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Text to a parsed query (`PQLParser.to_parsed_predictive_query` and
//! `PQLVisitor`).
//!
//! The parse tree becomes the syntax tree in the order Python's visitor
//! visits it, because the first node that fails decides the diagnostic: a
//! literal that cannot be read is `PQL_VALUE_INVALID_LITERAL`, anything else
//! `PQL_PARSE_SYNTAX`.

use std::cell::RefCell;

use crate::ast::{self, BuildError, ConstValue, Location, NodeDtype, NodeRef};
use crate::codes;
use crate::dates::{DayFirst, Reader};
use crate::dor::DateOffsetRange;
use crate::lexer::{self, Kind, Token};
use crate::names;
use crate::parser::{self, Aggr, AggrTarget, Cond, CondLhs, Const, Entity, Filtered, Span, Target};
use crate::profiles::Mode;
use crate::pyint::PyInt;
use crate::query::ParsedQuery;
use crate::syntax::{is_plain, is_space};
use crate::types::{AggregationType, BoolOp, Dtype, MemberOp, ProblemType, RelOp, StrOp, TimeUnit};

/// Why text is not a parsed query, as one diagnostic without a span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseFailure {
    pub code: &'static str,
    pub title: &'static str,
    pub message: String,
}

impl ParseFailure {
    fn syntax(message: String) -> Self {
        ParseFailure {
            code: codes::PARSE_SYNTAX,
            title: "Invalid Syntax",
            message,
        }
    }
}

/// Why text is not a parsed query: a diagnostic, a two-digit year with no
/// current year to place it, or a D8 date literal, refused by name where it
/// is (decision E79).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    Failure(ParseFailure),
    Unplaced,
    Packed(Option<crate::ast::Location>),
}

impl ParseError {
    fn syntax(message: String) -> Self {
        ParseError::Failure(ParseFailure::syntax(message))
    }
}

impl From<BuildError> for ParseError {
    fn from(error: BuildError) -> Self {
        match error {
            BuildError::Syntax(message) => ParseError::syntax(message),
            BuildError::Literal(message) => ParseError::Failure(ParseFailure {
                code: codes::VALUE_INVALID_LITERAL,
                title: "Invalid constant",
                message,
            }),
            BuildError::Unplaced => ParseError::Unplaced,
            BuildError::Packed(location) => ParseError::Packed(location),
        }
    }
}

type Visited<T> = Result<T, ParseError>;

/// A day-first date literal's warning (D4), and where the literal is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DateWarning {
    pub day_first: DayFirst,
    pub location: Option<crate::ast::Location>,
}

/// `PQLParser(mode).to_parsed_predictive_query(query)`, reading date
/// literals with `reader`, with the warnings of the literals it read, in
/// the order it read them, whether or not the query parses.
pub fn parse_query(
    query: &str,
    mode: Mode,
    reader: Reader,
) -> (Visited<ParsedQuery>, Vec<DateWarning>) {
    let warned = RefCell::new(Vec::new());
    let parsed = visit(query, mode, reader, &warned);
    (parsed, warned.into_inner())
}

fn visit(
    query: &str,
    mode: Mode,
    reader: Reader,
    warned: &RefCell<Vec<DateWarning>>,
) -> Visited<ParsedQuery> {
    // `EXPLAIN` and `EVALUATE` are flags, stripped before the grammar sees
    // the text; locations count from what is left.
    let mut text: Vec<char> = query.chars().skip_while(|&c| is_space(c)).collect();
    let first_space = text.iter().position(|&c| c == ' ');
    let prefix: String = match first_space {
        Some(k) => text[..k].iter().collect(),
        // `query[:-1]`: all but the last character.
        None => text[..text.len().saturating_sub(1)].iter().collect(),
    };
    let upper = prefix.to_uppercase();
    let (evaluate, explain) = (upper == "EVALUATE", upper == "EXPLAIN");
    if evaluate || explain {
        let from = first_space.map_or(0, |k| k + 1);
        text = text[from..]
            .iter()
            .copied()
            .skip_while(|&c| is_space(c))
            .collect();
        let after = text.iter().position(|&c| c == ' ').unwrap_or(text.len());
        let word: String = text[..after].iter().collect();
        if word.to_uppercase() != "PREDICT" {
            let got = if is_plain(&word) {
                format!(", not \"{word}\"")
            } else {
                String::new()
            };
            return Err(ParseError::syntax(format!(
                "\"{prefix}\" must be followed by \"PREDICT\"{got}."
            )));
        }
    }
    let tokens = lexer::tokenize(&text).map_err(|e| {
        let at = format!("row {}, column {}", e.line, e.column);
        ParseError::syntax(match e.found {
            '\'' | '"' => format!("The string that starts at {at} is never closed."),
            '`' => format!(
                "The quoted name that starts at {at} is never closed, or is empty or spans \
                 lines."
            ),
            c if is_plain(&c.to_string()) => format!(
                "Unexpected character {} at {at}; no part of a predictive query starts with it.",
                crate::syntax::quoted(&c.to_string())
            ),
            c => format!(
                "Unexpected character U+{:04X} at {at}; no part of a predictive query starts \
                 with it.",
                c as u32
            ),
        })
    })?;
    let prog = parser::parse(&tokens).map_err(|e| ParseError::syntax(e.0))?;
    let visitor = Visitor {
        tokens: &tokens,
        reader,
        warned,
    };
    visitor.prog(&prog, mode, evaluate, explain)
}

struct Visitor<'a> {
    tokens: &'a [Token],
    reader: Reader,
    warned: &'a RefCell<Vec<DateWarning>>,
}

impl Visitor<'_> {
    fn text(&self, index: usize) -> &str {
        &self.tokens[index].text
    }

    /// `get_interval`: the first token's position and the last token's.
    fn interval(&self, span: Span) -> Option<Location> {
        let (start, stop) = (&self.tokens[span.start], &self.tokens[span.stop]);
        Some(Location::new(
            start.line,
            start.column,
            stop.line,
            stop.column,
        ))
    }

    fn int(&self, index: usize) -> Visited<PyInt> {
        PyInt::parse(self.text(index)).map_err(ParseError::syntax)
    }

    fn prog(
        &self,
        prog: &parser::Prog,
        mode: Mode,
        evaluate: bool,
        explain: bool,
    ) -> Visited<ParsedQuery> {
        let (entity_ast, rfm_entity_ids) = self.entity(&prog.entity)?;
        let target_ast = self.target(&prog.target)?;
        let mut problem_type = prog
            .spec
            .problem_type
            .map(|t| ProblemType::parse(&self.text(t).to_uppercase()).unwrap_or(ProblemType::Rank));
        let top_k = prog.spec.top_k.map(|t| self.int(t)).transpose()?;
        let forecast = prog.spec.forecast.map(|t| self.int(t)).transpose()?;
        let whatif_ast = prog
            .whatif
            .as_ref()
            .map(|c| self.condition(c))
            .transpose()?;
        let for_each = self.text(prog.for_each).to_uppercase();
        if forecast.is_some() && problem_type.is_none() {
            problem_type = Some(ProblemType::Forecast);
        }
        Ok(ParsedQuery {
            entity_ast,
            target_ast,
            whatif_ast,
            top_k,
            problem_type,
            rfm_query: mode == Mode::Inference,
            for_each,
            rfm_entity_ids,
            num_forecasts: forecast.unwrap_or_else(|| PyInt::from_i64(1)),
            evaluate,
            explain,
        })
    }

    fn target(&self, target: &Target) -> Visited<NodeRef> {
        match target {
            Target::Cond(c) => self.condition(c),
            Target::Aggr(a) => self.aggregation(a),
            Target::Column(c) => self.column(*c),
        }
    }

    fn column(&self, index: usize) -> Visited<NodeRef> {
        let fqn = names::canonical_fqn(self.text(index)).map_err(ParseError::syntax)?;
        let location = self.interval(Span {
            start: index,
            stop: index,
        });
        Ok(ast::column(fqn, location)?)
    }

    /// The entity, and the entity IDs when the query names some.
    fn entity(&self, entity: &Entity) -> Visited<(NodeRef, Option<NodeRef>)> {
        match entity {
            Entity::Column(c) => Ok((self.column(*c)?, None)),
            Entity::Filtered(f) => Ok((self.filtered(f)?, None)),
            Entity::List {
                column,
                op,
                constant,
                span,
            } => {
                let entity_col = self.column(*column)?;
                let op = self.entity_op(*op)?;
                let ids = self.constant(constant)?;
                let location = self.interval(*span);
                let condition = ast::condition(entity_col.clone(), &op, ids, location)?;
                Ok((entity_col, Some(condition)))
            }
            Entity::FilteredList {
                column,
                op,
                constant,
                condition,
                span,
            } => {
                let entity_col = self.column(*column)?;
                let op = self.entity_op(*op)?;
                let ids = self.constant(constant)?;
                let condition = self.condition(condition)?;
                let location = self.interval(*span);
                let filter = ast::filter(entity_col.clone(), condition, location)?;
                let ids = ast::condition(entity_col, &op, ids, location)?;
                Ok((filter, Some(ids)))
            }
        }
    }

    /// `_rel_op_cast`: the entity IDs' operator, relational, membership or
    /// string, never `LIKE` or `IS`.
    fn entity_op(&self, index: usize) -> Visited<String> {
        let op = self.text(index).to_uppercase();
        if RelOp::parse(&op).is_some()
            || MemberOp::parse(&op).is_some()
            || StrOp::parse(&op).is_some()
        {
            return Ok(op);
        }
        Err(ParseError::syntax(format!(
            "Entity IDs are named with = or IN, not {op}."
        )))
    }

    fn filtered(&self, filtered: &Filtered) -> Visited<NodeRef> {
        let column = self.column(filtered.column)?;
        let condition = self.condition(&filtered.condition)?;
        let location = self.interval(filtered.span);
        Ok(ast::filter(column, condition, location)?)
    }

    fn condition(&self, cond: &Cond) -> Visited<NodeRef> {
        match cond {
            Cond::Not { inner, span, .. } => {
                let nested = self.condition(inner)?;
                Ok(ast::logical(
                    nested,
                    BoolOp::Not,
                    None,
                    self.interval(*span),
                )?)
            }
            Cond::Binary {
                left,
                op,
                right,
                span,
                ..
            } => {
                let left = self.condition(left)?;
                let right = self.condition(right)?;
                let op = if self.text(*op).to_uppercase() == "AND" {
                    BoolOp::And
                } else {
                    BoolOp::Or
                };
                Ok(ast::logical(left, op, Some(right), self.interval(*span))?)
            }
            Cond::Paren { inner, .. } => self.condition(inner),
            Cond::Rel {
                lhs,
                op,
                constant,
                span,
                ..
            } => {
                let rel_op = self.text(*op).to_uppercase();
                let value = self.constant(constant)?;
                let target = match lhs {
                    CondLhs::Aggr(a) => self.aggregation(a)?,
                    CondLhs::Column(c) => self.column(*c)?,
                };
                let op = match rel_op.as_str() {
                    "IS" => "=".to_string(),
                    "IS NOT" => "!=".to_string(),
                    other => other.to_string(),
                };
                Ok(ast::condition(target, &op, value, self.interval(*span))?)
            }
        }
    }

    fn aggregation(&self, aggr: &Aggr) -> Visited<NodeRef> {
        let kind = AggregationType::parse(&self.text(aggr.aggr).to_uppercase())
            .unwrap_or(AggregationType::Count);
        let target = match &aggr.target {
            AggrTarget::Column(c) => self.column(*c)?,
            AggrTarget::Filtered(f) => self.filtered(f)?,
        };
        let location = self.interval(aggr.span);
        let Some((start, end, unit)) = aggr.window else {
            return Ok(ast::aggregation(target, kind, None, location)?);
        };
        let start = if self.tokens[start].kind == Kind::Int {
            Some(self.int(start)?)
        } else {
            None
        };
        let end = self.int(end)?;
        let unit = unit.map_or(TimeUnit::Days, |u| {
            TimeUnit::parse(&self.text(u).to_lowercase()).unwrap_or(TimeUnit::Days)
        });
        let range = DateOffsetRange::new(start.as_ref(), &end, unit).map_err(ParseError::syntax)?;
        Ok(ast::aggregation(target, kind, Some(range), location)?)
    }

    fn constant(&self, constant: &Const) -> Visited<NodeRef> {
        let location = self.interval(constant.span());
        match constant {
            Const::Token(index) => {
                let token = &self.tokens[*index];
                let dtype = match token.kind {
                    Kind::Bool => Some(Dtype::Bool),
                    Kind::Int => Some(Dtype::Int),
                    Kind::Decimal => Some(Dtype::Float),
                    Kind::Str => {
                        check_escapes(&token.text)?;
                        Some(Dtype::String)
                    }
                    _ => None,
                };
                Ok(ast::constant(
                    ConstValue::Text(token.text.clone()),
                    dtype.map(NodeDtype::Plain),
                    location,
                    self.reader,
                )?)
            }
            Const::Datetime { date, time, .. } => {
                let mut text = self.text(*date).to_string();
                if let Some(time) = time {
                    text.push(' ');
                    text.push_str(self.text(*time));
                }
                let node = ast::constant(
                    ConstValue::Text(text),
                    Some(NodeDtype::Plain(Dtype::Time)),
                    location,
                    self.reader,
                )?;
                if let ast::Kind::Constant {
                    day_first: Some(day_first),
                    ..
                } = node.borrow().kind
                {
                    self.warned.borrow_mut().push(DateWarning {
                        day_first,
                        location,
                    });
                }
                Ok(node)
            }
            Const::Array { items, .. } => {
                let children = items
                    .iter()
                    .map(|c| self.constant(c))
                    .collect::<Visited<Vec<_>>>()?;
                let dtype = match ast::dtype(&children[0]) {
                    Some(NodeDtype::Plain(d)) => NodeDtype::array(d).map_err(ParseError::syntax)?,
                    Some(NodeDtype::Array(_)) => {
                        return Err(ParseError::syntax(
                            "An array cannot begin with an array; arrays hold numbers, \
                             strings, booleans or dates."
                                .to_string(),
                        ));
                    }
                    None => {
                        return Err(ParseError::syntax(
                            "An array cannot begin with NULL; put a value of the array's \
                             type first."
                                .to_string(),
                        ));
                    }
                };
                Ok(ast::constant(
                    ConstValue::List(children),
                    Some(dtype),
                    location,
                    self.reader,
                )?)
            }
        }
    }
}

/// A string's escapes must each be one the grammar names:
/// `\\ \" \' \/ \b \f \n \r \t \u`.
fn check_escapes(text: &str) -> Visited<()> {
    let chars: Vec<char> = text.chars().collect();
    let inner = if chars.len() >= 2 {
        &chars[1..chars.len() - 1]
    } else {
        &[][..]
    };
    let mut escaped = false;
    for &c in inner {
        if escaped && !"\\\"'/bfnrtu".contains(c) {
            return Err(ParseError::syntax(format!(
                "\\{c} is not a valid escape pattern."
            )));
        }
        escaped = !escaped && c == '\\';
    }
    Ok(())
}
