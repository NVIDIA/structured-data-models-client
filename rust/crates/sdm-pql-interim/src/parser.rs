// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A recursive-descent parser for `PQLGrammar.g4`, accepting exactly the
//! token sequences its ANTLR parser accepts and building the same tree.
//!
//! Conditions follow the precedence ANTLR gives the left-recursive rule:
//! `NOT` binds tighter than `AND`, which binds tighter than `OR`, and both
//! binary operators group to the left. An array needs two elements or more;
//! the grammar's actions refuse `(c)` and `(c,)`.

use crate::lexer::{Kind, Token};

/// Nesting deeper than this is refused rather than followed. Today's
/// Python gives up at about 400 levels, on its own recursion limit.
pub const MAX_DEPTH: usize = 1000;

/// A rule's first and last token, which give its location.
#[derive(Clone, Copy, Debug)]
pub struct Span {
    pub start: usize,
    pub stop: usize,
}

#[derive(Debug)]
pub struct Prog {
    pub target: Target,
    pub spec: Spec,
    pub for_each: usize,
    pub entity: Entity,
    pub whatif: Option<Cond>,
}

#[derive(Debug)]
pub enum Target {
    Cond(Cond),
    Aggr(Aggr),
    Column(usize),
}

#[derive(Debug, Default)]
pub struct Spec {
    pub forecast: Option<usize>,
    pub problem_type: Option<usize>,
    pub top_k: Option<usize>,
    pub empty: bool,
}

#[derive(Debug)]
pub enum Entity {
    Column(usize),
    Filtered(Filtered),
    List {
        column: usize,
        op: usize,
        constant: Const,
        span: Span,
    },
    FilteredList {
        column: usize,
        op: usize,
        constant: Const,
        condition: Cond,
        span: Span,
    },
}

/// `column WHERE condition`.
#[derive(Debug)]
pub struct Filtered {
    pub column: usize,
    pub condition: Box<Cond>,
    pub span: Span,
}

#[derive(Debug)]
pub enum AggrTarget {
    Column(usize),
    Filtered(Filtered),
}

#[derive(Debug)]
pub struct Aggr {
    pub aggr: usize,
    pub target: AggrTarget,
    /// `(start, end, unit)`; `start` is an `INT` or `NEG_INFTY` token.
    pub window: Option<(usize, usize, Option<usize>)>,
    pub span: Span,
}

#[derive(Debug)]
pub enum CondLhs {
    Aggr(Aggr),
    Column(usize),
}

#[derive(Debug)]
pub enum Cond {
    Rel {
        lhs: CondLhs,
        op: usize,
        constant: Const,
        span: Span,
    },
    Not {
        inner: Box<Cond>,
        span: Span,
    },
    Binary {
        left: Box<Cond>,
        op: usize,
        right: Box<Cond>,
        span: Span,
    },
    Paren {
        inner: Box<Cond>,
        span: Span,
    },
}

#[derive(Debug)]
pub enum Const {
    /// `BOOL`, `INT`, `STR`, `DECIMAL` or `NULL`.
    Token(usize),
    Datetime {
        date: usize,
        time: Option<usize>,
        span: Span,
    },
    Array {
        items: Vec<Const>,
        span: Span,
    },
}

impl Cond {
    pub fn span(&self) -> Span {
        match self {
            Cond::Rel { span, .. }
            | Cond::Not { span, .. }
            | Cond::Binary { span, .. }
            | Cond::Paren { span, .. } => *span,
        }
    }
}

impl Const {
    pub fn span(&self) -> Span {
        match self {
            Const::Token(t) => Span {
                start: *t,
                stop: *t,
            },
            Const::Datetime { span, .. } | Const::Array { span, .. } => *span,
        }
    }
}

/// Why a query is not PQL, said for its author.
#[derive(Debug)]
pub struct SyntaxError(pub String);

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    depth: usize,
}

type Parsed<T> = Result<T, SyntaxError>;

/// Parses the tokens of a whole query.
pub fn parse(tokens: &[Token]) -> Parsed<Prog> {
    let mut parser = Parser {
        tokens,
        pos: 0,
        depth: 0,
    };
    parser.prog()
}

impl Parser<'_> {
    fn peek(&self) -> Option<Kind> {
        self.tokens.get(self.pos).map(|t| t.kind)
    }

    fn at(&self, kind: Kind) -> bool {
        self.peek() == Some(kind)
    }

    fn error(&self, expected: &str) -> SyntaxError {
        let found = match self.tokens.get(self.pos) {
            Some(token) => format!(
                "found {} at row {}, column {}",
                describe_found(token),
                token.line,
                token.column
            ),
            None => "the query ends there".to_string(),
        };
        SyntaxError(format!("Expected {expected}, but {found}."))
    }

    fn expect(&mut self, kind: Kind, expected: &str) -> Parsed<usize> {
        if self.at(kind) {
            self.pos += 1;
            Ok(self.pos - 1)
        } else {
            Err(self.error(expected))
        }
    }

    fn descend(&mut self) -> Parsed<()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            let row = self.tokens.get(self.pos).map_or(0, |t| t.line);
            return Err(SyntaxError(format!(
                "The query nests parentheses, NOT or arrays more than {MAX_DEPTH} levels \
                 deep near row {row}; simplify it."
            )));
        }
        Ok(())
    }

    fn prog(&mut self) -> Parsed<Prog> {
        if !self.at(Kind::Predict) {
            return Err(self.error("the query to start with PREDICT"));
        }
        self.pos += 1;
        let target = self.target()?;
        let spec = self.spec()?;
        let for_each = match self.peek() {
            Some(Kind::ForEach | Kind::For) => {
                self.pos += 1;
                self.pos - 1
            }
            _ => {
                return Err(self.error(
                    "FOR EACH or FOR after the target (or RANK, CLASSIFY, TOP or FORECAST)",
                ));
            }
        };
        let entity = self.entity()?;
        let whatif = if self.at(Kind::Assuming) {
            self.pos += 1;
            Some(self.condition(0)?)
        } else {
            None
        };
        if self.pos != self.tokens.len() {
            return Err(self.error("the end of the query"));
        }
        Ok(Prog {
            target,
            spec,
            for_each,
            entity,
            whatif,
        })
    }

    fn is_column(&self) -> bool {
        matches!(self.peek(), Some(Kind::FqnColumn | Kind::WildcardColumn))
    }

    fn column(&mut self) -> Parsed<usize> {
        if self.is_column() {
            self.pos += 1;
            Ok(self.pos - 1)
        } else {
            Err(self.error("a column such as table.column or table.*"))
        }
    }

    fn target(&mut self) -> Parsed<Target> {
        match self.peek() {
            Some(Kind::Not | Kind::Open) => Ok(Target::Cond(self.condition(0)?)),
            Some(Kind::Aggr) => {
                let aggr = self.aggregation()?;
                if self.at(Kind::RelOp) {
                    let first = self.rel_rest(CondLhs::Aggr(aggr))?;
                    Ok(Target::Cond(self.condition_loop(first, 0)?))
                } else {
                    Ok(Target::Aggr(aggr))
                }
            }
            Some(Kind::FqnColumn | Kind::WildcardColumn) => {
                let column = self.column()?;
                if self.at(Kind::RelOp) {
                    let first = self.rel_rest(CondLhs::Column(column))?;
                    Ok(Target::Cond(self.condition_loop(first, 0)?))
                } else {
                    Ok(Target::Column(column))
                }
            }
            _ => Err(self.error("a target after PREDICT: a column, an aggregation or a condition")),
        }
    }

    fn spec(&mut self) -> Parsed<Spec> {
        let mut spec = Spec::default();
        match self.peek() {
            Some(Kind::Forecast) => {
                self.pos += 1;
                spec.forecast =
                    Some(self.expect(Kind::Int, "the number of timeframes after FORECAST")?);
                self.expect(Kind::Timeframes, "TIMEFRAMES after FORECAST and its number")?;
            }
            Some(Kind::Rank | Kind::Classify) => {
                spec.problem_type = Some(self.pos);
                self.pos += 1;
                if self.at(Kind::Top) {
                    spec.top_k = Some(self.top_k()?);
                }
            }
            Some(Kind::Top) => spec.top_k = Some(self.top_k()?),
            _ => spec.empty = true,
        }
        Ok(spec)
    }

    fn top_k(&mut self) -> Parsed<usize> {
        self.pos += 1;
        self.expect(Kind::Int, "an integer after TOP")
    }

    fn entity(&mut self) -> Parsed<Entity> {
        let column = self.column()?;
        if self.at(Kind::RelOp) {
            let op = self.pos;
            self.pos += 1;
            let constant = self.constant()?;
            if self.at(Kind::Where) {
                self.pos += 1;
                let condition = self.condition(0)?;
                let span = Span {
                    start: column,
                    stop: self.pos - 1,
                };
                return Ok(Entity::FilteredList {
                    column,
                    op,
                    constant,
                    condition,
                    span,
                });
            }
            let span = Span {
                start: column,
                stop: self.pos - 1,
            };
            return Ok(Entity::List {
                column,
                op,
                constant,
                span,
            });
        }
        if self.at(Kind::Where) {
            return Ok(Entity::Filtered(self.filtered(column)?));
        }
        Ok(Entity::Column(column))
    }

    fn filtered(&mut self, column: usize) -> Parsed<Filtered> {
        self.pos += 1;
        let condition = self.condition(0)?;
        Ok(Filtered {
            column,
            condition: Box::new(condition),
            span: Span {
                start: column,
                stop: self.pos - 1,
            },
        })
    }

    fn aggregation(&mut self) -> Parsed<Aggr> {
        let aggr = self.pos;
        self.pos += 1;
        self.expect(Kind::Open, "'(' after the aggregation's name")?;
        let column = self.column()?;
        let target = if self.at(Kind::Where) {
            AggrTarget::Filtered(self.filtered(column)?)
        } else {
            AggrTarget::Column(column)
        };
        let mut window = None;
        if self.at(Kind::Comma) {
            self.pos += 1;
            let start = match self.peek() {
                Some(Kind::Int | Kind::NegInfty) => {
                    self.pos += 1;
                    self.pos - 1
                }
                _ => return Err(self.error("the window's start, an integer or -INF")),
            };
            self.expect(Kind::Comma, "',' after the window's start")?;
            let end = self.expect(Kind::Int, "the window's end, an integer")?;
            let mut unit = None;
            if self.at(Kind::Comma) {
                self.pos += 1;
                unit = Some(self.expect(
                    Kind::TimeUnit,
                    "a time unit: seconds, minutes, hours, days, weeks or months",
                )?);
            }
            window = Some((start, end, unit));
        }
        let close = self.expect(Kind::Close, "')' to close the aggregation")?;
        Ok(Aggr {
            aggr,
            target,
            window,
            span: Span {
                start: aggr,
                stop: close,
            },
        })
    }

    /// The rest of `lhs REL_OP constant`, at the operator.
    fn rel_rest(&mut self, lhs: CondLhs) -> Parsed<Cond> {
        let start = match &lhs {
            CondLhs::Aggr(a) => a.span.start,
            CondLhs::Column(c) => *c,
        };
        let op = self.expect(Kind::RelOp, "an operator such as =, >, IN or LIKE")?;
        let constant = self.constant()?;
        Ok(Cond::Rel {
            lhs,
            op,
            constant,
            span: Span {
                start,
                stop: self.pos - 1,
            },
        })
    }

    fn condition(&mut self, precedence: u8) -> Parsed<Cond> {
        self.descend()?;
        let first = match self.peek() {
            Some(Kind::Aggr) => {
                let aggr = self.aggregation()?;
                self.rel_rest(CondLhs::Aggr(aggr))?
            }
            Some(Kind::FqnColumn | Kind::WildcardColumn) => {
                let column = self.column()?;
                self.rel_rest(CondLhs::Column(column))?
            }
            Some(Kind::Not) => {
                let start = self.pos;
                self.pos += 1;
                let inner = self.condition(4)?;
                Cond::Not {
                    inner: Box::new(inner),
                    span: Span {
                        start,
                        stop: self.pos - 1,
                    },
                }
            }
            Some(Kind::Open) => {
                let start = self.pos;
                self.pos += 1;
                let inner = self.condition(0)?;
                let stop = self.expect(Kind::Close, "')' to close the parenthesis")?;
                Cond::Paren {
                    inner: Box::new(inner),
                    span: Span { start, stop },
                }
            }
            _ => {
                return Err(self.error(
                    "a condition: a column or aggregation compared with a constant, NOT, or '('",
                ));
            }
        };
        let result = self.condition_loop_from(first, precedence);
        self.depth -= 1;
        result
    }

    fn condition_loop(&mut self, first: Cond, precedence: u8) -> Parsed<Cond> {
        self.descend()?;
        let result = self.condition_loop_from(first, precedence);
        self.depth -= 1;
        result
    }

    /// `AND` and `OR` after a condition, as far as `precedence` allows.
    fn condition_loop_from(&mut self, first: Cond, precedence: u8) -> Parsed<Cond> {
        let mut left = first;
        let mut left_start = left.span().start;
        loop {
            let (rhs_precedence, allowed) = match self.peek() {
                Some(Kind::And) => (4, 3 >= precedence),
                Some(Kind::Or) => (3, 2 >= precedence),
                _ => break,
            };
            if !allowed {
                break;
            }
            let op = self.pos;
            self.pos += 1;
            let right = self.condition(rhs_precedence)?;
            let span = Span {
                start: left_start,
                stop: self.pos - 1,
            };
            left = Cond::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
                span,
            };
            left_start = span.start;
        }
        Ok(left)
    }

    fn constant(&mut self) -> Parsed<Const> {
        match self.peek() {
            Some(Kind::Bool | Kind::Int | Kind::Str | Kind::Decimal | Kind::Null) => {
                self.pos += 1;
                Ok(Const::Token(self.pos - 1))
            }
            Some(Kind::Date) => {
                let date = self.pos;
                self.pos += 1;
                let time = if self.at(Kind::Time) {
                    self.pos += 1;
                    Some(self.pos - 1)
                } else {
                    None
                };
                Ok(Const::Datetime {
                    date,
                    time,
                    span: Span {
                        start: date,
                        stop: self.pos - 1,
                    },
                })
            }
            Some(Kind::Open) => self.array(),
            _ => {
                Err(self
                    .error("a constant: a number, a string, a date, true, false, NULL or an array"))
            }
        }
    }

    fn array(&mut self) -> Parsed<Const> {
        self.descend()?;
        let open = self.pos;
        self.pos += 1;
        let mut items = vec![self.constant()?];
        loop {
            match self.peek() {
                Some(Kind::Comma) => {
                    self.pos += 1;
                    if self.at(Kind::Close) && items.len() == 1 {
                        return Err(one_element(&self.tokens[open]));
                    }
                    items.push(self.constant()?);
                }
                Some(Kind::Close) => {
                    if items.len() == 1 {
                        return Err(one_element(&self.tokens[open]));
                    }
                    self.pos += 1;
                    break;
                }
                _ => return Err(self.error("',' or ')' in the array")),
            }
        }
        self.depth -= 1;
        Ok(Const::Array {
            items,
            span: Span {
                start: open,
                stop: self.pos - 1,
            },
        })
    }
}

fn one_element(open: &Token) -> SyntaxError {
    SyntaxError(format!(
        "The array at row {}, column {} has one element, and an array needs two or more. \
         To compare with one value, use '= value' instead of 'IN (value)'.",
        open.line, open.column
    ))
}

/// A token as a message names it: its text where that reads well, else its
/// kind.
fn describe_found(token: &Token) -> String {
    let text: String = token.text.chars().take(40).collect();
    if crate::syntax::is_plain(&text) {
        crate::syntax::quoted(&text)
    } else {
        token.kind.describe().to_string()
    }
}
