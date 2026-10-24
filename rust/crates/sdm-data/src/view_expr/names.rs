// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which words of a Databricks expression name values
//! (`RE/view_expr.py:1047-1732`): a scan of its tokens in order, keeping
//! whether an operand is expected next, which decides whether a word names
//! a value. A function's name, a type, a keyword where it is grammar, a
//! datetime unit, a named argument, a lambda's parameter and a JSON path's
//! step are none.
//!
//! Where a word may open a form or be a column, as `NOT`, `CASE` and a
//! `CASE`'s first `WHEN` may, Databricks' parser takes the form if the rest
//! parses, and so does the scan: a trial reads on as if the word opened the
//! form, to the end of its bracket. Meeting another such word in that
//! bracket, it takes the form and comes back to read a column if the form
//! fails; one in a deeper bracket is a trial of its own. What a trial finds
//! is kept by where it starts and the state it starts in, and past a budget
//! of steps, or the deepest trial, the scan stops and the expression is
//! refused as unreadable, so a long expression is read in time linear in its
//! length and none is read wrong.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Range;

use super::ExprDialect;
use super::lex::{Kind, Token, lex};
use super::words::{self, has};

/// A name an expression reads as a value, possibly qualified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameReference {
    /// Its parts, unquoted; `*` for every column of a table.
    pub parts: Vec<String>,
    /// Where it is in the expression as written, in bytes.
    pub span: Range<usize>,
    /// Whether it is one unquoted part.
    pub bare: bool,
}

/// An expression whose names could not be settled within the bounds the
/// scan keeps, so no reading of it is safe (`UnreadableExpressionError`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unreadable {
    TooDeep,
    NotUnderstood,
}

impl fmt::Display for Unreadable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Unreadable::TooDeep => "it is too deeply nested or too long to read safely",
            Unreadable::NotUnderstood => {
                "it has syntax that is not understood, so which of its words name columns \
                 cannot be settled"
            }
        })
    }
}

impl std::error::Error for Unreadable {}

/// The names a Databricks expression reads as values (`name_references`).
/// An expression that does not parse as the scan reads SQL is refused,
/// since a failed reading of a form makes its word a column.
pub fn name_references(expr: &str) -> Result<Vec<NameReference>, Unreadable> {
    let tokens = lex(expr, Some(ExprDialect::Databricks))?
        .into_iter()
        .filter(|token| token.kind != Kind::Comment)
        .collect();
    let mut scan = Scan::new(tokens);
    let references = scan.references();
    if scan.cut {
        return Err(Unreadable::TooDeep);
    }
    if !scan.parses {
        return Err(Unreadable::NotUnderstood);
    }
    Ok(references)
}

// A column may take any keyword's name but TRUE, FALSE and NULL.
const LITERAL_WORDS: &[&str] = &["false", "null", "true"];
// Words that, after an operand, take another, and a type, field, collation
// or window name after AS, COLLATE or OVER.
const TAKE_AN_OPERAND: &[&str] = &[
    "and", "as", "between", "collate", "div", "else", "escape", "filter", "for", "from", "ilike",
    "in", "is", "like", "or", "over", "placing", "range", "regexp", "rlike", "rows", "then",
    "when", "within",
];
const TAKE_A_NAME: &[&str] = &["as", "collate", "over"];
// Words before `(` that open no call, or a form with its own grammar.
const NO_CALL_OPENERS_BEYOND: &[&str] = &[
    "",
    "cast",
    "extract",
    "group",
    "overlay",
    "position",
    "struct",
    "substring",
    "trim",
    "try_cast",
];
const TRIAL_STEPS: i64 = 16;
const DEEPEST_TRIAL: usize = 64;
const OPERATOR_PAIRS: &[&str] = &["!=", "<<", "==", ">>", "||"];
// What may stand after IS.
const AFTER_IS: &[&str] = &[
    "distinct", "false", "from", "not", "null", "true", "unknown",
];
// What a window's frame says between ROWS or RANGE and its `)`, and what
// may start it.
const FRAME_WORDS: &[&str] = &[
    "between",
    "current",
    "following",
    "preceding",
    "row",
    "unbounded",
];
const FRAME_STARTS: &[&str] = &["between", "current", "interval", "unbounded"];
// Words before a string that type it.
const TYPED_LITERALS: &[&str] = &["date", "timestamp", "timestamp_ltz", "timestamp_ntz", "x"];
const TRIM_SIDES: &[&str] = &["both", "leading", "trailing"];
const WINDOW_BY: &[&str] = &["cluster", "distribute", "partition", "sort"];
const TYPES_WITH_FIELDS: &[&str] = &["array", "map", "struct"];
const INTERVAL_FIELDS: &[&str] = &["day", "hour", "minute", "month", "second", "year"];
const TIME_ZONES: [&[&str]; 2] = [
    &["with", "local", "time", "zone"],
    &["without", "time", "zone"],
];
// Functions whose first argument may be a datetime unit, as a bare word.
const UNIT_FIRST: &[&str] = &["timediff", "timestampadd", "timestampdiff"];
// These take a unit first only in their three-argument form.
const UNIT_FIRST_OF_THREE: &[&str] = &["date_add", "date_diff", "dateadd", "datediff"];

fn no_call_opener(word: &str) -> bool {
    has(TAKE_AN_OPERAND, word) || NO_CALL_OPENERS_BEYOND.contains(&word)
}

/// What a CASE may have read last before each of its words.
fn case_steps(word: &str) -> Option<&'static [&'static str]> {
    match word {
        "when" => Some(&["case", "then"]),
        "then" => Some(&["when"]),
        "else" => Some(&["then"]),
        "end" => Some(&["then", "else"]),
        _ => None,
    }
}

/// Words that follow an operand only before these: `x NOT IN (1)`,
/// `IGNORE NULLS`.
fn followed_by(word: &str) -> Option<&'static [&'static str]> {
    match word {
        "not" => Some(&["between", "ilike", "in", "like", "regexp", "rlike"]),
        "ignore" | "respect" => Some(&["nulls"]),
        _ => None,
    }
}

fn is_name(kind: Kind) -> bool {
    matches!(kind, Kind::Word | Kind::Quoted)
}

/// A bracket open while scanning, and what is open inside it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Group {
    /// The word before it, folded, if any.
    opener: String,
    /// Each CASE open, by what it last read: CASE, WHEN, THEN or ELSE.
    cases: Vec<&'static str>,
    /// Whether a window's frame is being read.
    frame: bool,
}

impl Group {
    fn new(opener: String) -> Group {
        Group {
            opener,
            cases: Vec::new(),
            frame: false,
        }
    }
}

/// What a scan carries from one token to the next.
#[derive(Clone, Debug)]
struct State {
    groups: Vec<Group>,
    /// An operand is expected next.
    operand: bool,
    /// A type, alias, collation or window comes.
    name_next: bool,
    /// Within `IS [NOT] ...`.
    after_is: bool,
    /// Where a CASE's subject or first WHEN stands.
    case_at: isize,
    /// Where the BY of `PARTITION BY` stands.
    by_at: isize,
    /// The end of a type.
    skip_to: isize,
}

/// What a reading to the end of the current bracket depends on, with where
/// it starts.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    index: usize,
    operand: bool,
    name_next: bool,
    after_is: bool,
    case_at: isize,
    by_at: isize,
    skip_to: isize,
    group: Group,
}

impl State {
    fn new() -> State {
        State {
            groups: vec![Group::new(String::new())],
            operand: true,
            name_next: false,
            after_is: false,
            case_at: -1,
            by_at: -1,
            skip_to: -1,
        }
    }

    /// What the last token left expected, cleared for the next.
    fn taken(&mut self) -> (bool, bool, bool) {
        let left = (self.operand, self.name_next, self.after_is);
        self.operand = false;
        self.name_next = false;
        self.after_is = false;
        left
    }

    /// A copy holding only the innermost group, which is all a trial to the
    /// bracket's end reads, under an outermost one.
    fn trial(&self) -> State {
        State {
            groups: vec![Group::new(String::new()), self.inner().clone()],
            ..self.clone()
        }
    }

    /// Whether the innermost group may end here, nothing left owed.
    fn settled(&self) -> bool {
        !(self.operand || self.name_next || !self.inner().cases.is_empty())
    }

    fn key(&self, index: usize) -> Key {
        Key {
            index,
            operand: self.operand,
            name_next: self.name_next,
            after_is: self.after_is,
            case_at: self.case_at,
            by_at: self.by_at,
            skip_to: self.skip_to,
            group: self.inner().clone(),
        }
    }

    fn inner(&self) -> &Group {
        self.groups.last().unwrap()
    }

    fn inner_mut(&mut self) -> &mut Group {
        self.groups.last_mut().unwrap()
    }
}

/// Where the bracket opening at `at` closes, or the last token.
fn closing(tokens: &[Token], at: usize) -> usize {
    let opening = tokens[at].text.as_str();
    let close = match opening {
        "(" => ")",
        "[" => "]",
        _ => ">",
    };
    let mut depth = 0;
    for (index, token) in tokens.iter().enumerate().skip(at) {
        if token.text == opening {
            depth += 1;
        } else if token.text == close {
            depth -= 1;
            if depth == 0 {
                return index;
            }
        }
    }
    tokens.len() - 1
}

/// Where the type named at `at` ends: past its parameters (`DECIMAL(38,
/// 2)`), its fields (`ARRAY<INT>`, `STRUCT<>`, `INTERVAL DAY TO SECOND`) or
/// its time zone. An alias, collation or window name ends where it starts.
fn type_end(tokens: &[Token], at: usize) -> usize {
    let (word, after) = (tokens[at].folded(), tokens[at + 1].text.as_str());
    if after == "(" || (after == "<" && TYPES_WITH_FIELDS.contains(&word.as_str())) {
        return closing(tokens, at + 1);
    }
    if after == "<>" && word == "struct" {
        return at + 1;
    }
    if word == "interval" && INTERVAL_FIELDS.contains(&tokens[at + 1].folded().as_str()) {
        return if tokens[at + 2].folded() == "to" {
            at + 3
        } else {
            at + 1
        };
    }
    if word == "timestamp" {
        let following: Vec<String> = tokens[at + 1..(at + 5).min(tokens.len())]
            .iter()
            .map(Token::folded)
            .collect();
        for zone in TIME_ZONES {
            if following.len() >= zone.len()
                && following[..zone.len()]
                    .iter()
                    .zip(zone)
                    .all(|(a, b)| a == b)
            {
                return at + zone.len();
            }
        }
    }
    at
}

/// Where the token after the one at `at` is, past the type a `::`, `?::`
/// or `AS` there introduces.
fn next(tokens: &[Token], at: usize) -> usize {
    let casts = matches!(tokens[at].text.as_str(), "::" | "?::") || tokens[at].folded() == "as";
    if casts && tokens[at + 1].kind == Kind::Word {
        return type_end(tokens, at + 1) + 1;
    }
    at + 1
}

/// How many arguments the call whose `(` is at `at` is given.
fn arguments(tokens: &[Token], mut at: usize) -> usize {
    let (mut commas, mut depth) = (0, 0i64);
    let end = closing(tokens, at);
    while at < end {
        let text = tokens[at].text.as_str();
        depth += match text {
            "(" | "[" => 1,
            ")" | "]" => -1,
            _ => 0,
        };
        if text == "," && depth == 1 {
            commas += 1;
        }
        at = next(tokens, at);
    }
    commas + 1
}

/// Each lambda's parameters, folded, with the tokens they are bound in:
/// from the parameters to the end of the lambda's argument.
fn lambda_scopes(tokens: &[Token]) -> Vec<(usize, usize, BTreeSet<String>)> {
    let mut scopes = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        if token.text != "->" || index == 0 {
            continue;
        }
        let before = &tokens[index - 1];
        let (start, names) = if before.text == ")" {
            let (mut depth, mut start) = (0i64, index - 1);
            while start > 0 {
                depth += match tokens[start].text.as_str() {
                    ")" => 1,
                    "(" => -1,
                    _ => 0,
                };
                if depth == 0 {
                    break;
                }
                start -= 1;
            }
            let names = tokens[start..index]
                .iter()
                .map(Token::folded_name)
                .filter(|name| !name.is_empty())
                .collect();
            (start, names)
        } else {
            (index - 1, BTreeSet::from([before.folded_name()]))
        };
        let (mut depth, mut end, mut at) = (0i64, tokens.len() - 1, start);
        while at < tokens.len() {
            let text = tokens[at].text.as_str();
            depth += match text {
                "(" | "[" => 1,
                ")" | "]" => -1,
                _ => 0,
            };
            if depth < 0 || (depth == 0 && text == ",") {
                end = at;
                break;
            }
            at = next(tokens, at);
        }
        scopes.push((start, end, names));
    }
    scopes
}

type Column = (usize, State, Key);

struct Scan {
    /// How many tokens the expression has.
    end: usize,
    /// Its tokens, then four of kind `End`, which also stand before the
    /// first.
    tokens: Vec<Token>,
    scopes: Vec<(usize, usize, BTreeSet<String>)>,
    parsed: BTreeMap<Key, bool>,
    /// For each trial under way, the readings as a column left to try.
    trials: Vec<Vec<Column>>,
    budget: i64,
    cut: bool,
    /// Whether the whole expression parses.
    parses: bool,
    /// Where the bracket around each token closes, or the end.
    closes: Vec<usize>,
}

impl Scan {
    fn new(tokens: Vec<Token>) -> Scan {
        let end = tokens.len();
        let mut around = Vec::with_capacity(end);
        let mut closed: BTreeMap<isize, usize> = BTreeMap::new();
        let mut open_at: Vec<isize> = vec![-1];
        for (index, token) in tokens.iter().enumerate() {
            if (token.text == ")" || token.text == "]") && open_at.len() > 1 {
                closed.insert(open_at.pop().unwrap(), index);
            }
            around.push(*open_at.last().unwrap());
            if token.text == "(" || token.text == "[" {
                open_at.push(isize::try_from(index).unwrap());
            }
        }
        let closes = around
            .iter()
            .map(|at| closed.get(at).copied().unwrap_or(end))
            .collect();
        let sentinel = Token {
            kind: Kind::End,
            text: String::new(),
            start: usize::MAX,
            end: usize::MAX,
        };
        let mut padded = tokens;
        padded.extend(std::iter::repeat_n(sentinel, 4));
        let scopes = lambda_scopes(&padded);
        Scan {
            end,
            tokens: padded,
            scopes,
            parsed: BTreeMap::new(),
            trials: Vec::new(),
            budget: TRIAL_STEPS * i64::try_from(end + 1).unwrap(),
            cut: false,
            parses: true,
            closes,
        }
    }

    /// The token at `at`; one of kind `End` before the first and past the
    /// last.
    fn token(&self, at: isize) -> &Token {
        match usize::try_from(at) {
            Ok(at) if at < self.tokens.len() => &self.tokens[at],
            _ => self.tokens.last().unwrap(),
        }
    }

    fn at(&self, index: usize, offset: isize) -> &Token {
        self.token(isize::try_from(index).unwrap() + offset)
    }

    fn references(&mut self) -> Vec<NameReference> {
        let mut state = State::new();
        let mut found = Vec::new();
        let mut index = 0;
        while index < self.end {
            let (following, parses) = self.step(index, &mut state, Some(&mut found));
            index = following;
            self.parses = self.parses && parses;
        }
        self.parses = self.parses && state.groups.len() == 1 && state.settled();
        found
    }

    /// Whether the tokens from `index` to the end of the current bracket
    /// parse, read from `state`.
    fn parses_from(&mut self, index: usize, state: &State) -> bool {
        let key = state.key(index);
        if let Some(&known) = self.parsed.get(&key) {
            return known;
        }
        if self.budget <= 0 || self.trials.len() >= DEEPEST_TRIAL {
            self.cut = true;
            return true; // Undecided: the expression will be refused.
        }
        self.trials.push(Vec::new());
        let (mut index, mut state) = (index, state.trial());
        let parses = loop {
            let parses = self.reads_on(index, &mut state);
            if parses {
                break true;
            }
            let Some((at, column, form)) = self.trials.last_mut().unwrap().pop() else {
                break false;
            };
            (index, state) = (at, column);
            self.keep(form, false);
        };
        // Each form taken on the way parsed.
        for (_, _, form) in self.trials.pop().unwrap() {
            self.keep(form, true);
        }
        self.keep(key, parses);
        parses
    }

    fn keep(&mut self, key: Key, parses: bool) {
        if !self.cut {
            self.parsed.insert(key, parses);
        }
    }

    /// Whether every token from `index` to the end of the current bracket
    /// may stand where it does, read from `state`.
    fn reads_on(&mut self, mut index: usize, state: &mut State) -> bool {
        while index < self.end && state.groups.len() > 1 {
            let known = self.parsed.get(&state.key(index)).copied();
            if let Some(known) = known
                && (!known || state.groups.len() == 2)
            {
                return known; // Read from here before, from this state.
            }
            if known == Some(true) {
                // The rest of this deeper bracket parses: go on after it.
                index = self.closes[index] + 1;
                state.groups.pop();
                state.operand = false;
                state.name_next = false;
                state.after_is = false;
                continue;
            }
            if self.budget <= 0 {
                self.cut = true;
                return true;
            }
            self.budget -= 1;
            let (following, parses) = self.step(index, state, None);
            index = following;
            if !parses {
                return false;
            }
        }
        // A `)` checks its own group as it closes it.
        state.groups.len() == 1 || state.settled()
    }

    /// Reads the token or name at `index`, adding a name read as a value to
    /// `found`. Gives where the next starts, and whether what was read may
    /// stand where it does.
    fn step(
        &mut self,
        index: usize,
        state: &mut State,
        found: Option<&mut Vec<NameReference>>,
    ) -> (usize, bool) {
        if isize::try_from(index).unwrap() <= state.skip_to {
            return (index + 1, true);
        }
        if !is_name(self.tokens[index].kind) {
            if let Some(found) = found
                && self.star(index, state)
            {
                let token = &self.tokens[index];
                found.push(NameReference {
                    parts: vec!["*".into()],
                    span: token.start..token.end,
                    bare: false,
                });
            }
            return (index + 1, self.punctuation(index, state));
        }
        let mut last = index;
        while self.tokens[last + 1].text == "." && is_name(self.tokens[last + 2].kind) {
            last += 2;
        }
        if self.tokens[last + 1].text == "." && self.tokens[last + 2].text == "*" {
            last += 2; // `t.*`, every column of `t`.
        }
        let (value, parses) = self.read_name(index, last, state);
        if value && let Some(found) = found {
            found.push(NameReference {
                parts: self.tokens[index..=last]
                    .iter()
                    .step_by(2)
                    .map(Token::unquoted)
                    .collect(),
                span: self.tokens[index].start..self.tokens[last].end,
                bare: last == index && self.tokens[index].kind == Kind::Word,
            });
        }
        (last + 1, parses)
    }

    /// Whether the token at `index` is a `*` reading every column, as in
    /// `struct(*)`; `COUNT(*)` counts rows instead.
    fn star(&self, index: usize, state: &State) -> bool {
        if self.tokens[index].text != "*" || !state.operand {
            return false;
        }
        !(state.inner().opener == "count"
            && self.at(index, -1).text == "("
            && self.at(index, 1).text == ")")
    }

    /// Reads the token at `index`, no name, and says whether it may stand
    /// where it does.
    fn punctuation(&mut self, index: usize, state: &mut State) -> bool {
        let token = self.tokens[index].clone();
        let previous = self.at(index, -1).clone();
        let (text, operand) = (token.text.as_str(), state.operand);
        state.name_next = matches!(text, "::" | "?::");
        state.after_is = false;
        if text == "(" || text == "[" {
            state.groups.push(Group::new(previous.folded()));
            state.operand = true;
            return operand || text == "[";
        }
        if text == ")" || text == "]" {
            let closed = state.groups.len() > 1 && state.groups.pop().unwrap().cases.is_empty();
            state.operand = false;
            return closed && (!operand || previous.text == "(");
        }
        if matches!(token.kind, Kind::Number | Kind::Str) {
            state.operand = false;
            // `INTERVAL 1 YEAR 2 MONTHS` gives a value after each unit.
            return operand || has(words::DATETIME_UNITS, &previous.folded());
        }
        if matches!(text, "+" | "-" | "~") || (text == "*" && operand) {
            state.operand = text != "*"; // A sign, or COUNT(*).
            return true;
        }
        if previous.end == token.start
            && OPERATOR_PAIRS.contains(&format!("{}{text}", previous.text).as_str())
        {
            return true; // The second half of an operator.
        }
        if text == "!" && self.at(index, 1).text != "=" {
            return operand; // Not, as in `!x`.
        }
        state.operand = true;
        !operand // An operator after an operand, or a comma.
    }

    /// Whether `word` at `at` may stand right after an operand.
    fn follows(&self, word: &str, state: &State, at: usize) -> bool {
        let group = state.inner();
        let before = self.at(at, -1);
        let after = self.at(at, 1).folded();
        if let Some(steps) = case_steps(word) {
            return group.cases.last().is_some_and(|last| steps.contains(last));
        }
        if matches!(word, "preceding" | "following" | "row") {
            return group.frame;
        }
        if let Some(next) = followed_by(word) {
            return next.contains(&after.as_str());
        }
        if word == "nulls" {
            return matches!(after.as_str(), "first" | "last")
                || matches!(before.folded().as_str(), "ignore" | "respect");
        }
        if matches!(word, "first" | "last") {
            return before.folded() == "nulls";
        }
        if word == "to" {
            return has(words::DATETIME_UNITS, &before.folded());
        }
        if has(words::DATETIME_UNITS, word) {
            // `INTERVAL 1 DAY`, `'1' DAY TO HOUR`, `INTERVAL (n) DAYS`.
            return matches!(before.kind, Kind::Number | Kind::Str)
                || before.text == ")"
                || before.folded() == "to"
                || self.at(at, -2).folded() == "interval";
        }
        if has(TAKE_AN_OPERAND, word) || matches!(word, "asc" | "desc") {
            return true;
        }
        self.implicit_alias(at, state)
    }

    /// Whether the name at `at`, right after an operand, is an alias STRUCT
    /// takes without AS: ``struct(x a, y `b c`)``.
    fn implicit_alias(&self, at: usize, state: &State) -> bool {
        matches!(self.at(at, 1).text.as_str(), "," | ")") && state.inner().opener == "struct"
    }

    /// Whether a FROM follows the token at `at` in its own bracket.
    fn a_from_follows(&self, at: usize) -> bool {
        let mut depth = 0i64;
        let end = self.closes[at];
        for token in self.tokens.get(at + 1..end).unwrap_or(&[]) {
            depth += match token.text.as_str() {
                "(" | "[" => 1,
                ")" | "]" => -1,
                _ => 0,
            };
            if depth == 0 && token.folded() == "from" {
                return true;
            }
        }
        false
    }

    /// Whether the name at `at`, first in its call, is the field of
    /// `EXTRACT(DOW FROM d)` or the unit of `dateadd(DAY, 1, d)`. A
    /// qualified call is someone's own function, whose arguments are none.
    fn names_a_field(&self, at: usize, state: &State) -> bool {
        let opener = state.inner().opener.as_str();
        if self.at(at, -1).text != "(" || self.at(at, -3).text == "." {
            return false;
        }
        if opener == "extract" {
            return true;
        }
        let three = UNIT_FIRST_OF_THREE.contains(&opener);
        let unit_first =
            UNIT_FIRST.contains(&opener) || (three && arguments(&self.tokens, at - 1) == 3);
        unit_first && self.at(at, 1).text == ","
    }

    /// Reads the name at `index` to `last`, and says whether it is read as
    /// a value and whether it may stand where it does.
    fn read_name(&mut self, index: usize, last: usize, state: &mut State) -> (bool, bool) {
        let previous = self.at(index, -1).clone();
        let after = self.tokens[last + 1].clone();
        let single = index == last;
        let word = if single {
            self.tokens[index].folded()
        } else {
            String::new()
        };
        let name = if single {
            self.tokens[index].folded_name()
        } else {
            String::new()
        };
        let (operand, name_next, after_is) = state.taken();
        if matches!(previous.text.as_str(), "." | ":") {
            return (false, true); // A field or a JSON path's step.
        }
        if name_next {
            let end = type_end(&self.tokens, last);
            if end > last {
                state.skip_to = isize::try_from(end).unwrap();
            }
            return (false, true);
        }
        if after.text == "(" && operand {
            // `NOT (x)` and `CASE (x) WHEN` open forms, and others call.
            self.opens_a_form(index, &word, state, after_is);
            state.operand = true;
            return (false, true);
        }
        if after.text == "=>" {
            return (false, operand); // A named argument.
        }
        if !name.is_empty()
            && self.scopes.iter().any(|(start, end, names)| {
                (*start..=*end).contains(&index) && names.contains(&name)
            })
        {
            return (false, operand); // A lambda's parameter.
        }
        if word.is_empty() {
            // Quoted or qualified: a value, but for a quoted field, unit or
            // alias, as in ``EXTRACT(`DOW` FROM d)``.
            if self.tokens[index].kind == Kind::Quoted
                && single
                && (if operand {
                    self.names_a_field(index, state)
                } else {
                    self.implicit_alias(last, state)
                })
            {
                return (false, true);
            }
            return (true, operand);
        }
        if isize::try_from(index).unwrap() == state.by_at {
            state.operand = true;
            return (false, true);
        }
        if after.folded() == "by"
            && (word == "order"
                || (WINDOW_BY.contains(&word.as_str()) && state.inner().opener == "over"))
        {
            // After an operand, or first in `OVER (`.
            state.by_at = isize::try_from(last + 1).unwrap();
            return (false, !operand || previous.text == "(");
        }
        if operand {
            return (!self.opens_a_form(index, &word, state, after_is), true);
        }
        if word == "except" && previous.text == "*" && after.text == "(" {
            // The columns `* EXCEPT (...)` leaves out are read as no value.
            state.skip_to = isize::try_from(closing(&self.tokens, last + 1)).unwrap();
            return (false, true);
        }
        // Right after an operand: an operator or a clause's word.
        let parses = self.follows(&word, state, index);
        state.operand = has(TAKE_AN_OPERAND, &word);
        state.name_next = TAKE_A_NAME.contains(&word.as_str());
        state.after_is = word == "is";
        let group = state.inner_mut();
        group.frame = group.frame || matches!(word.as_str(), "rows" | "range");
        if word == "end" {
            group.cases.pop();
        } else if let Some(step) = case_steps(&word).map(|_| case_word(&word))
            && !group.cases.is_empty()
        {
            *group.cases.last_mut().unwrap() = step;
        }
        (false, parses)
    }

    /// Whether `word`, where an operand is expected, is grammar, moving
    /// `state` on past it.
    fn opens_a_form(
        &mut self,
        index: usize,
        word: &str,
        state: &mut State,
        after_is: bool,
    ) -> bool {
        match self.fixed_form(index, word, state, after_is) {
            Some(form) => form,
            None => self.tried_form(index, word, state),
        }
    }

    /// Whether `word` at `index` is grammar by where it stands, or `None`
    /// if a trial decides.
    fn fixed_form(
        &self,
        index: usize,
        word: &str,
        state: &mut State,
        after_is: bool,
    ) -> Option<bool> {
        let after = self.at(index, 1).clone();
        let opens = self.at(index, -1).text == "(";
        if LITERAL_WORDS.contains(&word) {
            return Some(true);
        }
        if after_is && AFTER_IS.contains(&word) {
            state.after_is = matches!(word, "not" | "distinct");
            state.operand = matches!(word, "not" | "distinct" | "from");
            return Some(true);
        }
        if state.inner().frame && FRAME_WORDS.contains(&word) {
            state.operand = word == "between";
            return Some(true);
        }
        if self.names_a_field(index, state) {
            return Some(true);
        }
        let opener = state.inner().opener.clone();
        if opens && opener == "trim" && TRIM_SIDES.contains(&word) {
            // `TRIM(BOTH FROM s)`, `TRIM(BOTH from FROM s)`, and
            // `TRIM(both)` and `TRIM(both < x)`, which have no FROM.
            let form = self.a_from_follows(index);
            let third = self.at(index, 3);
            let column = self.at(index, 2).folded() == "from"
                && (matches!(
                    third.kind,
                    Kind::Word | Kind::Quoted | Kind::Number | Kind::Str
                ) || matches!(third.text.as_str(), "(" | "+" | "-" | "~"));
            state.operand = form && (after.folded() != "from" || column);
            return Some(form);
        }
        if opens
            && opener == "over"
            && matches!(word, "rows" | "range")
            && (FRAME_STARTS.contains(&after.folded().as_str()) || after.kind == Kind::Number)
        {
            state.inner_mut().frame = true;
            state.operand = true;
            return Some(true);
        }
        if word == "interval" {
            // `INTERVAL 1 DAY` and `INTERVAL -1 DAY`, where `interval - 1`
            // and `interval ASC` read a column.
            let signed = matches!(after.text.as_str(), "+" | "-");
            let value =
                signed || (after.kind == Kind::Word && !has(TAKE_AN_OPERAND, &after.folded()));
            let unit = has(
                words::DATETIME_UNITS,
                &self.at(index, 2 + isize::from(signed)).folded(),
            );
            state.operand = matches!(after.kind, Kind::Number | Kind::Str)
                || after.text == "("
                || (value && unit);
            return Some(state.operand);
        }
        if word == "where" {
            // FILTER takes a WHERE, and nothing else does.
            state.operand = opens && opener == "filter";
            return Some(state.operand);
        }
        if TYPED_LITERALS.contains(&word) && after.kind == Kind::Str {
            state.operand = true;
            return Some(true);
        }
        None
    }

    /// Whether `word` at `index`, which may open a form or be a column,
    /// opens the form: whether what follows then parses.
    fn tried_form(&mut self, index: usize, word: &str, state: &mut State) -> bool {
        let mut form = state.trial();
        let at = isize::try_from(index).unwrap();
        if word == "when" && at == state.case_at {
            // A searched CASE's first WHEN, or a subject named `when`.
            let cases = &mut form.inner_mut().cases;
            cases.pop();
            cases.push("when");
        } else if word == "case" {
            form.inner_mut().cases.push("case");
            form.case_at = at + 1;
        } else if word != "not"
            && !(matches!(word, "all" | "distinct")
                && self.at(index, -1).text == "("
                && !no_call_opener(&form.inner().opener))
        {
            return false;
        }
        form.operand = true;
        if !self.trials.is_empty() && state.groups.len() == 2 {
            // In the bracket this trial reads to the end of: take the form,
            // and keep the column to come back to.
            let column = (index + 1, state.clone(), form.key(index + 1));
            self.trials.last_mut().unwrap().push(column);
        } else if !self.parses_from(index + 1, &form) {
            return false;
        }
        state.inner_mut().cases = form.inner().cases.clone();
        state.case_at = form.case_at;
        state.operand = true;
        true
    }
}

/// The CASE step a word is, as a constant the groups can hold.
fn case_word(word: &str) -> &'static str {
    match word {
        "when" => "when",
        "then" => "then",
        "else" => "else",
        _ => "end",
    }
}
