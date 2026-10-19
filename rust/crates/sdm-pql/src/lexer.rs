// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The tokens of `PQLGrammar.g4`, read as its ANTLR lexer reads them.
//!
//! At each position every rule is tried, the longest match wins, and on equal
//! length the rule listed first in the grammar wins. Keywords are exact
//! strings, so `Count` is an identifier and `FOR  EACH` is `FOR` then `EACH`.
//! Positions count code points: rows from 1, ended by `\n` alone, and columns
//! from 0, as ANTLR's Python runtime counts them.

/// A token type, in the order the grammar defines them, which breaks ties.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Open,
    Close,
    Comma,
    Bool,
    FqnColumn,
    WildcardColumn,
    Predict,
    Classify,
    Rank,
    Top,
    Forecast,
    Timeframes,
    ForEach,
    For,
    Where,
    Assuming,
    And,
    Or,
    Not,
    Aggr,
    RelOp,
    Null,
    NegInfty,
    TimeUnit,
    Id,
    QuotedId,
    Int,
    Decimal,
    Str,
    Date,
    Time,
    Skip,
}

impl Kind {
    /// How a message names a token of this kind.
    pub fn describe(self) -> &'static str {
        match self {
            Kind::Open => "'('",
            Kind::Close => "')'",
            Kind::Comma => "','",
            Kind::Bool => "a boolean",
            Kind::FqnColumn => "a column",
            Kind::WildcardColumn => "a table's wildcard column",
            Kind::Predict => "PREDICT",
            Kind::Classify => "CLASSIFY",
            Kind::Rank => "RANK",
            Kind::Top => "TOP",
            Kind::Forecast => "FORECAST",
            Kind::Timeframes => "TIMEFRAMES",
            Kind::ForEach => "FOR EACH",
            Kind::For => "FOR",
            Kind::Where => "WHERE",
            Kind::Assuming => "ASSUMING",
            Kind::And => "AND",
            Kind::Or => "OR",
            Kind::Not => "NOT",
            Kind::Aggr => "an aggregation",
            Kind::RelOp => "an operator",
            Kind::Null => "NULL",
            Kind::NegInfty => "-INF",
            Kind::TimeUnit => "a time unit",
            Kind::Id => "a name",
            Kind::QuotedId => "a quoted name",
            Kind::Int => "an integer",
            Kind::Decimal => "a decimal number",
            Kind::Str => "a string",
            Kind::Date => "a date",
            Kind::Time => "a time of day",
            Kind::Skip => "whitespace",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Token {
    pub kind: Kind,
    pub text: String,
    /// The row of the first character, from 1.
    pub line: usize,
    /// The column of the first character, in code points from 0.
    pub column: usize,
}

/// Where the lexer found a character no token starts with.
#[derive(Debug)]
pub struct LexError {
    pub line: usize,
    pub column: usize,
    pub found: char,
}

const BOOLS: &[&str] = &["true", "false", "True", "False", "TRUE", "FALSE"];
const AGGRS: &[&str] = &[
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    "COUNT_DISTINCT",
    "FIRST",
    "LAST",
    "LIST_DISTINCT",
    "COUNT",
    "sum",
    "avg",
    "min",
    "max",
    "count_distinct",
    "first",
    "last",
    "list_distinct",
    "count",
];
const REL_OPS: &[&str] = &[
    "!=",
    "<=",
    ">=",
    "<",
    ">",
    "=",
    "IS",
    "IS NOT",
    "IN",
    "IS IN",
    "LIKE",
    "NOT LIKE",
    "CONTAINS",
    "NOT CONTAINS",
    "STARTS WITH",
    "ENDS WITH",
    "is",
    "is not",
    "in",
    "is in",
    "like",
    "not like",
    "contains",
    "not contains",
    "starts with",
    "ends with",
];
const NULLS: &[&str] = &["NULL", "null"];
const NEG_INFTYS: &[&str] = &["-INF", "-INFINITY", "-inf", "-infinity"];
const TIME_UNITS: &[&str] = &[
    "days", "weeks", "months", "hours", "minutes", "seconds", "DAYS", "WEEKS", "MONTHS", "MINUTES",
    "HOURS", "SECONDS",
];

/// The keyword rules, each a set of exact spellings.
const WORDS: &[(Kind, &[&str])] = &[
    (Kind::Predict, &["predict", "PREDICT"]),
    (Kind::Classify, &["classify", "CLASSIFY"]),
    (Kind::Rank, &["rank", "RANK"]),
    (Kind::Top, &["top", "TOP"]),
    (Kind::Forecast, &["forecast", "FORECAST"]),
    (Kind::Timeframes, &["timeframes", "TIMEFRAMES"]),
    (Kind::ForEach, &["for each", "FOR EACH"]),
    (Kind::For, &["for", "FOR"]),
    (Kind::Where, &["where", "WHERE"]),
    (Kind::Assuming, &["assuming", "ASSUMING"]),
    (Kind::And, &["and", "AND"]),
    (Kind::Or, &["or", "OR"]),
    (Kind::Not, &["not", "NOT"]),
];

/// The characters the grammar's `WS` rule skips (`PQLGrammar.g4:214`).
pub fn is_grammar_space(c: char) -> bool {
    matches!(
        c,
        ' ' | '\t'
            | '\n'
            | '\r'
            | '\u{0b}'
            | '\u{0c}'
            | '\u{1c}'
            | '\u{1d}'
            | '\u{1e}'
            | '\u{1f}'
            | '\u{85}'
            | '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'
            ..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}'
    )
}

fn is_id_start(c: char) -> bool {
    c == '_' || c.is_ascii_alphabetic()
}

fn is_id_char(c: char) -> bool {
    c == '_' || c.is_ascii_alphanumeric()
}

/// Matches of a rule at one position, as lengths in code points.
struct At<'a> {
    text: &'a [char],
    pos: usize,
}

impl At<'_> {
    fn get(&self, offset: usize) -> Option<char> {
        self.text.get(self.pos + offset).copied()
    }

    /// The longest of `words` spelled exactly here.
    fn longest_word(&self, words: &[&str]) -> usize {
        words
            .iter()
            .filter(|w| {
                let mut k = 0;
                w.chars().all(|c| {
                    let same = self.get(k) == Some(c);
                    k += 1;
                    same
                })
            })
            .map(|w| w.chars().count())
            .max()
            .unwrap_or(0)
    }

    fn run(&self, from: usize, pred: impl Fn(char) -> bool) -> usize {
        let mut k = from;
        while self.get(k).is_some_and(&pred) {
            k += 1;
        }
        k - from
    }

    /// `ID`, from `from`.
    fn id(&self, from: usize) -> usize {
        if !self.get(from).is_some_and(is_id_start) {
            return 0;
        }
        1 + self.run(from + 1, is_id_char)
    }

    /// `QUOTED_ID`, from `from`: `` '`' (~'`' | '``')+ '`' ``, a backtick
    /// in the name doubled and any other character, a line break included,
    /// as it stands (KO2-36; the reference's grammar takes neither).
    fn quoted_id(&self, from: usize) -> usize {
        if self.get(from) != Some('`') {
            return 0;
        }
        let mut k = from + 1;
        loop {
            match self.get(k) {
                None => return 0,
                Some('`') if self.get(k + 1) == Some('`') => k += 2,
                Some('`') => break,
                Some(_) => k += 1,
            }
        }
        if k == from + 1 { 0 } else { k + 1 - from }
    }

    fn name(&self, from: usize) -> usize {
        self.id(from).max(self.quoted_id(from))
    }

    /// `FQN_COLUMN` and `WILDCARD_COLUMN`.
    fn columns(&self) -> (usize, usize) {
        let first = self.name(0);
        if first == 0 || self.get(first) != Some('.') {
            return (0, 0);
        }
        let second = self.name(first + 1);
        let fqn = if second > 0 { first + 1 + second } else { 0 };
        let wildcard = if self.get(first + 1) == Some('*') {
            first + 2
        } else {
            0
        };
        (fqn, wildcard)
    }

    fn digits(&self, from: usize) -> usize {
        self.run(from, |c| c.is_ascii_digit())
    }

    /// `INT`: `-?[0-9]+`, from `from`.
    fn int(&self, from: usize) -> usize {
        let sign = usize::from(self.get(from) == Some('-'));
        let digits = self.digits(from + sign);
        if digits == 0 { 0 } else { sign + digits }
    }

    /// `EXP`: `[Ee] [+-]? INT`, from `from`.
    fn exp(&self, from: usize) -> usize {
        if !matches!(self.get(from), Some('e' | 'E')) {
            return 0;
        }
        let sign = usize::from(matches!(self.get(from + 1), Some('+' | '-')));
        let int = self.int(from + 1 + sign);
        if int > 0 {
            return 1 + sign + int;
        }
        // `e-5`: the '-' read as INT's own sign rather than EXP's.
        if sign == 1 {
            let int = self.int(from + 1);
            if int > 0 {
                return 1 + int;
            }
        }
        0
    }

    /// `DECIMAL`: `'-'? INT '.' [0-9]+ EXP? | '-'? INT EXP`.
    fn decimal(&self) -> usize {
        let mut best = 0;
        for sign in [0, 1] {
            if sign == 1 && self.get(0) != Some('-') {
                continue;
            }
            let int = self.int(sign);
            if int == 0 {
                continue;
            }
            let after = sign + int;
            if self.get(after) == Some('.') {
                let fraction = self.digits(after + 1);
                if fraction > 0 {
                    let end = after + 1 + fraction;
                    best = best.max(end + self.exp(end));
                }
            }
            let exp = self.exp(after);
            if exp > 0 {
                best = best.max(after + exp);
            }
        }
        best
    }

    /// `STR`, either quote, escapes allowed or not: the longest prefix that
    /// the rule's regular language holds.
    fn string(&self) -> usize {
        let Some(quote) = self.get(0) else { return 0 };
        if quote != '"' && quote != '\'' {
            return 0;
        }
        let escapable =
            |c: char| c == quote || matches!(c, '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u');
        // Two states: inside the string, and just after an escaping '\'.
        let (mut inside, mut escaping) = (true, false);
        let mut best = 0;
        let mut k = 1;
        while inside || escaping {
            let Some(c) = self.get(k) else { break };
            let mut next_inside = false;
            let mut next_escaping = false;
            if inside {
                if c == quote {
                    best = k + 1;
                } else {
                    next_inside = true;
                    if c == '\\' {
                        next_escaping = true;
                    }
                }
            }
            if escaping && escapable(c) {
                next_inside = true;
            }
            inside = next_inside;
            escaping = next_escaping;
            k += 1;
        }
        best
    }

    /// `DATEINT sep DATEINT sep DATEINT` for each separator in `seps`.
    fn three_runs(&self, seps: &[char]) -> usize {
        let first = self.digits(0);
        if first == 0 {
            return 0;
        }
        let Some(sep) = self.get(first).filter(|c| seps.contains(c)) else {
            return 0;
        };
        let second = self.digits(first + 1);
        if second == 0 || self.get(first + 1 + second) != Some(sep) {
            return 0;
        }
        let third = self.digits(first + second + 2);
        if third == 0 {
            0
        } else {
            first + second + third + 2
        }
    }

    fn whitespace(&self) -> usize {
        if self.get(0) == Some('#') {
            return 1 + self.run(1, |c| c != '\r' && c != '\n');
        }
        self.run(0, is_grammar_space)
    }

    /// The token here: the longest match, the first rule on a tie.
    fn token(&self) -> Option<(Kind, usize)> {
        let (fqn, wildcard) = self.columns();
        let mut candidates = vec![
            (Kind::Open, usize::from(self.get(0) == Some('('))),
            (Kind::Close, usize::from(self.get(0) == Some(')'))),
            (Kind::Comma, usize::from(self.get(0) == Some(','))),
            (Kind::Bool, self.longest_word(BOOLS)),
            (Kind::FqnColumn, fqn),
            (Kind::WildcardColumn, wildcard),
        ];
        for (kind, words) in WORDS {
            candidates.push((*kind, self.longest_word(words)));
        }
        candidates.extend([
            (Kind::Aggr, self.longest_word(AGGRS)),
            (Kind::RelOp, self.longest_word(REL_OPS)),
            (Kind::Null, self.longest_word(NULLS)),
            (Kind::NegInfty, self.longest_word(NEG_INFTYS)),
            (Kind::TimeUnit, self.longest_word(TIME_UNITS)),
            (Kind::Id, self.id(0)),
            (Kind::QuotedId, self.quoted_id(0)),
            (Kind::Int, self.int(0)),
            (Kind::Decimal, self.decimal()),
            (Kind::Str, self.string()),
            (Kind::Date, self.three_runs(&['/', '-', '.'])),
            (Kind::Time, self.three_runs(&[':'])),
            (Kind::Skip, self.whitespace()),
        ]);
        // `max_by_key` keeps the last of equal maxima, so walk in reverse to
        // keep the first rule.
        candidates
            .into_iter()
            .rev()
            .filter(|(_, len)| *len > 0)
            .max_by_key(|(_, len)| *len)
    }
}

/// Every token of `query`, whitespace and comments dropped, or the first
/// character no token starts with.
pub fn tokenize(query: &[char]) -> Result<Vec<Token>, LexError> {
    let mut tokens = Vec::new();
    let (mut pos, mut line, mut column) = (0, 1, 0);
    while pos < query.len() {
        let at = At { text: query, pos };
        let Some((kind, len)) = at.token() else {
            return Err(LexError {
                line,
                column,
                found: query[pos],
            });
        };
        if kind != Kind::Skip {
            tokens.push(Token {
                kind,
                text: query[pos..pos + len].iter().collect(),
                line,
                column,
            });
        }
        for &c in &query[pos..pos + len] {
            if c == '\n' {
                line += 1;
                column = 0;
            } else {
                column += 1;
            }
        }
        pos += len;
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(text: &str) -> Vec<(Kind, String)> {
        let chars: Vec<char> = text.chars().collect();
        tokenize(&chars)
            .expect("lexes")
            .into_iter()
            .map(|t| (t.kind, t.text))
            .collect()
    }

    #[test]
    fn longest_match_then_first_rule() {
        use Kind::*;
        let cases: &[(&str, &[(Kind, &str)])] = &[
            ("2024.01.02", &[(Date, "2024.01.02")]),
            ("2024.01", &[(Decimal, "2024.01")]),
            ("countx", &[(Id, "countx")]),
            ("count", &[(Aggr, "count")]),
            ("FOR EACH", &[(ForEach, "FOR EACH")]),
            ("FOR  EACH", &[(For, "FOR"), (Id, "EACH")]),
            ("is invalid", &[(RelOp, "is in"), (Id, "valid")]),
            ("--3.5", &[(Decimal, "--3.5")]),
            ("1e+-5", &[(Decimal, "1e+-5")]),
            ("-infinity", &[(NegInfty, "-infinity")]),
            ("true", &[(Bool, "true")]),
            ("`a b`.c", &[(FqnColumn, "`a b`.c")]),
            // KO2-36: a doubled backtick and a line break inside the quotes.
            ("t.`a``b`", &[(FqnColumn, "t.`a``b`")]),
            ("t.````", &[(FqnColumn, "t.````")]),
            ("t.`a\nb`", &[(FqnColumn, "t.`a\nb`")]),
            ("t.`a.b`", &[(FqnColumn, "t.`a.b`")]),
            ("`a``b`", &[(QuotedId, "`a``b`")]),
            ("x.*", &[(WildcardColumn, "x.*")]),
            ("12:00:00", &[(Time, "12:00:00")]),
            ("'it\\'s'", &[(Str, "'it\\'s'")]),
            ("'a\\'", &[(Str, "'a\\'")]),
            ("'a\\\\' = 'b", &[(Str, "'a\\\\' = '"), (Id, "b")]),
        ];
        for (text, want) in cases {
            let want: Vec<(Kind, String)> = want.iter().map(|(k, t)| (*k, t.to_string())).collect();
            assert_eq!(kinds(text), want, "{text:?}");
        }
    }

    #[test]
    fn a_quoted_name_is_closed_by_a_single_backtick() {
        let lexes = |text: &str| tokenize(&text.chars().collect::<Vec<_>>()).is_ok();
        // Empty, or never closed: a doubled backtick is part of the name.
        for text in ["t.``", "t.`a", "t.`a``", "t.```"] {
            assert!(!lexes(text), "{text:?}");
        }
        // A line break inside the quotes moves the next token's row.
        let chars: Vec<char> = "t.`a\nb` x".chars().collect();
        let tokens = tokenize(&chars).unwrap();
        assert_eq!((tokens[1].line, tokens[1].column), (2, 3));
    }
}
