# Phase 5 acceptance contract

The acceptance checks in this directory, and the Rust tests named `accept_*` in `rust/crates/pql/tests/`, test behavior through a small, fixed interface.
This file is that interface.
Everything it doesn't name is the implementer's choice.

## Results

A result is what `tests/parity/test_pql_corpus.py`'s `resolve` gives for a query, as JSON: an object with `warnings` and exactly one of `tree` and `errors`.

- `tree` is the `ValidatedPredictiveQuery` as `json_serde.to_json_dict` writes it.
- `errors` is a list of `{"code", "span", "message"}`; `span` is `null` or `{"start_row", "start_col", "end_row", "end_col"}`.
- `warnings` is a list of `{"category", "message"}`, one per `UserWarning` resolving the query gives today, in order, with `category` written `module.Class` (today, `builtins.UserWarning`).

Two results are equal when:

- trees are equal by value and by JSON type, so `1` and `1.0` differ;
- errors are equal in count, then in code and span, then in message;
- a `PQL_PARSE_SYNTAX` message may have any new text that is not empty and carries none of ANTLR's or pydantic's wording (the patterns in `_pql.py`, `FOREIGN_WORDING`), such as "mismatched input", a grammar token name like `FQN_COLUMN`, `<EOF>` or a pydantic error link;
- warnings are equal, except that P5.generated doesn't compare them for a query today's Python refuses with `PQL_PARSE_SYNTAX`.

## The Rust crate

The package is `sdm-pql` in `rust/crates/pql`, with the library `sdm_pql`.
Its root exports:

- `validate_json(query: &str, schema_json: &str, profile: &str, mode: Option<&str>) -> Result<String, E>`, where `E` implements `Display`:
  - `schema_json` is a `SchemaDefinition` as `predictive_query.schema.json_serde` writes it, the `schema` of a corpus case;
  - `profile` is the name of one of the five profiles: `RELATIONAL`, `TABULAR`, `TIMESERIES`, `ANOMALY` or `FINE_TUNING`;
  - `mode` is `None` for the profile's own mode, or `TRAINING` or `INFERENCE` to replace it, as `dataclasses.replace(profile, mode=Mode(mode))` does today;
  - `Ok` holds the result, as JSON text;
  - `Err` is for input it can't read: a schema that isn't one, or an unknown profile or mode.
    A query, whatever its text, gets `Ok`.
- `parse_date_json(literal: &str) -> String`, the reading of a date literal described below, as JSON text.

`rust/crates/pql/tests/accept_corpus.rs` reads every `*.json` under `tests/parity/pql/` and checks every case.
`rust/crates/pql/tests/accept_quirks.rs` checks every row of `accept_quirks.json`, beside it.
The crate's source, under `rust/crates/pql/src/`, spells every code of `schema/codes.py` as a string literal, such as `"PQL_PARSE_SYNTAX"`, the `PQL_REWRITE` ones included.

## The Python binding

`sdm_core._native` exposes the same two functions:

- `pql_validate(query, schema_json, profile, mode=None)` takes `str` arguments as `validate_json` does, `mode` being `None` or a `str`, and returns the result as a JSON `str`.
  - It raises `ValueError` for a schema it can't read, an unknown profile or an unknown mode, and `TypeError` for a `query` that isn't a `str`.
  - It never ends in a Rust panic or a crash, however deep the query nests; today's Python takes 300 nested parentheses.
- `pql_parse_date(literal)` returns the reading of `literal` as a JSON `str`.

The checks build the binding's wheel from this checkout with maturin and install it into the phase venv before calling it (`_rc.install_local_wheel`).

## Date literals

A date literal is the value of a date constant in today's tree: the grammar's `DATE` token, or `DATE`, one space and `TIME`.
Today `Constant.typed_value` reads it with `pd.to_datetime(literal)`; any exception, or a result that isn't a `Timestamp`, refuses it.

Its reading is `{"value", "warnings"}`:

- `value` is `null` when the literal is refused;
- otherwise it is `YYYY-MM-DDTHH:MM:SS`, with the year padded to four digits, followed by `.fffffffff` when pandas gives a fraction of a second, and by the UTC offset as `+HH:MM` (or `+HH:MM:SS`) when pandas gives a time zone, as `_pql.python_date` writes it;
- `warnings` are the `UserWarning`s `pd.to_datetime` gives, as in a result.

pandas reads some two-digit years relative to the current year, so a reading may depend on the date it runs; the checks compare both sides on the same day.

## `rust/crates/pql/DATES.md`

- Each rule is a line that starts with its id in bold, `**D1**`, `**D2**` and so on, after an optional list marker or heading marks.
  There are at least six, and no id repeats.
- Examples are rows of Markdown tables whose header starts `| Rule | Literal | Result |`.
  - The Rule cell holds one rule id.
  - The Literal cell holds the literal in backticks.
  - The Result cell holds either the word `refused`, or the `value` in backticks; it contains the word `warning` exactly when pandas warns.
- Every rule has at least one example, and every example must hold both for pandas and for `pql_parse_date`.
- Between them, the examples show a month-first reading, a day-first reading with its warning, a two-digit year, a time part and a refusal.

## `rust/crates/pql/UNREACHABLE_CODES.md`

Every code of `schema/codes.py` except the six `PQL_REWRITE` ones must be exercised by an invalid corpus case, or listed in this file as a line `` - `PQL_CODE`: reason ``.
A listed code must not be exercised by the corpus, and must not be produced today by any quirk row or by any of P5.generated's 50,000 queries.
The reason must be at least five words; whether it holds is the reviewer's call, and the report repeats the list.

## Changed syntax messages

`syntax_messages.py` prints every corpus `PQL_PARSE_SYNTAX` message whose text changed, with its old and new text, as a Markdown table.
The report lists the same changes, and the reviewer compares the two.

## Generators

`_generate.py` holds the seeded generators of P5.generated (`queries`, seed `20261002`) and P5.dates (`date_literals`, seed `20261003`).
Every choice comes from one `random.Random`, so a seed fixes the sequence, and the seeds are in the tests.
