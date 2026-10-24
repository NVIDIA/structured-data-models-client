# T3.5: views (Databricks metric views and Snowflake semantic views)

Lane j.
Brief: `$SDM_ORCH_STATE/phase-3/briefs/T3.5-v2.md`; design: `design-remaining.md`, the shared section and T3.5.
Parity reference: `old-behaviour/T3.5-views.md`.

## Stage 1: the pure pieces (no contract change)

| Piece | Where | Held by |
|---|---|---|
| YAML 1.1 as PyYAML 6.0.3 reads it, aliases as shared nodes | `sdm-schema/src/view/yaml.rs` | `view_yaml.rs`, 377 texts PyYAML read (`contract/view-yaml-vectors.json`) |
| One dialect-free `ViewSpec` for both kinds, entries read where the old converter reads them | `sdm-schema/src/view/{spec,metric,semantic}.rs` | `view_semantic.rs`; `sdm-core/tests/view_metric_model.rs` |
| The metric view's joins kept or skipped, in the old order | `sdm-schema/src/view/joins.rs`, `sdm-core/src/view_steps.rs` | `view_metric_model.rs`, whole against the old answer of 145 definitions |
| The logical model's steps: add_table, add_join with undo, add_relationship, measure notes, time column, invalid edges, validate | `sdm-schema/src/view/model.rs` | `view_model.rs` |
| Expression lexing, the unsafe-text guard, the allow-lists, the name scan, Snowflake qualifiers | `sdm-data/src/view_expr/{lex,check,names,qualifier,words,pattern}.rs` | `sdm-data/tests/view_expr.rs`, about 19,000 recorded answers and tokens of a megabyte |
| The metric view's expression helpers | `sdm-data/src/view_expr/metric.rs` | `sdm-data/tests/view_metric.rs` |
| `DESCRIBE TABLE EXTENDED`'s sections | `sdm-data/src/view_expr/describe.rs` (moves to `databricks.rs` once T3.4 lands) | `sdm-data/tests/view_describe.rs` |
| Python's `repr` and `str` of a quoted value | `sdm-pql/src/float_repr.rs` (`Shown`, `text_repr`), reached through `sdm-core/src/pql.rs` | `sdm-pql/tests/shown.rs` |
| The old client's sentences, from typed facts | `sdm-core/src/view_words.rs` (the dialect-naming ones in `sdm-data`) | `sdm-core/tests/view_words.rs` |

Vectors are recorded by `contract/fixtures/views/record.py`, which runs the old client's own view tests at 43e55f4f with its functions wrapped, and the review's cases through the same functions.

## Types for T3.27 to take into the contract

Each is crate-local until T3.27; none is in `sdm-contract` yet.

- `sdm_schema::view::yaml::{Document, NodeId, Value, Integer, Date, Timestamp}`: the definition's values; a default, a quoted section and a raw entry cross to the data layer and the words as nodes.
- `sdm_schema::view::spec::{ViewSpec, ViewKind, TableItem, TableSpec, Source, JoinSpec, Cardinality, JoinKeys, PrimaryKey, ColumnItem, ColumnSpec, Section, RelationshipItem, Relationship, ParameterItem, MeasureItem, Ignored, ViewRefusal, RefusalReason}`.
- `sdm_schema::view::joins::{JoinWalk, SkippedJoin, SkipReason, JoinMessage}`.
- `sdm_schema::view::model::{ViewModel, ModelTable, ModelColumn, Edge, JoinStep, KeyCheck, JoinNote, JoinRefusal, LinkFailure, RelationshipStep, RelationshipRefusal, ModelRefusal, KeyReason, EdgeFacts, EdgeReason, MeasureColumn, MeasureNote, Aggregation, TimeInput, RawAdd, Added, Choice, TimeColumnNote, TimeOutcome}`.
- `sdm_data::view_expr::{ExprDialect, ExprRefusal, ViewKind, NameReference, Unreadable}` and `metric::{KeyRef, ParameterDefault, DefaultKind, DefaultRefusal, ParameterInput, Parameter, ParameterRead, Unread, FilterRefusal, MeasureInput, DimensionRef, MeasureRead, MeasureRefusal, MeasureReason, JoinRef}`, `describe::{ViewDescription, DescribeRefusal}`: the inputs and answers of `DataLayer::view_text` (`ViewTextAsk`, `ViewTextAnswer`).
- `sdm_core::view_steps::JoinJudgement`.
- One `Sentence` per function of `sdm_core::view_words` and per refusal above: about 45 templates, each a sentence of the old client, its quoted values written by `sdm-pql`.

## Registered differences (proposed, Approver: owner)

- P3J-1: a definition that is no readable YAML is refused by name.
- P3J-2: a section that cannot be iterated is refused by name.
- P3J-3: a semantic definition missing an entry the old client reads is refused by name.
- P3J-4: a join nested in itself is refused by name.
- P3J-5: an expression a check's pattern cannot finish on is refused as unreadable (no input known).
- P3J-6: three YAML texts the scanner reads otherwise than PyYAML.
- P3J-7: a `!!set` is quoted in the order written.
- P3J-8: an integer past 4,300 digits is quoted whole.

Withdrawn in fix 1, now read as the old client reads them: integers past 128 bits; a composite relationship's tables.

## Stage 2 (after T3.27)

The ordered routine `sdm-core/src/view.rs` over these pieces, the wiring, providers, and each warehouse's `failure` facts.
