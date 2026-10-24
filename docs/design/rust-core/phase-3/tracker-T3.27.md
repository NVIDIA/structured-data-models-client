# T3.27: contract first

Lane a, from the phase head 569c6fc7 (T3.4 landed).
Brief: `$SDM_ORCH_STATE/phase-3/briefs/T3.27.md`; design: `design-remaining.md`, the shared section (S7) and the parts of T3.5, T3.6, T3.7, T3.28 and T3.13 that name contract types.
After T3.27 lands, no package edits `sdm-contract`; a missing sentence or method is a one-commit follow-up on the phase head.

## What changed (contract v28, the phase's working version)

- `ProviderError.error_class` in place of `refuses_expression`; the Databricks provider hands over its SQLSTATE and error class only.
- `DataLayer::failure` (`FailureFacts`), `spellings`, `bind` (`BoundRequest`, `BoundFetch`, `KeyData`) and `view_text` (`ViewTextAsk`, `ViewTextAnswer`), each with a default that keeps today's behaviour or refuses by name (`views_unread`).
- `ProviderChoice::Sql.view` (`ViewChoice`), refused by name by `sdm-data` until a dialect reads views; `SqlParam::Text`; `DescribePart::ViewDefinition` and `CatalogAnswer::ViewDefinition` (with its `ViewKind`); `DescribedSource.notes` (`ViewNote`, `Code::Note`).
- `Predict.model_endpoint` (`ModelEndpoint::ServingEndpoint`) and `ModelOutcome::Failed` (`ModelAnswer`).
- `ErrorCode::EndpointRefused` (fix 1), the fields `REFUSED_STATUS` and `REFUSED_CODE`, and `CoreError::refused_by_endpoint`; the Python host raises it as `NimRequestError(status, code=...)`.
- `sdm_contract::view`: the T3.5 pure modules' data types as they are at lane j's fix 2 (88850a2e), with `DefaultTexts`, `DeclaredType` and the asks `TypeClass`, `DeclaredType` and `ReasonWords`.
- The sentences of every message site below, the warning codes `VIEW_READ_IN_PART` and `SEED_IGNORED`, the group `view_read_in_part`, the advice `trust_expressions` (Python and Java spell it, and Java now spells `semantic_model` too), and `wording::PROPOSED`.

## Departures from the design

- **D1, core-written clauses (Claude r1 P2-6).**
  The view sentences take clauses the core writes (`{reason}`, `{lost}`, `{reasons}`, `{failure}`) and values the old client quoted, written by the core as Python wrote them (`sdm-pql`).
  The old client composes about 55 clauses into these messages (expression, filter, join, key, link, edge and relationship reasons); one sentence per clause and per composition would multiply the table, so S7's one sentence per message site holds and the clauses are core-written text.
  A clause that names a warehouse is the dialect's to word (`ViewTextAsk::ReasonWords`), so the core holds no dialect's name.
  No template before T3.27 took a clause; a later package adds none without the owner's yes.
- **D2, types left where they are.**
  `ViewSpec`, the YAML reader's types (`Document`, `NodeId`, `Value`, `Rules`, `YamlError`, `Outside`), the join walk's and the logical model's types, `NameReference`, `ExprDialect`, `DescribeRefusal`, `JoinJudgement` and `TooLarge` cross neither the data-layer port nor a host, so they stay in `sdm-schema`, `sdm-data` and `sdm-core`; putting `ViewSpec` in the contract would put a YAML parser there.
- **D3, a default crosses as texts.**
  Lane j's `read_parameters` takes a conversion closure, which no port carries; the core converts each default for each type it may be read as (`TypeClass` asked of the dialect) and hands the texts over (`DefaultTexts`).
- **D4, names.**
  The design's `Bound` is `BoundRequest` (the window bound has the name); `ModelAnswer` travels in `ModelOutcome::Failed`; the serving endpoint is `ServingEndpoint` and the Snowflake sentences name no dialect (`l0_scan`: a dialect is `sdm-data`'s alone), the warehouse said through `{database:backend}`.
- **D5, serving refusals (Codex r1 P1-4).**
  The old transport raises `NimRequestError(status, code=<the endpoint's code, the SDK's error code, INVALID_REQUEST or none>)`; the core's `ENDPOINT_REFUSED` carries the status and that code as fields, and the host raises the same class with them, so the caller's class, `status_code` and `code` match.

## Message sites and the sentence that covers each

"host" is a message the host says before or without the core, unchanged; "none" is a row with no message; "clause" is text inside another row's sentence (D1).

### `old-behaviour/T3.5-views.md`

| Row | Sentence |
|---|---|
| 1 | none |
| 2, 3, 4 | host |
| 5 | host (`{host.semantic_model!r}`) |
| 6, 7 | none |
| 8 | the view refusals below, `INVALID_REQUEST` |
| 9 | group `view_read_in_part` over the warnings of code `VIEW_READ_IN_PART` |
| 10 | none |
| 11 | `view_tables_absent` |
| 12 | none |
| 13 | `ViewNote`: `view_measure_note`, `view_time_column_note*`, `view_parameter_note` |
| 14, 15, 16, 17 | none |
| 18 | `view_name_invalid` |
| 19 | none |
| 20 | `view_not_found`; another failure `provider_failed` |
| 21, 22 | none |
| 23 | `view_not_a_metric_view` |
| 24 | `view_no_definition` |
| 25 | `view_not_a_mapping`; proposed `view_not_yaml` (P3J-1, P3J-9), `view_not_a_list` (P3J-2), `view_misses_entry` (P3J-3), `view_nests_itself` (P3J-4), `view_value_too_large` (P3J-10), `view_entry_unhashable` and `view_entry_not_text` (P3J-11) |
| 26 | `view_misses_source` |
| 27 | `view_source_unsupported` |
| 28 | `view_materialization_ignored` |
| 29 | clause of rows 32 and 35 |
| 30 | `view_source_unqualified`, `view_on_metric_view`, `no_such_table` |
| 31 | `view_join_not_a_mapping`, `view_join_nameless`, `view_join_skipped`, `view_join_skipped_with_it` |
| 32, 40 | `view_filter_refused`, `view_filter_refused_no_default`, `view_filter_refused_untrusted` |
| 33 | `view_fields_ignored` |
| 34 | `view_dimension_no_table`, `view_dimension_not_a_mapping`, `view_dimension_misses` |
| 35 | `view_dimension_refused` |
| 36 | `view_dimension_multiple_tables` |
| 37 | `view_measure_refused` |
| 38 | `view_join_name_collides` |
| 39 | `view_join_skipped` (its reasons clauses, from `FailureFacts::unusable`) |
| 41 | `view_expression_dropped`, `view_expression_dropped_untrusted` |
| 42 | `view_measure_column_shadowed` |
| 43 | none |
| 44 | `empty_table` |
| 45 | `view_column_refused` |
| 46 | `view_join_unchecked` |
| 47 | `view_join_skipped`, `view_join_skipped_with_it`, `view_derived_key_replaced` |
| 48 | `view_measure_note` |
| 49 | `view_time_column_note`, `view_time_column_note_added`, `view_time_column_note_renamed` |
| 50 | `view_parameter_note` |
| 51, 60 | `view_edge_dropped`; the validation failure `view_invalid` |
| 52 | `provider_failed`; proposed `view_semantic_not_found`, `view_semantic_definition_unread` |
| 53 | `view_composite_key` |
| 73 | `no_such_table` |
| 54 | `view_column_cross_table`, `view_columns_cross_table` |
| 55 | `view_table_expression_dropped`, `view_table_expression_dropped_untrusted` |
| 56 | none (`ViewTextAsk::DeclaredType`) |
| 57 | `view_table_cannot_hold`, `empty_table`; proposed: `view_column_refused` for a refused expression |
| 58 | none |
| 59 | `view_relationship_refused` |
| 61 | `view_expression_dropped*`, `view_table_expression_dropped*` |
| 62, 63 | none |
| 64, 65, 66, 67 | clause of rows 32, 35, 41 and 55 (`ViewTextAsk::ReasonWords` for a warehouse's words) |
| 68, 69, 70 | none |
| 71, 72 | `metric_view_as_table` |
| S5 (plan B) | `view_table_omitted`, `view_column_omitted`; a dialect reading no view `views_unread` |

### `old-behaviour/T3.6-snowflake.md`

| Row | Sentence |
|---|---|
| 1, 3, 4, 5, 6 | host |
| 2, 7, 8 | none |
| 9 | proposed `no_current_database` |
| 10 | `no_current_schema`; with `tables=`, `unspecified_schema` (`SNOW/table.py:88-93`) |
| 11 | none |
| 12 | `no_tables` |
| 13 | `provider_failed` |
| 14 | none |
| 15 | `no_such_table` |
| 16 | none |
| 17 | `table_no_columns` |
| 18, 19, 20, 21, 22 | none |
| 23 | as T3.4 says an unsupported column (`unsupported_type`) |
| 24 | `column_not_in_source` |
| 25 | none |
| 26, 38 | `empty_table` |
| 27, 28, 29, 30, 31, 32 | none |
| 33 | the shared key sentences (`key_not_unique_*`, `key_missing_values_*`, `key_no_value_*`, `key_repeats_*`) |
| 34, 35, 36 | none |
| 37 | none (proposed: a view counted by `COUNT(*)`, no refusal) |
| 39 | `reserved_name`, `reserved_names` |
| 40 | none |
| 41 | `seed_ignored` |
| 42, 43 | none |
| 44 | `entity_table_empty` |
| 45, 46 | none |
| 47 | `held_by_several_rows_one`, `held_by_several_rows`, `held_by_several_rows_many` |
| 48 | none |
| 49 | proposed `null_anchor` |
| 50, 51, 52, 53, 54, 55 | none |
| 56 | the core's refusals by their codes; a driver failure `provider_failed` |
| 57 | none (the core route has no `optimize`) |
| 58 | none |

### `old-behaviour/T3.7-lakebase.md`

| Row | Sentence |
|---|---|
| 1 to 9, 11, 12, 13 | host |
| 10, 14, 15, 16, 18, 20, 21, 23, 24, 26, 30, 33, 35, 36, 38 to 42, 44 | none |
| 17 | `no_current_schema` (`{database:backend}` PostgreSQL) |
| 19 | `no_tables` |
| 22 | `no_such_table` |
| 25, 27, 29 | as Phase 2's Postgres route says them (unchanged by T3.7) |
| 28, 37 | `empty_table` |
| 31 | the shared key sentences |
| 32, 43 | `provider_failed` |
| 34 | none (plan B) |

### `CL/core/databricks_serving.py` (T3.13)

| Site | Sentence |
|---|---|
| `_answered`, a value with no JSON form | `serving_value_not_json` |
| `_answered`, past the limit | `serving_past_limit` |
| `_response_body`, not one prediction row | `serving_prediction_row_count` |
| `_response_body`, no `response_json` string | `serving_response_column_missing` |
| `_decoded`, not JSON | `serving_not_json` |
| `_decoded`, not an object | `serving_not_an_object` |
| `_refusal`, with a code | `serving_refused` (`ENDPOINT_REFUSED`) |
| `_refusal`, with no code | `serving_failed`, or `serving_failed_coded` with the SDK's code |
| `_callers_fault` | `serving_callers_fault` (`ENDPOINT_REFUSED`, code `INVALID_REQUEST`) |
