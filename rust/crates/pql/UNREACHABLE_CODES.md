# Codes no invalid query can produce

Every code of `predictive_query.schema.codes` is exercised by an invalid case of the PQL corpus (`tests/parity/pql/invalid.json`), except the six `PQL_REWRITE` codes, which belong to query rewriting, and the codes below.
Each is listed with why no corpus case can give it as an error, and, where a caller can still reach it, how.
The core keeps every one of them, with today's message, so that it answers as Python does if the reason ever stops holding.

## Codes that only label warnings

A warning never refuses a query, so these codes appear in a resolved query's `validation_response.warnings` and never in a refusal's errors.

- `PQL_TARGET_FILTER_ONLY_ENTITY_COLUMNS`: only ever a warning, given when a target filter reads only the entity's table.
- `PQL_TIME_POTENTIAL_LEAKAGE`: only ever a warning, given when the entity table has no time column but other columns in the query do.
- `PQL_TYPE_AGGREGATION_ON_BOOL`: only ever a warning, given for `MIN` or `MAX` of a boolean column.
- `PQL_TYPE_AGGREGATION_WANTS_NUMERIC_TEXT_OR_TIME`: only ever a warning, given for `MIN` or `MAX` of a column that is not numerical, text or a timestamp.
- `PQL_TYPE_AGGREGATION_WANTS_NUMERIC`: only ever a warning, given for `SUM` or `AVG` of a column whose semantic type is not numerical.
- `PQL_TYPE_AGGREGATION_WANTS_CATEGORICAL`: only ever a warning, given for `LIST_DISTINCT` of a column that is not categorical, multicategorical or an ID.
- `PQL_TYPE_COMPARED_BY_VALUE`: only ever a warning, given when a categorical or ID column is compared with `<`, `>`, `<=`, `>=` or a string operator.

## Codes only a custom profile or validator option reaches

The corpus names each case's profile, so it can only reach the five profiles `predictive_query` defines, which are also all that `pql_validate` and `validate_json` take by name.
But `Profile`, `Capabilities` and `PredictiveQueryValidator` are public, in Python and in the core, so a caller can reach the codes below.
`rust/crates/pql/tests/custom_profiles.json` covers each of them, with today's Python answer recorded beside every row by `custom_profiles.py`, and `custom_profiles.rs` checks that the core gives the same codes, spans and messages.

- `PQL_MODEL_NO_COUNTERFACTUAL`: needs a model that refuses `ASSUMING` but takes a windowed target.
  Of the five profiles, only the tabular one refuses `ASSUMING`, and it refuses every windowed target before resolution, while `ASSUMING` on a static target is refused with `PQL_TIME_ASSUMING_REQUIRES_TEMPORAL`.
  `validate` with a custom `Profile` whose capabilities set `counterfactuals=False` gives it.
- `PQL_MODEL_NO_FORECASTING`: needs a model that refuses forecasting but takes a windowed target.
  Of the five profiles, only the tabular and anomaly ones refuse forecasting, and both refuse every windowed target before resolution, while a forecast of a static target is refused with `PQL_TASK_FORECAST_REQUIRES_TEMPORAL_TARGET`.
  `validate` with a custom `Profile` whose capabilities set `forecasting=False` gives it.
- `PQL_MODEL_NO_KEY_COLUMN_FILTER`: needs a model that refuses key columns in filters, and all five profiles allow them.
  `validate` with a custom `Profile` whose capabilities set `key_columns_in_filters=False` gives it.
- `PQL_TYPE_AGGREGATION_ON_TIMESTAMP_UNSUPPORTED`: needs `allow_array_targets=True` and `allow_timestamp_arrays=False` on `PredictiveQueryValidator`, whose defaults are `False` and `True`.
  `validate` builds the validator with the defaults, and so does every other caller in the repository (`rfm/query/parser.py` in the relational engine, and the rewriting in `bp_filter_transformer.py`).
- `PQL_TYPE_TARGET_DTYPE_NOT_PREDICTABLE`: needs a target of an array type, which only `allow_array_targets=True` creates; as above, nothing in the repository sets it.

## Codes an earlier check always takes first

- `PQL_MODEL_ENTITY_ID_LIST_NOT_ALLOWED`: an array compared with `=` is refused earlier, by the type check, with `PQL_TYPE_OPERATOR_ON_ARRAY`, before the model check runs.
- `PQL_MODEL_ENTITY_ID_LIST_REQUIRED`: a single value after `IN` is refused earlier, by the type check, with `PQL_TYPE_OPERATOR_NEEDS_ARRAY`, before the model check runs.
- `PQL_TARGET_LIST_DISTINCT_CLASSIFY_STATIC`: a `LIST_DISTINCT` target without a window under `CLASSIFY` is always refused earlier by the time-range check, with `PQL_TIME_LIST_DISTINCT_NEEDS_RANGE_FOR_RANK`.
- `PQL_TASK_PROBLEM_TYPE_UNKNOWN`: the grammar's problem types are only `RANK`, `CLASSIFY` and `FORECAST`, the three the check accepts.
