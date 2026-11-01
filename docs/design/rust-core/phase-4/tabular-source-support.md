# Tabular database inputs: keep the table explicit

**Decision: do not add `Schema.from_source()` support for Tabular.**
The earlier proposal in this file is superseded.
Tabular continues to accept in-memory rows; Relational keeps its existing source-backed schema and sampling paths.

## Why does `Schema.from_source()` take a dictionary?

```python
Schema.from_source(source, tables={"customers": "adventureworks_customers"})
```

The key is the name used in PQL, schema access, and metadata overrides.
The value is the table name in the source.
The implementation carries these separately as `name` and `source_name`.
This changes neither the database table nor the source object.

| Benefit | Example |
|---|---|
| Shorter queries | `customers.churn` instead of a long physical table name |
| Stable queries across environments | Map `customers` to a different physical table in each environment |
| Consistent schema references | Keys, time columns, and relationships refer to logical table names |

This is useful indirection, not a requirement that every caller rename tables.
PQL supports backtick-quoted names containing spaces or dots, so aliases are not simply a workaround for those characters.
PQL resolves `table.column` against the schema; warehouse namespace resolution belongs to the source.
These are verified behaviors, not a claim about the original author's motivation.

**Separate API improvement worth considering:** accept a list for selection without renaming, retaining the dictionary for aliases.

```python
# Proposed shorthand, not implemented:
Schema.from_source(source, tables=["customers", "orders"])

# Existing equivalent:
Schema.from_source(source, tables={"customers": "customers", "orders": "orders"})
```

Normalize the list into an identity mapping at the API boundary.
The Relational sampling design would not need to change.

## A simpler direction for Tabular

Separate reading data from asking the model a question.
There is no reason to require a graph-shaped schema just to read one table.

| Option | Assessment |
|---|---|
| Existing `read_table(connection, query=...)` | Use now: explicit SQL result becomes a DataFrame for existing Tabular APIs |
| Proposed `source.read_table(...)` convenience | Investigate next: reuse source configuration while returning ordinary rows |
| Lazy `source.table(...)` object accepted by prediction APIs | Defer: adds execution, lifetime, filtering and row-selection contracts |
| Automatic joins or relational feature generation | Separate feature; not implicit in a reader |

Existing usage, with an already-open driver connection:

```python
from sdm_connectors import read_table

rows = read_table(
    connection,
    query="SELECT id, age, income, churn FROM customer_features",
)
```

The caller then passes these rows through the existing in-memory Tabular API.
`read_table()` fetches the full query result; it does not automatically choose or bound context examples.
The caller must constrain large reads explicitly.

**Recommendation:** retain this explicit read-then-predict model for now.
If connection reuse is awkward, investigate a `Source.read_table()` convenience before inventing a new lazy table abstraction.
That method is only a proposal: driver connections and Spark/Snowpark source adapters need different execution support, so it cannot simply delegate every source to today's reader.
Keep data-reading mechanics in connectors and model preparation in Rust.
A shared API should reuse connections and prediction calls without pretending that a relational schema and a flat table are the same input.

## Combining tables remains explicit

An inner join can duplicate customers with several orders or drop customers with none.
[RDBLearn](https://arxiv.org/html/2602.18495v1) instead constructs relational features, including aggregates, for an augmented single table.
That approach has not been validated for this client and Kumo Tabular.
A caller-provided feature table or SQL result can already be read as rows; automatic feature generation needs a separate design.

## Evidence

- [Schema API](../../../../packages/structured-data-models-client/src/sdm_client/schema.py): `from_source()` documents logical-to-source names.
- [Source translation](../../../../packages/structured-data-models-client/src/sdm_client/materializers/sources.py): `engine_graph()` passes separate `name` and `source_name` fields.
- [PQL grammar](../../../../packages/structured-data-models-pql/src/predictive_query/grammar/PQLGrammar.g4): `NAME`, `QUOTED_ID`, and `FQN_COLUMN`.
- [Reader](../../../../packages/structured-data-models-connectors/src/sdm_connectors/reader.py): `read_table()` returns a DataFrame from a driver connection.
