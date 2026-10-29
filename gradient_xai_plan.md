# Gradient explanations (XAI) in the client: plan and tracker

The shared tracker for Jana, Claude and subagents. Update a row's status when it changes, and
record who decided and when. Nothing is built from a row marked NEEDS DECISION.

**Context**
- **Client PR #34:** branch `jg-enable-explain`. Rebased onto main at `12955d397` as
  `a01c08d3a` + `78b0a5ebb`, and pushed; the PR head is `78b0a5ebb`. The explanation work is
  local commits on top, `793a2a8c4`..`9de1214f8`, not pushed.
- **NIM:** `structured-data-nims` origin/main `0a8230d` explains every predict row and returns one
  `explanation` per prediction record.
- **Wire format** `sdm_gradient_v1`: described in the NIM repo at
  `sdm_profiles/kumo_{relational,tabular}_sdm/USER_GUIDE.md`, section "Gradient Explanations".
  `gradients` is `[output][row][column]`. `row_indices` are positions in the tables as submitted,
  not keys.

**Status values:** `NEEDS DECISION` (Jana chooses; the end-state column holds a proposal) ·
`TO DO` (agreed, not built yet; who agreed and when, in brackets) · `WAITING (on X)` (blocked by
another row or something outside) · `DONE` (finished, with where)

| Issue | End state / fix | Status |
|---|---|---|
| **R1** PR #34 is behind main and conflicts in `tabular.py`, `conftest.py` and `CHANGELOG.md` | Rebased onto main, conflicts resolved, the PR stays one commit; CI-equivalent checks pass locally | DONE: `a01c08d3a`; 2188 passed / 379 skipped (missing extras) / 0 failed |
| **R2** Docs said kumo-tabular doesn't explain; a test docstring stated the old `run_mode='fast'` rule as fact | Architecture capability table shows ✅; the docstring is scoped to "a deployment that…" | DONE: `78b0a5ebb` |
| **R3** The rebased branch is only local | Pushed; PR #34 shows the new commits | DONE: pushed 2026-10-28 18:59 PT, PR head `78b0a5ebb` |
| **N1** The NIM refused explanations over more than one row | The NIM explains every predict row | DONE: Jana, NIM `0a8230d` |
| **N2** Concern: a related row shared by two entities would get a mixed gradient credited to only one of them | No shared rows reach the NIM | DONE: verified by code reading plus a library test. The engine sampler copies rows per predict row, and the NIM scopes joins by `instance_id`. Not run end to end. |
| **L1** `explanation_predict_rows` in `TABULAR_LIMITS` and `RELATIONAL_LIMITS` (`models.py`) | Removed | DONE: `0da19289e` (`models.py`) |
| **L2** Tabular `_check_explanation_rows` (added by #34) | Removed, with its calls | DONE: `0da19289e` (`adapters/tabular.py`) |
| **L3** Relational one-entity explanation check (`adapters/relational.py` ~:380-400) | Removed | DONE: `0da19289e` (`adapters/relational.py`) |
| **L4** Other one-row assumptions may hide in the materializer, `decompose.py` or `client.py` | Grepped; none remain | DONE: `0da19289e`; grepped client, pql and engine, none left |
| **A1** `result.explanation` keeps only the first record's explanation and silently drops the rest | Removed. Replaced by `result.explanations`, a mapping from row key to that row's explanation, kept out of the DataFrame | DONE: `23db8a188` (`result.py` `explanations`) |
| **A2** Key for each explanation | Whatever identifies the row in `to_pandas()`: `id`; for forecasts `(id, forecast_step)`; `row_index` when the request has no entity key. The same rule for both models | DONE: `23db8a188` (`wire/explanations.py`, `client.py` `_by_step`, `_numbered_from`) |
| **A3** The instance block's `row_indices` is a position, not a key | Dropped, since the mapping key names the row; gradients become `[output][column]` | DONE: `23db8a188` |
| **A4** Related-table `row_indices` are positions in tables the caller never sees; keyless tables only carry the engine's synthetic `__node_id` (`payload.py:587-592`) | Each related row carries its primary key when the table has one, and always the row's values as sent; gradients stay `[output][row][column]`, aligned with those rows | DONE: `23db8a188` (`adapters/relational.py` `_related_rows_of`) |
| **A5** When a request is split across calls (#36), positions restart at 0 in each call | Translated inside each call's `answered()`, before merging | DONE: `23db8a188` (translated in each call's `answered()`) |
| **A7** Forecast `{'by_forecast_step': ...}` aggregation (`client.py` ~:1241-1305) | Removed; per-row keys `(id, forecast_step)` replace it | DONE: `23db8a188` (`client.py`) |
| **A9** `result.explanations` when explanations weren't requested | An empty dict | DONE: `23db8a188` |
| **A11** Type of an explanation entry | Plain dicts (A15). `result.explanations.to_pandas()` adds one flat table, with `hop` (A17). See "Reading it" in Proposed API | DONE: `23db8a188` (`Explanations.to_pandas()`) |
| **A12** `outputs` and `columns` repeat in every record | Kept per entry, as the NIM returns them | DONE: `23db8a188` |
| **A13** `columns` are the model's preprocessed columns; categoricals and datetimes appear as derived columns | Passed through as the NIM returns them, and documented (D3) | DONE: passed through; documented in `bff621aba` |
| **A14** `__contains__`, has-field and repr refer to `explanation` | Updated for `explanations` | DONE: `23db8a188` |
| **A15** Shape of one entry | `{'outputs': [...], 'instance': {'columns', 'gradients' [output][column]}, 'related': {table: {'keys', 'rows' (values as sent), 'columns', 'gradients' [output][row][column]}}}` | DONE: `23db8a188`, `ca99c782f` |
| **A16** Each model's entry has only the tables the caller sent. kumo-tabular has no related tables. For kumo-relational the client's engine sends the entity as two tables: a question row (`instance_table`: `instance_id`, entity reference, anchor time; no caller columns) and the entity's own table inside `related_tables`, and the NIM explains both | kumo-tabular entries have no `related` key. kumo-relational entries have no `instance` key: the client drops the question row's block when it reads the reply, and the entity's own table appears under `related` like every other table, at hop 0 | DONE: `ca99c782f` (relational entries have no `instance`) |
| **A17** The `hop` column in `to_pandas()` (A11) needs the client to work out each related row's distance from the entity | Computed from the relationships and foreign-key values in the payload sent, per call; checked against a real multi-hop payload | DONE: `23db8a188` (`_hops_from_the_entity`); checked on the golden one-hop payload; deeper hops are checked by T4 |
| **B2** The `live_nim` parity case in `test_every_backend_answers_alike.py` expects a NIM to refuse relational explanations | The case is removed; T1 covers both behaviours (the in-process backend refuses, the NIM answers) | DONE: `b2a30cd3e` |
| **T1** Tests for the new behaviour | Written per the `unittesting` skill. Cover tabular multi-row; relational multi-row with key and value translation; keyless related table; a split request whose second call restarts at 0; forecast keys; the in-process backend refusing `'explanation'`; `to_pandas()` | DONE: `a79c00aa0`, `ca99c782f` (unit tests with mocked NIM replies) |
| **T2** Old tests assert the one-row refusals and `result.explanation` | Updated or removed | DONE: `0da19289e`, `23db8a188`, `36de54755` |
| **T3** Local checks | ruff, format, mypy, hygiene and the CI pytest command all pass; counts recorded here | DONE: ruff, format, mypy (184 files), hygiene, client suite 1580 passed / 256 skipped / 0 failed, root tests 174 passed / 109 skipped (2026-10-29, rerun by the main agent) |
| **T4** The client has never run against a NIM that has `0a8230d` | Explanation cases (tabular and relational, multi-row, keyless related table, a split request) added to `test_live_nim.py` and `test_live_relational.py`, run once against a NIM built from `0a8230d`; output checked against A1-A5 | WAITING (on the rebuilt NIM image; Jana names the endpoint). Test code written: `9de1214f8` (`test_live_nim.py`, `test_live_relational.py`, `tests/test_skills.py`). Keyless related table added to the live fixture: `716f029cf` (agreed: Jana, 2026-10-29). Also at T4 time: add `explanation` to `SERVED` in `tests/test_skills.py` so the skill-matrix live test stops expecting a refusal (agreed: Jana, 2026-10-29). Both images triggered in the NIM repo CI (`SDM_PROFILE=both`) 2026-10-29 |
| **D1** `result.py` module docstring shows `result.explanation` | Shows `result.explanations` and its keys | DONE: `23db8a188` |
| **D2** `docs/reference/limits.md` explanation rows | The row limit is removed | DONE: `793a2a8c4` |
| **D3** `docs/reference/prediction-output.md` | Has a section on explanations: shape, keys, related rows, preprocessed columns | DONE: `793a2a8c4`, `964ded993`, `bff621aba` |
| **D4** `docs/about/architecture.md` | Explains the explanation design and why (from this file) | DONE: `793a2a8c4`, `bff621aba` |
| **D5** `skills/tabular/SKILL.md` and `skills/relational/SKILL.md` say one row per explanation | Multi-row, `result.explanations`, with an example | DONE: `793a2a8c4`, `bff621aba` |
| **D6** `qa/probe_tabular.py` | Probes a multi-row explanation | DONE: `793a2a8c4`; `qa/probe_relational.py` too (`964ded993`, `bff621aba`); not run, no NIM |
| **V1** CHANGELOG `## Unreleased` | Changed: multi-row explanations. Removed: `result.explanation` and `by_forecast_step`. Added: `result.explanations` | DONE: `793a2a8c4`, `bff621aba` |

## Proposed API

The IDs in brackets refer to rows in the tracker table above. Each item's status lives only
there.

### Asking

Asking is unchanged: add `'explanation'` to `outputs`. There is no row limit (L1-L3).

```python
result = client.predict(request, model=KumoRelational(),
                        outputs=['prediction', 'probabilities', 'explanation'])
```

### `result.explanations`

`result.explanations` is a mapping from a row's key to that row's explanation. It is kept out of
`result.to_pandas()` (A1). `result.explanation` and `{'by_forecast_step': ...}` no longer exist
(A1, A7).

| Request | Key | Example |
|---|---|---|
| has an entity id | the row's `id`, the same value as in `to_pandas()['id']` | `result.explanations['user_5']` |
| forecast | `(id, forecast_step)` | `result.explanations[('user_5', 3)]` |
| no entity id (single-table tabular) | `row_index` (A2) | `result.explanations[0]` |

### One entry (A15)

```python
{
  'outputs':  [...],                        # the explained scores, in order (see below)
  'instance': {
      'columns':   [...],                   # model's preprocessed numerical columns (A13)
      'gradients': [[...], ...],            # [output][column]
  },
  'related': {                              # absent for kumo-tabular; for kumo-relational it
                                            # holds every table incl. the entity's own, and
                                            # there is no 'instance' (A16)
      '<table>': {
          'keys':      [...],               # the table's primary key per row; absent if the table has none
          'rows':      [{...}, ...],        # each row's values as sent to the model
          'columns':   [...],               # model's preprocessed numerical columns (A13)
          'gradients': [[[...], ...], ...], # [output][row][column], rows aligned with keys and rows
      },
  },
}
```

`outputs` is copied from the NIM:
- classification gives every class probability in declared order, e.g.
  `{'kind': 'probability', 'class': 'yes'}`;
- regression gives the returned prediction, `{'kind': 'prediction', 'statistic': 'mean'}`
  (or `'median'`), plus each returned quantile when `quantiles` is requested.

A gradient is the change in that output score per unit change in that preprocessed column. It is
a diagnostic, not a causal effect or a feature-importance ranking.

### Which tables get gradients

| Model | Tables with gradients |
|---|---|
| kumo-tabular | the one table: each predicted row's own columns |
| kumo-relational | the predicted entity's own table, plus every related table in the subgraph the client gathered for that entity (all tables reached within the hop limit, up to 6). The engine's question row (`instance_table`) is explained by the NIM too, but the client drops that block (A16) |

- Only the rows being predicted and their related rows get gradients. The context rows (the
  labelled examples the model compares against) get none (`EnableInputGradients` in
  `sdm/explain/gradient.py`).
- Only numerical inputs as the model sees them get gradients. Categoricals and datetimes appear as
  derived numeric columns; a table with none has an empty `columns` (A13).
- Gradients are per row, not per path. A row reached through several paths, for example a product
  bought in two of the user's orders, appears once. Its gradient is its total effect through all
  of them, and the model gives no split by path.

### Examples (values illustrative)

kumo-tabular, request with an id column:

```python
result.explanations['cust_17']
# {'outputs': [{'kind': 'probability', 'class': 'no'},
#              {'kind': 'probability', 'class': 'yes'}],
#  'instance': {'columns': ['age', 'balance'],
#               'gradients': [[0.01, -0.03], [-0.01, 0.03]]}}
```

kumo-relational, user 5 with one order and two login events. `orders` has a primary key
(`order_id`), so its rows carry `keys`. `events` has none, so its rows are identified by their
values alone. Both are part of the same entry:

```python
result.explanations['user_5']
# {'outputs': [{'kind': 'probability', 'class': 'no'},
#              {'kind': 'probability', 'class': 'yes'}],
#  'related': {
#      'users':  {'keys': ['user_5'],
#                 'rows': [{'user_id': 'user_5', 'tenure_days': 410}],
#                 'columns': ['tenure_days'],
#                 'gradients': [[[-0.05]], [[0.05]]]},
#      'orders': {'keys': ['order_9003'],
#                 'rows': [{'user_id': 'user_5', 'order_id': 'order_9003', 'amount': 12.0}],
#                 'columns': ['amount'],
#                 'gradients': [[[-0.02]], [[0.02]]]},
#      'events': {'rows': [{'user_id': 'user_5', 'kind': 'login', 'at': '2026-10-01T09:00:00'},
#                          {'user_id': 'user_5', 'kind': 'login', 'at': '2026-10-03T18:20:00'}],
#                 'columns': ['kind_login', 'at_dayofweek'],    # model-derived names, illustrative (A13)
#                 'gradients': [[[0.01, 0.00], [0.02, -0.01]],
#                               [[-0.01, 0.00], [-0.02, 0.01]]]}}}
```

Keyed or keyless changes only how a row is labelled in the output. Both tables get gradients in
the same way.

### Reading it (A11)

Each entry is the plain nested dict shown above. `result.explanations.to_pandas()` gives one line
per (row, output, column), across all entities:

```
id      table     hop  key         row  output           column       gradient
user_5  users     0    user_5      0    probability_yes  tenure_days   0.05
user_5  orders    1    order_9003  0    probability_yes  amount        0.02
user_5  products  2    prod_77     0    probability_yes  price         0.004
user_5  events    1    None        1    probability_yes  kind_login   -0.02
```

- `hop` is how many links the row is from the entity (0 is the entity's own table; for kumo-tabular the one table, shown as `instance`). `None` for a row no relationship reaches. The client works
  it out from the relationships and foreign-key values it sent (A17).
- `output` is the matching `result.to_pandas()` column name (`probability_yes`, `prediction`, `quantile_0.1`).
- `key` is the row's primary key, or `None` for a keyless table. `row` is the row's position in
  that entity's `related[table]['rows']`.
