# Random streams of the reference (T0.6)

This lists every random draw the reference makes, where it is, what it draws, and how the public seed reaches it.
The reference is `rust-core-reference` at `70aa159`, with the owner's decisions D1 (the example shuffle stays on `default_rng`; `508845fe` is not adopted) and D2 (the seeded SQLite draw of `f0f324c` is accepted).
Entries marked "Register" are proposed for `register.yaml` in T0.8: the Rust core must reproduce them exactly.
Entries marked "Known non-repeatable" are draws the reference itself does not repeat, because their fixes were dropped in T0.3.

Paths are relative to `packages/` on the reference.
`engine/` stands for `structured-data-models-relational-engine/src/sdm_relational_engine/`, and `client/` for `structured-data-models-client/src/sdm_client/`.

## How this was checked

Every stream below that the core must reproduce was rewritten in plain Python (MT19937, PCG64 with its bounded draws, and the shuffles), and compared with the reference's own numpy, pandas and native sampler.
All four checks pass: the SQLite coefficients for seven seeds including `2**32`, `2**64 + 5` and `10**30`; `DataFrame.sample` for three seeds; `default_rng(seed).shuffle` for four seeds including `2**70`; and the native neighbor draw, its stream continuing across calls, and its default seed.
The script is kept outside the repository, at `~/.local/share/sdm-dbx/prototypes/random-streams/verify.py`.
A search of the four Python packages and the C++ source finds no other generator: no `random` module, no `np.random` calls other than those below.

## The public seed

The seed is the `random_seed` option of a relational `PredictiveQuery.materialize(...)` (`client/query.py:90`), passed to the engine's `_get_task_table` and `materialize_task` (`client/materializers/relational.py:186-230`).
The client refuses a `bool`, a non-integer and a negative seed with `INVALID_REQUEST` (`client/materializers/relational.py:662-676`).
Left out, the engine's default applies: `_RANDOM_SEED = 42` (`engine/rfm/rfm.py:91`, used as the default of every `random_seed` parameter from `rfm.py:822` on).
Passed explicitly as `None`, every stream below is unseeded, with the behavior given per stream.
The single-table (tabular) path takes no options at all (`client/query.py:298`), so its one draw has a fixed seed.

## Client-side generators

| ID | Generator | Where | What it draws | Seed | `None` | Status |
|---|---|---|---|---|---|---|
| G1 | numpy `default_rng` (PCG64) | `engine/rfm/base/sampler.py:644`, shuffle at `:728` | The order of candidate entity positions, one Fisher-Yates shuffle per call of `get_valid_entity_index` | `random_seed` | Fresh OS entropy on every `sample_target` call | Register |
| G2 | numpy `RandomState` (MT19937) through `random_state` | `engine/rfm/base/utils.py:184-208`, used at `engine/rfm/backend/sqlite/sampler.py:88` | The four coefficients of the SQLite row hash | `random_seed` | Not called; the unseeded SQLite path uses `RANDOM()` | Register (D2) |
| G3 | pandas `DataFrame.sample` (legacy `RandomState`) | `engine/rfm/backend/local/sampler.py:188-192` | Which entity rows, and in which order, the in-memory backend returns | `random_seed` | numpy's global `RandomState`: the process-wide stream, seeded from the OS at import and moved by any `np.random.seed` call | Register, with a decision for `None` |
| G4 | C++ `std::mt19937` in `relationallib.NeighborSampler` | `engine/csrc/neighbor_sampler.cpp:108-139`, `:282-312`, `:534`; seeded from `engine/rfm/backend/local/sampler.py:98-99` | Which neighbors a static hop keeps, in the in-memory backend | `random_seed`, re-applied at every subgraph (every batch) | The sampler's own stream continues from wherever the previous call left it; it starts at seed `123` | Register |
| G5 | pandas `DataFrame.sample` | `client/materializers/tabular.py:28`, `:288` | Which context rows a single-table request keeps when there are more than the endpoint accepts | Always `42`; `random_seed` does not reach it | Not applicable | Register |

### G1, the example shuffle

`sample_target` makes one `rng = np.random.default_rng(random_seed)` per call and shuffles in place each time it asks for candidate positions: once for a shared static train and test set, once each for test and train otherwise, once per step back in time for a temporal training set, and once to pick the forecast entity (`engine/rfm/base/sampler.py:708-850`).
The order of those calls is part of the stream.
The algorithm, checked against numpy 2.5.3:

1. `SeedSequence(seed).generate_state(4, uint64)` gives `s0..s3`.
2. PCG64 (128-bit LCG, XSL-RR output) is seeded with `initstate = s0 << 64 | s1` and `initseq = s2 << 64 | s3`, in PCG's `setseq` order (step, add `initstate`, step).
3. A 32-bit draw takes the low half of a 64-bit output and keeps the high half for the next 32-bit draw.
4. A bounded draw of `[0, i]` with `i < 2**32` masks a 32-bit draw to the smallest all-ones mask covering `i` and rejects values above `i`.
5. The shuffle swaps position `i` with a bounded draw of `[0, i]`, for `i` from `n - 1` down to `1`.

The core has to port numpy's `SeedSequence` mixing as well as PCG64, since D1 keeps this stream.
Any non-negative integer seeds it, including seeds of `2**64` and above.

### G2, the SQLite coefficients

`random_state(seed)` returns `RandomState(seed)` for a seed below `2**32`, which is MT19937's `init_genrand(seed)`.
A larger seed is split into its 32-bit words, least significant first, and seeds MT19937 by `init_by_array(words)`.
`randint(0, 2**31 - 1, size=4)` then draws each coefficient by masking a 32-bit output with `0x7FFFFFFF` and rejecting values above `2**31 - 2`.
NEP 19 freezes this stream across numpy releases.

### G3, the in-memory entity sample

`pkey_map.sample(n=num_rows, random_state=random_seed, ignore_index=True)` runs only when no entity ids are given and the table has more rows than asked for.
It equals `RandomState(seed).permutation(len(pkey_map))[:n]`: the legacy Fisher-Yates over `arange`, with the same MT19937 and bounded draws as G2, kept in drawn order.
pandas raises `ValueError` for a seed of `2**32` or more.
With `None`, pandas uses the global numpy stream, which the core cannot see; reproducing it is not possible, so the proposed entry is that `None` draws from OS entropy, which differs only in that a user's `np.random.seed` no longer makes it repeatable.

### G4, the native neighbor draw

The sampler is built once per in-memory model with no seed argument (`engine/rfm/backend/local/sampler.py:35`), so it is seeded with the default `123` (`neighbor_sampler.cpp:114`, binding default at `:554`).
`std::mt19937::seed(s)` is the same initialisation as numpy's `init_genrand(s)`, checked by comparing raw draws.
With a seed, `_sample_subgraph` reseeds before every call, so each batch of one request starts the stream afresh.
With `None`, nothing reseeds: the stream continues across batches and requests on that model, so a fresh process making the same calls draws the same neighbors, but any earlier call changes them.
Draws happen only on a static hop (the destination table has no time column, or there are no seed times) whose neighborhood is larger than the hop's count.
The draw is Floyd's sampling without replacement (`neighbor_sampler.cpp:296-310`): for `i` from `population - count` to `population - 1`, `r = rng() % (i + 1)`, and if `r` was already taken, `i` is taken instead.
The modulo bias is part of the stream and must be kept.
A temporal hop keeps the latest `count` neighbors with no draw (`:229`), a time-window hop keeps all of them (`:277`), and the query-data pass (`engine/rfm/backend/local/sampler.py:229`) asks for no draws, so none of these advance the stream.
The binding takes the seed as a C++ `int`: a public seed of `2**31` or more is refused with `INVALID_REQUEST` carrying pybind11's "incompatible function arguments" text, observed through `materialize` for `2**31`, `2**32` and `2**40`.

### G5, the tabular context sample

`candidates.sample(n=wanted, random_state=42).sort_index()` keeps the rows of `RandomState(42).permutation(len)[:wanted]` and then restores their table order, so only the chosen set depends on the stream.

## Not answer streams

- `uuid.uuid4()` names requests (`client/adapters/tabular.py:72`, `client/adapters/relational.py:72`) and predictions (`engine/rfm/rfm.py:1699`); none is part of an answer or a compared payload.
- `column_shuffle`, `category_shuffle`, `hop_shuffle` and `class_shuffle` (`engine/rfm/inference/config.py:53-68`) are flags sent to the endpoint (`engine/rfm/inference/payload.py:940-946`); any randomness they cause is the server's.

## Seeded draws done in SQL

Only the entity-table sample (`_sample_entity_table`) is random on SQL backends.
Neighbor traversal ignores the seed (`engine/rfm/base/sql_sampler.py:291-294`) and orders rows explicitly, as in `engine/rfm/backend/sqlite/sampler.py:543`.
A request with explicit entity ids draws nothing on any backend.

| Backend | Where | SQL with a seed | SQL without | Repeatable with a seed | Status |
|---|---|---|---|---|---|
| SQLite | `engine/rfm/backend/sqlite/sampler.py:63-92`, `:301-375` | `ORDER BY <hash>, <key>, rowid LIMIT n` for an integer key, else `ORDER BY <hash of rowid>, rowid LIMIT n` | `ORDER BY RANDOM() LIMIT n` | Yes: integer arithmetic only | Register (D2); unseeded is non-repeatable by design |
| DuckDB | `engine/rfm/backend/duckdb/sampler.py:116-132` | `SELECT ... FROM (SELECT * FROM t WHERE ...) USING SAMPLE reservoir(n ROWS) REPEATABLE (seed)` | The same without `REPEATABLE` | Yes on DuckDB 1.5.6: rows and order identical over repeated runs at 1, 8 and 10 threads on a three-million-row table | Register (SQL text; DuckDB's own stream, version recorded) |
| PostgreSQL | `engine/rfm/backend/postgres/sampler.py:111-122`, `:239-280` | `SELECT setseed(x)`, then `... ORDER BY random() LIMIT n` on the same cursor | The second statement only | Only on one server major version and one scan order; not checked live | Register (seed mapping and SQL text); repeatability not promised |
| Snowflake | `engine/rfm/backend/snow/sampler.py:79-140` | `SAMPLE ROW (n ROWS)`, with `n` capped at 1,000,000; the seed is ignored with one warning per sampler (`engine/rfm/base/sql_sampler.py:233-249`) | The same | No | Known non-repeatable (Snowflake half of `2e48359f`, and `7a01cd89`, `16e96df8` dropped) |
| Databricks (SQL warehouse or a notebook's Spark session) | `engine/rfm/backend/databricks/sampler.py:26-29`, `:249-270` | `TABLESAMPLE (p PERCENT) REPEATABLE (seed)`, the filters, then `LIMIT n` | The same without `REPEATABLE` | No: the sample is seeded, but `LIMIT` keeps arbitrary rows of it | Known non-repeatable (`300bab01` dropped); SQL text registered |

### The SQLite hash

With `p = 2**31 - 1` and `c0, c1, c2, c3` from G2, a key reference `ref` is hashed by the text

```sql
((((c3 * ((ref % p + p) % p) + c2) % p) * ((ref % p + p) % p) + c1) % p * ((ref % p + p) % p) + c0) % p
```

with the numbers written in decimal (`_seeded_hash`, `sqlite/sampler.py:69-92`).
The key is hashed when its declared or inferred dtype is an integer; otherwise `rowid` is hashed, so the draw then follows physical row ids, which `VACUUM` may renumber (`_seeded_order_by`, `:301-325`).
Every seed is accepted, including seeds of `2**32` and above (G2).

### The PostgreSQL seed

`x = 2 * v / (2**64 - 1) - 1`, where `v` is the big-endian unsigned value of `blake2b(str(seed).encode('ascii'), digest_size=8)` (`_postgres_seed`).
The core must compute the same double and pass it as the bound parameter, so the statement the server sees is the same.
The draw itself is the server's `random()`, which the core does not reproduce.

### The Databricks percentage

`p` is `min(100.0, 100.0 * n / total * 3.0)` when the table's row count is known and above `n`, else `100.0`, written into the SQL with Python's `repr` of a float.
A Rust float printed with `{}` differs from Python's `repr` for some values (Python writes `1e-05`, Rust `0.00001`), so the core has to format `p` as Python does, or register the SQL text change.

### Seed range per backend

| Backend | Seeds accepted |
|---|---|
| In-memory | `0` to `2**31 - 1`; larger seeds are refused with `INVALID_REQUEST` (G4), or by pandas with `ValueError` from `2**32` where G3 runs first |
| SQLite | Any non-negative integer |
| DuckDB | `0` to `2**31 - 1`; DuckDB's parser rejects a larger literal (observed on 1.5.6) |
| PostgreSQL | Any non-negative integer (it is hashed) |
| Snowflake | Ignored |
| Databricks | Not checked live |

### Rows read for type inference

These are not random draws, but they are unordered reads whose rows decide inferred types, listed so that no one mistakes them for repeatable.
Every SQL backend reads at most `_NUM_SAMPLE_ROWS = 1_000` rows (`engine/rfm/base/table.py:96`).
SQLite orders them by `rowid` (`engine/rfm/backend/sqlite/table.py:153`), and DuckDB returns a table in insertion order, so both repeat.
PostgreSQL, Snowflake and Databricks use a bare `LIMIT 1000` (`postgres/table.py:327`, `snow/table.py:197`, `databricks/table.py:338`), which is known non-repeatable (`f15d5bf4`, `440b8d60`, `572085ad` and their follow-ups dropped).

## Proposed register entries

1. RS-1, the example shuffle (G1): `default_rng(seed)` with `SeedSequence`, PCG64 and numpy's bounded Fisher-Yates, in `sample_target`'s call order; `None` draws OS entropy.
2. RS-2, the SQLite seeded entity draw (G2 and the SQLite hash): the coefficients, the SQL text, and the key or `rowid` fallback; seeds of any size.
3. RS-3, the in-memory neighbor draw (G4): MT19937 `init_genrand`, default seed `123`, reseeded per subgraph when seeded, continuing when not, Floyd's draw with `rng() % (i + 1)`, draws only on static hops.
4. RS-4, the in-memory entity sample (G3): legacy `RandomState(seed).permutation(len)[:n]` in drawn order; for `None`, the owner decides between OS entropy (proposed) and refusing to match the reference.
5. RS-5, the tabular context sample (G5): `RandomState(42).permutation(len)[:wanted]`, returned in table order.
6. RS-6, warehouse SQL text: DuckDB `reservoir ... REPEATABLE`, the PostgreSQL `setseed` mapping, Snowflake `SAMPLE ROW` with its warning, and the Databricks percentage formatted as Python's `repr`.
7. RS-7, the seed range: whether the core keeps the in-memory and DuckDB refusals of seeds from `2**31` on (today's behavior) or accepts them; accepting them needs a defined stream for those seeds, so this is an owner decision.

Known non-repeatable in the reference, and therefore compared by shape and not by rows: unseeded draws everywhere, the Snowflake sample, which rows Databricks's `LIMIT` keeps, the PostgreSQL draw across servers, and the inference reads of PostgreSQL, Snowflake and Databricks.

## A neighbouring ordering question

`sample_target` sorts a shared train set by entity time with `sort_values(ascending=False)` (`engine/rfm/base/sampler.py:777-782`), which is numpy's unstable quicksort; rows tying on time come out in introsort's order.
This is not a random stream, but the core has to reproduce that tie order or register the change; it is noted here for T0.8 and Phase 1.
