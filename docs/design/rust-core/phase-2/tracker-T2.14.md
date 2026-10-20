# Phase 2 tracker: T2.14

The design branch owns `tracker.md`; this file holds T2.14's sections until they fold in.
T2.14a, T2.14b, T2.14c and T2.14d land separately, each writing its own section.

## T2.14a

T2.14a widens the bridge freeze (`scripts/rust_core/bridge_freeze.py`, decision E50) to the rest of the core route's Python bridge (FR P3-2, the design's scope item 5), and records Phase 2's ledger of host-held rules, the "before" of the evidence pack.
It is a judging-path package and lands alone, before T2.14b relies on it.

### What changed

- `WHOLE` gains `_rust/codec.py`, `_rust/edits.py`, `_rust/errors.py` and `_rust/providers/postgres.py`: every function, method and skeleton of each is now frozen.
- `COVERED` gains `SDMClient._send_body` in `client.py`, the core's call bytes sent as the core wrote them.
- The docstring's list of frozen files names the four new files.
- `bridge_freeze.json` rewritten with `--write`: 69 entries added, none changed or gone.
- `bridge_changes.json` declares each of the 69 new entries `refactor` ("newly covered by the freeze (FR P3-2), unchanged; no rule added"), as the freeze requires of an entry new against the base.
- No product code changed.

### Phase 2's ledger (before)

Each row is a rule a core route still runs in the Python host at T2.14a's base (1d85bce3), from the mid-phase review MID2, with the package that moves it into the core and deletes it from the host.
Phase 1's ledger (`phase-1/design.md`, "Host-held semantics") stays in force for its own rows; its rows marked "stays host" are not repeated here.
Paths are under `packages/structured-data-models-client/src/sdm_client/`; a definition's line is its `def`.

| MID2 row | Rule held outside the core | Where, at 1d85bce3 | Owner |
|---|---|---|---|
| P2-4 | Host-resolved facts: whether the target is windowed and its entity table, read from the old validator's tree for every query but T2.18's `Unresolved` ones, though the core returns both (`QueryFacts`) | `_rust/materialize.py`'s `asked` (`:470`) | T2.14b |
| P2-4 | The old validator's pick: which validator resolves a query, chosen by a Python column-name check (`schema.py`'s `names_a_query_cannot_say`) | `query.py`'s `PredictiveQuery._resolve` (`:636`, the check at `:662-666`) | T2.14b |
| P2-4 | The host's own step numbering: its call counter stamped on a refusal as `_forecast_step`, which `client.py`'s `_said_of_the_part` prefers over the core's `forecast_step`, and set on batched calls that are not a forecast too | `_rust/predict.py`'s `predict` (`:395`, in `model_call`); read at `client.py:878` | T2.14b |
| P2-4 | Any ranking split still in Python: the split itself is the core's since T2.8 (`sdm-request`'s ranking); what remains is the host's call counter mapped to a candidate's position, stamped as `_ranking_candidate` | `_rust/predict.py`'s `predict` (`:398`); read at `client.py:895` | T2.14b |
| P2-2 (rest) | The candidate-repeat rule: a candidate equal to one given before it is a repeat by Python equality, as the reference's `dict.fromkeys` keeps the first | `_rust/materialize.py`'s `_candidates` (`:213`) | T2.14b |
| P2-3 | The Postgres value rules: an interval tagged `sdm:interval` = `timedelta` (`INTERVAL_TAG`, `:67`); a JSON scalar written with Python's `str()`, so JSON `true` reaches the core as `"True"` (`json_text`, `:222`); an interval built with months always 0 (`_interval`, `:234`); how a `numeric` past 76 digits or with no precision is read (`_Numeric`, `:325`); the core's key set encoded as JSON text for the dialect's `$1::jsonb` binding (`key_set_json`, `:494`); seed times written as pandas' `Timestamp.isoformat` writes them (`_isoformat`, `:524`) | `_rust/providers/postgres.py` | T2.14d |
| Phase 1 ledger (`phase-1/design.md`, design scope item 5) | The answer re-encoded before the core reads it, and the made-up too-deep body coupled to the core's reader's depth | `_rust/predict.py`'s `_encoded` (`:137`) and `_ENCODED_DEPTH` (`:134`), used at `:381` and `:409` | T2.14b |

### Deviations

None.
Packages landing after T2.14a (T2.11, T2.13b) declare their changes to the newly frozen files in their next fix round, as the brief says.
