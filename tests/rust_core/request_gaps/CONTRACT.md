# Phase 8.6 acceptance contract

The checks in this directory hold the request coverage-gap list (`scripts/parity/request_coverage_gaps.json`) to what Phase 8.5's coverage-gap verifier found (`investigations/P85-coverage-gaps-verifier.md` in the migration harness), and the branches it found reachable to recorded cases the core matches.
This file is the interface they test through.
Everything it doesn't name is the implementer's choice.

Phase 8.5's contract (`tests/rust_core/request/CONTRACT.md`) still holds: the host, the recordings, the fixtures and the coverage measure.
Today's Python, in the phase venv, is the reference throughout, and no file under `packages/` changes, so every line number below still points at the code the verifier read.

## Names

- A **named entry** is one of the gap-list entries `_reqgaps.py` lists in `ENTRIES`: every entry the verifier called wrong, by module and branch, in one of four groups.
  - **Public**: a branch ordinary public calls reach (the verifier's first list): quantile outputs, `return_embeddings`, a bool target gathered as multiclass, plain link prediction's context entity references, `SDMClient.fit` answered 400 with a categorical-cardinality body, tabular `fit` with holdout keys with and without a key column, a tabular target whose stype is set to `ID`, `result.absent` asked of an unknown name, rankings over a table or column whose name needs backticks, a frame given a multi-index, repeated or empty column names after `Schema.from_data`, a part of a composite key dropped from its frame after `Schema.from_data`, and a `SourceTable` put into `schema.tables` on the tabular route.
  - **Route**: a branch reached by a route the entry did not name, which no host test holds (the second list).
  - **Narrow**: a "Reachable only" entry that names too few routes (the third list), with the number of routes the verifier found.
  - **False**: a branch still untaken whose reason is false or cites the wrong guard (the fourth list and the minor findings), with the clauses of its base reason the verifier showed false.
- The **base** is `fd195c2`; `base.json` holds the gap list and the fixture exclusions as they stood there.
- A **held case** is a stored transcript (`tests/parity/transcripts/`) or a **request fixture case** (Phase 8.5's: a case of a `tests/parity/fixtures/*.json` file whose `operation` starts with `client.`, `decompose.`, `payload.`, `tabular.` or `wire.`) that `scripts/parity/request_fixture_exclusions.json` does not exclude.
  Phase 8.5's checks, which P86.core runs again, hold the core to every held case: payloads, statements, results, warnings, and errors' class, code and message.
- A **host test** is a pytest test under `tests/rust_core/` that runs a route through today's Python and through the core by `scripts/parity/request_host.py`, and requires the two to agree.
  A reason **cites** one by its pytest node id, written in full: `tests/rust_core/<path>.py::<test>` or `tests/rust_core/<path>.py::<test>[<param id>]`.

## Held cases (P86.cases)

`_held.py` replays every stored transcript (`_parity.corpus.check`) and runs every held request fixture case through the Python fixture runner (`tests/parity/_fixture_ops.py`'s `run`), under branch coverage of `sdm_relational_engine`, `sdm_client`, `sdm_connectors` and `predictive_query`, as `scripts/parity/coverage.py` measures them.

- Every transcript replays, and at least the base's 137 transcripts and 553 held fixture cases run.
- Every public branch is taken by the held cases.
- Every named entry the gap list no longer lists is taken by the held cases.
- `request_fixture_exclusions.json` excludes no case the base did not.

A route the corpus vocabularies cannot express (a frame changed in place between calls, `result.absent`, a `SourceTable` put into `schema.tables`) needs the vocabulary extended: the Python fixture runner's `Flow` (`tests/parity/_fixture_ops.py`) or the transcript runner and scenarios (`scripts/parity/_parity/runner.py`, `scenarios.py`, `data.py`), which this phase may change.
They are governed files, so each change is approved like a recording.
Phase 9 switches today's Python to the core, so a host test then compares the core with itself, while a recorded case keeps today's answer; this is why a public branch needs a recorded case.

## The gap list (P86.gaps)

- Every named entry was in the gap list at the base.
- No public branch is listed.
- A named entry still listed has a reason other than its base reason (compared with runs of whitespace as one space), holding none of its false clauses (case-insensitive).
- A route or narrow entry still listed has a reason that starts `Reachable only` and cites at least as many host tests, each a distinct node id, as the verifier found routes to it (one for a route entry).
- Every named entry still listed whose reason starts `Reachable only` cites at least one host test, and each it cites, run by pytest with `_host_plugin.py` (the binding built from this checkout first), passes (setup, call and teardown), makes at least one call into `sdm_core._native` (a function of it, or a method of an object of its types), and takes the entry's branch in today's Python under the same branch coverage.
- `tests/parity/coverage.json` is exactly what `scripts/parity/coverage.py --out` writes, and no public branch is among its `missing_branches`.
- A public branch's module is in `tests/parity/coverage.json`, unless the first whole-module rule matching it in the ledger rules `coverage.py` reads (`--rules`, `SDM_PARITY_LEDGER_RULES` or its default) is `python`, which `coverage.py` does not report.
  Such a branch must instead be taken by a held case (`_held.py`), as P86.cases measures it.

That each reason is true, and that each cited host test holds the route it is cited for, is the independent verifier's check (P86.coverage-gaps).
