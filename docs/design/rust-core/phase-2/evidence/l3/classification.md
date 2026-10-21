# L3: every divergence classified (check 6)

Tool: `contract/tools/differential.py run --corrected`, the reference at 43e55f4f against the core at be55838a (the tree of f133c4b8), on DuckDB sources.
KO2-26 asks for about 200 seeds per database; the tool generates DuckDB sources only, so SQLite and Postgres are not measured by L3 (on the packet's "not proven" list).

## Counts

| Corpus | Seed | Sources | agree | approved by an entry the tool names | UNCLASSIFIED by the tool |
|---|---|---|---|---|---|
| generated | 27 | 200 | 166 | 26 (P1B-11) | 8 |
| selfref | 27 | 9 (the corpus's own size) | 4 | 5 (P1B-13) | 0 |

The raw outputs are [generated-27-200.txt](generated-27-200.txt) and [selfref-27-200.txt](selfref-27-200.txt).

## The 8 the tool leaves unclassified, classified here

The tool has no rule for two approved entries, so it prints them UNCLASSIFIED; each was read in the run's own `old.json`, `core.json` and `spec.json`.

| Sources | What differs | Entry | Approved by |
|---|---|---|---|
| g31, g49, g52, g77, g94, g102, g168 | One column of the payload: table `a`'s `d`, a `DECIMAL(38, 18)` feature holding `1.25 + i * 1e-18`. The core sends each value's nearest double (for example `1.2500000000000002`); the reference sends a value one unit in the last place away (`1.2500000000000004`). Every differing value is in that column and is exactly one unit in the last place (2.22e-16) apart; nothing else in the payload, seeds or result differs (the core's own `labels` record has no reference counterpart). | P2B-1 (exact decimals; a decimal feature a float64 cannot hold exactly is sent as its nearest value) | owner, KO2-35 |
| g193 | A text column (`cat`, values `US`, `DE`, `IN`) made the time column by an edit: both refuse; the reference with `INVALID_REQUEST`, the core with `UNSUPPORTED_REQUEST`, rule `text_time` | DT-15 (text outside the core's list of formats refused by name) | owner, E66 |

So every divergence is classified under an approved entry, and none is a new refusal of an input the reference accepts.
The seven payload differences are the more-exact decimal class, not a defect.
On the same generated sources, T2.8c's and T2.13b's trackers reported a different difference: `from_source` read column `d` as categorical on one route and numerical on the other. At this tree the inferred schemas agree on every source, so that difference was closed by a later landing (not bisected; T2.14b's decimal stype fix is the likely one), and L3's first difference on these sources is now the payload's.
Teaching `differential.py` these two rules is carried into Phase 3 as P2-T2.16-5 (no new harness at the phase end, KO2-91).
