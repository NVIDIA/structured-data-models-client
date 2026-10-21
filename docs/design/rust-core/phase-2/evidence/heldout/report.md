| case (leg) | python-ref | python-core | python-old | rust-host | compare lines | entry approved before the run |
|---|---|---|---|---|---|---|
| h2.basic.anchor-before-the-data | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | 0 | |
| h2.basic.anchor-before-the-data@sqlite | refusal INVALID_REQUEST | refusal INVALID_REQUEST | not run there | refusal INVALID_REQUEST | 1 | |
| h2.basic.anchor-before-the-data@postgres | refusal INVALID_REQUEST | refusal INVALID_REQUEST | not run there | refusal INVALID_REQUEST | 37 | |
| h2.basic.default-anchor-multiclass | success | success | success | success | 10 | |
| h2.basic.default-anchor-multiclass@sqlite | success | success | not run there | success | 11 | |
| h2.basic.default-anchor-multiclass@postgres | success | success | not run there | success | 51 | |
| h2.basic.unknown-table-in-the-target | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | 6 | |
| h2.basic.unknown-table-in-the-target@sqlite | refusal INVALID_REQUEST | refusal INVALID_REQUEST | not run there | refusal INVALID_REQUEST | 1 | |
| h2.basic.unknown-table-in-the-target@postgres | refusal INVALID_REQUEST | refusal INVALID_REQUEST | not run there | refusal INVALID_REQUEST | 35 | |
| h2.calls.batched-forecast-refused | refusal UNSUPPORTED_REQUEST | refusal UNSUPPORTED_REQUEST | refusal UNSUPPORTED_REQUEST | refusal UNSUPPORTED_REQUEST | 0 | |
| h2.calls.batched-forecast-refused@sqlite | refusal UNSUPPORTED_REQUEST | refusal UNSUPPORTED_REQUEST | not run there | refusal UNSUPPORTED_REQUEST | 1 | |
| h2.calls.batched-forecast-refused@postgres | refusal UNSUPPORTED_REQUEST | refusal UNSUPPORTED_REQUEST | not run there | refusal UNSUPPORTED_REQUEST | 35 | |
| h2.calls.batched-ranking-refused | refusal UNSUPPORTED_REQUEST | refusal UNSUPPORTED_REQUEST | refusal UNSUPPORTED_REQUEST | refusal UNSUPPORTED_REQUEST | 0 | |
| h2.calls.batched-ranking-refused@sqlite | refusal UNSUPPORTED_REQUEST | refusal UNSUPPORTED_REQUEST | not run there | refusal UNSUPPORTED_REQUEST | 1 | |
| h2.calls.batched-ranking-refused@postgres | refusal UNSUPPORTED_REQUEST | refusal UNSUPPORTED_REQUEST | not run there | refusal UNSUPPORTED_REQUEST | 49 | |
| h2.calls.five-entities-two-a-call | success | success | success | success | 35 | |
| h2.calls.five-entities-two-a-call@sqlite | success | success | not run there | success | 31 | |
| h2.calls.five-entities-two-a-call@postgres | success | success | not run there | success | 71 | |
| h2.calls.last-of-three-calls-fails | refusal NimRequestError | refusal NimRequestError | refusal NimRequestError | refusal INVALID_RESPONSE | 35 | |
| h2.calls.last-of-three-calls-fails@sqlite | refusal NimRequestError | refusal NimRequestError | not run there | refusal INVALID_RESPONSE | 31 | |
| h2.calls.last-of-three-calls-fails@postgres | refusal NimRequestError | refusal NimRequestError | not run there | refusal INVALID_RESPONSE | 71 | |
| h2.calls.only-call-fails | refusal NimRequestError | refusal NimRequestError | refusal NimRequestError | refusal INVALID_RESPONSE | 10 | |
| h2.calls.only-call-fails@sqlite | refusal NimRequestError | refusal NimRequestError | not run there | refusal INVALID_RESPONSE | 11 | |
| h2.calls.only-call-fails@postgres | refusal NimRequestError | refusal NimRequestError | not run there | refusal INVALID_RESPONSE | 51 | |
| h2.calls.rows-per-call-written-as-a-float | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | 0 | |
| h2.calls.rows-per-call-written-as-a-float@sqlite | refusal INVALID_REQUEST | refusal INVALID_REQUEST | not run there | refusal INVALID_REQUEST | 1 | |
| h2.calls.rows-per-call-written-as-a-float@postgres | refusal INVALID_REQUEST | refusal INVALID_REQUEST | not run there | refusal INVALID_REQUEST | 35 | |
| h2.decimal.negative-fractional-key | success | success | success | success | 18 | |
| h2.decimal.negative-fractional-key@sqlite | success | success | not run there | success | 15 | |
| h2.decimal.negative-fractional-key@postgres | success | success | not run there | success | 58 | |
| h2.decimal.negative-sum-9-2 | success | success | success | success | 17 | |
| h2.decimal.negative-sum-9-2@sqlite | refusal INVALID_REQUEST | refusal INVALID_REQUEST | not run there | refusal INVALID_REQUEST | 1 | |
| h2.decimal.negative-sum-9-2@postgres | success | success | not run there | success | 55 | |
| h2.decimal.sum-above-a-finer-constant | success | success | success | success | 15 | |
| h2.decimal.sum-above-a-finer-constant@sqlite | refusal INVALID_REQUEST | refusal INVALID_REQUEST | not run there | refusal INVALID_REQUEST | 1 | |
| h2.decimal.sum-above-a-finer-constant@postgres | success | success | not run there | success | 55 | |
| h2.forecast.second-step-call-fails | refusal NimRequestError | refusal NimRequestError | refusal NimRequestError | refusal INVALID_RESPONSE | 30 | |
| h2.forecast.second-step-call-fails@sqlite | refusal NimRequestError | refusal NimRequestError | not run there | refusal INVALID_RESPONSE | 29 | |
| h2.forecast.second-step-call-fails@postgres | refusal NimRequestError | refusal NimRequestError | not run there | refusal INVALID_RESPONSE | 75 | |
| h2.forecast.weekly-counts-over-text-keys | success | success | success | success | 45 | |
| h2.forecast.weekly-counts-over-text-keys@sqlite | success | success | not run there | success | 43 | |
| h2.forecast.weekly-counts-over-text-keys@postgres | success | success | not run there | success | 95 | |
| h2.forecast.with-candidates | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | 0 | |
| h2.forecast.with-candidates@sqlite | refusal INVALID_REQUEST | refusal INVALID_REQUEST | not run there | refusal INVALID_REQUEST | 1 | |
| h2.forecast.with-candidates@postgres | refusal INVALID_REQUEST | refusal INVALID_REQUEST | not run there | refusal INVALID_REQUEST | 35 | |
| h2.keys.decimal-scales-differ-across-a-link | success | success | success | success | 18 | |
| h2.keys.decimal-scales-differ-across-a-link@sqlite | success | success | not run there | success | 15 | |
| h2.keys.decimal-scales-differ-across-a-link@postgres | success | success | not run there | success | 58 | |
| h2.keys.int64-at-the-top | success | success | success | success | 15 | |
| h2.keys.int64-at-the-top@sqlite | success | success | not run there | success | 15 | |
| h2.keys.int64-at-the-top@postgres | success | success | not run there | success | 55 | |
| h2.keys.negative-integers | success | success | success | success | 15 | |
| h2.keys.negative-integers@sqlite | success | success | not run there | success | 15 | |
| h2.keys.negative-integers@postgres | success | success | not run there | success | 55 | |
| h2.keys.text-differs-by-case | success | success | success | success | 15 | |
| h2.keys.text-differs-by-case@sqlite | success | success | not run there | success | 15 | |
| h2.keys.text-differs-by-case@postgres | success | success | not run there | success | 55 | |
| h2.keys.text-differs-by-trailing-space | success | success | success | success | 15 | |
| h2.keys.text-differs-by-trailing-space@sqlite | success | success | not run there | success | 15 | |
| h2.keys.text-differs-by-trailing-space@postgres | success | success | not run there | success | 55 | |
| h2.keys.utinyint-link-to-bigint-key | success | success | success | success | 15 | |
| h2.keys.utinyint-link-to-bigint-key@sqlite | success | success | not run there | success | 15 | |
| h2.keys.utinyint-link-to-bigint-key@postgres | success | success | not run there | success | 55 | |
| h2.name.double-quote-in-feature | success | success | success | success | 15 | |
| h2.name.double-quote-in-feature@sqlite | success | success | not run there | success | 15 | |
| h2.name.double-quote-in-feature@postgres | success | success | not run there | success | 55 | |
| h2.name.mixed-case-tables-and-columns | success | success | success | success | 15 | |
| h2.name.mixed-case-tables-and-columns@sqlite | success | success | not run there | success | 15 | |
| h2.name.mixed-case-tables-and-columns@postgres | success | success | not run there | success | 53 | |
| h2.name.non-ascii-feature | success | success | success | success | 15 | |
| h2.name.non-ascii-feature@sqlite | success | success | not run there | success | 15 | |
| h2.name.non-ascii-feature@postgres | success | success | not run there | success | 55 | |
| h2.name.space-in-aggregation | success | success | success | success | 15 | |
| h2.name.space-in-aggregation@sqlite | success | success | not run there | success | 15 | |
| h2.name.space-in-aggregation@postgres | success | success | not run there | success | 55 | |
| h2.rank.a-candidate-no-row-holds | success | success | success | success | 60 | |
| h2.rank.a-candidate-no-row-holds@sqlite | success | success | not run there | success | 55 | |
| h2.rank.a-candidate-no-row-holds@postgres | success | success | not run there | success | 121 | |
| h2.rank.second-candidate-call-fails | refusal NimRequestError | refusal NimRequestError | refusal NimRequestError | refusal INVALID_RESPONSE | 60 | |
| h2.rank.second-candidate-call-fails@sqlite | refusal NimRequestError | refusal NimRequestError | not run there | refusal INVALID_RESPONSE | 55 | |
| h2.rank.second-candidate-call-fails@postgres | refusal NimRequestError | refusal NimRequestError | not run there | refusal INVALID_RESPONSE | 121 | |
| h2.rank.text-candidates | success | success | success | success | 40 | |
| h2.rank.text-candidates@sqlite | success | success | not run there | success | 37 | |
| h2.rank.text-candidates@postgres | success | success | not run there | success | 97 | |
| h2.rank.top-k-option-past-the-most | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | 0 | |
| h2.rank.top-k-option-past-the-most@sqlite | refusal INVALID_REQUEST | refusal INVALID_REQUEST | not run there | refusal INVALID_REQUEST | 1 | |
| h2.rank.top-k-option-past-the-most@postgres | refusal INVALID_REQUEST | refusal INVALID_REQUEST | not run there | refusal INVALID_REQUEST | 49 | |
| h2.rank.top-two-of-three | success | success | success | success | 60 | |
| h2.rank.top-two-of-three@sqlite | success | success | not run there | success | 55 | |
| h2.rank.top-two-of-three@postgres | success | success | not run there | success | 121 | |
| h2.rank.without-candidates | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | 0 | |
| h2.rank.without-candidates@sqlite | refusal INVALID_REQUEST | refusal INVALID_REQUEST | not run there | refusal INVALID_REQUEST | 1 | |
| h2.rank.without-candidates@postgres | refusal INVALID_REQUEST | refusal INVALID_REQUEST | not run there | refusal INVALID_REQUEST | 49 | |
| h2.rule.composite-key-named | success | success | success | success | 0 | |
| h2.rule.composite-key-named@sqlite | success | success | not run there | success | 1 | |
| h2.rule.composite-key-named@postgres | success | success | not run there | success | 49 | |
| h2.rule.composite-key-sampled | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | 6 | |
| h2.rule.composite-key-sampled@sqlite | refusal INVALID_REQUEST | refusal INVALID_REQUEST | not run there | refusal INVALID_REQUEST | 3 | |
| h2.rule.composite-key-sampled@postgres | refusal INVALID_REQUEST | refusal INVALID_REQUEST | not run there | refusal INVALID_REQUEST | 51 | |
| h2.rule.signed-zeros-in-a-feature | success | success | success | success | 15 | |
| h2.rule.signed-zeros-in-a-feature@postgres | success | success | not run there | success | 55 | |
| h2.rule.two-key-candidates-tie | success | success | success | success | 0 | |
| h2.rule.two-key-candidates-tie@sqlite | success | success | not run there | success | 1 | |
| h2.rule.two-key-candidates-tie@postgres | success | success | not run there | success | 49 | |
| h2.shape.cycle-through-two-event-tables | success | success | success | success | 28 | |
| h2.shape.cycle-through-two-event-tables@sqlite | success | success | not run there | success | 27 | |
| h2.shape.cycle-through-two-event-tables@postgres | success | success | not run there | success | 93 | |
| h2.shape.entity-links-to-itself | success | success | success | success | 44 | |
| h2.shape.entity-links-to-itself@sqlite | success | success | not run there | success | 41 | |
| h2.shape.entity-links-to-itself@postgres | success | success | not run there | success | 79 | |
| h2.shape.entity-with-a-parent | success | success | success | success | 23 | |
| h2.shape.entity-with-a-parent@sqlite | success | success | not run there | success | 23 | |
| h2.shape.entity-with-a-parent@postgres | success | success | not run there | success | 75 | |
| h2.shape.two-paths-of-different-lengths | success | success | success | success | 30 | |
| h2.shape.two-paths-of-different-lengths@sqlite | success | success | not run there | success | 27 | |
| h2.shape.two-paths-of-different-lengths@postgres | success | success | not run there | success | 79 | |
| h2.target.count-filtered-by-text | success | success | success | success | 15 | |
| h2.target.count-filtered-by-text@sqlite | success | success | not run there | success | 15 | |
| h2.target.count-filtered-by-text@postgres | success | success | not run there | success | 55 | |
| h2.target.static-threshold | success | success | success | success | 10 | |
| h2.target.static-threshold@sqlite | success | success | not run there | success | 11 | |
| h2.target.static-threshold@postgres | success | success | not run there | success | 51 | |
| h2.target.two-conditions-joined-by-and | success | success | success | success | 17 | |
| h2.target.two-conditions-joined-by-and@sqlite | success | success | not run there | success | 17 | |
| h2.target.two-conditions-joined-by-and@postgres | success | success | not run there | success | 57 | |
| h2.types.blob-feature | success | success | success | success | 15 | |
| h2.types.blob-feature@sqlite | success | success | not run there | success | 15 | |
| h2.types.blob-feature@postgres | success | success | not run there | success | 55 | |
| h2.types.float32-feature | success | success | success | success | 15 | |
| h2.types.float32-feature@sqlite | success | success | not run there | success | 15 | |
| h2.types.float32-feature@postgres | success | success | not run there | success | 55 | |
| h2.types.smallint-keys-at-the-top | success | success | success | success | 15 | |
| h2.types.smallint-keys-at-the-top@sqlite | success | success | not run there | success | 15 | |
| h2.types.smallint-keys-at-the-top@postgres | success | success | not run there | success | 55 | |
| h2.types.times-before-1970 | success | success | success | success | 15 | |
| h2.types.times-before-1970@sqlite | success | success | not run there | success | 15 | |
| h2.types.times-before-1970@postgres | success | success | not run there | success | 55 | |
| h2.types.uinteger-keys-past-int32 | success | success | success | success | 15 | |
| h2.types.uinteger-keys-past-int32@sqlite | success | success | not run there | success | 15 | |
| h2.types.uinteger-keys-past-int32@postgres | success | success | not run there | success | 55 | |

cases: 47; legs: 140 (duckdb 47, sqlite 46, postgres 47); cases with a compare line: 47
Fill the last column from the register as it was at the run start only; counts.md holds the outcome counts.
