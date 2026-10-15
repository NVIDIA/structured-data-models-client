| case | python-ref | python-core | python-old | rust-host | compare lines | entry approved before the run |
|---|---|---|---|---|---|---|
| h11.decimal.binary-threshold-10-2 | success | success | success | success | 15 | |
| h11.decimal.feature-18-18-null-heavy | success | success | success | success | 10 | |
| h11.decimal.key-38-0-large | refusal INTERNAL_ERROR | success | refusal INTERNAL_ERROR | success | 2 | |
| h11.decimal.target-38-37 | success | success | success | success | 15 | |
| h11.decimal.target-sum-past-precision | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | 5 | |
| h11.edit.clear-time-column-window | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | 0 | |
| h11.edit.drop-target-column | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | 0 | |
| h11.edit.set-second-time-column | success | success | success | success | 15 | |
| h11.edit.stype-categorical-int-target | success | success | success | success | 10 | |
| h11.empty.entity-table | refusal RuntimeError | refusal INVALID_REQUEST | refusal RuntimeError | refusal INVALID_REQUEST | 0 | |
| h11.empty.single-entity | success | success | success | success | 15 | |
| h11.key.composite-primary-key | refusal INVALID_REQUEST | refusal UNSUPPORTED_REQUEST | refusal INVALID_REQUEST | refusal UNSUPPORTED_REQUEST | 0 | |
| h11.key.duplicate-primary-values | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | 0 | |
| h11.key.explicit-tables-and-link | refusal AttributeError | refusal AttributeError | refusal AttributeError | success | 1 | |
| h11.key.int64-near-max | success | success | success | success | 15 | |
| h11.key.negative-multiclass | success | success | success | success | 10 | |
| h11.key.uint32-binary | success | success | success | success | 15 | |
| h11.key.varchar-leading-zeros | success | success | success | success | 15 | |
| h11.key.varchar-unicode-binary | success | success | success | success | 15 | |
| h11.key.width-mismatch | success | success | success | success | 15 | |
| h11.nested.json-text | success | success | success | success | 15 | |
| h11.nested.list-of-list | refusal ValueError | refusal UNSUPPORTED_REQUEST | refusal ValueError | refusal UNSUPPORTED_REQUEST | 0 | |
| h11.nested.list-on-fact-table | success | success | success | success | 15 | |
| h11.nested.list-varchar-null-elements | success | success | success | success | 15 | |
| h11.nested.struct-on-fact-table | refusal RuntimeError | refusal UNSUPPORTED_REQUEST | refusal RuntimeError | refusal UNSUPPORTED_REQUEST | 0 | |
| h11.null.all-foreign-keys | success | success | success | success | 15 | |
| h11.null.nan-and-infinity-target | refusal INVALID_REQUEST | refusal UNSUPPORTED_REQUEST | refusal INVALID_REQUEST | refusal UNSUPPORTED_REQUEST | 5 | |
| h11.null.target-mostly-null-binary | success | success | success | success | 15 | |
| h11.null.time-column-all-null | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | 0 | |
| h11.option.inference-config | success | success | success | success | 15 | |
| h11.task.binary-count-equals | success | success | success | success | 15 | |
| h11.task.forecast-sum | success | success | success | success | 27 | |
| h11.task.multiclass-text-with-null-target | success | success | success | success | 10 | |
| h11.task.ranking-without-candidates | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal INVALID_REQUEST | refusal UNSUPPORTED_REQUEST | 0 | |
| h11.time.anchor-leap-day-binary | refusal UNSUPPORTED_REQUEST | refusal UNSUPPORTED_REQUEST | refusal UNSUPPORTED_REQUEST | refusal UNSUPPORTED_REQUEST | 15 | |
| h11.time.date-month-ends | success | success | success | success | 15 | |
| h11.time.dst-fall-back-binary | refusal UNSUPPORTED_REQUEST | refusal UNSUPPORTED_REQUEST | refusal UNSUPPORTED_REQUEST | refusal UNSUPPORTED_REQUEST | 15 | |
| h11.time.dst-spring-forward-hours | success | success | success | success | 15 | |
| h11.time.month-window-from-month-end | success | success | success | success | 15 | |
| h11.time.pre-epoch | success | success | success | success | 15 | |
| h11.time.text-iso-offset | success | success | success | success | 15 | |
| h11.time.text-iso-zulu-fraction | success | success | success | success | 15 | |

cases: 42; differing from the reference (any compare line): 33
panics or INTERNAL_ERROR or runner failures (H11: any = no-go): 0 []
new refusals of an input the reference accepts (H11: any = no-go unless owner-approved before the run): 0 []
H11: more than 4 cases differing WITHOUT an entry approved before the run = no-go.
Fill the last column from register-at-run-start.yaml only (entries approved before the run);
the count of differing cases with that column empty is the number H11 judges.
