# Package brief template

Every package brief is written from this template before its test author starts (learnings.md, lesson 16).
`scripts/rust_core/brief_check.py <file>` names every `##` heading below a brief leaves out (T3.1); the orchestrator runs it before each launch, and T3.2's tooling train makes `lane.py start --brief <file>` refuse such a brief.
A heading may say "none" with a reason; it may not be left out.
The standing rules of [implementer.md](implementer.md) and [test-author.md](test-author.md) are appended after it, as today.

```markdown
# <ID>: <one line on what users get>

## Scope and exits
What the package delivers, its design row, and each exit with the command that shows it.

## Threat model
Honest mistakes, unless the owner says otherwise. Hardening internal tools against forgery is P3 at most (Phase 2 decision KO2-52).

## Reference code to port
Each rule the package implements, with the old client's file and line at the reference commit. Port it; do not approximate it.

## Host or core
For each rule: can the core compute it from plain data, without the host language's runtime? If not, the host computes it and the contract carries the result (lesson 1).

## Input axes
The value classes, every reading the reference makes of a value, the databases and warehouse legs, and the CI cells the package's cases must cover (lesson 6), including the key-type pairs for any key it binds (lesson 7).

## When you change X, also update Y
The list for this package: lifting a refusal updates the census and the differential's classes; a new case needs its reference lines; a new register entry passes `register.py validate`; a contract change bumps the version once (lesson 11).

## Readers of new or changed files
Every reader of every file the package adds or changes, with the command that runs it, the landed judging copy included.

## Judging paths
Every path of `scripts/rust_core/judging-paths.txt` the package would change: none, or the change the orchestrator moves into the tooling train.

## Owner decisions
Each decision row the package relies on, with the owner's words verbatim and the question as asked.

## Owner questions
Any question still open, answered before the first review round, or "none".
```
