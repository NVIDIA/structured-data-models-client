# Phase 3 decisions

The owner's decisions for Phase 3, newest last.
As in Phase 2 (KO2-14), this file records only the owner's own decisions; the orchestrator's and implementers' choices go in [design.md](design.md) and each package's tracker.
Phase 1's and Phase 2's rows ([phase-1/decisions.md](../phase-1/decisions.md), [phase-2/decisions.md](../phase-2/decisions.md)) stay in force unless a row here changes them; cite them by ID, never copy them.
A status `approved by owner <date>` records an answer the owner wrote; an agent never writes that status for itself (plan section 2, decision 12).
Each Decision cell says no more than the owner's words, which the Owner's words cell quotes exactly, with the question as it was asked; a row that would need more is an open question in design.md instead.
A row is at most 1,000 characters.
Row IDs start `KO3-`: a capital letter, as `capture_check.py` and `coverage.py` read IDs, and a `3` so they never collide with earlier phases' rows.

**Reference approvals.**
A row approves a new reference commit exactly as Phase 2's do (phase-2/decisions.md, "Reference approvals"); Phase 3 has none yet.

## Before the design: package size (2026-10-20)

Source: the owner's answer on 2026-10-20 to the orchestrator's package-granularity question, recorded in the orchestrator's state file `phase-2/owner-answers-2026-10-17.md`, section "2026-10-20 (UTC): package granularity".
The question, as asked: "1 owner decision rows that approve recorded differences ride with the package that needs them (reference approvals and comparison loosenings go in the landing just ahead, never alone); 2 CI-config changes ride with the package that needs them; 3 Phases 3-5 at about 7-10 landings each, integer IDs (T3.1, T3.2), groups capped at about 10 hours of implementation."

| # | Decision | Owner's words | Status |
|---|---|---|---|
| KO3-1 | Owner decision rows that approve recorded differences ride with the package that needs them; reference approvals and comparison loosenings go in the landing just ahead, never alone; CI-config changes ride with the package that needs them; Phases 3 to 5 at about 7 to 10 landings each, with integer IDs (T3.1, T3.2), groups capped at about 10 hours of implementation | "yes to all" | approved by owner 2026-10-20 |

## The root-cause analysis's decisions (2026-10-21, evening)

Source: the owner's answers to the nine decisions at the end of Phase 2's root-cause analysis (`$SDM_ORCH_STATE/phase-3/rca/RCA.md`, "Decisions only the owner can make"), recorded in `$SDM_ORCH_STATE/phase-3/owner-answers-2026-10-21.md`.
Each question is summarized in the Decision cell as it was asked; the RCA file holds its full text and recommendation.

| # | Decision | Owner's words | Status |
|---|---|---|---|
| KO3-2 | RCA decision 1, one severity bar for both reviewers (any measured difference from the reference on a public call, any red gate or CI job the change caused, and any decision row wider than the owner's words is at least P2): not decided; the owner asked a question back | "1. Severity: Has this really been a problem?" | open: answered in design.md, owner question G2 |
| KO3-3 | RCA decision 2, stop and redesign: when the same family of finding blocks two rounds in a row, or a fix's own change causes a new blocking finding twice in a row, stop patching; without a large redesign session for a simple missed case | "2. Yes to Stop and redesign, but sometimes you might've just missed some dumb case so we shouldn't need a large redesign sesion for that..." | approved by owner 2026-10-21 |
| KO3-4 | RCA decision 3, speed infrastructure (the stacked queue, the CI fix, environment and disk hygiene, the command hook) is schedule work built in Phase 3's first package, not "machinery" | "3. Yes please." | approved by owner 2026-10-21 |
| KO3-5 | RCA decision 4, docs-only fix rounds: the reviewer who raised the finding confirms its fix; any round writing decision rows or register entries keeps Claude | "4. Sure..." | approved by owner 2026-10-21 |
| KO3-6 | RCA decision 5, Phase 3's budget (at most 10 landings and a 16-hour planned critical path): approved, then replaced the same evening by KO3-12, which rejects hard budgets | "5. Yes" | superseded by KO3-12 |
| KO3-7 | RCA decision 6, what "match the reference" covers: answers, refusal kinds and user-visible messages; statement counts, warning repeat counts, tie order among equal scores and hash order only when a user can see them, otherwise one standing register class the owner approves once | "6. Yes" | approved by owner 2026-10-21 |
| KO3-8 | RCA decision 7, a blocking hook for the banned shell commands for every Claude agent; Codex keeps the watcher | "7. Yes" | approved by owner 2026-10-21 |
| KO3-9 | RCA decision 8, overnight delegation for Phase 3, on condition each decision is written plainly when made and put to the owner in the morning, before the final candidate | "8. Yes" | approved by owner 2026-10-21 |
| KO3-10 | RCA decision 9, process lessons live in `learnings.md` with the product lessons | "9. Yes" | approved by owner 2026-10-21 |
| KO3-11 | The lessons are enforced by mechanisms, not by updating a Markdown file | "The problem is all these "lessons" aren't being enforced, you just seem to be updating some random md file and not officially enforcing them. How can you enforce? Think carefully. Could be templates, scripts, github actions, pre-commit hooks, etc. You have a lot of lessons learned / RCAs found but they don't translate to your decisions above" | the owner's instruction, 2026-10-21 |

## The five Phase 3 process changes (2026-10-21, evening)

Source: the owner's answers to the orchestrator's first proposal of five process changes and then to the revised five, recorded in `$SDM_ORCH_STATE/phase-3/owner-answers-2026-10-21.md`; T3.0 built the revised five.

| # | Decision | Owner's words | Status |
|---|---|---|---|
| KO3-12 | No hard budgets of landings or hours; change the process itself | "1. The problem with setting "budgets" like the 10 landings and 16-hour critical path is it can lead to shoddy work. Is there a way to change the process itself? Imagine 16 hours go by and it just throws its hands in the air" | approved by owner 2026-10-21 (with KO3-17) |
| KO3-13 | Stack all the time, with CI mandatory every N packages, for example 4 | "2. Yes we should be stacking all the time. But maybe this can be limited to every N packages (e.g. 4 packages) before CI is mandatory" | approved by owner 2026-10-21 (with KO3-17) |
| KO3-14 | Fix the slow checking-tool tests in CI, with a warning and fix system | "3. Yes please. What even are these tool tests? and yes we need some warning + fix system." | approved by owner 2026-10-21 (with KO3-17) |
| KO3-15 | Reviewers do not live-test; checking against the old client is the implementer's, the tests' and CI's | "4. Do we think this "compare with the old client" will add too many slowdowns in the review phases? The reviewers shouldn't really need to live test, that's the point of CI or the implementer and the tests we write." | approved by owner 2026-10-21 (with KO3-17) |
| KO3-16 | Test environments must fit the machine's cores, memory and disk | "5. This "own test environment" thing will blow up ssd and memory, remember that we can't run too many jobs in parallel due to limited cpu cores, memory, and disk." | approved by owner 2026-10-21 (with KO3-17) |
| KO3-17 | The revised five changes are built (T3.0), then Phase 3's design begins | "Yes, do these. After this is done, we will begin the phase 3 design doc" | approved by owner 2026-10-21 |

## Kickoff answers

None yet: the questions are in [design.md](design.md), "Owner questions", each with a recommendation.
