# Reviewer: standing rules (shared rules, independent starting points)

The orchestrator names the package, approved design, exact `BASE..CANDIDATE` range, round, already known findings and report path.
Every brief carries `Candidate: <full SHA>`.
The current [handbook](../README.md) and latest owner instructions supersede older process wording.

**Reviewer assignments**

Both reviewers use `gpt-6.1-sol` with high reasoning, fresh contexts and the same candidate SHA.
The broad reviewer examines the whole change, with priority on design, contracts, duplication, boundaries and correctness.
The focused reviewer starts with the package's concrete failure risks, relevant reference behavior and existing evidence.
Neither assignment limits what that reviewer may inspect or flag.
The focused assignment is not a request for a new compatibility audit or test campaign.
Both receive the approved requirements and decisions without the implementer's conclusions or a desired verdict.
Keep their initial reports independent; compare findings only after both have written them.
The test-execution and live-work restrictions below apply equally to both reviewers.

**How to review**

- Review the fixed SHA in a throwaway worktree or with `git show`, never the changing implementation worktree.
- Start immediately when the candidate is pushed, with no tests, lint, stamp or CI prerequisite.
  Package tests and cheap pre-push checks run only after both reviewers approve the same SHA.
  A failed check leads to a fix, both reviewers' renewed approval and then another check run.
- Round one covers the whole range; later rounds cover the fixes and verify resolution of earlier findings.
  Read the code instead of routinely rerunning suites or probes.
  If code suggests a specific defect, investigate with the smallest focused check and explain what prompted it.
  Never run package, broad or full CI suites as a routine review step.
- Verify that tests were committed before implementation.
  Implementers may write their own tests under the current rule; independent acceptance tests and reference observations must not be weakened to hide defects.
- Inspect design contracts, DRY, modularity, separation of concerns, interfaces and cross-language plain data.
  The core owns semantics, SQL and the chosen diagnostic message; hosts convert native data, perform I/O and render that message.
  Flag unnecessary fields or layers and misplaced rules according to realistic impact.
- List every measured behavior difference and its register entry.
  A better error or support for an input the old client fails on can be approved as an improvement with `APPROVE-<entry id>` and reasons.
  Accepted-answer changes, new refusals of supported input and comparison relaxations require the owner's actual authorization.
- Do not wait for CI or demand an exact-candidate CI run for a package landing.
  Full CI belongs to the nightly phase head and phase end.
  Report a real failure already observed, without launching broad CI to look for one.
- Keep any targeted probe bounded with stdin closed and use an isolated target directory.
  Change nothing in the implementation worktree or on GitHub.
  Respect command and live-access restrictions; do not wait on an unattended approval prompt.

**Severity**

Only P0 and P1 block under the owner's October 23 instruction, refined October 24.
P0/P1 cover serious defects and normal-use regressions: wrong answers, crashes, lost support, worse errors, invalid evidence accepted as valid, and authorization claims wider than the owner's words.
Rare sequences can be P2; minor issues are P3.
Do not relabel a realistic failure as rare merely to obtain approval.
Address useful nonblocking findings where practical and name the rest for carry-forward.
A cluster of repeated findings may require design reassessment rather than more patches.

The write guard protects against accidental writes from actual client/loader behavior.
Hand-crafted SQL bypasses outside plausible product behavior are not automatic P0 findings.
A clearer error or more supported input is a registered improvement, not a blocker merely because it differs from the reference.

**The report** (`land.py` reads its first two lines)

- Line 1: exactly `APPROVE` or `CHANGES REQUIRED`.
- Line 2: the full candidate SHA.
- Then `differences:`, listing each changed answer, refusal or diagnostic as better or worse, with its register entry; use `none` when appropriate.
- Then findings tagged P0, P1, P2 or P3.
  Only unresolved P0/P1 findings require `CHANGES REQUIRED`.
- Each finding identifies `file:line`, the concrete effect or reproduction and a concise suggested fix.
- Finish with what was verified, any focused check used and any limitation.
  Do not claim tests ran when the review was code-only.
