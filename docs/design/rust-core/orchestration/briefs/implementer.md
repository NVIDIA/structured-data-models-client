# Implementer: standing rules (appended to every package brief)

The package brief gives scope, exits, dependencies and approved design.
The current [handbook](../README.md) and latest owner instructions take precedence over older briefs or scripts.

**Workflow**

- Work in the package's assigned lane or topic worktree and preserve any unfinished changes already there.
  Push only the assigned package branch; do not move protected branches.
- Every commit subject starts with the package ID: `<ID>: ...`, `<ID> tests: ...`, `<ID> fix <n>: ...` or `<ID> docs: ...`.
  Use DCO sign-off and no agent co-author.
- Write and commit the package's tests before implementation.
  A separate test author is optional; preserve already independent acceptance tests and authentic recorded values.
  Use public calls and both routes where applicable.
- Push the candidate and report its exact SHA immediately so both independent reviewers can start.
  The two reviewers use Sol 6.1 High: one broad design/correctness review and one starting from the package's specific risks.
  Both may flag any defect; their focus does not limit review coverage.
  Do not run package tests, lint or cheap pre-push checks before both reviewers approve the code.
  Do not run full suites, workspace tests, affected suites or full CI within a topic package.
- After both reviewers approve the same SHA, coordinate the cheap pre-push checks and `handoff.py run --package <ID> --python <lane python>` as one bounded check of that clean SHA.
  Do not duplicate overlapping commands or accept the old pre-push script's wider tool self-tests and broad clippy scope.
  The handoff scope is lint and the package's own tests, nothing else.
  If a check fails, fix it, obtain both approvals on the new SHA, then run the checks again.
  An outdated automatic hook must be reconciled by the orchestrator, not bypassed with `--no-verify`.
- A product fix commit carries `Rule: <the old client's rule it ports, one sentence>` where the commit-message tooling requires it.
- Report the candidate, review/check state, deviations, measured differences and anything requiring the owner.
  End your turn rather than waiting on CI; the orchestrator handles nightly failures and continuation.

**Design and behavior**

- Keep the language-neutral core responsible for query semantics, SQL construction, policy and the chosen error/warning message.
  Hosts handle native conversions, drivers, authentication, transport and rendering the core's chosen structured message.
  Do not move SQL or business rules into Python/Java to avoid a contract or core change.
- Every measured difference from the reference gets a register entry with its exact scope.
  Never approve your own behavior change or describe a recommendation as the owner's approval.
  Better errors and more support may be reviewer-approved improvements under the current policy; accepted-answer changes and comparison relaxations still need the owner.
- Check realistic name resolution, views, namespaces and capabilities against the actual source behavior.
  Do not replace evidence with guesses or weaken a test to conceal a mismatch.
- After repeated findings on one theme, report the design issue rather than adding another local patch.
  The orchestrator reassesses after two rounds on a theme or by round four.
- Design review includes DRY, crate/module ownership, separation of concerns, interfaces and cross-language plain data.

**Tests and judging tools**

- Write regression tests for the actual public failure and check the tests-first history.
  Execute the package checks after both approvals, including any required baseline failure demonstration within that bounded check.
- Cover parser nesting and malformed inputs without crashes where the package introduces a reader.
  A new judging rule needs a seeded fault proving missing, empty or invalid evidence fails.
  Do not launch generated differential campaigns before review as a routine package step.
- Judging-path changes land with the product package that needs them; name them so both reviewers inspect them.
  The separate tooling train is gone.
  A guard or gate change must also be correct in its post-landing state.
- Regenerate generated files with their generators and refresh required snapshots for observable changes.
  Never hand-edit reference recordings or verified outputs.

**Environment and coordination**

- Use the lane's isolated `cells.sh` environment, current phase setting and assigned Cargo target directory.
  Concurrent worktrees must not share incremental build output.
- Heavy jobs go through the existing `heavy` wrapper.
  Do not create new resource guards or raise concurrency limits without authorization.
- Never wait unattended on an approval prompt.
  Report the specific command promptly to the orchestrator, which can coordinate an authorized alternative.
- Never open held-out data, touch another lane's worktree, expose credentials or private infrastructure, or send outbound messages without authorization.
- Keep probes bounded with stdin closed and noninteractive Git/GitHub settings.
  Respect the command guard and authorized cleanup scope; do not assume another runtime inherits Claude's hooks.
- Keep shared-document edits confined to assigned entries and existing ID blocks.
  Add prose only when needed or requested, keep explanations short, and put each full sentence on its own line in long Markdown.
