# T2.8c: multi-call `predict`

Lane f's worktree, on the phase head after T2.19's landing (dfdd532a).
`main`'s 1fc5a686 (in the reference at 6298913c) gathers a request of more entities than one call scores as several calls and predicts them as one; this package writes it in the core (design, choice 6), never in the host.
The acceptance cases are the test author's (`T2.8c tests:` commits on `t28c-tests`).

## What changed

In progress.
