# Offline diagram renderer

Mermaid 11.15.0 is vendored unchanged from https://cdn.jsdelivr.net/npm/mermaid@11.15.0/dist/mermaid.min.js.
SHA-256: `70137e77bb273bb2ef972b86e8b0400cca8be53cb25bfc45911a186dc98665de`.
Both Python and Java package these same resources.
`MERMAID_LICENSE` is the upstream license; `THIRD-PARTY-NOTICES-MERMAID.txt` retains the complete original dependency attribution and license texts.
To upgrade, replace the upstream bundle/license together, run `python scripts/mermaid_notices.py --refresh`, then `--write`, then review the digest and rendering tests.
Use `--check` to verify generated attribution without network access.
