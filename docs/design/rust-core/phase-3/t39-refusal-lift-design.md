# T3.9: owner-admitted sampling values

KO3-44 requires accepting the three inputs that Phase2 refused while the frozen client answered.
KO3-36 concerns the separate KO2-94 feature list and does not defer this work.
Public reproductions against genuine43e confirm that NaN lag_timesteps, negative num_neighbors and SQL-expression num_neighbors succeed on DuckDB and SQLite.
Current production refuses all three before materialization.

## Boundaries

Python and Java encode option values, without deciding their sampling meaning.
NumPy NaNs, including extended-precision scalar NaNs, are encoded as the same NaN token as Python floats only in the option attachment.
A dedicated option-attachment decoder reuses the wire codec's lossless nonfinite-token mechanism.
Ordinary message, attachment and request JSON remain strict, and strings such as "NaN" remain strings.
Rust admits only the nonfinite value accepted by the old client: a NaN lag_timesteps, meaning no extra lag.
Existing errors for other literal nonfinite option values remain named refusals.
Finite lossless JSON numbers outside double range retain their existing option rules; transport admission does not convert them to doubles.
The endpoint response decoder's encoding and surrogate rules do not apply to options.

The shared data contract carries an explicit neighbor limit.
It preserves the full unsigned64 range, signed negative numbers, and caller-authored SQL-expression text as distinct variants.
Positive numeric limits retain their existing encoding and behavior.
Only a numeric zero stops the walk; textual zero continues it as before.
Core and request code carry opaque provider-expression text separately from numbers, without interpreting a language or naming a dialect.
The data contract documents SQL-expression trust and retains the explicit sql_expression wire tag; only SQL providers render its text.
SQL dialects retain their original predicate or LIMIT placement and precedence, including their different answers to negative bounds.
An SQL expression is caller-authored query code under the existing derived-expression trust boundary, never an identifier or metadata value.
No invented SQL parser or new accepted-expression subset is introduced.

Capability-driven warehouse lookback remains numeric and preserves the old error order where text cannot participate in arithmetic.
Negative caller text retains its SQL-expression identity, distinct from the same numeric value; the existing P2I-1 policy for positive numeric text remains unchanged.
Signed negative lookback is preserved rather than converted to a positive or zero window.
The non-SQL Arrow and test providers use an explicit row-position bound: a negative numeric bound keeps no rows after ordinary lookup and scan work.
They refuse caller SQL expressions because they have no SQL execution language.
SQL providers retain their native negative LIMIT behavior, including SQLite’s unlimited result; no core-wide conversion discards the signed bound.
Unsigned limits continue through the existing generic fallback.

## Evidence and compatibility

Regression tests are written before implementation and exercise the public materialize API against the frozen client's payloads.
Negative limits and expressions require dialect-level evidence because DuckDB predicates and SQLite LIMIT do not have the same semantics.
The invalid-text control remains a refusal where the old SQL cannot execute it.
No warehouse acceptance cases, source datasets, seeds, anchors or model settings are added or changed.
Immutable contract29 remains unchanged; working30 and bindings are regenerated canonically when their schema changes.
Two independent code reviews precede candidate checks.
