# Native Arrow Decimal constructor

This is the source composition for built-in Arrow Decimal frame constructors under working internal contract 35.
Dependency reconciliation, canonical schema generation, two independent exact code approvals and actual package execution remain required before acceptance.
The unchanged frozen public source produced 97 constructor/edge cases in each of CP312/pandas3 and CP310/pandas2, with identical record arrays.
Its accepted native domain includes scale outside i8 and scale greater than precision.
Those observations cover128/256; a separate frozen packet is pending for native32/64 and timestamp-candidate mutation before the full family can be accepted.

## Host and core boundary

The Python host exports exact built-in native Decimal arrays as typed scale-zero views at their original32/64/128/256 width over their existing buffers.
It retains original precision and signed-i32 logical scale in a neutral NativeDecimal declaration bound to the same field ordinal/name in the same FrameInput attachment.
Chunks, offsets, validity and native frame ownership remain native facts.
No per-cell native tree, Binary interpretation, scalar coercion or schema answer is supplied by the host.

Shared physical Arrow admission validates and detaches the representable mantissa buffers.
Its existing32/64-to128 widening preserves those native mantissas exactly.
The core constructor validates logical declarations against unique field identity and exact admitted width/precision/scale-zero storage, and bounds32/64 mantissas by their declared original signed width.
This validation also accepts owned canonical128 rows when their original32/64 descriptors are rebound after a Table result.
The borrowed `sdm-schema::frames::Columns` input feeds the existing nomination, adoption, time, override, assembly and sample algorithms.
Existing entrypoints without logical declarations retain their signatures for other consumers.

Constructor equality uses exact mantissas within each fixed-type column.
SQL/provider scaled-decimal float inference remains its own existing policy.
Declared positive native scale infers float64 even for whole-valued, empty and all-null Arrow columns.
For nonpositive scale, native precision at most 18 preserves the frozen metadata shortcut, including values whose logical negative-scale expansion exceeds int64.
Wider precision uses bounded exact integer range comparison without expanding powers or strings by logical scale.

## Presentation and owned data

Only core-selected repeated-key scalars are sent for native warning/error presentation.
Their ordinary closed Decimal scalar carries the unscaled digits, and a parallel typed i32-scale list preserves the original logical values.
The Python presenter reconstructs Decimal from its digit tuple without context arithmetic, then uses actual native scalar or mixed-row presentation at the reached stage.
Each new constructor stage captures fresh native descriptors after earlier warning effects while keeping already inferred meanings fixed.

Legacy Table describe returns detached `frame_rows` with an optional `frame_decimals` bytes attachment for logical declarations.
Both are taken before the run is dropped and rebound under fresh attachment ids.
The control message retains fixed-size handles, and the descriptor bytes remain proportional to columns rather than rows.
SQL and final Schema results return neither owned frame attachment.

## Remaining delivery obligations

The partial frame materializer still lacks logical Decimal value consumption and must not treat constructor mantissas as model values.
The existing whole Decimal family remains outside that unfinished development route until materialization, wire/session conversion and checkpoint propagation consume the original descriptor.
Default public routing is unchanged.
Frame setters retain the existing frame-edit route pending the broader frame-edit migration.
Mixed timestamp-candidate mutation into Decimal needs actual native conversion/read evidence before full mutation parity can be claimed.
This constructor composition does not complete T4.2 or Phase 4.
