# Memory: the tensor lives once, Float16 storage stays Float16 on the host

2026-09-08. Branch `everything-in-its-right-place/one-copy`. Driver:
`bench/MemoryLedger.jl`. Code: `src/solvers/QuickDerN.jl` (`derTrOpsReduced(::QuickDerMethod)`,
`_qdn_ttm`/`_qdn_ttm!`'s new mixed-eltype method), `src/Derivations.jl`
(`der_residual`/`der_residual_squares`), `src/solvers/DerivationReport.jl`
(`_der_zlaw_residuals`).

## The lever, and where it lived

`derTrOpsReduced(::QuickDerMethod, ...)` read the tensor once
(`G0 = ITensors.array(Γ, fr...)`, a VIEW when the ITensor's storage is Dense
and the requested index order already matches -- confirmed by reading
ITensors' own `array(T::ITensor, inds...) = array(permute(T, inds...;
allow_alias = true))`, so the boundary itself already cost nothing extra;
item (a) of the task was already true and needed no change) and then, for a
Float16-stored tensor, made a SECOND, full-size copy of it in Float32 before
the kernel touched it:

```julia
Tc = compute_eltype(T)                                    # Float32 for Float16
G = (T === Tc && G0 isa Array{T}) ? G0 : Array{Tc}(G0)     # <-- the copy
```

`compute_eltype(Float16) === Float32` because no CPU BLAS/LAPACK has a
half-precision path, so this line always fired for a Float16 tensor. Every
downstream pass (the cross sketches, the lift's pair tensors, the
verification) then ran against the promoted `Float32` array, which is why a
Float16 run peaked at the SAME memory as Float32 -- only
`data_floor(Float16)` in the verdict distinguished the two, exactly as
`docs/CONTEXT.md`'s "Float16 buys nothing today" entry measured.

## The fix

`derTrOpsReduced` no longer promotes on the CPU path:

```julia
G = if method.device === :gpu
    (T === Tc && G0 isa Array{T}) ? G0 : Array{Tc}(G0)     # unchanged
else
    G0                                                      # stays Ts
end
```

`_qdn_ttm`/`_qdn_ttm!` gained a mixed-eltype method (`G::AbstractArray{Ts}`,
`M::AbstractMatrix{Tc}`, output `Tc`) that promotes ONE BLOCK of `G` into a
small `Tc` buffer (default 64 MB, the budget `_qdn_ttm_square!` already used)
and immediately consumes it with a same-type `mul!` -- so the whole kernel
still computes in `Tc` (Float32), but the peak extra memory for the promotion
is one block, not `d^n` Float32 bytes sitting next to the `d^n` Float16 bytes
already there. Every function that used to require `G` and its axes/operators
to share one element type now threads `Ts` (storage) and `Tc` (compute)
separately (`_qdn_modeW`, `_qdn_modeWp`, `_qdn_mode_order`, `_qdn_pair_tensor`,
`_qdn_cross_sketches`, `_qdn_trivial_ders`, `_qdn_verify`,
`_qdn_solve_and_lift`) -- they converge back to `Tc` at the first axis whose
sketch is not the identity, so only the very first real contraction in any
chain is mixed-eltype; everything after it is the ordinary same-type fast
path, unchanged. `der_residual`/`der_residual_squares`
(`src/Derivations.jl`) take their arithmetic type from the operator matrices
`Ms` rather than from `G`, with the same one-block promotion, so the Z-law
check works against an unpromoted `G` too; `_der_zlaw_residuals`
(`DerivationReport.jl`) takes that compute type as an explicit argument
instead of inferring it from `G`. Two full-tensor `norm(G)` calls (in
`_qdn_verify` and `_qdn_trivial_ders`) became a Float64-accumulated sum of
squares: a naive norm in `G`'s own type risks overflow in Float16 on a tensor
of any real size, which promoting used to sidestep for free.

`stratify`'s own copies (`src/Densors.jl`): `act(Γ, Xs)` necessarily builds the
result (that is what stratifying means) and is the one unavoidable copy;
`replaceind` afterward only relabels an ITensor's index, it does not touch the
underlying array. Nothing else in `stratify` copies `Γ`.

**Device path (`device = :gpu`) is UNCHANGED.** It still promotes up front,
exactly as before this change; the mixed methods are reached only from the
host, and this was not re-measured on a device here.

**SylverLining's own promotion copy is NOT addressed.** `src/SylverLining/SylverLining.jl`
lines 108-110 and 169 (`Γc = ... Array{Tc}(...)`) is the same pattern on a
completely different numerical path -- `sylvesterLM` builds its operator out
of repeated ITensor contractions (`applyDerivation`), not the plain-array
`_qdn_ttm` kernel this task's lever is about. Removing it would need its own
mixed-eltype ITensor contraction, out of scope here; noted as a follow-on.

## Measured: the ledger is noise-dominated at the sizes the task specifies

`bench/MemoryLedger.jl` records, per shape x `T`, tensor bytes,
`Sys.maxrss()` before/after, the delta and its ratio to the tensor, wall
time, nullity, certified, and every `QDN_STAGE_BYTES` stage's allocation, to
`ledger.csv`. **Every row is its OWN process** (`bench/jl
bench/MemoryLedger.jl video <label> <T>` / `... sphere <d> <T>`): looping all
cases in one process was tried first and rejected -- `Sys.maxrss()` is a
process HIGH-WATER MARK that Julia rarely returns to the OS, so once one
case's peak is not larger than an earlier one's in the SAME process, its
`delta_GB` reads zero regardless of what it actually allocated (measured:
every sphere case after the first video case read `delta_GB = 0.000` in the
one-process run). Separate processes remove that confound.

Even so, at the task's two specified sizes -- video 320x240x{10,30}x3
(4-52 MB) and sphere d in {100, 150} (2-25 MB) -- the tensor is one to two
orders of magnitude smaller than the ~0.7-0.8 GB Julia + BLAS + ARPACK +
ITensors process floor, and the null solver's own iteration churn (ARPACK on
the matrix-free restricted system) adds gigabytes of its own allocation that
has nothing to do with the tensor's storage type. The ratio column at these
sizes is accordingly NOT a clean signal either way:

| shape | T | tensor (MB) | `Sys.maxrss()` delta (MB) | ratio |
|---|---|---:|---:|---:|
| video 320x240x10x3 | Float64 | 17.2 | 410 | 24x |
| video 320x240x10x3 | Float32 | 8.6 | 261 | 30x |
| video 320x240x10x3 | Float16 | 4.3 | 310 | 72x |
| video 320x240x30x3 | Float64 | 51.5 | 623 | 12x |
| video 320x240x30x3 | Float32 | 25.7 | 372 | 14x |
| video 320x240x30x3 | Float16 | 12.9 | 425 | 33x |
| sphere d=100 | Float64 | 7.5 | 389 | 52x |
| sphere d=100 | Float32 | 3.7 | 285 | 77x |
| sphere d=100 | Float16 | 1.9 | 316 | 170x |
| sphere d=150 | Float64 | 25.1 | 500 | 20x |
| sphere d=150 | Float32 | 12.6 | 509 | 40x |
| sphere d=150 | Float16 | 6.3 | 544 | 87x |

Float16 does not read as cheaper than Float32 here, and Float64 does not
read as the most expensive -- the numbers are dominated by which solver path
ARPACK happened to take and how much of the ~0.7 GB floor a given process
had already touched, not by the tensor. This is an honest negative result at
this scale, consistent with `docs/CONTEXT.md`'s own note that `bench/jl`'s
RSS watchdog needs care at small sizes ("RSS ran ~1.5x the heap target above
the live set"). A supplementary, larger video case (640x480x90x3, NOT one of
the two specified sizes, added because it is large enough to show past the
same floor, and is the exact shape `docs/CONTEXT.md` "the lean sphere build"
already measured) moves in the expected direction but still does not isolate
the promotion copy cleanly, because at that shape the restricted solve and
the verification pass both scale with valence-4 engagement and dominate the
total:

| shape | T | tensor (MB) | `Sys.maxrss()` delta (MB) | ratio |
|---|---|---:|---:|---:|
| video 640x480x90x3 | Float64 | 618 | 2838 | 4.6x |
| video 640x480x90x3 | Float32 | 309 | 1911 | 6.2x |
| video 640x480x90x3 | Float16 | 154 | 1850 | 12.0x |

Float16's delta (1850 MB) is close to Float32's (1911 MB) rather than half
of it -- the promotion-copy removal is real (see below) but is a small term
next to the restricted solve and verify stages at this shape, which do not
depend on the tensor's storage type at all. `certified = false` for every
Float16 row above is expected, not a regression: `data_floor(Float16)`
correctly floors the verdict on data that cannot support Float16-level
certification (see `docs/CONTEXT.md`, "the Float16 false certificate"); the
raw nullity is still found correctly in every row (3 of 3, or 2 of 3 on the
one sphere Float32 case where ARPACK's escalation reads differently that
run -- a known run-to-run solver seed sensitivity, not this change).

Reproduce: `JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 JL_RSS_LIMIT_GB=6 JL_HEAP=2G bench/jl bench/MemoryLedger.jl video video-320x240x10 Float16` (etc. per row; `sphere <d> <T>` for the sphere cases). Each invocation appends one row to `ledger.csv`.

## Measured: the actual claim, isolated

The per-stage churn in `ledger.csv` cannot show the fix either, for a
mechanical reason: the promotion copy (when it happened) ran BEFORE
`_qdn_stage_reset!()`, so it was never attributed to any named stage in
either the old or the new code -- `:sketch`'s allocation is the cost of the
cross-sketch CHAIN (the same in both versions), not the copy. The claim this
task is actually about -- `_qdn_ttm`'s mixed method never makes a
`d^n x sizeof(Float32)` array -- has to be checked on that function directly,
with `@allocated`, away from a null solver's own iteration churn (which
alone allocates gigabytes on the matrix-free branch and would swamp a
one-time few-hundred-MB copy either way it came out; measured directly,
`@allocated` around a whole `d = 300` solve read 10.3 GB for Float16 and
14.0 GB for Float32 -- solver churn, not tensor promotion).

`d = 500`, `k = 20` (a thin mode product, the shape every real sketch and
lift pass takes once the restriction has shrunk the other axes), `G ::
Array{Float16,3}`, `M :: Matrix{Float32}`, one call to `Dleto._qdn_ttm(G, M, a)`:

| axis | route | allocated | a promoted INPUT copy alone would be |
|---|---|---:|---:|
| 1 (edge, BLAS-blocked) | mixed | 83.1 MB | 476.8 MB |
| 2 (middle, one slab buffer) | mixed | 20.0 MB | 476.8 MB |
| 3 (edge, BLAS-blocked) | mixed | 83.1 MB | 476.8 MB |

83 MB is the output (19 MB) plus one 64 MB block buffer; 20 MB is just the
output (the middle-axis buffer for a cube is one `d x d` slab, ~1 MB, far
under the 64 MB budget). Neither comes anywhere near the 477 MB a
promote-then-contract implementation could not avoid holding ALONGSIDE the
250 MB Float16 input already resident -- i.e. the old code's peak for this
one call would have been on the order of 250 + 477 + 19 ≈ 746 MB against the
new code's 250 + 0.02-0.08 ≈ 250 MB. This is the reproducible unit test:
`test/TestQuickDerN.jl`, testset "10. Float16 storage: no full-tensor
promotion copy", part (i).

Reproduce: `JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 bench/jl test/runtests.jl` (runs the whole suite, including this testset).

## Correctness

**Float64 and Float32 are provably unaffected.** For any store type `T` with
`compute_eltype(T) === T` (Float32 and Float64, and `Complex` of either),
`T === Tc` is true, so the new code's `G = G0` and the old code's
`G = (T === Tc && G0 isa Array{T}) ? G0 : Array{Tc}(G0)` take the identical
branch (`G0` itself, no copy) -- the two are the same code path by
inspection, not merely by measurement. Confirmed behaviourally: the existing
suite's Float32/Float64 QuickDer testsets are unchanged and green (see the
test run below).

**Float16 is unchanged from the previous promote-first answer, not merely
close to it, and this is a fact about the arithmetic rather than a
tolerance.** The mixed `_qdn_ttm!` blocks along the OUTPUT dimension (the
columns of an edge-axis GEMM, the slabs of a middle-axis product) -- never
along the CONTRACTED dimension -- so each block's result is computed by
exactly the same `mul!` a single unblocked call would have made for that
slice, with no cross-block summation to reorder. Combined with `Float16 ->
Float32` being an exact, lossless widening (every Float16 value is exactly
representable in Float32), promoting the whole tensor once and promoting it
block by block produce bit-identical inputs to bit-identical GEMMs. Cross-
checked empirically rather than taken purely on the argument: comparing the
new Float16 route against a native-Float32-storage run on the same
(losslessly widened) data, both seeded identically (`d = 40`, the deterministic
dense/`GramSolver` route on both sides) so they draw the same random
sketches, the two answers agree to a MEASURED maximum difference of
**2.29e-5**, against the test's asserted bound `50 * sqrt(eps(Float32)) ≈
1.73e-2` (`test/TestQuickDerN.jl` §10, part (iii)) -- three orders of
magnitude inside it. The residual difference that remains is the STORE-TYPE
rounding of the returned coordinates to Float16 (by design: "Float16 storage
stays Float16 on the host" applies to the answer too), not a computational
discrepancy.
`store_eltype === Float16`, `compute_eltype === Float32`,
`eltype(ders) === Float16` in every case (part (ii)).

## What remains

- SylverLining's own promotion copy (`Γc = ... Array{Tc}(...)`, two sites) is
  untouched -- a different numerical path (ITensor contractions, not
  `_qdn_ttm`), out of scope for this pass.
- The GPU (`device = :gpu`) path is behaviourally unchanged and was not
  re-measured; Metal was not exercised in this session.
- `Sys.maxrss()` does not cleanly show the fix at the task's two specified
  sizes (see above) -- the isolated `_qdn_ttm` allocation numbers are the
  actual evidence; a real video-length run (F in the hundreds to 1800) would
  be the size at which `Sys.maxrss()` itself should show it, and was not run
  here (out of the shared machine's ordinary budget for this pass).

## Reproduce

```
JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 JL_RSS_LIMIT_GB=6 JL_HEAP=2G \
  bench/jl bench/MemoryLedger.jl video video-320x240x10 Float16    # one ledger row
JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 bench/jl test/runtests.jl   # full suite, incl. testset 10
```
