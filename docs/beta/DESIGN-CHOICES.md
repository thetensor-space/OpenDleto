# Design choices on `beta`, and whether they hold up

A design review of the `beta` branch (tag `v1.5-beta-2026-09-04`), written 2026-09-08. Each
entry names a decision that was actually taken in the code, the reasoning on record (mostly
`docs/CONTEXT.md` and `docs/design/`), and a verdict. Verdicts: **keep**, **keep but finish**,
**revisit**. Line-level findings that back these verdicts are in [REVIEW.md](REVIEW.md).

The frame for the whole assessment: the package had two authors with opposing styles, an
incremental problem-solver and an over-designer, and the September work was asked to strike a
balance and leave room for extension. Read the verdicts with that in mind. Most of what beta
did is right; the recurring weakness is that a good idea was implemented in one place and not
carried through the rest of the package.

---

## 1. One required method per derivation solver: `derTrOpsReduced`

**Decision.** Every derivation method is a struct `<: DerivationMethod` implementing exactly
`derTrOpsReduced(method, Ω, P, Γ; tol, nd, ...) -> (rΩ, expand_map, coords)`. The user-facing
tiers `der`, `derReduced`, `derTrOps` are written once on top of it.

**Reasoning.** The smallest seam that lets five methods (SylverLining, FastDer3Valent,
QuickSylver, QuickDer, AutoDer) coexist; the reduction to engaged axes is where the methods
genuinely differ.

**Verdict: keep, but finish.** The seam is real and all five methods use it. But the contract
around it has drifted into four dialects: `tol` is `Real` in three methods and `Float64` in
QuickSylver and in `der` itself; `nd > 0` means a policy in QuickDer and SylverLining, a
warning-and-refusal after restriction in QuickDer, and a silent column truncation in
FastDer3Valent and QuickSylver; `rΩ` is the engagement-reduced space for SylverLining and `Ω`
itself for the other three; `coords` come back in the storage type from two methods and in
Float64 from the others; `return_diagnostics` is honoured by three and swallowed by two. The
abstract type's docstring still says a method "should implement der and den". Write the
contract once on the abstract type (see REVIEW.md for a proposed form), add a `supports`
trait so AutoDer stops using `try/catch` as applicability, and turn `get_derivation_method`'s
`if/elseif` into a registry like the null solvers have.

## 2. QuickDer restricts by a random orthogonal sketch, not a corner

**Decision.** For any valence, sketch each output axis with a random orthogonal `W_a`
(`d_a × r_a`); the restricted system has `∏ r_a` equations and `Σ d_a r_a` unknowns. The
corner slice (`W_a = I[:, 1:r_a]`) is the special case.

**Reasoning.** The corner is what breaks on structured tensors: the unscrambled sphere's corner
is all zeros. The sketch makes the restriction generic with probability one.

**Verdict: keep.** This is the single decision that moved the frontier from `d ≈ 60` to
`d = 1000`. It is well argued in `docs/design/QuickDer-valence-n.md` and pinned by a test
that shows `:random` seeing what `:corner` cannot. Open point (recorded, not a defect): the
restriction sizes rest on the reference implementation's choices, not on a stated
sufficiency theorem.

## 3. Whitened restriction on by default

**Decision.** Thin QR of each mode unfolding so every diagonal Gram block of the restricted
system becomes a multiple of the identity; `whiten = true` is the default and `:Auto`
inherits it. Rank-deficient modes fall to an SVD and their trivial derivations are written
down exactly and published factored.

**Reasoning.** The matrix-free branch was not slow, it was not converging: ARPACK hit its cap
and returned nullity 0. Whitening fixes conditioning without touching the null space at a cost
of `n` small QRs.

**Verdict: keep.** Measured 5–15× fewer applies, a correct answer where there was none, and a
correctness fix for degenerate modes as a side effect. The one caveat is that it adds another
global tunable family (`QDN_TRIVIAL_MAX_BYTES`, `QDN_TRIVIAL_FACTORED`) to a file that already
has fifteen; see §10.

## 4. The verdict is a value, and two independent vetoes withhold it

**Decision.** `solve_nullspace` returns a `NullVerdict`; `certified` requires a spectral gap
*and* `status == :ok` *and* `undecidable == 0`. `status` (solver converged?) and `data_floor`
(can the input's rounding decide?) are independent reasons a spectrum is not evidence.
`DerivationReport` carries the verdict, both element types, restriction sizes and residuals to
the caller as a fourth return value behind `return_diagnostics = true`.

**Reasoning.** Three observed false certificates: an unconverged block Lanczos reading
"nullity 0, certified"; a Float16 tensor certifying a cut inside its own rounding; consumers
reading diagnostics out of `@warn` text that `maxlog = 1` silences.

**Verdict: keep, but finish.** This is the best idea on beta and the part that makes Float16
honest instead of merely fast. Unfinished edges: `rule` and `status` are bare `Symbol`s, so a
typo in a comparison is silently `false`; `NullVerdict` has seventeen positional fields and is
constructed positionally in two places; `store_eltype` defaults to the compute type, which is
the one default that reproduces the Float16 false certificate, so callers were patched instead
of the default; `den` still destructures `(vals, vecs)` and drops the verdict, returning an
empty densor on a failed solve. Make the two fields enums, add a keyword constructor, require
`store_eltype` when it differs from the compute type, and thread the verdict through `den`.

## 5. One floating-point policy in `Precision.jl`

**Decision.** Every tolerance is derived from the element type by a small set of functions:
`compute_eltype` (Float16 → Float32), `precision_floor` (where arithmetic stops separating
from zero), `data_floor` (where the input's rounding stops), `tol_default` (relative ceiling,
squared for Gram maps), `iter_tol`, `rank_rtol`, `qd_tolerance`. Constants were tuned by
experiment and the design is documented.

**Reasoning.** Fixed literal tolerances cannot serve Float16, Float32 and Float64 at once, and
the mixed case (Float16 stored, Float32 computed) needs two floors, not one.

**Verdict: keep, but finish.** The policy is coherent and the store/compute/verdict split is
correct. It is "one place" for the verdict, not for the package: the extensions still carry
`1e-10`/`1e-12` defaults that the policy then floors; `ShiftInvertSolver`'s CG tolerance,
`GRAM_SHIFT_REL`, `engaged`'s cutoff, `realCanonicalForm`'s and `nondeg`'s tolerances are
un-floored literals; `sqrt(eps(T))` is hand-written in QuickDerN where `qd_tolerance` exists;
`LUSolver` uses `max(m,n)·eps` directly though the docs say it uses `rank_rtol`; several
docstrings still describe `FLOOR_EPS` as 5 or 100 after it became 8, and one documented table
value (`tol_default(Float64; squared = true)`) is wrong. A grep-driven sweep closes this.

## 6. Solver registry with traits, instead of an `if/elseif` on symbols

**Decision.** `SOLVER_REGISTRY::Dict{Symbol, NullSolver}` populated by `register_solver!`,
extensions registering inside `__init__`; per-solver behaviour as traits (`wants_square`,
`densifies`, `initial_request`, `wants_seed`); `AutoSolver` densifies when cheap by bytes and
otherwise walks a preference list of matrix-free solvers.

**Reasoning.** Extension types are invisible to the parent module, so the old chain raised
`UndefVarError` for five of seven names. Traits let the central layer square the map or seed
the start vector on behalf of any solver.

**Verdict: keep.** Idiomatic Julia and the pattern the derivation-method factory should copy.
Gaps: the preference list is a hard-coded symbol list, so a third-party solver is reachable by
name but never chosen automatically; `GramSolver` and `LSMRSolver` draw from the global RNG yet
declare `wants_seed = false`, so seeded `den` is not reproducible; inner-solver options cannot
pass through `AutoSolver` (the dense branch rejects unknown kwargs); the registry is a global
`Dict` mutated without a lock though `register_solver!` is exported.

## 7. `den` as a rectangular `LinearMap` with a genuine adjoint; never densify

**Decision.** The densor map reuses `applyDerivation` forward and has a real adjoint; `den`
hands the rectangular map to `solve_nullspace` and lets the solver decide whether to square
it. `LSMRSolver` finds the null space by projection without squaring.

**Reasoning.** Densifying cost `O(n^7)` memory for an answer that is a handful of vectors.
Squaring loses half the digits of separation; the projection method needs no spectral
knowledge.

**Verdict: keep.** Correct at `n = 19` where every alternative failed, and consistent with the
Z-law by construction. Two defects to fix: `denLM` hard-codes `Vector{Float64}` and untyped
`LinearMap`s, so the Float32/Float16 direction stops at the densor; `den` defaults `nd = 10`
while `der` defaults `nd = -1`, so the T-set is silently truncated by default while the Z-set
is not. Also `den` ignores `Ω`'s operator restriction, so the `𝕋` slot of the design's
`(𝕋, Ω, P)` chisel is still absent.

## 8. `:Auto` is a composition, and it falls back on failure

**Decision.** `AutoDerMethod(quick, fallback, min_entries)`: QuickDer when `Ω` is
`IndTransverseOps`, some axis is engaged and the tensor is big enough; SylverLining otherwise
or on any non-interrupt exception. `stratify` defaults to `:Auto`.

**Reasoning.** QuickDer is generically correct and 65–150× faster; SylverLining is exact and
always applicable.

**Verdict: keep, but finish.** Composition over flags is right. But the fallback catches
*every* exception except `InterruptException` and logs at `@info`, so a `MethodError` or an
`OutOfMemoryError` inside QuickDer is invisible and an OOM on a `d^n` tensor triggers a fallback
whose operator is `n·d^{n+1}`. Applicability is a hand-written predicate plus that `try/catch`,
so a new method or a new `TransverseOps` kind silently gets SylverLining only; and `der(Γ)` and
`den(Γ)` still default to `:SylverLining` while `stratify` defaults to `:Auto`. Three entry
points, two default methods. Throw a dedicated `QuickDerDeclined` at the deliberate decline
sites and catch only that; pick `:Auto` everywhere, or say why not.

## 9. Device hooks as `Ref`s; an explicit backend request never degrades

**Decision.** `gpu_available`, `to_gpu`, `to_cpu`, `gpu_sync` are `Ref`s set by the Metal
extension's `__init__`; `sylvesterLM(backend = :metal)` errors if the extension is absent and
`:auto` never picks the GPU; QuickDer's device path is a hybrid (Gram and `M*X` on device,
factorizations on host).

**Reasoning.** An extension may only add methods on Julia 1.12; redefining a hook blocks its
precompilation. Metal is Float32-only, so the GPU cannot be a silent default.

**Verdict: keep, with one correctness fix.** The gating is correct. But `:metal` accepts a
Float64 tensor and computes it in fp32 while the precision policy sees Float64, so the null
solver runs with a Float64 floor nine decades below fp32 noise. Either refuse Float64 on the
Metal path or set the compute type to Float32 when the backend is `:metal`. Also the hooks
live in `NullSolvers.jl`, which has nothing to do with them; a `Device.jl` would be cleaner.
Writing the QuickDer kernel against `AbstractArray` and `similar`, so an `MtlArray` flows
through with no second implementation, is the right small design; but a second device (CUDA)
is hard today because two backends cannot both own the four global hooks, the Float32-only
rule and the bandwidth constants are Metal's measured on one M4 Max, and `G isa Array` is the
de-facto device predicate at six sites. A `Device` trait on `QuickDerMethod` with
`supported_eltypes`, the bandwidths and `on_device` would let the extension define the methods.
The measured conclusion that the eigensolve, not the tensor stages, is the movie's cost is
recorded correctly in CONTEXT but the superseded numbers still sit above it as live text.

## 10. Tunables as module-level `Ref`s

**Decision.** `QDN_DENSE_BUDGET_BYTES`, `QDN_GRAM_MIN_COLS`, `QDN_LIFT_CEILING`,
`QDN_TRIVIAL_MAX_BYTES`, `QDN_APPLY_COUNT`, `QDN_LAST_SOLVE_STATUS`, `QDN_STAGE_TIMES`,
`FASTDER_RESTRICT_CEILING`, `GRAM_GPU_FACTOR` and about ten more are `const X = Ref(...)`
read inside the solvers.

**Reasoning.** Benchmarks need to move a budget or count applies without a keyword on every
call.

**Verdict: revisit.** For measurement counters (`QDN_APPLY_COUNT`, stage times) a `Ref` is
fine and the `DerivationReport` is already replacing most of them. For *policy* (budgets,
ceilings, gap ratios) hidden global state means two concurrent solves in one process
interfere, tests that set a `Ref` and forget to reset it leak into later testsets, and a
consumer cannot see what a result was computed with. Move the policy knobs into
`QuickDerMethod` fields (it already has `whiten`, `device`, `seed`), and record them in the
report. `QDN_LIFT_CEILING` and `FASTDER_RESTRICT_CEILING` are the same constant with the same
reasoning and should be one entry in `Precision.jl`.

## 11. Shared numerics live in `FastDer3Valent.jl`

**Decision.** FastDer3Valent is retired as an algorithm (`:Auto` never picks it) but kept as a
line-for-line transcription of the reference and as the valence-3 oracle. The helpers QuickDer
and QuickSylver depend on in production, `_fastder_restrict_to_ops`, `_fastder_tall_nullspace`,
`_qd_tolerance`, `_qd_linear_equals_affine`, are defined in that file.

**Reasoning.** History: the helpers were written for FastDer first and reused later.

**Verdict: revisit.** A file named after a retired method holds the most delicate shared
numerics in the package (the Float32 restriction cut). Move the shared helpers into their own
file, un-export `FastDer3ValentMethod`, keep the transcription reachable as `:QuickDer3` for
the comparison tests. The restrict/solve/lift pattern is also written three times; the
tolerance fixes made to FastDer's copies were never applied to QuickSylver's, whose
verification is still absolute and therefore scale-dependent.

## 12. Include order is load-bearing

**Decision.** `Dleto.jl` includes `Precision.jl` first, `DerivationReport.jl` after
`NullSolvers.jl`, and the comments explain that field-type and annotation evaluation at
definition time force the order. Three `derTrOpsReduced` methods dropped their return-type
annotations to cope.

**Verdict: revisit.** The comments are honest; the cure is structural. The real cycle is
that `NullSolvers.jl` calls `_qdn_stage!` from `QuickDerN.jl`, which is included before it;
one of the four workaround comments rests on a false premise (`GAP_RATIO` is in scope, since
`Precision.jl` is included first). Moving the stage and budget knobs into a small file
included right after `Precision.jl`, and more generally one types file first
(`abstract type`s and `struct`s) then methods, removes the constraint and lets the
annotations come back. At the same time, replace the blanket `using ITensors` in a util file
included last, which is currently what makes `hasind`/`replaceind` resolve in core files, with
a curated import at the module top.

## 13. Chisels stay bare matrices

**Decision.** `UniversalChisel`, `TuckerChisel`, `AdjointChisel`, `CentroidChisel` return a
`Matrix`; `Chisel` (renamed from `ChiselFramed`) wraps `P` plus a frame. The decision to make
`Chisel` the full `(𝕋, Ω, P)` keyed by `Index`, with curried builders and a fixed transpose
convention, was taken on 2026-09-02 and not yet implemented.

**Verdict: keep the decision, implement it next.** The explicit-matrix parametrisation is the
advantage over Magma and must be preserved. But with no chisel type nothing dispatches on
kind, `AdjointChisel(valence, left, right)` is positional (the hazard the refactor plan
names), `normalize_chisel` still has no caller, and `engaged(P::Matrix)` rejects views so
AutoDer already works around it. This is the largest open design item in the core.

## 14. Package spine and dependencies

**Decision (inherited from `main`, not changed on beta).** `IJulia`, `PlotlyJS`, `PlotlyBase`,
`PlotlyKaleido`, `Plots`, `CSV`, `DataFrames`, `JSON` are hard dependencies. `KrylovKit` and
`IterativeSolvers` are hard dependencies *and* extension triggers; `Arpack` and `Metal` are
weak. `__init__` sets `ENV["WEBIO_WARN"]`. No `julia` compat; eight packages without compat.
Version 0.1.0.

**Verdict: revisit before any public beta.** A numerics library should not pull a notebook and
plotting stack. An extension whose trigger is unconditionally imported is a module with an
extra indirection, a "Loading …" banner on every `using Dleto`, and an empty registry during
precompilation; either fold those two extensions into `src/` or make the packages real
weakdeps. The tuned `AutoSolver` default is Arpack, a weakdep, so the same call runs ARPACK
under `bench/jl` and KrylovKit under `runtests`, which CONTEXT records as "two code paths, one
testset". Decide, then add compat for everything.

## 15. Process: `beta` as an approval gate, tagged; `CONTEXT.md` as the record

**Decision.** Daily branches named after song lyrics; `beta` advances only in reviewed, tested,
tagged steps; a downstream consumer pins to `beta`; `docs/CONTEXT.md` is handed to every new
session.

**Verdict: keep, tidy the record.** The process produced 113 commits with unusually good
messages and five tagged, green integration points in three days. `CONTEXT.md` has grown to
1,037 lines with no table of contents, the 2026-09-04 material split across three non-adjacent
sections, and a corrected measurement still present as live text above its correction. Add a
TOC, one dated section per session newest first, and apply corrections in place. There is no
docs index; `Refactor-Plan.md` still says nothing has been implemented.

---

## What the two-coder collision still looks like

The seam runs almost exactly along the SylverLining / solvers boundary: camelCase
(`derTrOpsReduced`, `embedITensors`, `globalDim`, `sylvesterLM`) on one side, snake_case
(`solve_nullspace`, `der_residual`, `_fastder_restrict_to_ops`) on the other; `@assert` for
user-facing validation in one, `error` with a message in the other; abstract-typed struct
fields (`ch::AbstractMatrix`, `val::Integer`, `Dict{Index,Integer}`) and `Int8`/`Int16` index
maps in the older core against concrete, parametrised types in the new solvers; `Base.:+` and
`Base.:*` pirated on `ITensor × AbstractArray` in `DletoBase.jl`. Beta did not create these
and repaired several; it also did not pick a side. The recommendation is to pick one
convention now, before the public surface grows further, and deprecate with aliases.
