# QuickDer for any valence: sketch-restrict, solve, lift, verify

Design note, 2026-09-03.  This generalises Liu's solve-and-lift (`:QuickDer`,
`src/solvers/FastDer3Valent.jl`, valence 3, corner restriction, dense solve) to
tensors of any valence `n`, any per-axis dimensions, any chisel, with a
restriction that survives sparse/structured tensors and a restricted solve that
stays feasible at d = 500..1000.  The valence-3 transcription stays as the
reference oracle (`:FastDer3Valent` / `:QuickDer3`).

## 0. Conventions

Tensor `Γ` with axes `a = 1..n`, dims `d_a`.  Mode product with a matrix
`M` (d_a × k):

    (Γ ×_a M)[.., i, ..] = Σ_p Γ[.., p, ..] M[p, i]        # M's FIRST index meets Γ

This is exactly what `applyDerivation` / `embedITensors(Ω, ·)` do: an operator
ITensor carries (frame index, temporary index) and the frame index is
contracted.  Therefore the kernel below needs NO transposes at the ITensor
boundary (unlike FastDer3Valent, whose `X` acted by left multiplication).

Chisel `P` (m × n).  Derivation equation, one residual tensor per row `ρ`:

    E_ρ(M) := Σ_a P[ρ,a] (Γ ×_a M_a) = 0,   M_a ∈ F^{d_a × d_a}      (E)

Axes with `P[:,a] == 0` (disengaged) carry no unknown.  `Der(P,Γ)` is the
solution space; `stratify` needs a basis of it intersected with the operator
space `Ω` (e.g. `SymmetricOp()`), which `_fastder_restrict_to_ops` already does
for any valence -- generalise its input from triples to `Vector{Matrix}`.

## 1. Restriction by sketching (coordinate-free corner)

For each axis choose an orthogonal `Q_a = [W_a | W_a⊥]`, `W_a` of size d_a × r_a.
Contract every output axis of (E) with `W`:

    E_ρ ×_1 W_1 ⋯ ×_n W_n = Σ_a P[ρ,a] (Γ ×_{b≠a} W_b) ×_a (M_a W_a)

Define the CROSS SKETCH `S_a := Γ ×_{b≠a} W_b` (shape r_1..r_{a-1}, d_a, r_{a+1}..r_n)
and the RESTRICTED UNKNOWN `Y_a := M_a W_a` (d_a × r_a).  The restricted system

    R(Y) := Σ_a P[ρ,a] (S_a ×_a Y_a) = 0    for all ρ                         (R)

has ∏ r_a · m equations and Σ_a d_a r_a unknowns (engaged axes only).

* `W_a = I[:, 1:r_a]` reproduces Liu's corner restriction exactly (cheap: `S_a`
  is a slice, no arithmetic).  Call this `restriction = :corner`.
* `W_a` = Q-factor of a randn(d_a, r_a) is `restriction = :random`.  It costs
  one pass `Γ ×_{b≠a} W_b` per axis, O(n · r · d^n) flops (sparse Γ: O(n·r·nnz)),
  and it is what makes structured tensors work: the UNscrambled sphere
  (support i_1+…+i_n = d-1) has an all-zero corner, so `:corner` returns
  garbage there while `:random` is generic.  Default `:random`.

Choosing `r`.  Generic solvability needs (i) ∏ r_a ≥ Σ_a d_a r_a + slack (the
restricted system has full column rank up to the true nullity) and (ii) for
every engaged axis, `R_a := ∏_{b≠a} r_b ≥ d_a + slack` (the lift operator has
full column rank).  Start from the balanced value
`r_a = min(d_a, ceil((n·max d)^(1/(n-1))) + 1)` and bump the axis with the
smallest `r_a/d_a` that is not saturated (`r_a < d_a`) until both hold.  Tiny
axes (a colour axis of length 3) saturate, `W_a = I`, and never need lifting.
Examples: (100,100,100,3) → r = (11,11,11,3); (100,100,100) → r = 19;
(1000,1000,1000) → r = 57.  Allow an explicit override `sizes = (r_1,…,r_n)`.

## 2. The restricted solve

(R) is linear in `y = vcat(vec(Y_a) for engaged a)`.  Two representations:

* DENSE, when `(∏r · m) · (Σ d_a r_a)` entries fit `DENSE_BUDGET_BYTES`
  (d ≤ ~120 at valence 3, ≤ ~100 at valence 4): the block for axis `a` is
  `Perm_a · kron(I_{r_a}, P[ρ,a]·S_a(a)ᵗ)` where `S_a(a)` is the mode-a
  unfolding (d_a × R_a) and `Perm_a` reorders rows from (i_{-a}, i_a) to
  column-major (i_1..i_n): `Perm_a = vec(permutedims(reshape(1:∏r, r...), (setdiff(1:n,a)..., a)))`
  i.e. `Block[Perm_a, :] = kron(I, Sᵗ)`.  Hand it to
  `solve_nullspace(Matrix, :SVDSolver; tol)`.  (Do NOT go through
  `Matrix(::FunctionMap)`, which would apply the map ∏r times.)
* MATRIX-FREE otherwise: a `LinearMaps.LinearMap` whose forward is `y ↦ vec(R(Y))`
  (n mode products of the cross sketches, cost ∏r · Σ d_a per apply) and whose
  adjoint is `E ↦ (P[ρ,a]·(S_a contracted with E over all axes but a))_a`.
  Hand it to `solve_nullspace(L, solver; tol)` with `solver = :AutoSolver`
  (rectangular ⇒ LSMR projection first, no squaring; eigensolvers square it
  as a composition).  At (1000,1000,1000) the map is 185k × 171k and an apply
  is ~5e8 flops -- roughly 6000× cheaper than one apply of the full
  `sylvesterLM` operator.

Both return `(vals, vecs, verdict)`; `vecs` columns are restricted solutions
`y`.  Use the caller's `tol`; the verdict's `certified` flag is reported.

## 3. The lift

For engaged axis `a` with `r_a < d_a`, the unknown part is `Z_a := M_a W_a⊥`
(d_a × (d_a − r_a)); then `M_a = [Y_a  Z_a] · Q_aᵗ`.  Use the equations of (E)
whose axis-a output is contracted with `W_a⊥` and every other axis with `W_b`:

    P[ρ,a] · (S_a ×_a Z_a) = − Σ_{b≠a} P[ρ,b] · (H_{ab} ×_b Y_b)              (L_a)

where `H_{ab} := Γ ×_a W_a⊥ ×_{c≠a,b} W_c` (shape: r on axes c, d_b on b,
d_a − r_a on a).  Form it by contracting the small `W_c` first (O(r · d^n)),
then `W_a⊥` on the reduced tensor (O(d_a² d_b r^{n-2})) -- never `Γ ×_a W_a⊥`
directly, which is O(d^{n+1}).  Unfold (L_a) with axis a last: LHS operator is
`P[ρ,a]·S_a(a)ᵗ` (R_a × d_a, rows stacked over ρ), full column rank by (ii);
RHS is R_a·m × (d_a − r_a).  One thin QR per axis, shared by every basis
vector; solve with `\`.  For the corner restriction `H_{ab}` is a plain slice
`Γ[I,…,Î_a,…,(:)_b,…,I]`, exactly Liu's `T_I_hat_J` etc.

Consistency filter (replaces Liu's "Θ_feas ≠ Θ" error).  A restricted
solution that is not the restriction of a true derivation leaves a residual in
(L_a).  The residual is LINEAR in `y`, so stack the residuals of all axes for
each basis column into a matrix `Res` (∑ rows × k) and keep
`nullspace(Res; atol = tol·scale)` combinations (same trick as
`_fastder_restrict_to_ops`).  This makes the method correct, not merely
generic: on a non-generic tensor whose restricted nullspace is too large, the
spurious directions are filtered rather than erroring.  If nothing survives,
retry once with `r` bumped by 50 % (capped at `d`), then give up with a clear
error naming the sizes (so `AutoDer` can fall back to `:SylverLining`).

## 4. Verification (the Z-law, always)

Solve-and-lift is only generically correct, so verify with the defining
equation.  `verify = :random` (default): pick `nslices = 4` random output
indices along the largest engaged axis `â` and check
`‖E_ρ(M)[.., i_â, ..]‖ ≤ tol · ‖Γ‖ · Σ_a ‖M_a‖` for every basis element
(cost O(n · nslices · d^n) total per element -- contract axis â with the
selected columns FIRST for the a ≠ â terms; for the a = â term use
`M_â[:, slice]`).  `verify = :full` runs the whole equation when
`prod(dims) ≤ 2e7`, else falls back to `:random`.  `verify = :none` for
benchmarking only.  Failure is an error (the tensor was not generic enough for
these sizes: report `r`, suggest `sizes`).

## 5. API

    QuickDerMethod(; restriction = :random, sizes = nothing, solver = :AutoSolver,
                     verify = :random, nslices = 4, seed = nothing)
    get_derivation_method(:QuickDer; kwargs...)    # ANY valence now
    :QuickDer3 / :FastDer3Valent                   # the valence-3 transcription, kept as oracle

`derTrOpsReduced(::QuickDerMethod, Ω, P, Γ; tol, nd)` returns
`(Ω, id_map, ders)` with `ders` the Ω-coordinates, like FastDer3Valent does
today.  Requirements relaxed to: `Ω isa IndTransverseOps`, `Γ` carries its
frame, `P` any m × n with at least one engaged axis.  Element type follows
`eltype(Γ)` (Float32 stays Float32; tolerances floored at `sqrt(eps(T))` as
`_qd_tolerance` does).  Read `Γ` with `ITensors.array(Γ, fr...)` where the
order already matches to avoid a d^n copy.

Kernel lives in `src/solvers/QuickDerN.jl`, plain Arrays inside, ITensor only
at the boundary, a `ttm!(out, G, M, a)` helper built on `permutedims` +
`mul!` (reshape to (d_a, rest); `out = Mᵗ·G_(a)`) so a later swap to
TensorOperations/Strided/Finch is one function.

## 6. Cost model (for `AutoDer`, section 7 of the plan)

| stage | flops | memory |
|---|---|---|
| sketch `S_a`, all a | n · r · d^n  (sparse: n·r·nnz) | n · r^{n-1} · d |
| restricted solve, dense | (∏r)(Σdr)² | (∏r)(Σdr) |
| restricted solve, iterative | ~500 · ∏r · Σd | Σ d r · (nv+8) |
| lift, per axis | R_a d_a² + R_a d_a (d_a − r_a) | R_a · d_a |
| verify `:random` | n · nslices · d^n per basis element | d^n |

Full `sylvesterLM` for comparison: ~500 · n · d^{n+1} flops.  QuickDer wins
whenever the tensor is dense-generic and d ≳ 20; SylverLining remains the
fallback for non-generic tensors and small cases.

## 7. Tests (test/TestQuickDerN.jl)

Every test verifies with `der_residual` (Z-law) and compares nullity with
`:SylverLining`/dense SVD where feasible:
* random dense valence 3/4/5, unequal dims incl. a length-3 axis: nullity =
  n−1 (scalars), residual < 1e-10 (Float64), Float32 works;
* diagonal valence-4 tensor, d = 4..6: nullity (n−1)·d;
* valence-3 random vs `:QuickDer3` (same basis span);
* unscrambled sparse sphere valence 3 and 4 (corner is all-zero): `:random`
  succeeds, `:corner` is expected to fail/filter;
* scrambled sphere (bench/SphereHarness.jl, valence 3 and 4, d = 12..20):
  with `SymmetricOp()` nullity = n (n−1 scalars + Euler), stratification
  `lsq_err < 1e-8`;
* disengaged axis: `P = [1 1 0]` on valence 3 (compare with SylverLining);
* two-row chisel (CentroidChisel(3)) on a random tensor vs SylverLining.
