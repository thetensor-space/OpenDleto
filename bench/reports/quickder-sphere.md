# QuickDer on the scrambled sphere

Input: `bench/SphereHarness.jl`, `build_sphere(d)` -- the sphere octant sampled
in x^2 (support i + j + k = d-1, random entries), scrambled by random
orthogonal matrices, then `nondeg`.  Method under test: `:QuickDer`
(`src/solvers/FastDer3Valent.jl`, Liu's solve-and-lift), compared with
`:SylverLining` and the `:ArpackSolver` / `:SVDSolver` null solvers on the
`SymmetricOp()` operator space that fits the scramble.

Before this work, `run_stratify(inp; method=:QuickDer)` needed
`ops = UniversalOp()`, returned nullity 13, and stratified badly (lsq_err
6.9e-2 at d = 20); in Float32 it errored with
"LinearEqualsAffine assumes every parameter is feasible".

## 1. The universal derivation space is 13-dimensional, and QuickDer's count is exact

Dense SVD of the densor map (`Dleto.sylvesterLM(Ω, ch, Γ)` densified, op dim
3d^2) on the `UniversalOp()` input:

| d  | densor map      | singular values around the gap        | nullity (rtol 1e-3 .. 1e-12) | QuickDer nullity | QuickDer residuals |
|----|-----------------|---------------------------------------|------------------------------|------------------|--------------------|
| 10 | 1000 x 300      | ..., 7.6e-2, 5.7e-4, 1.9e-15, ...     | 14 at 1e-3, else 13          | 13               | ~3e-15             |
| 15 | 3375 x 675      | ..., 1.6e-2, 9.1e-3, 3.7e-15, ...     | 13                           | 13               | ~8e-15             |
| 20 | 8000 x 1200     | ..., 4.5e-2, 1.1e-2, 7.2e-15, ...     | 13                           | 13               | ~1e-14             |

QuickDer's 13 is the true dimension, not an artefact of its `atol = 1e-6`: the
spectral gap is 1e-2 -> 1e-15 and the rank of its basis is 13 at rtol 1e-8.
(Only at d = 10 is there a stray singular value of 5.7e-4 that the harness's
`SVDSolver` at tol 1e-6 also admits, giving 14; that is a property of the
d = 10 tensor, not of either solver.)

Why 13, independent of d.  Read the sphere tensor as a generic ternary form
F(x, y, z) = sum s_ijk x^i y^j z^k of degree n = d-1; axis a of the tensor is
the exponent of one variable, so an axis operator D_a is a linear map on
polynomials in that variable, and D_a[i, i'] has *degree* i' - i.  The
derivation equation decomposes by degree.  Every output monomial
x^i' y^j' z^k' receives exactly one term from each axis (the source exponent is
forced by i + j + k = n), so the degree-s block is a small linear system:

* degree 0 (diagonal maps): 3(n+1) unknowns, C(n+2, 2) equations, generic
  solution space of dimension 3 -- the two scalar derivations and the Euler
  operator (x d/dx, y d/dy, z d/dz - (d-1)), which is the sphere derivation.
* degree -s (lowering by s, output plane of total degree m = n - s): 3(m+1)
  unknowns against C(m+2, 2) equations.  For m = 0, 1, 2, 3 that is 3, 6, 9,
  12 unknowns against 1, 3, 6, 10 equations, so 2 + 3 + 3 + 2 = 10 solutions;
  for m >= 4 the block is over-determined and generically empty.
* degree +s (raising): always over-determined, empty.

Total 3 + 10 = 13 for every d >= 6.  Verified numerically at d = 6, 7, 12: the
pulled-back derivations use only degrees 0, -(d-1), -(d-2), -(d-3), -(d-4),
with ranks 3, 2, 3, 3, 2.  The ten extra derivations are strictly lower
triangular in the sphere frame, hence nilpotent, hence *not symmetric* in any
orthogonal frame; the dimension of (universal derivations) ∩ (symmetric
operators) is exactly 3 at every d tested.

This is why QuickDer's stratification was wrong: `stratify` puts a random
element of the returned space into real canonical form, and a random element
of the 13-dimensional algebra has a nilpotent part, so its eigenbasis does not
diagonalise the sphere.

## 2. Design: QuickDer honours the operator space Ω

`derTrOpsReduced(::FastDer3ValentMethod, Ω, P, Γ)` used to reject any
`Ω` that was not `UniversalOp()` on every axis.  It now accepts any
`IndTransverseOps` of valency 3, runs the kernel over all matrices as before,
and then intersects the universal basis with Ω:

1. `_fastder_triple_matrices(X, Y, Z)` turns each solution triple into the
   three (frame x temporary) operator matrices `embedITensors` expects (X is
   transposed, as `_encode_basis_vector` did).  The basis is normalised.
2. `_fastder_projector(op, n, T)` builds, for each axis whose local operator is
   not `UniversalOp`, the least-squares coordinate map onto the local space
   from the two primitives every `Operator` provides: `unsafe_embed` (E,
   coordinates -> matrix) and `unsafe_transposeEmbed` (E').  The Gram matrix
   G = E'E is assembled column by column (localDim calls of cost n^2), and is
   diagonal for every operator in `src/ops/OperatorImpls.jl`, so
   `coords(M) = G \ E'M` costs O(n^2) and `embed(coords(M))` is the orthogonal
   projection of M onto the space.  A non-diagonal G falls back to Cholesky.
3. `_fastder_restrict_to_ops(Ω, basis, atol)`: a combination
   sum_i c_i D^(i) lies in Ω iff on every axis (I - Π_a) sum_i c_i D_a^(i) = 0.
   The residuals of the k basis elements are stacked into a (sum_a n_a^2) x k
   matrix whose null space (absolute tolerance `atol`, `rtol = 0`, because when
   everything is already in Ω the residual matrix is pure roundoff) is the
   coefficient space.  Each null vector is assembled, projected axis by axis
   with `coords`, and written into the `Ω.soffsets[a]:Ω.eoffsets[a]` block --
   i.e. genuine Ω coordinates, so `embedITensors(Ω, ders * c)` works unchanged
   and the return value has the same meaning as SylverLining's on that Ω.
   For all-universal Ω the step reduces to `unsafe_coordinates(Ω, ·)`, the
   old behaviour.

The cost is k = 13 residuals per axis plus one Gram matrix per axis
(localDim^2 entries: 1275^2 for SymmetricOp at d = 50, about 13 MB), which is
small next to the kernel's dense restricted system.

Checked at d = 20: QuickDer's 3 symmetric derivations and `SVDSolver`'s 3 span
the same subspace (principal-angle cosines 1.0, 1.0, 1.0; densor residuals
2e-14 .. 2e-13), and `run_stratify(build_sphere(20); method=:QuickDer)` gives
nullity 3, lsq_err 4.5e-14, support 1.0.  With `DiagonalOp()` both methods
return the 2 scalar derivations (the sphere derivation is not diagonal in the
nondeg frame), also in agreement.

Not changed: the frame check.  Γ is now read as `Array(Γ, frames(Ω)...)` so the
coordinates follow Ω's axis order rather than Γ's index order, and
`_fastder_validate_compatibility` requires Γ to carry Ω's frame.

## 3. Float32: tolerances are relative and floored at sqrt(eps)

The reference hard-coded `atol = 1e-6` as an *absolute* tolerance in three
places: the rank / null-space decisions, the lift's feasibility test
`|K' N_i| ≈ 0`, and the final verification `|X R_k + S_k Y - T Z_k| ≈ 0`.
For a Float32 tensor of these sizes the roundoff in `K' N_i` is ~1e-6..1e-5,
so every Float32 run failed the feasibility check.  Now:

* `_qd_tolerance(T, tol) = max(T(tol), sqrt(eps(T)))`: the working tolerance
  is the caller's `tol` (still 1e-6 by default) floored at what the element
  type can certify (1.5e-8 for Float64, 3.4e-4 for Float32).
* Rank and null-space decisions use `rtol` only (relative to the largest
  singular value; the absolute `atol` made the answer depend on the scale of Γ).
* Feasibility in `_qd_linear_equals_affine` is `|K' N| <= atol * scale`, where
  the lift passes `scale = |R| + |S| + |T|`: the restricted basis vectors are
  unit vectors, so every right-hand side is bounded by the tensor slices it
  was built from.  (Scaling by `|N_i|` alone fails when `N_i` is itself
  roundoff, which happens for the diagonal tensor in the test suite.)
* Verification uses one scale per triple, `|X||R| + |S||Y| + |T||Z|`, applied
  to every slice, for the same reason.
* `zeros(...)` in the lift are typed `Tnum`, so Float32 stays Float32.

Result: `run_stratify(build_sphere(20; T=Float32); method=:QuickDer)` runs,
nullity 3, lsq_err 2.3e-5, support 1.0 (SVDSolver in Float32: 1.5e-5).  The
Float32 `UniversalOp()` input also runs (nullity 13).

`test/TestFastDer3Valent.jl` and `test/TestDerivationLaws.jl` pass (181/181),
as does `test/TestSylverLining.jl`.  Float32 at d = 40 and 50 fails with a
detected error, see section 4.

## 4. Timings

`bench/reports/QuickDerSphereBench.jl` (rows also in
`bench/reports/quickder-sphere.csv`, log in `quickder-sphere.log`).  Warm-up at
d = 8, two passes per config; then one timed `run_stratify` per (d, config) on
the `SymmetricOp()` input, `julia -t 2`, under CPU contention from other
processes, so only relative comparisons within the table are meaningful.

Machine state during the run: load average 39-67 on 8 cores (other agents),
`BLAS.get_num_threads() == 4`; a bare `svd` of a 1728 x 1440 Float64 matrix
(QuickDer's restricted system at d = 40) took 2.0 s in that state.

| d  | config (SymmetricOp input) | seconds | bytes (MB) | nullity | lsq_err  | support |
|----|----------------------------|--------:|-----------:|--------:|---------:|--------:|
| 20 | QuickDer (Float64)         |   0.720 |       83.7 | 3       | 4.5e-14  | 1.0000  |
| 20 | ArpackSolver               |   0.315 |      559.6 | 3       | 2.6e-14  | 1.0000  |
| 20 | SVDSolver                  |   1.034 |      754.6 | 3       | 1.9e-14  | 1.0000  |
| 20 | QuickDer (Float32)         |   0.481 |       51.3 | 3       | 2.0e-05  | 1.0000  |
| 30 | QuickDer (Float64)         |   7.582 |      302.2 | 3       | 1.4e-13  | 1.0000  |
| 30 | ArpackSolver               |   4.179 |     2013.0 | 3       | 3.8e-14  | 1.0000  |
| 30 | SVDSolver                  |  22.274 |     4770.2 | 3       | 1.9e-13  | 1.0000  |
| 30 | QuickDer (Float32)         |   3.897 |      170.8 | 3       | 6.0e-05  | 1.0000  |
| 40 | QuickDer (Float64)         |  14.016 |      777.2 | 3       | 7.4e-14  | 1.0000  |
| 40 | ArpackSolver               |  29.720 |     4334.3 | 3       | 7.9e-14  | 1.0000  |
| 40 | SVDSolver                  | 162.691 |    17793.2 | 3       | 5.8e-14  | 1.0000  |
| 40 | QuickDer (Float32)         |   9.398 |      128.2 | error: verification failed at (a',b',c') = (12,12,12) | | |
| 50 | QuickDer (Float64)         |  35.288 |     1702.6 | 3       | 3.0e-12  | 1.0000  |
| 50 | ArpackSolver               |  42.024 |     8982.6 | 4       | 3.8e-08  | 1.0000  |
| 50 | SVDSolver                  | 292.431 |    50904.9 | 4       | 1.2e-07  | 1.0000  |
| 50 | QuickDer (Float32)         |  24.622 |      264.2 | error: lift matrix not of full column rank | | |

Reading the table:

* QuickDer now returns the same 3-dimensional symmetric derivation space as
  the null solvers and stratifies to the same precision (1e-14 .. 1e-12).
* Wall time: Arpack wins at d = 20 and 30, QuickDer at d = 40 and 50 (2x
  Arpack at d = 40, 1.2x at d = 50); SVDSolver is 8-12x slower than QuickDer
  from d = 30 on.  Under this much contention the absolute numbers are
  inflated several-fold; the ordering within a d is what the run supports.
* Memory: QuickDer allocates 5-7x less than ArpackSolver and 9-30x less than
  SVDSolver at every d.
* At d = 50 the null solvers report nullity 4 and recover only to 1e-7: the
  sphere tensor at that size has a near-derivation whose singular value sits
  around the 1e-6 tolerance (the restricted system shows it too: 14 relative
  singular values below 1e-6 at d = 50, 13 at d = 40).  QuickDer's
  verification and the Ω-intersection filter it out and return the true 3.
  This is a property of the benchmark tensor at d = 50, not of any solver.

Where QuickDer's time goes (`scratchpad/phases.jl`, Float64, same machine
state; the `Ω-intersect` column is the new step):

| d  | (a',b',c') | kernel (solve-and-lift) | verification | Ω-intersect     | stratify |
|----|------------|-------------------------|--------------|-----------------|----------|
| 20 | (9,9,9)    | 1.27 s, 43 MB           | 0.00 s       | 0.23 s, 32 MB   | 0.00 s   |
| 30 | (11,11,11) | 5.93 s, 133 MB          | 0.04 s       | 0.48 s, 145 MB  | 0.00 s   |
| 40 | (12,12,12) | 14.08 s, 260 MB         | 0.24 s       | 0.57 s, 438 MB  | 0.02 s   |

The kernel is the dense SVD of the restricted system ((a'b'c') x 3a'd:
1728 x 1440 at d = 40, 2744 x 2100 at d = 50); building the system takes
0.1 s and the lift is negligible.  The intersection with Ω is under 5% of the
time; its allocations come from assembling the Gram matrix through
`unsafe_embed` / `unsafe_transposeEmbed` one coordinate at a time and could be
cut with a SymmetricOp-specific path, which did not seem worth the loss of
generality.

### Float32 at d >= 40

Both Float32 failures are detected failures, not wrong answers.  The
restricted system's relative singular values at d = 40 are: 13 below 1e-6 (the
true null space) and a 14th between 1e-6 and 3.4e-4; at d = 50, 14 below 1e-6
and 24 below 3.4e-4.  With the Float32 floor `sqrt(eps(Float32)) = 3.4e-4`
the null space comes back with 14 (d = 40) or more vectors, the lift is then
inconsistent, and either the verification (d = 40) or the full-column-rank
check on the lift matrix (d = 50) reports it.  The spectral gap of this tensor
at those sizes is simply below what Float32 can resolve for a system of this
size; Float64 is needed from d = 40 on.

## Files

* `src/solvers/FastDer3Valent.jl` -- the change (tolerances, Ω intersection).
* `bench/reports/QuickDerSphereBench.jl`, `quickder-sphere.csv`,
  `quickder-sphere.log` -- the sweep.
