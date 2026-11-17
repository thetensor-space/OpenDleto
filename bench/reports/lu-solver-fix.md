# LUSolver: wrong rank and a self-repeating basis, fixed with column-pivoted QR

Branch `signed-sealed-delivered/der-and-derivation-law`, 2026-09-03.
File: `src/solvers/NullSolvers.jl`, the `solve(::LUSolver, L::LinearMap; nv, tol)`
method and its docstring.  Harness: `bench/SphereHarness.jl`
(`build_sphere(d)`, `run_stratify(inp; solver)`), sphere octant with
`SymmetricOp()`, exact derivation space of dimension 3.

## Symptom

In the stratification sweep (`bench/stratify-solver-profile.csv`) `LUSolver`
returned nullity 1 with reconstruction error 0.1-0.9 at d = 10, 15, 20, 30,
35, 40, 45, while `SVDSolver`, `ArpackSolver` and `KrylovSolver` returned
nullity 3 with error ~1e-14.  At d = 25 it returned nullity 3 with error
4.6e-2, in 4.3 s and 13 GB where SVD took 0.9 s and 2 GB.

## Diagnosis

The operator handed to the solver is the derivation-densor map: square
(n = 3·d(d+1)/2: 165, 360, 630, 975, 1395 for d = 10..30), symmetric to
5e-17, positive semidefinite, with three singular values at 1e-15 and the next
at 0.24 (d = 10).  Any rank-revealing method sees a 14-orders-of-magnitude gap.

`lu` with partial pivoting is not rank revealing.  It pivots rows only: in the
current column it divides by the largest available entry, but it never
reorders columns.  When column j is a linear combination of columns 1:j-1,
every entry of column j has already been eliminated to roundoff by the time it
is reached, so the pivot is ~1e-15 whichever row is chosen, and elimination
carries on dividing by it.  The tiny pivot therefore lands wherever the first
dependent column is, not at the trailing end.

On this operator that position is structural.  The operator space is three
axis blocks of d(d+1)/2; the two scalar derivations make the first column of
the third block a combination of the first two blocks, so `lu` puts one tiny
pivot at exactly 2n/3 at every size and the other two last:

    d = 10   n = 165    tiny pivots at 110, 164, 165
    d = 25   n = 975    tiny pivots at 650, 974, 975
    d = 30   n = 1395   tiny pivots at 930, 1394, 1395

The old code counted `abs.(diag(U)) .> 1e-8` to get r = n - 3 (correct), then
assumed the last n - r columns were the free ones and solved
`U[1:r,1:r] x = -U[1:r,j]` for each.  `U[1:r,1:r]` contains the 1e-15 pivot at
2n/3, so those solves divide by it.  The result is
`1e15 · (the null vector belonging to column 2n/3) + (garbage of size 1)`:

* Where the garbage dominated, the residual `‖Mv‖/‖v‖` was large and the
  vector was filtered out: d = 10 gave residuals 0.0255, 2.4 and 2.8e-15, d = 30
  gave 1.74, 0.613 and 2.8e-14 -- nullity 1.
* At d = 25 the amplification won: residuals 7.9e-14, 7.8e-14, 2.1e-14 with
  `‖v‖ = 2.8e14, 6.4e13, 6.2`.  All three passed the threshold, but two of them
  are the same vector to thirteen digits -- singular values of the normalized
  basis are `1.43, 0.98, 5.7e-13`.  The "basis" spans a two-dimensional space.
  `stratify` then drew a random derivation missing one direction, giving the
  4.6e-2 error; the 13 GB / 4.3 s are downstream of that degenerate input.

So there were two defects: the rank/column assumption of partial-pivoting LU,
and a per-vector residual check that cannot detect a basis repeating itself.

## Fix

`LUSolver` now uses column-pivoted QR, `qr(M, ColumnNorm())` (LAPACK `geqp3`).
At each step it moves the column of largest remaining norm to the front, so
`|R[1,1]| >= |R[2,2]| >= ...` and every dependent column is pushed to the
trailing block.  With `M P = Q R`:

* rank `r = count(|R[j,j]| > tol · |R[1,1]|)`, `tol` defaulting to
  `max(m,n)·eps(T)` (`LinearAlgebra.rank`'s convention; the old `tol` was an
  absolute 1e-8);
* null vectors `P [ -R[1:r,1:r] \ R[1:r,j] ; e_j ]` for j = n, n-1, ..., r+1,
  each with `M v = Q [0; R[j,j]; 0] ≈ 0` by construction -- `R[1:r,1:r]` is
  well conditioned because every small diagonal has been pivoted out of it;
* when `nv` exceeds the nullity, padding vectors continue the same formula
  into the independent columns j = r, r-1, ... using `R[1:j-1,1:j-1]`, so
  `solve_nullspace` (which doubles `nv` while every value is below threshold)
  always sees a non-null value once the null space is exhausted;
* the whole returned set is orthonormalized in order (thin `qr`), so the
  leading n - r columns still span the null space, and only then are
  `vals = ‖M q‖` computed.  Duplicate directions would cancel to noise and fail
  the residual test; padding vectors become orthogonal to the null space, so
  their residual is at least the smallest non-zero singular value.

The method stays factorization-based and shape-agnostic (rectangular, non
symmetric, indefinite inputs all fine; tested on random 40x100 rank-30 and
100x60 rank-25 matrices: nullities 70 and 35, residuals 6e-14).  Pivoted
Cholesky of the symmetric PSD operator would be cheaper still (0.02 s at
n = 1395) but LAPACK `pstrf` reported full rank on these matrices at its
default tolerance, it needs a square PSD input (so `LᵗL` for the rectangular
densor maps, the squaring the LSMR path was rewritten to avoid), and the
saving is tenths of a second against a densification costing seconds.

## Before / after

Before (from `bench/stratify-solver-profile.csv`, earlier session, unloaded
machine):

    d   n     LUSolver (old)                          SVDSolver
             nullity  lsq_err   sec    GB alloc       nullity  lsq_err   sec    GB alloc
    10  165     1     8.55e-01  0.030   0.05            3     1.44e-14  0.023   0.05
    15  360     1     4.21e-01  0.102   0.23            3     3.56e-14  0.123   0.23
    20  630     1     3.75e-01  0.237   0.75            3     1.72e-14  0.302   0.75
    25  975     3     4.60e-02  4.287  14.00            3     3.04e-14  0.928   2.05
    30 1395     1     9.64e-01  1.297   4.74            3     3.24e-14  1.964   4.76

After (this session; Float64; post warm-up).  NOTE the machine carried a load
average of 42 on 8 cores from concurrent agents, which inflated every wall time
here (SVD at d = 30 took 44.9 s against 1.96 s in the earlier CSV); the
LU/SVD ratio is the meaningful column.

    d   n     LUSolver (CPQR)                         SVDSolver                    LU/SVD
             nullity  lsq_err   sec    GB alloc       nullity  lsq_err   sec    GB alloc
    10  165     3     3.64e-14  0.051   0.05            3     1.44e-14  0.174   0.05    0.29
    15  360     3     1.44e-14  0.186   0.23            3     3.56e-14  0.823   0.23    0.23
    20  630     3     2.15e-14  0.570   0.74            3     1.72e-14  1.945   0.75    0.29
    25  975     3     2.56e-14  2.269   2.01            3     3.04e-14  9.689   2.05    0.23
    30 1395     3     3.04e-14 25.577   4.70            3     3.24e-14 44.878   4.76    0.57

Factorization step alone on the captured operators (same session, lighter
load; `Matrix(L)` densification, shared by both solvers, excluded):

    n      lu      qr(ColumnNorm)   cholesky(RowMaximum)   svd
    165    0.000s      0.001s            0.000s            0.003s
    975    0.009s      0.061s            0.009s            0.320s
    1395   0.015s      0.141s            0.021s            0.638s

Float32 input (`build_sphere(d; T=Float32)`, scored in Float64):

    d    LUSolver nullity  lsq_err     SVDSolver nullity  lsq_err
    10        3            2.63e-06         3             5.43e-06
    15        3            3.71e-06         3             8.04e-05
    20        3            9.88e-06         3             8.68e-06
    25        3            7.49e-06         3             1.52e-05
    30        3            1.10e-05         3             1.04e-04

Float32 rank cut at n = 1395: trailing `|R_jj|/|R_11|` are
`0.0781, 0.0538, 1.3e-7, 6.7e-8, 4.8e-8` against a cut of 1.7e-4.

Contract checks on the d = 10 operator: `solve(LUSolver(), L; nv=16)` returns
16 orthonormal vectors (`‖QᵗQ - I‖ = 2.7e-15`) with vals
`4.4e-15, 4.7e-15, 4.5e-15, 0.327, 0.45, ...`; the leading three agree with
SVD's null space to 8.7e-15 (projector distance); `nv=2` returns two null
vectors (so `solve_nullspace` doubles); `solve_nullspace(L, :LUSolver)`
returns nullity 3.

## Test suite

`julia -t 2 --project=. -e 'using Pkg; Pkg.test()'` on the final code:
36 passed, 0 failed, 1 errored.  The one error is
`stratify returns a usable change of frame / structured / QuickDer`
(`test/TestDerivationLaws.jl:287`): `FastDer3Valent did not find a correct
solution triple at (a',b',c') = (5,5,5)`, raised from
`src/solvers/FastDer3Valent.jl:328`.  That path (`FastDer3ValentMethod`) never
calls `NullSolvers`, and `FastDer3Valent.jl` carries another agent's
uncommitted edits, so it is pre-existing and unrelated to this change.  All
other testsets pass, including `structured / SylverLining` (16/16) and
`generic / SylverLining` (12/12), which are the ones that go through
`solve_nullspace`.
