# The restricted eigensolve: four candidates, two adopted

2026-09-08. Branch `under-pressure/restricted-solve`. Raw numbers:
`baseline.csv`, `mf-baseline.csv`, `knobs.csv`, `arpack-tol.csv`, `gram-mixed.csv`,
`range-finder.csv` in this directory. Harness: `bench/RestrictedSolve.jl`
(+ `RestrictedSolveGram.jl`, `RestrictedSolveRangeFinder.jl`). Design:
`docs/design/Native-Core-Plan.md` "Phase 2 -- the restricted solve". Code:
`src/solvers/QuickDerN.jl`, `src/solvers/NullSolvers.jl`.

## The question

On `640x480xFx3` the restricted eigensolve is ~90% of the movie's cost
(`docs/CONTEXT.md`, "Movie regime measured per stage"), and the restricted
system grows with `F`. Four candidates were on the table, cheapest first:
(1) solver parameters on the matrix-free branch, (2) `GramSolver` in Float32
+ Float64 Rayleigh-Ritz on the dense branch, (3) a randomized range finder
for the (tiny) complement of the null space, (4) a block-diagonal Kronecker
preconditioner. The brief was to make the solve cheaper WITHOUT changing
what it returns, at sizes affordable on this machine: video-shaped random
tensors `160x120xFx3` Float32 for `F in {10, 30, 60}` and scrambled sphere
octants valence 3, `d in {100, 150, 200}` Float64.

**The first thing measured changed which candidate mattered.** `bench/jl
bench/RestrictedSolve.jl estimate` (restriction sizes alone, no tensor
built) shows all six cases take the DENSE route (`GramSolver`) under the
CURRENT production budget (`QDN_DENSE_BUDGET_BYTES = 2.5 GB`,
`QDN_GRAM_MIN_COLS = 1000`):

| case | dims | r | restricted rows x cols | dense GB | branch |
|---|---|---|---|---|---|
| video10 | 160,120,10,3 | 11,10,10,3 | 3300 x 3069 | 0.038 | dense (GramSolver) |
| video30 | 160,120,30,3 | 13,10,10,3 | 3900 x 3589 | 0.052 | dense (GramSolver) |
| video60 | 160,120,60,3 | 14,11,10,3 | 4620 x 4169 | 0.072 | dense (GramSolver) |
| sphere100 | 100,100,100 | 19,19,19 | 6859 x 5700 | 0.291 | dense (GramSolver) |
| sphere150 | 150,150,150 | 23,23,23 | 12167 x 10350 | 0.938 | dense (GramSolver) |
| sphere200 | 200,200,200 | 26,26,26 | 17576 x 15600 | 2.043 | dense (GramSolver) |

So candidate 1 (matrix-free solver tuning) needed
`Dleto.QDN_DENSE_BUDGET_BYTES[] = 0.0` forced to even be exercised, exactly as
the brief anticipated ("force the matrix-free branch where needed"); candidate
2 is the lever that already applies at these sizes with no forcing at all.

## Baseline: the code as found

`bench/jl bench/RestrictedSolve.jl baseline <case>` (production defaults,
whichever branch the budget picks), `mf-baseline <case>` (forced matrix-free,
`solver = :AutoSolver` -> ARPACK, `_qdn_default_free_solver()`):

| case | branch (as found) | seconds | applies | nullity | certified | residual | forced-mf seconds | forced-mf applies |
|---|---|---|---|---|---|---|---|---|
| video10 | dense/GramSolver | 0.53-1.22 | 0 | 3/3 | true | 2.26e-06 | 0.77-1.51 | 7392 |
| video30 | dense/GramSolver | 0.63-0.66 | 0 | 3/3 | true | 2.90e-06 | 1.00-1.02 | 8002 |
| video60 | dense/GramSolver | 0.77 | 0 | 3/3 | true | 1.50e-06 | 1.15-1.18 | 7992 |
| sphere100 | dense/GramSolver | 1.97 | 0 | 3/3 | true | 2.74e-14 | 9.53-9.91 | 36572 |
| sphere150 | dense/GramSolver | 8.98 | 0 | 3/3 | true | 3.50e-15 | 10.37-10.74 | 30138 |
| sphere200 | dense/GramSolver | 28.06 | 0 | 3/3 | true | 2.26e-14 | 18.42-18.68 | 35050 |

(Ranges are two runs each; `bench/jl` shares the machine with four other
agents' Julia jobs, so wall time is noisy -- applies is not, and is the
number the rest of this report leans on.) Two things worth registering
before the candidates: **for the video shapes the dense route is already
faster than forced matrix-free** (0.5-1.2s vs 0.8-1.5s), consistent with the
dense route being the intended fast path at sizes small enough to reach it.
**For the sphere, past d=100 forced matrix-free is faster than dense**
(sphere200: 18.7s matrix-free vs 28.1s dense) -- the dense route's own cost
is what candidate 2 targets.

## Candidate 1: solver parameters on the matrix-free branch

Forced matrix-free throughout this section (`QDN_DENSE_BUDGET_BYTES = 0`).
New plumbing to make it testable at all: `QuickDerMethod` gained
`solver_kwargs` (`src/solvers/QuickDerN.jl`), forwarded to `solve_nullspace`
on the matrix-free branch only -- `ncv`, `nv0`, `min_above` reach the solver
this way. `tol` does NOT: it is reserved (the constructor refuses it) because
it names two different things at two layers -- `solve_nullspace`'s own outer
relative gap-test ceiling (which `_qdn_solve_and_lift` sets explicitly) and
ARPACK's internal Ritz tolerance (a `kwarg` on `Dleto.solve(::ArpackSolver,
...)`) -- and the outer one always wins the name collision. ARPACK's own
`tol` is reachable only by calling `Dleto.solve` directly on the map
(`build_whitened_map`, below); through `derTrOpsReduced` or
`solve_nullspace` it is unreachable AT ALL, at any layer, for any caller.
That is itself a finding, not only a limitation of this harness.

### Solver comparison (each at its own defaults)

| case | ArpackSolver applies / s | KrylovSolver applies / s (status) | LSMRSolver applies / s (status) |
|---|---|---|---|
| video10 | 7392 / 0.83 | 18874 / 4.73 (ok) | **6,493,664 / 671.0 (capped, uncertified)** |
| video30 | 8002 / 1.02 | 18490 / 2.53 (ok) | not run (see below) |
| video60 | 7992 / 1.18 | 18586 / 2.69 (ok) | not run |
| sphere100 | 36572 / 9.63 | 95544 / 30.7 (**unconverged**) | not run |
| sphere150 | 30138 / 10.74 | 198326 / 158.0 (**unconverged**) | not run |
| sphere200 | 35050 / 18.42 | 209390 / 279.0 (**unconverged**) | not run |

**LSMRSolver is not competitive and was run once, not six times.** At
video10 it took 6.49 million applies and 671 seconds against ARPACK's 7392
and 0.83s for the SAME nullity (`solve_nullspace`'s escalation trace:
`[16, 32, 64, 128, 256, 512, 1024, 2048, 3069]` -- it ran the request all the
way to the FULL dimension of the restricted system before the bracket test
was satisfied, because the projection-based method never returns a value
clearly "above the cut" early). One measurement is the finding; repeating it
on sphere200 (5x the columns) would cost an unknown number of hours on a
machine four other agents are sharing, for a conclusion already reached.
Excluded from `run_solvers`'s default list; `solvers = (...,:LSMRSolver)`
still runs it if wanted.

**KrylovSolver's disadvantage GROWS with size and it stops certifying.**
2.5x more applies than ARPACK at video10, but 5.7x at sphere100, 6.6x at
sphere150, 6.0x at sphere200 -- and from sphere100 up it reports
`:unconverged`, clearing `certified` even though the computed residual is
fine (1.9e-10 to 2.4e-9). ARPACK-first stays the right default, exactly as
`bench/reports/2026-09-04/whitened/README.md` already found for LOBPCG.

### ncv, nv0, min_above (ArpackSolver, full pipeline with escalation)

| case | default (ncv auto) | ncv=32 | ncv=64 | ncv=256 | nv0=32 | min_above=4 |
|---|---|---|---|---|---|---|
| video10 | 7392 / 0.83 | **FAILS** | 7338 / 0.64 | 7654 / 0.86 | 10386 / 1.49 | 7392 / 0.80 |
| video30 | 8002 / 1.02 | **FAILS** | 8160 / 0.89 | 8520 / 1.08 | 10884 / 1.72 | 8002 / 0.97 |
| video60 | 7992 / 1.18 | **FAILS** | 8272 / 1.09 | 8542 / 1.29 | 10902 / 2.02 | 7992 / 1.18 |
| sphere100 | 36572 / 9.63 | **FAILS** | **FAILS** | 41260 / 9.39 | 29262 / 7.98 | 36572 / 9.39 |
| sphere150 | 30138 / 10.74 | **FAILS** | **FAILS** | 54370 / 20.60 | 30288 / 14.21 | 44086 / 18.24 |
| sphere200 | 35050 / 18.42 | -- | -- | 43778 / 26.35 | -- | -- |

(applies / seconds; sphere200's ncv=32/64 and nv0/min_above cells were not
re-run given the pattern was already clear at three smaller sizes and the
machine is shared -- ncv=256 was run to confirm the trend at the largest
size.) Two results, and both are the opposite of "bigger knob, better":

* **A FIXED `ncv` that is fine on the escalation loop's FIRST request can
  starve a LATER, bigger one and make ARPACK fail outright.** `ncv=32`
  succeeds at `nev=16` (the first request) and then the confirmation step
  doubles `nev` to `>= 32`; ARPACK needs `ncv > nev` strictly, and `32`
  clamped up to `33` is only marginally wider than `nev`, which starves the
  Krylov subspace and ARPACK exhausts `maxiter` (`XYAUPD_Exception`,
  `info = 1`) instead of converging. Reproduced directly (see "ARPACK's own
  `tol`, direct" below): the SAME map, single ARPACK call, `nev` forced
  to 32-64, fails identically. `ncv=64` is not even safe at sphere100/150,
  where escalation reaches `nev = 64` (see the default column's
  `nv_requests`, `arpack-tol.csv`) -- `ncv == nev` fails the same way.
  **A fixed `ncv` is a real hazard once the caller's request can grow; the
  policy already in `ext/DletoArpackExt.jl` (`ncv = 8*nev`, scaling with the
  request) is not a decoration, it is what keeps this failure mode from
  ever triggering, and no fixed override should replace it as a default.**
* **A LARGER `ncv` than the default is not free on this operator, and is
  usually worse.** On the sphere, `ncv=256` costs 13-80% MORE applies and
  8-43% MORE time than the default at every size tried (sphere100: +13%
  applies/-3%s -- noisy; sphere150: +80%/+92%; sphere200: +25%/+43%). On
  video it is roughly neutral (+3-7% applies). ARPACK's own docstring
  reports `ncv=8*nev` as "the flattest cost" on a DIFFERENT benchmark
  (`bench/reports/exp2-seeds.csv`); on the whitened restricted map of a
  scrambled sphere, the default is already at or past that flat point and
  widening the Krylov subspace further buys nothing but more Lanczos
  restart work.
* `nv0=32` (skip the first, small escalation request) costs MORE on video
  (+40-45% applies, since the true nullity is found at the default's first
  request already) but LESS on sphere100 (29262 vs 36572, -20%) -- there the
  default's `[16, 32, 64]` sequence is itself the cost, and starting at 32
  skips the wasted `k=16` attempt. Mixed; not adopted as a default (it made
  video worse by exactly the amount it helped sphere100, and sphere150 was a
  wash after the naturally larger request `[32, 64]` still needed).
* `min_above=4` (vs default 2) is neutral on video and on sphere100, and
  costs sphere150 an extra escalation round (`[16, 32, 64]` vs the default's
  `[16, 32]`) for the same answer -- not adopted.

**None of candidate 1's parameters are adopted as new defaults.** The
existing defaults (`ncv = 8*nev`, `nv0` from `initial_request`, `min_above =
2`, ARPACK as `_qdn_default_free_solver()`) were already at or near a local
optimum on every case measured; the experiments here mostly document WHY,
and one of them (`ncv=32`) documents a real hazard other callers should not
walk into by hand.

### ARPACK's own `tol`, direct on the SAME map

`Dleto.solve(Dleto.SOLVER_REGISTRY[:ArpackSolver], L; nv, ncv, tol)` called
directly on `build_whitened_map`'s `LinearMap` (one call, `nv = restricted
oracle + 4`, no escalation loop) -- the only route that reaches ARPACK's own
tolerance, since `solve_nullspace`'s outer `tol` shadows the name at every
higher layer:

| case | tol=1e-10 | tol=1e-6 | tol=1e-14 | ncv=128 (~default) | ncv=256 |
|---|---|---|---|---|---|
| video10 | 2790 / 0.22 | 2790 / 0.22 | 2790 / 0.22 | 2814 / 0.26 | 3292 / 0.35 |
| video30 | 3154 / 0.28 | 3154 / 0.28 | 3154 / 0.28 | 3220 / 0.35 | 3744 / 0.45 |
| video60 | 3070 / 0.33 | 3070 / 0.33 | 3070 / 0.33 | 3014 / 0.36 | 3748 / 0.51 |
| sphere100 | 13180 / 2.54 | 5544 / 1.04 | 13724 / 2.53 | 13080 / 2.43 | 19634 / 4.27 |
| sphere150 | 8188 / 2.44 | 6652 / 1.96 | 13912 / 4.12 | 8028 / 2.34 | 25492 / 9.17 |
| sphere200 | 16660 / 8.84 | -- | -- | -- | 24606 / 14.90 |

`tol` (applies / seconds) does nothing on video (the null cluster there sits
at the Float32 floor regardless), and on the sphere `tol=1e-6` (looser)
sometimes converges in FEWER applies (sphere100: 5544 vs 13180 at 1e-10,
sphere150: 6652 vs 8188) by accepting a cruder Ritz residual -- but ARPACK's
stopping test is already close to machine precision for a null eigenvalue
regardless of `tol` (per `ext/DletoArpackExt.jl`'s own docstring), so this is
noise in which Ritz values happen to cross ARPACK's OWN internal test at a
given iteration, not a controllable lever; `tol=1e-14` never converges FEWER
applies than `1e-10`, consistent with that. `ncv` again shows the same
"default is already near the flat point, wider is worse" pattern the
full-pipeline table found. **Not adopted**: no change to `ArpackSolver`'s
own default `tol = 1e-10`.

## Candidate 2: widen the dense/Gram route (ADOPTED)

`GramSolver` gained `gram_eltype` (`src/solvers/NullSolvers.jl`): stage 1
(the Gram `MᵗM`, its Cholesky, the shifted subspace iteration) runs in
`gram_eltype` when it narrows the matrix's own type; stage 2 (Rayleigh-Ritz,
`svd(M X)`) ALWAYS measures the ORIGINAL matrix `Mh`, never
`gram_eltype.(Mh)` -- the returned `vecs` are always the matrix's own type.
Measured directly with `Dleto.solve` on `build_dense_matrix`'s matrix
(`bench/RestrictedSolveGram.jl`), `:GramSolver` vs
`GramSolver(gram_eltype = Float32)`, `nv = restricted oracle + 8`:

| case | T | seconds f64 -> mixed | gram+cholesky s, f64 -> mixed | worst reldiff at oracle | worst reldiff overall |
|---|---|---|---|---|---|
| video10 | Float32 | 0.54 -> 0.13 | 0.24 -> 0.11 | 7.77e-07 | 1.21e-02 |
| video30 | Float32 | 0.61 -> 0.20 | 0.31 -> 0.17 | 3.74e-07 | 6.39e-03 |
| video60 | Float32 | 0.71 -> 0.30 | 0.40 -> 0.27 | 2.83e-07 | 8.56e-03 |
| sphere100 | Float64 | 1.73 -> 1.15 (1.50x) | 1.48 -> 0.84 (1.76x) | 2.58e-05 | 1.42e-04 |
| sphere150 | Float64 | 8.55 -> 4.67 (1.83x) | 8.00 -> 3.49 (2.29x) | 2.99e-05 | 1.16e-04 |

Video's `T` is already Float32, so `gram_eltype = Float32` is a no-op there
(`Tg === T`) -- the small differences shown are two independent random
subspace draws (no seed passed to this direct call), the honest "noise
floor" any two `GramSolver` calls differ by regardless of precision, and the
`worst reldiff overall` column is dominated by it. The sphere rows are where
`gram_eltype` does something: ~1.5-1.8x on total time, ~1.8-2.3x on the
Gram+Cholesky substages specifically, at a worst relative disagreement in
the near-null singular values of 2.6-3.0e-5 -- two orders of magnitude
looser than machine epsilon, but still four orders of magnitude tighter than
`sqrt(eps(Float64))`, and (below) it changes no certified answer.

**Verified through the real verdict machinery, not just the raw solver
call** (`solve_nullspace`, `:GramSolver` vs `GramSolver(gram_eltype =
Float32)`, seeded, on the sphere100/150 dense matrices):

| d | nullity (f64 / mixed) | certified (f64 / mixed) | rule (f64 / mixed) | principal angle cos |
|---|---|---|---|---|
| 100 | 13 / 13 | true / true | gap / gap | 1.00000000 |
| 150 | 13 / 13 | true / true | gap / gap | 1.00000000 |

Identical nullity, certification, rule, and subspace (principal angle cosine
1.0 to 8 digits) at both sizes. **Adopted as the new default** for the dense
branch's Float64 case, behind `Dleto.QDN_GRAM_MIXED_PRECISION[]` (default
`true`, a `Ref` so it is one line to turn off): `_qdn_solve_and_lift` now
builds `GramSolver(device = ..., gram_eltype = _qdn_gram_narrow(T))`, and
`_qdn_gram_narrow` maps `Float64 -> Float32` when the Ref is set and leaves
every other type (Float32 already, any complex type, untested) alone.
Pinned end-to-end in `test/TestQuickDerN.jl` section 10 (d=48 valence 3,
`QDN_GRAM_MIXED_PRECISION` true vs false through `derTrOpsReduced` itself)
and directly in `test/TestNullVerdict.jl` (`GramSolver(gram_eltype =
...)`: reproduces the unmodified solver when `nothing`, finds the same
subspace when Float32 narrows it, refuses `device = :gpu` together with
`gram_eltype`).

**sphere200 was not measured directly** -- `GramSolver()` and
`GramSolver(gram_eltype = Float32)` together, on a 17576x15600 matrix, do
not fit this machine's 6 GB per-process budget (measured: killed at 6.23 GB
mid-comparison). Extrapolating from the sphere100/150 ratios (~1.5-1.8x
total time), the dense route's 28.06s baseline would land around 16-19s --
close to forced-matrix-free's measured 18.4-18.7s at this size -- but that
is an extrapolation, not a measurement, and is reported as one.

## Candidate 3: randomized range finder for the complement (NOT adopted)

Prototype in `bench/RestrictedSolveRangeFinder.jl`: the SAME algorithm
`GramSolver` uses (shifted inverse power iteration on a random subspace,
Rayleigh-Ritz on the unsquared matrix) but with the inverse taken
MATRIX-FREE via `Dleto.shift_invert_map` (CG on `(LᵗL + shift I)`, never
forming the Gram) instead of a dense Cholesky -- candidate (c)'s literal
reading, "a randomized range finder for the complement, matrix-free."
Measured against `ArpackSolver` directly on the SAME map (`nv = restricted
oracle + 4`, one call, no escalation):

| case | range finder applies / s | ArpackSolver applies / s | principal angle cos |
|---|---|---|---|
| video10 | 36863 / 3.17 | 2814 / 1.41 | 1.000000 |
| video30 | 36863 / 3.48 | 3220 / 1.50 | 1.000000 |
| video60 | 36863 / 4.01 | 3014 / 1.52 | 1.000000 |
| sphere100 | 54474 / 6.34 | 13180 / 3.61 | 0.999995 |
| sphere150 | 54474 / 11.11 | 8188 / 3.61 | 4.15e-07 (see below) |
| sphere200 | 54474 / 16.45 | 16660 / 8.45 | 1.000000 |

The range finder costs 3-13x more applies than ARPACK at every size, and is
2-3x slower in wall time despite that cost being "only" CG iterations (an
apply here is far cheaper than a full ARPACK restart step) -- it simply
needs a lot more of them. **Why it costs this much is itself the finding,
and it was predicted before this prototype existed**: the applies count is
IDENTICAL across all three video sizes (36863) and across all three sphere
sizes (54474), regardless of the restricted system's actual dimension --
the signature of an inner solve that ALWAYS exhausts its iteration cap
(`cgmaxiter = 200`) rather than converging, so the cost here is fixed by
`(k, p, steps, cgmaxiter)` alone and never adapts to the problem. This is
exactly what `ext/DletoArpackExt.jl`'s own `ArpackSolver` docstring already
says about this approach: "Shift-invert is not offered on a matrix-free map:
the inner solve would be CG on `L + shift*I`, whose condition number is
`‖L‖/shift`... [and] costs more CG steps per application than the whole
`:SM` solve." Measured here rather than only asserted, and confirmed:
candidate 3 does not win.

The sphere150 row's principal angle (4.15e-07, i.e. orthogonal, an apparent
disagreement) is a comparison artifact, not a solver failure: at `nv = 17`
with NO escalation, ARPACK's single-shot call under-resolves sphere150's
13-dimensional near-null cluster (finds only 7 of it -- matching
`arpack-tol.csv`'s own `sphere150, ncv=auto, ...: below_1e-6 = 7` row), so
comparing 13 columns of each vector set puts 6 genuinely NONZERO,
essentially arbitrary ARPACK directions against the range finder's true null
ones. This is a limitation of the single-shot (no-escalation) comparison
methodology used for candidates 1's `arpack-tol` and candidate 3 alike, not
evidence against either algorithm; the properly-escalated numbers elsewhere
in this report (`mf-baseline.csv`, `knobs.csv`) are unaffected and are what
production actually runs.

## Candidate 4: block-diagonal Kronecker preconditioner (not built)

The design note's own question for this candidate was whether there is
anything left to precondition once the restriction is whitened -- and there
is not, by construction. Whitening (`docs/design/QuickDer-valence-n.md`
section 2a, `_qdn_whiten_axis`) makes every diagonal block of the restricted
Gram EXACTLY `c_a * I` (verified to 5e-16 in `test/TestQuickDerN.jl`,
"whiten: the restricted Gram has Kronecker diagonal blocks", already in the
suite before this report). A scalar multiple of the identity has condition
number 1 -- there is no anisotropy within an axis block for a
block-diagonal preconditioner to correct, and the cross-axis blocks (which
carry the actual derivation condition, per the same design note) are
exactly what whitening does NOT touch. Building a preconditioner whose job
is already done by construction would add code and a maintenance surface for
no measured gain; not attempted, and this is the argument for not
attempting it, not an oversight.

## Correctness

Every result above compares nullity, certification, and subspace (principal
angle) against the unmodified path on the SAME matrix/map -- the numbers in
each section's own table ARE the correctness check for that candidate.

Two new tests: `test/TestNullVerdict.jl` (`GramSolver(gram_eltype = ...)`:
matches the unmodified solver at `gram_eltype = nothing`, finds the same
subspace via principal angle at `gram_eltype = Float32` while its own
near-null values sit at Float32 precision rather than Float64's -- a real,
documented cost of narrowing stage 1 -- and refuses `device = :gpu` together
with a `gram_eltype`) and `test/TestQuickDerN.jl` section 10 (mixed
precision through the real `derTrOpsReduced` pipeline, `QDN_GRAM_MIXED_PRECISION`
true vs false, same nullity/certified/rule/span, both answers genuine
derivations by `der_residual`).

Full suite: `JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 bench/jl test/runtests.jl`
-- see the commit history on this branch for the result of that run.

## What did NOT help, in one place

* **LSMRSolver** on the matrix-free branch: 450-900x more applies than
  ARPACK, capped rather than converged, at video10 (one measurement, not
  repeated at larger sizes -- see candidate 1).
* **KrylovSolver** on the matrix-free branch: competitive in applies at
  video sizes (2.5x ARPACK) but not at sphere sizes (6-7x), and stops
  reporting `:ok` from sphere100 up even though its numbers are fine.
* **A fixed `ncv`** smaller than what escalation will eventually need
  (`ncv=32` at every size, `ncv=64` at sphere100/150): ARPACK fails outright
  once the request grows past it.
* **A larger `ncv` than the default** (`ncv=256`): costs more applies and
  more time on the sphere at every size tried; roughly neutral on video.
  The default (`8*nev`) is already at or past the point where widening helps.
* **ARPACK's own `tol`**: unreachable from any layer above
  `Dleto.solve(::ArpackSolver, ...)` itself (shadowed by
  `solve_nullspace`'s outer `tol`), and moving it directly changes nothing
  systematic once reached.
* **`nv0` and `min_above` off their defaults**: mixed results that cancel
  across cases; not worth a default change.
* **The randomized range finder (candidate 3)**: matrix-free shift-invert's
  inner CG never converges on this operator's condition number and pays for
  it in iterations that do not shrink with problem size while applies do
  not adapt to it -- exactly what was already documented as the reason
  ARPACK does not offer shift-invert here.
* **The Kronecker preconditioner (candidate 4)**: whitening already leaves
  nothing to precondition; not built.

## Reproduce

```
bench/jl bench/RestrictedSolve.jl estimate
bench/jl bench/RestrictedSolve.jl baseline <case>              # as found
bench/jl bench/RestrictedSolve.jl mf-baseline <case>            # forced matrix-free
bench/jl bench/RestrictedSolve.jl solvers <case>                # Arpack vs Krylov (vs LSMR: pass solvers=...)
bench/jl bench/RestrictedSolve.jl knobs <case> <solver> <ncv> <nv0> <min_above>
bench/jl bench/RestrictedSolve.jl arpack-tol <case> <ncv> <tol>  # direct on the same map
bench/jl bench/RestrictedSolve.jl gram-mixed <case>              # candidate 2
bench/jl bench/RestrictedSolve.jl range-finder <case>            # candidate 3
bench/jl bench/RestrictedSolve.jl sweep <case>                   # everything for candidate 1 on one case
```
`<case>` is one of `video10`, `video30`, `video60`, `sphere100`, `sphere150`,
`sphere200`. Every task forces `QDN_DENSE_BUDGET_BYTES = 0` where the brief
calls for the matrix-free branch and restores it afterward; `estimate` and
`baseline` do not force anything.
