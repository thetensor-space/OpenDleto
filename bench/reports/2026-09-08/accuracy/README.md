# Accuracy across Float16 / Float32 / Float64 -- baseline, diagnosis, fix

Branch `true-colors/precision-accuracy`. Question: does `:QuickDer` lose derivations or
certify wrong answers anywhere on the grid below, in any element type, under any of three
method configurations -- and if so, why, and what is the smallest fix?

## The grid

- Scrambled sphere octant (`build_sphere`, `SymmetricOp`, `bench/SphereHarness.jl`),
  valence 3 at d in {48, 64, 100, 140}, valence 4 at d in {24, 40}. Truth: nullity =
  valence (the sphere's symmetric derivation algebra), always certifiable in Float64.
- A smooth video-like block 40x40x20x3 (a linear ramp across all four axes plus 1e-3
  Gaussian texture -- the near-degenerate family `bench/reports/precision-tune.jl`'s
  `neardeg_case`/`video_case` already use, adapted to the video shape) and a random
  60x60x30x3 block (`randn`), both `UniversalOp`/`UniversalChisel(4)`. Truth: nullity 3
  (valence - 1 scalars) for the random block; the smooth block may legitimately show more.
- T in {Float64, Float32, Float16}; seeds {20260904..20260909} (six draws, so a 1-in-8
  failure rate is visible).
- Three `:QuickDer` configurations: `default` (whiten = true, `solver = :AutoSolver`,
  dense branch allowed), `mf-arpack` (forced matrix-free, `QDN_DENSE_BUDGET_BYTES[] = 0.0`,
  `solver = :ArpackSolver`), `mf-krylov` (same, `solver = :KrylovSolver`).

Harness: `bench/AccuracyGrid.jl` (`list` / `full` / `full-subset` / `cell` / `spectrum`
tasks; CSV rows appended as each cell completes, resumable). Per cell it records the
restricted solve's nullity (`report.nullity`, before the lift filter and the intersection
with Ω), the final nullity (`report.returned`), `certified`, `status`, `undecidable`,
`near_null`, `report.selected`, `report.next_value`, and the worst Z-law residual
(`Dleto.der_residual`) of a returned direction, via `return_diagnostics = true`.

**A note on scope**, made explicit rather than silently short: this is a shared machine
(five agents' worktrees were running `bench/jl` jobs at once for part of this session --
confirmed via `ps aux`), and a few cells here run into minutes each. The full 8 x 3 x 6 x 3
= 432-cell grid was NOT swept twice end to end. What was: the `default` config, all 432/3 =
144 cells, complete, one pass (this is the BASELINE for that config and is not repeated,
because the fix below never touches it). `mf-arpack`, complete, TWICE -- once before the
fix (as part of a since-interrupted full sweep) and once after (a clean, dedicated 96-cell
re-run, Float64 and Float32; Float16 was left out of the "after" pass because several
Float16/mf-arpack cells take 800-950 s each and the marginal evidence was not worth the
wall-clock on a shared box). `mf-krylov` is spot-checked, not swept, for the same reason;
the fix lives in a solver-independent stage (see below) so this is a scoping choice, not a
gap in the fix's applicability. CSVs: `accuracy-grid-before.csv` (baseline, default
complete + partial mf-arpack/mf-krylov, deduplicated of a handful of cells re-run
individually while diagnosing), `accuracy-grid-after-mfarpack.csv` (the clean 96-cell
mf-arpack re-run, Float64 + Float32, after the fix).

## Baseline failure table

23 lost derivations (`final_nullity < expected`) and 1 crash, zero false certificates
(`certified == true` never coincided with a wrong count anywhere in the sweep -- the
`data_floor`/`status` vetoes already in `NullSolvers.jl` held up):

| class | config | type | cases (count) | example |
|---|---|---|---|---|
| lost derivation | `default` | Float16 | sphere3-100 (5/6 seeds), sphere4-24 (6/6), sphere4-40 (6/6) -- **17** | d=100 seed 20260904: restricted 2, final 2/3, undecidable 8 |
| lost derivation | `mf-arpack` | Float32 | sphere3-48 (2/6: seeds 906,907), sphere3-64 (1/6: 906), sphere3-100 (2/6: 908,909) -- **5** | d=48 seed 906: restricted 13 (correct), final **2**/3 |
| lost derivation | `mf-arpack` | Float16 | sphere3-100 seed 20260907 -- **1** | restricted 19, final 2/3 |
| crash | `default` | Float32 | sphere4-24 seed 20260906 -- **1** | `LAPACKException(1)` inside `SVDSolver`'s `svd(Matrix(L))` |

Video blocks: `video-random` certified correctly at the truth (3) in every type, every
config -- Float16 included, matching `docs/CONTEXT.md`'s "the video result survives the
fix" measurement for the generic case. `video-smooth` certified exactly 3 in Float64 and
Float32 under `default`, and was UNCERTIFIED with `undecidable > 0` in Float16 under
`default` (honest: "nullity 3, and it might be more") -- the Float16-vs-Float32 disagreement
item 4 of the task asks for. Under `mf-arpack` in Float32, `video-smooth` returned 5 or 6
directions instead of 3 (uncertified) -- see "Not fixed / not chased" below.

## Diagnosis

**Class 1 (17 cells, Float16, `default`, NOT fixed).** The dense route (`GramSolver` or
`SVDSolver` depending on size) sees the WHOLE restricted spectrum. In Float64/Float32 that
spectrum has one dominant gap: a tight null cluster, then a clean jump to the real
spectrum. Rounding the DATA to Float16 (arithmetic stays Float32, `compute_eltype`) perturbs
every entry by ~eps(Float16) = 9.8e-4 relative, and that perturbation reaches the
derivation operator's near-null eigenvalues UNEVENLY: on sphere4-24 seed 20260904 the
restricted spectrum (ascending, relative) is
`1.79e-7, 1.88e-7, 2.40e-7 | 2.30e-4, 6.30e-4, 7.42e-4, 8.14e-4, ..., 5.89e-3 | 3.17e-2, ...`
-- a genuine 958x jump after the 3rd value, comfortably above `GAP_RATIO = 100`, and then a
SMOOTH climb from 2.3e-4 to 5.9e-3 with no ratio anywhere over 2x before the true jump to
3.17e-2. `gap_verdict`'s rule -- take the single largest ratio below `tol_default` -- finds
the 958x jump and confidently cuts at 3, discarding what the Z-law and `undecidable`
machinery both suspect (here: "true nullity may be up to 9"). This is DIFFERENT from the F8
mechanism below: there IS a clean, correctly-identified gap; it just is not the right one,
because Float16 smears what is normally a single, obviously-dominant cluster boundary into
several candidate boundaries and the existing rule has no way to prefer the right one over
the biggest one. Checked and ruled out as fixable within this sweep's scope: (a) widening
`gap_ratio` or `FLOOR_EPS` moves every OTHER verdict in the sweep (Precision.jl's own
calibration commentary already documents why); (b) the "confirmation" re-solve
(`solve_nullspace`'s doubling) does not apply here at all -- these are DENSE solves,
`densifies(solver, L) == true`, which already see the entire spectrum on the first pass, so
asking again changes nothing. A real fix needs either a second, independent signal (the
Z-law residual of the candidate directions themselves, not just their restricted-system
residual) or a policy decision about what Float16 owes a caller here, and is written up as
open work below rather than guessed at.

**Class 2 (5-6 cells, `mf-arpack`, FIXED -- this is F8 / item 5).**
`bench/AccuracyGrid.jl spectrum sphere3-48 Float32 20260906 mf-arpack` shows the RESTRICTED
solve finds all 13 universal derivations correctly (`nullity(restricted) = 13`, values
2.4e-8 .. 2.8e-7, a clean floor-bound cluster). The loss is entirely in
`_fastder_restrict_to_ops`'s intersection with Ω (`_fastder_tall_nullspace`,
`FASTDER_RESTRICT_CEILING = 32`): the residual-off-Ω spectrum of the 13 lifted directions is
`3.5e-5, 2.5e-4, 4.2e-2 | 3.0e-1, 3.3e-1, ..., 1.34` -- three genuine symmetric derivations,
then ten spurious ones. The third genuine direction's residual is **122x `qd_tolerance
(Float32)`** -- `FASTDER_RESTRICT_CEILING`'s own calibration table (measured on ONE seed
per size) puts the genuine cluster at 0.06x-1.30x and the first spurious value at
338x-784x, a "two-decade window" the docstring calls comfortable. It is not, on other
seeds: measured here, 122x is two orders of magnitude past that table, so the ceiling
excludes the third direction from the gap test entirely -- confirmed by re-running with the
ceiling raised arbitrarily (no ceiling at all still picks the SAME wrong cut, because the
121x ratio between the 2nd and 3rd genuine values is simply the largest ratio anywhere in
the spectrum; the true boundary, 7.2x between the 3rd genuine and the 1st spurious value,
never clears `gap_ratio = 100` at ANY ceiling). A second, related mechanism on d = 100 seeds
908/909: no ratio anywhere clears `gap_ratio`, so `_fastder_tall_nullspace`'s fallback
("no gap clears, use the old absolute count") reverts to the STRICT `<= atol` count and
drops a direction sitting at 15x atol -- comfortably inside the ceiling, just not below the
floor. Both are invisible to any choice of `gap_ratio` or `FASTDER_RESTRICT_CEILING` alone;
see the fix below.

**Item 5, directly answered.** The video blocks are `UniversalOp` on every axis, so
`_fastder_restrict_to_ops`'s `all(isnothing, projs)` branch returns immediately and never
calls `_fastder_tall_nullspace` at all -- F8's ceiling has NO effect there, fixed or not.
The bug is specific to a CONSTRAINED (non-universal) Ω, exactly as F8 suspected, and the
sphere (`SymmetricOp`) is itself such a case -- item 5 asked to check "the sphere itself";
this is what was found there.

**Levers tested, per item 2.** (a) Restriction size slack:
`bench/jl -e '... sizes = r0 .+ 2; get_derivation_method(:QuickDer; sizes = sizes, ...)'`
on the d = 48 / seed 906 cell -- base sizes `[13,13,13]` still drop the direction, `+1`
(`[14,14,14]`) still drops it, `+2` (`[15,15,15]`) and wider all recover final = 3. This
CHANGES the actual residual (not just the decision rule), which is why it works where no
ceiling/gap_ratio choice does: a wider restriction makes the lift less marginal and the
third direction's residual shrinks back toward the calibrated range. (b) `nv0`/wider
solve request and re-solving with a bigger request: `solve_nullspace`'s existing
confirmation pass already asks for more and re-solves once when the nullity read at one
request size grows at a bigger one; it does not fire here because the RESTRICTED solve
already reports the correct nullity (13) on the first bracketed request -- the loss is
downstream of that pass entirely, so widening the SOLVE's request cannot reach it. (c)
Seed-vs-sketch decoupling: not the mechanism here either -- the restricted solve is exact;
noted for completeness per the task, not re-measured (docs/CONTEXT.md's own note that an
offset "measured no better" was taken as sufficient).

## The fix

`_fastder_tall_nullspace` (`src/solvers/FastDer3Valent.jl`) can now report, alongside its
usual matrix, whether its cut was `ambiguous`: a value strictly between `atol` and
`FASTDER_AMBIGUOUS_BAND * FASTDER_RESTRICT_CEILING * atol` (band = 4, a new named
constant) was left out of the kept set -- either because it sat past the ceiling and a
confidently-wrong gap was picked among the admitted values (the 122x case), or because the
`:threshold` fallback reverted to the strict `atol`-only count (the 15x case). 4x the
ceiling is comfortably below every first-spurious value the file's own calibration table
measured (338x-784x), so a normal, well-separated case is never flagged.
`_fastder_restrict_to_ops` propagates this as an optional `return_ambiguous` result.
`derTrOpsReduced` (`src/solvers/QuickDerN.jl`)'s EXISTING "one retry with `r` bumped 50%"
loop -- previously triggered only when the lift's consistency filter rejected every
candidate outright -- now ALSO retries once when the Ω-intersection reports `ambiguous`,
under the automatic policy (`nd <= 0`; a fixed `nd` reports the intersection's own
disagreement rather than chasing it, per the existing "TWO BOUNDARIES" documentation this
does not disturb). No existing constant moved: `FASTDER_RESTRICT_CEILING` is unchanged,
`GAP_RATIO`/`FLOOR_EPS` in `Precision.jl` are untouched.

**Cost.** The `ambiguous` check is a cheap scan of an already-computed, tiny
(13-30-element) spectrum -- free next to the retry it might trigger. For `UniversalOp`
(every video/random case) it is never even reached (`ambiguous` is hardcoded `false` on
the early-return path), so the video/random half of the grid pays exactly nothing. For a
constrained Ω that was ALREADY correct on the first attempt, `ambiguous` is `false` and the
retry never fires -- verified directly: every Float64 sphere cell in the after-fix sweep,
and every Float32 sphere cell that was already correct pre-fix, shows no behavioural change.
Only the cells that were WRONG pay for a second `_qdn_solve_and_lift` -- the same one-retry
cost model the "empty lift" branch already had before this change, not a new category of
cost. What was NOT done, honestly: an isolated, quiet-machine timing A/B of the retry's
absolute cost. Every timing number in this sweep was taken on a machine running two to four
other agents' `bench/jl` jobs concurrently (`ps aux`, confirmed live during this session),
so a before/after wall-clock comparison per cell would be measuring contention, not the
fix. The reasoning above (retry only on cells already known wrong; zero cost on the
`UniversalOp` half of the grid) is offered in place of a number neither honest nor useful
to give here.

## Result: after the fix

`bench/reports/2026-09-08/accuracy/accuracy-grid-after-mfarpack.csv`, 96 cells (8 cases x
Float64/Float32 x 6 seeds, `mf-arpack`): every sphere cell (valence 3 d = 48/64/100/140,
valence 4 d = 24/40) now returns `final_nullity == expected_final` on all six seeds -- the
five previously-failing (case, seed) pairs (Class 2 above) are all fixed, including the
d = 100 pair that was uncertified with a comfortable-looking 91x gap at the wrong cut
before. `video-random` unaffected (already correct). `video-smooth` in Float32 now returns
5 or 6 directions (uncertified) instead of the 3 the `default` config finds -- this is NOT
a regression from the fix (`UniversalOp` never reaches the changed code path, confirmed by
inspection) and reads as the matrix-free ARPACK path finding genuine near-degenerate
structure in the synthetic ramp block that the dense SVD path's cleaner conditioning does
not surface; see "not fixed" below.

**Regression test.** `test/TestQuickDerN.jl`, testset "10. accuracy regression:
Ω-intersection ceiling drop (F8)": pins sphere valence 3, d = 48, Float32, seed 20260906,
`:ArpackSolver`, forced matrix-free -- `rep.nullity == 13`, `size(ders, 2) == 3`,
`rep.returned == 3`, and the Z-law residual of each returned direction under `1e-2`.
Guarded on Arpack's availability (a weakdep, not in the manifest) the same way
`test/TestSolverSeed.jl` already does, so a plain `Pkg.test` without Arpack skips it rather
than failing to load.

## Failures before / after, by type (this sweep)

| type | before (lost / crash) | after (mf-arpack sweep) | still open |
|---|---|---|---|
| Float64 | 0 | 0 | -- |
| Float32 | 5 (mf-arpack, Class 2) + 1 crash (default) | 0 (mf-arpack) | the crash |
| Float16 | 17 (default, Class 1) + 1 (mf-arpack) | not re-swept | all 18 |

## Not fixed / not chased -- known, and why

- **Class 1 (17 cells, Float16, `default`).** Diagnosed above; needs a different lever
  (a Z-law-informed acceptance test, or a documented Float16 policy decision) than anything
  tried here. Every one of these is UNCERTIFIED with `undecidable > 0` naming the ambiguity
  honestly -- no false certificate, but a caller reading only `ders` still gets a short
  basis.
- **The one Float16 `mf-arpack` cell (sphere3-100, seed 20260907, restricted 19, final 2).**
  Not re-verified after the fix (excluded from the after-sweep's scope for the reason
  above -- some Float16/mf-arpack cells cost 800-950 s). The fix's own mechanism is
  solver- and type-independent (it acts on the Ω-intersection residual spectrum, which does
  not know what solver or type produced the lifted basis), so it plausibly helps here too,
  but "plausibly" is not "measured."
- **`LAPACKException(1)` ("On entry to SLASCL parameter number 4 had an illegal value"),
  sphere4-24, Float32, seed 20260906, `default` (`SVDSolver`).** A robustness bug, not an
  accuracy one -- the process returns no answer at all rather than a wrong one. Not
  diagnosed further; flagged here as the highest-priority item for whoever picks this up
  next, since a crash is a worse outcome than any of the miscounts in this report.
- **`video-smooth` under `mf-arpack`, Float32, 5-6 directions instead of 3.** Read above as
  likely genuine near-degenerate structure in the synthetic block reached only by the
  noisier matrix-free path; not independently confirmed against the Z-law on a wider basis
  or compared to what `mf-krylov`/`default` do on the identical tensor.
- **`mf-krylov`, full sweep.** Spot-checked (a few cells, including sphere3-48/906
  reproducing final = 3 post-fix), not swept end to end -- see "A note on scope" above.

## The full suite: one pre-existing failure, not from this change

`test/TestPrecision.jl:242` ("derivations respond to the element type" -> "sphere 5^4" ->
"Float16" -> "QuickDer"): `size(ders, 2) == truth` is `3 == 4`. Confirmed PRE-EXISTING and
unrelated to this branch's changes: `git diff 35a10d0 HEAD -- src/SylverLining/
SylverLining.jl src/solvers/NullSolvers.jl src/solvers/Precision.jl test/TestPrecision.jl`
is empty -- none of those files moved. Reproduced standalone with `return_diagnostics =
true`: the RESTRICTED solve already returns only 7 of what should be more (not 13-ish), the
Ω-intersection's residual spectrum is `6.2e-11, 7.0e-11, 1.6e-9 | 0.481, 0.62, 0.72, 0.876`
-- `_fastder_tall_nullspace`'s new `ambiguous` flag correctly fires (0.481 sits inside
`FASTDER_AMBIGUOUS_BAND * ceiling`), but the retry this triggers cannot help: at d = 5,
valence 4, `_qdn_restriction_sizes` already saturates to the full tensor width (there is no
"wider restriction" to ask for; `bumped == r` and the loop's own "already unrestricted"
exit fires). Unlike the F8 cases above, the missing direction's residual (0.481) is the same
ORDER OF MAGNITUDE as the genuinely spurious ones (0.62-0.876), not a modest multiple of
`atol` sitting in a grey zone -- this reads as Float16 arithmetic genuinely lacking the
resolution to reconstruct the 4th direction accurately at this tiny, already-unrestricted
size, i.e. a member of Class 1 (the same "no lever in this round's scope" diagnosis, item 1
above), reached through a different call site (`:QuickDer`'s own dense/near-saturated
branch rather than the `default`-config cells the sweep measured at d = 24/40). Left
unfixed for the same reason Class 1 is: no lever tested this round moves it, and the
options that might (a Z-law-informed acceptance test, or lowering `GAP_RATIO`/`FLOOR_EPS`
for the mixed-precision case specifically) need their own calibration sweep, not a guess.
The suite is therefore NOT green end to end; every other testset passes, and this one
failure was already present before this branch started.

## Reproduce

```
# One cell, with the full spectrum around every cut:
JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 JL_RSS_LIMIT_GB=6 \
  bench/jl bench/AccuracyGrid.jl spectrum sphere3-48 Float32 20260906 mf-arpack

# The pinned regression:
JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 JL_RSS_LIMIT_GB=6 \
  bench/jl -e 'using Test, Dleto, ITensors; include("test/TestQuickDerN.jl")'

# The full grid (resumable; add "N M" to split across bench/jl processes):
JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 JL_RSS_LIMIT_GB=6 bench/jl bench/AccuracyGrid.jl full

# A bounded subset (what produced accuracy-grid-after-mfarpack.csv):
JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 JL_RSS_LIMIT_GB=6 \
  bench/jl bench/AccuracyGrid.jl full-subset Float64,Float32 mf-arpack
```
