# The sigma_(e+1) gap verdict

`src/solvers/NullSolvers.jl`: `NullVerdict`, `gap_verdict`, `GAP_RATIO`,
`FLOOR_EPS`, and the nullity decision inside `solve_nullspace`.

## Problem

`solve_nullspace` used to decide nullity with one fixed relative threshold,
`max(tol, 100*eps(T)) * ‖L‖`: count how many returned values fall below it.
`krylov-calibration.md` and `precision-study.md` showed this has no right
value on the sphere stratification benchmark (true nullity exactly 3,
`bench/SphereHarness.jl`): the fourth eigenvalue relative to the top wanders
from `2.5e-3` down to `8e-5` across seeds at `d = 32`, trends to `~1e-4` at
`d = 45..50`, and at `d = 50`, seed 50, a genuine near-derivation sits at
`1.2e-8` relative. `tol = 1e-6` gave nullity 4 for every solver at every seed
(too loose). `tol = 1e-8` gave nullity 3 in Float64 -- but Float32, whose own
null values sit at `~6e-7` relative (its noise floor is `eps(Float32) * ‖L‖ ~
1.2e-5`), *also* reports 3, for the wrong reason: the fixed cut throws away
information that is actually present in the spectrum's shape.

## Design

`gap_verdict(vals, scale; threshold, floor, gap_ratio, nd, requested)`:

1. Sort `|vals|`, divide by `scale` (an operator-norm estimate) to get a
   RELATIVE spectrum.
2. Floor every value at `floor` (default `FLOOR_EPS * eps(T)`, relative) --
   below this a value is arithmetic noise, not a measurement.
3. Consider cuts `k = 1 .. min(below_thr, n-1)`, where `below_thr` is how many
   values are strictly below `threshold` (the old fixed rule, still a
   CEILING -- see below). For each candidate cut, take the ratio
   `rel[k+1] / max(rel[k], floor)`.
4. Take the cut with the LARGEST ratio. If that ratio clears `gap_ratio`,
   declare nullity `k`, `rule = :gap`, `certified = true`.
5. Otherwise fall back to the old fixed count (`below_thr`), `rule =
   :threshold`, `certified = false` (except nullity 0, certified when the
   smallest value already clears `gap_ratio` above `threshold`).
6. `nd > 0` bypasses all of this: `rule = :fixed`, the threshold count capped
   at `nd`, never certified.

**Why a ceiling, not just a gap.** A gap alone cannot tell "zero" from "the
smallest thing this operator happens to have": a value at `1e-8` looks like a
huge jump from the floor whether it is a genuine near-derivation or the
entire nonzero spectrum of some other, unrelated operator. Restricting
candidate cuts to `k <= below_thr` keeps `tol`'s old meaning -- an upper bound
on what "null" can mean -- and lets the gap only REFINE that count, never
exceed it. Concretely, this is what keeps a smoothly-graded spectrum with no
real null cluster (`bench/reports/gap-verdict.md` test "no dominant gap") from
being cut wherever the largest of many unremarkable ratios happens to fall
outside the threshold.

**`NullVerdict`** carries the decision back to the caller: `nullity`, `rule`,
`certified`, the `gap` found and the `gap_ratio` required, the `floor` and
whether the cut was `floor_binding` (the last value counted null sits *below*
the floor -- routine for a clean Float64 null cluster, and the informative
signal for Float32), `near_null` (values below `threshold` but above the cut
-- the near-derivations a fixed rule would have swallowed), a few `below`/
`above` values for a human to eyeball, the full `spectrum`, `scale`, and
`requested`.

`solve_nullspace` returns `(; vals, vecs, verdict)`. This still destructures
as `(vals, vecs) = solve_nullspace(...)` (Julia's tuple-destructuring assigns
from an iterable and simply ignores the rest), so both existing callers
(`src/Densors.jl`, `src/SylverLining/SylverLining.jl`) are unchanged. Logging:
an `@warn` (one-line, `maxlog=1`) when the verdict is uncertified; an `@info`
(also `maxlog=1`) when it IS certified but `floor_binding` -- the routine
Float64 case, downgraded from warning to note because raising it every call
would be pure noise, but still on the record.

## Why `GAP_RATIO = 100`

With the floor at `100*eps(T)` relative:

- **Float64.** The null cluster sits at `1e-19..1e-13` relative (measured
  below), so it floors to `2.2e-14`. The first nonzero eigenvalue ranges from
  `1.2e-8` (the near-degenerate `d=50`, seed 50) up to `1.1e-2`. Gaps
  (floored-null-cluster to first nonzero) measured `1.3e10..1.7e11` at the
  ordinary seeds, and `5.4e5` at the one near-degenerate case -- five to seven
  orders of magnitude above `100`.
- **Float32.** The null cluster sits at `3e-8..3e-7` relative -- *under* the
  `1.19e-5` floor -- so every clean case floors identically and the gap to the
  first nonzero eigenvalue (`2.5e-3..1.1e-2` at `d<=40`) comes out at
  `200..300`: just over `100`, comfortably certified. At `d=50` the first
  nonzero eigenvalue has drifted down to `5e-5..3e-4` and a near-derivation
  sits at `8e-8..2e-7` -- almost exactly at the floor -- so the floored gap
  collapses to `4.6` and `25.3`: both **fail** to clear `100`, and the verdict
  correctly falls back to `:threshold`, uncertified.

`100` sits comfortably below every ordinary Float64 or Float32 gap (by two to
nine decades) and above nothing worth confusing it with -- the smallest
"real" gap in this data is Float32's `207` at `d=32`. It also sits well above
what a `:threshold`-rule fallback could produce by accident: two Float32 runs
above (`d=50`) land at `4.6` and `25.3`, an order of magnitude short, so `100`
does not need to be tuned finer than "the nearest power of ten with headroom
on both sides." A much larger ratio (`1e3`+) would have no effect on the
Float64 cases (their gaps are enormous regardless) but would push the
ordinary Float32 `d<=40` cases (`200..300`) into the same uncertified bucket
as the genuinely marginal `d=50` ones, discarding exactly the distinction the
gap test exists to make.

## Validation

`julia -t 2 --project=. -e 'using KrylovKit, IterativeSolvers, Arpack;
include("bench/SphereHarness.jl")'`, then for each `(d, seed)` and element
type, `run_stratify(inp; solver=:ArpackSolver)` plus a direct
`Dleto.solve_nullspace(Dleto.sylvesterLM(inp.Ω, inp.ch, inp.Γ)[1],
:ArpackSolver)` for the verdict. True nullity is 3 at every case.

| d  | seed | T       | nullity | lsq_err  | rule      | certified | gap      | floor-bound | note |
|----|------|---------|---------|----------|-----------|-----------|----------|-------------|------|
| 32 | 32   | Float64 | 3       | 2.8e-14  | :gap      | yes       | 1.1e11   | yes         | |
| 32 | 32   | Float32 | 3       | 2.1e-5   | :gap      | yes       | 207      | yes         | |
| 32 | 1032 | Float64 | 3       | 1.7e-14  | :gap      | yes       | 1.7e11   | yes         | |
| 32 | 1032 | Float32 | 3       | 1.5e-5   | :gap      | yes       | 323      | yes         | |
| 40 | 40   | Float64 | 3       | 2.5e-13  | :gap      | yes       | 1.2e11   | yes         | |
| 40 | 40   | Float32 | 3       | 1.5e-4   | :gap      | yes       | 217      | yes         | |
| 50 | 50   | Float64 | **3**   | 4.8e-13  | :gap      | **yes**   | 5.4e5    | yes         | near-derivation at `1.2e-8` relative flagged, `near_null=1` |
| 50 | 50   | Float32 | 4       | 1.5e-4   | :threshold| **NO**    | 4.6      | yes         | **UNCERTIFIED** -- correctly refuses to certify a silent 4th vector |
| 50 | 1050 | Float64 | 3       | 1.0e-13  | :gap      | yes       | 1.3e10   | yes         | |
| 50 | 1050 | Float32 | 3       | 1.1e-4   | :threshold| NO        | 25.3     | yes         | gap present but under `gap_ratio`; count still happens to be right, flagged anyway |

The `d=50`, seed 50 pair is the case this feature exists for. In Float64 the
verdict is `NullVerdict(nullity=3, rule=:gap, certified, gap=5.39e5,
floor-bound, near-null above cut: 1; below=[8.6e-17, 1.0e-16, 2.5e-16],
above=[1.2e-8, 5.4e-5, 1.0e-3])` -- exactly nullity 3, with the near-derivation
at `1.2e-8` visible in `above` and counted in `near_null`, not folded into the
answer. In Float32 the same operator gives `NullVerdict(nullity=4,
rule=:threshold, UNCERTIFIED, gap=4.6, floor-bound; below=[8.4e-8, 1.4e-7,
2.1e-7], above=[5.5e-5, 1.0e-3, 2.8e-3])`: the old fixed threshold alone would
have silently reported nullity 4 (as it used to, for every `tol` tried in the
original study); the gap verdict still reports 4 as its best guess but marks
it uncertified and (per `solve_nullspace`, `maxlog=1`) emits

```
Warning: null solve: nullity 4 is UNCERTIFIED -- no gap of 100.0x in the
spectrum below the threshold 1.19e-5 (relative). Values around the cut:
[8.4e-8, 1.4e-7, 2.1e-7] | [5.5e-5, 1.0e-3, 2.8e-3]; the value below the cut
is under the Float32 precision floor 1.19e-5, so a near-derivation there
cannot be told from zero. Consider Float64, an explicit `tol`, or `nd`.
```

For comparison, the certified-but-floor-bound `@info` (the routine Float64
case) reads:

```
Info: null solve: nullity 3 certified (rule :gap), cut at the Float64
precision floor 2.22e-14 (relative) -- a near-derivation hidden below that
floor would look identical. Values around the cut: [2.6e-17, 1.9e-16,
3.4e-16] | [2.5e-3, 8.0e-3, 9.1e-3]
```

(Both examples show only one occurrence in the log: `maxlog=1` caps repeats
of the same log statement across the run, so of the eight cases above that
qualify for the `@info`, only the first is actually printed. This is a
pre-existing convention used for the old `@warn`, not new here.)

## Tests

`test/TestNullVerdict.jl` (wired into `test/runtests.jl`):

- `gap_verdict` directly, on hand-built relative spectra: a clean Float64
  nullity-3 cluster; a near-derivation below `tol` that is NOT swallowed
  because the null-cluster-to-derivation gap dominates the
  derivation-to-real-eigenvalue gap (the `d=50`/seed-50 shape, reproduced
  synthetically); a smoothly graded spectrum with no dominant gap, falling
  back to `:threshold` uncertified; nullity 0 certified; `nd > 0` forcing
  `:fixed`; a Float32 near-derivation sitting below the Float32 floor.
- `solve_nullspace` end-to-end on a `Diagonal` `LinearMap` via `:SVDSolver`
  (no optional extension needed): the `(vals, vecs, verdict)` contract, the
  legacy `(vals, vecs) = solve_nullspace(...)` destructuring, and a
  `Test.collect_test_logs` check that the Float32 floor-bound/certified case
  emits an `@info` (not a `@warn`) naming the precision floor.

`julia -t 2 --project=. -e 'using Pkg; Pkg.test()'`: **`Testing Dleto tests
passed`**, exit code 0. Every testset green, `TestNullVerdict` included:

```
gap_verdict                                                  |   22    22   0.1s
solve_nullspace end-to-end (diagonal LinearMap, :SVDSolver)  |   17    17   1.2s
```

alongside the pre-existing suite (`realCanonicalForm` 4500, `Chisel` 1280,
`Operators` 190, `TransverseOpsIndependant` 900, `TransverseOpsSymmetires`
1340, `Testing Tensor Synthesis` 32, `FastDer3Valent` 8, the four
`TestDerivationLaws` sets 32+29+23+52+18+19, `SylverLining preserves Float32`
8, `SylverLining Independent/Trivial Symmetry/Symmetry Tests`
1750+1750+232) -- none of it touched by this change, all of it still passing.

(Note: `test/Project.toml` does not exist and `Logging` is not in `[extras]`,
so `Pkg.test()`'s sandboxed environment does not resolve `using Logging` even
though it always works in the main `--project=.` environment; the test file
uses `Base.CoreLogging.Info`/`.Warn` -- part of Base, no package needed --
instead, so it needs no `Project.toml` change.)
