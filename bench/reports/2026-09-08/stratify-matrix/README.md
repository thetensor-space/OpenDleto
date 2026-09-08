# The stratify benchmark matrix, and what it says about auto-selection

2026-09-08. Raw numbers: `baseline.csv`/`baseline-summary.md` (code as found),
`tuned.csv`/`tuned-summary.md` (after the one `src/` fix below),
`knob-gram-min-cols.csv`, `knob-dense-budget.csv`. Harness: `bench/StratifyMatrix.jl`,
`bench/StratifyOverheadProfile.jl`, `bench/StratifyKnobTune.jl`. Code touched:
`src/DletoBase.jl` (`realCanonicalForm`).

## The question

`stratify` picks its derivation method and null solver through several global
`Ref` knobs (`QDN_DENSE_BUDGET_BYTES`, `QDN_GRAM_MIN_COLS`, `AUTODER_MIN_ENTRIES`,
the per-eltype `matrix_free_solvers` order). Do the shipped defaults actually
win on a grid that spans the shapes `stratify` is used on -- scrambled spheres
at valence 3 and 4, and a video-shaped random tensor -- in Float64, Float32 and
Float16? And separately: how much of one `stratify` call is `stratify`'s OWN
work (the change-of-frame construction) rather than the derivation solve it
calls out to?

## Setup

`bench/StratifyMatrix.jl` runs `:Auto` (method/solver overridable from the
command line) over: scrambled sphere valence 3 at d in {30, 60, 100, 150},
valence 4 at d in {20, 40, 60} (`bench/SphereHarness.jl`, `SymmetricOp`, oracle
nullity = valence -- the `n-1` scalar derivations plus the sphere/Euler one);
a video-shaped random tensor `120x90xFx3` for F in {10, 30} (`UniversalOp`,
oracle nullity 3 -- a random dense tensor has no structure beyond the
valence-4 chisel's scalars). Each cell in Float64/Float32/Float16, one CSV row
per cell as it completes. Per cell: wall time (a tiny warm-up run before the
grid excludes JIT), `Sys.maxrss()` delta, nullity found vs. the oracle, the
reconstruction `lsq_err` (sphere only -- video has no known scramble to score
against) and the Z-law residual (both, from
`derTrOpsReduced(...; return_diagnostics = true)`'s `DerivationReport`, which
`run_stratify` does not expose), the certified flag, which route/solver
answered, and QuickDer's per-stage times (`Dleto.QDN_STAGE_TIMES`).

Reproduce:

```
JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 JL_RSS_LIMIT_GB=6 bench/jl bench/StratifyMatrix.jl Auto AutoSolver baseline
JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 JL_RSS_LIMIT_GB=6 bench/jl bench/StratifyOverheadProfile.jl
JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 JL_RSS_LIMIT_GB=6 bench/jl bench/StratifyKnobTune.jl
```

Machine note: this ran on a shared box with other agents' Julia jobs
concurrent (per the ground rules, `bench/jl` queues rather than
oversubscribing its own 3 slots, but other processes on the box do not all go
through it). Two baseline-run cells came back as huge outliers -- sphere v3
d=30 Float32 at 933s and sphere v3 d=150 Float16 at 409s, both against
neighbours under 10s. Rerun three times each in isolation, both settled to
their normal range (0.48-0.49s and 25-29s) with identical nullity/certified --
confirmed contention noise, not a reproducible cost, matching the project's
own precedent ("wall times are noisy because a second job shared the
machine," `bench/reports/2026-09-04/whitened/README.md`). `baseline.csv` still
carries the two outlier rows as measured, since the brief says a killed run
keeps its rows and the point of a CSV is not to editorialise after the fact;
read the times in this README instead.

## Part 1: stratify's own overhead, and the one fix

`bench/StratifyOverheadProfile.jl` isolates `realCanonicalForm` (one `eigen`
per axis operator), `act` (the change-of-frame contraction), the retag loop,
and the `@info` line -- everything in `stratify(Γ, δ)` /
`stratify(Ω, ch, Γ)` (`src/Densors.jl`) that is not the derivation solve --
on three cells: sphere v3 d=100 Float64, sphere v4 d=40 Float32, video F=30
Float32.

**As found:**

| cell | solve | info | rcf (eigen) | act | retag | overhead % |
|---|---|---|---|---|---|---|
| sphere v3 d=100 F64 | 4.31s | 0.0001s | 0.0109s | 0.0107s | 0.0104s | 0.7% |
| sphere v4 d=40 F32 | 0.19s | 0.0001s | 0.0008s | 0.0105s | 0.00005s | 5.6% |
| video F=30 F32 | 0.22s | 0.0001s | **0.0580s** | 0.0047s | 0.00004s | **22.5%** |

The video cell's overhead is real and almost entirely `realCanonicalForm`: a
random dense tensor's only derivations are the chisel's scalars
`D_a = c_a·I`, and a scalar matrix stays *exactly* scalar under any per-axis
change of basis -- so all four returned operators, at every axis, are `c·I`,
and `eigen` on `c·I` is pure waste. `_isdiag_within` measured the actual
off-diagonal noise on those matrices at **~2e-5 relative** (Float32
arithmetic building up over the sketch/lift/verify pipeline), three orders of
magnitude above `realCanonicalForm`'s existing `tol = 1e-10` default -- so a
literal `tol` comparison would never have caught this for exactly the types
(Float32, Float16) it matters most for; the fast path added below tests
against `max(tol, sqrt(eps(RT)))` instead.

**Fixed** (`src/DletoBase.jl`, `realCanonicalForm`): a cheap `O(n^2)`
diagonal scan before the `O(n^3)` `eigen`, returning `D = M`, `T = I` when it
fires -- exact, since `M = I·M` trivially satisfies the law. Re-measured:

| cell | solve | overhead (info+rcf+act+retag) | overhead % |
|---|---|---|---|
| sphere v3 d=100 F64 | 1.68s | 0.0148s | 0.9% |
| sphere v4 d=40 F32 | 0.077s | 0.0039s | 4.8% |
| video F=30 F32 | 0.081s | **0.0025s** | **3.0%** |

(Solve times differ from the "as found" table run-to-run -- shared-machine
noise on a ~0.1-0.5s cell -- the overhead COLUMN is what the fix targets and
what moved: video's `rcf` went from 0.058s to 0.00007s.) Correctness: verified
directly (scalar, noisy-scalar-with-2e-5-off-diagonal-noise, dense symmetric,
dense asymmetric real spectrum, complex conjugate pair, zero, nonuniform
diagonal all satisfy `M*T == T*D`), the existing `realCanonicalForm Tests`
testset (4512 passes) is green, and the full suite is green (below). The
`stratify` full-matrix baseline-vs-tuned run (Part 3) shows identical
nullity/certified on every one of the 27 cells before and after -- the fix
changes speed, not the answer.

`act` and the retag loop were NOT a measurable fraction anywhere in this
grid (both under 1.5% even on the cell that most exercises them) and were
left alone; `@info`'s own cost is unmeasurable (0.0001s, logging-backend
overhead, not string formatting) and was also left alone -- REVIEW.md's M7
already tracks turning library `println`/`@info` into `@debug` as a separate,
larger hygiene item, not something this grid gives evidence for doing here.

## Part 2: the auto-selection knobs

**`QDN_GRAM_MIN_COLS` (default 1000) is the one knob this grid actually
exercises.** The main grid crosses its `:SVDSolver`/`:GramSolver` boundary
twice (valence 3 between d=30 and d=60, valence 4 between d=20 and d=40).
`bench/StratifyKnobTune.jl` forces each solver at every cell near the
boundary (`QDN_GRAM_MIN_COLS[] = 0` or `typemax(Int)`) to check the DEFAULT
against both, not just the one it happened to pick:

| ncols | valence,d | :GramSolver | :SVDSolver | winner |
|---|---|---|---|---|
| 480 (< 1000) | v4 d=20 | 0.84 - 2.32s | **0.05 - 0.08s** | SVD, by 15-40x |
| 990 (< 1000) | v3 d=30 | 1.80 - 6.15s | **0.33 - 0.49s** | SVD, by 4-12x |
| 1120 (>= 1000) | v4 d=40 | **0.25 - 0.33s** | 0.48 - 0.69s | Gram, by ~2x |
| 1920 (>= 1000) | v4 d=60 | **1.83 - 2.06s** | 2.70 - 3.61s | Gram, by ~1.4x |
| 2700 (>= 1000) | v3 d=60 | **0.14 - 0.24s** | 2.02 - 3.35s | Gram, by 10-24x |
| 5700 (>= 1000) | v3 d=100 | **0.87 - 1.73s** | 16.6 - 33.5s | Gram, by 15-32x |

(times are min-max across Float64/Float32/Float16, `knob-gram-min-cols.csv`
has every cell; nullity and certified match the DEFAULT's own choice in every
row except one aside below.) The boundary at 1000 sits exactly where the
data crosses from "SVD wins by an order of magnitude" (480, 990) to "Gram
wins" (1120 and up) -- **no change made; the default is validated, not
guessed at.** One aside, not a knob question: at ncols=5700, Float16 SVD
found nullity 3/3 (uncertified) where Float16 Gram found only 2/3
(uncertified) -- a solver-accuracy difference at Float16, already the kind of
thing REVIEW.md tracks under the null-solver findings, not one of the four
knobs this task named.

**`QDN_DENSE_BUDGET_BYTES` (default 2.5 GB) and `AUTODER_MIN_ENTRIES`
(default 2000) are NOT exercised by the main grid at all.** Every one of the
27 cells stayed on the dense branch (`Dleto.QDN_APPLY_COUNT` was 0 throughout
`baseline.csv`/`tuned.csv`), and every cell's entry count is 13x-oracle-nullity
times over the 2000-entry floor (the smallest cell, sphere v3 d=30, is 27000).
Per the coordinator's request, two cells chosen to be large enough to test
the dense/matrix-free boundary, still under the 6 GB budget (`bytes*6` for
sphere v3 d=200 Float32 is 192 MB): sphere v3 d=200 Float32
(oracle 3) and video 120x90x60x3 Float32 (oracle 3), each run at the DEFAULT
budget and forced matrix-free (`QDN_DENSE_BUDGET_BYTES[] = 0.0`):

| case | route | seconds | applies | nullity | certified |
|---|---|---|---|---|---|
| sphere v3 d=200 F32 | default (dense, ncols=15600) | **14.9s** | 0 | 3/3 | true |
| sphere v3 d=200 F32 | forced matrix-free | 39.2s | 57296 | 2/3 | **false** |
| video F=60 F32 | default (dense, ncols=2919) | **0.30s** | 0 | 3/3 | true |
| video F=60 F32 | forced matrix-free | 4.29s | 19482 | 3/3 | true |

Even at d=200 (valence 3, Float32, the largest cell in this budget), the
DEFAULT 2.5 GB budget still picks dense automatically -- the boundary was
never naturally reached even by the two cells added to look for it. Forcing
matrix-free anyway shows it losing on both counts everywhere it was tried:
2.6-14.5x slower, and on the sphere case it also drops a nullity and its
certificate. **No change made** -- this is evidence FOR the current budget
(if anything, evidence that it could be raised further, but with no cell in
this budget that fails dense, there is nothing to raise it against).
`AUTODER_MIN_ENTRIES` similarly has no crossing in this grid to test against
(nor an obvious cell to add: the smallest useful sphere, v3 d=30, is already
13x over it) -- left alone.

**The per-eltype `matrix_free_solvers` order (`src/solvers/NullSolvers.jl`)
was never exercised either**, for the same reason: the matrix-free branch
never ran under the default budget anywhere in this grid, forced or not
(`apply_count = 0` in every `baseline.csv`/`tuned.csv` row). Left alone --
there is no evidence at this grid's sizes.

## Part 3: baseline vs. tuned, the whole grid

Every cell's nullity/oracle/certified is IDENTICAL between `baseline.csv`
(before the `realCanonicalForm` fix) and `tuned.csv` (after it) -- the fix
does not change any answer:

| family | valence | d/F | T | nullity/oracle (both runs) | certified (both runs) |
|---|---|---|---|---|---|
| sphere | 3 | 30 | F64/F32/F16 | 3/3, 3/3, 3/3 | true, true, false |
| sphere | 3 | 60 | F64/F32/F16 | 3/3, 3/3, 2/3 | true, false, false |
| sphere | 3 | 100 | F64/F32/F16 | 3/3, 3/3, 2/3 | true, true, false |
| sphere | 3 | 150 | F64/F32/F16 | 3/3, 3/3, 3/3 | true, true, false |
| sphere | 4 | 20 | F64/F32/F16 | 4/4, 4/4, 3/4 | true, true, false |
| sphere | 4 | 40 | F64/F32/F16 | 4/4, 4/4, 3/4 | true, true, false |
| sphere | 4 | 60 | F64/F32/F16 | 4/4, 4/4, 3/4 | true, true, false |
| video | 4 | F=10 | F64/F32/F16 | 3/3, 3/3, 3/3 | true, true, true |
| video | 4 | F=30 | F64/F32/F16 | 3/3, 3/3, 3/3 | true, true, true |

(Float16 undercounts nullity on the larger spheres -- 2/3 at v3 d=60/100, 3/4
at every v4 case -- and is never certified on the sphere family; this is the
Float16 precision floor doing its job, a pre-existing and separate story from
this task, not something introduced or fixed here.) Wall-clock time is NOT a
useful baseline-vs-tuned column at the full-matrix level: most cells are
already under 0.5s and the fix's own saving (tens of milliseconds, Part 1) is
smaller than this shared machine's run-to-run jitter at that scale. Part 1's
controlled, repeated, warm measurement is the real evidence for the fix; this
table is the correctness check that it does not disturb anything else.

## What was NOT changed, and why

- **`QDN_DENSE_BUDGET_BYTES`, `AUTODER_MIN_ENTRIES`, the `matrix_free_solvers`
  order**: no default changed. Not exercised by this grid (see Part 2); the
  two cells added specifically to probe the dense/matrix-free boundary still
  landed on the dense side by default, and forcing the alternative loses on
  time everywhere and on correctness once.
- **`QDN_GRAM_MIN_COLS`**: no default changed. The measured table places the
  1000-column boundary exactly where the winner switches; there is no ncols
  in this grid where the default's choice loses.
- **`act`, the retag loop, `@info`**: measured, none is a significant
  fraction anywhere in this grid (Part 1); left alone.
- **Float16's undercounted nullity and lack of certification on structured
  (sphere) inputs, and the SVD/Gram accuracy difference noted at ncols=5700**:
  real, visible in this data, but a null-solver/precision-policy question
  (already tracked in `docs/beta/REVIEW.md`), not one of the four
  auto-selection knobs this task named.

## Suite

`JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 bench/jl test/runtests.jl`: 52
testsets, 14299 passes, 0 failures, exit code 0, after the `realCanonicalForm`
fix.
