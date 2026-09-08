# Review E — repository hygiene, bench/, labs/, docs/ (beta vs main)

Scope: everything outside `src/ ext/ test/`. Read-only inspection of `the beta worktree`
(113 commits over main, 2026-09-02..04, 88 of them with Claude co-author trailers). No Julia was run.

**Headline numbers.** Tracked tree: main 6.35 MB / 54 files -> beta 26.58 MB / 214 files (4.2x).
`labs/` is 21.7 MB (82% of the tree); `docs/` 2.96 MB (2.66 MB of it pre-existing images); `bench/` 1.22 MB
in 122 files (41 .jl 310 KB, 50 .csv 430 KB, 16 .log 113 KB, 12 .md 225 KB, 1 .png 111 KB). The shared
`.git` is 208 MB packed + 83 MB loose; history already carries 30 MB / 20 MB / 13 MB notebook blobs from
before beta, so beta continues an existing habit rather than starting one.

## 1. Committed artifacts

| Path | Size | What it is | Verdict |
|---|---|---|---|
| `labs/FastDerSphereComparison.ipynb` | 12.97 MB | 13 cells, 7 KB of source, **8.2 MB of outputs** (4.6 MB text/html, 2.5 MB plotly JSON, 0.74 MB SVG, 0.39 MB PNG) | Strip outputs; keep the 7 KB source. |
| `labs/WWEIA2.ipynb` | 4.97 MB | 19 code cells + **3,413 markdown cells that are byte-identical copies** of one 1,198-char "### Interpretation" block (median = max = 1198). Only 612 B of output. | Corrupted by a generation loop; dedupe to 1 cell (file becomes ~30 KB) or drop. Never a result. |
| `labs/MovieRuntime.ipynb` + 3 PNGs + `MovieRuntime.jl` | 173 KB + 120 KB + 6 KB | Notebook with 160 KB embedded PNG output; the same 3 figures also committed as files; a "script twin" duplicates the cells. | Keep `.jl` (it is the reproducible one) + PNGs (referenced from CONTEXT.md); strip notebook outputs. |
| `labs/*.pdf` (2), `labs/FPED_1720.csv` | 1.46 + 0.82 + 1.07 MB | Pre-existing on main. | Out of scope for beta, but the same policy question applies. |
| `bench/*.csv|log|png` at bench root | 6 files, ~170 KB | Outputs of ChiselOperationBench / DenseSphereProfile / Stratify*Profile written next to the scripts. | Move under `bench/reports/<date>/`; drop the two `.log`s. |
| `bench/reports/**/*.csv` | 50 files, 430 KB | Raw sweep tables; `precision-tune-a.csv` alone is 254 KB (1,945 rows). 2 orphans (`hypersphere-baseline-v3/v4.csv`, referenced nowhere). | Keep: small, cited by the .md reports, and the reports' tables are derived from them. Drop orphans. |
| `bench/reports/**/*.log` | 16 files, 113 KB | stdout captures. **15 of 16 are referenced by no .md or .jl.** They contain `/Users/algeboy/...` and `.claude/worktrees/agent-aa6f98900b62c46d8/...` paths and Julia crash backtraces. | Drop all; the CSV + README carry the result. |

Notebook outputs account for ~8.4 MB and the duplicated-cell bug for ~4.9 MB: **13.3 MB of the 20.2 MB beta adds
is neither code nor data.** The genuinely valuable results are the dated READMEs and the small CSVs (~0.65 MB).

Policy: (a) `nbstripout` as a pre-commit filter for `labs/*.ipynb` (or commit `.jl` twins, as
`MovieRuntime.jl` already does); (b) results live under `bench/reports/<YYYY-MM-DD>/<name>/` as
`README.md` + `*.csv` (+ at most one `.png`), never `.log`; (c) anything > 1 MB that is not source goes to a
release asset or a `data` orphan branch; (d) the pre-beta 30 MB blobs are a `git filter-repo` decision for the
owner, not something to do on beta.

## 2. bench/ organisation

41 Julia scripts + 1 Python + 1 zsh wrapper, in two places (`bench/` and `bench/reports/`) with no README.

- **Harness / infrastructure (2):** `SphereHarness.jl` (included by 30 of the 41 scripts; no header run-hint),
  `jl` (the budgeted wrapper; good header).
- **Profiles (7):** `DenseSphereProfile`, `MemoryProfile`, `StratifyLeadersProfile`, `StratifySolverProfile`,
  `DerivationSolverAudit`, `SylvesterKernelBench`, `SylvesterKernelEquivalence`.
- **Sweeps / frontiers (11):** `Frontier`, `QuickDerScaling`, `QuickDerLargeD`, `StratifyScaling`,
  `HypersphereBaseline`, `SparseSphereDer`, `VideoDenseBench`, `D500Matrix`, `WhitenedRestriction`,
  `ChiselOperationBench`, `reports/QuickDerSphereBench`.
- **Device (4):** `Float16Metal`, `GpuMovie`, `QuickDerMetalBench`, `SylverMetalBench`.
- **One-off experiments living inside reports/ (17):** `exp1-grid`, `exp2-seeds`, `exp3-endtoend`,
  `precision-probe0`, `precision-exp1..5b` (7), `precision-frontier`, `precision-lift-residual`,
  `precision-qd-law`, `precision-qd-tol-frontier`, `precision-tune`, `precision-tune-iter`,
  `precision-tune-report.py`, `2026-09-04/whitened/summarise.jl`.

Findings:
- Headers: every script has a prose header (good, mostly excellent). 20 of 41 never mention `bench/jl`;
  7 (`SphereHarness`, `ChiselOperationBench`, `StratifySolverProfile`, `SylvesterKernelBench`,
  `SylvesterKernelEquivalence`, `GpuMovie`, `Float16Metal`) give no run command at all. Two headers name a
  different file than the one they are in (`SylvesterKernelBench.jl` says "KernelProfile3.jl",
  `SylvesterKernelEquivalence.jl` says "Equivalence.jl") — renamed without editing.
- `bench/SylvesterKernelBench.jl:12` — `include("/Users/algeboy/CODE/OpenDleto/bench/SphereHarness.jl")`;
  the only script that does not use `@__DIR__`. Runs on one machine, and against main, not beta.
- `bench/jl:32` — `PROJECT=${JL_PROJECT:-/Users/algeboy/CODE/OpenDleto}`. The wrapper defaults to the
  *primary* worktree, so `bench/jl test/runtests.jl` executed inside `OpenDleto-beta` certifies **main**
  unless `JL_PROJECT` is exported. Should default to `$(dirname $0)/..`.
- Output locations are inconsistent: scripts write to `bench/` root, `bench/reports/`,
  `bench/reports/night-2026-09-03/`, or `bench/reports/2026-09-04/<name>/`; and several committed CSVs no
  longer match what the script writes (`HypersphereBaseline` writes `hypersphere-baseline.csv`, committed
  are `-v3`/`-v4`; `VideoDenseBench` writes `video-dense-sylver.csv`, committed `v3-`/`v4-dense-sylver.csv`;
  `Frontier` writes `restricted-solvers.csv`, not committed). Hand-renamed results are not reproducible.
- Superseded: `precision-video-verdict.jl` was correctly dropped (14995b6). Still present but likely
  superseded: `HypersphereBaseline.jl` (its only outputs are the two orphan CSVs; `Frontier.jl` covers it),
  `precision-probe0.jl` (baseline probe, no CSV), `reports/QuickDerSphereBench.jl` vs `QuickDerScaling.jl`.
  `SylvesterKernelBench` and `SylvesterKernelEquivalence` share ~all helper names and could be one file.
- `precision-tune-report.py` is the only non-Julia tool in the repo; `summarise.jl` does the same job for
  another sweep. Pick one language for table rendering.
- Records: `bench/reports/2026-09-04/{d500,gpu-movie,whitened}/README.md` are coherent reports
  (question / setup / tables / correctness / files) — the model to copy. `CHECKPOINT-2026-09-03.md` is a good
  handoff note but its "Done (uncommitted)" status is a stale snapshot. `night-2026-09-03/BOARD.md` (1,206
  lines, 15 timestamped agent entries) is a multi-agent journal: valuable pitfalls (Metal.jl reclaim, scratch
  buffers) buried in a log with `cd /Users/algeboy/...` instructions. Mine it into `contraction-options.md`
  and the design docs, then archive.

## 3. docs/

**`docs/CONTEXT.md` (1,037 lines, 72 KB).** No table of contents. 18 H2 sections, not chronological:
Session 4 (09-04) at line 8, Session 3 at 371, static "What this package is" at 451, session-2 work at 559,
then *two more* 09-04 sections at 885 and 1004. A reader must know that 09-04 is split into three places.
The **wrong movie-cost attribution is still present verbatim** (lines 887-924: "~171 s of which ~20 s is the
eigensolve ... flat in F", "the minute would need ~92 GB today"), followed at 925 by a bold "Correction, later
the same evening" and at 1004 by the corrected per-stage section. It does not contradict itself silently,
but the obsolete numbers are the ones a skim lands on, and "Next up" (973) sits between them. One dead
reference: `src/Invariants.m` (a Magma sibling file). Identifier check: 107/135 backticked names resolve in
`src/ext`; the misses are Magma names or planned artefacts (`Dleto_jll`, `DletoNativeExt`).

**`docs/design/*.md`, `docs/review/*.md`.** Spot-check against `src/ ext/`: Precision-Policy 31/34,
QuickDer-valence-n 19/21, Float16-Metal 3/4, Timing-Results 13/13, Native-Core-Plan 27/44 (misses are Rust/
C toolchain words), Deployment-Plan 24/48 (misses are proposed `libdleto`, `juliac`, `stratify_video`),
Refactor-Plan 41/59, OpenDleto-vs-Magma 22/46 (Magma type names — expected). Current enough; the misses are
honest proposals, not stale APIs. Decision-vs-proposal marking is good where it exists: Deployment-Plan and
Native-Core-Plan open with "Decision document, 2026-09-04"; Prior-Art carries a "parent's note: unverified";
Refactor-Plan marks RESOLVED/DEFERRED; OpenDleto-vs-Magma says IN PROGRESS with a Status line. Two gaps:
Refactor-Plan still says "Nothing here has been implemented yet" (09-02) although `ChiselFramed -> Chisel`
(c86ece9) and other Phase items have landed; Precision-Policy and QuickDer-valence-n have a date but no
status. **There is no docs index**: `docs/Dleto-Design.md`, `docs/Timing-Results.md`, the six design and two
review docs are linked from nowhere but CONTEXT.md.

**Install docs vs `Project.toml`.** `README.md` says "Dleto is now a Julia package", "Julia 1.12 or later",
and lists `Arpack` as "installed automatically" — Arpack is a *weakdep*. `docs/Installing-Dleto.md` says the
opposite: "not a formal Julia package ... compatible with Julia 1.7", with Julia 1.10.3 examples. README's
"Contents" TOC points at `#samples`/`#usage` sections that are commented out; the WhatWeEat Binder URL lacks
the `labs/` prefix; and `labs/geometry/SphereLab.html` / `.pdf` **do not exist** (the directory was deleted;
its 30 MB notebook blobs remain in history).

## 4. Repo config

- `Project.toml`: `version = "0.1.0"` after 113 commits and API changes (`export der` dropped, `Chisel`
  rename, new `DletoMetalExt`). Hard deps include `IJulia`, `CSV`, `DataFrames`, `JSON`, `PlotlyJS`,
  `PlotlyBase`, `PlotlyKaleido`, `Plots` — notebook tooling in a library's dependency closure.
  `IterativeSolvers`, `KrylovKit`, `Plots` are simultaneously hard deps and extension triggers, so those
  extensions always load and the weakdep pattern buys nothing. `[compat]` lacks `julia`, `KrylovKit`,
  `LinearMaps`, `PlotlyBase`, `PlotlyKaleido`, `Arpack`, `Metal`, `Random`, `SparseArrays` (registry blocker).
- `.gitignore`: orphan `!` line (a negation of nothing); no rule for `*.log`, `.ipynb_checkpoints/`,
  `.claude/worktrees/`, `.claude/settings.local.json`. `Manifest.toml` *is* ignored (good).
- `.claude/settings.json` (new in beta) whitelists `Bash(julia --project=. -e ...)` — bare `julia`, the very
  thing `bench/jl` exists to prevent, and it is per-machine policy. Move to `settings.local.json` (ignored) or
  make the allow-rule `bench/jl`.
- No `.github/` (no CI, no test run on push), no `CHANGELOG.md`, no `CITATION.cff` (an academic package with
  three named authors and grant acknowledgements). `LICENSE` MIT, present, current years.
- `test/old-tests/` (4 files, pre-existing on main) is unwired and acknowledged stale in `runtests.jl:39`.
- `src/SylverLining/SylverLininig.jl` (typo twin) was deleted on beta — good.

## 5. Coordination / private material

- `grep '/Users/'` over tracked files: `bench/jl:32`, `bench/SylvesterKernelBench.jl:12`, 15 `.log` files
  (also leaking the agent worktree name `.claude/worktrees/agent-aa6f98900b62c46d8`), `BOARD.md:687`.
- Emails: only `noreply@anthropic.com` in a quoted commit trailer (`CHECKPOINT:54`). No personal emails.
- Tokens/keys: none. `.coop`: not referenced in any tracked file. Downstream project not named in the tree.
- Not a secret but machine-specific: `bench/jl`'s budget constants and `~/.cache/opendleto-jl-slots` are
  documented as this machine's policy; fine as long as the header says so (it does).

## Findings (ranked)

**Must fix before release**
1. `labs/WWEIA2.ipynb` — 3,413 duplicated markdown cells (4.9 MB of one paragraph). Dedupe or delete.
2. `labs/FastDerSphereComparison.ipynb` — 8.2 MB of embedded plotly/HTML output. Strip outputs; add
   `nbstripout` filter for `labs/*.ipynb`.
3. `bench/jl:32` — default project is `/Users/algeboy/CODE/OpenDleto`; running it from beta tests main.
   Default to `$(cd "$(dirname "$0")/.." && pwd)`.
4. `README.md` — dead links `labs/geometry/SphereLab.{html,pdf}`, wrong Binder path for WhatWeEat, `Arpack`
   listed as auto-installed; `docs/Installing-Dleto.md` contradicts README (not a package / Julia 1.7).
   Rewrite the install section once, from `Project.toml`.
5. `Project.toml` — bump version (0.2.0), add missing `[compat]` entries incl. `julia`, decide whether
   `IterativeSolvers`/`KrylovKit`/`Plots` are deps or weakdeps (not both), move `IJulia`/`CSV`/`DataFrames`/
   `Plotly*`/`JSON` out of the library's hard deps (labs environment or weakdeps).
6. 15 orphan `bench/**/*.log` files with `/Users/algeboy` and agent-worktree paths — `git rm`, add `*.log`
   to `.gitignore`.

**Should fix**
7. `bench/SylvesterKernelBench.jl:12` absolute `include` — use `joinpath(@__DIR__, "SphereHarness.jl")`;
   fix the two stale filename headers.
8. `docs/CONTEXT.md` — add a TOC; make sections chronological (or reverse-chronological throughout); collapse
   the three 09-04 sections; strike through or box the superseded movie-cost numbers at 887-924 rather than
   leaving them live above the correction.
9. `docs/README.md` (new) indexing design/review/reports with one status word each (measured / decided /
   proposal / superseded); update Refactor-Plan's "nothing implemented" header.
10. `.claude/settings.json` — move to `settings.local.json`, or change the allow rule to `bench/jl`.
11. Consolidate result locations: move `bench/{*.csv,*.png}` and `bench/reports/*.csv` into dated
    subdirectories; delete `hypersphere-baseline-v3/v4.csv`; make committed filenames match what the script
    writes (or have the script take an `--out` argument).
12. Add `.github/workflows/test.yml` (Julia 1.12, `Pkg.test()`, ubuntu; skip Metal ext) and `CITATION.cff`.

**Nice to have**
13. Retire `HypersphereBaseline.jl`, `precision-probe0.jl`; merge `SylvesterKernelBench` +
    `SylvesterKernelEquivalence`; replace `precision-tune-report.py` with Julia.
14. Add run-hint lines (`bench/jl bench/X.jl args`) to the 7 scripts lacking one; a `bench/README.md` grouping
    scripts as above with the CSV each produces.
15. Mine `night-2026-09-03/BOARD.md` for the Metal.jl pitfalls into `docs/design/Float16-Metal.md`, then
    move it and `CHECKPOINT-2026-09-03.md` under `bench/reports/archive/`.
16. Delete `test/old-tests/` (already stale on main) or wire it into Refactor-Plan Phase 0 with a date.
17. Owner decision on `git filter-repo` for the pre-beta 30/20/13 MB notebook blobs (`labs/geometry/`,
    `labs/Noisy-stratification.ipynb`, `labs/data/VideoLab.ipynb`, `examples/Hypergraph.ipynb`).

## Proposed repo layout / policy

```
Project.toml            deps = what src/ needs; weakdeps = Arpack, Metal, KrylovKit, IterativeSolvers, Plots
.github/workflows/      test.yml (CPU only)
.gitignore              + *.log  .ipynb_checkpoints/  .claude/settings.local.json  .claude/worktrees/
CITATION.cff  CHANGELOG.md
docs/README.md          index: one line + status per document
docs/CONTEXT.md         TOC at top; one dated H2 per session, newest first, corrections applied in place
docs/design/            "Decision"/"Design note"/"Measured"/"Scout" in line 2 of every file
docs/review/
bench/README.md         script table: purpose, run line, output path
bench/jl                default project = repo containing the script
bench/*.jl              scripts only; every output goes to bench/reports/<date>/<name>/
bench/reports/<date>/<name>/{README.md, *.csv, one .png}   -- no .log, no hand-renamed files
bench/reports/archive/  BOARD.md, CHECKPOINT-*.md once mined
labs/*.ipynb            outputs stripped (nbstripout); heavy figures as .png next to a .jl twin
labs/data/              anything > 1 MB -> release asset or `data` branch, referenced by URL
```

Rule of thumb: a file is versioned if a human wrote it or a script cannot regenerate it in under an hour on
the documented budget; everything else is regenerated, and the README says how.
