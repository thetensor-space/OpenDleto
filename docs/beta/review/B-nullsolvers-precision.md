# Review B — null-space solver layer and floating-point policy (`beta`)

Scope: `src/solvers/NullSolvers.jl`, `src/solvers/Precision.jl`, `src/solvers/SolverProgress.jl`,
`ext/DletoArpackExt.jl`, `ext/DletoKrylovKitExt.jl`, `ext/DletoIterativeSolversExt.jl`,
`test/TestNullVerdict.jl`, `test/TestPrecision.jl`, `test/TestSolverSeed.jl`,
`docs/design/Precision-Policy.md`, `Project.toml`, `src/Dleto.jl`. All paths below are relative to
`the beta worktree`. Read-only review; no Julia was run.

## 1. What it is

**Registry.** `abstract type NullSolver` (NullSolvers.jl:20) with a global
`const SOLVER_REGISTRY = Dict{Symbol,NullSolver}()` (:372). `register_solver!(name, solver)` (:379)
is a bare `setindex!`; `available_solvers()` (:382) sorts the keys. Core registers
`:SVDSolver, :LUSolver, :GramSolver, :AutoSolver` (:394-397) and `:ShiftInvertSolver` (:1285).
Extensions register in their `__init__` (Arpack:119-123, KrylovKit:189-194, IterativeSolvers:285-290),
adding `:ArpackSolver, :ArpackDenseSolver, :KrylovSolver, :LanczosSolver, :CGSolver, :LSMRSolver`.
Per-solver behaviour is expressed as **traits** on the solver type: `wants_square` (:174),
`densifies` (:184), `initial_request` (:201-202), `wants_seed` (:226), plus `seed_opts` (:235) to
splat a seed only into solvers that accept the keyword. `matrix_free_solvers([L])` (:141-154) is the
AutoSolver preference order, shape- and eltype-dependent.

**`solve` contract.** `solve(method::NullSolver, L::LinearMap; nv=10)` returns `(; vals, vecs)` with
`vecs` a matrix whose columns are the candidate null vectors and `vals` the matching singular-type
values (σ for rectangular solvers, λ = σ² for eigensolvers on `AᵗA`, residual norms for LSMR/LU)
(:22-37). Optional fields `converged`, `nconv`, `niter`, `nconverged`, `restarts`, `iterations`
travel on the NamedTuple and are read by `solver_converged` (:683). The symbol form is
`solve(L, sym::Symbol=:SVDSolver; kwargs...)` (:384) — argument order reversed relative to the
instance form (acknowledged as a wart at :889-891).

**`solve_nullspace(L, solver; tol, atol, nd, nv0, gap_ratio, min_above, squared, seed, store_eltype,
progress, label, kwargs...)`** (:781-1053) returns `(; vals, vecs, verdict)`. Inputs: any `LinearMap`
(or matrix wrapped as one); rectangular maps are squared **as a composition** `L' * L` for solvers
whose `wants_square` is true (:806-807); `squared=true` says the caller already handed in a Gram map
(SylverLining.jl:151). `tol` is RELATIVE to an `opnorm_estimate` of the map (:855-863), floored by the
policy, and squared when the values are σ² (`tol_default(RT; squared)`); `atol` overrides with an
absolute threshold. `nd <= 0` means "the whole null space" — request `initial_request(solver, L)`,
double while fewer than `min_above` values lie above the cut (:1048), and re-solve once at a doubled
request to confirm an iterative count (:939-965). `nd > 0` means `:fixed`: the threshold count capped
at `nd`, never certified (:637-639). `seed` is forwarded only to `wants_seed` solvers and also seeds
the `opnorm_estimate` power iteration with `MersenneTwister(seed + 1)` (:855-858).

**`NullVerdict`** (:514-532): 17 positional fields — `nullity, rule (:gap/:threshold/:fixed),
certified, status (:ok/:unconverged/:capped), gap, gap_ratio, floor, floor_binding, data_floor,
undecidable, threshold, near_null, below, above, spectrum, scale, requested`. `gap_verdict`
(:610-672) computes the cut and the two vetoes: `certified = certified && undecidable == 0 && status
=== :ok` (:663). Because the status is usually known only after the bracket test, `_with_status`
(:697-701) re-stamps it and re-applies the veto; it can only withdraw a certificate. "Status vetoes
certification" therefore means: the numbers (cut, gap, spectrum) are reported as computed, but a
solver that reported non-convergence, or an escalation that hit `k >= N` without bracketing, cannot
certify, and nullity 0 with a non-`:ok` status is `@warn`ed as a FAILED solve (:982-991).

**Precision policy** (Precision.jl). `TOL_DEFAULT = 1e-6` (:61); `FLOOR_EPS = 8` (:149), tuned in a
window `[4.92, 12.6]` from the frontier sweep; `GAP_RATIO = 100.0` (:193); `ITER_TOL_EPS = 100` (:373).
`compute_eltype(T)` promotes only Float16 → Float32 (:231-236). `precision_floor(T) = FLOOR_EPS *
eps(compute_eltype(T))` (:258) places the cut; `data_floor(T) = eps(T)` (:290) vetoes certification
only; `tol_default(T; tol, squared)` (:322-327) is the relative ceiling, squared then floored;
`iter_tol(T, tol)` (:375) floors a Krylov stopping tolerance at `100 eps`; `qd_tolerance(T, tol)`
(:407) floors solve-and-lift at `max(tol, sqrt(data_floor), precision_floor)`; `rank_rtol(T, m, n) =
max(m,n) eps` (:423) is the factorization pivot cut; `precision_policy(T, n)` (:439) bundles them.

## 2. Design assessment

**Registry vs. dispatch — a good fit, with one caveat.** The registry solves a real problem the
if/elseif chain could not (extension types are invisible to the parent module, :365-371), and the
actual polymorphism still goes through multiple dispatch on `NullSolver` subtypes — the `Dict` only
maps user-facing symbols to instances. That is the idiomatic Julia shape (cf. `Plots` backends). The
caveat: the registry is also a **policy** surface — `matrix_free_solvers` (:141-154) is a hard-coded
symbol list, so a third-party solver is reachable by name but will never be chosen by `AutoSolver`
(see §5). A `priority`/`shape` trait would let the registry be the single source of truth.

**Tolerance model — coherent, well argued, and mostly one place.** Relative-to-norm is right and the
`‖AᵗA‖ ≈ 1e25` story (:809-823) justifies it. Squaring `tol` but not the floor for Gram maps is
subtle and correctly reasoned (Precision.jl:182-191, the thirteen-decade Arpack cluster). The
three-type split (stored / compute / verdict) is the best idea in the slice and is what makes Float16
honest instead of accurate. Weak spots: (a) `store_eltype` defaults to `real(eltype(L))` (:795),
i.e. the compute type — the one wrong default for exactly the mixed case the second floor exists for
(QuickDerN.jl:1611-1621 documents having been bitten); (b) `tol` is consumed by `solve_nullspace`
and never forwarded, so the inner solvers' `tol` (Arpack:63, KrylovKit:62, LSMR:76, LU:1396) is
unreachable from the public entry point — the un-floored defaults `1e-10`/`1e-12` living in the
extensions are what always run; (c) the "one place" claim is true for the *verdict* but not for the
package: see the magic-constant list under Findings M2.

**Verdict object — well designed, loosely typed.** Reporting `below/above/spectrum/scale/requested`
alongside the decision is exactly right for a certificate. But `rule` and `status` are `Symbol`s,
not enums; a caller comparing `verdict.status == :not_converged` silently gets `false`
(DerivationReport.jl:235, QuickDerN.jl:1785 both compare against literals). The 17-field positional
constructor is called positionally in QuickDerN.jl:1574-1577 and `_with_status` (:698-701) — any
reordering is a silent field swap. There is no `ok`; the dual `certified`/`status` is explained well
(:445-484) but a caller can still read `certified == false` as "no derivations" — `den`
(Densors.jl:148) drops the verdict entirely and returns `ITensor[]` on an empty basis whatever the
status, which is the exact failure the `status` field was added to prevent.

**Seeding — correct where the trait says so, incomplete elsewhere.** `MersenneTwister(seed)` per
call makes Arpack (`v0`, Arpack:101), KrylovKit (block, :105-122), and LOBPCG (`X0`, :255-259)
deterministic across threads and tasks. `opnorm_estimate` being seeded separately (`seed + 1`) is a
good catch. But two solvers with random starts declare `wants_seed = false`: `GramSolver`
(`randn(RT, n, kp)` at :1592, global RNG) and `LSMRSolver` (`randn(T, n)` at IterativeSolvers:126).
TestSolverSeed.jl:71 asserts `!wants_seed(GramSolver())` with the comment "a dense factorization has
no random start to fix" — GramSolver is subspace iteration from a random block, so that premise is
false. `ShiftInvertSolver` forwards `kwargs...` but not a seed to its outer Krylov solve (:1268).

**Good decisions worth naming.** Composition-not-matrix squaring (:427, :807); the byte gate instead
of a dimension gate (:96-111); the dense-shortcut widening below Float64 with the measurement that
forced it (KrylovKit:76-99); the LSMR "return one column above threshold so the caller can bracket"
rule (IterativeSolvers:146-157); CPQR replacing LU with a written post-mortem (:1305-1394); the
Gram shift escalation *upward* from the floor (:1452-1456, :1567-1579); `check = false` on the
Cholesky rather than try/catch (:1573); the wide-SVD fix (:1294-1300); `maxlog = 1` on every
verdict message; and registering inside `__init__` with the precompile reason stated (KrylovKit:191).

## 3. Findings (ranked)

### Bugs / correctness risks

**B1. `GramSolver` and `LSMRSolver` are non-reproducible and say they are not seedable.**
NullSolvers.jl:1592, IterativeSolvers:126, trait defaults :226 / :19,:73. LSMR *leads* the
rectangular matrix-free order (:141-143), so matrix-free `den` is never reproducible even with a
seed; GramSolver is QuickDer's dense route above `QDN_GRAM_MIN_COLS` (QuickDerN.jl:1688). Fix: give
both a `seed` kwarg, `wants_seed = true`, and draw from `MersenneTwister(seed)`; fix the test comment.

**B2. `:CGSolver` is excluded below Float64 for square maps but not for rectangular ones.**
:141-143 vs :147-153. The Float32 LOBPCG breakdown (Cholesky of the block Gram fails, block
collapses) is a property of the solver, not of the map shape; a Float32 rectangular map whose LSMR
call is skipped or absent falls to LOBPCG. Fix: apply the eltype filter in both branches.

**B3. `den` reads a non-`:ok` empty result as "no densors".** Densors.jl:148-150 destructures
`(vals, vecs)` and returns `ITensor[]` when `size(vecs, 2) == 0`, ignoring `verdict.status`. This
is the failure mode `status` was introduced for (:756-765); QuickDerN.jl:1785 does the check,
`den` does not. Also the docstring says `nd=-1` (Densors.jl:130) while the signature says `nd=10`
(:135). Out of slice but a direct consumer of the contract.

**B4. `tol_default(Float64; squared = true)` is documented as 1.8e-15; it is 1e-12.**
Precision.jl:310 and Precision-Policy.md:65 give the Float64 squared ceiling as the floor, but
`max(1e-6^2, 1.8e-15) = 1e-12`. The code is right; the table is wrong, and it is the table users
read to know what a Gram-map solve accepts.

**B5. `store_eltype` defaults to the compute type.** :795. The default is the one value that
produces a false certificate on a promoted Float16 solve; the fix commit (bcf0f74 / QuickDerN
:1611-1621) patched callers rather than the default. Fix: make `store_eltype` required when
`eltype(L) !== compute_eltype(store)`, or carry it on the map (a thin `PromotedMap{Tstore}`
wrapper) so it cannot be forgotten.

**B6. `ArpackDenseSolver` is dead-on-arrival.** Arpack:111-117: `eigs(M; sigma = 0.0)` factors
`M - 0·I = M`, which is singular by construction on every input this package produces; it also
returns no `converged` field and prints. Delete it or implement shift-invert with a nonzero
relative shift as `shift_invert_map` does.

### API design

**A1. Two default solvers.** `solve(L, sym = :SVDSolver)` (:384) vs `solve_nullspace(L, solver =
:AutoSolver)` (:1055). Pick one.

**A2. Inner-solver options cannot pass through `AutoSolver`.** `solve(::AutoSolver)` forwards
`kwargs...` to `m.dense` (:413), but `solve(::SVDSolver, L; nv)` (:1288) accepts no kwargs, so any
Krylov option (`maxiter`, `krylovdim`) passed to `solve_nullspace(L, :AutoSolver; ...)` throws on
the dense branch and works on the matrix-free one. Same for `tol` (see §2). Fix: a
`solver_opts::NamedTuple` kwarg forwarded verbatim to the chosen solver, and `kwargs...` on the
dense solvers so they can ignore unknown options loudly (`@debug`) rather than throw.

**A3. `rule`/`status` as `Symbol`.** :515-517. Use `@enum NullRule gap threshold fixed` and
`@enum SolverStatus ok unconverged capped`, or validate in an inner constructor; add a keyword
constructor so QuickDerN.jl:1574 and `_with_status` stop being positional.

**A4. `AutoSolver`'s default behaviour depends on the environment.** `matrix_free_solvers` prefers
`:ArpackSolver` (:152), a weakdep; `:KrylovSolver` otherwise. CONTEXT.md:340-345 records the
consequence ("two code paths, one testset"): the same call is Arpack in `bench/jl` and KrylovKit in
`runtests`. Either make the tuned default a hard dep or make the fallback explicit in a warning.

**A5. `AutoSolver.dense_limit` and `DENSE_LIMIT` are dead.** :46, :337, :96-111: the body of
`dense_is_cheap` never reads `dense_limit` ("kept in the signature for callers that pass it").
Remove both or restore the shortcut.

### Maintainability

**M1. `println` to stdout in a library.** NullSolvers.jl:1289,1291; Arpack:67,112,120;
KrylovKit:66,190; IterativeSolvers:78,186,222,286. The three `__init__` messages fire on every
`using Dleto` (see D1). Replace with `@debug`; the progress tracker already has a proper `io`.

**M2. Magic constants outside Precision.jl.** Un-floored defaults that the policy then floors:
`tol = 1e-10` (Arpack:63, IterativeSolvers:76,185,219), `1e-12` (KrylovKit:62), `lsmr_tol = 1e-12,
rank_tol = 1e-8` (IterativeSolvers:68). Not floored at all: `cg_solve tol = 1e-10` (:1148),
`shift_rel = 1e-10, cgtol = 1e-4` (:1220-1221, :1251) — in Float32 CG spins to `maxiter`;
`GRAM_SHIFT_REL = 1e-10` (:1458, floored ad hoc at `10 eps`, :1490); LUSolver uses `max(m,n) *
eps(real(T))` directly (:1407) although Precision.jl:147 says it uses `rank_rtol`. Elsewhere in
src: `engaged(P, cutoff = 1e-6)` and `normalize_chisel` (Chisels.jl:50,60); `realCanonicalForm
tol = 1e-10` (DletoBase.jl:201); `nondeg tol = 1e-10` absolute (Nondegenerate.jl:29-105);
`+ 1e-15` (TensorSynthesis.jl:145); `sqrt(eps(RT))` hand-written at QuickDerN.jl:1194,1200,1216,
1961-1962 instead of `qd_tolerance`/`precision_floor`. Move the constants to Precision.jl or, at
minimum, express them as `iter_tol(RT, …)` / `rank_rtol` at the definition site.

**M3. Duplicated logic across the three extensions.** Dense shortcut `eigen(Symmetric(Matrix(L)))`
+ `converged = true` (Arpack:75-79, KrylovKit:94-99); `@assert square` (Arpack:69, KrylovKit:68,
IterativeSolvers:224); `RT = typeof(real(zero(T)))` (Arpack:71, KrylovKit:70, IterativeSolvers:82 —
`real(T)` does the same); sort-take-return (Arpack:106-107, KrylovKit:177-179, IterativeSolvers:
199-201, 273-274, NullSolvers:1276-1278, 1615-1617). A core helper `dense_shortcut(L, nev)` and
`smallest(vals, vecs, nv)` would remove ~40 lines and one class of shape bugs.

**M4. Functions over 80 lines.** `solve_nullspace` 272 lines (:781-1053), `solve(::GramSolver)`
115 (:1505-1620), `solve(::KrylovSolver)` 125 (KrylovKit:62-187), `solve(::LSMRSolver)` 87
(IterativeSolvers:76-163). Most of the length is commentary, which is valuable, but the
escalation loop, the status stamping, and the five-way message `if` (:967-1044) are three
functions living in one.

**M5. GPU hooks live in NullSolvers.jl.** :1065-1108 (`GPU_AVAILABLE`, `to_gpu`, `to_cpu`,
`gpu_sync`) have nothing to do with null spaces; `to_cpu(x::Array)` (:1099) duplicates the
`AbstractArray` method. Move to a `Device.jl`.

**M6. `progress = true` defeats the `_gram_dense` no-copy path.** `progress_wrap` (:882,
SolverProgress.jl:159-166) turns the `WrappedMap{StridedMatrix}` into a `FunctionMap`, so
`_gram_dense(L)` (:1502-1503) copies 1.1–3.3 GB purely because progress was requested.

**M7. Registry without locking.** :372, :379. Fine at `__init__`, but `register_solver!` is
exported (DletoExports.jl:76) and a Dict mutated from one task while `AutoSolver` iterates it in
another is UB. A `ReentrantLock` around writes, or freezing after load, costs nothing.

**M8. Stale docstrings after `FLOOR_EPS` 100 → 5 → 8.** Precision.jl:42 ("100x"), :158
("`FLOOR_EPS = 5`"), :285 ("`FLOOR_EPS = 100`"), :396-398 (`precision_floor` column shows
`2.2e-14`/`1.2e-5` = 100 eps), :433-436 (`precision_policy` example shows `5.96e-7` for both
`precision_floor` and `tol`; `tol_default(Float16)` is `1e-6`). TestNullVerdict.jl:282-284
("retuned from 100 to 5"). The KrylovKit warning at :172-174 has no `maxlog` and fires once per
escalation step. `solve_nullspace`'s docstring lists `nv0, gap_ratio, min_above, seed` but not
`squared`, `store_eltype`, `atol`'s interaction with `squared` (:706).

### Style / minor

`take(r) = rel[r]` closure (:665) is noise; `opnorm_estimate` returns `0.0::Float64` on one branch
and `RT` on the other (:1191, :1195); `floor` shadows `Base.floor` inside `gap_verdict` (:611);
`ShiftInvertSolver` sorts `dot(v, M*v)` (:1273-1276), which is complex for complex `T`;
`solve(::LanczosSolver)` is registered (:287) but documented as pointing at the wrong end
(:170-177) and excluded from every preference list — keep or delete, not both.

## 4. Test coverage

**Covered well.** `gap_verdict` on hand-built spectra: clean cluster, near-derivation kept out,
graded spectrum → `:threshold`, nullity 0 certified, `:fixed`, Float32 sub-floor cases
(TestNullVerdict.jl:25-128). End-to-end `solve_nullspace` on a diagonal map for Float64 and Float32
(:130-211). Status plumbing with stub solvers (`converged = false` vs absent field) (:234-301).
Policy arithmetic per type, both floors, monotone `qd_tolerance`, the data-floor veto
(TestPrecision.jl:50-143), the three routes × three types × four cases end-to-end (:228-251),
Float32 stays Float32 including an `@allocated` check (:253-299), the frontier constants as a
regression guard (:370-407), the storage type reaching QuickDer's verdict (:426-463). Seed
reproducibility for Arpack (when present), KrylovKit, LOBPCG; `:capped` forced via `min_above`
(TestSolverSeed.jl:64-179); independence of the two vetoes (:181-213).

**Weak or wrong.** `@test precision_floor(Float32) == precision_floor(Float32)` (TestPrecision.jl:87)
is a tautology — it should compare against a dimensioned variant or assert independence from `n`
via `precision_policy(T, n)`. The CGSolver testset checks `a.nullity == b.nullity` but not `== 3`
(TestSolverSeed.jl:126). TestNullVerdict.jl:108 accepts either of two outcomes. TestPrecision.jl:26
`using Logging` — Logging is not in `[deps]`/`[extras]` (Project.toml), so `Pkg.test` in a clean
sandbox fails; TestNullVerdict.jl:19-23 explicitly works around this, TestPrecision does not.
TestSolverSeed.jl:18-21 and runtests.jl:34-37 claim it is "the only file that loads the solver
extensions" — false: `src/Dleto.jl:41-42` imports KrylovKit and IterativeSolvers, so the
extensions are loaded by `using Dleto` and every earlier testset already runs against a registry
containing `:KrylovSolver`, `:CGSolver`, `:LSMRSolver`.

**Missing.** No direct test of `LUSolver`, `GramSolver`, `LSMRSolver`, `LanczosSolver`,
`ShiftInvertSolver`, `ArpackDenseSolver` (grep of `test/`: GramSolver appears once, in a trait
assertion). No "every registered solver on the same matrix agrees" sweep. No end-to-end empty null
space on a real solver (only the stub at :254-281) and no full null space (`L = 0`, `nullity == N`,
the wide-SVD path at :1294-1300 exercised only indirectly). No rank-deficient-but-non-null case
through `solve_nullspace` with an iterative solver. Float16 goes through `der` routes but never
through `solve_nullspace(...; store_eltype = Float16)` directly. Seed reproducibility is untested
for `AutoSolver`, `ShiftInvertSolver`, `GramSolver`, `LSMRSolver` (the last two cannot pass, B1).
No test that `solve_nullspace(L, :Unknown)` and `ShiftInvertSolver(:Missing)` error cleanly.
`SolverProgress.jl` has no tests at all (tag parsing, `progress_wrap` tick counts).

## 5. Extensibility and the dependency arrangement

**Adding a solver** (undocumented outside docstrings and the ext files themselves): define
`struct MySolver <: Dleto.NullSolver`, a method `Dleto.solve(::MySolver, L::LinearMap; nv, kwargs...)
-> (; vals, vecs[, converged])`, optionally `wants_square`/`densifies`/`initial_request`/
`wants_seed`, and call `Dleto.register_solver!(:MySolver, MySolver())` from `__init__`. The contract
is consistent across the three extensions and the traits make it honest. Gaps: (1) the `(; vals,
vecs)` shape is stated only in `solve`'s docstring (:22-37) and enforced nowhere — a
`validate_result` in `solve_nullspace` (matrix, `size(vecs,2) == length(vals)`) would catch the
`Vector`-of-vectors regression noted at :1391-1394 for third parties; (2) a registered solver is
never chosen by `AutoSolver` (:141-154 is a literal list); (3) the KrylovKit Arnoldi fallback
convention "never claim convergence when copies may be missing" (KrylovKit:136-149) is a rule an
extension author needs and is not in the `solve` contract; (4) there is no `docs/` page — the
`register_solver!` mechanism is mentioned only in CONTEXT.md:593-597 and Deployment-Plan.md:174.

**Dependency arrangement.** `Project.toml` lists KrylovKit and IterativeSolvers in `[deps]` (:12,:9)
*and* as triggers in `[extensions]` (:24-25), with `import KrylovKit; import IterativeSolvers` in
`src/Dleto.jl:41-42` so the extensions always load; Arpack and Metal are `[weakdeps]` (:19-20). Also
in `[deps]`: Plots (also an extension trigger, :26), IJulia, PlotlyJS, PlotlyKaleido, CSV,
DataFrames — a plotting/notebook stack as hard dependencies of a numerical kernel.

Assessment: a `[deps]` package as an extension trigger is accepted by recent Julia (the loader
falls back from `[weakdeps]` to `[deps]` when resolving triggers); the tests and CONTEXT.md confirm
it works on this machine's 1.12. But `Project.toml` has **no `julia` compat entry**, and no compat
for KrylovKit, LinearMaps, Random, SparseArrays, Arpack or Metal, so whether the arrangement resolves
on a user's Julia is unstated. More importantly it is self-defeating: an extension whose trigger is
a hard dep is loaded unconditionally, so the ext boundary buys only (a) a separate precompile unit,
(b) `__init__`-time registration that is invisible during Dleto's own `__init__` and during
precompilation of any dependent (SOLVER_REGISTRY is empty then), and (c) the stdout "Loading …"
banner on every `using Dleto`. The stated motive (Dleto.jl:34-39: AutoSolver must see the solvers it
was tuned with) is only half met, because the solver it was tuned with first is Arpack, a weakdep (A4).

Alternatives, in preference order: **(i)** make KrylovKit and IterativeSolvers genuine weakdeps and
move the tuned default into a documented `using KrylovKit` opt-in — cleanest dependency graph, but
AutoSolver silently degrades to dense SVD without it; **(ii)** keep them as hard deps and fold
`ext/DletoKrylovKitExt.jl` and `ext/DletoIterativeSolversExt.jl` into `src/solvers/`, registering at
Dleto's own load — honest about what is required, removes two extension modules, and makes the
registry complete at `__init__`; **(iii)** split into `Dleto` (core, LinearAlgebra + LinearMaps only)
and a `DletoSolvers`/`DletoPlots` glue package, moving the plotting/notebook stack out of `[deps]`.
Whichever is chosen, promote Arpack to the same class as KrylovKit or stop preferring it; add
`julia` and per-package compat; add `Logging` to `[extras]`/test target.
