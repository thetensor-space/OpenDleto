#
# RestrictedSolve -- can the restricted eigensolve, the ~90% of the movie's
# cost (docs/CONTEXT.md, "Movie regime measured per stage"), be made cheaper
# WITHOUT changing what it returns (nullity, certification, Z-law residual)?
#
# Candidates, cheapest first (docs/design/Native-Core-Plan.md "Phase 2"):
#   1. solver parameters on the matrix-free branch (ARPACK ncv, tol, the
#      first request nv0, min_above; ArpackSolver vs KrylovSolver vs
#      LSMRSolver on the SAME whitened map).
#   2. widen the dense/Gram route: GramSolver in Float32 syrk + Float64
#      Rayleigh-Ritz (candidate (b)).
#   3. randomized range finder for the COMPLEMENT of the (tiny) null space
#      (candidate (c)).
#   4. the block-diagonal Kronecker preconditioner, if anything is left to
#      precondition after whitening (candidate (d)/(a)).
#
# Sizes: video-shaped random tensors 160x120xFx3 Float32 (F in 10, 30, 60,
# UniversalOp/UniversalChisel(4)) and scrambled sphere octants valence 3,
# d in 100, 150, 200, Float64 (bench/SphereHarness.jl build_sphere).
#
# RUN ONE CASE PER COMMAND (bench/jl, one Julia process per invocation), as in
# bench/WhitenedRestriction.jl and bench/Frontier.jl.
#
# Usage:
#   bench/jl bench/RestrictedSolve.jl estimate
#       Restriction sizes, restricted shape and the branch the PRODUCTION
#       budget picks for every case in the sweep, from _qdn_restriction_sizes
#       alone -- allocates nothing.
#
#   bench/jl bench/RestrictedSolve.jl baseline <case>
#       The code AS FOUND: default QuickDerMethod (AutoSolver, whiten=true),
#       whichever branch the production budget picks.  <case> is one of
#       video10, video30, video60, sphere100, sphere150, sphere200.
#       -> reports/baseline.csv
#
#   bench/jl bench/RestrictedSolve.jl mf-baseline <case>
#       Same case, forced matrix-free (QDN_DENSE_BUDGET_BYTES = 0), default
#       free solver.  The baseline candidate 1's knob sweep is measured
#       against.
#       -> reports/mf-baseline.csv
#
#   bench/jl bench/RestrictedSolve.jl knobs <case> <solver> <ncv> <nv0> <min_above>
#       One (solver, ncv, nv0, min_above) combination, forced matrix-free, via
#       QuickDerMethod's `solver_kwargs`.  Pass 0 for a knob to leave it at its
#       default.
#       -> reports/knobs.csv
#
#   bench/jl bench/RestrictedSolve.jl solvers <case>
#       :ArpackSolver, :KrylovSolver, :LSMRSolver, each at its own defaults,
#       forced matrix-free, on the case.
#       -> reports/solvers.csv
#
#   bench/jl bench/RestrictedSolve.jl arpack-tol <case> <ncv> <tol>
#       `Dleto.solve(ArpackSolver(), L; nv, ncv, tol)` DIRECTLY on the
#       whitened map (`build_whitened_map`), one call, nv = oracle + 4.
#       ARPACK's own `tol` is shadowed by `solve_nullspace`'s outer `tol` at
#       every higher layer, so this is the one route that reaches it.  ncv=0
#       leaves it at ARPACK's own default.
#       -> reports/arpack-tol.csv
#
#   bench/jl bench/RestrictedSolve.jl sweep <case>
#       ONE process, ONE tensor for <case>: baseline, mf-baseline, the solver
#       comparison, a small ncv/nv0/min_above grid and a small ARPACK-tol
#       grid, all reusing the process's warm JIT (bench/jl's own startup is
#       the dominant cost of a lightweight parameter sweep otherwise).
#       -> every CSV above, one row each per sub-measurement.
#
#   bench/jl bench/RestrictedSolve.jl gram-mixed <case>
#       :GramSolver (Float64 throughout) vs :GramSolverMixed (Float32 Gram +
#       subspace iteration, Float64 Rayleigh-Ritz on the unsquared matrix),
#       dense branch (budget raised so the case reaches it).
#       -> reports/gram-mixed.csv
#
#   bench/jl bench/RestrictedSolve.jl range-finder <case>
#       Prototype: a randomized range finder for the row space (the
#       COMPLEMENT of the null space, which is tiny) versus the production
#       solver, forced matrix-free.
#       -> reports/range-finder.csv
#
using Arpack
using Dleto
using ITensors
using IterativeSolvers
using KrylovKit
using LinearAlgebra
using LinearMaps
using Logging
using Printf
using Random

include(joinpath(@__DIR__, "Frontier.jl"))     # der_residual, csv_header, branch_info, THREAD_NOTE
# Included unconditionally, at top level, not lazily inside `rmain`: Julia 1.12's
# world-age rules mean a function DEFINED by an `include` inside another
# function cannot be CALLED in the same top-level statement (`rmain`'s call
# to it would need `Base.invokelatest`).  Included here, before `the_case` and
# friends are defined below, which is fine -- the functions these files define
# only REFERENCE those names, and are not CALLED until the CLI dispatch at the
# bottom of this file, by which point every name is bound.
include(joinpath(@__DIR__, "RestrictedSolveGram.jl"))          # run_gram_mixed
include(joinpath(@__DIR__, "RestrictedSolveRangeFinder.jl"))   # run_range_finder

const RREPORT_DIR = joinpath(@__DIR__, "reports", "2026-09-08", "restricted-solve")
mkpath(RREPORT_DIR)
rcsv(name) = joinpath(RREPORT_DIR, name)

force_matrix_free!() = (Dleto.QDN_DENSE_BUDGET_BYTES[] = 0.0)
restore_dense_budget!(x) = (Dleto.QDN_DENSE_BUDGET_BYTES[] = x)

# --------------------------------------------------------------- the cases

"""
    the_case(name) -> (; Ω, ch, Γ, dims, valence, T, oracle, restricted_oracle)

`name` in `video10`, `video30`, `video60` (160x120xFx3 Float32, randn,
UniversalOp/UniversalChisel(4)) or `sphere100`, `sphere150`, `sphere200`
(valence 3 scrambled sphere octant, Float64, `build_sphere`, SymmetricOp).

TWO different "oracle" counts, and both matter here.  `oracle` is the FINAL
answer `derTrOpsReduced` returns (through `timed_run`, the FULL pipeline,
`Ω`-intersection included): 3 in both families -- the scalar derivations every
generic tensor has, video's whole answer since `Ω = UniversalOp` matches the
chisel exactly.  `restricted_oracle` is the dimension of the RESTRICTED
SYSTEM's OWN null space, which the low-level harnesses (`build_whitened_map`,
`build_dense_matrix`, used by `run_arpack_tol`/`run_gram_mixed`/
`run_range_finder` -- everything that stops at the solve and never runs
`_fastder_restrict_to_ops`) actually measure: for video it is 3 too (nothing
to intersect against), but for the sphere it is 13 -- `build_sphere`'s `ch`
is ALWAYS `UniversalChisel(3)` regardless of `Ω`, so the restricted solve
finds the full 13-dimensional universal derivation space of the (scrambled)
octant, of which only 3 lie in `Ω = SymmetricOp()` (docs/CONTEXT.md, Session 4
part 2: "the scrambled sphere's 13 universal derivations are 3 symmetric
ones"; `13` is a fixed combinatorial constant of the octant, independent of
`d` -- measured the same at d = 24, 40, 48, 64, 100, 140, 300 elsewhere in
that log, and confirmed again for d = 100/150/200 in this report).  Using the
wrong one as `nv` on a low-level call under-requests the sphere by 10 and
leaves nothing above the null cluster to scale a relative comparison against
-- the bug this docstring exists to prevent a repeat of.
"""
function the_case(name::AbstractString)
    if startswith(name, "video")
        F = parse(Int, name[6:end])
        H, W = 160, 120
        Random.seed!(20260908 + F)
        fr = [Index(H, "h"), Index(W, "w"), Index(F, "f"), Index(3, "c")]
        Γ = ITensor(Array{Float32}(randn(H, W, F, 3)), fr...)
        return (; Ω = IndTransverseOps(fr, UniversalOp()), ch = UniversalChisel(4), Γ,
                  dims = [H, W, F, 3], valence = 4, T = Float32, oracle = 3,
                  restricted_oracle = 3, label = "video-$(H)x$(W)x$(F)x3")
    elseif startswith(name, "sphere")
        d = parse(Int, name[7:end])
        inp = build_sphere(d; valence = 3, T = Float64, seed = d)
        return (; Ω = inp.Ω, ch = inp.ch, Γ = inp.Γ, dims = collect(inp.dims),
                  valence = 3, T = Float64, oracle = 3, restricted_oracle = 13,
                  label = "sphere-v3-d$d" * (inp.lean ? "-lean" : ""))
    else
        error("unknown case $name; expected video<F> or sphere<d>")
    end
end

const CASES = ["video10", "video30", "video60", "sphere100", "sphere150", "sphere200"]

# --------------------------------------------------------------- the timed call

"""
    timed_run(case; method) -> NamedTuple

One `derTrOpsReduced(method, Ω, ch, Γ)` with the apply counter and the stage
clock on, the null-solver's own log captured, and the answer's Z-law residual
measured.  Mirrors `bench/WhitenedRestriction.jl`'s `timed_quickder`, with the
restricted SHAPE and `certified` added -- this report tables both.
"""
function timed_run(case; method::Dleto.QuickDerMethod, tol::Real = 1e-6)
    Ω, ch, Γ = case.Ω, case.ch, case.Γ
    io = IOBuffer()
    logger = Logging.ConsoleLogger(io, Logging.Debug)
    status = "ok"
    nullity = 0
    resid = NaN
    stages = Dict{Symbol,Float64}()
    Dleto.QDN_STAGE_TIMES[] = stages
    Dleto.QDN_APPLY_COUNT[] = 0
    local ders, expand_map, report
    GC.gc()
    st = @timed try
        Logging.with_logger(logger) do
            redirect_stdout(devnull) do
                (_, expand_map, ders, report) =
                    Dleto.derTrOpsReduced(method, Ω, ch, Γ; tol = tol, return_diagnostics = true)
            end
        end
        nullity = size(ders, 2)
        if nullity > 0
            D = embedITensors(Ω, expand_map(ders[:, 1]))
            resid = der_residual(Γ, D, ch)
        end
    catch e
        # The first NON-BLANK line: some exceptions here (Arpack's own,
        # notably) print a blank first line, which made every such row's CSV
        # `status` column read as a bare "error: " with nothing after it.
        errlines = split(sprint(showerror, e), '\n')
        firstline = errlines[something(findfirst(l -> !isempty(strip(l)), errlines), 1)]
        status = "error: " * strip(firstline)
    end
    applies = Dleto.QDN_APPLY_COUNT[]
    Dleto.QDN_APPLY_COUNT[] = -1
    Dleto.QDN_STAGE_TIMES[] = nothing
    logtxt = String(take!(io))
    nv_requests = [parse(Int, m.captures[1])
                   for m in eachmatch(r"quickder restricted: request (\d+) of", logtxt)]
    certified = status == "ok" && nullity > 0
    verdict_certified = try; report !== nothing ? report.verdict.certified : missing; catch; missing; end
    restricted_size = try; report !== nothing ? report.restricted_size : (missing, missing); catch; (missing, missing); end
    solver_used = try; report !== nothing ? report.solver : missing; catch; missing; end
    return (; seconds = st.time, bytes = st.bytes, applies, nullity, resid, status,
              nv_requests, restricted_size, solver_used,
              certified = verdict_certified,
              solver_status = match(r"reported :(\w+)", logtxt) === nothing ? "ok" :
                              match(r"reported :(\w+)", logtxt).captures[1],
              solve_s = get(stages, :solve, NaN), whiten_s = get(stages, :whiten, 0.0),
              sketch_s = get(stages, :sketch, NaN), lift_s = get(stages, :lift, NaN),
              maxrss_GB = Sys.maxrss() / 2^30)
end

function csv_row(path, header, cols...)
    csv_header(path, THREAD_NOTE, header)
    open(path, "a") do io
        println(io, join(cols, ","))
    end
    return nothing
end

fmt(x) = x isa AbstractFloat ? @sprintf("%.6g", x) : string(x)

# --------------------------------------------------------------- estimate

"""
    real_branch_info(dims, eng, valence; T) -> (; r, R, sum_dr, dense_bytes, branch)

`branch_info` (bench/Frontier.jl) compares against a STALE constant
(`DENSE_BUDGET_BYTES/2 == 2^29`, session 3's pre-whitening threshold): the
production one is `Dleto.QDN_DENSE_BUDGET_BYTES[]` (2.5 GB as of 2026-09-04,
`_qdn_solve_and_lift` compares against it directly, no /2), and columns also
decide `GramSolver` vs `SVDSolver` on the dense route (`QDN_GRAM_MIN_COLS`,
default 1000).
"""
function real_branch_info(dims::Vector{Int}, eng::Vector{Bool}, valence::Int; T::Type)
    r = Dleto._qdn_restriction_sizes(dims, eng, valence)
    R = prod(r)
    sum_dr = sum(Int[dims[a] * r[a] for a in 1:valence if eng[a]])
    dense_bytes = float(R) * sum_dr * sizeof(T)
    branch = dense_bytes <= Dleto.QDN_DENSE_BUDGET_BYTES[] ? "dense" : "matrix-free"
    dsolver = branch == "dense" ?
              (sum_dr >= Dleto.QDN_GRAM_MIN_COLS[] ? "GramSolver" : "SVDSolver") : "-"
    return (; r, R, sum_dr, dense_bytes, branch, dsolver)
end

"""
    estimate()

Sizes and shapes for every case in the sweep from `_qdn_restriction_sizes`
alone (no tensor is built) plus which branch and dense solver the CURRENT
production budget picks -- run this before a big case.
"""
function estimate()
    @printf("%-14s %-18s %-16s %10s %10s %10s %-12s %s\n",
            "case", "dims", "r", "rows", "cols", "dense_GB", "branch", "dense solver")
    for name in CASES
        dims, T, valence, ch = case_shape(name)
        eng = Dleto.engaged(Matrix{Float64}(ch))
        bi = real_branch_info(dims, eng, valence; T = T)
        @printf("%-14s %-18s %-16s %10d %10d %10.3f %-12s %s\n",
                name, string(dims), string(bi.r), bi.R, bi.sum_dr,
                bi.dense_bytes / 2^30, bi.branch, bi.dsolver)
    end
    println("\n(production budget: QDN_DENSE_BUDGET_BYTES = ",
            Dleto.QDN_DENSE_BUDGET_BYTES[] / 2^30, " GB, QDN_GRAM_MIN_COLS = ",
            Dleto.QDN_GRAM_MIN_COLS[], ")")
    return nothing
end

"""Shape and chisel for `name`, with no tensor allocated -- `estimate`'s helper."""
function case_shape(name::AbstractString)
    if startswith(name, "video")
        F = parse(Int, name[6:end])
        return ([160, 120, F, 3], Float32, 4, UniversalChisel(4))
    else
        d = parse(Int, name[7:end])
        return ([d, d, d], Float64, 3, UniversalChisel(3))
    end
end

# --------------------------------------------------------------- baseline / mf-baseline

"""
    warmup_case(name) -> (; Ω, ch, Γ)

A tiny tensor through the SAME code path as `name` (video shape or sphere),
so the timed call does not pay first-call JIT for the chosen branch/solver.
"""
function warmup_case(name::AbstractString)
    if startswith(name, "video")
        fr = [Index(10, "h"), Index(10, "w"), Index(6, "f"), Index(3, "c")]
        Γ = ITensor(Array{Float32}(randn(10, 10, 6, 3)), fr...)
        return (; Ω = IndTransverseOps(fr, UniversalOp()), ch = UniversalChisel(4), Γ)
    else
        inp = build_sphere(10; valence = 3, T = Float64)
        return (; Ω = inp.Ω, ch = inp.ch, Γ = inp.Γ)
    end
end

function run_baseline(name::AbstractString; forced::Bool)
    c = the_case(name)
    old = Dleto.QDN_DENSE_BUDGET_BYTES[]
    forced && force_matrix_free!()
    try
        method = Dleto.get_derivation_method(:QuickDer; seed = 20260908)
        try
            timed_run(warmup_case(name); method)
        catch e
            @warn "warmup failed (continuing)" exception = (e, catch_backtrace())
        end
        res = timed_run(c; method)
        rows, cols = res.restricted_size
        csv_row(rcsv(forced ? "mf-baseline.csv" : "baseline.csv"),
                "case,label,dims,valence,eltype,forced_mf,solver,restricted_rows,restricted_cols," *
                "applies,seconds,solve_seconds,sketch_seconds,whiten_seconds,lift_seconds," *
                "maxrss_GB,nullity,oracle_nullity,residual,certified,solver_status,nv_requests,status",
                name, "\"$(c.label)\"", "\"$(c.dims)\"", c.valence, c.T, forced ? 1 : 0,
                res.solver_used, rows, cols, res.applies, fmt(res.seconds), fmt(res.solve_s),
                fmt(res.sketch_s), fmt(res.whiten_s), fmt(res.lift_s), fmt(res.maxrss_GB),
                res.nullity, c.oracle, fmt(res.resid), res.certified, res.solver_status,
                "\"$(res.nv_requests)\"", "\"$(res.status)\"")
        @printf("%-14s forced=%d solver=%-13s branch_rows=%s cols=%s applies=%-7d \
seconds=%8.2f solve=%8.2f nullity=%d(oracle=%d) certified=%s resid=%.2e status=%s\n",
                name, forced, string(res.solver_used), string(rows), string(cols), res.applies,
                res.seconds, res.solve_s, res.nullity, c.oracle, string(res.certified), res.resid,
                res.status)
    finally
        restore_dense_budget!(old)
    end
    return nothing
end

# --------------------------------------------------------------- knobs (candidate 1)

"""
    run_knobs(name, solver, ncv, nv0, min_above)

One (solver, ncv, nv0, min_above) combination through the REAL pipeline
(`derTrOpsReduced`, forced matrix-free), via `QuickDerMethod`'s
`solver_kwargs` (src/solvers/QuickDerN.jl).  `tol` is NOT tunable here: it
names two different things at two layers (`solve_nullspace`'s own relative
gap-test ceiling, which `_qdn_solve_and_lift` sets explicitly and which
`solver_kwargs` is therefore forbidden from repeating, vs ARPACK's internal
Ritz tolerance, which is shadowed by the outer name and never reaches
`Dleto.solve(::ArpackSolver, ...)` through this route at all) -- see
`run_arpack_tol` below for the one route that reaches the inner knob.
"""
function run_knobs(name::AbstractString, solver::Symbol, ncv::Int, nv0::Int, min_above::Int)
    c = the_case(name)
    old = Dleto.QDN_DENSE_BUDGET_BYTES[]
    force_matrix_free!()
    try
        skw = Dict{Symbol,Any}()
        ncv > 0 && (skw[:ncv] = ncv)
        nv0 > 0 && (skw[:nv0] = nv0)
        min_above > 0 && (skw[:min_above] = min_above)
        method = Dleto.get_derivation_method(:QuickDer; seed = 20260908, solver = solver,
                                             solver_kwargs = NamedTuple(skw))
        try
            timed_run(warmup_case(name); method)
        catch e
            @warn "warmup failed (continuing)" exception = (e, catch_backtrace())
        end
        res = timed_run(c; method)
        rows, cols = res.restricted_size
        csv_row(rcsv("knobs.csv"),
                "case,label,solver,ncv,nv0,min_above,restricted_rows,restricted_cols," *
                "applies,seconds,solve_seconds,nullity,oracle_nullity,residual,certified," *
                "solver_status,nv_requests,status",
                name, "\"$(c.label)\"", solver, ncv, nv0, min_above, rows, cols,
                res.applies, fmt(res.seconds), fmt(res.solve_s), res.nullity, c.oracle,
                fmt(res.resid), res.certified, res.solver_status, "\"$(res.nv_requests)\"",
                "\"$(res.status)\"")
        @printf("%-14s solver=%-13s ncv=%-4d nv0=%-4d min_above=%-3d applies=%-7d \
seconds=%8.2f nullity=%d(oracle=%d) certified=%s resid=%.2e status=%s\n",
                name, solver, ncv, nv0, min_above, res.applies, res.seconds,
                res.nullity, c.oracle, string(res.certified), res.resid, res.status)
    finally
        restore_dense_budget!(old)
    end
    return nothing
end

"""
    run_solvers(name; solvers)

`solvers` defaults to `(:ArpackSolver, :KrylovSolver)`, NOT `:LSMRSolver` --
measured once, on `video10` (`bench/reports/2026-09-08/restricted-solve/`):
6,493,664 applies and 671 s against ARPACK's 7392 applies and 1.5 s for the
SAME nullity, because `solve_nullspace`'s escalation loop cannot bracket
LSMR's projection early and runs `k` up toward the full dimension (3069).
That one measurement is the finding; repeating it on the larger cases here
(sphere200's restricted system is 5x wider) would cost hours on a machine
four other agents are sharing, for a conclusion already reached.  Pass
`solvers = (:ArpackSolver, :KrylovSolver, :LSMRSolver)` explicitly to repeat
it anyway.
"""
function run_solvers(name::AbstractString; solvers = (:ArpackSolver, :KrylovSolver))
    for solver in solvers
        run_knobs(name, solver, 0, 0, 0)
    end
    return nothing
end

# --------------------------------------------------------------- the SAME map, directly

"""
    build_whitened_map(c; seed) -> (; L, ncols, nrows, r, dims, T, eaxes)

Reproduces `_qdn_solve_and_lift`'s matrix-free branch (QuickDerN.jl:1636-1710,
sketch -> whiten -> `_qdn_restricted_map`) UP TO the `LinearMap`, using the
same internal functions, so that several solver calls afterwards are genuinely
"the same map" -- one sketch, one whitening, reused -- rather than one map per
solver call.  `seed` fixes the sketch (same seed => bit-identical `L` on a
repeat, since `_qdn_axis` draws from a fresh `MersenneTwister(seed)` every
time this is called).
"""
function build_whitened_map(c; seed::Integer = 20260908)
    Ω, P, Γ = c.Ω, Matrix{c.T}(c.ch), c.Γ
    fr = frames(Ω)
    G = ITensors.array(Γ, fr...)
    T = eltype(G)
    N = ndims(G)
    dims = collect(size(G))
    eng = Dleto.engaged(Matrix{Float64}(P))
    r = Dleto._qdn_restriction_sizes(dims, eng, N)
    eaxes = [a for a in 1:N if eng[a]]
    rng = MersenneTwister(seed)
    axs = [Dleto._qdn_axis(T, dims[a], r[a], :random, rng) for a in 1:N]
    S = Dleto._qdn_cross_sketches(G, axs, eng)
    Uf = Dict{Int, Matrix{T}}(a => Dleto._qdn_host(Dleto._qdn_unfold(S[a], a)) for a in eaxes)
    wh = Dleto._qdn_whiten(Uf, eaxes, dims, r, N)
    Us, sdims = wh[2], wh[4]
    coff = Dict{Int,Int}(); ncols = 0
    for a in eaxes
        coff[a] = ncols
        ncols += sdims[a] * r[a]
    end
    Sh = wh[3]
    L = Dleto._qdn_restricted_map(Sh, Us, P, eaxes, r, sdims, coff, ncols)
    nrows = size(L, 1)
    return (; L, ncols, nrows, r, dims, T, eaxes)
end

"""
    run_arpack_tol(name, ncv, tol)

`Dleto.solve(Dleto.ArpackSolver(), L; nv, ncv, tol)` DIRECTLY on the map
`build_whitened_map` returns, for a fixed `nv = oracle + 4` -- one ARPACK call,
not the escalation loop -- since ARPACK's own `tol` (relative to the Ritz
value inside ARPACK) is shadowed by `solve_nullspace`'s outer `tol` (the
gap-test ceiling) at every higher layer and never reaches
`Dleto.solve(::ArpackSolver, ...)` through `derTrOpsReduced` or
`solve_nullspace` at all.  This is the one route that reaches it.
"""
function run_arpack_tol(name::AbstractString, ncv::Int, tol::Real)
    c = the_case(name)
    m = build_whitened_map(c)
    Lsq = m.L' * m.L        # ArpackSolver needs a square map, as solve_nullspace forms it
    nv = c.restricted_oracle + 4
    # `ArpackSolver` the STRUCT lives in the `DletoArpackExt` weakdep module,
    # not in `Dleto` itself (`Dleto.ArpackSolver` is undefined) -- the
    # registry is the portable way to get the singleton instance any caller
    # who only knows the symbol `:ArpackSolver` uses.
    arpack = Dleto.SOLVER_REGISTRY[:ArpackSolver]
    Dleto.QDN_APPLY_COUNT[] = 0
    GC.gc()
    st = @timed Dleto.solve(arpack, Lsq; nv = nv,
                            ncv = ncv > 0 ? ncv : min(m.ncols, max(2nv + 1, 8nv)),
                            tol = tol, seed = 20260908 + 1)
    applies = Dleto.QDN_APPLY_COUNT[]
    Dleto.QDN_APPLY_COUNT[] = -1
    res = st.value
    nbelow = count(v -> abs(v) <= 1e-6 * maximum(abs, res.vals), res.vals)
    csv_row(rcsv("arpack-tol.csv"),
            "case,label,ncv,tol,restricted_rows,restricted_cols,nv,applies,seconds," *
            "converged,nconv,niter,below_1e-6,oracle_nullity",
            name, "\"$(c.label)\"", ncv, tol, m.nrows, m.ncols, nv, applies, fmt(st.time),
            res.converged, get(res, :nconv, missing), get(res, :niter, missing), nbelow, c.oracle)
    @printf("%-14s ncv=%-5s tol=%-8.1e nv=%-4d applies=%-7d seconds=%7.3f converged=%s \
below=%d(oracle=%d)\n",
            name, ncv > 0 ? string(ncv) : "auto", tol, nv, applies, st.time,
            string(res.converged), nbelow, c.oracle)
    return nothing
end

# --------------------------------------------------------------- sweep (one process, one case)

"""
    run_sweep(name)

Everything candidate 1 needs on ONE case in ONE process: `bench/jl`'s own
startup (package load, JIT) dominates a lightweight parameter sweep otherwise,
since this file's cases cost milliseconds to seconds each, not the minutes the
d = 1000 frontier runs did.
"""
function run_sweep(name::AbstractString)
    println("=== baseline (as found) ===")
    run_baseline(name; forced = false)
    println("=== mf-baseline (forced matrix-free, AutoSolver) ===")
    run_baseline(name; forced = true)
    println("=== solver comparison (forced matrix-free, each solver's own defaults) ===")
    run_solvers(name)
    println("=== ncv / nv0 / min_above grid (ArpackSolver, forced matrix-free) ===")
    for (ncv, nv0, min_above) in [(0, 0, 0), (32, 0, 0), (64, 0, 0), (256, 0, 0), (0, 32, 0), (0, 0, 4)]
        run_knobs(name, :ArpackSolver, ncv, nv0, min_above)
    end
    println("=== ARPACK tol / ncv, direct on the SAME map ===")
    for (ncv, tol) in [(0, 1e-10), (0, 1e-6), (0, 1e-14), (128, 1e-10), (256, 1e-10)]
        run_arpack_tol(name, ncv, tol)
    end
    return nothing
end

# --------------------------------------------------------------- CLI

function rmain(args)
    isempty(args) && error("first argument must be estimate, baseline, mf-baseline, knobs, " *
                           "solvers, gram-mixed or range-finder")
    task = args[1]
    if task == "estimate"
        estimate()
    elseif task == "baseline"
        run_baseline(args[2]; forced = false)
    elseif task == "mf-baseline"
        run_baseline(args[2]; forced = true)
    elseif task == "knobs"
        run_knobs(args[2], Symbol(args[3]), parse(Int, args[4]), parse(Int, args[5]),
                 parse(Int, args[6]))
    elseif task == "solvers"
        run_solvers(args[2])
    elseif task == "arpack-tol"
        run_arpack_tol(args[2], parse(Int, args[3]), parse(Float64, args[4]))
    elseif task == "sweep"
        run_sweep(args[2])
    elseif task == "gram-mixed"
        run_gram_mixed(args[2])
    elseif task == "range-finder"
        run_range_finder(args[2])
    else
        error("unknown task $task")
    end
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && rmain(ARGS)
