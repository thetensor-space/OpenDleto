#
# QuickDer at large valence-3 dimension: how far the dense-restricted-solve
# path reaches, which null solver wins once the restricted system goes
# matrix-free, and whether the answer is still correct on the scrambled
# sphere.  See bench/reports/night-2026-09-03/BOARD.md, "### quickder-n" and
# "### quickder-large-d".
#
# RUN ONE SIZE PER COMMAND (bench/jl, one Julia process per invocation).
#
# Usage:
#   bench/jl bench/QuickDerLargeD.jl task1 d [T]
#       Random dense valence-3 randn(d,d,d), UniversalOp, UniversalChisel(3),
#       Dleto.derTrOpsReduced(:QuickDer, ...). T = Float64 (default) or Float32.
#       -> bench/reports/night-2026-09-03/quickder-v3-large.csv
#
#   bench/jl bench/QuickDerLargeD.jl task2 d solver [sizes r1,r2,r3]
#       Restricted-map solver comparison. solver in
#       LSMRSolver/ArpackSolver/KrylovSolver/CGSolver. Pass an explicit
#       "sizes r1,r2,r3" to force the matrix-free branch (e.g. at d=100).
#       -> bench/reports/night-2026-09-03/quickder-restricted-solvers.csv
#
#   bench/jl bench/QuickDerLargeD.jl task3 d
#       Scrambled sphere octant (bench/SphereHarness.jl), valence 3,
#       SymmetricOp, run_stratify(...; method=:QuickDer).
#       -> bench/reports/night-2026-09-03/quickder-v3-sphere.csv
#
using Dleto
using ITensors
using LinearAlgebra
using LinearMaps
using Random
using Printf
using Logging
try
    using Arpack
catch e
    @warn "Arpack unavailable in this environment" exception = (e, catch_backtrace())
end
try
    using KrylovKit
catch e
    @warn "KrylovKit unavailable in this environment" exception = (e, catch_backtrace())
end
try
    using IterativeSolvers
catch e
    @warn "IterativeSolvers unavailable in this environment" exception = (e, catch_backtrace())
end

include(joinpath(@__DIR__, "SphereHarness.jl"))

const REPORT_DIR = joinpath(@__DIR__, "reports", "night-2026-09-03")
mkpath(REPORT_DIR)

# --------------------------------------------------------------- Z-law residual

"""der_residual(Γ, D, P) -- same definition as test/TestDerivationLaws.jl."""
function der_residual(Γ::ITensor, D::Vector{ITensor}, P::AbstractMatrix)
    C = Chisel(P, collect(inds(Γ)))
    R = applyDerivation(Γ, D, C)
    scale = norm(Γ) * maximum(norm.(D))
    return norm(R) / max(scale, eps())
end

# --------------------------------------------------------------- branch inference

"""
Which branch `_qdn_solve_and_lift` takes for a fully-engaged, one-row chisel
(UniversalChisel) on a tensor of shape `dims`, replicating the exact
`dense_bytes <= DENSE_BUDGET_BYTES / 4` test at QuickDerN.jl:518-519.
"""
function qdn_branch(dims::Vector{Int}, r::Vector{Int}, m::Int, ::Type{T}) where {T}
    R = prod(r)
    ncols = sum(dims[a] * r[a] for a in eachindex(dims))
    rows = Dleto._qdn_system_rows(m * R, ncols)
    dense_bytes = float(rows) * ncols * sizeof(T)
    branch = dense_bytes <= Dleto.DENSE_BUDGET_BYTES / 4 ? "dense" : "matrix-free"
    return (; branch, rows, ncols, dense_bytes)
end

# --------------------------------------------------------------- warmup
#
# Every `bench/jl` invocation is a fresh process, so the FIRST call into
# derTrOpsReduced/solve_nullspace pays full JIT compilation of Dleto,
# ITensors and (for task 2) whichever extension solver is asked for. Run a
# tiny (d=6) case through the exact same method/solver path first so the
# timed call at the real `d` is compiled code, not compilation -- matching
# the convention `bench/SphereHarness.jl`'s `warmup!` uses.
function qdn_warmup(; d::Int = 6, solver::Union{Nothing,Symbol} = nothing,
                    forced_sizes::Union{Nothing,Vector{Int}} = nothing)
    frame = [Index(d, "a_$i") for i in 1:3]
    Γ = ITensor(randn(d, d, d), frame...)
    P = UniversalChisel(3)
    Ω = IndTransverseOps(frame, UniversalOp())
    kw = solver === nothing ? (;) : (; solver = solver)
    sizes_kw = forced_sizes === nothing ? (;) : (; sizes = forced_sizes)
    method = Dleto.get_derivation_method(:QuickDer; verify = :random, kw..., sizes_kw...)
    try
        Logging.with_logger(Logging.NullLogger()) do
            Dleto.derTrOpsReduced(method, Ω, P, Γ; tol = 1e-6)
        end
    catch e
        @warn "warmup call failed (continuing to the timed call anyway)" exception = (e, catch_backtrace())
    end
    return nothing
end

# --------------------------------------------------------------- task 1

function task1(d::Int, T::Type = Float64)
    Random.seed!(20260904 + d)
    dims = [d, d, d]
    frame = [Index(d, "a_$i") for i in 1:3]
    G = T === Float64 ? randn(d, d, d) : Float32.(randn(d, d, d))
    Γ = ITensor(G, frame...)
    P = UniversalChisel(3)
    Ω = IndTransverseOps(frame, UniversalOp())
    eng = Dleto.engaged(Matrix{Float64}(P))
    r = Dleto._qdn_restriction_sizes(dims, eng, 3)
    br = qdn_branch(dims, r, size(P, 1), T)

    qdn_warmup()
    GC.gc()
    method = Dleto.get_derivation_method(:QuickDer; verify = :random)
    local rΩ, expand_map, ders, st
    st = @timed ((rΩ, expand_map, ders) = Dleto.derTrOpsReduced(method, Ω, P, Γ; tol = 1e-6))
    rss = Sys.maxrss() / 2^30
    nullity = size(ders, 2)
    resid = if nullity > 0
        D = embedITensors(Ω, expand_map(ders[:, 1]))
        der_residual(Γ, D, P)
    else
        NaN
    end
    oracle_nullity = 2

    @printf("task1 d=%d T=%s branch=%s r=%s seconds=%.4f bytes=%d maxrss_GB=%.3f nullity=%d(oracle=%d) resid=%.3e\n",
            d, T, br.branch, string(r), st.time, st.bytes, rss, nullity, oracle_nullity, resid)
    flush(stdout)

    csv = joinpath(REPORT_DIR, "quickder-v3-large.csv")
    newfile = !isfile(csv)
    open(csv, "a") do io
        newfile && println(io, "d,T,seconds,bytes,maxrss_GB,nullity,oracle_nullity,residual,r1,r2,r3,branch,dense_bytes")
        @printf(io, "%d,%s,%.6f,%d,%.6f,%d,%d,%.6e,%d,%d,%d,%s,%d\n",
                d, T, st.time, st.bytes, rss, nullity, oracle_nullity, resid,
                r[1], r[2], r[3], br.branch, round(Int, br.dense_bytes))
    end
    return nothing
end

# --------------------------------------------------------------- task 2

function task2(d::Int, solver::Symbol, forced_sizes::Union{Nothing,Vector{Int}} = nothing)
    Random.seed!(20260904 + d)
    dims = [d, d, d]
    frame = [Index(d, "a_$i") for i in 1:3]
    G = randn(d, d, d)
    Γ = ITensor(G, frame...)
    P = UniversalChisel(3)
    Ω = IndTransverseOps(frame, UniversalOp())
    eng = Dleto.engaged(Matrix{Float64}(P))

    r = forced_sizes === nothing ? Dleto._qdn_restriction_sizes(dims, eng, 3) : forced_sizes
    br = qdn_branch(dims, r, size(P, 1), Float64)
    if br.branch == "dense"
        @warn "task2: chosen/forced sizes still land in the DENSE branch -- not a matrix-free comparison" r = r branch = br.branch
    end

    sizes_kw = forced_sizes === nothing ? (;) : (; sizes = forced_sizes)
    method = Dleto.get_derivation_method(:QuickDer; verify = :random, solver = solver, sizes_kw...)

    # A d=26 saturated (r=d, no restriction at all) matrix-free warmup was
    # tried here and abandoned: with no restriction the cross-sketch itself
    # costs O(d^3) at full size AND the iterative solver runs on a 17576-row
    # system, so the "warmup" ended up more expensive than the real d=100
    # call it was meant to precompile for (minutes, not seconds). JIT cost
    # for a solver's `solve(...)` method is small next to its own iteration
    # count at these sizes, so only the cheap default dense (d=6) warmup
    # below is worth paying for.
    qdn_warmup()

    # Capture whether solve_nullspace warned about an uncertified verdict.
    io = IOBuffer()
    logger = Logging.ConsoleLogger(io, Logging.Warn)
    local rΩ, expand_map, ders, st, err
    err = nothing
    GC.gc()
    st = try
        Logging.with_logger(logger) do
            @timed ((rΩ, expand_map, ders) = Dleto.derTrOpsReduced(method, Ω, P, Γ; tol = 1e-6))
        end
    catch e
        err = e
        nothing
    end
    rss = Sys.maxrss() / 2^30
    logtxt = String(take!(io))
    uncertified = occursin("UNCERTIFIED", logtxt)

    if err !== nothing
        @printf("task2 d=%d r=%s solver=%s ERROR: %s\n", d, string(r), solver,
                first(split(sprint(showerror, err), '\n')))
        flush(stdout)
        csv = joinpath(REPORT_DIR, "quickder-restricted-solvers.csv")
        newfile = !isfile(csv)
        open(csv, "a") do io2
            newfile && println(io2, "d,r1,r2,r3,branch,solver,seconds,nullity,residual,uncertified,status")
            @printf(io2, "%d,%d,%d,%d,%s,%s,,,,,\"error: %s\"\n",
                    d, r[1], r[2], r[3], br.branch, solver,
                    first(split(sprint(showerror, err), '\n')))
        end
        return nothing
    end

    nullity = size(ders, 2)
    resid = if nullity > 0
        D = embedITensors(Ω, expand_map(ders[:, 1]))
        der_residual(Γ, D, P)
    else
        NaN
    end

    @printf("task2 d=%d r=%s branch=%s solver=%s seconds=%.4f maxrss_GB=%.3f nullity=%d resid=%.3e uncertified=%s\n",
            d, string(r), br.branch, solver, st.time, rss, nullity, resid, uncertified)
    flush(stdout)

    csv = joinpath(REPORT_DIR, "quickder-restricted-solvers.csv")
    newfile = !isfile(csv)
    open(csv, "a") do io2
        newfile && println(io2, "d,r1,r2,r3,branch,solver,seconds,nullity,residual,uncertified,status")
        @printf(io2, "%d,%d,%d,%d,%s,%s,%.6f,%d,%.6e,%s,ok\n",
                d, r[1], r[2], r[3], br.branch, solver, st.time, nullity, resid, uncertified)
    end
    return nothing
end

# --------------------------------------------------------------- task 3

function task3(d::Int)
    warmup!([(; method = :QuickDer)]; d = 8, valence = 3)
    inp = build_sphere(d; valence = 3, T = Float64, ops = SymmetricOp())
    GC.gc()
    r = run_stratify(inp; method = :QuickDer, tol = 1e-6)
    rss = Sys.maxrss() / 2^30

    @printf("task3 d=%d seconds=%.4f maxrss_GB=%.3f nullity=%d(need 3) lsq_err=%.3e(need <1e-8) status=%s\n",
            d, r.seconds, rss, r.nullity, r.lsq_err, r.status)
    flush(stdout)

    csv = joinpath(REPORT_DIR, "quickder-v3-sphere.csv")
    newfile = !isfile(csv)
    open(csv, "a") do io
        newfile && println(io, "d,seconds,bytes,maxrss_GB,nullity,lsq_err,support,perm_ok,status")
        @printf(io, "%d,%.6f,%d,%.6f,%d,%.6e,%.6f,%s,\"%s\"\n",
                d, r.seconds, r.bytes, rss, r.nullity, r.lsq_err, r.support,
                r.perm_ok, r.status)
    end
    return nothing
end

# --------------------------------------------------------------- CLI

function main(args)
    task = args[1]
    if task == "task1"
        d = parse(Int, args[2])
        T = length(args) >= 3 ? (args[3] == "Float32" ? Float32 : Float64) : Float64
        task1(d, T)
    elseif task == "task2"
        d = parse(Int, args[2])
        solver = Symbol(args[3])
        forced = nothing
        for a in args[4:end]
            if startswith(a, "sizes=")
                forced = [parse(Int, x) for x in split(a[7:end], ",")]
            end
        end
        task2(d, solver, forced)
    elseif task == "task3"
        d = parse(Int, args[2])
        task3(d)
    else
        error("unknown task $task; expected task1, task2 or task3")
    end
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
