#
# Frontier -- the CPU "frontier" table at the raised (5-thread) bench/jl
# budget for the finished stack (:Auto = QuickDer-first with SylverLining
# fallback; GramSolver dense branch to ~d=130 at valence 3; matrix-free
# restricted branch above), plus the restricted-map solver comparison an
# overnight agent never finished.  See bench/reports/night-2026-09-03/
# BOARD.md ("### quickder-n", "### sylver-v4-baseline", "### orchestrator").
#
# RUN ONE SIZE PER COMMAND (bench/jl, one Julia process per invocation) --
# the stopping rule ("stop a sweep at the first point over 4 minutes") is
# applied by the caller across successive commands, not inside this file.
#
# Usage:
#   bench/jl bench/Frontier.jl frontier-cpu <valence> <d>
#       run_stratify(inp; method=:Auto), Float64, scrambled sphere octant
#       (bench/SphereHarness.jl build_sphere, default SymmetricOp).
#       -> bench/reports/night-2026-09-03/frontier-cpu.csv
#
#   bench/jl bench/Frontier.jl video-cpu <H> <W> <F> <T>
#       Derivation only, Dleto.derTrOpsReduced(get_derivation_method(:QuickDer),
#       Ω, ch, Γ; tol=1e-6), UniversalOp, UniversalChisel(4), on
#       randn(T,H,W,F,3).  T = Float64 or Float32.
#       -> bench/reports/night-2026-09-03/video-cpu.csv
#
#   bench/jl bench/Frontier.jl restricted-solvers <d> <solver>
#       Matrix-free restricted branch: random randn(d,d,d) Float64,
#       :QuickDer with solver in {LSMRSolver, ArpackSolver, KrylovSolver,
#       CGSolver} (:GramSolver is dense-only, do not pass it here).
#       -> bench/reports/night-2026-09-03/restricted-solvers.csv
#
#   bench/jl bench/Frontier.jl sparse-cpu <valence> <d>
#       Raw sphere_octant(d; valence) (unscrambled), UniversalOp, :QuickDer
#       default (restriction=:random).
#       -> bench/reports/night-2026-09-03/sparse-cpu.csv
#
using Arpack
using Dleto
using ITensors
using LinearAlgebra
using LinearMaps
using Logging
using Printf
using Random

include(joinpath(@__DIR__, "SphereHarness.jl"))

BLAS.set_num_threads(Threads.nthreads())

const REPORT_DIR = joinpath(@__DIR__, "reports", "night-2026-09-03")
mkpath(REPORT_DIR)
const THREAD_NOTE = "# 5-thread timings (bench/jl now pins 5 Julia + 5 OpenBLAS threads, " *
                     "10G heap hint, 16 GB RSS kill); may be noisy if another agent held " *
                     "a bench/jl slot concurrently"

# --------------------------------------------------------------- helpers

"""der_residual(Γ, D, P) -- same definition as test/TestDerivationLaws.jl."""
function der_residual(Γ::ITensor, D::Vector{ITensor}, P::AbstractMatrix)
    C = Chisel(P, collect(inds(Γ)))
    R = applyDerivation(Γ, D, C)
    scale = norm(Γ) * maximum(norm.(D))
    return norm(R) / max(scale, eps())
end

csv_header(path, lines...) = isfile(path) || open(path, "w") do io
    println(io, THREAD_NOTE)
    for l in lines
        println(io, l)
    end
end

"""
    branch_info(dims, engaged, valence; T = Float64) -> (; r, branch, dense_bytes)

Which branch `_qdn_solve_and_lift` takes for a fully-engaged, one-row chisel
(UniversalChisel) on a tensor of shape `dims`, per QuickDerN.jl:518-527:
dense iff `(∏r)·(Σ_a d_a·r_a)·sizeof(T) <= DENSE_BUDGET_BYTES/2`
(`DENSE_BUDGET_BYTES == 2^30`, so the gate is `2^30/2`).
"""
function branch_info(dims::Vector{Int}, eng::Vector{Bool}, valence::Int; T::Type = Float64)
    r = Dleto._qdn_restriction_sizes(dims, eng, valence)
    R = prod(r)
    sum_dr = sum(Int[dims[a] * r[a] for a in 1:valence if eng[a]])
    dense_bytes = float(R) * sum_dr * sizeof(T)
    branch = dense_bytes <= (2.0^30) / 2 ? "dense" : "matrix-free"
    return (; r, R, sum_dr, dense_bytes, branch)
end

# --------------------------------------------------------------- task 1: frontier-cpu

"""
Same body as `run_stratify(inp; method=:Auto)`, but with a real (capturing)
logger instead of `quietly()`'s NullLogger, so we can tell whether AutoDer
fell back to :SylverLining (its `@info "AutoDer: QuickDer declined ..."`
message, src/solvers/AutoDer.jl:85).
"""
function timed_stratify_auto(inp; tol::Real = 1e-6)
    Ω, ch, Γ = inp.Ω, inp.ch, inp.Γ
    io = IOBuffer()
    logger = Logging.ConsoleLogger(io, Logging.Info)
    status = "ok"
    nullity = 0
    res = nothing
    GC.gc()
    st = @timed try
        Logging.with_logger(logger) do
            redirect_stdout(devnull) do
                m = Dleto.get_derivation_method(:Auto)
                (rΩ, expand_map, ders) = derTrOpsReduced(m, Ω, ch, Γ; tol = tol)
                nullity = size(ders, 2)
                nullity == 0 && error("no nontrivial derivations")
                δ = embedITensors(Ω, expand_map(ders * randn(eltype(ders), nullity)))
                res = stratify(Γ, δ)
            end
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
    end
    logtxt = String(take!(io))
    fellback = occursin("falling back to SylverLining", logtxt)
    sc = res !== nothing ? reconstruction(inp, res.Σ, res.Xs) :
         (; lsq_err = NaN, support = NaN, perm_ok = false)
    return (; seconds = st.time, bytes = st.bytes, nullity, sc.lsq_err, sc.perm_ok, status,
              fellback)
end

function frontier_cpu(valence::Int, d::Int)
    Random.seed!(20260904 + 1000 * valence + d)
    wd = valence == 3 ? 14 : 8   # small warmup, but >= AUTODER_MIN_ENTRIES(2000) so :Auto
                                 # actually exercises the QuickDer path during warmup, not
                                 # just the SylverLining fallback (14^3=2744, 8^4=4096)
    warmup!([(; method = :Auto)]; d = wd, valence, T = Float64)

    inp = build_sphere(d; valence, T = Float64)
    dims = collect(inp.dims)
    eng = Dleto.engaged(Matrix{Float64}(inp.ch))
    bi = branch_info(dims, eng, valence)

    r = timed_stratify_auto(inp; tol = 1e-6)
    rss = Sys.maxrss() / 2^30
    oracle = valence

    @printf("frontier-cpu valence=%d d=%d branch=%s r=%s seconds=%.4f maxrss_GB=%.3f \
nullity=%d(oracle=%d) lsq_err=%.3e fellback=%s status=%s\n",
            valence, d, bi.branch, string(bi.r), r.seconds, rss, r.nullity, oracle,
            r.lsq_err, r.fellback, r.status)
    flush(stdout)

    csv = joinpath(REPORT_DIR, "frontier-cpu.csv")
    csv_header(csv, "valence,d,seconds,bytes,maxrss_GB,nullity,oracle_nullity,lsq_err," *
                     "branch,r,dense_bytes_MB,fellback_to_sylverlining,status")
    open(csv, "a") do io
        @printf(io, "%d,%d,%.6f,%d,%.6f,%d,%d,%.6e,%s,\"%s\",%.3f,%s,\"%s\"\n",
                valence, d, r.seconds, r.bytes, rss, r.nullity, oracle, r.lsq_err,
                bi.branch, string(bi.r), bi.dense_bytes / 2^20, r.fellback, r.status)
    end
    return nothing
end

# --------------------------------------------------------------- task 2: video-cpu

function video_cpu(H::Int, W::Int, F::Int, T::Type)
    Random.seed!(20260904 + H + 100W + 10_000F)

    # small warmup on the same element type
    wH, wW, wF = 10, 10, 10
    Gw = T === Float64 ? randn(wH, wW, wF, 3) : Float32.(randn(wH, wW, wF, 3))
    frw = [Index(wH, "h"), Index(wW, "w"), Index(wF, "f"), Index(3, "c")]
    Γw = ITensor(Gw, frw...)
    chw = UniversalChisel(4)
    Ωw = IndTransverseOps(frw, UniversalOp())
    method = Dleto.get_derivation_method(:QuickDer)
    try
        quietly() do
            Dleto.derTrOpsReduced(method, Ωw, chw, Γw; tol = 1e-6)
        end
    catch e
        @warn "video-cpu warmup failed (continuing to the timed call anyway)" exception = (e, catch_backtrace())
    end

    G = T === Float64 ? randn(H, W, F, 3) : Float32.(randn(H, W, F, 3))
    fr = [Index(H, "h"), Index(W, "w"), Index(F, "f"), Index(3, "c")]
    Γ = ITensor(G, fr...)
    ch = UniversalChisel(4)
    Ω = IndTransverseOps(fr, UniversalOp())
    dims = [H, W, F, 3]
    eng = Dleto.engaged(Matrix{Float64}(ch))
    bi = branch_info(dims, eng, 4; T = T)

    GC.gc()
    status = "ok"
    nullity = 0
    resid = NaN
    local ders, expand_map
    st = @timed try
        (rΩ, expand_map, ders) = Dleto.derTrOpsReduced(method, Ω, ch, Γ; tol = 1e-6)
        nullity = size(ders, 2)
        if nullity > 0
            D = embedITensors(Ω, expand_map(ders[:, 1]))
            resid = der_residual(Γ, D, ch)
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
    end
    rss = Sys.maxrss() / 2^30
    oracle = 3

    @printf("video-cpu H=%d W=%d F=%d T=%s branch=%s r=%s seconds=%.4f maxrss_GB=%.3f \
nullity=%d(oracle=%d) resid=%.3e status=%s\n",
            H, W, F, T, bi.branch, string(bi.r), st.time, rss, nullity, oracle, resid, status)
    flush(stdout)

    csv = joinpath(REPORT_DIR, "video-cpu.csv")
    csv_header(csv, "H,W,F,T,seconds,bytes,maxrss_GB,nullity,oracle_nullity,residual," *
                     "branch,r,status")
    open(csv, "a") do io
        @printf(io, "%d,%d,%d,%s,%.6f,%d,%.6f,%d,%d,%.6e,%s,\"%s\",\"%s\"\n",
                H, W, F, T, st.time, st.bytes, rss, nullity, oracle, resid, bi.branch,
                string(bi.r), status)
    end
    return nothing
end

# --------------------------------------------------------------- task 3: restricted-solvers

function restricted_solvers(d::Int, solver::Symbol)
    Random.seed!(20260904 + d)
    dims = [d, d, d]
    frame = [Index(d, "a_$i") for i in 1:3]
    G = randn(d, d, d)
    Γ = ITensor(G, frame...)
    P = UniversalChisel(3)
    Ω = IndTransverseOps(frame, UniversalOp())
    eng = Dleto.engaged(Matrix{Float64}(P))
    r = Dleto._qdn_restriction_sizes(dims, eng, 3)
    bi = branch_info(dims, eng, 3)
    if bi.branch != "matrix-free"
        @warn "restricted-solvers: d=$d still lands in the DENSE branch -- not a " *
              "matrix-free comparison" r = r branch = bi.branch
    end

    # Cheap dense (d=6) warmup only -- forcing a matrix-free warmup at a small d is not
    # possible (r <= d always) and forcing it at a size big enough to actually cross the
    # dense/matrix-free gate was tried and abandoned by an earlier agent (see
    # bench/QuickDerLargeD.jl task2 comment): the cross-sketch cost at "no restriction"
    # dwarfs the JIT tax it was meant to save.  The chosen solver's `solve(...)` method
    # therefore pays its own first-call JIT inside the timed call below, same as every
    # prior run of this comparison.
    frame6 = [Index(6, "a_$i") for i in 1:3]
    Γ6 = ITensor(randn(6, 6, 6), frame6...)
    method6 = Dleto.get_derivation_method(:QuickDer; verify = :random, solver = solver)
    try
        Logging.with_logger(Logging.NullLogger()) do
            Dleto.derTrOpsReduced(method6, IndTransverseOps(frame6, UniversalOp()),
                                  UniversalChisel(3), Γ6; tol = 1e-6)
        end
    catch e
        @warn "restricted-solvers warmup failed (continuing to the timed call anyway)" exception = (e, catch_backtrace())
    end

    method = Dleto.get_derivation_method(:QuickDer; verify = :random, solver = solver)
    io = IOBuffer()
    logger = Logging.ConsoleLogger(io, Logging.Warn)
    status = "ok"
    nullity = 0
    resid = NaN
    err = nothing
    local ders, expand_map
    GC.gc()
    st = try
        Logging.with_logger(logger) do
            @timed ((_, expand_map, ders) = Dleto.derTrOpsReduced(method, Ω, P, Γ; tol = 1e-6))
        end
    catch e
        err = e
        nothing
    end
    rss = Sys.maxrss() / 2^30
    logtxt = String(take!(io))
    uncertified = occursin("UNCERTIFIED", logtxt)
    oracle = 2

    csv = joinpath(REPORT_DIR, "restricted-solvers.csv")
    csv_header(csv, "d,r1,r2,r3,branch,solver,seconds,nullity,oracle_nullity,residual," *
                     "uncertified,status")
    if err !== nothing
        status = "error: " * first(split(sprint(showerror, err), '\n'))
        @printf("restricted-solvers d=%d r=%s solver=%s ERROR: %s\n", d, string(r), solver, status)
        flush(stdout)
        open(csv, "a") do io2
            @printf(io2, "%d,%d,%d,%d,%s,%s,,,%d,,,\"%s\"\n",
                    d, r[1], r[2], r[3], bi.branch, solver, oracle, status)
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

    @printf("restricted-solvers d=%d r=%s branch=%s solver=%s seconds=%.4f maxrss_GB=%.3f \
nullity=%d(oracle=%d) resid=%.3e uncertified=%s\n",
            d, string(r), bi.branch, solver, st.time, rss, nullity, oracle, resid, uncertified)
    flush(stdout)

    open(csv, "a") do io2
        @printf(io2, "%d,%d,%d,%d,%s,%s,%.6f,%d,%d,%.6e,%s,ok\n",
                d, r[1], r[2], r[3], bi.branch, solver, st.time, nullity, oracle, resid,
                uncertified)
    end
    return nothing
end

# --------------------------------------------------------------- task 4: sparse-cpu

function sparse_cpu(valence::Int, d::Int)
    Random.seed!(20260904 + 10_000 * valence + d)

    wd = valence == 3 ? 12 : 8
    Sw = sphere_octant(wd; valence)
    frw = collect(inds(Sw))
    method = Dleto.get_derivation_method(:QuickDer)
    try
        quietly() do
            Dleto.derTrOpsReduced(method, IndTransverseOps(frw, UniversalOp()),
                                  UniversalChisel(valence), Sw; tol = 1e-6)
        end
    catch e
        @warn "sparse-cpu warmup failed (continuing to the timed call anyway)" exception = (e, catch_backtrace())
    end

    S = sphere_octant(d; valence)
    fr = collect(inds(S))
    ch = UniversalChisel(valence)
    Ω = IndTransverseOps(fr, UniversalOp())
    dims = collect(ITensors.dim.(fr))
    eng = Dleto.engaged(Matrix{Float64}(ch))
    bi = branch_info(dims, eng, valence)

    GC.gc()
    status = "ok"
    nullity = 0
    resid = NaN
    local ders, expand_map
    st = @timed try
        (rΩ, expand_map, ders) = Dleto.derTrOpsReduced(method, Ω, ch, S; tol = 1e-6)
        nullity = size(ders, 2)
        if nullity > 0
            D = embedITensors(Ω, expand_map(ders[:, 1]))
            resid = der_residual(S, D, ch)
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
    end
    rss = Sys.maxrss() / 2^30
    oracle = 13

    @printf("sparse-cpu valence=%d d=%d branch=%s r=%s seconds=%.4f maxrss_GB=%.3f \
nullity=%d(oracle=%d) resid=%.3e status=%s\n",
            valence, d, bi.branch, string(bi.r), st.time, rss, nullity, oracle, resid, status)
    flush(stdout)

    csv = joinpath(REPORT_DIR, "sparse-cpu.csv")
    csv_header(csv, "valence,d,seconds,bytes,maxrss_GB,nullity,oracle_nullity,residual," *
                     "branch,r,status")
    open(csv, "a") do io
        @printf(io, "%d,%d,%.6f,%d,%.6f,%d,%d,%.6e,%s,\"%s\",\"%s\"\n",
                valence, d, st.time, st.bytes, rss, nullity, oracle, resid, bi.branch,
                string(bi.r), status)
    end
    return nothing
end

# --------------------------------------------------------------- CLI

function main(args)
    isempty(args) && error("first argument must be frontier-cpu, video-cpu, " *
                            "restricted-solvers, or sparse-cpu")
    task = args[1]
    if task == "frontier-cpu"
        frontier_cpu(parse(Int, args[2]), parse(Int, args[3]))
    elseif task == "video-cpu"
        T = args[5] == "Float32" ? Float32 : Float64
        video_cpu(parse(Int, args[2]), parse(Int, args[3]), parse(Int, args[4]), T)
    elseif task == "restricted-solvers"
        restricted_solvers(parse(Int, args[2]), Symbol(args[3]))
    elseif task == "sparse-cpu"
        sparse_cpu(parse(Int, args[2]), parse(Int, args[3]))
    else
        error("unknown task $task; expected frontier-cpu, video-cpu, restricted-solvers, " *
              "or sparse-cpu")
    end
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
