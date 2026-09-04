#
# WhitenedRestriction -- does whitening the restricted system (QuickDer-W,
# `whiten = true`) buy iterations on the MATRIX-FREE branch, which is the only
# branch left above d ~ 200 at valence 3?
#
# The dense branch is at the BLAS floor (docs/design/Native-Core-Plan.md), so
# on the matrix-free branch iteration count is the whole story.  Every run here
# therefore FORCES the matrix-free branch (`QDN_DENSE_BUDGET_BYTES[] = 0`) so
# that the same size can be compared with and without whitening, and reads the
# iteration count directly out of `Dleto.QDN_APPLY_COUNT` instead of inferring
# it from wall time.
#
# Reuses bench/Frontier.jl (`branch_info`, `der_residual`, `csv_header`,
# THREAD_NOTE) and bench/SphereHarness.jl (`build_sphere`).
#
# RUN ONE CASE PER COMMAND -- one Julia process per invocation, as in
# bench/Frontier.jl.
#
# Usage:
#   bench/jl bench/WhitenedRestriction.jl estimate
#       Print, for every case in the sweep, the restriction sizes, the
#       restricted system's shape, the branch the production budget would pick,
#       and an RSS estimate.  Allocates nothing: run this BEFORE a big case.
#
#   bench/jl bench/WhitenedRestriction.jl sphere <valence> <d> <whiten> <solver> [T]
#       Scrambled sphere octant (build_sphere, SymmetricOp), derivation only
#       (`derTrOpsReduced` on :QuickDer, verify = :random so the Z-law runs).
#       <whiten> is 0 or 1; <solver> is the restricted null solver
#       (ArpackSolver, CGSolver = LOBPCG on AᵗA, LSMRSolver, KrylovSolver, or
#       AutoSolver for QuickDer's own default).  T defaults to Float64.
#
#   bench/jl bench/WhitenedRestriction.jl video <H> <W> <F> <whiten> <solver> <T>
#       randn(T, H, W, F, 3), UniversalOp, UniversalChisel(4).
#
#   bench/jl bench/WhitenedRestriction.jl random <d> <whiten> <solver> [T]
#       randn(T, d, d, d), UniversalOp: the GENERIC family, and the only one
#       where the unwhitened matrix-free branch converges, so the only one
#       with a finite iteration ratio.
#
#   bench/jl bench/WhitenedRestriction.jl degenerate <d> <whiten> <solver>
#       randn(d,d,d) with mode 1 projected to rank d-2: the case where the
#       whitening has to truncate and report the trivial derivations.
#
# -> bench/reports/2026-09-04/whitened/whitened.csv
#
using Arpack
using Dleto
using ITensors
using IterativeSolvers
using LinearAlgebra
using LinearMaps
using Logging
using Printf
using Random

include(joinpath(@__DIR__, "Frontier.jl"))

const WREPORT_DIR = joinpath(@__DIR__, "reports", "2026-09-04", "whitened")
mkpath(WREPORT_DIR)
const WCSV = joinpath(WREPORT_DIR, "whitened.csv")

# The comparison is about the matrix-free branch, so take it at every size.
# Restored nowhere: one case per process.
force_matrix_free!() = (Dleto.QDN_DENSE_BUDGET_BYTES[] = 0.0)

# --------------------------------------------------------------- the timed call

"""
    timed_quickder(Ω, ch, Γ; whiten, solver, tol) -> NamedTuple

One `derTrOpsReduced(:QuickDer, ...)` with the apply counter and the stage
clock on, the null-solver warnings captured, and the answer's Z-law residual
measured.  `applies` counts forward + adjoint applications of the matrix-free
restricted map; `uncertified` is `solve_nullspace`'s own verdict word.
"""
function timed_quickder(Ω, ch, Γ; whiten::Bool, solver::Symbol, tol::Real = 1e-6)
    method = Dleto.get_derivation_method(:QuickDer; whiten = whiten, solver = solver,
                                         verify = :random, seed = 20260904)
    io = IOBuffer()
    logger = Logging.ConsoleLogger(io, Logging.Info)
    status = "ok"
    nullity = 0
    resid = NaN
    stages = Dict{Symbol,Float64}()
    Dleto.QDN_STAGE_TIMES[] = stages
    Dleto.QDN_APPLY_COUNT[] = 0
    local ders, expand_map
    GC.gc()
    st = @timed try
        Logging.with_logger(logger) do
            redirect_stdout(devnull) do
                (_, expand_map, ders) = Dleto.derTrOpsReduced(method, Ω, ch, Γ; tol = tol)
            end
        end
        nullity = size(ders, 2)
        if nullity > 0
            D = embedITensors(Ω, expand_map(ders[:, 1]))
            resid = der_residual(Γ, D, ch)
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
    end
    applies = Dleto.QDN_APPLY_COUNT[]
    Dleto.QDN_APPLY_COUNT[] = -1
    Dleto.QDN_STAGE_TIMES[] = nothing
    logtxt = String(take!(io))
    return (; seconds = st.time, bytes = st.bytes, applies, nullity, resid, status,
              uncertified = occursin("UNCERTIFIED", logtxt),
              trivial = occursin("rank-deficient mode", logtxt),
              solve_s = get(stages, :solve, NaN), whiten_s = get(stages, :whiten, 0.0),
              sketch_s = get(stages, :sketch, NaN), lift_s = get(stages, :lift, NaN),
              maxrss_GB = Sys.maxrss() / 2^30)
end

function record!(case, dims, r, T, whiten, solver, oracle, res)
    csv_header(WCSV,
        "# forced matrix-free branch (QDN_DENSE_BUDGET_BYTES = 0) so both settings " *
        "meet the same solver at the same size",
        "case,dims,valence,eltype,whiten,solver,r,restricted_rows,restricted_cols," *
        "applies,seconds,solve_seconds,whiten_seconds,sketch_seconds,lift_seconds," *
        "maxrss_GB,bytes,nullity,oracle_nullity,residual,uncertified,trivial_reinjected,status")
    n = length(dims)
    eng = Dleto.engaged(Matrix{Float64}(UniversalChisel(n)))   # rows of a 1-row chisel
    rows = prod(r)
    cols = sum(Int[dims[a] * r[a] for a in 1:n if eng[a]])
    @printf("%-22s dims=%-18s T=%-7s whiten=%d solver=%-13s r=%-16s applies=%-7d \
seconds=%8.2f solve=%8.2f whiten_s=%6.3f maxrss=%.2fGB nullity=%d(oracle=%d) \
resid=%.2e uncert=%s status=%s\n",
            case, string(dims), T, whiten, solver, string(r), res.applies, res.seconds,
            res.solve_s, res.whiten_s, res.maxrss_GB, res.nullity, oracle, res.resid,
            res.uncertified, res.status)
    flush(stdout)
    open(WCSV, "a") do io
        @printf(io, "%s,\"%s\",%d,%s,%d,%s,\"%s\",%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,\
%.6f,%d,%d,%d,%.6e,%s,%s,\"%s\"\n",
                case, string(dims), n, T, whiten, solver, string(r), rows, cols,
                res.applies, res.seconds, res.solve_s, res.whiten_s, res.sketch_s,
                res.lift_s, res.maxrss_GB, res.bytes, res.nullity, oracle, res.resid,
                res.uncertified, res.trivial, res.status)
    end
    return nothing
end

# --------------------------------------------------------------- cases

function sphere_case(valence::Int, d::Int, whiten::Bool, solver::Symbol, T::Type)
    force_matrix_free!()
    # Warm the whole path up at a size where it is cheap, so the timed call does
    # not pay first-call JIT for the chosen solver.
    let wd = valence == 3 ? 10 : 6
        inp = build_sphere(wd; valence, T)
        try
            timed_quickder(inp.Ω, inp.ch, inp.Γ; whiten, solver)
        catch e
            @warn "warmup failed (continuing)" exception = (e, catch_backtrace())
        end
    end
    inp = build_sphere(d; valence, T)
    dims = collect(inp.dims)
    r = Dleto._qdn_restriction_sizes(dims, Dleto.engaged(Matrix{Float64}(inp.ch)), valence)
    res = timed_quickder(inp.Ω, inp.ch, inp.Γ; whiten, solver)
    # `-lean` in the case name, not a new column: above `SPHERE_LEAN_BYTES`
    # `build_sphere` takes its memory-lean path, which is the same input family
    # up to a per-axis orthogonal change of basis but NOT the same array -- so
    # the applies and the residual digits of a `-lean` row are not comparable
    # with a classic row at the same d, while the nullity and verdict are.
    record!("sphere-v$valence-d$d" * (inp.lean ? "-lean" : ""),
            dims, r, T, whiten, solver, valence, res)
    return nothing
end

function video_case(H::Int, W::Int, F::Int, whiten::Bool, solver::Symbol, T::Type)
    force_matrix_free!()
    let (h, w, f) = (10, 10, 10)
        frw = [Index(h, "h"), Index(w, "w"), Index(f, "f"), Index(3, "c")]
        Γw = ITensor(Array{T}(randn(h, w, f, 3)), frw...)
        try
            timed_quickder(IndTransverseOps(frw, UniversalOp()), UniversalChisel(4), Γw;
                           whiten, solver)
        catch e
            @warn "warmup failed (continuing)" exception = (e, catch_backtrace())
        end
    end
    Random.seed!(20260904 + H + 100W + 10_000F)
    fr = [Index(H, "h"), Index(W, "w"), Index(F, "f"), Index(3, "c")]
    Γ = ITensor(Array{T}(randn(H, W, F, 3)), fr...)
    dims = [H, W, F, 3]
    r = Dleto._qdn_restriction_sizes(dims, Dleto.engaged(Matrix{Float64}(UniversalChisel(4))), 4)
    res = timed_quickder(IndTransverseOps(fr, UniversalOp()), UniversalChisel(4), Γ;
                         whiten, solver)
    record!("video-$(H)x$(W)x$(F)x3", dims, r, T, whiten, solver, 3, res)
    return nothing
end

"""
    random_case(d, whiten, solver, T)

`randn(T, d, d, d)`, UniversalOp, UniversalChisel(3): the GENERIC tensor, where
the unwhitened matrix-free branch is already known to converge
(bench/reports/night-2026-09-03/quickder-restricted-solvers.csv).  This is the
only family that gives a finite iteration RATIO -- on the structured sphere the
unwhitened branch does not converge at any d, so its count is just ARPACK's
cap.  It is also the case where the whitening should help LEAST: a random
tensor's mode unfoldings are already well conditioned (measured cond(M_a) ~ 3
against 20..150 on the sphere), which is the analysis's own prediction.
"""
function random_case(d::Int, whiten::Bool, solver::Symbol, T::Type)
    force_matrix_free!()
    let wd = 10
        frw = [Index(wd, "a$a") for a in 1:3]
        Γw = ITensor(Array{T}(randn(wd, wd, wd)), frw...)
        try
            timed_quickder(IndTransverseOps(frw, UniversalOp()), UniversalChisel(3), Γw;
                           whiten, solver)
        catch e
            @warn "warmup failed (continuing)" exception = (e, catch_backtrace())
        end
    end
    Random.seed!(20260904 + d)
    fr = [Index(d, "a$a") for a in 1:3]
    Γ = ITensor(Array{T}(randn(d, d, d)), fr...)
    dims = [d, d, d]
    r = Dleto._qdn_restriction_sizes(dims, Dleto.engaged(Matrix{Float64}(UniversalChisel(3))), 3)
    res = timed_quickder(IndTransverseOps(fr, UniversalOp()), UniversalChisel(3), Γ;
                         whiten, solver)
    record!("random-v3-d$d", dims, r, T, whiten, solver, 2, res)
    return nothing
end

"""A tensor whose mode-1 unfolding has rank `d-2`: the truncation case."""
function degenerate_case(d::Int, whiten::Bool, solver::Symbol)
    force_matrix_free!()
    Random.seed!(20260904 + d)
    U = Matrix(qr(randn(d, d)).Q)
    G = Dleto._qdn_ttm(randn(d, d, d), U[:, 1:(d - 2)] * transpose(U[:, 1:(d - 2)]), 1)
    fr = [Index(d, "c$a") for a in 1:3]
    Γ = ITensor(G, fr...)
    dims = [d, d, d]
    r = Dleto._qdn_restriction_sizes(dims, Dleto.engaged(Matrix{Float64}(UniversalChisel(3))), 3)
    res = timed_quickder(IndTransverseOps(fr, UniversalOp()), UniversalChisel(3), Γ;
                         whiten, solver)
    # 2 scalar derivations + the full trivial space of the deficient mode
    record!("degenerate-d$d", dims, r, Float64, whiten, solver, 2 + 2 * d, res)
    return nothing
end

# --------------------------------------------------------------- estimate

"""
    estimate()

Sizes, branches and an RSS estimate for every case in the sweep, computed from
`_qdn_restriction_sizes` and `branch_info` alone -- nothing is allocated.  The
tensor estimate is `3 x prod(dims) x sizeof(T)`: `build_sphere` holds the
sphere, its scramble and the nondeg tensor at once, and `_qdn_pair_tensor`
`permutedims` the full tensor once per lift axis whose first small contraction
is not axis 1.
"""
function estimate()
    cases = Tuple{String, Vector{Int}, Type}[]
    for d in (200, 300, 500, 1000); push!(cases, ("sphere-v3-d$d", [d, d, d], Float64)); end
    for d in (60, 80, 100, 150, 200); push!(cases, ("sphere-v4-d$d", [d, d, d, d], Float64)); end
    for d in (100, 150, 200); push!(cases, ("sphere-v4-d$d-f32", [d, d, d, d], Float32)); end
    push!(cases, ("video-200x200x100x3", [200, 200, 100, 3], Float32))
    push!(cases, ("video-150x150x80x3", [150, 150, 80, 3], Float32))
    push!(cases, ("video-100x100x100x3", [100, 100, 100, 3], Float32))
    @printf("%-24s %-20s %-8s %-18s %10s %10s %10s %9s %s\n",
            "case", "dims", "eltype", "r", "rows", "cols", "dense_GB", "tensor_GB", "branch")
    for (name, dims, T) in cases
        n = length(dims)
        eng = Dleto.engaged(Matrix{Float64}(UniversalChisel(n)))
        bi = branch_info(dims, eng, n; T = T)
        rows = bi.R
        cols = bi.sum_dr
        tensorGB = 3 * prod(float.(dims)) * sizeof(T) / 2^30
        @printf("%-24s %-20s %-8s %-18s %10d %10d %10.2f %9.2f %s\n",
                name, string(dims), T, string(bi.r), rows, cols,
                bi.dense_bytes / 2^30, tensorGB, bi.branch)
    end
    println("\nfits a 12 GB process when tensor_GB + dense_GB (if dense) stays under ~10 GB")
    return nothing
end

# --------------------------------------------------------------- CLI

function wmain(args)
    isempty(args) && error("first argument must be estimate, sphere, video, random " *
                           "or degenerate")
    task = args[1]
    if task == "estimate"
        estimate()
    elseif task == "sphere"
        T = length(args) >= 6 && args[6] == "Float32" ? Float32 : Float64
        sphere_case(parse(Int, args[2]), parse(Int, args[3]), args[4] == "1",
                    Symbol(args[5]), T)
    elseif task == "video"
        T = args[7] == "Float32" ? Float32 : Float64
        video_case(parse(Int, args[2]), parse(Int, args[3]), parse(Int, args[4]),
                   args[5] == "1", Symbol(args[6]), T)
    elseif task == "random"
        T = length(args) >= 5 && args[5] == "Float32" ? Float32 : Float64
        random_case(parse(Int, args[2]), args[3] == "1", Symbol(args[4]), T)
    elseif task == "degenerate"
        degenerate_case(parse(Int, args[2]), args[3] == "1", Symbol(args[4]))
    else
        error("unknown task $task; expected estimate, sphere, video, random or " *
              "degenerate")
    end
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && wmain(ARGS)
