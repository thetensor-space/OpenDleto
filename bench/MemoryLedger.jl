#
# MemoryLedger.jl -- peak RSS, tensor bytes, and the Float16/Float32 ratio,
# on the shapes the 2026-09-08 memory task measures: two video shapes and two
# sphere sizes, each in Float64/Float32/Float16.
#
#   JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 JL_RSS_LIMIT_GB=6 JL_HEAP=2G \
#     bench/jl bench/MemoryLedger.jl
#
# Writes bench/reports/2026-09-08/memory/ledger.csv (one row per shape x
# eltype) and prints the same table to stdout.  Reuses `build_sphere`
# (bench/SphereHarness.jl) for the sphere cases; the video cases are built
# inline, the same recipe `bench/MemoryProfile.jl`'s `profile_video` uses,
# kept here rather than `include`d so this script owns its own stage
# boundaries end to end (one `Sys.maxrss()` delta over the WHOLE build+solve,
# which is what a caller of `stratify`/`der` actually pays).
#
using Arpack          # so the matrix-free branch's default solver is ArpackSolver
using Printf
using Random

include(joinpath(@__DIR__, "SphereHarness.jl"))

const ML_GB = 1 / 2^30

"""
    ml_video(H, W, F, T) -> (Γ, Ω, ch)

A random `H x W x F x 3` tensor, `UniversalOp` on every axis (the movie
regime's operator space) -- the same recipe `bench/MemoryProfile.jl`'s
`profile_video` uses.
"""
function ml_video(H::Integer, W::Integer, F::Integer, T::Type)
    Random.seed!(20260908 + H + 100W + 10_000F)
    A = randn(T, H, W, F, 3)
    fr = [Index(H, "h"), Index(W, "w"), Index(F, "f"), Index(3, "c")]
    Γ = ITensors.itensor(A, fr...)
    Ω = IndTransverseOps(fr, UniversalOp())
    return (Γ, Ω, UniversalChisel(4))
end

"""
    ml_sphere(d, T) -> (Γ, Ω, ch)

The scrambled, nondegenerate sphere octant at valence 3, `build_sphere`'s
default (classic path below `SPHERE_LEAN_BYTES`, lean above it -- unaffected
by this call, `build_sphere` decides).
"""
function ml_sphere(d::Integer, T::Type)
    inp = build_sphere(d; valence = 3, T)
    return (inp.Γ, inp.Ω, inp.ch)
end

"""
    ml_solve(Γ, Ω, ch; seed) -> (; seconds, nullity, certified, status, stage_bytes)

One `derTrOpsReduced(:QuickDer, ...)`, matrix-free (as at these sizes: the
dense budget is forced to zero so the ledger measures the branch the movie
regime actually takes), with `QDN_STAGE_BYTES` on so the per-stage allocation
is available afterward -- `stage_bytes` is a COPY of the dict, taken before
the kernel resets the `Ref` to `nothing` at the end of its own call.
"""
function ml_solve(Γ::ITensor, Ω::TransverseOps, ch::AbstractMatrix; seed::Integer = 20260908)
    Dleto.QDN_DENSE_BUDGET_BYTES[] = 0.0
    Random.seed!(seed)
    method = Dleto.get_derivation_method(:QuickDer; whiten = true, solver = :AutoSolver,
                                         verify = :random, seed = seed)
    bytes = Dict{Symbol,NTuple{2,Float64}}()
    Dleto.QDN_STAGE_BYTES[] = bytes
    Dleto.QDN_STAGE_TIMES[] = Dict{Symbol,Float64}()
    t0 = time()
    nullity = 0
    certified = false
    status = "ok"
    try
        (_, _, ders, rep) = Dleto.derTrOpsReduced(method, Ω, ch, Γ; tol = 1e-6,
                                                   return_diagnostics = true)
        nullity = size(ders, 2)
        certified = rep.verdict.certified
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
    end
    secs = time() - t0
    stage_bytes = copy(bytes)
    Dleto.QDN_STAGE_BYTES[] = nothing
    Dleto.QDN_STAGE_TIMES[] = nothing
    return (; seconds = secs, nullity, certified, status, stage_bytes)
end

"""
    ml_row(label, T, tensor_bytes, build) -> NamedTuple

One ledger row: full GC before the build (so the delta measured is THIS
shape's, not carryover from the previous one), `Sys.maxrss()` before and
after the WHOLE build+solve, and the solve's own per-stage bytes.

At the tensor sizes this ledger measures (tens of MB), the ABSOLUTE process
peak is dominated by the Julia runtime and package floor (~1-2 GB: BLAS,
ARPACK, ITensors, the GC's own heap-size hint), not by the tensor -- so the
number that isolates what THIS call cost is the DELTA,
`Sys.maxrss()` after minus before, which is what the ratio column below is
built from.  The absolute peak is still recorded (`peak_GB`), for the case
where a run is large enough that the floor is negligible next to it.
"""
function ml_row(label::AbstractString, T::Type, tensor_bytes::Real, build)
    GC.gc(); GC.gc()
    rss0 = Sys.maxrss()
    (Γ, Ω, ch) = build()
    r = ml_solve(Γ, Ω, ch)
    peak_gb = Sys.maxrss() * ML_GB
    base_gb = rss0 * ML_GB
    delta_gb = max(peak_gb - base_gb, 0.0)
    tensor_gb = tensor_bytes * ML_GB
    row = (; label, T = string(T), tensor_GB = tensor_gb, base_GB = base_gb,
             peak_GB = peak_gb, delta_GB = delta_gb, ratio = delta_gb / tensor_gb,
             seconds = r.seconds, nullity = r.nullity, certified = r.certified,
             status = r.status, stage_bytes = r.stage_bytes)
    @printf("%-24s %-8s tensor=%7.4f GB base=%6.3f GB peak=%7.3f GB delta=%6.3f GB ratio=%6.2fx t=%6.2fs nullity=%d certified=%s %s\n",
            row.label, row.T, row.tensor_GB, row.base_GB, row.peak_GB, row.delta_GB,
            row.ratio, row.seconds, row.nullity, row.certified, row.status)
    return row
end

function mlmain()
    rows = NamedTuple[]

    # 640x480x90x3 is NOT one of the two sizes the task specifies -- it is
    # added because at 320x240x{10,30}x3 the tensor (4-52 MB) is far smaller
    # than the ~1-2 GB Julia/BLAS/ARPACK process floor, so `Sys.maxrss()`'s
    # page-granularity delta is pure noise there (see the README: the ratio
    # column at the small sizes does not move in the expected direction).
    # 640x480x90x3 Float32 is 331 MB, large enough that a removed 331 MB
    # promotion copy is visible over that same floor, and it is the exact
    # shape session 4 already measured (`docs/CONTEXT.md`, "the lean sphere
    # build"), so its Float32 numbers here are a cross-check against that
    # entry, not a new claim.
    for (H, W, F) in ((320, 240, 10), (320, 240, 30), (640, 480, 90))
        for T in (Float64, Float32, Float16)
            tb = float(H) * W * F * 3 * sizeof(T)
            push!(rows, ml_row("video $(H)x$(W)x$(F)x3", T, tb,
                                () -> ml_video(H, W, F, T)))
        end
    end

    for d in (100, 150)
        for T in (Float64, Float32, Float16)
            tb = float(d)^3 * sizeof(T)
            push!(rows, ml_row("sphere d=$d valence=3", T, tb, () -> ml_sphere(d, T)))
        end
    end

    outdir = joinpath(@__DIR__, "reports", "2026-09-08", "memory")
    mkpath(outdir)
    stage_order = [:upload, :sketch, :whiten, :restricted, :solve, :lift, :filter,
                   :verify, :restrict_ops]
    csv = joinpath(outdir, "ledger.csv")
    open(csv, "w") do io
        println(io, join(vcat(["label", "T", "tensor_GB", "base_GB", "peak_GB", "delta_GB",
                                "ratio", "seconds", "nullity", "certified", "status"],
                               [string(s, "_alloc_GB") for s in stage_order]), ","))
        for row in rows
            sb = row.stage_bytes
            stagevals = [haskey(sb, s) ? sb[s][1] * ML_GB : 0.0 for s in stage_order]
            println(io, join(vcat([row.label, row.T, row.tensor_GB, row.base_GB,
                                    row.peak_GB, row.delta_GB, row.ratio, row.seconds,
                                    row.nullity, row.certified, row.status], stagevals), ","))
        end
    end
    println("wrote ", csv)
    return rows
end

abspath(PROGRAM_FILE) == (@__FILE__) && mlmain()
