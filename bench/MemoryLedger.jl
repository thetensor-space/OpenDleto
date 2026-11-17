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

const ML_STAGE_ORDER = [:upload, :sketch, :whiten, :restricted, :solve, :lift, :filter,
                        :verify, :restrict_ops]
const ML_HEADER = vcat(["label", "T", "tensor_GB", "base_GB", "peak_GB", "delta_GB",
                        "ratio", "seconds", "nullity", "certified", "status"],
                       [string(s, "_alloc_GB") for s in ML_STAGE_ORDER])
const ML_CASES = [
    ("video-320x240x10", () -> (320, 240, 10)),
    ("video-320x240x30", () -> (320, 240, 30)),
    # NOT one of the two sizes the task specifies -- added because at
    # 320x240x{10,30}x3 the tensor (4-52 MB) is far smaller than the ~1-2 GB
    # Julia/BLAS/ARPACK process floor, so `Sys.maxrss()`'s page-granularity
    # delta is dominated by that floor, not by the tensor (see the README).
    # 640x480x90x3 Float32 is 331 MB, large enough to show against the same
    # floor, and it is the exact shape docs/CONTEXT.md already measured, so
    # its Float32 row here is a cross-check, not a new claim.
    ("video-640x480x90", () -> (640, 480, 90)),
]

_ml_row_line(row) = join(vcat([row.label, row.T, row.tensor_GB, row.base_GB,
                               row.peak_GB, row.delta_GB, row.ratio, row.seconds,
                               row.nullity, row.certified, row.status],
                              [haskey(row.stage_bytes, s) ? row.stage_bytes[s][1] * ML_GB : 0.0
                               for s in ML_STAGE_ORDER]), ",")

"""
    ml_append(csv, row)

Append one row to the CSV, writing the header first if the file is new.
Used by the `--case` CLI form, where EACH row is its own fresh process (see
the file header): appending, not rewriting, is what lets a bash loop of
separate `bench/jl` invocations build up one ledger.
"""
function ml_append(csv::AbstractString, row)
    isnew = !isfile(csv)
    open(csv, "a") do io
        isnew && println(io, join(ML_HEADER, ","))
        println(io, _ml_row_line(row))
    end
end

"""
    mlmain(args)

`args` empty: run every case in ONE process (loops all shapes x eltypes) and
OVERWRITE the CSV.  Convenient for a quick look, but `Sys.maxrss()` is a
process HIGH-WATER MARK that Julia rarely returns to the OS, so once a later
case's peak is not larger than an earlier one's, `delta_GB` reads zero even
though that case allocated plenty -- measured directly: running the sizes in
one process shows exactly this (bench/reports/2026-09-08/memory/README.md,
"why every row is its own process").

`args = ["sphere", d, T]` or `["video", label, T]` (label from `ML_CASES`,
e.g. `video-640x480x90`): run exactly ONE case and APPEND it to the CSV --
each invocation is its OWN `bench/jl` process, so `Sys.maxrss()` is that
process's alone and the confound above does not apply.  This is the form the
reproduce commands in the README actually use.
"""
function mlmain(args)
    outdir = joinpath(@__DIR__, "reports", "2026-09-08", "memory")
    mkpath(outdir)
    csv = joinpath(outdir, "ledger.csv")

    if !isempty(args) && args[1] == "sphere"
        d = parse(Int, args[2])
        T = args[3] == "Float64" ? Float64 : args[3] == "Float32" ? Float32 : Float16
        tb = float(d)^3 * sizeof(T)
        row = ml_row("sphere d=$d valence=3", T, tb, () -> ml_sphere(d, T))
        ml_append(csv, row)
        println("appended -> ", csv)
        return [row]
    elseif !isempty(args) && args[1] == "video"
        label = args[2]
        (H, W, F) = only(f() for (l, f) in ML_CASES if l == label)
        T = args[3] == "Float64" ? Float64 : args[3] == "Float32" ? Float32 : Float16
        tb = float(H) * W * F * 3 * sizeof(T)
        row = ml_row("video $(H)x$(W)x$(F)x3", T, tb, () -> ml_video(H, W, F, T))
        ml_append(csv, row)
        println("appended -> ", csv)
        return [row]
    end

    rows = NamedTuple[]
    for (label, shape) in ML_CASES
        (H, W, F) = shape()
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
    open(csv, "w") do io
        println(io, join(ML_HEADER, ","))
        for row in rows
            println(io, _ml_row_line(row))
        end
    end
    println("wrote ", csv, " (one process, see the docstring for the caveat)")
    return rows
end

abspath(PROGRAM_FILE) == (@__FILE__) && mlmain(ARGS)
