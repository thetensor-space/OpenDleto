#
# Float16Metal.jl -- does the Apple GPU do anything special for Float16 on our
# sketch/lift contractions, in throughput AND in accuracy?
#
# The movie regime (640x480xFx3, F up to 1800) runs in Float16 for storage
# (3.3 GB vs 6.6 GB in Float32 at F = 1800).  The hot op is the sketch/lift
# contraction: an unfolding of the tensor along one axis, A (m x k), times a
# small random orthogonal sketch W (k x r).  This script measures, on an M4
# Max (40 GPU cores) via Metal.jl/MPS:
#
#   1. THROUGHPUT: median-of-5 `Metal.@sync mul!(C, A, W)` for realistic
#      movie-regime shapes and a 4096^3 square control, Float16 vs Float32;
#      plus `permutedims` and host<->device transfer of a 640x480x90x3 tensor.
#   2. ACCURACY: the same product on the CPU in Float64 (from the SAME
#      Float64-drawn, then T-rounded, inputs) as the reference, at
#      accumulation lengths k = 480, 1800, 8192 -- to see whether MPS
#      accumulates fp16 GEMM in fp16 (error ~ sqrt(k)*1e-3, growing) or fp32
#      (error ~ 5e-4 flat, one half-rounding of the output).  Also tries the
#      mixed contract Metal.jl/MPS would need for our pipeline: Float16
#      operands, C::MtlMatrix{Float32} (fp32 accumulate, no final half
#      rounding) -- recorded as accepted/rejected either way.
#   3. CPU Float16 control: one reduced movie-regime shape with
#      `Array{Float16}` `*`, to see whether the CPU path is BLAS-backed or a
#      slow generic fallback.
#
# Run through the wrapper (Metal is a weakdep of Dleto; this script does not
# need Dleto itself, only `using Metal`):
#
#   JL_PROJECT=$(pwd) JL_THREADS=4 JL_HEAP=3G JL_RSS_LIMIT_GB=6 \
#     bench/jl bench/Float16Metal.jl
#
# One process, all arrays under 2 GB (the largest is 720 MB), whole campaign
# well under a minute of GPU time -- the wall clock here is dominated by
# Metal.jl's first-use shader compilation, not the measured kernels.
#
# One row per shape x eltype x operation goes to
#   bench/reports/2026-09-04/float16-metal/float16-metal.csv
#
using Metal
using LinearAlgebra
using Printf
using Statistics
using Random

const OUTDIR = joinpath(@__DIR__, "reports", "2026-09-04", "float16-metal")
const CSVPATH = joinpath(OUTDIR, "float16-metal.csv")
mkpath(OUTDIR)

Random.seed!(20260904)

println("Metal.functional() = ", Metal.functional())
if !Metal.functional()
    println("STOP: Metal.jl is loaded but not functional on this machine; " *
            "no GPU measurements are possible.  That is itself the answer.")
    exit(1)
end
println("device = ", Metal.device().name)

# --- CSV ----------------------------------------------------------------

const COLS = ["section", "shape", "m", "k", "r", "eltype", "op", "seconds",
              "gflops", "gbps", "max_rel_err", "rms_rel_err", "accepted",
              "note"]

isfile(CSVPATH) && rm(CSVPATH)   # one clean run, not an append log
function emit(row::Dict)
    fresh = !isfile(CSVPATH)
    open(CSVPATH, "a") do io
        fresh && println(io, join(COLS, ","))
        println(io, join([string(get(row, c, "")) for c in COLS], ","))
    end
end

# --- timing ---------------------------------------------------------------

"""
    median_metal(f; n = 5)

Warm up once (compiles the MPS kernel / shader), then time `n` more calls of
`f()` each wrapped in `Metal.@sync` so the timer sees completion, not just
enqueue.  Returns the median second.
"""
function median_metal(f; n::Int = 5)
    f()                       # warmup / compile, untimed
    ts = Float64[]
    for _ in 1:n
        push!(ts, @elapsed (Metal.@sync f()))
    end
    return median(ts)
end

function median_cpu(f; n::Int = 5)
    f()
    ts = Float64[]
    for _ in 1:n
        push!(ts, @elapsed f())
    end
    return median(ts)
end

# --- section 1: GEMM throughput -------------------------------------------
#
# Shapes from the movie regime (640x480xFx3, Float16 storage): two
# orientations of the frame-axis unfolding, the accumulation-length-1800 case
# (a frame-axis contraction, m subsampled to keep the array under 2 GB), the
# channel/spatial unfolding, and a 4096^3 square control comparable to
# published GEMM rates.  Every A is under 1 GB in both eltypes.
const SHAPES = [
    ("unfold-1 (640 x 480*90*3)",       640,      480 * 90 * 3, 33),
    ("unfold-1T (480*90*3 x 640)",      480 * 90 * 3, 640,       33),
    ("frame-accum (1e5 x 1800)",        100_000,  1800,          33),
    ("unfold-3 (640*480 x 270)",        640 * 480, 270,          16),
    ("square control (4096^3)",         4096,     4096,        4096),
]

function bytes_of(T, m, k, r)
    return (m * k + k * r + m * r) * sizeof(T)
end

println("\n--- section 1: GEMM throughput (Metal.jl / MPS) ---")
@printf("%-28s %-8s %10s %10s %10s\n", "shape", "eltype", "seconds",
        "GFlop/s", "GB/s")
for (name, m, k, r) in SHAPES
    A64 = randn(Float64, m, k)
    W64 = randn(Float64, k, r)
    for T in (Float16, Float32)
        AT = T.(A64); WT = T.(W64)
        Ad = MtlArray(AT); Wd = MtlArray(WT)
        Cd = MtlArray{T}(undef, m, r)
        t = median_metal(() -> mul!(Cd, Ad, Wd))
        gflops = 2.0 * m * k * r / t / 1e9
        gbps = bytes_of(T, m, k, r) / t / 1e9
        @printf("%-28s %-8s %10.6f %10.2f %10.2f\n", name, T, t, gflops, gbps)
        emit(Dict("section" => "throughput", "shape" => name, "m" => m,
                  "k" => k, "r" => r, "eltype" => T, "op" => "mul!",
                  "seconds" => t, "gflops" => round(gflops; digits = 3),
                  "gbps" => round(gbps; digits = 3), "accepted" => true))
        Ad = nothing; Wd = nothing; Cd = nothing
    end
    # Mixed contract: Float16 operands, Float32 accumulate/output -- the ideal
    # storage-vs-arithmetic split for the movie pipeline.  Try it, record
    # whichever way it goes.
    A16 = Float16.(A64); W16 = Float16.(W64)
    Ad = MtlArray(A16); Wd = MtlArray(W16)
    Cd32 = MtlArray{Float32}(undef, m, r)
    try
        t = median_metal(() -> mul!(Cd32, Ad, Wd))
        gflops = 2.0 * m * k * r / t / 1e9
        gbps = (m * k + k * r) * sizeof(Float16) / t / 1e9 +
               (m * r) * sizeof(Float32) / t / 1e9
        @printf("%-28s %-8s %10.6f %10.2f %10.2f  (mixed f16->f32)\n",
                name, "mixed", t, gflops, gbps)
        emit(Dict("section" => "throughput", "shape" => name, "m" => m,
                  "k" => k, "r" => r, "eltype" => "f16in_f32out",
                  "op" => "mul!_mixed", "seconds" => t,
                  "gflops" => round(gflops; digits = 3),
                  "gbps" => round(gbps; digits = 3), "accepted" => true))
    catch e
        msg = replace(sprint(showerror, e), "," => ";")[1:min(end, 150)]
        @printf("%-28s %-8s  MIXED mul! REJECTED: %s\n", name, "mixed", msg)
        emit(Dict("section" => "throughput", "shape" => name, "m" => m,
                  "k" => k, "r" => r, "eltype" => "f16in_f32out",
                  "op" => "mul!_mixed", "accepted" => false, "note" => msg))
    end
    A64 = nothing; W64 = nothing; Ad = nothing; Wd = nothing; Cd32 = nothing
    GC.gc(false)
end

# --- section 1b: permutedims and host<->device transfer --------------------
#
# The known bottleneck from last night's SylverLining GPU work: permutedims!
# on the device beat the GEMM it feeds.  640x480x90x3 is a movie-regime
# fragment (331 MB Float32, 165 MB Float16).
println("\n--- section 1b: permutedims and host<->device transfer ---")
tdims = (640, 480, 90, 3)
tperm = (2, 1, 3, 4)
for T in (Float16, Float32)
    host = T.(randn(Float32, tdims))
    nbytes = length(host) * sizeof(T)

    dev = MtlArray(host)
    t_pd = median_metal(() -> permutedims(dev, tperm))
    gbps_pd = 2 * nbytes / t_pd / 1e9     # read + write
    @printf("permutedims  %-8s %10.6f s  %8.2f GB/s\n", T, t_pd, gbps_pd)
    emit(Dict("section" => "permutedims", "shape" => "640x480x90x3", "m" => "",
              "k" => "", "r" => "", "eltype" => T, "op" => "permutedims",
              "seconds" => t_pd, "gbps" => round(gbps_pd; digits = 2),
              "accepted" => true))

    t_up = median_metal(() -> MtlArray(host))
    gbps_up = nbytes / t_up / 1e9
    @printf("host->device %-8s %10.6f s  %8.2f GB/s\n", T, t_up, gbps_up)
    emit(Dict("section" => "transfer", "shape" => "640x480x90x3", "eltype" => T,
              "op" => "host_to_device", "seconds" => t_up,
              "gbps" => round(gbps_up; digits = 2), "accepted" => true))

    t_down = median_metal(() -> Array(dev))
    gbps_down = nbytes / t_down / 1e9
    @printf("device->host %-8s %10.6f s  %8.2f GB/s\n", T, t_down, gbps_down)
    emit(Dict("section" => "transfer", "shape" => "640x480x90x3", "eltype" => T,
              "op" => "device_to_host", "seconds" => t_down,
              "gbps" => round(gbps_down; digits = 2), "accepted" => true))
    dev = nothing; host = nothing
    GC.gc(false)
end

# --- section 2: accuracy vs accumulation length k --------------------------
#
# Same Float64-drawn inputs, rounded independently to each T (so each T sees
# consistent, self-rounded operands); the Float64 reference is the exact
# product of THOSE rounded values, computed on the CPU.  m, r kept modest --
# this section is about the error's dependence on k, not about throughput.
println("\n--- section 2: accuracy vs accumulation length k ---")
@printf("%-8s %-14s %10s %10s\n", "k", "eltype", "max_rel", "rms_rel")
function relerr(Chost, Cref)
    d = Float64.(Chost) .- Cref
    denom = max.(abs.(Cref), 1e-30)
    rel = abs.(d) ./ denom
    return maximum(rel), sqrt(mean(rel .^ 2))
end
const KVALS = (480, 1800, 8192)
const ACC_M, ACC_R = 256, 33
for k in KVALS
    A64 = randn(Float64, ACC_M, k)
    W64 = randn(Float64, k, ACC_R)

    A16 = Float16.(A64); W16 = Float16.(W64)
    Cref16 = Float64.(A16) * Float64.(W16)
    Ad16 = MtlArray(A16); Wd16 = MtlArray(W16)

    Cd16 = MtlArray{Float16}(undef, ACC_M, ACC_R)
    Metal.@sync mul!(Cd16, Ad16, Wd16)
    me, rms = relerr(Array(Cd16), Cref16)
    @printf("%-8d %-14s %10.3e %10.3e\n", k, "Float16", me, rms)
    emit(Dict("section" => "accuracy", "shape" => "k=$k", "m" => ACC_M,
              "k" => k, "r" => ACC_R, "eltype" => "Float16", "op" => "mul!",
              "max_rel_err" => me, "rms_rel_err" => rms, "accepted" => true))

    A32 = Float32.(A64); W32 = Float32.(W64)
    Cref32 = Float64.(A32) * Float64.(W32)
    Ad32 = MtlArray(A32); Wd32 = MtlArray(W32)
    Cd32 = MtlArray{Float32}(undef, ACC_M, ACC_R)
    Metal.@sync mul!(Cd32, Ad32, Wd32)
    me32, rms32 = relerr(Array(Cd32), Cref32)
    @printf("%-8d %-14s %10.3e %10.3e\n", k, "Float32", me32, rms32)
    emit(Dict("section" => "accuracy", "shape" => "k=$k", "m" => ACC_M,
              "k" => k, "r" => ACC_R, "eltype" => "Float32", "op" => "mul!",
              "max_rel_err" => me32, "rms_rel_err" => rms32, "accepted" => true))

    # Mixed: same Float16 operands as above, Float32 accumulate/output,
    # compared against the SAME Float16-input reference (isolates the effect
    # of skipping the final half-rounding of the output).
    Cmix = MtlArray{Float32}(undef, ACC_M, ACC_R)
    try
        Metal.@sync mul!(Cmix, Ad16, Wd16)
        memix, rmsmix = relerr(Array(Cmix), Cref16)
        @printf("%-8d %-14s %10.3e %10.3e\n", k, "f16in_f32out", memix, rmsmix)
        emit(Dict("section" => "accuracy", "shape" => "k=$k", "m" => ACC_M,
                  "k" => k, "r" => ACC_R, "eltype" => "f16in_f32out",
                  "op" => "mul!_mixed", "max_rel_err" => memix,
                  "rms_rel_err" => rmsmix, "accepted" => true))
    catch e
        msg = replace(sprint(showerror, e), "," => ";")[1:min(end, 150)]
        emit(Dict("section" => "accuracy", "shape" => "k=$k", "m" => ACC_M,
                  "k" => k, "r" => ACC_R, "eltype" => "f16in_f32out",
                  "op" => "mul!_mixed", "accepted" => false, "note" => msg))
    end
    A64 = nothing; W64 = nothing
end

# --- section 3: CPU Float16 control -----------------------------------------
#
# One reduced movie-regime shape (a 1/27-scale version of the unfold-1 shape,
# to stay inside the campaign's wall-clock budget on a shared machine):
# 640 x 4800 x 33.  This documents the CPU path's character, not its rate at
# full scale.
println("\n--- section 3: CPU Float16 control ---")
cm, ck, cr = 640, 4800, 33
Ac64 = randn(Float64, cm, ck); Wc64 = randn(Float64, ck, cr)
for T in (Float16, Float32, Float64)
    Ac = T.(Ac64); Wc = T.(Wc64)
    t = median_cpu(() -> Ac * Wc)
    gflops = 2.0 * cm * ck * cr / t / 1e9
    @printf("CPU %-8s %6dx%6dx%3d: %9.5f s  %8.3f GFlop/s\n",
            T, cm, ck, cr, t, gflops)
    emit(Dict("section" => "cpu_control", "shape" => "640x4800x33 (1/27 scale)",
              "m" => cm, "k" => ck, "r" => cr, "eltype" => T, "op" => "*",
              "seconds" => t, "gflops" => round(gflops; digits = 3),
              "accepted" => true))
end

println("\nwrote $CSVPATH")
