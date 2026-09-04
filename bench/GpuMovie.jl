#
# GpuMovie -- the movie regime, stage by stage, CPU against Metal.
#
# The question this answers: of the wall time a `640 x 480 x F x 3` QuickDer run
# spends, how much is TENSOR-SHAPED work (the sketches, the lift's pair tensors,
# the verification -- everything that makes a pass over `d^n` and therefore
# grows with the frame count), how much is the restricted eigensolve, and what
# does moving the first group to the GPU actually buy?
#
# TWO THINGS MAKE THIS DIFFERENT FROM bench/WhitenedRestriction.jl, and both are
# the reason it is a separate driver rather than a flag on that one.
#
#   1. IT READS EVERY STAGE, not the four `whitened.csv` carries.  `verify` is
#      the biggest tensor stage on the device and `whitened.csv` has no column
#      for it, so the interesting number was invisible there.
#
#   2. IT WARMS UP ON THE SHAPE IT MEASURES.  This is not a nicety.  A device
#      run pays Metal pipeline specialization the first time each kernel meets a
#      new shape class, and on the movie tensor that cost is SECONDS -- measured
#      3.26 s inside `verify` on the first `640x480x90x3` GPU run of a process
#      and 0.018 s on the second, with a `10x10x10x3` warm-up in between that
#      did not cover it.  A cold number is a measurement of the compiler.
#
# The tensor stays out of the timed region on both devices; `upload` is a stage
# of its own so the host->device transfer is visible rather than hidden.
#
# RUN ONE CASE PER COMMAND, as in bench/Frontier.jl.
#
# Usage:
#   bench/jl bench/GpuMovie.jl <H> <W> <F> <T> <device> [reps]
#       e.g.  bench/jl bench/GpuMovie.jl 640 480 90 Float32 gpu
#       <T> is Float16, Float32 or Float64; <device> is cpu or gpu.  `reps`
#       (default 1) times that many warm runs and reports the median.
#
# -> bench/reports/2026-09-04/gpu-movie/gpu-movie.csv
#
using Arpack
using Dleto
using ITensors
using LinearAlgebra
using Logging
using Printf
using Random

include(joinpath(@__DIR__, "Frontier.jl"))       # der_residual, csv_header, THREAD_NOTE

const HAVE_GPU = try
    @eval using Metal
    Dleto.gpu_available()
catch
    false
end

const GREPORT_DIR = joinpath(@__DIR__, "reports", "2026-09-04", "gpu-movie")
mkpath(GREPORT_DIR)
const GCSV = joinpath(GREPORT_DIR, "gpu-movie.csv")

# Every stage `_qdn_solve_and_lift` and `derTrOpsReduced` mark, in the order they
# happen, so the CSV has one column each and a row reads left to right as the run.
const STAGES = [:upload, :sketch, :whiten, :restricted, :solve, :lift, :filter,
                :verify, :restrict_ops]

const GHEADER = "case,dims,valence,eltype,device,r,restricted_rows,restricted_cols," *
                "reps,seconds," * join(string.(STAGES) .* "_seconds", ",") *
                ",tensor_seconds,maxrss_GB,gpu_peak_GB,nullity,oracle_nullity," *
                "residual,status"

# --------------------------------------------------------------- GPU polling

"""
    gpu_poll_start() / gpu_poll_stop!(poll)

Peak `Metal.device().currentAllocatedSize` across a run, sampled from a task --
the same instrument `bench/D500Matrix.jl` uses.  A no-op without a GPU.
"""
function gpu_poll_start()
    peak = Ref(0)
    stop = Ref(false)
    HAVE_GPU || return (peak, stop, nothing)
    dev = Metal.device()
    t = Threads.@spawn begin
        while !stop[]
            try
                v = Int(dev.currentAllocatedSize)
                v > peak[] && (peak[] = v)
            catch
            end
            sleep(0.02)
        end
    end
    return (peak, stop, t)
end

function gpu_poll_stop!(poll)
    (peak, stop, t) = poll
    stop[] = true
    t === nothing || wait(t)
    return peak[]
end

# --------------------------------------------------------------- the timed call

"""
    timed_stages(Ω, ch, Γ; device) -> NamedTuple

One `derTrOpsReduced(:QuickDer, ...)` with the stage clock on.  `tensor_s` is
the sum of the stages that make a pass over the full tensor -- `sketch`, `lift`
and `verify` -- i.e. exactly the part a device port can move, as against
`solve`, which is the restricted eigensolve and stays on the host.
"""
function timed_stages(Ω, ch, Γ; device::Symbol, tol::Real = 1e-6)
    method = Dleto.get_derivation_method(:QuickDer; whiten = true, solver = :AutoSolver,
                                         device = device, verify = :random,
                                         seed = 20260904)
    stages = Dict{Symbol,Float64}()
    Dleto.QDN_STAGE_TIMES[] = stages
    status = "ok"
    nullity = 0
    resid = NaN
    local ders, expand_map
    GC.gc()
    poll = gpu_poll_start()
    st = @timed try
        Logging.with_logger(Logging.NullLogger()) do
            (_, expand_map, ders) = Dleto.derTrOpsReduced(method, Ω, ch, Γ; tol = tol)
        end
        nullity = size(ders, 2)
        if nullity > 0
            D = embedITensors(Ω, expand_map(ders[:, 1]))
            resid = der_residual(Γ, D, ch)
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
    end
    peak_bytes = gpu_poll_stop!(poll)
    Dleto.QDN_STAGE_TIMES[] = nothing
    tensor_s = sum(get(stages, s, 0.0) for s in (:sketch, :lift, :verify))
    return (; seconds = st.time, stages, tensor_s, nullity, resid, status,
              maxrss_GB = Sys.maxrss() / 2^30,
              gpu_peak_GB = device === :gpu ? peak_bytes / 2^30 : NaN)
end

# --------------------------------------------------------------- the case

function movie_case(H::Int, W::Int, F::Int, T::Type, device::Symbol, reps::Int)
    # The matrix-free restricted branch on BOTH devices.  The movie shape is
    # over the host dense budget anyway; the GPU budget has to be zeroed too or
    # a device run silently takes the Gram route instead and the two columns
    # stop comparing the same computation (it also puts a 4.8 GB restricted
    # matrix next to the tensor, which is what the RSS budget cannot hold).
    Dleto.QDN_DENSE_BUDGET_BYTES[] = 0.0
    Dleto.QDN_GPU_DENSE_BUDGET_BYTES[] = 0.0

    Random.seed!(20260904 + H + 100W + 10_000F)
    fr = [Index(H, "h"), Index(W, "w"), Index(F, "f"), Index(3, "c")]
    Γ = ITensor(Array{T}(randn(H, W, F, 3)), fr...)
    Ω = IndTransverseOps(fr, UniversalOp())
    ch = UniversalChisel(4)
    dims = [H, W, F, 3]
    r = Dleto._qdn_restriction_sizes(dims, Dleto.engaged(Matrix{Float64}(ch)), 4)

    # WARM UP ON THIS TENSOR, on this device -- see the header.  Discarded.
    warm = timed_stages(Ω, ch, Γ; device = device)
    @printf("warm-up (discarded): %.2f s  tensor %.3f s  %s\n",
            warm.seconds, warm.tensor_s, warm.status)
    flush(stdout)

    # The MEDIAN run, not the mean: the restricted eigensolve is the noisy term
    # (Arpack has been seen to take 17 s and 54 s on the same input in the same
    # process, see docs/CONTEXT.md on eigensolver outliers) and one outlier
    # would swamp an average of the stages this driver is actually about.
    runs = [timed_stages(Ω, ch, Γ; device = device) for _ in 1:reps]
    res = sort(runs; by = x -> x.seconds)[(length(runs) + 1) ÷ 2]

    rows = prod(r)
    cols = sum(Int[dims[a] * r[a] for a in 1:4])
    @printf("%-22s T=%-8s dev=%-4s r=%-16s seconds=%7.2f solve=%7.2f tensor=%6.3f \
(sketch %.3f lift %.3f verify %.3f) maxrss=%.2fGB gpu=%.2fGB nullity=%d resid=%.2e %s\n",
            "movie-$(H)x$(W)x$(F)x3", T, device, string(r), res.seconds,
            get(res.stages, :solve, NaN), res.tensor_s,
            get(res.stages, :sketch, NaN), get(res.stages, :lift, NaN),
            get(res.stages, :verify, NaN), res.maxrss_GB, res.gpu_peak_GB,
            res.nullity, res.resid, res.status)
    flush(stdout)

    csv_header(GCSV,
        "# movie regime, stage by stage, CPU vs Metal.  Every row is a WARM run: " *
        "the same tensor is solved once on the same device first and discarded, " *
        "because the first movie-shaped GPU run pays seconds of Metal pipeline " *
        "specialization (3.26 s inside verify, against 0.018 s warm).  " *
        "tensor_seconds = sketch + lift + verify, the part a device port can move.",
        GHEADER)
    open(GCSV, "a") do io
        @printf(io, "%s,\"%s\",%d,%s,%s,\"%s\",%d,%d,%d,%.6f",
                "movie-$(H)x$(W)x$(F)x3", string(dims), 4, T, device, string(r),
                rows, cols, reps, res.seconds)
        for s in STAGES
            @printf(io, ",%.6f", get(res.stages, s, NaN))
        end
        @printf(io, ",%.6f,%.6f,%.6f,%d,%d,%.6e,\"%s\"\n",
                res.tensor_s, res.maxrss_GB, res.gpu_peak_GB, res.nullity, 3,
                res.resid, res.status)
    end
    return nothing
end

# --------------------------------------------------------------- CLI

function gmain(args)
    length(args) >= 5 || error("usage: bench/jl bench/GpuMovie.jl <H> <W> <F> <T> " *
                               "<device> [reps]")
    T = args[4] == "Float16" ? Float16 : args[4] == "Float64" ? Float64 : Float32
    device = Symbol(args[5])
    device === :gpu && !HAVE_GPU &&
        error("device = :gpu asked for but Dleto.gpu_available() is false")
    movie_case(parse(Int, args[1]), parse(Int, args[2]), parse(Int, args[3]),
               T, device, length(args) >= 6 ? parse(Int, args[6]) : 1)
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && gmain(ARGS)
