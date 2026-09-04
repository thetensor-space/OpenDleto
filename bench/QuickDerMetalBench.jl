#
# bench/QuickDerMetalBench.jl -- QuickDer on the Apple GPU vs the CPU.
#
#   bench/jl bench/QuickDerMetalBench.jl [--only=NAME,NAME] [--csv=PATH]
#                                        [--devices=cpu,gpu] [--tol=1e-6]
#
# Everything here is Float32, because Metal is: Apple GPUs have no Float64 at
# all, so `device = :gpu` is an exploratory-precision run and the certified
# answer is the Float64 CPU one.  The CPU column is Float32 too, so the
# comparison is device-vs-device and not also precision-vs-precision.
#
# WHAT IS MEASURED.  `Dleto.QDN_STAGE_TIMES` (see `src/solvers/QuickDerN.jl`)
# is an opt-in per-stage clock that `derTrOpsReduced` and `GramSolver` fill in;
# this script turns it on, runs the real entry point once per (case, device),
# and writes the stages out.  Nothing is re-implemented here, so the totals are
# the totals a caller would see.  The stages:
#
#   upload        host -> device copy of Γ (zero on the CPU)
#   sketch        the cross sketches S_a = Γ ×_{b≠a} W_b, the d^n work
#   restricted    assembling the dense restricted matrix (host, both devices)
#   solve         the whole restricted null solve, i.e. GramSolver, of which
#                 gram / cholesky / subspace / ritz are the parts
#   lift          pair tensors (d^n, on device) + per-axis least squares (host)
#   filter        the lift-consistency nullspace (host)
#   verify        the Z-law check on `nslices` slices (on device)
#   restrict_ops  intersecting the universal answer with Ω (host)
#
# CAVEAT on the dense route.  The default host budget declines a restricted
# matrix over 0.5 GB and goes matrix free, which at valence 3 happens from
# about d = 130 in Float32.  A CPU-vs-GPU comparison has to run the SAME
# algorithm on both sides, so the script raises `QDN_DENSE_BUDGET_BYTES` to
# match the GPU budget.  Timings here are therefore "dense route at d = 200",
# not "what the default does at d = 200".
#
# CAVEAT on repeats.  Metal's kernel/pipeline cache warms over the first few
# calls even for identical shapes: measured on the verification stage at
# valence 3 d = 100, three back-to-back runs in one process took 6.32 s,
# 0.257 s, 0.055 s.  A single warm-up on a differently shaped case is NOT
# enough, so each (case, device) is run `--passes` times (2 by default) and the
# LAST pass is what is recorded.  Cases whose memory estimate is over 60% of
# the budget run once, and say so in the `status` column.
#
# CAVEAT on memory.  `bench/jl` kills the process over 16 GB RSS.  A device
# run holds Γ twice (host + device) and the restricted matrix twice, and
# `_qdn_ttm` on an axis other than the first allocates a full permuted copy of
# whatever it is given -- so the first cross sketch of a 4.8 GB tensor wants
# another 4.8 GB.  Every case therefore carries an estimate, is skipped when
# the estimate exceeds `--budget` (default 11 GB), and rows are written and
# flushed one at a time so a kill on the last case does not lose the rest.
#

using Dleto
using ITensors
using LinearAlgebra
using Random
using Printf

try
    using Metal
catch err
    @warn "Metal is not loadable; only the :cpu device can run." err
end

include(joinpath(@__DIR__, "SphereHarness.jl"))

const REPORT_DIR = joinpath(@__DIR__, "reports", "night-2026-09-03")

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
#
# `kind = :random` is a dense Gaussian tensor with the universal chisel and
# `UniversalOp()`; the oracle nullity is `n - 1` (the scalars).  `kind =
# :sphere` is the scrambled hypersphere octant of `bench/SphereHarness.jl`
# with `SymmetricOp()`, oracle nullity `n` and a stratification score
# (`lsq_err`), which is the end-to-end correctness measure.

struct Case
    name::String
    kind::Symbol
    dims::Vector{Int}
end

const CASES = Case[
    Case("v3-random-d100",   :random, [100, 100, 100]),
    Case("v3-random-d200",   :random, [200, 200, 200]),
    Case("v4-random-100c3",  :random, [100, 100, 100, 3]),
    Case("v3-sphere-d100",   :sphere, [100, 100, 100]),
    Case("v3-sphere-d200",   :sphere, [200, 200, 200]),
    Case("v4-random-200c3",  :random, [200, 200, 100, 3]),
    Case("v3-random-d300",   :random, [300, 300, 300]),
]

"""
    est_bytes(case, device) -> Float64

Peak host+device bytes this case is expected to want, so an impossible one is
skipped rather than killed.  Counts, in Float32: Γ (once per device it lives
on), one full permuted copy of Γ (what the first non-leading-axis mode product
allocates), and the dense restricted matrix plus its Gram plus the Cholesky's
copy of the Gram (once per device the solve runs on).
"""
function est_bytes(case::Case, device::Symbol)
    d = case.dims
    n = length(d)
    eng = trues(n)
    r = Dleto._qdn_restriction_sizes(d, eng, n)
    tensor = prod(float.(d)) * 4
    ncols = sum(float(d[a]) * r[a] for a in 1:n)
    rows = max(prod(float.(r)), ncols)
    mat = rows * ncols * 4
    gram = ncols * ncols * 4
    host = tensor + tensor + mat                      # Γ, permuted copy, matrix
    dev = device === :gpu ? (tensor + tensor + mat + 2 * gram) : (2 * gram)
    return host + dev
end

function build_case(case::Case)
    if case.kind === :sphere
        d = case.dims[1]
        inp = build_sphere(d; valence = length(case.dims), T = Float32)
        return (; inp.Γ, inp.Ω, inp.ch, inp, oracle = length(case.dims))
    end
    Random.seed!(hash(case.name) % 100000)
    fr = [Index(dd, "a$i") for (i, dd) in enumerate(case.dims)]
    Γ = ITensor(randn(Float32, case.dims...), fr...)
    Ω = IndTransverseOps(fr, UniversalOp())
    ch = UniversalChisel(length(case.dims))
    return (; Γ, Ω, ch, inp = nothing, oracle = length(case.dims) - 1)
end

const STAGES = (:upload, :sketch, :restricted, :solve, :gram, :cholesky,
                :subspace, :ritz, :lift, :filter, :verify, :restrict_ops)

"""One (case, device) measurement, with the stage clock on."""
function measure(case::Case, device::Symbol; tol::Real)
    c = build_case(case)
    GC.gc()
    st = Dict{Symbol,Float64}()
    Dleto.QDN_STAGE_TIMES[] = st
    nullity = 0
    lsq = NaN
    status = "ok"
    seconds = NaN
    try
        if case.kind === :sphere
            r = run_stratify(c.inp; method = :QuickDer, device = device, tol = tol)
            nullity = r.nullity
            lsq = r.lsq_err
            seconds = r.seconds
            status = r.status
        else
            m = Dleto.get_derivation_method(:QuickDer; device = device, seed = 11)
            seconds = @elapsed ((rΩ, em, ders) =
                derTrOpsReduced(m, c.Ω, c.ch, c.Γ; tol = tol))
            nullity = size(ders, 2)
        end
    catch err
        status = "error: " * first(split(sprint(showerror, err), '\n'))
    finally
        Dleto.QDN_STAGE_TIMES[] = nothing
    end
    rss = Sys.maxrss() / 2^30
    return (; seconds, nullity, lsq, status, st, rss)
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

function main(args)
    only = String[]
    csv = joinpath(REPORT_DIR, "quickder-metal.csv")
    devices = [:cpu, :gpu]
    tol = 1e-6
    budget = 11.0 * 2^30
    passes = 2
    for a in args
        if startswith(a, "--only=")
            only = split(a[8:end], ',')
        elseif startswith(a, "--csv=")
            csv = a[7:end]
        elseif startswith(a, "--devices=")
            devices = Symbol.(split(a[11:end], ','))
        elseif startswith(a, "--tol=")
            tol = parse(Float64, a[7:end])
        elseif startswith(a, "--budget=")
            budget = parse(Float64, a[10:end]) * 2^30
        elseif startswith(a, "--passes=")
            passes = parse(Int, a[10:end])
        else
            error("unknown argument $a")
        end
    end
    cases = isempty(only) ? CASES : [c for c in CASES if c.name in only]

    have_gpu = Dleto.gpu_available()
    have_gpu || (devices = filter(!=(:gpu), devices))
    println("threads = ", Threads.nthreads(), "  BLAS = ", BLAS.get_num_threads(),
            "  gpu_available = ", have_gpu)

    # The CPU side has to take the same dense route as the GPU side, or the
    # comparison is between two different algorithms (see the header).
    Dleto.QDN_DENSE_BUDGET_BYTES[] = Dleto.QDN_GPU_DENSE_BUDGET_BYTES[]

    # Warm up every (kind, device) pair so no row pays for compilation.
    print("warming up... ")
    for dev in devices
        warm = Case("warm3", :random, [24, 24, 24])
        measure(warm, dev; tol = 1e-6)
        measure(Case("warm4", :random, [12, 12, 12, 3]), dev; tol = 1e-6)
        measure(Case("warms", :sphere, [12, 12, 12]), dev; tol = 1e-6)
    end
    println("done")

    mkpath(dirname(csv))
    open(csv, "w") do io
        println(io, "# QuickDer CPU vs Apple GPU (Metal), Float32 throughout, " *
                    "$(Threads.nthreads()) Julia threads via bench/jl.")
        println(io, "# Seconds are wall clock; stage columns come from " *
                    "Dleto.QDN_STAGE_TIMES. gram/cholesky/subspace/ritz are " *
                    "parts of `solve`.")
        println(io, "case,kind,dims,device,seconds,nullity,oracle,lsq_err," *
                    join(string.(STAGES) .* "_s", ",") * ",maxrss_GB,status")
        flush(io)
        for case in cases, dev in devices
            eb = est_bytes(case, dev)
            if eb > budget
                @printf("%-18s %-4s SKIPPED: estimated %.1f GB > budget %.1f GB\n",
                        case.name, dev, eb / 2^30, budget / 2^30)
                println(io, "$(case.name),$(case.kind),\"$(join(case.dims,'x'))\",$dev," *
                            "NaN,0,0,NaN," * join(fill("NaN", length(STAGES)), ",") *
                            ",NaN,\"skipped: estimated $(round(eb/2^30; digits=1)) GB\"")
                flush(io)
                continue
            end
            np = eb > 0.6 * budget ? 1 : passes
            local r
            for _ in 1:np
                r = measure(case, dev; tol = tol)
                GC.gc()
            end
            note = np == 1 && passes > 1 ? " (single pass: memory)" : ""
            oracle = case.kind === :sphere ? length(case.dims) : length(case.dims) - 1
            @printf("%-18s %-4s %8.2fs  nullity %3d (oracle %d)  lsq %-10s  rss %.1f GB  %s\n",
                    case.name, dev, r.seconds, r.nullity, oracle,
                    isnan(r.lsq) ? "-" : @sprintf("%.1e", r.lsq), r.rss, r.status)
            for s in STAGES
                haskey(r.st, s) && @printf("      %-13s %8.3fs\n", s, r.st[s])
            end
            println(io, "$(case.name),$(case.kind),\"$(join(case.dims,'x'))\",$dev," *
                        "$(r.seconds),$(r.nullity),$oracle,$(r.lsq)," *
                        join([get(r.st, s, NaN) for s in STAGES], ",") *
                        ",$(r.rss),\"$(r.status)$note\"")
            flush(io)
            GC.gc()
        end
    end
    println("\nwrote ", csv)
    return nothing
end

main(ARGS)
