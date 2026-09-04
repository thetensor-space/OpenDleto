#
# Serial re-timing of the leading stratification configurations on a quiet
# machine, after the 2026-09-03 calibration work (bench/reports/*.md).
#
# Same input and metric as bench/SphereHarness.jl.  One process, BLAS pinned,
# every configuration warmed up per element type before anything is timed, so
# the numbers are full runs of compiled code with no CPU contention.
#
# Configurations:
#   Float64  SVDSolver (dense), LUSolver (column-pivoted QR, dense),
#            ArpackSolver, KrylovSolver (block Lanczos), AutoSolver (the new
#            calibration), QuickDer (solve-and-lift, intersected with Ω)
#   Float32  ArpackSolver, KrylovSolver, SVDSolver, QuickDer
#
# Usage:
#   julia -t 4 --project=. bench/StratifyLeadersProfile.jl [dims...] [--tol=1e-8] [--seeds=1]
#        [--only=Auto/F64,QuickDer/F64] [--csv=path]
#   default dims: 20 30 40 50 64; default: every configuration below
#
# Prints a row the moment it is measured; appends to
# bench/stratify-leaders-profile.csv.
#
using KrylovKit
using IterativeSolvers
using Arpack
using LinearAlgebra
using Printf
include(joinpath(@__DIR__, "SphereHarness.jl"))

BLAS.set_num_threads(Threads.nthreads())

const TOL   = let a = filter(startswith("--tol="), ARGS); isempty(a) ? 1e-8 : parse(Float64, a[1][7:end]) end
const SEEDS = let a = filter(startswith("--seeds="), ARGS); isempty(a) ? 1 : parse(Int, a[1][9:end]) end
const DIMS  = let a = filter(x -> !startswith(x, "--"), ARGS); isempty(a) ? [20, 30, 40, 50, 64] : parse.(Int, a) end
const ONLY  = let a = filter(startswith("--only="), ARGS); isempty(a) ? String[] : split(a[1][8:end], ",") end
const CSV   = let a = filter(startswith("--csv="), ARGS); isempty(a) ? joinpath(@__DIR__, "stratify-leaders-profile.csv") : a[1][7:end] end
const BUDGET = 600.0     # drop a configuration for larger d once it exceeds this

# (label, element type, run_stratify keywords)
const CONFIGS = [
    ("SVD/F64",      Float64, (; solver = :SVDSolver)),
    ("LU-CPQR/F64",  Float64, (; solver = :LUSolver)),
    ("Arpack/F64",   Float64, (; solver = :ArpackSolver)),
    ("Krylov/F64",   Float64, (; solver = :KrylovSolver)),
    ("Auto/F64",     Float64, (; solver = :AutoSolver)),
    ("QuickDer/F64", Float64, (; method = :QuickDer)),
    ("Arpack/F32",   Float32, (; solver = :ArpackSolver)),
    ("Krylov/F32",   Float32, (; solver = :KrylovSolver)),
    ("SVD/F32",      Float32, (; solver = :SVDSolver)),
    ("QuickDer/F32", Float32, (; method = :QuickDer)),
]

function main()
    println("Dleto stratification: leaders, serial, quiet machine")
    println("  threads = $(Threads.nthreads()), BLAS = $(BLAS.get_num_threads()), tol = $TOL, seeds = $SEEDS")
    println("  dims = $DIMS")
    isfile(CSV) || open(CSV, "w") do io
        println(io, "d,seed,config,T,seconds,bytes,peak_rss_gb,nullity,lsq_err,support,perm_ok,status")
    end

    print("\nWarm-up (d = 8, both element types, two passes) ... "); flush(stdout)
    tw = @elapsed for T in (Float64, Float32)
        cfgs = [CONFIGS[k][3] for k in 1:length(CONFIGS)
                if CONFIGS[k][2] === T && (isempty(ONLY) || CONFIGS[k][1] in ONLY)]
        isempty(cfgs) || warmup!(cfgs; T, passes = 2)
    end
    @printf("done (%.1fs, not counted)\n", tw)

    active = [k for k in 1:length(CONFIGS) if isempty(ONLY) || CONFIGS[k][1] in ONLY]
    isempty(active) && error("--only matched no configuration; labels: " * join(first.(CONFIGS), ", "))
    for d in DIMS
        isempty(active) && break
        println()
        @printf("==== d = %d   op dim = %d   %s\n", d, 3 * d * (d + 1) ÷ 2, Libc.strftime("%H:%M:%S", time()))
        @printf("%-14s %4s %9s %9s %8s %5s %9s %8s %5s  %s\n",
                "config", "seed", "time(s)", "alloc(GB)", "rss(GB)", "null", "lsq_err", "support", "perm", "status")
        flush(stdout)
        worst = Dict{Int,Float64}()
        for s in 1:SEEDS
            seed = d + 1000 * (s - 1)
            inputs = Dict(T => build_sphere(d; T, seed) for T in (Float64, Float32))
            for k in active
                label, T, kw = CONFIGS[k]
                r = run_stratify(inputs[T]; kw..., tol = TOL)
                rss = Sys.maxrss() / 2^30
                @printf("%-14s %4d %9.3f %9.3f %8.3f %5d %9.2e %8.4f %5s  %s\n",
                        label, seed, r.seconds, r.bytes / 2^30, rss, r.nullity, r.lsq_err, r.support,
                        r.perm_ok ? "yes" : "no", r.status)
                flush(stdout)
                open(CSV, "a") do io
                    @printf(io, "%d,%d,%s,%s,%.6f,%d,%.4f,%d,%.6e,%.6f,%s,\"%s\"\n",
                            d, seed, label, T, r.seconds, r.bytes, rss, r.nullity, r.lsq_err, r.support,
                            r.perm_ok, r.status)
                end
                worst[k] = max(get(worst, k, 0.0), r.seconds)
            end
        end
        slow = [k for k in active if worst[k] > BUDGET]
        for k in slow
            println("  dropping $(CONFIGS[k][1]): $(round(worst[k]; digits = 1))s exceeds $(BUDGET)s")
        end
        active = filter(k -> !(k in slow), active)
    end
    println("\nFinished. Results in $CSV")
end

main()
