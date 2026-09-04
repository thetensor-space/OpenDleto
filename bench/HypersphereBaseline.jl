#
# HypersphereBaseline -- solver comparison on the valence-n hypersphere octant.
#
# Runs `:SylverLining` with `:AutoSolver`, `:ArpackSolver`, `:SVDSolver`, and
# `:QuickDer` (Liu's solve-and-lift, `src/solvers/FastDer3Valent.jl`, valence 3
# only -- at any other valence it errors, and that error string is recorded as
# the row's `status` rather than crashing the sweep) on
# `bench/SphereHarness.jl`'s `build_sphere(d; valence)`.
#
# `:ArpackSolver` needs the `Arpack` package loaded to be reachable
# (`src/solvers/NullSolvers.jl`'s solver registry); this script tries to load
# it and, if the active project does not have it resolved, falls back to
# recording the registry's own "unavailable" error as the status instead of
# crashing.
#
# Usage:
#   bench/jl bench/HypersphereBaseline.jl [dims...] [--valence=4] [--seeds=1]
#        [--tol=1e-8] [--only=AutoSolver,QuickDer] [--csv=path]
#   default dims: 8 10 12 16 20; default valence: 3; default: every solver below
#
# Warms up every active configuration on d = 6 (two passes, not counted) so
# the timed rows are full runs of compiled code.  Prints a row the moment it
# is measured; appends to bench/hypersphere-baseline.csv.
#
# NOTE ON TIMINGS: run through `bench/jl`, which pins 2 Julia threads / 2
# OpenBLAS threads per process -- these are 2-thread timings.  Compare rows
# within one run of this script, not against numbers taken with more threads.
#
using LinearAlgebra
using Printf
include(joinpath(@__DIR__, "SphereHarness.jl"))

# Best-effort: if Arpack is not a resolvable dependency of the active
# project, `:ArpackSolver` simply stays unregistered and every row for it
# reports the solver registry's own clean error as `status` (see
# `src/solvers/NullSolvers.jl`, `solve(L, sym::Symbol)`).
try
    @eval using Arpack
catch e
    @warn "Arpack did not load; :ArpackSolver rows will report the registry's " *
          "\"unavailable\" error instead of a real timing" exception = e
end

BLAS.set_num_threads(Threads.nthreads())

const VALENCE = let a = filter(startswith("--valence="), ARGS); isempty(a) ? 3 : parse(Int, a[1][11:end]) end
const TOL     = let a = filter(startswith("--tol="), ARGS); isempty(a) ? 1e-8 : parse(Float64, a[1][7:end]) end
const SEEDS   = let a = filter(startswith("--seeds="), ARGS); isempty(a) ? 1 : parse(Int, a[1][9:end]) end
const DIMS    = let a = filter(x -> !startswith(x, "--"), ARGS); isempty(a) ? [8, 10, 12, 16, 20] : parse.(Int, a) end
const ONLY    = let a = filter(startswith("--only="), ARGS); isempty(a) ? String[] : split(a[1][8:end], ",") end
const CSV     = let a = filter(startswith("--csv="), ARGS); isempty(a) ? joinpath(@__DIR__, "hypersphere-baseline.csv") : a[1][7:end] end

# (label, run_stratify keywords)
const CONFIGS = [
    ("AutoSolver",   (; method = :SylverLining, solver = :AutoSolver)),
    ("ArpackSolver", (; method = :SylverLining, solver = :ArpackSolver)),
    ("KrylovSolver", (; method = :SylverLining, solver = :KrylovSolver)),
    ("SVDSolver",    (; method = :SylverLining, solver = :SVDSolver)),
    ("QuickDer",     (; method = :QuickDer)),
]

function main()
    println("Dleto hypersphere baseline -- valence $VALENCE (2-thread timings under bench/jl)")
    println("  threads = $(Threads.nthreads()), BLAS = $(BLAS.get_num_threads()), tol = $TOL, seeds = $SEEDS")
    println("  dims = $DIMS")
    isfile(CSV) || open(CSV, "w") do io
        println(io, "# timings are 2-thread timings (bench/jl pins 2 Julia + 2 OpenBLAS threads)")
        println(io, "d,seed,valence,solver,seconds,bytes,nullity,lsq_err,status")
    end

    active = [k for k in 1:length(CONFIGS) if isempty(ONLY) || CONFIGS[k][1] in ONLY]
    isempty(active) && error("--only matched no configuration; labels: " * join(first.(CONFIGS), ", "))

    print("\nWarm-up (d = 6, valence $VALENCE, two passes) ... "); flush(stdout)
    tw = @elapsed warmup!([CONFIGS[k][2] for k in active]; d = 6, valence = VALENCE, passes = 2)
    @printf("done (%.1fs, not counted)\n", tw)

    for d in DIMS
        println()
        @printf("==== d = %d   valence = %d   %s\n", d, VALENCE, Libc.strftime("%H:%M:%S", time()))
        @printf("%-12s %4s %9s %9s %5s %9s  %s\n",
                "solver", "seed", "time(s)", "alloc(GB)", "null", "lsq_err", "status")
        flush(stdout)
        for s in 1:SEEDS
            seed = d + 1000 * (s - 1)
            inp = build_sphere(d; valence = VALENCE, seed)
            for k in active
                label, kw = CONFIGS[k]
                r = run_stratify(inp; kw..., tol = TOL)
                @printf("%-12s %4d %9.3f %9.3f %5d %9.2e  %s\n",
                        label, seed, r.seconds, r.bytes / 2^30, r.nullity, r.lsq_err, r.status)
                flush(stdout)
                open(CSV, "a") do io
                    @printf(io, "%d,%d,%d,%s,%.6f,%d,%d,%.6e,\"%s\"\n",
                            d, seed, VALENCE, label, r.seconds, r.bytes, r.nullity, r.lsq_err, r.status)
                end
            end
        end
    end
    println("\nFinished. Results in $CSV")
end

main()
