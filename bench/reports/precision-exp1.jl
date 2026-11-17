#
# Precision study, experiment 1: Float64 vs Float32 at d = 30, 35, 40 for the
# SVD, Arpack, Krylov and CG null solvers on the sphere benchmark.
#
#   julia -t 2 --project=. bench/reports/precision-exp1.jl [ds...]
#
# Appends rows to bench/reports/precision-exp1.csv and prints them as it goes.
# Randomized solvers (Arpack, Krylov, CG) run on three seeds; SVD once.
# Timings are under contention (other agents share the machine): compare
# relatively within this file, not across sessions.
#
using KrylovKit, IterativeSolvers, Arpack
include(joinpath(@__DIR__, "..", "SphereHarness.jl"))
using Printf
LinearAlgebra.BLAS.set_num_threads(2)

const DS = isempty(ARGS) ? [30, 35, 40] : parse.(Int, ARGS)
const SOLVERS = [:SVDSolver, :ArpackSolver, :KrylovSolver, :CGSolver]
const CSV = joinpath(@__DIR__, "precision-exp1.csv")
isfile(CSV) || open(CSV, "w") do io
    println(io, "d,T,solver,seed,N,seconds,bytes,peak_rss_gb,nullity,lsq_err,support,perm_ok,status")
end

function row(d, T, s, seed, N, r)
    line = @sprintf("%d,%s,%s,%d,%d,%.4f,%d,%.3f,%d,%.3e,%.6f,%s,\"%s\"",
                    d, T, s, seed, N, r.seconds, r.bytes, Sys.maxrss() / 2^30,
                    r.nullity, r.lsq_err, r.support, r.perm_ok ? "yes" : "no", r.status)
    println(line); flush(stdout)
    open(CSV, "a") do io; println(io, line); end
end

for T in (Float64, Float32)
    println("== warmup $T"); flush(stdout)
    warmup!([(; solver = s) for s in SOLVERS]; T)
    for d in DS
        for (k, seed) in enumerate((d, d + 1000, d + 2000))
            inp = build_sphere(d; T, seed)
            N = Dleto.globalDim(inp.Ω)
            for s in SOLVERS
                s === :SVDSolver && k > 1 && continue
                r = run_stratify(inp; solver = s)
                row(d, T, s, seed, N, r)
            end
        end
    end
end
