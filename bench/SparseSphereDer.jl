#
# SparseSphereDer -- derivation-only benchmark on the RAW (unscrambled) sparse
# hypersphere octant, universal operators, universal chisel.
#
# Unlike `bench/HypersphereBaseline.jl` (which scrambles the sphere by random
# orthogonal matrices and measures stratification recovery), this script
# builds `S = sphere_octant(d; valence)` directly -- sparse, on the lattice
# `i_1 + ... + i_valence = d - 1` -- and times only
# `Dleto.derTrOpsReduced(Dleto.get_derivation_method(:SylverLining; solver),
# Ω, ch, S; tol=1e-8)` against the universal operator space.  Expected nullity
# is 13 at every valence and every d (the universal derivation algebra of the
# sphere; see bench/reports/night-2026-09-03/BOARD.md, "tests-quickdern").
#
# Also records the max Z-law residual of the returned basis (`der_residual`,
# same formula as test/TestDerivationLaws.jl but defined locally here so this
# script does not `include` a test file) and the density nnz/d^valence of S.
#
# Usage (CLI mirrors bench/HypersphereBaseline.jl so each d can be run as its
# own `bench/jl` invocation -- keeps any single process short):
#   bench/jl bench/SparseSphereDer.jl [dims...] [--valence=4] [--only=Auto,Arpack]
#        [--csv=path]
#   default dims: 10 16 20 24 30 40; default valence: 4; default solvers: all three
#
# NOTE ON TIMINGS: run through `bench/jl`, 2 Julia threads / 2 OpenBLAS
# threads per process -- these are 2-thread timings.
#
using LinearAlgebra
using Printf
include(joinpath(@__DIR__, "SphereHarness.jl"))

try
    @eval using Arpack
catch e
    @warn "Arpack did not load; :ArpackSolver rows will error" exception = e
end

BLAS.set_num_threads(Threads.nthreads())

const TOL     = 1e-8
const VALENCE = let a = filter(startswith("--valence="), ARGS); isempty(a) ? 4 : parse(Int, a[1][11:end]) end
const DIMS    = let a = filter(x -> !startswith(x, "--"), ARGS); isempty(a) ? [10, 16, 20, 24, 30, 40] : parse.(Int, a) end
const ONLYRAW = let a = filter(startswith("--only="), ARGS); isempty(a) ? String[] : split(a[1][8:end], ",") end
const CSV     = let a = filter(startswith("--csv="), ARGS); isempty(a) ? joinpath(@__DIR__, "reports", "night-2026-09-03", "sparse-sphere-der.csv") : a[1][7:end] end

const ALL_SOLVERS = [:AutoSolver, :ArpackSolver, :KrylovSolver]
const SOLVERS = isempty(ONLYRAW) ? ALL_SOLVERS : [s for s in ALL_SOLVERS if String(s) in ONLYRAW]

"""Silence solver stdout chatter, but NOT @warn/@info -- those carry the
nullity-verdict diagnostics (`solve_nullspace`'s "UNCERTIFIED" warning) that
this script needs to see when the returned nullity falls short of 13."""
quietly(f) = redirect_stdout(devnull) do
    f()
end

"""
    der_residual(Γ, D, P) -> Real

Relative Z-law residual of derivation basis element `D` (one ITensor per
axis) against chisel `P`.  Same formula as test/TestDerivationLaws.jl,
redefined locally so this script does not need to `include` a test file.
"""
function der_residual(Γ::ITensor, D::Vector{ITensor}, P::AbstractMatrix)
    C = Chisel(P, collect(inds(Γ)))
    R = applyDerivation(Γ, D, C)
    scale = norm(Γ) * maximum(norm.(D))
    return norm(R) / max(scale, eps())
end

"""One timed derTrOpsReduced call plus a (untimed) Z-law residual check."""
function one_case(S::ITensor, Ω, ch, solver::Symbol; tol::Real = TOL)
    status = "ok"
    nullity = 0
    maxres = NaN
    GC.gc()
    st = @timed try
        quietly() do
            m = Dleto.get_derivation_method(:SylverLining; solver)
            (rΩ, expand_map, ders) = Dleto.derTrOpsReduced(m, Ω, ch, S; tol = tol)
            nullity = size(ders, 2)
            if nullity > 0
                basis = [embedITensors(Ω, expand_map(ders[:, i])) for i in 1:nullity]
                maxres = maximum(der_residual(S, D, ch) for D in basis)
            end
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
    end
    return (; seconds = st.time, bytes = st.bytes, nullity, maxres, status)
end

function warmup!(d::Integer = 8)
    print("Warm-up (d = $d, valence $VALENCE, active solvers) ... "); flush(stdout)
    tw = @elapsed for solver in SOLVERS
        S = sphere_octant(d; valence = VALENCE)
        Ω = IndTransverseOps(collect(inds(S)), UniversalOp())
        ch = UniversalChisel(VALENCE)
        one_case(S, Ω, ch, solver)
    end
    @printf("done (%.1fs, not counted)\n", tw)
end

function main()
    println("Dleto sparse-sphere derivation benchmark -- valence $VALENCE (2-thread timings under bench/jl)")
    println("  threads = $(Threads.nthreads()), BLAS = $(BLAS.get_num_threads()), tol = $TOL, dims = $DIMS")
    isfile(CSV) || open(CSV, "w") do io
        println(io, "# timings are 2-thread timings (bench/jl pins 2 Julia + 2 OpenBLAS threads)")
        println(io, "# S = sphere_octant(d; valence) raw/unscrambled, UniversalOp(), UniversalChisel(valence)")
        println(io, "valence,d,nnz,density,solver,seconds,bytes,nullity,maxres,status")
    end

    warmup!(8)

    for d in DIMS
        println()
        @printf("==== valence = %d   d = %d   %s\n", VALENCE, d, Libc.strftime("%H:%M:%S", time()))
        S = sphere_octant(d; valence = VALENCE)
        fr = collect(inds(S))
        nnzS = count(!=(0), Array(S, fr...))
        density = nnzS / d^VALENCE
        Ω = IndTransverseOps(fr, UniversalOp())
        ch = UniversalChisel(VALENCE)
        @printf("%-12s %9s %9s %5s %9s  %s\n", "solver", "time(s)", "alloc(GB)", "null", "maxres", "status")
        flush(stdout)
        for solver in SOLVERS
            r = one_case(S, Ω, ch, solver)
            @printf("%-12s %9.3f %9.3f %5d %9.2e  %s\n",
                    solver, r.seconds, r.bytes / 2^30, r.nullity, r.maxres, r.status)
            flush(stdout)
            open(CSV, "a") do io
                @printf(io, "%d,%d,%d,%.6g,%s,%.6f,%d,%d,%.6e,\"%s\"\n",
                        VALENCE, d, nnzS, density, solver, r.seconds, r.bytes, r.nullity, r.maxres, r.status)
            end
        end
    end
    println("\nFinished. Results in $CSV")
end

main()
