#
# VideoDenseBench -- derivation-only benchmark on a random dense valence-4
# tensor shaped like a colour video: `randn(T, H, W, F, 3)` (height, width,
# frame count, RGB).  Universal operators, universal chisel.  The oracle is
# nullity 3 -- the three scalar derivations that every generic valence-4
# tensor has (one per axis pair up to the overall scaling ambiguity; a
# *random* dense tensor has no extra symmetry, unlike the hypersphere).
#
# Records seconds, bytes, nullity, max Z-law residual of the returned basis,
# and RSS peak (`Sys.maxrss()`, cumulative-peak for the process, not just the
# one call).  `T = Float32` is warmed up separately from `Float64` (different
# compiled specialisation).
#
# Usage:
#   bench/jl bench/VideoDenseBench.jl
#   (no CLI args -- the (H,W,F) x T x solver matrix is fixed below)
#
# NOTE ON TIMINGS: run through `bench/jl`, 2 Julia threads / 2 OpenBLAS
# threads per process -- these are 2-thread timings.
#
using LinearAlgebra
using Printf
using Random
using Dleto
using ITensors
using Logging

try
    @eval using Arpack
catch e
    @warn "Arpack did not load; :ArpackSolver rows will error" exception = e
end

BLAS.set_num_threads(Threads.nthreads())

const TOL = 1e-8
const CSV = joinpath(@__DIR__, "reports", "night-2026-09-03", "video-dense-sylver.csv")
const SOLVERS = [:AutoSolver, :ArpackSolver]
const SIZES = [20, 30, 40, 50, 64]      # (H,W,F) = (n,n,n), colour axis fixed at 3
const TYPES = [Float64, Float32]
const STOP_SECONDS = 180.0

quietly(f) = with_logger(NullLogger()) do
    redirect_stdout(devnull) do
        f()
    end
end

function der_residual(Γ::ITensor, D::Vector{ITensor}, P::AbstractMatrix)
    C = Chisel(P, collect(inds(Γ)))
    R = applyDerivation(Γ, D, C)
    scale = norm(Γ) * maximum(norm.(D))
    return norm(R) / max(scale, eps())
end

"""Build the (H,W,F,3) random video-shaped tensor of element type T, seeded."""
function build_video(n::Integer, T::Type; seed::Integer = n)
    Random.seed!(seed)
    A = randn(T, n, n, n, 3)
    frames = [Index(n, "h"), Index(n, "w"), Index(n, "f"), Index(3, "c")]
    Γ = ITensor(A, frames...)
    Ω = IndTransverseOps(frames, UniversalOp())
    ch = UniversalChisel(4)
    return (; Γ, Ω, ch)
end

function one_case(Γ::ITensor, Ω, ch, solver::Symbol; tol::Real = TOL)
    status = "ok"
    nullity = 0
    maxres = NaN
    GC.gc()
    st = @timed try
        quietly() do
            m = Dleto.get_derivation_method(:SylverLining; solver)
            (rΩ, expand_map, ders) = Dleto.derTrOpsReduced(m, Ω, ch, Γ; tol = tol)
            nullity = size(ders, 2)
            if nullity > 0
                basis = [embedITensors(Ω, expand_map(ders[:, i])) for i in 1:nullity]
                maxres = maximum(der_residual(Γ, D, ch) for D in basis)
            end
        end
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
    end
    rss = Sys.maxrss() / 2^30
    return (; seconds = st.time, bytes = st.bytes, nullity, maxres, status, rss)
end

function warmup!(n::Integer, T::Type)
    print("Warm-up (n = $n, T = $T, all solvers) ... "); flush(stdout)
    tw = @elapsed begin
        inp = build_video(n, T)
        for solver in SOLVERS
            one_case(inp.Γ, inp.Ω, inp.ch, solver)
        end
    end
    @printf("done (%.1fs, not counted)\n", tw)
end

function main()
    println("Dleto video-shaped dense derivation benchmark (2-thread timings under bench/jl)")
    println("  threads = $(Threads.nthreads()), BLAS = $(BLAS.get_num_threads()), tol = $TOL")
    isfile(CSV) || open(CSV, "w") do io
        println(io, "# timings are 2-thread timings (bench/jl pins 2 Julia + 2 OpenBLAS threads)")
        println(io, "# Γ = randn(T, n, n, n, 3), UniversalOp(), UniversalChisel(4); oracle nullity = 3")
        println(io, "n,T,solver,seconds,bytes,nullity,maxres,rss_gb,status")
    end

    for T in TYPES
        warmup!(8, T)
    end

    for T in TYPES
        stopped = false
        for n in SIZES
            stopped && break
            println()
            @printf("==== T = %s   n = %d (H=W=F=%d, c=3)   %s\n", T, n, n, Libc.strftime("%H:%M:%S", time()))
            inp = build_video(n, T)
            @printf("%-12s %9s %9s %5s %9s %8s  %s\n", "solver", "time(s)", "alloc(GB)", "null", "maxres", "rss(GB)", "status")
            flush(stdout)
            for solver in SOLVERS
                r = one_case(inp.Γ, inp.Ω, inp.ch, solver)
                @printf("%-12s %9.3f %9.3f %5d %9.2e %8.2f  %s\n",
                        solver, r.seconds, r.bytes / 2^30, r.nullity, r.maxres, r.rss, r.status)
                flush(stdout)
                open(CSV, "a") do io
                    @printf(io, "%d,%s,%s,%.6f,%d,%d,%.6e,%.3f,\"%s\"\n",
                            n, T, solver, r.seconds, r.bytes, r.nullity, r.maxres, r.rss, r.status)
                end
                if r.seconds > STOP_SECONDS
                    @printf("STOPPED: T=%s solver %s at n=%d exceeded %.0fs (%.1fs)\n",
                            T, solver, n, STOP_SECONDS, r.seconds)
                    open(CSV, "a") do io
                        @printf(io, "%d,%s,%s,stopped,,,,, \"exceeded %ds\"\n", n, T, solver, Int(STOP_SECONDS))
                    end
                    stopped = true
                end
            end
        end
    end
    println("\nFinished. Results in $CSV")
end

main()
