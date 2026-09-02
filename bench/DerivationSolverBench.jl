#
# Derivation solver comparison: speed AND accuracy.
#
# Every solver is run on the *same* tensors, of growing dimension, and we
# record both how long it took and how badly it satisfies the defining
# equation.  Accuracy is the point: a fast solver that returns vectors which
# are not derivations is not a faster solver.
#
# The tensor family is Liu's `K_M_field_tensor` from ../fast-der-solver: powers
# of a cyclic shift matrix with random coefficients, then scrambled by random
# basis changes on the row and column axes.  That is the multiplication tensor
# of a polynomial algebra K[x]/(x^n - c) -- Example 5.2 of null_patterns.pdf --
# so it has a large, known derivation space, unlike a generic tensor whose
# derivations are only the scalars.  Scrambling is what makes it a real test:
# the structure is present but not visible in the given basis.
#
# Usage:
#   julia --project=. bench/DerivationSolverBench.jl          # short
#   julia --project=. bench/DerivationSolverBench.jl long
#
# Writes bench/derivation-solver-results.csv and, if a plotting backend is
# available, bench/derivation-solver-results.png on log-log axes.
#
using Dleto
using ITensors
using LinearAlgebra
using Random
using Printf
using Plots

# Loading the trigger packages activates Dleto's solver extensions.  Both are
# in [deps]; Arpack is only a weakdep and is skipped when absent, which is why
# ArpackSolver / ArpackDenseSolver are not in the roster below.
using KrylovKit
using IterativeSolvers

# ---------------------------------------------------------------- tensor family

"""
    K_M_field_tensor(n; scramble=true) -> Array{Float64,3}

Transcribed from ../fast-der-solver/quicksylver-vs-dleto-bench.jl.
"""
function K_M_field_tensor(n; scramble = true)
    M = zeros(Float64, n, n)
    M[1, n] = 1
    for i in 1:(n - 1)
        M[i + 1, i] = 1
    end

    coefficients = rand(vcat(-10:-1, 1:10), n)
    slices = [coefficients[k] * M^(k - 1) for k in 1:n]
    if !scramble
        return cat(slices...; dims = 3)
    end

    row_basis = randn(n, n)
    col_basis = randn(n, n)
    return cat([row_basis \ slice * col_basis for slice in slices]...; dims = 3)
end

# ---------------------------------------------------------------- measurement

"""
    residuals(Γ, basis, P) -> (max, median)

Relative size of the defining equation's residual, per derivation, via
`applyDerivation` -- the same oracle the Z-law uses in
test/TestDerivationLaws.jl.  Returns (Inf, Inf) for an empty basis, since
"found nothing" must not score as "perfectly accurate".
"""
function residuals(Γ::ITensor, basis::Vector{Vector{ITensor}}, P::AbstractMatrix)
    isempty(basis) && return (Inf, Inf)
    C = Chisel(P, collect(inds(Γ)))
    rs = Float64[]
    for D in basis
        scale = norm(Γ) * maximum(norm.(D))
        push!(rs, norm(applyDerivation(Γ, D, C)) / max(scale, eps()))
    end
    return (maximum(rs), median(rs))
end

median(v::Vector{Float64}) = isempty(v) ? Inf : sort(v)[cld(length(v), 2)]

"""
The solvers under comparison.

Two independent axes, which the package currently conflates: the derivation
*method* (the formulation) and the *null solver* (the linear algebra).  Note
that SylverLining only dispatches to the null solver when
globalDim(Ω) >= 1000, i.e. 3n^2 >= 1000, i.e. n >= 19 for this family; below
that it uses a dense `eigen` and the solver choice is inert.  The size grid
therefore has to cross n = 19 for the solver axis to mean anything.
"""
const UNIVERSAL = UniversalChisel(3)
const ADJOINT = AdjointChisel(3, 1, 2)

# (label, chisel, solve) -- the chisel is carried per case because QuickSylver
# handles adjoint-type chisels (two engaged axes) while the others take the
# universal chisel, so they are not solving the same system.  Residuals are
# always measured against the chisel that was actually used.
const SOLVER_CASES = [
    ("SylverLining/SVD  [1,1,1]", UNIVERSAL,
        (Γ; tol) -> der(:SylverLining, UNIVERSAL, Γ; tol = tol, solver = :SVDSolver)),
    ("SylverLining/LU   [1,1,1]", UNIVERSAL,
        (Γ; tol) -> der(:SylverLining, UNIVERSAL, Γ; tol = tol, solver = :LUSolver)),
    ("FastDer3Valent    [1,1,1]", UNIVERSAL,
        (Γ; tol) -> der(:FastDer3Valent, UNIVERSAL, Γ; tol = tol)),
    ("SylverLining/Krylov  [1,1,1]", UNIVERSAL,
        (Γ; tol) -> der(:SylverLining, UNIVERSAL, Γ; tol = tol, solver = :KrylovSolver)),
    ("SylverLining/Lanczos [1,1,1]", UNIVERSAL,
        (Γ; tol) -> der(:SylverLining, UNIVERSAL, Γ; tol = tol, solver = :LanczosSolver)),
    ("SylverLining/CG      [1,1,1]", UNIVERSAL,
        (Γ; tol) -> der(:SylverLining, UNIVERSAL, Γ; tol = tol, solver = :CGSolver)),
    ("SylverLining/SVD  [1,-1,0]", ADJOINT,
        (Γ; tol) -> der(:SylverLining, ADJOINT, Γ; tol = tol, solver = :SVDSolver)),
    ("QuickSylver       [1,-1,0]", ADJOINT,
        (Γ; tol) -> der(:QuickSylver, ADJOINT, Γ; tol = tol)),
]

# NOTE: the solver name only bites for SylverLining, and only when
# globalDim(Ω) = 3n^2 >= 1000, i.e. n >= 19 for this family; below that
# SylverLining uses a dense `eigen` and every /solver variant is the same
# code path.  FastDer3Valent and QuickSylver call `nullspace` directly, as
# their reference implementations do, and ignore the setting entirely.

struct Row
    solver::String
    n::Int
    trial::Int
    seconds::Float64
    dim::Int
    residual_max::Float64
    residual_med::Float64
    status::String
end

function run_case(label, P, f, n, trial; tol = 1e-6)
    A = K_M_field_tensor(n)
    frame = [Index(size(A, i), "a_$i") for i in 1:3]
    Γ = ITensor(A, frame...)

    GC.gc()
    basis = Vector{Vector{ITensor}}()
    status = "ok"
    seconds = NaN
    try
        seconds = @elapsed basis = f(Γ; tol = tol)
    catch e
        status = "error: " * first(split(sprint(showerror, e), '\n'))
        return Row(label, n, trial, seconds, 0, Inf, Inf, status)
    end

    rmax, rmed = residuals(Γ, basis, P)
    isempty(basis) && (status = "no solutions")
    rmax > 1e-6 && status == "ok" && (status = "inaccurate")
    return Row(label, n, trial, seconds, length(basis), rmax, rmed, status)
end

# ---------------------------------------------------------------- driver

function bench(; sizes, trials, tol = 1e-6)
    rows = Row[]
    for n in sizes, (label, P, f) in SOLVER_CASES, trial in 1:trials
        r = run_case(label, P, f, n, trial; tol = tol)
        push!(rows, r)
        @printf("%-20s n=%-4d trial=%d  %8.3fs  dim=%-4d  resid=%.2e  %s\n",
                r.solver, r.n, r.trial, r.seconds, r.dim, r.residual_max, r.status)
    end
    return rows
end

function write_csv(path, rows::Vector{Row})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "solver,n,trial,seconds,dim,residual_max,residual_median,status")
        for r in rows
            println(io, join((r.solver, r.n, r.trial, r.seconds, r.dim,
                              r.residual_max, r.residual_med, r.status), ","))
        end
    end
    @info "wrote $path"
end

"""Median over trials, per (solver, n), skipping non-finite entries."""
function summarise(rows::Vector{Row}, field::Symbol)
    out = Dict{String, Vector{Tuple{Int, Float64}}}()
    for solver in unique(r.solver for r in rows)
        pts = Tuple{Int, Float64}[]
        for n in sort(unique(r.n for r in rows if r.solver == solver))
            vals = [getfield(r, field) for r in rows if r.solver == solver && r.n == n]
            vals = filter(isfinite, vals)
            isempty(vals) || push!(pts, (n, sort(vals)[cld(length(vals), 2)]))
        end
        out[solver] = pts
    end
    return out
end

function plot_results(rows::Vector{Row}, path)
    try
        times = summarise(rows, :seconds)
        resid = summarise(rows, :residual_max)

        p1 = plot(; xscale = :log10, yscale = :log10,
                  xlabel = "n (axis dimension)", ylabel = "seconds",
                  title = "Derivation solve time", legend = :topleft)
        for (solver, pts) in sort(collect(times))
            isempty(pts) && continue
            plot!(p1, first.(pts), last.(pts); label = solver, marker = :circle)
        end

        p2 = plot(; xscale = :log10, yscale = :log10,
                  xlabel = "n (axis dimension)",
                  ylabel = "max relative residual",
                  title = "Accuracy of the derivations found", legend = :topleft)
        for (solver, pts) in sort(collect(resid))
            isempty(pts) && continue
            safe = [(n, max(v, 1e-18)) for (n, v) in pts]   # log axis needs > 0
            plot!(p2, first.(safe), last.(safe); label = solver, marker = :circle)
        end

        fig = plot(p1, p2; layout = (1, 2), size = (1100, 450))
        savefig(fig, path)
        @info "wrote $path"
    catch e
        @warn "plotting skipped" exception = (e, catch_backtrace())
    end
end

bench_sizes(mode) =
    mode === :short ? [4, 6, 8, 10] :
    mode === :long  ? [4, 6, 8, 10, 12, 15, 19, 22, 26] :
    error("Unknown mode $mode; use short or long.")

if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(20260902)
    mode = isempty(ARGS) ? :short : Symbol(ARGS[1])
    trials = mode === :short ? 3 : 5
    rows = bench(; sizes = bench_sizes(mode), trials = trials)
    write_csv(joinpath(@__DIR__, "derivation-solver-results.csv"), rows)
    plot_results(rows, joinpath(@__DIR__, "derivation-solver-results.png"))
end
