#
# Comparison of the three chiseling operations -- speed AND accuracy.
#
#   der       the Z-set: the derivations of a tensor for a chisel
#   den       the T-set: the tensors admitting a given set of derivations
#   stratify  the pattern: a change of frame putting the derivation in real
#             canonical form, so the tensor's sparsity becomes visible
#
# Every category is run on the *same* tensors, of growing dimension, and we
# record both how long each operation took and how badly its defining equation
# is violated.  Accuracy is the point: a fast solver that returns vectors which
# are not derivations is not a faster solver.
#
# THE TWO AXES.  A "category" here is a (method, chisel) pair, because the two
# are not independent -- the solve-and-lift methods each handle only the chisel
# shape their restriction is built for:
#
#   SylverLining   any chisel, any valency; builds the derivation-densor
#                  operator as a LinearMap and hands it to a null solver.
#   QuickDer       Liu's derivation solve-and-lift (quick-der-lib.jl): all
#                  three axes restricted, then lifted.  Valency 3, one-row
#                  fully engaged chisel.
#   QuickSylver    Liu's Sylvester solve-and-lift (quicksylver-lib.jl): two
#                  axes restricted, an affine frame lifted.  Chisels with
#                  exactly two engaged axes, e.g. AdjointChisel.
#
# `den` and `stratify` are both driven by whichever method produced the
# derivations, so the method axis carries through all three operations.
#
# The tensor family is Liu's `K_M_field_tensor` from ../fast-der-solver: powers
# of a cyclic shift matrix with random coefficients, then scrambled by random
# basis changes on the row and column axes.  That is the multiplication tensor
# of a polynomial algebra K[x]/(x^n - c) -- Example 5.2 of null_patterns.pdf --
# so it has a large, known derivation space, unlike a generic tensor whose
# derivations are only the scalars.  Scrambling is what makes it a real test:
# the structure is present but not visible in the given basis.  It also matters
# for `den`: a generic tensor's densor is the *whole* tensor space (section 5.4,
# "scalar derivations reveal nothing"), so measuring `den` there would measure
# nothing.
#
# Usage:
#   julia --project=. bench/DerivationSolverBench.jl          # short
#   julia --project=. bench/DerivationSolverBench.jl long
#
# Writes bench/chisel-operation-results.csv and, if a plotting backend is
# available, bench/chisel-operation-results.png on log-log axes.
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

med(v) = isempty(v) ? Inf : sort(v)[cld(length(v), 2)]

"""
    der_residual(Γ, D, P) -> Real

Relative size of the chisel-weighted sum  Σ_a P[:,a] ⊗ (Γ · D_a), which is zero
exactly when D is a P-derivation of Γ.  This single quantity scores all three
operations, because the derivation equation and the densor equation are one
equation with a different unknown -- only the slot being solved for changes.
"""
function der_residual(Γ::ITensor, D::Vector{ITensor}, P::AbstractMatrix)
    C = Chisel(P, collect(inds(Γ)))
    scale = norm(Γ) * maximum(norm.(D))
    return norm(applyDerivation(Γ, D, C)) / max(scale, eps())
end

"""Z-law scores: residual of every derivation against the tensor."""
function score_der(Γ, basis::Vector{Vector{ITensor}}, P)
    isempty(basis) && return (Inf, Inf)
    rs = [der_residual(Γ, D, P) for D in basis]
    return (maximum(rs), med(rs))
end

"""
T-law scores: every tensor of the densor must admit every derivation that
defined it.  Both sets can be large, so a fixed number of random (tensor,
derivation) pairs is sampled rather than the full product.
"""
function score_den(tset::Vector{ITensor}, Δ::Vector{Vector{ITensor}}, P; samples = 40)
    (isempty(tset) || isempty(Δ)) && return (Inf, Inf)
    rs = Float64[]
    for _ in 1:samples
        s = tset[rand(eachindex(tset))]
        ω = Δ[rand(eachindex(Δ))]
        push!(rs, der_residual(s, ω, P))
    end
    return (maximum(rs), med(rs))
end

# ---------------------------------------------------------------- categories

const UNIVERSAL = UniversalChisel(3)
const ADJOINT = AdjointChisel(3, 1, 2)

# (label, chisel, method-symbol, extra der kwargs)
const CATEGORIES = [
    ("SylverLining/SVD  [1,1,1]",  UNIVERSAL, :SylverLining, (; solver = :SVDSolver)),
    ("QuickDer-lift     [1,1,1]",  UNIVERSAL, :QuickDer,     (;)),
    ("SylverLining/SVD  [1,-1,0]", ADJOINT,   :SylverLining, (; solver = :SVDSolver)),
    ("QuickSylver-lift  [1,-1,0]", ADJOINT,   :QuickSylver,  (;)),
]

# The null-solver axis, exercised through SylverLining on the universal chisel.
# It only bites when globalDim(Ω) = 3n^2 >= 1000, i.e. n >= 19 for this family;
# below that SylverLining uses a dense `eigen` and every variant is the same
# code path.  The `long` size grid therefore crosses n = 19.  The lift methods
# call `nullspace` directly, as their reference implementations do, and ignore
# the setting entirely.
const NULL_SOLVERS = [:SVDSolver, :LUSolver, :KrylovSolver, :LanczosSolver, :CGSolver]

struct Row
    operation::String
    category::String
    n::Int
    trial::Int
    seconds::Float64
    dim::Int
    residual_max::Float64
    residual_med::Float64
    conditioning::Float64
    status::String
end

blank(op, cat, n, trial, status) = Row(op, cat, n, trial, NaN, 0, Inf, Inf, NaN, status)
errline(e) = "error: " * first(split(sprint(showerror, e), '\n'))

# ---------------------------------------------------------------- one trial

"""
    run_trial(label, P, method, kw, Γ, n, trial) -> Vector{Row}

Runs all three operations for one category on one tensor, in order, because
each feeds the next: `der` produces the derivations, `den` takes them as its
constraints, and `stratify` puts one of them into canonical form.  A failure in
`der` is recorded and the two dependents are recorded as skipped rather than
silently omitted.
"""
function run_trial(label, P, method, kw, Γ, n, trial; tol = 1e-6)
    rows = Row[]
    fr = collect(inds(Γ))
    Ω = IndTransverseOps(fr, UniversalOp())

    # -- der: the Z-set ------------------------------------------------------
    GC.gc()
    Δ = Vector{Vector{ITensor}}()
    local t_der
    try
        t_der = @elapsed Δ = der(method, Ω, P, Γ; tol = tol, kw...)
    catch e
        push!(rows, blank("der", label, n, trial, errline(e)))
        push!(rows, blank("den", label, n, trial, "skipped: der failed"))
        push!(rows, blank("stratify", label, n, trial, "skipped: der failed"))
        return rows
    end
    rmax, rmed = score_der(Γ, Δ, P)
    status = isempty(Δ) ? "no solutions" : (rmax > 1e-6 ? "inaccurate" : "ok")
    push!(rows, Row("der", label, n, trial, t_der, length(Δ), rmax, rmed, NaN, status))

    if isempty(Δ)
        push!(rows, blank("den", label, n, trial, "skipped: no derivations"))
        push!(rows, blank("stratify", label, n, trial, "skipped: no derivations"))
        return rows
    end

    # -- den: the T-set of those derivations ---------------------------------
    # `den` is method-independent: it solves the same bilinear system with the
    # tensor as the unknown, so the method axis enters only through which Δ it
    # was handed.  That is the honest comparison -- a method that returns an
    # under-determined Δ gets an over-large densor, and this is where that
    # shows up.
    GC.gc()
    try
        t_den = @elapsed tset = den(Ω, P, Δ; tol = tol, nd = -1)
        dmax, dmed = score_den(tset, Δ, P)
        dstatus = isempty(tset) ? "no solutions" : (dmax > 1e-6 ? "inaccurate" : "ok")
        push!(rows, Row("den", label, n, trial, t_den, length(tset), dmax, dmed, NaN, dstatus))
    catch e
        push!(rows, blank("den", label, n, trial, errline(e)))
    end

    # -- stratify: the pattern -----------------------------------------------
    # There is no residual oracle for stratification yet: the sigma_{e+1}
    # verdict (Algorithm 2 of null_patterns.pdf) that decides whether a pattern
    # is genuinely present is not implemented, and `stratify` does not return
    # the block structure it found.  Two things ARE measurable and are what is
    # reported instead:
    #
    #   conditioning  max over axes of kappa(X_a) for the frame it returns.
    #                 This is the number that decides whether the stratified
    #                 tensor is usable: the frame is applied to Gamma, so its
    #                 condition number multiplies straight into the answer.
    #
    #                 On this family it comes out IDENTICAL across all four
    #                 categories, to about ten digits, while varying from
    #                 trial to trial.  That is the right answer, not a bug:
    #                 K[x]/(x^n - c) is commutative, so its elements share an
    #                 eigenbasis, and the frame that puts a random derivation
    #                 into real canonical form is the same frame whichever
    #                 derivation was drawn and whichever chisel or method
    #                 found it.  The pattern is a property of the tensor.  So
    #                 kappa does not discriminate methods here -- it is a
    #                 cross-check that they agree, and the discriminating
    #                 axis for `stratify` is time.
    #   dim preserved der(Sigma) must have the same dimension as der(Gamma),
    #                 since a change of frame conjugates the derivation algebra
    #                 rather than changing it.  A drop means the frame was too
    #                 ill-conditioned to survive.
    GC.gc()
    try
        t_str = @elapsed res = stratify(Ω, P, Γ; tol = tol, method = method, kw...)
        κ = maximum(cond(Array(X, inds(X)...)) for X in res.Xs)

        # Cheap invariant: conjugation must not change the dimension of the
        # Z-set.  Only checked with the general method, so that a lift method's
        # own restriction cannot mask a bad frame.
        Δσ = der(:SylverLining, Ω, P, res.Σ; tol = tol, solver = :SVDSolver)
        sstatus = length(Δσ) == length(Δ) ? "ok" :
                  "dim changed $(length(Δ)) -> $(length(Δσ))"
        push!(rows, Row("stratify", label, n, trial, t_str, length(res.Xs),
                        Inf, Inf, κ, sstatus))
    catch e
        push!(rows, blank("stratify", label, n, trial, errline(e)))
    end

    return rows
end

"""Null-solver comparison: `der` only, since the solver choice is what varies."""
function run_null_solvers(Γ, n, trial; tol = 1e-6)
    rows = Row[]
    fr = collect(inds(Γ))
    Ω = IndTransverseOps(fr, UniversalOp())
    for s in NULL_SOLVERS
        label = "null-solver/$s"
        GC.gc()
        try
            local Δ
            t = @elapsed Δ = der(:SylverLining, Ω, UNIVERSAL, Γ; tol = tol, solver = s)
            rmax, rmed = score_der(Γ, Δ, UNIVERSAL)
            status = isempty(Δ) ? "no solutions" : (rmax > 1e-6 ? "inaccurate" : "ok")
            push!(rows, Row("der", label, n, trial, t, length(Δ), rmax, rmed, NaN, status))
        catch e
            push!(rows, blank("der", label, n, trial, errline(e)))
        end
    end
    return rows
end

# ---------------------------------------------------------------- driver

function bench(; sizes, trials, tol = 1e-6, null_solvers = false)
    rows = Row[]
    for n in sizes, trial in 1:trials
        A = K_M_field_tensor(n)
        frame = [Index(size(A, i), "a_$i") for i in 1:3]
        Γ = ITensor(A, frame...)

        batch = Row[]
        for (label, P, method, kw) in CATEGORIES
            append!(batch, run_trial(label, P, method, kw, Γ, n, trial; tol = tol))
        end
        null_solvers && append!(batch, run_null_solvers(Γ, n, trial; tol = tol))

        for r in batch
            @printf("%-9s %-28s n=%-4d t=%d %9.3fs dim=%-5d resid=%-9.2e κ=%-9.2e %s\n",
                    r.operation, r.category, r.n, r.trial, r.seconds, r.dim,
                    r.residual_max, r.conditioning, r.status)
        end
        append!(rows, batch)
    end
    return rows
end

function write_csv(path, rows::Vector{Row})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "operation,category,n,trial,seconds,dim,residual_max," *
                    "residual_median,conditioning,status")
        for r in rows
            println(io, join((r.operation, r.category, r.n, r.trial, r.seconds,
                              r.dim, r.residual_max, r.residual_med,
                              r.conditioning, r.status), ","))
        end
    end
    @info "wrote $path"
end

"""Median over trials, per (category, n), for one operation, skipping non-finite."""
function summarise(rows::Vector{Row}, operation::String, field::Symbol)
    sel = filter(r -> r.operation == operation, rows)
    out = Dict{String, Vector{Tuple{Int, Float64}}}()
    for cat in unique(r.category for r in sel)
        pts = Tuple{Int, Float64}[]
        for n in sort(unique(r.n for r in sel if r.category == cat))
            vals = filter(isfinite, [getfield(r, field) for r in sel
                                     if r.category == cat && r.n == n])
            isempty(vals) || push!(pts, (n, med(vals)))
        end
        out[cat] = pts
    end
    return out
end

function panel(rows, operation, field, ylabel, title)
    p = plot(; xscale = :log10, yscale = :log10, xlabel = "n (axis dimension)",
             ylabel = ylabel, title = title, legend = :topleft, legendfontsize = 5,
             titlefontsize = 9)
    for (cat, pts) in sort(collect(summarise(rows, operation, field)))
        isempty(pts) && continue
        safe = [(n, max(v, 1e-18)) for (n, v) in pts]   # log axis needs > 0
        plot!(p, first.(safe), last.(safe); label = cat, marker = :circle)
    end
    return p
end

function plot_results(rows::Vector{Row}, path)
    try
        panels = [
            panel(rows, "der", :seconds, "seconds", "der (Z-set): time"),
            panel(rows, "der", :residual_max, "max rel. residual", "der: accuracy"),
            panel(rows, "den", :seconds, "seconds", "den (T-set): time"),
            panel(rows, "den", :residual_max, "max rel. residual", "den: accuracy"),
            panel(rows, "stratify", :seconds, "seconds", "stratify: time"),
            panel(rows, "stratify", :conditioning, "max κ(X_a)", "stratify: frame conditioning"),
        ]
        fig = plot(panels...; layout = (3, 2), size = (1200, 1250))
        savefig(fig, path)
        @info "wrote $path"
    catch e
        @warn "plotting skipped" exception = (e, catch_backtrace())
    end
end

function summarise_text(rows::Vector{Row})
    println("\n=== median over trials ===")
    @printf("%-9s %-28s %-5s %10s %6s %11s %11s\n",
            "operation", "category", "n", "seconds", "dim", "resid", "kappa")
    for op in ("der", "den", "stratify")
        for cat in sort(unique(r.category for r in rows if r.operation == op))
            for n in sort(unique(r.n for r in rows))
                sel = filter(r -> r.operation == op && r.category == cat && r.n == n, rows)
                isempty(sel) && continue
                @printf("%-9s %-28s %-5d %10.3f %6d %11.2e %11.2e\n", op, cat, n,
                        med(filter(isfinite, [r.seconds for r in sel])),
                        round(Int, med([Float64(r.dim) for r in sel])),
                        med(filter(isfinite, [r.residual_max for r in sel])),
                        med(filter(isfinite, [r.conditioning for r in sel])))
            end
        end
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
    rows = bench(; sizes = bench_sizes(mode), trials = trials,
                 null_solvers = (mode === :long))
    write_csv(joinpath(@__DIR__, "chisel-operation-results.csv"), rows)
    plot_results(rows, joinpath(@__DIR__, "chisel-operation-results.png"))
    summarise_text(rows)
end
