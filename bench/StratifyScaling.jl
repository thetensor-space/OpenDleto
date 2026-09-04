#
# How long does QuickDer take to stratify a random dense d x d x d tensor?
#
# Grows d by 10 and stops after the first run that exceeds a time budget
# (default 60s).  Prints each row as it completes and writes a CSV.
#
# What this measures.  A random (generic) tensor's only derivations are the
# scalars -- dim = dim null([1,1,1]) = 2 -- so the *answer* is degenerate: the
# canonical form of a scalar matrix is an arbitrary eigenbasis.  The *work* is
# not degenerate, because the solver still has to solve the whole derivation
# system to discover that nothing but the scalars is there.  So this is the
# cost of asking "does this tensor have a pattern?" and being told no, which is
# the common case on real data.
#
# Usage:  julia --project=. bench/StratifyScaling.jl [budget_seconds] [method]
#
using Dleto
using ITensors
using LinearAlgebra
using Random
using Printf

function stratify_once(d, method)
    fr = [Index(d, "a_$i") for i in 1:3]
    Γ = ITensor(randn(d, d, d), fr...)
    P = UniversalChisel(3)
    Ω = IndTransverseOps(fr, UniversalOp())
    GC.gc()
    res = nothing
    t = @elapsed res = stratify(Ω, P, Γ; tol = 1e-6, method = method)
    κ = maximum(cond(Array(X, inds(X)...)) for X in res.Xs)
    return (t, length(res.Xs), κ)
end

function run_scaling(; budget = 60.0, method = :QuickDer, step = 10, dmax = 1000)
    @printf("stratify scaling, method = %s, stop after a run over %.0fs\n\n",
            method, budget)
    @printf("%-6s %12s %10s %14s %s\n", "d", "seconds", "axes", "max kappa", "status")
    flush(stdout)

    rows = NamedTuple[]
    d = step
    while d <= dmax
        local t, nx, κ
        try
            (t, nx, κ) = stratify_once(d, method)
        catch e
            @printf("%-6d %12s %10s %14s %s\n", d, "-", "-", "-",
                    first(split(sprint(showerror, e), '\n')))
            flush(stdout)
            break
        end
        @printf("%-6d %12.3f %10d %14.3e ok\n", d, t, nx, κ)
        flush(stdout)
        push!(rows, (; d = d, seconds = t, axes = nx, kappa = κ))
        t > budget && break
        d += step
    end
    return rows
end

function write_csv(path, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "d,seconds,axes,kappa")
        for r in rows
            println(io, join((r.d, r.seconds, r.axes, r.kappa), ","))
        end
    end
    @info "wrote $path"
end

if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(20260902)
    budget = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 60.0
    method = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :QuickDer

    # Warm up at d = 4 so the first row is not dominated by compile time.
    try; stratify_once(4, method); catch; end

    rows = run_scaling(; budget = budget, method = method)
    write_csv(joinpath(@__DIR__, "stratify-scaling-$(method).csv"), rows)

    println("\n--- log-log table (base 10) ---")
    @printf("%-6s %10s %10s %10s\n", "d", "log10 d", "log10 s", "slope")
    prev = nothing
    for r in rows
        slope = prev === nothing ? NaN :
            (log10(r.seconds) - log10(prev.seconds)) / (log10(r.d) - log10(prev.d))
        @printf("%-6d %10.3f %10.3f %10s\n", r.d, log10(r.d), log10(r.seconds),
                isnan(slope) ? "-" : string(round(slope; digits = 2)))
        prev = r
    end
end
