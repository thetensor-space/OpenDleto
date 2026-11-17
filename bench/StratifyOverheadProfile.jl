#
# StratifyOverheadProfile -- how much of a `stratify` call is NOT the
# derivation solve?  `realCanonicalForm` (an `eigen` per axis operator),
# `act` (the change-of-frame contraction), the retag loop, and the `@info`
# line, replicated here from `stratify(Γ, δ)` / `stratify(Ω, ch, Γ)`
# (src/Densors.jl) rather than instrumenting the source, so this measures
# exactly what ships.
#
# Usage:
#   JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 JL_RSS_LIMIT_GB=6 bench/jl \
#       bench/StratifyOverheadProfile.jl
#
# Three cells, as named in the task: sphere valence 3 d = 100 Float64,
# sphere valence 4 d = 40 Float32, video 120x90x30x3 Float32.
#
using Dleto
using ITensors
using LinearAlgebra
using Random
using Printf

include(joinpath(@__DIR__, "SphereHarness.jl"))
include(joinpath(@__DIR__, "StratifyMatrix.jl"))   # sphere_cell_input, video_cell_input (does not auto-run: no PROGRAM_FILE match)

"""
    profile_overhead(inp; method, solver, tol, reps) -> NamedTuple

Times the derivation solve once (warm), then times, separately, every
non-solve step `stratify` performs on the resulting derivations: building
each axis operator's real canonical form, `act`, the retag loop, and an
`@info` call shaped like the one in `stratify(Ω, ch, Γ)`.  Repeats `reps`
times and reports the mean of each, plus the solve time for scale.
"""
function profile_overhead(inp; method::Symbol = :Auto, solver::Symbol = :AutoSolver,
                          tol::Real = 1e-6, reps::Integer = 3)
    Ω, ch, Γ = inp.Ω, inp.ch, inp.Γ
    kw = solver === :AutoSolver ? NamedTuple() : (; solver)
    m = Dleto.get_derivation_method(method; kw...)

    # Warm-up (JIT), silenced -- the WHOLE pipeline, not just the solve: the
    # first call to `realCanonicalForm`/`act`/`replaceind` at a new (size, T)
    # combination pays its own compile cost, which swamped `act` and `rcf`
    # below (up to 0.57s) before this warm-up covered them too.
    quietly() do
        (_, expand_map0, ders0) = derTrOpsReduced(m, Ω, ch, Γ; tol = tol)
        if size(ders0, 2) > 0
            δ0 = embedITensors(Ω, expand_map0(ders0 * randn(eltype(ders0), size(ders0, 2))))
            Xs0 = [let X = δ0[i]
                       D, T = Dleto.realCanonicalForm(Array(X, inds(X)...))
                       ITensor(Matrix(T), inds(X)...)
                   end for i in 1:length(δ0)]
            Σ0 = act(Γ, Xs0)
            for X in Xs0
                orig = filter(i -> hasind(Γ, i), collect(inds(X)))
                temp = filter(i -> !hasind(Γ, i), collect(inds(X)))
                length(orig) == 1 && length(temp) == 1 || continue
                Σ0 = replaceind(Σ0, temp[1], orig[1])
            end
        end
        @info "Found $(size(ders0, 2)) derivations for stratification."
    end

    t_solve = Float64[]
    t_info = Float64[]
    t_rcf = Float64[]
    t_act = Float64[]
    t_retag = Float64[]
    nullity = 0

    for _ in 1:reps
        local expand_map, ders
        ts = @elapsed quietly() do
            (_, expand_map, ders) = derTrOpsReduced(m, Ω, ch, Γ; tol = tol)
        end
        push!(t_solve, ts)
        nullity = size(ders, 2)
        nullity == 0 && error("no nontrivial derivations at this cell; cannot profile stratify")
        δ = embedITensors(Ω, expand_map(ders * randn(eltype(ders), nullity)))

        ti = @elapsed quietly() do
            @info "Found $(nullity) derivations for stratification."
        end
        push!(t_info, ti)

        local Xs
        tr = @elapsed begin
            Xs = [let X = δ[i]
                      D, T = Dleto.realCanonicalForm(Array(X, inds(X)...))
                      ITensor(Matrix(T), inds(X)...)
                  end for i in 1:length(δ)]
        end
        push!(t_rcf, tr)

        local Σ
        ta = @elapsed (Σ = act(Γ, Xs))
        push!(t_act, ta)

        tg = @elapsed begin
            for X in Xs
                orig = filter(i -> hasind(Γ, i), collect(inds(X)))
                temp = filter(i -> !hasind(Γ, i), collect(inds(X)))
                length(orig) == 1 && length(temp) == 1 || continue
                Σ = replaceind(Σ, temp[1], orig[1])
            end
        end
        push!(t_retag, tg)
    end

    mean(v) = sum(v) / length(v)
    return (; nullity, solve = mean(t_solve), info = mean(t_info), rcf = mean(t_rcf),
              act = mean(t_act), retag = mean(t_retag))
end

function main()
    cells = [("sphere v3 d=100 F64", sphere_cell_input(3, 100, Float64)),
             ("sphere v4 d=40 F32", sphere_cell_input(4, 40, Float32)),
             ("video F=30 F32", video_cell_input(30, Float32))]

    @printf("%-22s %10s %10s %10s %10s %10s %10s\n",
            "cell", "solve(s)", "info(s)", "rcf(s)", "act(s)", "retag(s)", "overhead %")
    for (name, inp) in cells
        r = profile_overhead(inp)
        overhead = r.info + r.rcf + r.act + r.retag
        pct = 100 * overhead / (r.solve + overhead)
        @printf("%-22s %10.4f %10.6f %10.6f %10.6f %10.6f %9.2f%%\n",
                name, r.solve, r.info, r.rcf, r.act, r.retag, pct)
        flush(stdout)
    end
end

main()
