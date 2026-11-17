#
# Exp. C: the constant in `iter_tol`, i.e. the floor on an iterative
# eigensolver's own stopping tolerance, in units of `eps(T_compute)`.
#
# This one cannot be swept on a captured spectrum: the tolerance changes what
# the solver returns.  What it costs is map applications; what it buys is Ritz
# values accurate enough that the null cluster is still recognisable.  At
# `100 eps(Float32)` block Lanczos returns Ritz values good to only ~1e-5
# relative, and on the valence-4 sphere the whole null cluster (true values at
# 4e-8 relative) comes back at 3e-4..6e-3 -- indistinguishable from the first
# nonzero eigenvalue at 1.5e-3, so the solver reports nullity 0 on a
# 4-dimensional null space.
#
#   JL_PROJECT=$(pwd) bench/jl bench/reports/precision-tune-iter.jl
#
using Dleto, ITensors, LinearAlgebra, Random, Printf
using Arpack, KrylovKit, IterativeSolvers

include(joinpath(@__DIR__, "..", "SphereHarness.jl"))
include(joinpath(@__DIR__, "precision-tune.jl"))   # CASES, sylvester_map, quiet

const C_CASES = ("sphere3-d10", "sphere3-d16", "sphere4-d6",
                 "video-20x20x10x3", "rand4-d6")
const C_MULTS = (1.0, 3.0, 10.0, 30.0, 100.0)
const C_SOLVERS = (:ArpackSolver, :KrylovSolver)

rows = String[]
push!(rows, "case,truth,T,solver,mult,eigtol,seconds,nullity,certified,rule,gap," *
            "spec1,spec2,spec3,spec4,spec5")
byname = Dict(c.name => c for c in CASES)

for cname in C_CASES, T in (Float64, Float32, Float16), solver in C_SOLVERS
    case = byname[cname]
    Γ0 = case.build()
    fr = collect(inds(Γ0))
    Γ = T === Float64 ? Γ0 : ITensor(Array{T}(ITensors.array(Γ0, fr...)), fr...)
    Tc = Dleto.compute_eltype(T)
    L, scale, n = sylvester_map(Γ, case.ops)
    for mult in C_MULTS
        et = mult * Float64(eps(real(Tc)))
        t = 0.0; rel = Float64[]
        try
            t = @elapsed res = quiet() do
                Dleto.solve(Dleto.SOLVER_REGISTRY[solver], L;
                            nv = min(24, n), tol = et)
            end
            rel = sort!(Float64[abs(v) / scale for v in res.vals])
        catch e
            push!(rows, join((cname, case.truth, T, solver, mult,
                              @sprintf("%.3g", et), @sprintf("%.2f", t), -1, "throw",
                              first(split(sprint(showerror, e), '\n')), "", "", "", "", "", ""), ","))
            continue
        end
        # score with the tuned verdict policy: floor = 5 eps, gap ratio 100
        fl = 5 * Float64(eps(real(Tc)))
        _, vd = Dleto.gap_verdict(rel, 1.0; threshold = max(1e-12, fl), floor = fl,
                                  data_floor = Dleto.data_floor(T),
                                  gap_ratio = Dleto.GAP_RATIO)
        spec5 = [i <= length(rel) ? rel[i] : NaN for i in 1:5]
        push!(rows, join((cname, case.truth, T, solver, mult, @sprintf("%.3g", et),
                          @sprintf("%.2f", t), vd.nullity, vd.certified, vd.rule,
                          @sprintf("%.4g", vd.gap),
                          (@sprintf("%.3g", x) for x in spec5)...), ","))
        @printf("%-18s %-8s %-13s x%-6g eigtol=%-9.2g %5.2fs nullity=%-3d %s\n",
                cname, string(T), string(solver), mult, et, t, vd.nullity,
                vd.certified ? "certified" : "-")
    end
    L = nothing; Γ = nothing; GC.gc()
end

open(joinpath(@__DIR__, "precision-tune-c.csv"), "w") do io
    for r in rows; println(io, r); end
end
println("wrote precision-tune-c.csv (", length(rows) - 1, " rows)")
