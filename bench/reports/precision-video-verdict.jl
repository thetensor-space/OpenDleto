#
# The claim in docs/design/Precision-Policy.md section 5 that most needs
# checking end to end rather than on a captured spectrum: at `FLOOR_EPS = 5`
# a Float32 video-shaped box is CERTIFIED, and the same box in Float16 is
# reported UNDECIDABLE rather than certified or silently wrong.
#
#   JL_PROJECT=$(pwd) bench/jl bench/reports/precision-video-verdict.jl
#
using Dleto, ITensors, LinearAlgebra, Random, Printf
using Arpack, KrylovKit, IterativeSolvers

include(joinpath(@__DIR__, "precision-tune.jl"))   # CASES, sylvester_map, quiet

byname = Dict(c.name => c for c in CASES)

for cname in ("video-20x20x10x3", "video-40x40x20x3", "sphere3-d16", "neardeg-1e-3")
    case = byname[cname]
    Γ0 = case.build()
    fr = collect(inds(Γ0))
    for T in (Float64, Float32, Float16)
        Γ = T === Float64 ? Γ0 : ITensor(Array{T}(ITensors.array(Γ0, fr...)), fr...)
        L, _, n = sylvester_map(Γ, case.ops)
        for solver in (:SVDSolver, :ArpackSolver)
            out = try
                quiet() do
                    # Exactly the call `derTrOpsReduced(::SylverLiningMethod, ...)`
                    # makes: the squared map, and the STORED type declared.
                    solve_nullspace(L, solver; squared = true, store_eltype = real(T),
                                    label = "video")
                end
            catch e
                @printf("%-18s %-8s %-13s THROWN %s\n", cname, string(T),
                        string(solver), first(split(sprint(showerror, e), '\n')))
                continue
            end
            v = out.verdict
            @printf("%-18s %-8s %-13s truth=%-3d nullity=%-3d %-12s gap=%-10.3g %s%s\n",
                    cname, string(T), string(solver), case.truth, v.nullity,
                    string(v.rule) * (v.certified ? " CERTIFIED" : " -"),
                    v.gap, v.undecidable > 0 ? "UNDECIDABLE($(v.undecidable))" : "",
                    v.floor_binding ? " floor-bound" : "")
        end
        L = nothing; GC.gc()
    end
end
