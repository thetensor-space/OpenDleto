#
# Baseline probe: what does each (element type, solver route) do TODAY?
#
# Answers the question the precision policy has to be built against: which
# routes throw, which return a wrong nullity silently, and which are already
# fine.  One small d so this is seconds, not minutes.
#
#   JL_PROJECT=$(pwd) bench/jl bench/reports/precision-probe0.jl
#
using Dleto, ITensors, LinearAlgebra, Random
using Arpack, KrylovKit, IterativeSolvers

include(joinpath(@__DIR__, "..", "SphereHarness.jl"))

const ROUTES = [
    (; method = :SylverLining, solver = :SVDSolver),
    (; method = :SylverLining, solver = :GramSolver),
    (; method = :SylverLining, solver = :ArpackSolver),
    (; method = :SylverLining, solver = :KrylovSolver),
    (; method = :SylverLining, solver = :LSMRSolver),
    (; method = :SylverLining, solver = :AutoSolver),
    (; method = :QuickDer,),
]

routelabel(r) = haskey(r, :solver) ? "$(r.method)/$(r.solver)" : "$(r.method)"

function probe(d, valence, T)
    inp = try
        build_sphere(d; valence, T)
    catch e
        println("  build_sphere FAILED: ", first(split(sprint(showerror, e), '\n')))
        return
    end
    println("  Γ eltype = ", eltype(inp.Γ), ", dims = ", inp.dims)
    for r in ROUTES
        kw = haskey(r, :solver) ? (; solver = r.solver) : (;)
        out = try
            run_stratify(inp; method = r.method, kw...)
        catch e
            println("    ", rpad(routelabel(r), 30), " THROWN  ",
                    first(split(sprint(showerror, e), '\n')))
            continue
        end
        println("    ", rpad(routelabel(r), 30), " nullity=", rpad(out.nullity, 4),
                " lsq=", rpad(string(round(out.lsq_err; sigdigits = 3)), 12),
                " t=", rpad(string(round(out.seconds; digits = 2)), 8),
                out.status == "ok" ? "" : out.status)
    end
end

for (d, valence) in ((10, 3), (6, 4))
    for T in (Float64, Float32, Float16)
        println("=== d=$d valence=$valence T=$T  (truth: nullity $valence) ===")
        probe(d, valence, T)
    end
end
