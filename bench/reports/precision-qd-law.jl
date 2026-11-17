#
# Exp. G: the LAW.  What tolerance does QuickDer need in Float32, as a function
# of `d`?
#
# Exp. F established that the Float32 undercount at large `d` is a tolerance and
# nothing else: at d = 64 and d = 100 the nullity is 2 and 1 at every `tol` up
# to `qd_tolerance(Float32) = sqrt(eps(Float32)) = 3.45e-4`, and 3 at every
# `tol` from 1e-3 up.  A clean step, so there is a threshold to measure.
#
# This finds it per `d`: the smallest `tol` on a 1/3-decade grid that recovers
# the full nullity.  From the exponent, `qd_tolerance` gets a size term with a
# constant fitted here rather than guessed -- and the d = 500 report becomes a
# check on the extrapolation instead of the only data point.
#
#   JL_PROJECT=$(pwd) bench/jl bench/reports/precision-qd-law.jl
#
using Dleto, ITensors, LinearAlgebra, Random, Printf
using Arpack, KrylovKit, IterativeSolvers

include(joinpath(@__DIR__, "..", "SphereHarness.jl"))

# 1/3-decade grid from well below `sqrt(eps(Float32))` to well above
const TOLS = [1e-4, 2.15e-4, 3.45e-4, 4.64e-4, 6.81e-4, 1e-3, 1.47e-3, 2.15e-3,
              3.16e-3, 4.64e-3, 1e-2, 2.15e-2, 4.64e-2, 1e-1]

const HEADER = "d,valence,T,r,need_tol,need_tol_eps,need_over_sqrteps,nullity_at_policy," *
               "policy_tol,seconds_at_need"
rows = String[HEADER]

"""
    smallest_working_tol(d, valence, T) -> (tol, nullity_at_policy, seconds)

Walk `TOLS` upward and stop at the first value that returns the full nullity.
`nothing` if none does.  Also reports what the policy's own tolerance gives, so
the row says whether the default is on the right side of the threshold.
"""
function smallest_working_tol(d, valence, T)
    inp = build_sphere(d; valence, T)
    truth = valence
    run(tol) = try
        redirect_stdout(devnull) do
            size(derTrOpsReduced(Dleto.QuickDerMethod(), inp.Ω, inp.ch, inp.Γ;
                                 tol = tol)[3], 2)
        end
    catch
        -1
    end
    at_policy = run(Dleto.TOL_DEFAULT)
    need, secs = nothing, 0.0
    for tol in TOLS
        # `qd_tolerance` floors, so a `tol` under the floor is the floor; skip
        # the duplicates rather than re-running them.
        # `qd_tolerance` floors, so any `tol` under the floor IS the floor and
        # would re-run the identical solve; skip those.
        if Dleto.qd_tolerance(T, tol) ≈ Dleto.qd_tolerance(T) &&
           !(tol ≈ Dleto.qd_tolerance(T))
            continue
        end
        t = @elapsed k = run(tol)
        if k == truth
            need, secs = tol, t
            break
        end
    end
    return (need, at_policy, secs, inp)
end

println("=== the smallest tol that recovers the full nullity, vs d (valence 3) ===")
for T in (Float32, Float64), d in (16, 32, 48, 64, 100, 140)
    T === Float64 && !(d in (64, 140)) && continue      # a control at two points
    local need, at_policy, secs
    try
        (need, at_policy, secs, inp) = smallest_working_tol(d, 3, Float32 === T ? Float32 : Float64)
        eng = engaged(inp.ch)
        r = Dleto._qdn_restriction_sizes(collect(inp.dims), eng, 3)
        e = eps(real(Dleto.compute_eltype(T)))
        needv = need === nothing ? NaN : Float64(need)
        push!(rows, join((d, 3, T, join(r, "x"),
                          @sprintf("%.3g", needv), @sprintf("%.4g", needv / e),
                          @sprintf("%.3g", needv / sqrt(e)),
                          at_policy, @sprintf("%.3g", Dleto.qd_tolerance(T)),
                          @sprintf("%.2f", secs)), ","))
        println("d=", rpad(d, 5), rpad(string(T), 8), "r=", rpad(join(r, "x"), 12),
                "need_tol=", rpad(@sprintf("%.3g", needv), 10),
                "= ", rpad(@sprintf("%.4g", needv / e), 9), "eps  = ",
                rpad(@sprintf("%.3g", needv / sqrt(e)), 8), "sqrt(eps)   ",
                "policy(", @sprintf("%.3g", Dleto.qd_tolerance(T)), ") gives ",
                at_policy, " of 3", at_policy == 3 ? "" : "   <-- UNDERCOUNT")
        flush(stdout)
    catch e
        println("d=$d $T FAILED: ", first(split(sprint(showerror, e), '\n')))
    end
    GC.gc()
end

open(joinpath(@__DIR__, "precision-qd-law.csv"), "w") do io
    for r in rows; println(io, r); end
end
println("wrote precision-qd-law.csv (", length(rows) - 1, " rows)")
