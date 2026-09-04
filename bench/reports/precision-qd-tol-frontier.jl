#
# Exp. F: is QuickDer's Float32 undercount at large `d` a TOLERANCE, or not?
#
# Exp. E brackets the failure: on the scrambled sphere at valence 3, QuickDer in
# Float32 returns the correct nullity 3 at d = 16 and 32, then 2 at d = 64 and 1
# at d = 100 -- the same undercount reported at d = 500 (nullity 1 of 3,
# residual 2.23e-4), reproduced 5x smaller.  Float64 returns 3 at every d, and
# so does Float16, whose `qd_tolerance` is 90x coarser (3.1e-2 against 3.5e-4).
#
# That Float16 -- the WORST arithmetic -- succeeds where Float32 fails is the
# whole clue: if a coarser tolerance is what saves it, the failure is a
# tolerance and `qd_tolerance` needs a term that grows with `d`.  If sweeping
# `tol` upward does not recover the derivations, it is not, and the fault is in
# the solve rather than in the policy.
#
# One sweep, therefore: `tol` from the policy's value up to Float16's, at the
# `d` where Float32 is known to fail.
#
#   JL_PROJECT=$(pwd) bench/jl bench/reports/precision-qd-tol-frontier.jl
#
using Dleto, ITensors, LinearAlgebra, Random, Printf
using Arpack, KrylovKit, IterativeSolvers

include(joinpath(@__DIR__, "..", "SphereHarness.jl"))

const HEADER = "d,valence,T,tol,nullity,truth,seconds,rss_GB,status"
rows = String[HEADER]

function point(d, valence, T, tol)
    inp = build_sphere(d; valence, T)
    nullity, status, t = -1, "ok", 0.0
    try
        t = @elapsed out = redirect_stdout(devnull) do
            derTrOpsReduced(Dleto.QuickDerMethod(), inp.Ω, inp.ch, inp.Γ; tol = tol)
        end
        nullity = size(out[3], 2)
    catch e
        status = replace(first(split(sprint(showerror, e), '\n')), ',' => ';')
    end
    push!(rows, join((d, valence, T, @sprintf("%.3g", tol), nullity, valence,
                      @sprintf("%.2f", t), @sprintf("%.2f", Sys.maxrss() / 2^30),
                      status), ","))
    println("d=", rpad(d, 5), rpad(string(T), 8),
            "tol=", rpad(@sprintf("%.3g", tol), 10),
            "-> qd_tolerance=", rpad(@sprintf("%.3g", Dleto.qd_tolerance(T, tol)), 10),
            "nullity=", rpad(nullity, 4), "(truth ", valence, ")  ",
            rpad(string(round(t; digits = 1)) * "s", 8),
            "rss=", round(Sys.maxrss() / 2^30; digits = 1), "G  ", status)
    flush(stdout)
end

# warm up
try
    redirect_stdout(devnull) do
        for T in (Float32, Float64)
            point(6, 3, T, 1e-6)
        end
    end
catch
end
empty!(rows); push!(rows, HEADER)

println("=== does a coarser tol recover the Float32 derivations? ===")
for d in (64, 100)
    for tol in (1e-6, 1e-5, 1e-4, 3.45e-4, 1e-3, 3e-3, 1e-2, 3e-2)
        try
            point(d, 3, Float32, tol)
        catch e
            println("FAILED: ", first(split(sprint(showerror, e), '\n')))
        end
        GC.gc()
    end
    # Float64 control at the same `d`, to confirm nothing else changed.
    try
        point(d, 3, Float64, 1e-6)
    catch
    end
    GC.gc()
end

open(joinpath(@__DIR__, "precision-qd-tol-frontier.csv"), "w") do io
    for r in rows; println(io, r); end
end
println("wrote precision-qd-tol-frontier.csv (", length(rows) - 1, " rows)")
