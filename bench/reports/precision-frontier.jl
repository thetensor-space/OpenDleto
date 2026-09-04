#
# Exp. D: WHERE DOES Float32 STOP?  The frontier in `d` and in nullity.
#
# Stage A of bench/reports/precision-tune.jl reached n = 3609 and found the
# relative null values FALLING with size, which is why `precision_floor` has no
# dimension term.  A d = 500 valence-3 run then returned nullity 1 of 3 in
# Float32 (residual 2.23e-4, uncertified) on both CPU and GPU, so something
# grows between n = 3609 and n = 376500 that the small cases do not show.  This
# sweep brackets it, and separates the two candidate causes:
#
#   * `d` -- the arithmetic noise in the null values, which grows with the
#     number of contractions each map application performs, and
#   * the NULLITY -- how many copies of the zero eigenvalue a solver has to
#     resolve at once, which is what `UniversalOp` varies at fixed `d`
#     (nullity 13 on the raw sphere octant against 3 for `SymmetricOp`).
#
# Matrix-free throughout, so memory stays under a gigabyte and the frontier is
# set by wall time, not by RSS.
#
#   JL_PROJECT=$(pwd) bench/jl bench/reports/precision-frontier.jl [maxd]
#
using Dleto, ITensors, LinearAlgebra, Random, Printf
using Arpack, KrylovKit, IterativeSolvers

include(joinpath(@__DIR__, "..", "SphereHarness.jl"))

const MAXD = isempty(ARGS) ? 96 : parse(Int, ARGS[1])
quietly2(f) = redirect_stdout(f, devnull)

"""
    frontier_row(d, valence, ops, T) -> NamedTuple

One point: the scrambled sphere at valence `valence`, dimension `d`, operator
space `ops`, element type `T`, solved matrix-free through the same call
`derTrOpsReduced(::SylverLiningMethod, ...)` makes -- the SQUARED map with the
stored type declared -- plus QuickDer end to end, the route the d = 500 run
took.
"""
function frontier_row(d, valence, ops, T)
    inp = build_sphere(d; valence, T, ops)
    fr = collect(inds(inp.Γ))
    Tc = Dleto.compute_eltype(T)
    Γc = T === Tc ? inp.Γ : ITensor(Array{Tc}(ITensors.array(inp.Γ, fr...)), fr...)
    eng = engaged(inp.ch)
    (Ωr, _) = reduceByEngaged(inp.Ω, eng, Tc)
    L, _ = Dleto.sylvesterLM(Ωr, Matrix{Tc}(inp.ch[:, eng]), Γc)
    n = size(L, 2)
    scale = max(sqrt(max(Dleto.opnorm_estimate(L' * L; iters = 10), 0.0)), eps(real(Tc)))

    nullity, rule, cert, gap, und, spec, tsolve = -1, :throw, false, NaN, 0, Float64[], 0.0
    try
        tsolve = @elapsed out = quietly2() do
            solve_nullspace(L, :ArpackSolver; squared = true, store_eltype = real(T),
                            label = "frontier")
        end
        v = out.verdict
        nullity, rule, cert, gap, und = v.nullity, v.rule, v.certified, v.gap, v.undecidable
        spec = v.spectrum
    catch e
        rule = Symbol(first(split(sprint(showerror, e), '\n')))
    end

    qd_nullity, qd_t, qd_status = -1, 0.0, "ok"
    try
        qd_t = @elapsed out = quietly2() do
            derTrOpsReduced(Dleto.QuickDerMethod(), inp.Ω, inp.ch, inp.Γ)
        end
        qd_nullity = size(out[3], 2)
    catch e
        qd_status = first(split(sprint(showerror, e), '\n'))
    end

    # The two numbers the policy turns on: how high the null cluster sits in
    # units of `eps(T_compute)`, and how far the first nonzero value is above
    # the threshold that was actually used.
    truth = ops isa SymmetricOp ? valence : -1
    nulls = (truth > 0 && length(spec) >= truth) ? spec[1:truth] : Float64[]
    null_max = isempty(nulls) ? NaN : maximum(nulls)
    thr = max(Dleto.TOL_DEFAULT^2, Dleto.precision_floor(T))
    return (; d, valence, ops = string(nameof(typeof(ops))), T, n, truth,
              nullity, rule, cert, gap, und, qd_nullity, qd_status,
              null_max, null_max_eps = null_max / eps(real(Tc)),
              first_nonzero = (truth > 0 && length(spec) > truth) ? spec[truth + 1] : NaN,
              threshold = thr, tsolve, qd_t, rss_GB = Sys.maxrss() / 2^30)
end

const HEADER = "d,valence,ops,T,n,truth,nullity,rule,certified,gap,undecidable," *
               "qd_nullity,qd_status,null_max,null_max_eps,first_nonzero,threshold," *
               "solve_s,qd_s,rss_GB"

rowstr(r) = join((r.d, r.valence, r.ops, r.T, r.n, r.truth, r.nullity, r.rule,
                  r.cert, @sprintf("%.4g", r.gap), r.und, r.qd_nullity,
                  replace(r.qd_status, ',' => ';'),
                  @sprintf("%.4g", r.null_max), @sprintf("%.4g", r.null_max_eps),
                  @sprintf("%.4g", r.first_nonzero), @sprintf("%.4g", r.threshold),
                  @sprintf("%.2f", r.tsolve), @sprintf("%.2f", r.qd_t),
                  @sprintf("%.2f", r.rss_GB)), ",")

rows = String[HEADER]
open(joinpath(@__DIR__, "precision-frontier.csv"), "w") do io
    println(io, HEADER)
end
function emit!(r)
    push!(rows, rowstr(r))
    open(joinpath(@__DIR__, "precision-frontier.csv"), "a") do io
        println(io, rowstr(r))
    end
    fmt(x) = isfinite(x) ? string(round(x; sigdigits = 3)) : "-"
    println("d=", rpad(r.d, 5), "v=", r.valence, " ", rpad(r.ops, 13),
            rpad(string(r.T), 8), "n=", rpad(r.n, 8),
            "truth=", rpad(r.truth, 4),
            "arpack=", rpad(r.nullity, 4),
            rpad(string(r.rule) * (r.cert ? " CERT" : " -"), 12),
            "gap=", rpad(fmt(r.gap), 11),
            rpad(r.und > 0 ? "UND($(r.und))" : "", 8),
            "qd=", rpad(r.qd_nullity, 4),
            "nullmax=", rpad(fmt(r.null_max), 11),
            "(", rpad(fmt(r.null_max_eps), 9), "eps)  ",
            "1st=", rpad(fmt(r.first_nonzero), 11),
            rpad(string(round(r.tsolve; digits = 1)) * "s/" *
                 string(round(r.qd_t; digits = 1)) * "s", 14),
            "rss=", round(r.rss_GB; digits = 1), "G")
    flush(stdout)
end

# Warm up so the first timing is not JIT, per element type (the trap
# precision-study.md section 7.5 records).
for T in (Float64, Float32, Float16)
    try
        quietly2() do
            frontier_row(6, 3, SymmetricOp(), T)
        end
    catch
    end
end

println("=== valence 3, SymmetricOp (truth 3): d frontier ===")
for d in (16, 24, 32, 48, 64, 96, 128, 160)
    d > MAXD && break
    for T in (Float64, Float32, Float16)
        try
            emit!(frontier_row(d, 3, SymmetricOp(), T))
        catch e
            println("d=$d $T FAILED: ", first(split(sprint(showerror, e), '\n')))
        end
        GC.gc()
    end
end

println("=== valence 4, SymmetricOp (truth 4): d frontier ===")
for d in (6, 8, 10, 12, 16)
    for T in (Float64, Float32, Float16)
        try
            emit!(frontier_row(d, 4, SymmetricOp(), T))
        catch e
            println("d=$d v4 $T FAILED: ", first(split(sprint(showerror, e), '\n')))
        end
        GC.gc()
    end
end

# NULLITY at fixed d: the raw octant under UniversalOp has a much larger
# derivation space (13 at valence 3), so this varies the multiplicity the
# solver must resolve without changing the arithmetic's depth.
println("=== valence 3, UniversalOp (larger nullity): fixed-d comparison ===")
for d in (12, 16, 24, 32)
    for T in (Float64, Float32, Float16)
        try
            emit!(frontier_row(d, 3, UniversalOp(), T))
        catch e
            println("d=$d univ $T FAILED: ", first(split(sprint(showerror, e), '\n')))
        end
        GC.gc()
    end
end

println("wrote precision-frontier.csv (", length(rows) - 1, " rows)")
