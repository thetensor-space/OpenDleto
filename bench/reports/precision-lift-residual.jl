#
# Exp. E: how does QuickDer's LIFT RESIDUAL grow with `d`, and in which type?
#
# This is the quantity behind the d = 500 report (Float32, valence 3: nullity 1
# of 3, residual 2.23e-4, uncertified -- identically on CPU and GPU, so it is
# arithmetic and not the device).  `_qdn_solve_and_lift` filters the restricted
# null space by
#
#     C = nullspace(Rall; atol = qd_tolerance(T), rtol = 0)
#
# where `Rall` is the stack of lift-equation residuals, already divided by the
# tensor's scale.  A restricted solution that is not the restriction of a true
# derivation leaves a residual there; a true one leaves only rounding.  So the
# smallest singular values of `Rall` ARE the rounding in the lift, and
# `qd_tolerance(T)` has to sit above them.  It currently does not depend on `d`
# at all (`max(tol, sqrt(eps(T_store)))`), while the lift contracts over a full
# axis of length `d` -- so if that rounding grows with `d`, the filter
# eventually discards true derivations, which is exactly the reported failure
# mode: an UNDERCOUNT, not a merge.
#
# QuickDer's cost is set by the restriction sizes `r` (~sqrt(3d) at valence 3),
# not by `d`, so this reaches d = 200 cheaply where a full solve could not.
# It records, per (d, T): the `Rall` spectrum around the cut, the tolerance in
# force, and the nullity QuickDer actually returned.
#
#   JL_PROJECT=$(pwd) bench/jl bench/reports/precision-lift-residual.jl [maxd]
#
using Dleto, ITensors, LinearAlgebra, Random, Printf, Logging
using Arpack, KrylovKit, IterativeSolvers

include(joinpath(@__DIR__, "..", "SphereHarness.jl"))

const MAXD = isempty(ARGS) ? 200 : parse(Int, ARGS[1])

"""
A logger that keeps the `svdvals(Rall)` string out of QuickDer's `@debug
"QuickDer lift residual spectrum"` and drops everything else, so the
measurement costs one debug message per lift instead of a source edit.
"""
mutable struct LiftSpy <: AbstractLogger
    spectra::Vector{String}
end
Logging.min_enabled_level(::LiftSpy) = Logging.Debug
Logging.shouldlog(::LiftSpy, level, _module, group, id) = true
Logging.catch_exceptions(::LiftSpy) = false
function Logging.handle_message(s::LiftSpy, level, message, _module, group, id,
                                file, line; kwargs...)
    occursin("lift residual spectrum", string(message)) || return nothing
    for (k, v) in kwargs
        k === :svals && push!(s.spectra, string(v))
    end
    return nothing
end

const HEADER = "d,valence,T,r,qd_tol,qd_nullity,truth,status,seconds,rss_GB," *
               "n_below_tol,smallest5,around_cut"
rows = String[HEADER]
open(joinpath(@__DIR__, "precision-lift-residual.csv"), "w") do io
    println(io, HEADER)
end

"""Parse the logged `svdvals` string back into numbers, smallest first."""
function parse_svals(s::AbstractString)
    body = strip(s, ['[', ']'])
    vals = Float64[]
    for tok in split(body, ',')
        v = tryparse(Float64, strip(tok))
        v === nothing || push!(vals, v)
    end
    return sort!(vals)
end

function run_point(d, valence, T)
    inp = build_sphere(d; valence, T)
    eng = engaged(inp.ch)
    r = Dleto._qdn_restriction_sizes(collect(inp.dims), eng, valence)
    spy = LiftSpy(String[])
    nullity, status, t = -1, "ok", 0.0
    try
        t = @elapsed out = with_logger(spy) do
            redirect_stdout(devnull) do
                derTrOpsReduced(Dleto.QuickDerMethod(), inp.Ω, inp.ch, inp.Γ)
            end
        end
        nullity = size(out[3], 2)
    catch e
        status = replace(first(split(sprint(showerror, e), '\n')), ',' => ';')
    end
    tol = Dleto.qd_tolerance(T)
    # The LAST logged lift spectrum is the one the returned answer came from
    # (an `r` bump would log a second).
    vals = isempty(spy.spectra) ? Float64[] : parse_svals(last(spy.spectra))
    nbelow = count(<(tol), vals)
    fmt(x) = @sprintf("%.3g", x)
    small5 = join(map(fmt, vals[1:min(5, length(vals))]), " ")
    # the values bracketing the cut: last below the tolerance, first above
    icut = nbelow
    around = (icut >= 1 && icut < length(vals)) ?
             "$(fmt(vals[icut])) | $(fmt(vals[icut + 1]))" :
             (isempty(vals) ? "-" : "all " * (nbelow == 0 ? "above" : "below"))
    row = join((d, valence, T, join(r, "x"), fmt(tol), nullity, valence, status,
                @sprintf("%.2f", t), @sprintf("%.2f", Sys.maxrss() / 2^30),
                nbelow, small5, around), ",")
    push!(rows, row)
    open(joinpath(@__DIR__, "precision-lift-residual.csv"), "a") do io
        println(io, row)
    end
    println("d=", rpad(d, 5), "v=", valence, " ", rpad(string(T), 8),
            "r=", rpad(join(r, "x"), 12), "qd_tol=", rpad(fmt(tol), 11),
            "nullity=", rpad(nullity, 4), "(truth ", valence, ")  ",
            "below_tol=", rpad(nbelow, 4),
            "Rall smallest: ", rpad(small5, 46),
            "cut: ", rpad(around, 22),
            rpad(string(round(t; digits = 1)) * "s", 9),
            "rss=", round(Sys.maxrss() / 2^30; digits = 1), "G  ", status)
    flush(stdout)
    return nothing
end

# Warm up per element type so the first timing is not JIT.
for T in (Float64, Float32, Float16)
    try
        redirect_stdout(devnull) do
            run_point(6, 3, T)
        end
    catch
    end
end
empty!(rows); push!(rows, HEADER)
open(joinpath(@__DIR__, "precision-lift-residual.csv"), "w") do io
    println(io, HEADER)
end

println("=== valence 3: the lift residual against qd_tolerance, as d grows ===")
for d in (16, 32, 64, 100, 140, 200, 280)
    d > MAXD && break
    for T in (Float64, Float32, Float16)
        try
            run_point(d, 3, T)
        catch e
            println("d=$d $T FAILED: ", first(split(sprint(showerror, e), '\n')))
        end
        GC.gc()
    end
end

println("=== valence 4 ===")
for d in (8, 12, 16, 24)
    for T in (Float64, Float32, Float16)
        try
            run_point(d, 4, T)
        catch e
            println("d=$d v4 $T FAILED: ", first(split(sprint(showerror, e), '\n')))
        end
        GC.gc()
    end
end

println("wrote precision-lift-residual.csv (", length(rows) - 1, " rows)")
