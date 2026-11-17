#
# QuickDer vs the symmetric-operator null solvers on the scrambled sphere.
#
#   julia -t 2 --project=. bench/reports/QuickDerSphereBench.jl [ds...]
#
# Warm-up at d = 8 for every config, then one timed stratification per
# (d, config) on the SymmetricOp input, printed (and appended to
# bench/reports/quickder-sphere.csv) as soon as it is measured.  Timings are
# under CPU contention from other processes; compare within a run.
#
include(joinpath(@__DIR__, "..", "SphereHarness.jl"))
using Printf
try
    @eval using Arpack
catch e
    @warn "Arpack not loadable: $(sprint(showerror, e))"
end

ds = isempty(ARGS) ? [20, 30, 40, 50] : parse.(Int, ARGS)
configs = [
    (; label = "QuickDer",        T = Float64, cfg = (; method = :QuickDer)),
    (; label = "ArpackSolver",    T = Float64, cfg = (; solver = :ArpackSolver)),
    (; label = "SVDSolver",       T = Float64, cfg = (; solver = :SVDSolver)),
    (; label = "QuickDer",        T = Float32, cfg = (; method = :QuickDer)),
]
csv = joinpath(@__DIR__, "quickder-sphere.csv")
isfile(csv) || open(csv, "w") do io; println(io, "d,T,config,seconds,bytes,nullity,lsq_err,support,perm_ok,status"); end

println("warm-up ..."); flush(stdout)
for T in (Float64, Float32)
    warmup!([c.cfg for c in configs if c.T === T]; d = 8, T = T, passes = 2)
end
@printf("%4s  %-8s %-14s %9s %12s %8s %10s %8s %s\n", "d", "T", "config", "seconds", "MB", "nullity", "lsq_err", "support", "status"); flush(stdout)
for d in ds
    inps = Dict(T => build_sphere(d; T) for T in (Float64, Float32))
    for c in configs
        r = run_stratify(inps[c.T]; c.cfg...)
        @printf("%4d  %-8s %-14s %9.3f %12.1f %8d %10.2e %8.4f %s\n", d, c.T, c.label, r.seconds, r.bytes / 1e6, r.nullity, r.lsq_err, r.support, r.status); flush(stdout)
        open(csv, "a") do io
            println(io, join([d, c.T, c.label, r.seconds, r.bytes, r.nullity, r.lsq_err, r.support, r.perm_ok, r.status], ","))
        end
    end
end
println("done")
