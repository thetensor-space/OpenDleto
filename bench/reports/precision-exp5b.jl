#
# Precision study, experiment 5b: the CURRENT KrylovKit extension's block
# Lanczos with a type-aware tolerance floor, in Float32 and Float64.
#
#   julia -t 2 --project=. bench/reports/precision-exp5b.jl
#
# Copies ext/DletoKrylovKitExt.jl's solve body verbatim except for one line:
#     atol = max(tol, floor(RT)) * max(scale, eps(RT))
# with floor = 100*eps(RT) (:KrylovB100) or sqrt(eps(RT)) (:KrylovBsqrt).
# The stock :KrylovSolver (floor none) is run alongside for reference.
#
using KrylovKit, IterativeSolvers, Arpack
using LinearMaps
include(joinpath(@__DIR__, "..", "SphereHarness.jl"))
using Printf
LinearAlgebra.BLAS.set_num_threads(2)

const CSV = joinpath(@__DIR__, "precision-exp5b.csv")
isfile(CSV) || open(CSV, "w") do io
    println(io, "d,T,solver,seed,seconds,bytes,nullity,lsq_err,support,status")
end
function emit(line)
    println(line); flush(stdout)
    open(CSV, "a") do io; println(io, line); end
end

struct KrylovB <: Dleto.NullSolver
    floor::Function
end

function Dleto.solve(m::KrylovB, L::LinearMap; nv::Integer = 10, tol::Real = 1e-12,
                     krylovdim = nothing, maxiter::Integer = 200, blocksize = nothing)
    n = size(L, 1)
    T = eltype(L)
    RT = typeof(real(zero(T)))
    nev = clamp(nv, 1, n)
    if n <= max(32, 2 * (nev + 1))
        E = eigen(Symmetric(Matrix(L)))
        take = min(nev, n)
        return (; vals = RT.(E.values[1:take]), vecs = T.(E.vectors[:, 1:take]))
    end
    scale = Dleto.opnorm_estimate(L; iters = 10)
    atol = max(tol, m.floor(RT)) * max(scale, eps(RT))        # <-- the one changed line
    bs = blocksize === nothing ? nev : clamp(blocksize, 1, nev)
    kd = krylovdim === nothing ? max(128, 8 * bs) : Int(krylovdim)
    kd = clamp(max(kd, nev), nev, max(nev, n - bs))
    x0 = KrylovKit.Block([randn(T, n) for _ in 1:bs])
    alg = KrylovKit.BlockLanczos(; krylovdim = kd, maxiter = maxiter, tol = atol, verbosity = 0)
    vals_a, vecs_a, info = KrylovKit.eigsolve(L, x0, nev, :SR, alg)
    λ = real.(vals_a)
    ord = sortperm(λ)
    V = isempty(vecs_a) ? zeros(T, n, 0) : reduce(hcat, (real.(vecs_a[j]) for j in ord))
    return (; vals = λ[ord], vecs = V)
end

Dleto.register_solver!(:KrylovB100, KrylovB(RT -> 100 * eps(RT)))
Dleto.register_solver!(:KrylovBsqrt, KrylovB(RT -> sqrt(eps(RT))))

const SOLVERS = [:KrylovSolver, :KrylovB100, :KrylovBsqrt]
for T in (Float32, Float64)
    println("== warmup $T"); flush(stdout)
    warmup!([(; solver = s) for s in SOLVERS]; T)
    for d in (30, 40), seed in (d, d + 1000, d + 2000)
        inp = build_sphere(d; T, seed)
        for s in SOLVERS
            # The stock solver in Float32 is the known-slow case; one seed per d is enough.
            T === Float32 && s === :KrylovSolver && seed != d && continue
            r = run_stratify(inp; solver = s)
            emit(@sprintf("%d,%s,%s,%d,%.3f,%d,%d,%.3e,%.6f,\"%s\"",
                d, T, s, seed, r.seconds, r.bytes, r.nullity, r.lsq_err, r.support, r.status))
        end
    end
end
println("EXP5B DONE")
