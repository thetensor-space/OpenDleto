#
# Precision study, experiment 5: validate the proposed tolerance rules end to
# end, without editing src/ or ext/.
#
#   julia -t 2 --project=. bench/reports/precision-exp5.jl
#
# Proposed rule: an iterative eigensolver's stopping tolerance must scale with
# the element type.  KrylovKit and LOBPCG take an ABSOLUTE residual norm, so
# use sqrt(eps(T)) * ||L||; ARPACK's is relative to the Ritz value, so use
# sqrt(eps(T)).  Registered here as :KrylovP, :CGP, :ArpackP.
#
# Mixed variants (:KrylovP64, :CGP64, :ArpackP64) apply the Float32 map inside
# a Float64 eigensolver -- Float32 storage and contractions, Float64 Krylov
# arithmetic -- to see whether that recovers Float64-like robustness at
# Float32 memory.
#
# Everything runs through run_stratify, so nullity and recovery are scored the
# same way as experiment 1.
#
using KrylovKit, IterativeSolvers, Arpack
using LinearMaps
include(joinpath(@__DIR__, "..", "SphereHarness.jl"))
using Printf
LinearAlgebra.BLAS.set_num_threads(2)

const DIR = @__DIR__
const CSV = joinpath(DIR, "precision-exp5.csv")
isfile(CSV) || open(CSV, "w") do io
    println(io, "d,T,solver,seed,tol,seconds,bytes,nullity,lsq_err,support,status")
end
function emit(line)
    println(line); flush(stdout)
    open(CSV, "a") do io; println(io, line); end
end

# ------------------------------------------------------------ proposed solvers
struct KrylovP <: Dleto.NullSolver end
struct CGP <: Dleto.NullSolver end
struct ArpackP <: Dleto.NullSolver end
struct Promote64 <: Dleto.NullSolver
    inner::Dleto.NullSolver
end

"Float64 view of a map of any real element type: promote in, demote out."
function promote64(L::LinearMap)
    n = size(L, 2); T = eltype(L)
    T === Float64 && return L
    LinearMap{Float64}(v -> Float64.(L * T.(v)), n, n; issymmetric = true, ismutating = false)
end

function Dleto.solve(::KrylovP, L::LinearMap; nv::Integer = 10, tol = nothing)
    n = size(L, 2); T = eltype(L); RT = real(T)
    nrm = Dleto.opnorm_estimate(L; iters = 5)
    τ = tol === nothing ? sqrt(eps(RT)) * max(nrm, one(RT)) : tol
    nev = min(nv, n)
    x0 = randn(T, n)
    vals, vecs, info = KrylovKit.eigsolve(L, x0, nev, :SR; tol = τ, krylovdim = min(max(10, 2nev), n),
                                          maxiter = 100, verbosity = 0)
    V = isempty(vecs) ? zeros(T, n, 0) : reduce(hcat, real.(vecs))
    return (; vals = real.(vals), vecs = V)
end

function Dleto.solve(::CGP, L::LinearMap; nv::Integer = 10, tol = nothing, maxiter = 500)
    n = size(L, 2); T = eltype(L); RT = real(T)
    nrm = Dleto.opnorm_estimate(L; iters = 5)
    τ = tol === nothing ? sqrt(eps(RT)) * max(nrm, one(RT)) : tol
    blocksize = clamp(nv + max(8, nv ÷ 2), 1, max(1, n ÷ 3))
    while true
        try
            res = IterativeSolvers.lobpcg(L, false, blocksize; tol = τ, maxiter)
            ord = sortperm(res.λ)[1:min(nv, length(res.λ))]
            return (; vals = res.λ[ord], vecs = res.X[:, ord])
        catch e
            e isa PosDefException || rethrow()
            blocksize == 1 && rethrow()
            blocksize ÷= 2
        end
    end
end

function Dleto.solve(::ArpackP, L::LinearMap; nv::Integer = 20, tol = nothing)
    RT = real(eltype(L))
    τ = tol === nothing ? sqrt(eps(RT)) : tol
    nev = clamp(nv, 1, size(L, 2) - 2)
    vals, vecs = Arpack.eigs(L; nev, which = :SM, tol = τ)
    return (; vals = real.(vals), vecs = real.(vecs))
end

function Dleto.solve(m::Promote64, L::LinearMap; kwargs...)
    r = Dleto.solve(m.inner, promote64(L); kwargs...)
    T = eltype(L)
    return (; vals = real(T).(r.vals), vecs = T.(r.vecs))
end

Dleto.register_solver!(:KrylovP, KrylovP())
Dleto.register_solver!(:CGP, CGP())
Dleto.register_solver!(:ArpackP, ArpackP())
Dleto.register_solver!(:KrylovP64, Promote64(KrylovP()))
Dleto.register_solver!(:CGP64, Promote64(CGP()))
Dleto.register_solver!(:ArpackP64, Promote64(ArpackP()))

# ------------------------------------------------------------------ the runs
const SOLVERS32 = [:ArpackSolver, :ArpackP, :ArpackP64, :KrylovSolver, :KrylovP, :KrylovP64, :CGP, :CGP64]
const SOLVERS64 = [:ArpackSolver, :ArpackP, :KrylovSolver, :KrylovP, :CGSolver, :CGP]

for (T, solvers) in ((Float32, SOLVERS32), (Float64, SOLVERS64))
    println("== warmup $T"); flush(stdout)
    warmup!([(; solver = s) for s in solvers]; T)
    for d in (30, 40), seed in (d, d + 1000, d + 2000)
        inp = build_sphere(d; T, seed)
        for s in solvers
            # In Float32 also try the threshold lifted to 100*eps(Float32) ~ 1.2e-5.
            tols = T === Float32 ? (1e-6, 1.2e-5) : (1e-6,)
            for tol in tols
                # The stock CGSolver in Float32 is known broken (exp1); skip.
                r = run_stratify(inp; solver = s, tol)
                emit(@sprintf("%d,%s,%s,%d,%.1e,%.3f,%d,%d,%.3e,%.6f,\"%s\"",
                    d, T, s, seed, tol, r.seconds, r.bytes, r.nullity, r.lsq_err, r.support, r.status))
            end
        end
    end
end
println("EXP5 DONE")
