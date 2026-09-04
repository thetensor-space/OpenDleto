module DletoArpackExt
using Dleto
using Arpack
using LinearMaps
using LinearAlgebra

export ArpackDenseSolver, ArpackSolver

struct ArpackSolver <: Dleto.NullSolver end
struct ArpackDenseSolver <: Dleto.NullSolver end

"""
    solve(::ArpackSolver, L; nv, tol, ncv, maxiter, min_request)

    The `nv` smallest-magnitude eigenpairs of the square symmetric map `L` by
    implicitly restarted Lanczos (ARPACK `dsaupd`, `which = :SM`).

    ARPACK is asked for at least `min_request = 16` pairs even when fewer are
    wanted, and the `nv` smallest are returned.  This is the resilience knob,
    and it is nearly free.  A null space is a multiple eigenvalue, and a
    single-vector Krylov method only picks up its extra copies through roundoff
    amplified by restarts; the more pairs requested, the more restarts before
    convergence is declared and the more reliably the copies emerge.  Measured
    on the sphere benchmark (bench/reports/exp2-seeds.csv, nullity 3, 5 seeds
    at d = 32, 36, 40): `nev = 4` missed a null vector in 3 of 15 runs,
    `nev = 16` in none, and the number of map applications was *flat* in
    `nev` -- 409..684 for `nev = 16` against 503..2484 for `nev = 4`, because
    ARPACK's restart with very few wanted values is inefficient.  Returning
    only the `nv` smallest keeps the caller's escalation rule sound: when the
    nullity is at least `nv`, every returned value is null and the caller
    doubles its request.

    `ncv = 8*nev` (128 for the default request) gave the flattest cost,
    437..546 applications across every seed; `tol = 1e-10` is relative to the
    Ritz value inside ARPACK and, for a zero eigenvalue, the stopping test is
    effectively at machine precision whatever `tol` is, which is why the null
    residuals come out at 1e-15 and the reconstruction matches the dense SVD.
    Shift-invert (`sigma`) is not offered on a matrix-free map: the inner
    solve would be CG on `L + shift*I`, whose condition number is
    `‖L‖/shift`, and a shift small enough to separate the null space from
    the first nonzero eigenvalue costs more CG steps per application than the
    whole `:SM` solve.
"""
function Dleto.solve(::ArpackSolver, L::LinearMap; nv::Integer = 20, tol::Real = 1e-10,
                     ncv::Union{Nothing,Integer} = nothing, maxiter::Integer = 300,
                     min_request::Integer = 16)
    println("Using ArpackSolver...")
    n = size(L, 2)
    @assert size(L, 1) == n "ArpackSolver needs a square map; pass AᵗA."
    T = eltype(L)
    RT = typeof(real(zero(T)))
    want = clamp(nv, 1, n)

    # ARPACK needs nev < n - 1 and ncv <= n; below that a dense eigen is free.
    if n <= 32
        E = eigen(Symmetric(Matrix(L)))
        return (; vals = RT.(E.values[1:want]), vecs = T.(E.vectors[:, 1:want]),
                  converged = true)
    end

    nev = clamp(max(want, min_request), 1, n - 2)
    ncv_ = ncv === nothing ? min(n, max(2 * nev + 1, 8 * nev)) : clamp(Int(ncv), nev + 1, n)
    # ARPACK's test is relative to the Ritz value with an eps^(2/3) floor, so
    # it survives Float32 at any `tol`; flooring at 100*eps(T) just stops it
    # polishing below the element type's precision (~20% fewer applications
    # in Float32, no change in Float64).
    #
    # `nconv` and `niter` are ARPACK's own account of what it managed, and
    # `solve_nullspace` needs them: a non-converged solve whose Ritz values
    # have merely STALLED near zero produces a spectrum with a clean gap in
    # it, so a caller reading the spectrum alone reports a confident nullity
    # (often 0) for what is a failure.  ARPACK does not say WHICH pairs
    # converged, so `converged` is the honest conservative test, `nconv >=
    # nev`; the caller acts on it only where it can matter (see
    # `NullVerdict`'s `status`).  Note that exhausting `maxiter` is not this
    # case: `Arpack.eigs` throws `XYAUPD_Exception` for that, which propagates.
    vals, vecs, nconv, niter = Arpack.eigs(L; nev = nev, ncv = ncv_, which = :SM,
                                           tol = max(tol, 100 * eps(RT)),
                                           maxiter = maxiter)
    λ = real.(vals)
    ord = sortperm(abs.(λ))[1:min(want, length(λ))]
    return (; vals = λ[ord], vecs = real.(vecs[:, ord]),
              converged = nconv >= nev, nconv = Int(nconv), niter = Int(niter))
end

function Dleto.solve(::ArpackDenseSolver, L::LinearMap; nv::Integer = 20)
    println("Using ArpackDenseSolver...")
    M = Matrix(L) # Convert LinearMap to dense Matrix to allow LU-Factorization
    nev = clamp(nv, 1, size(M, 2) - 2)
    vals, vecs = Arpack.eigs(M; nev=nev, which=:LM, sigma=0.0)
    return (;vals=real.(vals), vecs=real.(vecs))
end

function __init__()
    println("Loading Dleto Arpack Extension")
    Dleto.register_solver!(:ArpackSolver, ArpackSolver())
    Dleto.register_solver!(:ArpackDenseSolver, ArpackDenseSolver())
end




end
