module DletoKrylovKitExt

using Dleto
using LinearMaps
using LinearAlgebra

import KrylovKit
using KrylovKit: eigsolve

export KrylovSolver

struct KrylovSolver <: Dleto.NullSolver end

"""
    solve(::KrylovSolver, L; nv, tol)

    Smallest-magnitude eigenpairs of the (square, symmetric) map `L` by Arnoldi.

    `L` is the composed derivation--densor operator, i.e. `AᵗA` for the densor
    map `A`, so its condition number is `κ(A)²`.  Arnoldi is the right tool for
    a *few* extremal eigenpairs of such an operator and the wrong tool for all
    of them: asking for the full space is what `nd <= 0` does, and Arnoldi will
    then stop on an invariant subspace short of the request.  That is reported
    once and the partial result returned, since a partial basis of genuine null
    vectors is still useful and the caller filters on `vals`.

    Two bugs fixed here.  It returned `vecs` as a `Vector` of vectors while the
    `solve` contract (and every caller) wants the vectors as *columns of a
    matrix*: `derTrOpsReduced` does `result.vecs[:, i] for i in 1:size(vecs,2)`,
    which for a 1-D array is a single element, and the subsequent
    eigenvalue filter then indexed a 1-element vector with the list of valid
    positions -- `BoundsError`.  And the retry loop re-ran the identical solve
    five times when KrylovKit had already reported a fixed invariant subspace,
    each attempt dumping the full residual vector to stderr (25MB at n = 19).
"""
function Dleto.solve(::KrylovSolver, L::LinearMap; nv::Integer = 10, tol::Float64 = 1e-8)
    println("Using KrylovSolver...")
    n = size(L, 1)
    @assert n == size(L, 2) "KrylovSolver needs a square map; pass AᵗA."
    nev = min(nv, n)
    x0 = randn(size(L, 2))

    max_attempts = 5
    maxiter = 100
    krylovdim = min(max(10, 2 * nev), n)

    λ = Float64[]
    vecs = Vector{Float64}[]
    last_converged = -1

    for attempt in 1:max_attempts
        # verbosity=0 keeps KrylovKit from printing the whole residual vector
        # on every non-convergence; we summarise below instead.
        vals_a, vecs_a, info = eigsolve(L, x0, nev, :SR;
            maxiter=maxiter,
            krylovdim=krylovdim,
            tol=tol,
            verbosity=0,
        )

        λ = real.(vals_a)
        vecs = [real.(v) for v in vecs_a]

        info.converged >= nev && break

        # No progress since the previous attempt means Arnoldi has found all
        # there is to find at this tolerance -- retrying is pure waste.
        if info.converged <= last_converged || krylovdim >= n
            @warn "KrylovSolver: $(info.converged) of $nev eigenvalues converged; " *
                  "returning the converged subspace. Use :SVDSolver for a full basis."
            break
        end
        last_converged = info.converged

        maxiter = Int(round(maxiter * 1.5))
        krylovdim = min(Int(round(krylovdim * 1.5)), n)
        x0 = randn(size(L, 2))

        if attempt == max_attempts
            @warn "KrylovSolver: $(info.converged) of $nev eigenvalues converged " *
                  "after $max_attempts attempts; returning what converged."
        end
    end

    # Contract: `vecs` is a matrix whose COLUMNS are the vectors.
    V = isempty(vecs) ? zeros(Float64, size(L, 2), 0) : reduce(hcat, vecs)
    return (; vals = λ, vecs = V)
end

function __init__()
    println("Loading Dleto KrylovKit Extension")
    # Registration must happen here, not at module level: top-level effects in
    # a precompiled module are captured at precompile time and discarded.
    Dleto.register_solver!(:KrylovSolver, KrylovSolver())
end



end
