module DletoIterativeSolversExt

using Dleto
using LinearMaps
using LinearAlgebra

import IterativeSolvers: lobpcg
using IterativeSolvers

export  LanczosSolver,  CGSolver

struct LanczosSolver <: Dleto.NullSolver end
struct CGSolver <: Dleto.NullSolver end



"""
    solve(::LanczosSolver, L; nv, tol)

    Approximate null vectors from a partial SVD by Lanczos bidiagonalization.

    CAVEAT: `svdl` converges to the *largest* singular values, and the null
    space is at the *smallest* end, so reaching it means asking for essentially
    the whole spectrum.  That is expensive and, on the composed operator (whose
    spread here is some twenty-five orders of magnitude), the small end is the
    least accurate part of what comes back.  `SVDSolver` is preferable whenever
    the map can be densified; this exists for the case where it cannot.

    `svdl` returns `(F, history)` with `F` an `SVD` object -- not `(S, V)`.
    Destructuring it as `S, V` bound `S` to the whole factorization and `V` to
    the convergence history, so the very next line died with
    `MethodError: no method matching length(::SVD{...})`.  It also `println`ed
    the entire factorization.
"""
function Dleto.solve(::LanczosSolver, L::LinearMap; nv::Integer = 10, tol = 1e-10)
    println("Using LanczosSolver...")
    nsv = min(nv, minimum(size(L)) - 1)   # svdl needs nsv < min(size)
    nsv < 1 && return (; vals = Float64[], vecs = zeros(Float64, size(L, 2), 0))

    F, _ = IterativeSolvers.svdl(L; nsv = nsv, vecs = :right, tol = tol)
    S = F.S
    V = F.Vt'                              # columns are right singular vectors

    # Smallest first: those are the candidate null directions.
    ord = sortperm(S)
    keep = ord[1:min(nv, length(ord))]
    return (; vals = S[keep], vecs = V[:, keep])
end


"""
    solve(::CGSolver, L; nv, tol, maxiter)

    Smallest eigenpairs of the square symmetric `L` by LOBPCG.

    CAVEAT: LOBPCG is a preconditioned method and is used here without a
    preconditioner.  `L` is the composed operator `AᵗA`, whose condition number
    is `κ(A)²`, so on a poorly conditioned chisel/tensor pair it can return
    nothing below tolerance at all.  When that happens the honest answer is an
    empty basis, which is what the caller's eigenvalue filter produces.

    It used to hardwire a block size of 10, ignoring `nv` entirely, and ignore
    `tol`, so `nv` was decorative.
"""
function Dleto.solve(::CGSolver, L::LinearMap; nv::Integer = 10, tol = 1e-10,
                     maxiter::Integer = 500)
    println("Using CGSolver...")
    n = size(L, 2)
    @assert size(L, 1) == n "CGSolver needs a square map; pass AᵗA."
    # LOBPCG refuses a block larger than a third of the dimension ("not stable
    # to use when the matrix size is less than 3 times the block size"), so a
    # request for the whole space has to be clamped rather than forwarded.
    blocksize = clamp(nv, 1, max(1, n ÷ 3))

    # Even within that bound a large block is Cholesky-factorized internally,
    # and on `AᵗA` -- condition number κ(A)² -- that factorization fails with
    # PosDefException.  Halving the block until it succeeds is what makes this
    # solver usable at all: a smaller block still finds the smallest
    # eigenpairs, which is the only end we want.
    while true
        try
            res = lobpcg(L, false, blocksize; tol = tol, maxiter = maxiter)
            blocksize < nv && @warn "CGSolver: LOBPCG block $blocksize of $nv " *
                "requested (dim = $n); the basis returned may be partial."
            return (; vals = res.λ, vecs = res.X)
        catch e
            e isa PosDefException || rethrow()
            blocksize == 1 && rethrow()
            blocksize = blocksize ÷ 2
        end
    end
end

function __init__()
    println("Loading Dleto IterativeSolvers Extension")
    Dleto.register_solver!(:LanczosSolver, LanczosSolver())
    Dleto.register_solver!(:CGSolver, CGSolver())
end



end
