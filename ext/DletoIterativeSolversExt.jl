module DletoIterativeSolversExt

using Dleto
using LinearMaps
using LinearAlgebra

import IterativeSolvers: lobpcg, lsmr!
using IterativeSolvers

export  LanczosSolver,  CGSolver, LSMRSolver

struct LanczosSolver <: Dleto.NullSolver end
struct CGSolver <: Dleto.NullSolver end

# ============================================================================
# LSMRSolver -- smallest singular vectors WITHOUT squaring the operator
# ============================================================================
#
# Every other iterative solver here is an eigensolver, so it needs `AᵗA`, and
# squaring is not merely inelegant -- it costs half the available precision at
# exactly the place we cannot afford it.
#
# Measured on the densor map at n = 10 (20000 x 1000): σ_max = 1.45e3, the ten
# null directions sit at σ ≈ 3e-13, and the smallest NONZERO singular value is
# 2.2e-2.  So the null space is separated by 1.5e-5 relative in σ -- an
# enormous, comfortable gap -- and by 2.3e-10 relative in σ².  Still above eps
# at this size, but it is losing an order of magnitude of headroom per step in
# n, and it is entirely unnecessary.
#
# THE METHOD IS NOT AN EIGENSOLVER.  It is a projection, which is what makes it
# work without needing to know anything about the spectrum:
#
#     null(A) = { w - A⁺(A w) : w arbitrary }
#
# because `A⁺A` is the orthogonal projector onto the row space, so `w - A⁺A w`
# is the component of `w` orthogonal to it -- exactly the null space.  And
# `A⁺y` is what LSMR computes: the minimum-norm least-squares solution, using
# only `A` and `Aᵗ`.  One LSMR solve per candidate vector, no shift, no
# eigenvalue iteration, nothing squared.
#
# This replaced a shift-invert subspace iteration, which is the textbook
# approach and does not work here: its inner solve is
# `(AᵗA + σI)x = v`, conditioned at `σ_max/√σ`, and choosing σ requires
# knowing the spectral gap in advance.  With σ small enough to separate the
# null space, the stacked map had κ ≈ 1e7 and LSMR needed thousands of
# iterations; capped at 400 it returned mid-spectrum directions (σ ≈ 250-500
# out of 1451), i.e. noise.  The projection has no such parameter: the only
# conditioning that matters is `σ_max/σ_min⁺` restricted to the row space,
# 6.6e4 here, which LSMR handles in a few hundred iterations.
#
# The nullity comes out for free.  Project `k > d` random vectors and they span
# the d-dimensional null space, so a rank-revealing QR reports `d` -- no need
# to guess how many vectors to ask for, and the caller's escalation only has to
# fire when `d == k`.
struct LSMRSolver <: Dleto.NullSolver
    lsmr_tol::Float64       # least-squares tolerance for the projection
    lsmr_maxiter::Int
    rank_tol::Float64       # relative pivot threshold for the revealing QR
    margin::Int             # extra candidate vectors beyond nv
    refine::Int             # iterative-refinement passes over the projection
end
LSMRSolver(; lsmr_tol = 1e-12, lsmr_maxiter = 1000, rank_tol = 1e-8,
           margin = 8, refine = 1) =
    LSMRSolver(lsmr_tol, lsmr_maxiter, rank_tol, margin, refine)

# It takes the rectangular map -- that is the entire point.
Dleto.wants_square(::LSMRSolver) = false
Dleto.densifies(::LSMRSolver, L) = false

function Dleto.solve(m::LSMRSolver, L::LinearMap; nv::Integer = 10, tol = 1e-10,
                     kwargs...)
    println("Using LSMRSolver...")
    rows, n = size(L)
    k = clamp(nv + m.margin, 1, n)
    T = eltype(L)
    RT = typeof(real(zero(T)))

    # One projection step:  z <- z - A⁺(A z).
    # FLOOR BOTH TOLERANCES ON `T`.  The struct's defaults (`lsmr_tol = 1e-12`,
    # `rank_tol = 1e-8`) are Float64 numbers chosen before the element type is
    # known, and both sit BELOW `eps(Float32) = 1.2e-7`, so in Float32 the
    # stopping test was unreachable and the rank cut was made inside the
    # roundoff -- which is how this solver returned nullity 20 on a Float16
    # tensor whose true nullity was 3.
    #
    # `lsmr_tol` gets `eps(T)`, NOT the `100 eps(T)` that `Dleto.iter_tol` gives
    # the Krylov methods, and that difference is measured rather than stylistic.
    # LSMR's `atol`/`btol` are relative tests INSIDE the inner least-squares
    # solve, and `project!` applies the projector repeatedly (see the
    # refinement note below): each pass starts from a smaller `‖y‖`, so the
    # composite accuracy is not capped by one pass's `eps(T) ‖A‖‖z‖` the way a
    # single Lanczos residual is.  At `100 eps(Float32) = 1.2e-5` the sphere at
    # d = 10 lost 270x of recovery (1.4e-6 -> 4.0e-4 relative) for a 15% time
    # saving, and the valence-4 case lost a null vector outright (4 -> 3).  At
    # `eps(Float32) = 1.2e-7` both are restored.  Below `eps(T)` a single pass
    # cannot resolve anything, and `lsmr_maxiter` is the real bound.
    lsmr_tol = max(Float64(m.lsmr_tol), Float64(eps(RT)))
    x = Vector{T}(undef, n)
    function project!(z::AbstractVector)
        y = L * z                        # y is in range(A) by construction,
        fill!(x, zero(T))                # so this system is consistent and
        lsmr!(x, L, y; atol = lsmr_tol, btol = lsmr_tol,
              maxiter = m.lsmr_maxiter)  # LSMR returns the min-norm solution.
        z .-= x
        return z
    end

    # Project k random vectors off the row space, then REFINE.
    #
    # A single pass leaves an error of roughly `lsmr_tol * kappa_row`, because
    # the least-squares residual is relative to ‖Az‖ and gets divided by the
    # smallest nonzero singular value on the way back.  Measured at n = 10:
    # sigma came out 2e-5 where the true null directions are at 3e-13.  The
    # projector is idempotent in exact arithmetic, so applying it again to an
    # already-nearly-null vector is standard iterative refinement -- ‖Az‖ is
    # now tiny, so the same relative tolerance buys a far smaller absolute
    # correction, and the error contracts by the same factor each pass.
    Z = Matrix{T}(undef, n, k)
    for j in 1:k
        z = randn(T, n)
        project!(z)
        for _ in 1:m.refine
            project!(z)
        end
        Z[:, j] = z
    end

    # Rank-revealing QR gives both an orthonormal basis and the nullity.  The
    # column-pivoted factorization is essential here: `qr` without pivoting
    # would hand back k columns whatever the true rank, and the extra ones
    # would be numerical noise indistinguishable from null vectors.
    F = qr(Z, ColumnNorm())
    R = F.R
    lead = abs(R[1, 1])
    rank_tol = max(Float64(m.rank_tol), Dleto.rank_rtol(RT, size(Z)...))
    d = lead == 0 ? 0 :
        count(i -> abs(R[i, i]) > rank_tol * lead, 1:min(size(R)...))
    d == 0 && return (; vals = RT[], vecs = zeros(T, n, 0))

    # Return the d null vectors PLUS one column above the rank threshold.
    #
    # Not cosmetic: `solve_nullspace` decides it has seen the whole null space
    # when something comes back that is NOT null, and doubles its request when
    # everything is.  A solver that returns only the vectors it believes are
    # null therefore never lets the caller bracket, and the request doubles up
    # to the full dimension -- 868,000 map applications at n = 10 before this
    # was caught.  Handing back the first rejected direction lets the generic
    # bracketing work unchanged, and when d == k there is nothing to hand back,
    # which is exactly the case where escalating IS correct.
    take = min(d + 1, k)
    V = Matrix(F.Q)[:, 1:take]

    # Report sigma = ‖Av‖ straight from A: no squaring anywhere in the chain.
    vals = [norm(L * V[:, j]) for j in 1:take]
    ord = sortperm(vals)
    return (; vals = vals[ord], vecs = V[:, ord])
end



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
    T = eltype(L)
    nsv < 1 && return (; vals = typeof(real(zero(T)))[], vecs = zeros(T, size(L, 2), 0))

    # `tol` is a relative residual tolerance; floored on `T` so a Float32 run
    # stops instead of iterating below its own noise (src/solvers/Precision.jl).
    F, _ = IterativeSolvers.svdl(L; nsv = nsv, vecs = :right,
                                 tol = Dleto.iter_tol(real(T), tol))
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
    # HEADROOM.  A block method converges to the `blocksize` smallest
    # eigenpairs at a rate governed by the gap to eigenvalue `blocksize + 1`.
    # Asked for exactly as many vectors as the null space has, that gap is the
    # one we care about but the block has no room to work in, and convergence
    # is worst precisely in the case of interest.  So search in a larger
    # subspace than requested and return the smallest `nv` of it.
    #
    # This is why the early measurements looked better than the method is:
    # asked for 40 vectors of a 10-dimensional null space it converged nicely,
    # and asked for 16 of a 15-dimensional one it returned nothing.
    margin = max(8, nv ÷ 2)
    blocksize = clamp(nv + margin, 1, max(1, n ÷ 3))
    # LOBPCG's `tol` is an absolute residual norm; floored on `T`.  This does
    # NOT make LOBPCG usable below Float64 -- its Float32 Cholesky of the block
    # Gram matrix fails whatever the tolerance and the block collapses to 1-3
    # vectors (precision-study.md, Exp. 2b, 5), which is why
    # `matrix_free_solvers` drops :CGSolver below Float64.  It only stops a
    # Float32 call that does get here from running to `maxiter`.
    tol = Dleto.iter_tol(real(eltype(L)), tol)

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
            # Return the smallest `nv` of the enlarged block, smallest first:
            # the extra vectors were search space, not answers, and the caller
            # counts how many fall below tolerance.
            ord = sortperm(res.λ)[1:min(nv, length(res.λ))]
            return (; vals = res.λ[ord], vecs = res.X[:, ord])
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
    Dleto.register_solver!(:LSMRSolver, LSMRSolver())
end



end
