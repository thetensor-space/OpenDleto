module DletoKrylovKitExt

using Dleto
using LinearMaps
using LinearAlgebra

import KrylovKit
using KrylovKit: eigsolve
using Random

export KrylovSolver

struct KrylovSolver <: Dleto.NullSolver end

# Takes `seed` and draws its start block from it; see the `rng` line in `solve`.
Dleto.wants_seed(::KrylovSolver) = true

# Block Lanczos puts a block as wide as the request through the map on every
# step, so its cost is close to linear in the request (bench/reports/
# exp3-nv0.csv: ~500 applications for 4, ~1800 for 16).  Start small and let
# `solve_nullspace` double the request when the block saturates.
Dleto.initial_request(::KrylovSolver) = 4

"""
    solve(::KrylovSolver, L; nv, tol, krylovdim, maxiter, blocksize, seed)

    The `nv` smallest eigenpairs of the square symmetric map `L` by **block**
    Lanczos, with the block as wide as the request.

    WHY BLOCK.  A null space is a *multiple* eigenvalue -- every null vector has
    eigenvalue exactly 0 -- and a Krylov space grown from a single start vector
    contains exactly one direction of a multiple eigenspace (the projection of
    the start vector onto it).  The other copies appear only through roundoff,
    amplified by restarts, and whether they have appeared by the time the
    requested number of Ritz pairs has converged is luck.  Measured on the
    sphere benchmark (bench/reports/exp1-grid.csv, exp2-seeds.csv): with the
    nullity 3, single-vector Lanczos or Arnoldi asked for 4 pairs returned 1 or
    2 null vectors and then *converged* on genuine nonzero eigenvalues, so the
    caller saw a clean bracket and reported nullity 1 or 2; asked for 16 pairs
    it was right most of the time but still missed 1 seed in 5.  This was the
    "KrylovSolver returned nullity 2 at d = 35" failure.  A block of `b`
    independent start vectors carries `min(b, multiplicity)` copies from the
    first step, so with the block equal to the request the caller's escalation
    rule -- double the request while every returned value is null -- is sound:
    the block saturates exactly when the nullity is at least the request.
    Cost is the same as ARPACK's (390-550 map applications per solve on the
    benchmark at d = 32..40, block 4), and it was right on every seed.

    TOLERANCE is relative to `‖L‖` (power-iteration estimate, 10 matvecs):
    KrylovKit's `tol` is an absolute residual norm, and the operators here
    range over many orders of magnitude in scale.  `1e-12` relative gives null
    residuals of 1e-13..1e-15 and a reconstruction indistinguishable from the
    dense SVD.  The tolerance is floored at `100*eps(T)`, so in Float32 it is
    1.2e-5 relative.

    `krylovdim` defaults to `max(128, 8*blocksize)`; the per-solve cost was
    flat in it from 64 upward and it bounds the memory (`krylovdim * n`
    numbers).  A non-symmetric map falls back to single-vector Arnoldi, which
    is not a path Dleto's callers take -- `solve_nullspace` squares every
    rectangular map into a symmetric one.
"""
function Dleto.solve(::KrylovSolver, L::LinearMap; nv::Integer = 10, tol::Real = 1e-12,
                     krylovdim::Union{Nothing,Integer} = nothing, maxiter::Integer = 200,
                     blocksize::Union{Nothing,Integer} = nothing,
                     seed::Union{Nothing,Integer} = nothing)
    println("Using KrylovSolver...")
    n = size(L, 1)
    @assert n == size(L, 2) "KrylovSolver needs a square map; pass AᵗA."
    T = eltype(L)
    RT = typeof(real(zero(T)))
    nev = clamp(nv, 1, n)

    # A Krylov method on a space this small is pure overhead, and the block
    # (as wide as the request) would not fit in it.
    #
    # BELOW Float64 THE GATE HAS TO BE 4x WIDER, and the reason is a measured
    # breakdown, not a preference.  KrylovKit's `BlockLanczos` in Float32 loses
    # null vectors when the block is a large fraction of `n`: with the default
    # request (nev = 24) on the scrambled sphere it returns nullity 0 of 4 at
    # n = 84 and 1 of 3 at n = 165, is correct at n = 408, 909 and 3609, and is
    # correct at every one of those in Float64.  Sweeping its stopping
    # tolerance over 1 .. 100 x eps(Float32) changes nothing, so this is the
    # block re-orthogonalization losing rank in half the mantissa, not a
    # tolerance (bench/reports/precision-tune-c.csv).
    #
    # It matters more than a wrong count usually would: the values it returns
    # are 3e-4..6.5e-3 relative, ALL above the null threshold, so the verdict
    # is a CERTIFIED nullity 0 -- confidently "this tensor has no derivations"
    # about a tensor with four.  A dense `eigen` at n = 165 costs milliseconds,
    # so the gate is simply widened where the block method cannot be trusted.
    # `8 * (nev + 1)` is 200 at the default request: past the largest observed
    # failure (165) and below the smallest observed success (408).
    dense_gate = RT === Float64 ? 2 * (nev + 1) : 8 * (nev + 1)
    if n <= max(32, dense_gate)
        E = eigen(Symmetric(Matrix(L)))
        take = min(nev, n)
        return (; vals = RT.(E.values[1:take]), vecs = T.(E.vectors[:, 1:take]),
                  converged = true)
    end

    # The start block is arbitrary but it must be the SAME arbitrary block on a
    # repeated call, or a multiple eigenvalue is resolved differently each time
    # (see `Dleto.wants_seed`).  Without a `seed` this is the task-local
    # default RNG, i.e. the old behaviour.
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    scale = Dleto.opnorm_estimate(L; iters = 10, rng = rng)
    # KrylovKit's `tol` is an absolute residual norm, and no Lanczos residual
    # gets below a small multiple of `eps(T) * ‖L‖`.  `1e-12` relative is
    # reachable in Float64 but 1e5 times below the Float32 floor, where block
    # Lanczos then restarts to `maxiter` for nothing: 7-38 s and 19-50 GB per
    # solve at d = 30..40 against 0.3-1 s with the floor, same nullity
    # (bench/reports/precision-study.md, Exp. 5 and 5b).  `100*eps(RT)` is
    # 2.2e-14 in Float64, so Float64 behaviour is unchanged.
    atol = Dleto.iter_tol(RT, tol) * max(scale, eps(RT))

    if LinearAlgebra.issymmetric(L)
        bs = blocksize === nothing ? nev : clamp(blocksize, 1, nev)
        kd = krylovdim === nothing ? max(128, 8 * bs) : Int(krylovdim)
        kd = clamp(max(kd, nev), nev, max(nev, n - bs))
        x0 = KrylovKit.Block([randn(rng, T, n) for _ in 1:bs])
        alg = KrylovKit.BlockLanczos(; krylovdim = kd, maxiter = maxiter, tol = atol,
                                     verbosity = 0)
        # `BlockLanczos` normalizes its block through a Cholesky-type step, and
        # in Float32 that step can see a numerically negative Gram: measured on
        # the near-degenerate `ramp + noise` tensors, `DomainError with
        # -1.25e-7` out of a `sqrt`, and `ArgumentError: blocklength must be
        # >(0)` when the block collapses entirely.  Neither is a statement
        # about the tensor, so neither should reach the caller as a crash --
        # fall back to the single-vector Arnoldi, which is less reliable on
        # multiplicities (it finds one copy of a repeated eigenvalue per start
        # vector) but does not break down.  The undercount that risks is
        # reported by the convergence warning below and by an uncertified
        # verdict; a `DomainError` from inside a dependency is reported by
        # nothing.
        vals_a, vecs_a, info = try
            eigsolve(L, x0, nev, :SR, alg)
        catch err
            err isa Union{DomainError,ArgumentError,LinearAlgebra.PosDefException} || rethrow()
            @warn "KrylovSolver: block Lanczos broke down in $RT " *
                  "($(first(split(sprint(showerror, err), '\n')))); retrying with " *
                  "single-vector Arnoldi, which may undercount a repeated " *
                  "eigenvalue. Consider :ArpackSolver or Float64." maxlog = 1
            eigsolve(L, randn(rng, T, n), nev, :SR;
                     krylovdim = clamp(max(kd, nev), nev, n),
                     maxiter = maxiter, tol = atol, verbosity = 0)
        end
    else
        kd = krylovdim === nothing ? min(n, max(64, 4 * nev)) : min(n, Int(krylovdim))
        vals_a, vecs_a, info = eigsolve(L, randn(rng, T, n), nev, :SR;
                                        krylovdim = kd, maxiter = maxiter, tol = atol,
                                        verbosity = 0)
    end

    converged = info.converged >= nev
    info.converged < nev && @warn "KrylovSolver: $(info.converged) of $nev eigenvalues " *
        "converged to relative tolerance $tol in $(info.numiter) restarts " *
        "($(info.numops) map applications); returning the Ritz pairs as computed."

    λ = real.(vals_a)
    ord = sortperm(λ)
    # Contract: `vecs` is a matrix whose COLUMNS are the vectors.
    V = isempty(vecs_a) ? zeros(T, n, 0) : reduce(hcat, (real.(vecs_a[j]) for j in ord))
    # `converged` travels with the answer, not only into a warning: block
    # Lanczos that converged NOTHING still returns Ritz values, and on the
    # whitened restricted map at d = 300..500 those stall at 1e-9 relative and
    # read as a clean, certified nullity of 0.  `solve_nullspace` needs the
    # flag to call that what it is (see `NullVerdict`'s `status`).
    return (; vals = λ[ord], vecs = V, converged = converged,
              nconverged = Int(info.converged), restarts = Int(info.numiter))
end

function __init__()
    println("Loading Dleto KrylovKit Extension")
    # Registration must happen here, not at module level: top-level effects in
    # a precompiled module are captured at precompile time and discarded.
    Dleto.register_solver!(:KrylovSolver, KrylovSolver())
end



end
