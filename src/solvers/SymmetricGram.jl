#
# SymmetricGram -- dense normal equations for cubic symmetric derivations.
#
# This is deliberately a narrow method.  It is the production form of the
# symmetric-sphere Gram prototype: a direct dense normal matrix is much smaller
# than the rectangular derivation map when every axis is restricted to
# `SymmetricOp()`, but it is not a general replacement for SylverLining.
#

using LinearAlgebra
using LinearMaps
using Random
using SparseArrays

"""
    SymmetricGramMethod(; eigensolver=:inverse, seed=20260914,
                         iterations=12, oversample=24, shift=1e-10,
                         max_bytes=1 << 30)

Dense normal-equation solver for a real cubic tensor with `SymmetricOp()` on
each axis and `UniversalChisel(3)`.  `nd` is required at the call and means
exactly the requested number of smallest modes; it is never inferred from
`tol`.  The modes are returned in Dleto's raw `SymmetricOp` coordinates.

`:inverse` uses shifted inverse subspace iteration.  Its `tol` is the relative
normal-eigen residual required to stop before `iterations`; the report has
`status = :unconverged` if the cap is reached.  `:eigen` is the dense direct
eigensolve validation path and does not need an iterative stopping test.

`shift` is relative to the largest diagonal entry of the normal matrix.
`max_bytes` is a conservative pre-allocation cap for the normal matrix and
its dense construction temporaries.
"""
struct SymmetricGramMethod <: DerivationMethod
    eigensolver::Symbol
    seed::Int
    iterations::Int
    oversample::Int
    shift::Float64
    max_bytes::Int
end

function SymmetricGramMethod(; eigensolver::Symbol=:inverse,
                             seed::Integer=20260914,
                             iterations::Integer=12,
                             oversample::Integer=24,
                             shift::Real=1e-10,
                             max_bytes::Integer=1 << 30)
    eigensolver in (:inverse, :eigen) || throw(ArgumentError(
        "SymmetricGramMethod: eigensolver must be :inverse or :eigen, got :$eigensolver"))
    iterations > 0 || throw(ArgumentError("SymmetricGramMethod: iterations must be positive"))
    oversample >= 0 || throw(ArgumentError("SymmetricGramMethod: oversample must be nonnegative"))
    isfinite(shift) && shift > 0 || throw(ArgumentError(
        "SymmetricGramMethod: shift must be finite and positive"))
    max_bytes > 0 || throw(ArgumentError("SymmetricGramMethod: max_bytes must be positive"))
    return SymmetricGramMethod(eigensolver, Int(seed), Int(iterations), Int(oversample),
                               Float64(shift), Int(max_bytes))
end

function _symmetric_gram_validate(method::SymmetricGramMethod, Ω::TransverseOps,
                                  P::AbstractMatrix, Γ::ITensor, nd, tol)
    ndims(Γ) == 3 || throw(ArgumentError("SymmetricGram requires a three-way ITensor"))
    Ω isa IndTransverseOps || throw(ArgumentError(
        "SymmetricGram requires independent transverse operators (SymmetricOps(Γ))"))
    valence(Ω) == 3 || throw(ArgumentError("SymmetricGram requires valence 3"))
    all(op -> op isa SymmetricOp, Ω.localOps) || throw(ArgumentError(
        "SymmetricGram requires SymmetricOp() on every axis"))
    frames(Ω) == collect(inds(Γ)) || throw(ArgumentError(
        "SymmetricGram requires Ω frames to match Γ in order"))
    size(P) == (1, 3) && all(x -> x == one(x), P) || throw(ArgumentError(
        "SymmetricGram requires the exact 1×3 all-ones UniversalChisel(3)"))
    # Read only metadata until the allocation budget has passed: `array` can
    # densify a sparse ITensor, which is precisely the allocation this guard is
    # meant to prevent on an unsupported size.
    T = eltype(Γ)
    (T === Float64 || T === Float32) || throw(ArgumentError(
        "SymmetricGram supports only real Float64 or Float32 tensors, got $T"))
    dims = ITensors.dim.(frames(Ω))
    length(unique(dims)) == 1 || throw(ArgumentError(
        "SymmetricGram requires equal axis dimensions, got $(Tuple(dims))"))
    nd isa Integer && nd > 0 || throw(ArgumentError(
        "SymmetricGram requires a positive integer nd; it is a fixed mode count"))
    isfinite(tol) && tol > 0 || throw(ArgumentError(
        "SymmetricGram requires a finite positive tol for inverse-iteration residuals"))
    d = only(unique(dims))
    db = BigInt(d)
    lbig = db * (db + 1) ÷ 2
    nbig = 3 * lbig
    nd <= nbig || throw(ArgumentError("SymmetricGram requested nd = $nd, but only $nbig modes exist"))

    # One normal matrix, its factor/work copy, construction temporaries, and
    # the inverse-iteration probes.  This is intentionally conservative: fail
    # before `zeros(T, n, n)` rather than relying on an allocator failure.
    kp = min(nbig, BigInt(nd) + method.oversample)
    estimated_big = BigInt(sizeof(T)) * (3 * nbig * nbig + 4 * db^4 + 6 * nbig * kp)
    estimated_big <= method.max_bytes || throw(ArgumentError(
        "SymmetricGram estimates $(estimated_big) bytes for d=$d, above max_bytes=$(method.max_bytes); " *
        "raise max_bytes explicitly or use :SylverLining"))
    nbig <= typemax(Int) || throw(ArgumentError(
        "SymmetricGram normal matrix dimension $nbig cannot be represented on this platform"))
    return T, d, Int(lbig), Int(nbig), Int(nd), Int(estimated_big)
end

function _symmetric_frobenius_basis(::Type{T}, d::Int) where {T}
    l = d * (d + 1) ÷ 2
    rows = Int[]
    cols = Int[]
    vals = T[]
    invsqrt2 = inv(sqrt(T(2)))
    col = 0
    for j in 1:d, i in 1:j
        col += 1
        push!(rows, i + (j - 1) * d); push!(cols, col)
        push!(vals, i == j ? one(T) : invsqrt2)
        if i != j
            push!(rows, j + (i - 1) * d); push!(cols, col); push!(vals, invsqrt2)
        end
    end
    return sparse(rows, cols, vals, d * d, l)
end

function _symmetric_gram_normal(G::Array{T,3}, d::Int, l::Int) where {T<:Union{Float32,Float64}}
    B = _symmetric_frobenius_basis(T, d)
    H = zeros(T, 3l, 3l)
    for a in 1:3
        rest = [q for q in 1:3 if q != a]
        A = reshape(permutedims(G, (a, rest...)), d, :)
        C = A * transpose(A)
        ra = (a - 1) * l + 1:a * l
        H[ra, ra] .= Matrix(transpose(B) * kron(Matrix{T}(I, d, d), C) * B)
    end
    for a in 1:2, b in (a + 1):3
        other = only(setdiff(1:3, (a, b)))
        U = reshape(permutedims(G, (a, b, other)), d * d, d)
        K = U * transpose(U)
        Hab = reshape(permutedims(reshape(K, d, d, d, d), (1, 3, 4, 2)), d * d, d * d)
        cab = Matrix(transpose(B) * Hab * B)
        ra = (a - 1) * l + 1:a * l
        rb = (b - 1) * l + 1:b * l
        H[ra, rb] .= cab
        H[rb, ra] .= transpose(cab)
    end
    return H
end

function _symmetric_raw_scale(::Type{T}, d::Int, l::Int) where {T}
    s = ones(T, l)
    p = 0
    for j in 1:d, i in 1:j
        p += 1
        i == j || (s[p] = inv(sqrt(T(2))))
    end
    return vcat(s, s, s)
end

function _symmetric_inverse_subspace(H::Matrix{T}, k::Int, method::SymmetricGramMethod,
                                     tol::Real) where {T<:Union{Float32,Float64}}
    n = size(H, 1)
    kp = min(n, k + method.oversample)
    maxdiag = max(maximum(abs, diag(H)), eps(T))
    shift = T(method.shift) * maxdiag
    factor = nothing
    for _ in 1:10
        trial = cholesky(Symmetric(H + shift * I); check=false)
        if issuccess(trial)
            factor = trial
            break
        end
        shift *= T(10)
    end
    factor === nothing && error("SymmetricGram: shifted Cholesky failed after 10 attempts")

    rng = MersenneTwister(method.seed)
    Q = Matrix(qr(randn(rng, T, n, kp)).Q)[:, 1:kp]
    hnorm = max(opnorm(H, Inf), eps(T))
    used = 0
    for iter in 1:method.iterations
        Q = Matrix(qr(factor \ Q).Q)[:, 1:kp]
        small = eigen(Symmetric(transpose(Q) * H * Q))
        vectors = Q * small.vectors[:, 1:k]
        normal_residuals = [norm(H * view(vectors, :, j) - small.values[j] * view(vectors, :, j)) / hnorm
                            for j in 1:k]
        used = iter
        # Keep the prototype's eight-pass baseline before optional residual
        # stopping.  The residual alone can be tiny after one pass when the
        # shift dominates, while the subspace is still not useful for the final
        # unsquared Rayleigh--Ritz extraction.
        if iter >= 8 && maximum(normal_residuals) <= tol
            break
        end
    end
    return Q, used, shift
end

function _symmetric_unsquared_ritz(Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor,
                                   Q::Matrix{T}, scale::Vector{T}, k::Int,
                                   H::Matrix{T}) where {T<:Union{Float32,Float64}}
    # The normal matrix finds the right subspace cheaply.  Final selection is
    # deliberately on the rectangular derivation map, not on H: squaring loses
    # digits in the near modes that matter to a noisy sphere.
    _, E = sylvesterLM(Ω, P, Γ)
    rawQ = Q .* reshape(scale, :, 1)
    Y = hcat((E * view(rawQ, :, j) for j in axes(rawQ, 2))...)
    # `Y` is wide at d=1 (and can be wide at d=2): thin SVD omits the right
    # null complement exactly where the requested modes live.  Keep the full
    # right factor and explicitly append its mathematically zero singulars.
    # Full factors are needed only for a wide map, where the omitted right
    # complement consists of exact zero modes.  On the normal d=50 path Y is
    # tall; requesting `full=true` there would needlessly form a d^3-by-d^3 U.
    F = svd(Y; full=size(Y, 1) < size(Y, 2))
    singulars = zeros(T, size(rawQ, 2))
    singulars[1:length(F.S)] .= F.S
    order = sortperm(singulars)[1:k]
    values = singulars[order] .^ 2
    vectors = Q * F.V[:, order]
    hnorm = max(opnorm(H, Inf), eps(T))
    normal_residuals = [norm(H * view(vectors, :, j) - values[j] * view(vectors, :, j)) / hnorm
                        for j in 1:k]
    return values, vectors, normal_residuals
end

"""
    derTrOpsReduced(::SymmetricGramMethod, Ω, P, Γ; nd, tol,
                    return_diagnostics=false)

Return exactly `nd` smallest normal modes for the narrow symmetric cubic
setting accepted by `SymmetricGramMethod`.  They are fixed approximate modes;
the result never claims an exact nullity.  With diagnostics, the appended
`DerivationReport` has `policy = :fixed_nd`, `certified = false`, the
normal-eigen convergence status, and a full Z-law residual for every returned
operator tuple.
"""
function derTrOpsReduced(method::SymmetricGramMethod, Ω::TransverseOps,
                         P::AbstractMatrix, Γ::ITensor;
                         tol::Real=TOL_DEFAULT, nd=-1, progress=false,
                         return_diagnostics::Bool=false)
    T, d, l, n, k, estimated = _symmetric_gram_validate(method, Ω, P, Γ, nd, tol)
    G0 = ITensors.array(Γ, frames(Ω)...)
    G = G0 isa Array{T,3} ? G0 : Array{T,3}(G0)
    tbuild = time_ns()
    H = _symmetric_gram_normal(G, d, l)
    build_seconds = (time_ns() - tbuild) / 1e9
    scale = _symmetric_raw_scale(T, d, l)
    tsolve = time_ns()
    values, vectors, normal_residuals, status, used, shift = if method.eigensolver === :eigen
        F = eigen(Symmetric(H), 1:k)
        hnorm = max(opnorm(H, Inf), eps(T))
        r = [norm(H * view(F.vectors, :, j) - F.values[j] * view(F.vectors, :, j)) / hnorm
             for j in 1:k]
        (F.values, F.vectors, r, :ok, 0, zero(T))
    else
        Q, niter, shift_used = _symmetric_inverse_subspace(H, k, method, tol)
        vals, vecs, residuals = _symmetric_unsquared_ritz(Ω, P, Γ, Q, scale, k, H)
        state = maximum(residuals) <= tol ? :ok : :unconverged
        (vals, vecs, residuals, state, niter, shift_used)
    end
    solve_seconds = (time_ns() - tsolve) / 1e9

    # Orthonormal Frobenius coordinates use (E_ij + E_ji)/sqrt(2); Dleto's
    # SymmetricOp coordinate is the raw shared off-diagonal matrix entry.
    ders = copy(vectors)
    ders .*= reshape(scale, :, 1)
    id_map = LinearMaps.LinearMap(identity, identity, n, n; ismutating=false)
    if !return_diagnostics
        status === :ok || @warn "SymmetricGram inverse iteration did not reach tol = $tol " *
            "within $(method.iterations) iterations; returning fixed approximate modes. " *
            "Use return_diagnostics=true to inspect residuals." maxlog=1
        return (Ω, id_map, ders)
    end

    hnorm = max(opnorm(H, Inf), eps(T))
    spectrum = sqrt.(max.(Float64.(values), 0.0)) ./ sqrt(Float64(hnorm))
    report = DerivationReport(; method=:SymmetricGram, store_eltype=T,
                              compute_eltype=T, dims=[d, d, d],
                              policy=:fixed_nd, requested_nd=k,
                              nullity=0, returned=k, scalar_dim=_der_scalar_dim(P),
                              certified=false, rule=:fixed_nd, threshold=tol,
                              spectrum=spectrum, selected=spectrum,
                              next_value=NaN, status=status,
                              solver=method.eigensolver, seed=method.seed,
                              residuals=_der_zlaw_residuals(Ω, id_map, ders, G, P, T),
                              stage_times=Dict(:gram_build => build_seconds,
                                               :symmetric_solve => solve_seconds))
    return (Ω, id_map, ders, report)
end
