# Direct normal matrix for a real, dense, three-way tensor with SymmetricOp()
# on every axis and the one-row universal chisel.  This is deliberately a
# benchmark/prototype rather than production solver code: it trades the full
# derivation map (d^3 by 3d(d+1)/2) for its 3d(d+1)/2 square normal matrix.

using Dleto
using ITensors
using LinearAlgebra
using Random
using SparseArrays

"""
    sphere_symmetric_gram(Γ::ITensor; nd=3, eigensolver=:inverse, seed=20260914)

Build the normal matrix of the universal-chisel derivation map restricted to
real symmetric matrices, in *orthonormal Frobenius coordinates*.  The columns
are diagonal matrix units and `(Eᵢⱼ + Eⱼᵢ)/sqrt(2)`, ordered as
`SymmetricOp` (columns, then rows through the diagonal).

The returned `ders` are converted back to Dleto's `SymmetricOp` coordinates:
their off-diagonal entries are divided by `sqrt(2)`.  Thus `ders` can be used
with `embedITensors(SymmetricOps(Γ), ders[:, j])`; `normal` and its
`eigenvectors` use the orthonormal coordinates instead.

`eigensolver = :inverse` (the default) uses a shifted Cholesky inverse
subspace iteration, then measures the Rayleigh--Ritz vectors against the
*unsquared* rectangular derivation map.  This avoids the expensive full
eigendecomposition of the d=50 normal matrix.  `eigensolver = :eigen` retains
the dense `eigen(Symmetric(H), 1:nd)` oracle for small validation cases.

This prototype is for dense real cubic tensors and `UniversalChisel(3)` only.
It returns a named tuple containing the requested smallest eigenpairs, the
orthonormal normal matrix, and build/eigensolve timings.
"""
function sphere_symmetric_gram(Γ::ITensor; nd::Integer=3,
                               eigensolver::Symbol=:inverse,
                               seed::Integer=20260914)
    nd > 0 || throw(ArgumentError("nd must be positive, got $nd"))
    eigensolver in (:inverse, :eigen) || throw(ArgumentError(
        "eigensolver must be :inverse or :eigen, got :$eigensolver"))
    ndims(Γ) == 3 || throw(ArgumentError("requires a three-way ITensor"))
    fr = collect(inds(Γ))
    dims = [ITensors.dim(i) for i in fr]
    length(unique(dims)) == 1 || throw(ArgumentError(
        "requires equal axis dimensions, got $(Tuple(dims))"))
    T = eltype(Γ)
    T <: AbstractFloat || throw(ArgumentError("requires a real floating tensor, got $T"))
    G = Array(Γ, fr...)
    d = only(unique(dims))
    l = d * (d + 1) ÷ 2
    n = 3l
    invsqrt2 = inv(sqrt(T(2)))

    # B maps orthonormal Frobenius symmetric coordinates to vec(M), in the
    # same upper-triangle-by-columns order SymmetricOp uses.
    function symmetric_basis()
        rows = Int[]
        cols = Int[]
        vals = T[]
        col = 0
        for j in 1:d, i in 1:j
            col += 1
            push!(rows, i + (j - 1) * d); push!(cols, col)
            push!(vals, i == j ? one(T) : invsqrt2)
            if i != j
                push!(rows, j + (i - 1) * d); push!(cols, col)
                push!(vals, invsqrt2)
            end
        end
        return sparse(rows, cols, vals, d * d, l)
    end

    t0 = time_ns()
    B = symmetric_basis()
    H = zeros(T, n, n)
    blocks = zeros(Float64, 3, 3)

    # The a-th mode covariance gives the a=a normal block.  `A` is the
    # d-by-d^2 mode unfolding, hence vec(M) has Gram kron(I_d, A*A').
    for a in 1:3
        rest = [q for q in 1:3 if q != a]
        A = reshape(permutedims(G, (a, rest...)), d, :)
        C = A * transpose(A)
        ta = time_ns()
        H[(a - 1) * l + 1:a * l, (a - 1) * l + 1:a * l] .=
            Matrix(transpose(B) * kron(Matrix{T}(I, d, d), C) * B)
        blocks[a, a] += (time_ns() - ta) / 1e9
    end

    # For a != b, contract the remaining mode.  The reshape/permutation makes
    # H[(p,q),(r,s)] = sum_k G[p,s,k] * G[q,r,k] in the (a,b,other) ordering.
    for a in 1:2, b in (a + 1):3
        other = only(setdiff(1:3, (a, b)))
        ta = time_ns()
        U = reshape(permutedims(G, (a, b, other)), d * d, d)
        K = U * transpose(U)
        Hab = reshape(permutedims(reshape(K, d, d, d, d), (1, 3, 4, 2)), d * d, d * d)
        cab = Matrix(transpose(B) * Hab * B)
        ra = (a - 1) * l + 1:a * l
        rb = (b - 1) * l + 1:b * l
        H[ra, rb] .= cab
        H[rb, ra] .= transpose(cab)
        blocks[a, b] = blocks[b, a] = (time_ns() - ta) / 1e9
    end
    build_seconds = (time_ns() - t0) / 1e9

    # Dleto's symmetric coordinates put the same raw value in each off-
    # diagonal entry, whereas B uses 1/sqrt(2) in each; q -> x is therefore
    # q/sqrt(2) off diagonal and identity on the diagonal.
    scale = ones(T, l)
    p = 0
    for j in 1:d, i in 1:j
        p += 1
        i == j || (scale[p] = invsqrt2)
    end

    k = min(Int(nd), n)
    te = time_ns()
    eigenvalues = Vector{T}(undef, k)
    eigenvectors = Matrix{T}(undef, n, k)
    cholesky_seconds = 0.0
    iteration_seconds = 0.0
    ritz_seconds = 0.0
    shift_used = zero(T)
    if eigensolver === :eigen
        F = eigen(Symmetric(H), 1:k)
        eigenvalues .= F.values
        eigenvectors .= F.vectors
    else
        # H is positive semidefinite.  Its exact scalar derivations make a
        # shift necessary; the shift is deliberately tiny relative to the
        # assembled normal matrix and is escalated only if Cholesky says so.
        tc = time_ns()
        maxdiag = max(maximum(abs, diag(H)), eps(T))
        shift = T(1e-10) * maxdiag
        C = nothing
        for attempt in 1:10
            F = cholesky(Symmetric(H + shift * I); check=false)
            if issuccess(F)
                C = F
                break
            end
            shift *= T(10)
        end
        C === nothing && error("shifted Cholesky failed after 10 attempts")
        shift_used = shift
        cholesky_seconds = (time_ns() - tc) / 1e9

        # Oversampling is essential when several exact scalars and nearby
        # directions share the bottom of the spectrum.  A local RNG makes the
        # probe reproducible without touching the caller's global RNG.
        kp = min(n, k + 24)
        rng = MersenneTwister(seed)
        ti = time_ns()
        Q = Matrix(qr(randn(rng, T, n, kp)).Q)[:, 1:kp]
        for _ in 1:8
            Q = Matrix(qr(C \ Q).Q)[:, 1:kp]
        end
        iteration_seconds = (time_ns() - ti) / 1e9

        # Refine on the UNSQUARED derivation map in Dleto's raw SymmetricOp
        # coordinates.  This is both a better numerical measurement and the
        # direct check that the orthonormal-coordinate scaling is applied in
        # the correct direction.
        tr = time_ns()
        rawQ = copy(Q)
        for a in 1:3
            rawQ[(a - 1) * l + 1:a * l, :] .*= scale
        end
        Ω = SymmetricOps(Γ)
        _, E = sylvesterLM(Ω, UniversalChisel(3), Γ)
        Y = hcat((E * view(rawQ, :, j) for j in 1:kp)...)
        Fy = svd(Y)
        order = sortperm(Fy.S)[1:k]
        eigenvalues .= Fy.S[order] .^ 2
        eigenvectors .= Q * Fy.V[:, order]
        ritz_seconds = (time_ns() - tr) / 1e9
    end
    eig_seconds = (time_ns() - te) / 1e9

    ders = copy(eigenvectors)
    for a in 1:3
        ders[(a - 1) * l + 1:a * l, :] .*= scale
    end

    hnorm = max(opnorm(H, Inf), eps(T))
    spectral_residuals = [norm(H * view(eigenvectors, :, j) -
                               eigenvalues[j] * view(eigenvectors, :, j)) / hnorm
                          for j in 1:k]
    return (; ders, eigenvalues, eigenvectors, normal=H, spectral_residuals,
            timings=(; build_seconds, eig_seconds, cholesky_seconds,
                     iteration_seconds, ritz_seconds, shift=shift_used,
                     block_seconds=blocks))
end
