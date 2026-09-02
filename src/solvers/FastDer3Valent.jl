#
# Strata Dleto: FastDer3Valent
#   Fast solve-and-lift derivation strategy for 3-valent tensors.
#
# NOTICE:
#   This integration adapts the fast-der solve-and-lift strategy developed in
#   companion work by Chris Liu, Joshua Maglione, and James B. Wilson.
#   Please retain this attribution when reusing this implementation.

using LinearAlgebra
using LinearMaps

"""
    FastDer3ValentMethod

Derivation method using the solve-and-lift strategy from fast-der-solver.
Current implementation is intentionally limited to 3-valent tensors with:
- `IndTransverseOps`
- `UniversalOp()` on each axis
- a single-row, fully engaged chisel
"""
struct FastDer3ValentMethod <: DerivationMethod
    triple_restriction_size_override::Union{Nothing, NTuple{3, Int}}
    solver::Symbol
    faster_randomized_check::Bool
end

FastDer3ValentMethod(; triple_restriction_size_override=nothing, solver::Symbol=:SVDSolver, faster_randomized_check::Bool=false) =
    FastDer3ValentMethod(triple_restriction_size_override, solver, faster_randomized_check)

function _selector_tensor(i::Index, kept::Vector{Int}; tag::String="rstr")
    j = Index(length(kept), "$(tag),$(i)")
    E = ITensor(i, j)
    for (jj, ii) in enumerate(kept)
        E[i => ii, j => jj] = 1.0
    end
    return E, j
end

function _restrict_index(Γ::ITensor, i::Index, kept::Vector{Int}; tag::String="rstr")
    E, j = _selector_tensor(i, kept; tag=tag)
    return Γ * E, j
end

function _matrix_from_tensor3(Γ::ITensor, r::Index, c1::Index, c2::Index)
    A = Array(Γ, r, c1, c2)
    M = Matrix{Float64}(undef, dim(c1) * dim(c2), dim(r))
    row = 1
    for j in 1:dim(c1), k in 1:dim(c2)
        for i in 1:dim(r)
            M[row, i] = A[i, j, k]
        end
        row += 1
    end
    return M
end

function _rhs_tensor_to_matrix(Γ::ITensor, row1::Index, row2::Index, col::Index)
    A = Array(Γ, row1, row2, col)
    M = Matrix{Float64}(undef, dim(row1) * dim(row2), dim(col))
    row = 1
    for i in 1:dim(row1), j in 1:dim(row2)
        for c in 1:dim(col)
            M[row, c] = A[i, j, c]
        end
        row += 1
    end
    return M
end

function _encode_basis_vector(X::AbstractMatrix, Y::AbstractMatrix, Z::AbstractMatrix)
    v = Float64[]
    append!(v, vec(Matrix(X)))
    append!(v, vec(Matrix(Y)))
    append!(v, vec(Matrix(Z)))
    return v
end

function _fastder_dertr_basis_to_xyz(
    DerTR_basis_vector::AbstractVector,
    a_prime::Int,
    b_prime::Int,
    c_prime::Int,
    r_dim::Int,
    s_dim::Int,
    t_dim::Int,
)
    X_I = zeros(Float64, a_prime, r_dim)
    Y_J = zeros(Float64, s_dim, b_prime)
    Z_K = zeros(Float64, t_dim, c_prime)

    offset = 0
    for i in 1:a_prime, j in 1:r_dim
        offset += 1
        X_I[i, j] = DerTR_basis_vector[offset]
    end
    for i in 1:s_dim, j in 1:b_prime
        offset += 1
        Y_J[i, j] = DerTR_basis_vector[offset]
    end
    for i in 1:c_prime, j in 1:t_dim
        offset += 1
        Z_K[j, i] = DerTR_basis_vector[offset]
    end

    return X_I, Y_J, Z_K
end

function _fastder_select_restriction_sizes(R::ITensor, S::ITensor, T::ITensor)
    r_dim, b, c = (dim(i) for i in inds(R))
    a, s_dim, _ = (dim(i) for i in inds(S))
    _, _, t_dim = (dim(i) for i in inds(T))
    rst_max = max(r_dim, s_dim, t_dim)

    balanced_block_size = ceil(Int, sqrt(3.0 * rst_max^3))

    a_prime = min(a, ceil(Int, balanced_block_size / r_dim) + 1)
    b_prime = min(b, ceil(Int, balanced_block_size / s_dim) + 1)
    c_prime = min(c, ceil(Int, balanced_block_size / t_dim) + 1)

    num_equations = a_prime * b_prime * c_prime
    num_unknowns = a_prime * r_dim + b_prime * s_dim + c_prime * t_dim
    num_equations >= num_unknowns || error("Restriction sizes are underconstrained for solve-and-lift.")

    return a_prime, b_prime, c_prime
end

function _fastder_restricted_basis(M::AbstractMatrix; tol::Float64=1e-6, nv::Int=8, solver::Symbol=:SVDSolver)
    L = LinearMaps.LinearMap(M)
    # Black-box path first; if it does not return usable vectors, fallback to exact dense nullspace.
    res = solve(L, solver; nv=min(nv, min(size(M)...)))
    vals = collect(res.vals)
    vecs = isa(res.vecs, AbstractVector) ? res.vecs : [res.vecs[:, i] for i in 1:size(res.vecs, 2)]
    if !isempty(vals) && !isempty(vecs)
        order = sortperm(abs.(vals))
        vals = vals[order]
        vecs = vecs[order]
        keep = findall(abs.(vals) .< tol)
        if !isempty(keep)
            return hcat(vecs[keep]...)
        end
        # Fallback: No eigenvalues strictly below tolerance.
        # Use relative tolerance approach: find where spectrum has a significant gap.
        # Keep vectors corresponding to eigenvalues <= median or until we see a big jump.
        abs_vals = abs.(vals)
        if length(abs_vals) > 1
            # Look for the largest relative jump in consecutive eigenvalues
            max_jump_idx = 1
            max_jump_ratio = 1.0
            for i in 1:(length(abs_vals)-1)
                if abs_vals[i] > 0
                    ratio = abs_vals[i+1] / abs_vals[i]
                    if ratio > max_jump_ratio && ratio > 5.0  # At least 5x jump
                        max_jump_idx = i
                        max_jump_ratio = ratio
                    end
                end
            end
            # Keep vectors up to the jump, or at least the first 3 if no clear jump
            keep_count = max(1, min(3, max_jump_idx))
            return hcat(vecs[1:keep_count]...)
        else
            return hcat(vecs[1:1]...)
        end
    end
    # If solver returns nothing, try nullspace directly
    return nullspace(M; atol=tol, rtol=tol)
end

function _fastder_solve_and_lift(
    R::ITensor,
    S::ITensor,
    T::ITensor;
    a_prime::Int,
    b_prime::Int,
    c_prime::Int,
    tol::Float64=1e-6,
    solver::Symbol=:SVDSolver,
)
    r_idx, b_idx, c_idx = inds(R)
    a_idx, s_idx, cS_idx = inds(S)
    aT_idx, bT_idx, t_idx = inds(T)

    cS_idx == c_idx || error("Incompatible c-index between R and S.")
    aT_idx == a_idx || error("Incompatible a-index between S and T.")
    bT_idx == b_idx || error("Incompatible b-index between R and T.")

    r_dim = dim(r_idx)
    a = dim(a_idx)
    b = dim(b_idx)
    c = dim(c_idx)
    s_dim = dim(s_idx)
    t_dim = dim(t_idx)

    Ipos = collect(1:a_prime)
    I_hat_pos = collect(a_prime + 1:a)
    I_hat_dim = length(I_hat_pos)

    Jpos = collect(1:b_prime)
    J_hat_pos = collect(b_prime + 1:b)
    J_hat_dim = length(J_hat_pos)

    Kpos = collect(1:c_prime)
    K_hat_pos = collect(c_prime + 1:c)
    K_hat_dim = length(K_hat_pos)

    R_JK, j_idx = _restrict_index(R, b_idx, Jpos; tag="J")
    R_JK, k_idx = _restrict_index(R_JK, c_idx, Kpos; tag="K")

    S_IK, i_idx = _restrict_index(S, a_idx, Ipos; tag="I")
    S_IK, kS_idx = _restrict_index(S_IK, cS_idx, Kpos; tag="K")

    T_IJ, iT_idx = _restrict_index(T, aT_idx, Ipos; tag="I")
    T_IJ, jT_idx = _restrict_index(T_IJ, bT_idx, Jpos; tag="J")

    # Align restricted index identities for safe contractions.
    S_IK = replaceind(S_IK, kS_idx, k_idx)
    T_IJ = replaceind(T_IJ, iT_idx, i_idx)
    T_IJ = replaceind(T_IJ, jT_idx, j_idx)

    M_R = _matrix_from_tensor3(R_JK, r_idx, j_idx, k_idx)
    M_S = _matrix_from_tensor3(S_IK, s_idx, i_idx, k_idx)
    M_T = _matrix_from_tensor3(T_IJ, t_idx, i_idx, j_idx)
    M = hcat(
        kron(Matrix(LinearAlgebra.I, dim(i_idx), dim(i_idx)), M_R),
        kron(Matrix(LinearAlgebra.I, dim(j_idx), dim(j_idx)), M_S),
        -kron(Matrix(LinearAlgebra.I, dim(k_idx), dim(k_idx)), M_T),
    )

    DerTR_basis = _fastder_restricted_basis(M; tol=tol, nv=max(8, a_prime + b_prime + c_prime), solver=solver)

    DerTR_basis_size = size(DerTR_basis, 2)
    DerTR_basis_size == 0 && return NTuple{3, Matrix{Float64}}[]

    M_C = _matrix_from_tensor3(R_JK, r_idx, j_idx, k_idx)

    T_I_hat_J, ihatT_idx = _restrict_index(T, aT_idx, I_hat_pos; tag="Ihat")
    T_I_hat_J, jhatT_idx = _restrict_index(T_I_hat_J, bT_idx, Jpos; tag="J")
    T_I_hat_J = replaceind(T_I_hat_J, jhatT_idx, j_idx)

    S_I_hat_K, ihatS_idx = _restrict_index(S, a_idx, I_hat_pos; tag="Ihat")
    S_I_hat_K, khatS_idx = _restrict_index(S_I_hat_K, cS_idx, Kpos; tag="K")
    S_I_hat_K = replaceind(S_I_hat_K, ihatS_idx, ihatT_idx)
    S_I_hat_K = replaceind(S_I_hat_K, khatS_idx, k_idx)

    M_R_lift = _matrix_from_tensor3(S_IK, s_idx, i_idx, k_idx)

    T_I_J_hat, i2_idx = _restrict_index(T, aT_idx, Ipos; tag="I")
    T_I_J_hat, jhat_idx = _restrict_index(T_I_J_hat, bT_idx, J_hat_pos; tag="Jhat")
    T_I_J_hat = replaceind(T_I_J_hat, i2_idx, i_idx)

    R_J_hat_K, j2hat_idx = _restrict_index(R, b_idx, J_hat_pos; tag="Jhat")
    R_J_hat_K, k2_idx = _restrict_index(R_J_hat_K, c_idx, Kpos; tag="K")
    R_J_hat_K = replaceind(R_J_hat_K, j2hat_idx, jhat_idx)
    R_J_hat_K = replaceind(R_J_hat_K, k2_idx, k_idx)

    M_D = _matrix_from_tensor3(T_IJ, t_idx, i_idx, j_idx)

    R_J_K_hat, j3_idx = _restrict_index(R, b_idx, Jpos; tag="J")
    R_J_K_hat, khat_idx = _restrict_index(R_J_K_hat, c_idx, K_hat_pos; tag="Khat")
    R_J_K_hat = replaceind(R_J_K_hat, j3_idx, j_idx)

    S_I_K_hat, i3_idx = _restrict_index(S, a_idx, Ipos; tag="I")
    S_I_K_hat, khatS2_idx = _restrict_index(S_I_K_hat, cS_idx, K_hat_pos; tag="Khat")
    S_I_K_hat = replaceind(S_I_K_hat, i3_idx, i_idx)
    S_I_K_hat = replaceind(S_I_K_hat, khatS2_idx, khat_idx)

    function N_C(Y_J::AbstractMatrix, Z_K::AbstractMatrix)
        Yten = ITensor(Matrix(Y_J), s_idx, j_idx)
        Zten = ITensor(Matrix(Z_K), t_idx, k_idx)
        NC = (T_I_hat_J * Zten) - (S_I_hat_K * Yten)
        return _rhs_tensor_to_matrix(NC, j_idx, k_idx, ihatT_idx)
    end

    function N_R(X_I::AbstractMatrix, Z_K::AbstractMatrix)
        Xten = ITensor(Matrix(X_I), i_idx, r_idx)
        Zten = ITensor(Matrix(Z_K), t_idx, k_idx)
        NR = (T_I_J_hat * Zten) - (Xten * R_J_hat_K)
        return _rhs_tensor_to_matrix(NR, i_idx, k_idx, jhat_idx)
    end

    function N_D(X_I::AbstractMatrix, Y_J::AbstractMatrix)
        Xten = ITensor(Matrix(X_I), i_idx, r_idx)
        Yten = ITensor(Matrix(Y_J), s_idx, j_idx)
        ND = (Xten * R_J_K_hat) + (S_I_K_hat * Yten)
        return _rhs_tensor_to_matrix(ND, i_idx, j_idx, khat_idx)
    end

    function _linear_equals_affine(Msys::AbstractMatrix, N0::AbstractMatrix, Ndirections::Vector{<:AbstractMatrix})
        rank(Msys; atol=tol, rtol=tol) >= size(Msys, 2) || error("Solve-and-lift requires full-column-rank lift systems.")
        M_left_inverse = pinv(Msys)
        U0 = M_left_inverse * N0
        Udirections = [M_left_inverse * N for N in Ndirections]
        return U0, Udirections
    end

    der_vectors = [_fastder_dertr_basis_to_xyz(DerTR_basis[:, i], a_prime, b_prime, c_prime, r_dim, s_dim, t_dim) for i in 1:DerTR_basis_size]

    col_rhs_directions = [N_C(Y_J, Z_K) for (_, Y_J, Z_K) in der_vectors]
    row_rhs_directions = [N_R(X_I, Z_K) for (X_I, _, Z_K) in der_vectors]
    depth_rhs_directions = [N_D(X_I, Y_J) for (X_I, Y_J, _) in der_vectors]

    _, X_hat_directions = _linear_equals_affine(M_C, zeros(b_prime * c_prime, I_hat_dim), col_rhs_directions)
    _, Y_hat_directions = _linear_equals_affine(M_R_lift, zeros(a_prime * c_prime, J_hat_dim), row_rhs_directions)
    _, Z_hat_directions = _linear_equals_affine(M_D, zeros(a_prime * b_prime, K_hat_dim), depth_rhs_directions)

    solution_basis = NTuple{3, Matrix{Float64}}[]
    for (i, (X_I, Y_J, Z_K)) in enumerate(der_vectors)
        X = vcat(X_I, Transpose(X_hat_directions[i]))
        Y = hcat(Y_J, Y_hat_directions[i])
        Z = hcat(Z_K, Z_hat_directions[i])
        push!(solution_basis, (X, Y, Z))
    end

    return solution_basis
end

function _fastder_solve_basis(
    R::ITensor,
    S::ITensor,
    T::ITensor;
    triple_restriction_size_override::Union{Nothing, NTuple{3, Int}}=nothing,
    faster_randomized_check::Bool=false,
    tol::Float64=1e-6,
    solver::Symbol=:SVDSolver,
)
    r_dim, b, c = (dim(i) for i in inds(R))
    a, s_dim, cS = (dim(i) for i in inds(S))
    aT, bT, _ = (dim(i) for i in inds(T))
    (aT == a && bT == b && cS == c) || throw(DimensionMismatch("R, S, and T must have compatible dimensions."))

    if triple_restriction_size_override === nothing
        a_prime, b_prime, c_prime = _fastder_select_restriction_sizes(R, S, T)
    else
        a_prime, b_prime, c_prime = triple_restriction_size_override
    end

    solution_basis = _fastder_solve_and_lift(
        R,
        S,
        T;
        a_prime=a_prime,
        b_prime=b_prime,
        c_prime=c_prime,
        tol=tol,
        solver=solver,
    )

    return solution_basis
end

function _fastder_validate_compatibility(Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor)
    ndims(Γ) == 3 || error("FastDer3ValentMethod currently supports only valency-3 tensors.")
    Ω isa IndTransverseOps || error("FastDer3ValentMethod currently requires IndTransverseOps.")
    valency(Ω) == 3 || error("FastDer3ValentMethod currently requires valency-3 transverse operators.")
    all(op -> op isa UniversalOp, Ω.localOps) || error("FastDer3ValentMethod currently requires UniversalOp() on every axis.")

    size(P, 2) == 3 || error("FastDer3ValentMethod currently requires a 3-column chisel.")
    size(P, 1) == 1 || error("FastDer3ValentMethod currently supports one-row chisels only.")
    all(engaged(P)) || error("FastDer3ValentMethod requires all three axes to be engaged.")

    return nothing
end

function derTrOpsReduced(
    method::FastDer3ValentMethod,
    Ω::TransverseOps,
    P::AbstractMatrix,
    Γ::ITensor;
    tol::Float64=1e-6,
    nd=-1,
    kwargs...,
)::Tuple{TransverseOps, LinearMaps.LinearMap, AbstractMatrix{<:Number}}
    _fastder_validate_compatibility(Ω, P, Γ)

    # The solve-and-lift kernel solves  X*R + S*Y - T*Z = 0  (note the minus on
    # the third slot; see quick-der-lib.jl check_derivation_solution).  The
    # derivation condition we want is  c1*XΓ + c2*ΓY + c3*ΓZ = 0, so the third
    # coefficient has to be negated on the way in.  Without this, a chisel
    # [c1,c2,c3] was silently solved as [c1,c2,-c3] -- with the default
    # UniversalChisel(3) = [1,1,1] that meant computing the [1,1,-1]
    # derivations, a different Z-set.  Caught by the Z-law in
    # test/TestDerivationLaws.jl.
    coeffs = vec(P[1, :])
    R = coeffs[1] * Γ
    S = coeffs[2] * Γ
    T = -coeffs[3] * Γ

    basis = _fastder_solve_basis(
        R,
        S,
        T;
        triple_restriction_size_override=method.triple_restriction_size_override,
        faster_randomized_check=method.faster_randomized_check,
        solver=method.solver,
        tol=tol,
    )

    cols = Vector{Vector{Float64}}()
    for (X, Y, Z) in basis
        push!(cols, _encode_basis_vector(X, Y, Z))
    end

    ders = isempty(cols) ? zeros(Float64, globalDim(Ω), 0) : hcat(cols...)
    if nd > 0 && size(ders, 2) > nd
        ders = ders[:, 1:nd]
    end

    id_map = LinearMaps.LinearMap(identity, identity, globalDim(Ω), globalDim(Ω); ismutating=false)
    return (Ω, id_map, ders)
end
