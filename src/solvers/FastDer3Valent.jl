#
# Strata Dleto: FastDer3Valent
#   Fast solve-and-lift derivation strategy for 3-valent tensors.
#
# NOTICE:
#   This integration adapts the fast-der solve-and-lift strategy developed in
#   companion work by Chris Liu, Joshua Maglione, and James B. Wilson.
#   Please retain this attribution when reusing this implementation.
#
# The numerical core below is a faithful transcription of the reference
# implementation in ../fast-der-solver/quick-der-lib.jl (with lin_solve and
# linear_equals_affine from linear-algebra-lib.jl), operating on plain Arrays.
# ITensors appear only at the boundary.
#
# It is a transcription on purpose.  The previous port re-derived the linear
# system in ITensor contractions and got all three blocks wrong: the reference
# builds `kron(I_a, R_flat)` with rows striding c then b, a *vertical stack*
# over i of `kron(I_b, Sᵢ)`, and `kron(T_flat, I_c)` with the identity on the
# right; the port used a single `kron(I, ·)` for each, and kron does not
# commute.  The system therefore had no nullspace where the reference's has
# one, which the Z-law in test/TestDerivationLaws.jl caught.  Reimplementing
# verified numerics in a different data structure is what introduced that, so
# the reference is followed line for line here.
#

using LinearAlgebra
using LinearMaps

"""
    FastDer3ValentMethod

Derivation method using the solve-and-lift strategy from fast-der-solver.
Intentionally limited to 3-valent tensors with:
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

# ---------------------------------------------------------------------------
# linear-algebra-lib.jl
# ---------------------------------------------------------------------------

_qd_nullspace(M; atol::Float64=1e-6) = nullspace(Matrix{Float64}(M); atol=atol, rtol=atol)

"""
    _qd_linear_equals_affine(M, N_0, N_directions; atol) -> (U_0, U_directions)

Transcription of `linear_equals_affine`.  Assumes `M` has full column rank and
that every parameter is feasible, both of which the reference checks.
"""
function _qd_linear_equals_affine(M, N_0, N_directions; atol::Float64=1e-6)
    M_matrix = Matrix{Float64}(M)
    N_0_matrix = Matrix{Float64}(N_0)

    if rank(M_matrix; atol=atol, rtol=atol) < size(M_matrix, 2)
        error("LinearEqualsAffine assumes the linear system matrix has full column rank.")
    end

    rhs_rows, rhs_cols = size(M_matrix, 1), size(N_0_matrix, 2)
    size(N_0_matrix, 1) == rhs_rows ||
        throw(DimensionMismatch("N_0 has incompatible row dimension."))

    N_direction_matrices = [Matrix{Float64}(N_i) for N_i in N_directions]
    for N_i in N_direction_matrices
        size(N_i) == (rhs_rows, rhs_cols) ||
            throw(DimensionMismatch("All direction matrices must have the same size as N_0."))
    end

    left_kernel = _qd_nullspace(M_matrix'; atol=atol)
    blocks = Matrix{Float64}[left_kernel' * N_0_matrix]
    append!(blocks, [left_kernel' * N_i for N_i in N_direction_matrices])
    if any(!isapprox(B, zeros(size(B)); atol=atol, rtol=atol) for B in blocks)
        error("LinearEqualsAffine assumes every parameter is feasible, but Theta_feas != Theta.")
    end

    left_inverse = pinv(M_matrix)
    return left_inverse * N_0_matrix, [left_inverse * N_i for N_i in N_direction_matrices]
end

# ---------------------------------------------------------------------------
# quick-der-lib.jl
# ---------------------------------------------------------------------------

"""
    _qd_derivation_system_matrix(R, S, T)

Transcription of `derivation_system_matrix`.  The row order is (a, b, c) with
rows striding c fastest, then b, then a; the columns are the unknowns of X, Y
and Z in that order.  Every block below must agree on that row order, which is
why the kron factors sit where they do.
"""
function _qd_derivation_system_matrix(R, S, T)
    r_dim, b, c = size(R)
    a, s_dim, _ = size(S)
    _, _, t_dim = size(T)

    # rows first stride by c then b
    R_flat = reshape(permutedims(R, (3, 2, 1)), (c * b, r_dim))
    I_a = Matrix(LinearAlgebra.I, a, a)
    # rows stride c, b, a; columns stride r_dim then a
    R_mat = kron(I_a, R_flat)

    I_b = Matrix(LinearAlgebra.I, b, b)
    S_slices = [transpose(S[i, :, :]) for i in 1:a]   # each (c x s_dim)
    # a vertical stack of per-slice krons -- NOT a single kron
    S_mat = vcat([kron(I_b, M) for M in S_slices]...)

    T_flat = reshape(permutedims(T, (2, 1, 3)), (b * a, t_dim))
    I_c = Matrix(LinearAlgebra.I, c, c)
    # identity on the right
    T_mat = kron(T_flat, I_c)

    return hcat(R_mat, S_mat, -T_mat)
end

"""Transcription of `solve_dense_derivation_system`."""
function _qd_solve_dense(R, S, T; atol::Float64=1e-6)
    _, b, c = size(R)
    a, _, c_S = size(S)
    a_T, b_T, _ = size(T)
    ((a_T, b_T) == (a, b) && c_S == c) ||
        throw(DimensionMismatch("R, S, and T must have compatible derivation dimensions."))

    M = _qd_derivation_system_matrix(R, S, T)
    return _qd_nullspace(M; atol=atol)
end

"""Transcription of `outer_action`: apply Z along the third axis."""
function _qd_outer_action(T, Z)
    a, b, c = size(T)
    return reshape(reshape(T, (a * b, c)) * Z, (a, b, size(Z, 2)))
end

"""
    _qd_check_solution(R, S, T, basis; faster_randomized_check) -> Bool

Transcription of `check_derivation_solution`.  The reference always runs this,
because solve-and-lift is only *generically* correct at a given (a',b',c').
The previous port dropped it, which turned a detectable failure into a silent
wrong answer.
"""
function _qd_check_solution(R, S, T, basis; faster_randomized_check::Bool=false, atol::Float64=1e-6)
    isempty(basis) && return true

    ok_on_slice(X, Y, Z, k) = begin
        TZ = _qd_outer_action(T, Z)
        isapprox(X * R[:, :, k] + S[:, :, k] * Y - TZ[:, :, k],
                 zeros(size(TZ[:, :, k])); rtol=atol, atol=atol)
    end

    if faster_randomized_check
        k = rand(axes(R, 3))
        X, Y, Z = basis[rand(eachindex(basis))]
        return ok_on_slice(X, Y, Z, k)
    end

    for (X, Y, Z) in basis
        TZ = _qd_outer_action(T, Z)
        for k in axes(R, 3)
            isapprox(X * R[:, :, k] + S[:, :, k] * Y - TZ[:, :, k],
                     zeros(size(TZ[:, :, k])); rtol=atol, atol=atol) || return false
        end
    end
    return true
end

"""Transcription of `select_restriction_sizes`."""
function _qd_select_restriction_sizes(R, S, T)
    r_dim, b, c = size(R)
    a, s_dim, _ = size(S)
    _, _, t_dim = size(T)
    rst_max = max(r_dim, s_dim, t_dim)

    balanced_block_size = ceil(Int, sqrt(3.0 * rst_max^3))

    a_prime = min(a, ceil(Int, balanced_block_size / r_dim) + 1)
    b_prime = min(b, ceil(Int, balanced_block_size / s_dim) + 1)
    c_prime = min(c, ceil(Int, balanced_block_size / t_dim) + 1)

    num_equations = a_prime * b_prime * c_prime
    num_unknowns = a_prime * r_dim + b_prime * s_dim + c_prime * t_dim
    if num_equations < num_unknowns
        error("The restriction sizes do not make TripleRestrictedDer generically full column rank.")
    end

    return a_prime, b_prime, c_prime
end

"""Transcription of `solve_and_lift_derivation_system`."""
function _qd_solve_and_lift(R, S, T; a_prime::Int, b_prime::Int, c_prime::Int, atol::Float64=1e-6)
    r_dim, b, c = size(R)
    a, s_dim, _ = size(S)
    _, _, t_dim = size(T)

    Ir = 1:a_prime;  I_hat = (a_prime + 1):a;  I_hat_dim = a - a_prime
    Jr = 1:b_prime;  J_hat = (b_prime + 1):b;  J_hat_dim = b - b_prime
    Kr = 1:c_prime;  K_hat = (c_prime + 1):c;  K_hat_dim = c - c_prime

    R_JK = R[:, Jr, Kr];  S_IK = S[Ir, :, Kr];  T_IJ = T[Ir, Jr, :]

    DerTR_basis = _qd_solve_dense(R_JK, S_IK, T_IJ; atol=atol)
    n_basis = size(DerTR_basis, 2)
    n_basis == 0 && return NTuple{3, Matrix{Float64}}[]

    M_C = Matrix(transpose(reshape(R[:, Jr, Kr], (r_dim, b_prime * c_prime))))
    T_I_hat_J = T[I_hat, Jr, :];  S_I_hat_K = S[I_hat, :, Kr]

    M_R = reshape(permutedims(S[Ir, :, Kr], (1, 3, 2)), (a_prime * c_prime, s_dim))
    T_I_J_hat = T[Ir, J_hat, :];  R_J_hat_K = R[:, J_hat, Kr]

    M_D = reshape(T[Ir, Jr, :], (a_prime * b_prime, t_dim))
    R_J_K_hat = R[:, Jr, K_hat];  S_I_K_hat = S[Ir, :, K_hat]

    function unpack(v)
        x_stop = a_prime * r_dim
        y_stop = x_stop + b_prime * s_dim
        X_I = permutedims(reshape(v[1:x_stop], (r_dim, a_prime)), (2, 1))
        Y_J = reshape(v[(x_stop + 1):y_stop], (s_dim, b_prime))
        Z_K = permutedims(reshape(v[(y_stop + 1):end], (c_prime, t_dim)), (2, 1))
        return X_I, Y_J, Z_K
    end

    function N_C(Y_J, Z_K)
        S_Y = Matrix(transpose(hcat([S_I_hat_K[:, :, k] * Y_J for k in 1:c_prime]...)))
        TZ_matrix = reshape(T_I_hat_J, (I_hat_dim * b_prime, t_dim)) * Z_K
        TZ = reshape(permutedims(reshape(TZ_matrix, (I_hat_dim, b_prime, c_prime)), (2, 3, 1)),
                     (b_prime * c_prime, I_hat_dim))
        return TZ - S_Y
    end

    function N_R(X_I, Z_K)
        X_R = vcat([X_I * R_J_hat_K[:, :, k] for k in 1:c_prime]...)
        TZ_matrix = reshape(T_I_J_hat, (a_prime * J_hat_dim, t_dim)) * Z_K
        TZ = reshape(permutedims(reshape(TZ_matrix, (a_prime, J_hat_dim, c_prime)), (1, 3, 2)),
                     (a_prime * c_prime, J_hat_dim))
        return TZ - X_R
    end

    function N_D(X_I, Y_J)
        X_R = vcat([X_I * R_J_K_hat[:, j, :] for j in 1:b_prime]...)
        S_Y_slices = [transpose(Y_J) * S_I_K_hat[i, :, :] for i in 1:a_prime]
        S_Y = reshape(permutedims(reshape(vcat(S_Y_slices...), b_prime, a_prime, K_hat_dim), (2, 1, 3)),
                      (a_prime * b_prime, K_hat_dim))
        return X_R + S_Y
    end

    vectors = [unpack(DerTR_basis[:, i]) for i in 1:n_basis]

    col_rhs = [N_C(Y_J, Z_K) for (_, Y_J, Z_K) in vectors]
    row_rhs = [N_R(X_I, Z_K) for (X_I, _, Z_K) in vectors]
    dep_rhs = [N_D(X_I, Y_J) for (X_I, Y_J, _) in vectors]

    _, X_hat = _qd_linear_equals_affine(M_C, zeros(b_prime * c_prime, I_hat_dim), col_rhs; atol=atol)
    _, Y_hat = _qd_linear_equals_affine(M_R, zeros(a_prime * c_prime, J_hat_dim), row_rhs; atol=atol)
    _, Z_hat = _qd_linear_equals_affine(M_D, zeros(a_prime * b_prime, K_hat_dim), dep_rhs; atol=atol)

    basis = NTuple{3, Matrix{Float64}}[]
    for (i, (X_I, Y_J, Z_K)) in enumerate(vectors)
        push!(basis, (Matrix(vcat(X_I, transpose(X_hat[i]))),
                      Matrix(hcat(Y_J, Y_hat[i])),
                      Matrix(hcat(Z_K, Z_hat[i]))))
    end
    return basis
end

"""Transcription of `derivation_solver`, including the verification step."""
function _qd_derivation_solver(R, S, T;
                               triple_restriction_size_override=nothing,
                               faster_randomized_check::Bool=false,
                               atol::Float64=1e-6)
    if triple_restriction_size_override !== nothing
        a_prime, b_prime, c_prime = triple_restriction_size_override
        a, b, c = size(T)
        (1 <= a_prime <= a && 1 <= b_prime <= b && 1 <= c_prime <= c) ||
            error("The TripleRestrictedDer override must satisfy 1 <= a' <= a, 1 <= b' <= b, and 1 <= c' <= c.")
    else
        a_prime, b_prime, c_prime = _qd_select_restriction_sizes(R, S, T)
    end

    basis = _qd_solve_and_lift(R, S, T; a_prime=a_prime, b_prime=b_prime, c_prime=c_prime, atol=atol)
    isempty(basis) && return basis

    _qd_check_solution(R, S, T, basis; faster_randomized_check=faster_randomized_check, atol=atol) ||
        error("FastDer3Valent did not find a correct solution triple at " *
              "(a',b',c') = ($a_prime,$b_prime,$c_prime). Retry with larger sizes via " *
              "triple_restriction_size_override.")

    return basis
end

# ---------------------------------------------------------------------------
# ITensor boundary
# ---------------------------------------------------------------------------

"""
    _encode_basis_vector(X, Y, Z) -> Vector{Float64}

Pack a solution triple into `Ω` coordinates.

`X` is transposed on the way in.  In the kernel `X` acts by left
multiplication, `(X·Γ)[i,j,k] = Σ_p X[i,p]·Γ[p,j,k]`, so it is `X`'s *second*
index that meets the tensor; `Y` and `Z` act on the right and so meet the
tensor with their *first*.  `embedITensors` puts the frame index first, i.e.
the contracted one first, which is the convention `applyDerivation` reads --
so only `X` needs flipping to agree.
"""
function _encode_basis_vector(X::AbstractMatrix, Y::AbstractMatrix, Z::AbstractMatrix)
    v = Float64[]
    append!(v, vec(Matrix(transpose(X))))
    append!(v, vec(Matrix(Y)))
    append!(v, vec(Matrix(Z)))
    return v
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

    fr = collect(inds(Γ))
    G = Array(Γ, fr...)

    # The kernel solves  X*R + S*Y - T*Z = 0, so the third slot enters negated:
    # the derivation condition c1*XΓ + c2*ΓY + c3*ΓZ = 0 needs T = -c3*Γ.
    coeffs = vec(P[1, :])
    R = coeffs[1] * G
    S = coeffs[2] * G
    T = -coeffs[3] * G

    basis = _qd_derivation_solver(
        R, S, T;
        triple_restriction_size_override=method.triple_restriction_size_override,
        faster_randomized_check=method.faster_randomized_check,
        atol=tol,
    )

    cols = [_encode_basis_vector(X, Y, Z) for (X, Y, Z) in basis]
    ders = isempty(cols) ? zeros(Float64, globalDim(Ω), 0) : hcat(cols...)
    if nd > 0 && size(ders, 2) > nd
        ders = ders[:, 1:floor(Int, nd)]
    end

    id_map = LinearMaps.LinearMap(identity, identity, globalDim(Ω), globalDim(Ω); ismutating=false)
    return (Ω, id_map, ders)
end
