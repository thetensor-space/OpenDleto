#
# Strata Dleto: QuickSylver
#   Fast double-restriction solve-and-lift for adjoint-type (nucleus) chisels.
#
# NOTICE:
#   This integration adapts the QuickSylver solve-and-lift strategy developed
#   in companion work by Chris Liu, Joshua Maglione, and James B. Wilson.
#   Please retain this attribution when reusing this implementation.
#
# Transcribed from ../fast-der-solver/quicksylver-lib.jl (with lin_solve from
# linear-algebra-lib.jl), operating on plain Arrays with ITensors only at the
# boundary -- the same discipline as FastDer3Valent.jl, and for the same
# reason.
#
# This is a DIFFERENT algorithm from quick-der / FastDer3Valent:
#
#   quick-der    X*R + S*Y - T*Z = 0   3 slots, triple restriction (a',b',c')
#   QuickSylver  X*R + S*Y     = T     2 slots + RHS, double restriction (a',b')
#
# In chisel terms QuickSylver handles a one-row chisel with exactly two engaged
# axes -- the adjoint chisels of null_patterns.pdf section 7.2, whose
# derivations are the nuclei.  With a zero right-hand side, `X*R + S*Y = 0`
# with R = c1*Γ and S = c2*Γ is precisely the [c1,c2,0]-derivation condition.
#
# The reference solves the *inhomogeneous* problem and so returns an affine
# frame: a particular solution plus offsets.  For the homogeneous derivation
# case the particular solution is 0 and the frame's offsets from its first
# point are a linear basis, which is what `der` wants.
#

using LinearAlgebra
using LinearMaps

"""
    QuickSylverMethod

Derivation method for one-row chisels with exactly two engaged axes, using the
double-restriction solve-and-lift strategy.  The engaged pair may be any two
axes; the tensor is permuted so that they come first, since the kernel fixes X
on axis 1, Y on axis 2 and slices along the third.
"""
struct QuickSylverMethod <: DerivationMethod
    double_restriction_size_override::Union{Nothing, NTuple{2, Int}}
    faster_randomized_check::Bool
end

QuickSylverMethod(; double_restriction_size_override=nothing, faster_randomized_check::Bool=false) =
    QuickSylverMethod(double_restriction_size_override, faster_randomized_check)

# ---------------------------------------------------------------------------
# linear-algebra-lib.jl : lin_solve with a right-hand side
# ---------------------------------------------------------------------------

"""
    _qs_lin_solve(M, rhs; atol) -> (particular, nullspace) or nothing

Transcription of `lin_solve` with a right-hand side.  `nothing` means the
system is inconsistent.
"""
function _qs_lin_solve(M, rhs; atol::Float64=1e-6)
    A = Matrix{Float64}(M)
    b = vec(Array{Float64}(rhs))
    size(A, 1) == length(b) ||
        throw(DimensionMismatch("Right-hand side has incompatible dimension."))

    F = svd(A; full=true)
    r = rank(F; atol=atol, rtol=atol)
    x = F \ b
    isapprox(A * x, b; atol=atol, rtol=atol) || return nothing
    N = Matrix(transpose(F.Vt[(r + 1):end, :]))
    return x, N
end

# ---------------------------------------------------------------------------
# quicksylver-lib.jl
# ---------------------------------------------------------------------------

_qs_rhs_vector(T) = vcat((vec(T[:, :, k]) for k in axes(T, 3))...)

"""Transcription of `sylvester_system_matrix`."""
function _qs_system_matrix(R, S)
    r_dim, b, c = size(R)
    a, s_dim, c_S = size(S)
    c_S == c || throw(DimensionMismatch("R and S must have the same number of slices."))

    I_a = Matrix(LinearAlgebra.I, a, a)
    I_b = Matrix(LinearAlgebra.I, b, b)
    blocks = [
        hcat(kron(transpose(R[:, :, k]), I_a), kron(I_b, S[:, :, k]))
        for k in axes(R, 3)
    ]
    return vcat(blocks...)
end

"""Transcription of `solve_dense_sylvester_system`; returns an affine frame."""
function _qs_solve_dense(R, S, T; atol::Float64=1e-6)
    r_dim, b, c = size(R)
    a, s_dim, c_S = size(S)
    a_T, b_T, c_T = size(T)
    ((a_T, b_T, c_T) == (a, b, c) && c_S == c) ||
        throw(DimensionMismatch("R, S, and T must have compatible Sylvester dimensions."))

    M = _qs_system_matrix(R, S)
    sol = _qs_lin_solve(M, _qs_rhs_vector(T); atol=atol)
    sol === nothing && return NTuple{2, Matrix{Float64}}[]
    particular, N = sol

    function unpack(v)
        stop = a * r_dim
        X = reshape(v[1:stop], (a, r_dim))
        Y = reshape(v[(stop + 1):end], (s_dim, b))
        return Matrix(X), Matrix(Y)
    end

    frame = NTuple{2, Matrix{Float64}}[unpack(particular)]
    for i in axes(N, 2)
        push!(frame, unpack(particular + N[:, i]))
    end
    return frame
end

"""Transcription of `check_sylvester_solution`."""
function _qs_check_solution(R, S, T, frame; faster_randomized_check::Bool=false, atol::Float64=1e-6)
    isempty(frame) && return true

    ok(X, Y, k) = isapprox(X * R[:, :, k] + S[:, :, k] * Y, T[:, :, k]; atol=atol, rtol=atol)

    if faster_randomized_check
        k = rand(axes(R, 3))
        X, Y = frame[rand(eachindex(frame))]
        return ok(X, Y, k)
    end
    return all(ok(X, Y, k) for (X, Y) in frame for k in axes(R, 3))
end

"""Transcription of `select_double_restriction_sizes`."""
function _qs_select_restriction_sizes(R, S, T)
    r_dim, b, c = size(R)
    a, s_dim, c_S = size(S)
    a_T, b_T, c_T = size(T)
    ((a_T, b_T, c_T) == (a, b, c) && c_S == c) ||
        throw(DimensionMismatch("R, S, and T must have compatible Sylvester dimensions."))

    rs_max = max(r_dim, s_dim)
    balanced_block_size = ceil(Int, 2.0 * rs_max^2 / c)

    a_prime = min(a, max(1, ceil(Int, balanced_block_size / r_dim) + 1))
    b_prime = min(b, max(1, ceil(Int, balanced_block_size / s_dim) + 1))

    num_equations = a_prime * b_prime * c
    num_unknowns = a_prime * r_dim + b_prime * s_dim
    if num_equations < num_unknowns
        error("The restriction sizes do not make DoubleRestrictedSylvester generically full column rank.")
    end
    return a_prime, b_prime
end

"""Transcription of `solve_and_lift_sylvester_system`."""
function _qs_solve_and_lift(R, S, T; a_prime::Int, b_prime::Int, atol::Float64=1e-6)
    r_dim, b, c = size(R)
    a, s_dim, _ = size(S)

    Ir = 1:a_prime;  I_hat = (a_prime + 1):a
    Jr = 1:b_prime;  J_hat = (b_prime + 1):b

    DR_frame = _qs_solve_dense(R[:, Jr, :], S[Ir, :, :], T[Ir, Jr, :]; atol=atol)
    isempty(DR_frame) && return NTuple{2, Matrix{Float64}}[]

    M_R = vcat([S[Ir, :, k] for k in 1:c]...)
    M_C = vcat([Matrix(transpose(R[:, Jr, k])) for k in 1:c]...)

    N_R(X_I) = vcat([T[Ir, J_hat, k] - X_I * R[:, J_hat, k] for k in 1:c]...)
    N_C(Y_J) = vcat([Matrix(transpose(T[I_hat, Jr, k] - S[I_hat, :, k] * Y_J)) for k in 1:c]...)

    dim = length(DR_frame) - 1
    X_I_0, Y_J_0 = DR_frame[1]
    row_rhs_0, col_rhs_0 = N_R(X_I_0), N_C(Y_J_0)
    row_dirs = [N_R(X_I) - row_rhs_0 for (X_I, _) in DR_frame[2:end]]
    col_dirs = [N_C(Y_J) - col_rhs_0 for (_, Y_J) in DR_frame[2:end]]

    Y_hat_0, Y_hat_dirs = _qd_linear_equals_affine(M_R, row_rhs_0, row_dirs; atol=atol)
    X_hat_0, X_hat_dirs = _qd_linear_equals_affine(M_C, col_rhs_0, col_dirs; atol=atol)

    frame = NTuple{2, Matrix{Float64}}[]
    push!(frame, (Matrix(vcat(X_I_0, transpose(X_hat_0))), Matrix(hcat(Y_J_0, Y_hat_0))))
    for i in 1:dim
        X_I_i, Y_J_i = DR_frame[i + 1]
        push!(frame, (Matrix(vcat(X_I_i, transpose(X_hat_0 + X_hat_dirs[i]))),
                      Matrix(hcat(Y_J_i, Y_hat_0 + Y_hat_dirs[i]))))
    end
    return frame
end

"""Transcription of `sylvester_solver`, including the verification step."""
function _qs_sylvester_solver(R, S, T;
                              double_restriction_size_override=nothing,
                              faster_randomized_check::Bool=false,
                              atol::Float64=1e-6)
    if double_restriction_size_override !== nothing
        a_prime, b_prime = double_restriction_size_override
        a, b, _ = size(T)
        (1 <= a_prime <= a && 1 <= b_prime <= b) ||
            error("The DoubleRestrictedSylvester override must satisfy 1 <= a' <= a and 1 <= b' <= b.")
    else
        a_prime, b_prime = _qs_select_restriction_sizes(R, S, T)
    end

    frame = _qs_solve_and_lift(R, S, T; a_prime=a_prime, b_prime=b_prime, atol=atol)
    isempty(frame) && return frame

    _qs_check_solution(R, S, T, frame; faster_randomized_check=faster_randomized_check, atol=atol) ||
        error("QuickSylver did not find a correct affine frame at " *
              "(a',b') = ($a_prime,$b_prime). Retry with larger sizes via " *
              "double_restriction_size_override.")

    return frame
end

# ---------------------------------------------------------------------------
# ITensor boundary
# ---------------------------------------------------------------------------

function _qs_validate(Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor)
    ndims(Γ) == 3 || error("QuickSylverMethod currently supports only valency-3 tensors.")
    Ω isa IndTransverseOps || error("QuickSylverMethod currently requires IndTransverseOps.")
    valency(Ω) == 3 || error("QuickSylverMethod currently requires valency-3 transverse operators.")
    all(op -> op isa UniversalOp, Ω.localOps) ||
        error("QuickSylverMethod currently requires UniversalOp() on every axis.")
    size(P, 2) == 3 || error("QuickSylverMethod currently requires a 3-column chisel.")
    size(P, 1) == 1 || error("QuickSylverMethod currently supports one-row chisels only.")

    eng = engaged(P)
    sum(eng) == 2 || error("QuickSylverMethod handles adjoint-type chisels: exactly two " *
                           "engaged axes, got $(sum(eng)). Use FastDer3Valent for three.")
    return findall(eng)
end

function derTrOpsReduced(
    method::QuickSylverMethod,
    Ω::TransverseOps,
    P::AbstractMatrix,
    Γ::ITensor;
    tol::Float64=1e-6,
    nd=-1,
    kwargs...,
)::Tuple{TransverseOps, LinearMaps.LinearMap, AbstractMatrix{<:Number}}
    engaged_axes = _qs_validate(Ω, P, Γ)
    fr = collect(inds(Γ))
    dims = [ITensors.dim(i) for i in fr]

    # The kernel fixes X on axis 1, Y on axis 2 and slices along axis 3, so
    # permute the engaged pair to the front.  `perm` maps kernel position ->
    # original axis, and is inverted when the operators are packed back.
    disengaged = only(setdiff(1:3, engaged_axes))
    perm = vcat(engaged_axes, disengaged)
    G = permutedims(Array(Γ, fr...), perm)

    coeffs = vec(P[1, :])
    R = coeffs[engaged_axes[1]] * G
    S = coeffs[engaged_axes[2]] * G
    RHS = zeros(Float64, size(G))

    frame = _qs_sylvester_solver(
        R, S, RHS;
        double_restriction_size_override=method.double_restriction_size_override,
        faster_randomized_check=method.faster_randomized_check,
        atol=tol,
    )

    # A homogeneous system: the frame's offsets from its first point span the
    # solution space, and the first point is itself a solution (the zero one),
    # so the differences are the linear basis `der` expects.
    cols = Vector{Vector{Float64}}()
    if length(frame) >= 2
        X0, Y0 = frame[1]
        for (X, Y) in frame[2:end]
            dX = X - X0
            dY = Y - Y0
            ops = Matrix{Float64}[zeros(dims[a], dims[a]) for a in 1:3]
            # X acts by left multiplication, so its second index meets the
            # tensor; embedITensors puts the contracted index first, hence the
            # transpose.  Y acts on the right and needs none.  The disengaged
            # axis carries the zero operator (Remark 5.3).
            ops[engaged_axes[1]] = Matrix(transpose(dX))
            ops[engaged_axes[2]] = Matrix(dY)
            v = Float64[]
            for a in 1:3
                append!(v, vec(ops[a]))
            end
            push!(cols, v)
        end
    end

    ders = isempty(cols) ? zeros(Float64, globalDim(Ω), 0) : hcat(cols...)
    if nd > 0 && size(ders, 2) > nd
        ders = ders[:, 1:floor(Int, nd)]
    end

    id_map = LinearMaps.LinearMap(identity, identity, globalDim(Ω), globalDim(Ω); ismutating=false)
    return (Ω, id_map, ders)
end
