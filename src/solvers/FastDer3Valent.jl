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
using Random

"""
    FastDer3ValentMethod

Derivation method using the solve-and-lift strategy from fast-der-solver.
Intentionally limited to 3-valent tensors with:
- `IndTransverseOps`
- a single-row, fully engaged chisel

The kernel itself always solves over *all* matrices (`UniversalOp()` on every
axis).  When `Ω` names a smaller operator space on some axis -- `SymmetricOp()`,
`DiagonalOp()`, ... -- the universal basis is intersected with `Ω` afterwards
(`_fastder_restrict_to_ops`), so the result has the same meaning as
`SylverLining` on that `Ω`: coordinates in `Ω`, of the derivations that lie in
`Ω`.  For the scrambled sphere in bench/SphereHarness.jl this is the difference
between the 13-dimensional universal derivation algebra (three diagonal ones
plus ten nilpotent lowering maps) and the 3-dimensional symmetric one that
stratification needs.

Tolerances.  `tol` is *relative* throughout and is floored at
`sqrt(eps(eltype(Γ)))`: the reference hard-coded `atol = 1e-6` as an absolute
tolerance, which is below Float32 roundoff for these system sizes and so made
every Float32 run fail the feasibility check.
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

"""
    _qd_tolerance(T, tol)

The working tolerance for element type `T`: the caller's `tol`, floored at
`sqrt(eps(T))` because none of the rank, feasibility or verification checks
below can certify anything finer than that.  For Float64 and the default
`tol = 1e-6` this is `1e-6`; for Float32 it is `3.4e-4`, for Float16 `3.1e-2`.

The rule now lives in `qd_tolerance` (src/solvers/Precision.jl), which floors
at `precision_floor(T)` as well -- inert at every type measured, since
`sqrt(eps)` dominates `100 eps` for `eps < 1e-4` -- so that one file states the
whole floating-point policy.  This name is kept because it is what the solvers
call.
"""
_qd_tolerance(::Type{T}, tol::Real) where {T<:Number} = qd_tolerance(T, tol)
_qd_tolerance(::Type, tol::Real) = Float64(tol)

# Rank decisions are relative to the largest singular value.  The reference
# passed `atol = rtol = 1e-6`; an absolute `atol` makes the answer depend on the
# scale of Γ, and the relative part alone is what actually decides anything at
# the scales it was run at.
_qd_nullspace(M; atol::Real=TOL_DEFAULT) = nullspace(Matrix(M); rtol=atol)

"""Relative zero test: `|B| <= atol * scale`, with `scale` the size of the data `B` came from."""
_qd_isrelzero(B, scale; atol::Real) = norm(B) <= atol * scale

"""
    _qd_linear_equals_affine(M, N_0, N_directions; atol, scale) -> (U_0, U_directions)

Transcription of `linear_equals_affine`.  Assumes `M` has full column rank and
that every parameter is feasible, both of which the reference checks.  The
feasibility test is `|K' N| <= atol * scale` with `K` the left kernel of `M`;
`scale` defaults to the largest right-hand side, and the caller passes the
size of the data the right-hand sides were built from when it knows it (a
right-hand side that is itself pure roundoff is feasible, not infeasible).
"""
function _qd_linear_equals_affine(M, N_0, N_directions; atol::Real=TOL_DEFAULT, scale::Union{Nothing, Real}=nothing)
    M_matrix = Matrix(M)
    N_0_matrix = Matrix(N_0)

    if rank(M_matrix; rtol=atol) < size(M_matrix, 2)
        error("LinearEqualsAffine assumes the linear system matrix has full column rank.")
    end

    rhs_rows, rhs_cols = size(M_matrix, 1), size(N_0_matrix, 2)
    size(N_0_matrix, 1) == rhs_rows ||
        throw(DimensionMismatch("N_0 has incompatible row dimension."))

    N_direction_matrices = [Matrix(N_i) for N_i in N_directions]
    for N_i in N_direction_matrices
        size(N_i) == (rhs_rows, rhs_cols) ||
            throw(DimensionMismatch("All direction matrices must have the same size as N_0."))
    end

    # Feasibility is relative: the left kernel is orthonormal, so
    # `left_kernel' * N` carries roundoff of order `eps * |data|`, and an
    # absolute test at 1e-6 fails in Float32.
    ref = scale === nothing ? max(norm(N_0_matrix), maximum(norm, N_direction_matrices; init=zero(real(eltype(M_matrix))))) : scale
    left_kernel = _qd_nullspace(M_matrix'; atol=atol)
    feasible(N) = _qd_isrelzero(left_kernel' * N, ref; atol=atol)
    if !(feasible(N_0_matrix) && all(feasible, N_direction_matrices))
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
function _qd_solve_dense(R, S, T; atol::Real=TOL_DEFAULT)
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
function _qd_check_solution(R, S, T, basis; faster_randomized_check::Bool=false, atol::Real=TOL_DEFAULT)
    isempty(basis) && return true

    # The residual is measured against the size of the triple and the tensor,
    # not against an absolute 1e-6 (a solution is a solution at any scale) and
    # not slice by slice (a slice where all three terms are roundoff would fail
    # a per-slice relative test; the diagonal tensor in the tests has those).
    triple_scale(X, Y, Z) = norm(X) * norm(R) + norm(S) * norm(Y) + norm(T) * norm(Z)
    ok_on_slice(X, Y, TZ, k, scale) =
        _qd_isrelzero(X * R[:, :, k] + S[:, :, k] * Y - TZ[:, :, k], scale; atol=atol)

    if faster_randomized_check
        k = rand(axes(R, 3))
        X, Y, Z = basis[rand(eachindex(basis))]
        return ok_on_slice(X, Y, _qd_outer_action(T, Z), k, triple_scale(X, Y, Z))
    end

    for (X, Y, Z) in basis
        TZ = _qd_outer_action(T, Z)
        scale = triple_scale(X, Y, Z)
        for k in axes(R, 3)
            ok_on_slice(X, Y, TZ, k, scale) || return false
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
function _qd_solve_and_lift(R, S, T; a_prime::Int, b_prime::Int, c_prime::Int, atol::Real=TOL_DEFAULT)
    r_dim, b, c = size(R)
    a, s_dim, _ = size(S)
    _, _, t_dim = size(T)

    Ir = 1:a_prime;  I_hat = (a_prime + 1):a;  I_hat_dim = a - a_prime
    Jr = 1:b_prime;  J_hat = (b_prime + 1):b;  J_hat_dim = b - b_prime
    Kr = 1:c_prime;  K_hat = (c_prime + 1):c;  K_hat_dim = c - c_prime

    R_JK = R[:, Jr, Kr];  S_IK = S[Ir, :, Kr];  T_IJ = T[Ir, Jr, :]

    DerTR_basis = _qd_solve_dense(R_JK, S_IK, T_IJ; atol=atol)
    n_basis = size(DerTR_basis, 2)
    Tnum = promote_type(eltype(R), eltype(S), eltype(T))
    n_basis == 0 && return NTuple{3, Matrix{Tnum}}[]

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

    # The restricted basis vectors are unit vectors, so every right-hand side
    # is bounded by the size of the tensor slices it was built from.
    scale = norm(R) + norm(S) + norm(T)
    _, X_hat = _qd_linear_equals_affine(M_C, zeros(Tnum, b_prime * c_prime, I_hat_dim), col_rhs; atol=atol, scale=scale)
    _, Y_hat = _qd_linear_equals_affine(M_R, zeros(Tnum, a_prime * c_prime, J_hat_dim), row_rhs; atol=atol, scale=scale)
    _, Z_hat = _qd_linear_equals_affine(M_D, zeros(Tnum, a_prime * b_prime, K_hat_dim), dep_rhs; atol=atol, scale=scale)

    basis = NTuple{3, Matrix{Tnum}}[]
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
                               atol::Real=TOL_DEFAULT)
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
    _fastder_triple_matrices(X, Y, Z) -> [D_1, D_2, D_3]

The solution triple as the three (frame x temporary) operator matrices that
`embedITensors` expects, one per axis.

`X` is transposed on the way in.  In the kernel `X` acts by left
multiplication, `(X·Γ)[i,j,k] = Σ_p X[i,p]·Γ[p,j,k]`, so it is `X`'s *second*
index that meets the tensor; `Y` and `Z` act on the right and so meet the
tensor with their *first*.  `embedITensors` puts the frame index first, i.e.
the contracted one first, which is the convention `applyDerivation` reads --
so only `X` needs flipping to agree.
"""
_fastder_triple_matrices(X::AbstractMatrix, Y::AbstractMatrix, Z::AbstractMatrix) =
    AbstractMatrix[Matrix(transpose(X)), Matrix(Y), Matrix(Z)]

"""
    _fastder_projector(op, n, T) -> (coords, embed) or nothing

Least-squares coordinates in the local operator space `op` on an `n`-dimensional
axis.  `embed(x)` is the operator with coordinates `x`; `coords(M)` is the
coordinate vector of the orthogonal projection of `M` onto the space, so
`embed(coords(M)) == M` exactly when `M` is a member.  Built from the two
primitives every `Operator` provides -- `unsafe_embed` (the map E from
coordinates to matrices) and `unsafe_transposeEmbed` (its transpose) -- via the
Gram matrix `G = EᵗE`, which is diagonal for every operator in
src/ops/OperatorImpls.jl and is factored densely otherwise.  Returns `nothing`
for `UniversalOp`, where the projection is the identity.
"""
function _fastder_projector(op::Operator, n::Integer, ::Type{T}) where {T}
    op isa UniversalOp && return nothing
    ld = localDim(op, n)
    embed(x) = Matrix{T}(unsafe_embed(op, n, x))
    if ld == 0
        return (coords = M -> zeros(T, 0), embed = embed)
    end
    e = zeros(T, ld)
    unit(j) = (fill!(e, zero(T)); e[j] = one(T); embed(e))
    # Every operator in src/ops/OperatorImpls.jl has a DIAGONAL Gram matrix
    # (its basis matrices are orthogonal), known in closed form for the
    # built-in ones.  Building it by embedding each basis element was 2.5 GB
    # of churn per axis for SymmetricOp at d = 100 (ld = 5050), which the GC
    # let sit until the process passed its memory budget.  An operator without
    # a closed form still gets the full Gram and a Cholesky factorisation.
    known = _fastder_gram_diag(op, n, T)
    Gf = if known !== nothing
        Diagonal(known)
    else
        G = zeros(T, ld, ld)
        for j in 1:ld
            G[:, j] = unsafe_transposeEmbed(op, unit(j))
        end
        isdiag(G) ? Diagonal(diag(G)) : cholesky(Symmetric(G))
    end
    coords(M) = Gf \ Vector{T}(unsafe_transposeEmbed(op, M))
    return (coords = coords, embed = embed)
end

"""
    _fastder_gram_diag(op, n, T) -> Vector or nothing

The diagonal of the Gram matrix `EᵗE` of the operator's coordinate basis, for
the operators whose basis is orthogonal and whose norms are known in closed
form: a coordinate shared by two matrix entries (the off-diagonal ones of a
symmetric or antisymmetric matrix) has norm² 2, every other coordinate 1.  The
order follows `unsafe_coordinates`: column by column, `M[1:i, i]` for
`SymmetricOp`, `M[1:i-1, i]` for `AntiSymmetricOp`.  Returns `nothing` for an
operator without a closed form, which then gets the general build.
"""
_fastder_gram_diag(::Operator, n::Integer, ::Type{T}) where {T} = nothing
_fastder_gram_diag(::UniversalOp, n::Integer, ::Type{T}) where {T} = ones(T, n * n)
_fastder_gram_diag(::DiagonalOp, n::Integer, ::Type{T}) where {T} = ones(T, n)
_fastder_gram_diag(::SymmetricOp, n::Integer, ::Type{T}) where {T} =
    T[i == j ? 1 : 2 for j in 1:n for i in 1:j]
_fastder_gram_diag(::AntiSymmetricOp, n::Integer, ::Type{T}) where {T} =
    fill(T(2), (n * (n - 1)) ÷ 2)

"""
    _fastder_restrict_to_ops(Ω, basis, atol) -> Matrix

Intersect the universal derivation space spanned by `basis` with the operator
space `Ω`, and return the result as columns of `Ω`-coordinates, i.e. exactly
what `derTrOpsReduced` on `Ω` returns.

`basis` is a vector of derivations, each given as one operator matrix PER AXIS
in the `embedITensors` convention (frame index first, i.e. the index that meets
the tensor).  Any valency: `QuickDerN` hands in `Vector{Matrix}` of length
`valency(Ω)` directly.  The valence-3 kernel's `(X, Y, Z)` triples are accepted
too, through the method below that runs them past `_fastder_triple_matrices`
first -- that kernel's `X` acts by LEFT multiplication and so needs a
transpose, which is exactly the transposing this function must not do for
anyone else.

A combination `Σ_i c_i D^{(i)}` lies in `Ω` iff on every axis `a` its component
is fixed by the projector onto `Ω`'s local space: `(I - Π_a) Σ_i c_i D_a^{(i)} = 0`.
Stacking those residuals over the axes gives a `(Σ_a n_a²) x k` matrix whose
null space is the coefficient space `c`; `k` is the universal nullity (13 for
the sphere), so this is small.  The basis is normalised first, so every column
of the residual matrix has norm at most 1 and `atol` is a meaningful ABSOLUTE
threshold on it -- a relative one would keep nothing when every basis element
already lies in `Ω` and the residual matrix is pure roundoff.

The cut on that residual spectrum is a GAP, not the bare `atol`; see
`_fastder_tall_nullspace`, which is where a Float32 undercount lived.
"""
function _fastder_restrict_to_ops(Ω::IndTransverseOps,
                                  basis::AbstractVector{<:AbstractVector{<:AbstractMatrix}},
                                  atol::Real)
    val = valency(Ω)
    dims = axisDims(Ω)
    k = length(basis)
    Tnum = k == 0 ? Float64 : promote_type(map(eltype, first(basis))...)
    k == 0 && return zeros(Tnum, globalDim(Ω), 0)
    all(m -> length(m) == val, basis) ||
        error("_fastder_restrict_to_ops: every basis element needs one matrix per " *
              "axis (valency $(val)).")

    # `collect` so the normalisation below cannot write through to the caller's
    # own matrices.
    mats = [collect(m) for m in basis]
    for m in mats
        s = sqrt(sum(M -> sum(abs2, M), m))
        s > 0 && for a in 1:val; m[a] = m[a] ./ s; end
    end

    projs = [_fastder_projector(Ω.localOps[a], dims[a], Tnum) for a in 1:val]
    if all(isnothing, projs)
        return hcat([Vector{Tnum}(unsafe_coordinates(Ω, m)) for m in mats]...)
    end

    # residual of each basis element off Ω, axis blocks stacked
    rows = cumsum([0; dims .^ 2])
    Res = zeros(Tnum, rows[end], k)
    for i in 1:k, a in 1:val
        projs[a] === nothing && continue
        Da = mats[i][a]
        Res[(rows[a] + 1):rows[a + 1], i] = vec(Da - projs[a].embed(projs[a].coords(Da)))
    end
    C = _fastder_tall_nullspace(Res, Tnum(atol))
    nc = size(C, 2)

    ders = zeros(Tnum, globalDim(Ω), nc)
    for j in 1:nc
        for a in 1:val
            Da = sum(C[i, j] * mats[i][a] for i in 1:k)
            ders[Ω.soffsets[a]:Ω.eoffsets[a], j] =
                projs[a] === nothing ? Vector{Tnum}(unsafe_coordinates(Ω.localOps[a], Da)) :
                                       projs[a].coords(Da)
        end
    end
    return ders
end

"""
    FASTDER_RESTRICT_CEILING :: Ref{Float64}

Multiplies `atol` to bound which cuts `_fastder_tall_nullspace`'s gap test may
consider at all, exactly as `QDN_LIFT_CEILING` does for the lift consistency
filter, and with the same value (32) for the same reason: it has to clear the
top of the genuine cluster and stay well under the first spurious direction.

Measured on the scrambled sphere valence 3 in Float32 (`atol` = 3.45e-4 there,
which is `sqrt(eps(Float32))`), residual spectrum of the restriction matrix,
genuine cluster | first spurious value:

| d | genuine (x `atol`) | first spurious | ceiling headroom |
|---|---|---|---|
|  48 | 1.30, 0.58, 0.34 | 0.244 (707 x `atol`) | 24x above the cluster, 22x below |
|  64 | 0.74, 0.44, 0.13 | 0.270 (784 x `atol`) | 43x / 24x |
| 100 | 0.14, 0.12, 0.06 | 0.201 (581 x `atol`) | 223x / 18x |
| 140 | 0.74, 0.10, 0.09 | 0.117 (338 x `atol`) | 43x / 11x |

so 32 sits in the middle of a two-decade window in every case.
"""
const FASTDER_RESTRICT_CEILING = Ref(32.0)

"""
    _fastder_tall_nullspace(A, atol) -> Matrix

Null space of a TALL matrix (m >= n) from its thin SVD.  `LinearAlgebra.nullspace`
computes `svd(A; full = true)`, whose `U` is m x m -- for the 30000 x 13
residual matrix of the sphere at d = 100 that is a 7 GB array for 13 columns
of answer, and it is what took the process past its memory budget.  For a
wide matrix the thin SVD has no room for the null space, so that case keeps
`nullspace`.

THE CUT IS A GAP, and this is where the Float32 undercount lived.  `A` is the
residual of a normalised derivation basis off `Ω`, so its singular values split
into two populations: the combinations that DO lie in `Ω` leave a residual at
the noise of the basis they were built from, and the ones that do not leave an
O(1) one.  Taking every value `<= atol` puts a hard cutoff at the noise level
itself, and in Float32 the noise level is where the answer is: the genuine
cluster runs from 0.06 to 1.30 times `atol` (`qd_tolerance(Float32)` =
`sqrt(eps(Float32))`, the accuracy of the lift that produced the basis), so at
d = 48 the third genuine direction sat at 1.30 `atol` and was DROPPED -- 13
lifted derivations in, 2 out of a true 3.  The gap between the populations is
338x to 4056x across d = 48..140, and every ratio inside the genuine cluster is
under 4x, so the two are never in doubt; only their absolute position is.

So: floor at `atol` (everything at or under the basis's own noise is zero),
bound the eligible cuts at `FASTDER_RESTRICT_CEILING * atol`, and take the
largest consecutive ratio that clears `gap_ratio` -- the same rule
`gap_verdict` applies to a null solver's spectrum and the lift filter applies
to its residual spectrum.  `atol` is `qd_tolerance(T)`, so the floor follows
the element type without a constant of its own: 1e-6 in Float64, where the null
cluster sits 9 decades below it and nothing about the old behaviour moves.

WHEN NO GAP CLEARS, the old absolute count is used unchanged.  A gap can only
REFINE the answer here, never invent one: with no jump of `gap_ratio` anywhere
in the spectrum there is no evidence for a split, and the historical count
below `atol` is what every existing caller was calibrated on -- including the
case where EVERY direction lies in `Ω` (the raw sphere's nullity 13, whose
residual matrix is pure roundoff and has no gap in it at all).
"""
function _fastder_tall_nullspace(A::AbstractMatrix, atol::Real;
                                 gap_ratio::Real = GAP_RATIO,
                                 ceiling::Real = FASTDER_RESTRICT_CEILING[] * atol)
    m, n = size(A)
    m < n && return nullspace(A; atol = atol, rtol = zero(atol))
    n == 0 && return zeros(eltype(A), 0, 0)
    F = svd(A)
    below = count(<=(atol), F.S)
    # `gap_verdict` reads an ASCENDING spectrum and `F.S` is descending, so the
    # null end is the tail of `F.S` and the head of `asc`.  `scale = 1.0`: these
    # are already absolute numbers on a normalised basis (see
    # `_fastder_restrict_to_ops`), not ratios to an operator norm.
    asc = reverse(Float64.(F.S))
    (_, v) = gap_verdict(asc, 1.0; threshold = Float64(ceiling),
                         floor = Float64(atol), gap_ratio = gap_ratio)
    keep = v.rule === :gap ? v.nullity : below
    @debug "restrict_to_ops residual spectrum" atol ceiling below keep rule = v.rule gap = v.gap svals = string(round.(asc; sigdigits = 3))
    return F.V[:, (n - keep + 1):n]
end

"""
The valence-3 kernel's own form: solution triples `(X, Y, Z)` in which `X` acts
by left multiplication.  `_fastder_triple_matrices` puts all three into the
one-matrix-per-axis convention above.
"""
_fastder_restrict_to_ops(Ω::IndTransverseOps,
                         basis::AbstractVector{<:NTuple{3, AbstractMatrix}},
                         atol::Real) =
    _fastder_restrict_to_ops(Ω, [_fastder_triple_matrices(X, Y, Z) for (X, Y, Z) in basis],
                             atol)

function _fastder_validate_compatibility(Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor)
    ndims(Γ) == 3 || error("FastDer3ValentMethod currently supports only valency-3 tensors.")
    Ω isa IndTransverseOps || error("FastDer3ValentMethod currently requires IndTransverseOps.")
    valency(Ω) == 3 || error("FastDer3ValentMethod currently requires valency-3 transverse operators.")
    all(i -> hasind(Γ, i), frames(Ω)) || error("FastDer3ValentMethod: Γ does not carry the frame of Ω.")

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
    tol::Real=TOL_DEFAULT,
    nd=-1,
    kwargs...,
)::Tuple{TransverseOps, LinearMaps.LinearMap, AbstractMatrix{<:Number}}
    _fastder_validate_compatibility(Ω, P, Γ)

    # Read Γ in Ω's frame order, which is the order the coordinates use.
    fr = frames(Ω)
    G = Array(Γ, fr...)

    # The kernel solves  X*R + S*Y - T*Z = 0, so the third slot enters negated:
    # the derivation condition c1*XΓ + c2*ΓY + c3*ΓZ = 0 needs T = -c3*Γ.
    Tnum = eltype(G)
    atol = _qd_tolerance(Tnum, tol)
    coeffs = Tnum.(vec(P[1, :]))
    R = coeffs[1] * G
    S = coeffs[2] * G
    T = -coeffs[3] * G

    basis = _qd_derivation_solver(
        R, S, T;
        triple_restriction_size_override=method.triple_restriction_size_override,
        faster_randomized_check=method.faster_randomized_check,
        atol=atol,
    )

    # Universal derivations, cut down to the ones that live in Ω.
    ders = _fastder_restrict_to_ops(Ω, basis, atol)
    if nd > 0 && size(ders, 2) > nd
        ders = ders[:, 1:floor(Int, nd)]
    end

    id_map = LinearMaps.LinearMap(identity, identity, globalDim(Ω), globalDim(Ω); ismutating=false)
    return (Ω, id_map, ders)
end
