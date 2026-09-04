#
# Strata Dleto: QuickDerN
#   Solve-and-lift derivations for tensors of ANY valence.
#
# NOTICE:
#   The solve-and-lift strategy generalised here was developed in companion
#   work by Chris Liu, Joshua Maglione, and James B. Wilson.  Please retain
#   this attribution when reusing this implementation.
#
# This implements docs/design/QuickDer-valence-n.md.  `FastDer3Valent.jl` is a
# line-for-line transcription of the valence-3 reference and stays as the
# oracle (`:QuickDer3` / `:FastDer3Valent`); this file is the general method
# (`:QuickDer`) and is written from the equations rather than from that code,
# so the two are independent implementations that can be compared.
#
# THE SHAPE OF THE METHOD, in one paragraph.  A derivation is a tuple of
# matrices `M_a` with `Σ_a P[ρ,a] (Γ ×_a M_a) = 0` for every chisel row `ρ`.
# Contract every OUTPUT axis of that equation with a thin orthonormal `W_a`
# (d_a x r_a) and it becomes a much smaller system in the restricted unknowns
# `Y_a = M_a W_a` -- one whose size is governed by `r`, not by `d`.  Solve that,
# then recover the missing half `Z_a = M_a W_a⊥` one axis at a time by least
# squares, then CHECK the answer against the defining equation, because
# solve-and-lift is only *generically* correct at a given `r`.
#
# WHY THE SKETCH IS RANDOM BY DEFAULT.  Liu's valence-3 kernel restricts to a
# CORNER, `W_a = I[:, 1:r_a]`, which is free (the sketches are slices).  That is
# a fine choice on a dense generic tensor and a fragile one on a structured
# tensor, where the corner block can miss the support.  Measured on the
# unscrambled sphere octant (support `i+j+k = d-1`), valence 3, `:corner`:
# d = 10, 12 land the answer; from d = 16 the LIFT operator `S_a(a)ᵗ` loses
# column rank -- the corner cross sketch still meets the support hyperplane,
# because one axis of `S_a` stays full, but the part of it the lift sees no
# longer determines `Z_a` -- and only the minimum-norm fallback below saves the
# answer.  A random orthonormal `W_a` costs one pass over Γ per axis and is
# generic for any Γ, with no such cliff.  Both are available; `:corner` is kept
# because it is genuinely cheaper when the tensor is dense and generic.
#
# THE KERNEL IS PLAIN ARRAYS.  ITensors appear only at the boundary, and every
# contraction goes through `_qdn_ttm`, so swapping in TensorOperations,
# Strided, Finch or a GPU backend is one function (see the contraction study in
# bench/reports/night-2026-09-03/contraction-options.md).
#

using LinearAlgebra
using LinearMaps
using Random

# ---------------------------------------------------------------------------
# The method
# ---------------------------------------------------------------------------

"""
Restricted systems with at least this many unknowns take the Gram route
(`:GramSolver`) instead of a full SVD in the dense branch.  Below it the SVD
costs well under a second and keeps full precision.  A `Ref` so benchmarks can
flip the route (`QDN_GRAM_MIN_COLS[] = typemax(Int)` forces the SVD).
"""
const QDN_GRAM_MIN_COLS = Ref(1000)

"""
    QuickDerMethod(; restriction = :random, sizes = nothing, solver = :AutoSolver,
                     verify = :random, nslices = 4, seed = nothing)

Solve-and-lift derivations for a tensor of any valence `n >= 2`, any per-axis
dimensions, any chisel with at least one engaged axis, and any
`IndTransverseOps` operator space.

- `restriction`  `:random` (default) sketches each axis with the Q factor of a
  random square Gaussian; `:corner` uses `I[:, 1:r_a]`, which makes every
  sketch a slice and every lift right-hand side a sub-block -- Liu's original
  choice, and the fast one on a dense generic tensor.  `:corner` is fragile on
  a tensor whose leading corner is degenerate: on the unscrambled sphere its
  lift operator loses column rank from d = 16 up (see the file header), so
  `:random` is the default.
- `sizes`        override the restriction sizes `r_1..r_n` (see
  `_qdn_restriction_sizes` for what is chosen otherwise and why).
- `solver`       null solver for the MATRIX-FREE restricted solve.  The dense
  branch always uses `:SVDSolver`, because when the restricted matrix fits in
  memory a real SVD is both faster and more accurate than any iterative
  method at these sizes.
- `verify`       `:random` (default) checks the defining equation on `nslices`
  output slices of the largest engaged axis; `:full` checks all of it when
  `prod(dims) <= 2e7`; `:none` skips the check and is for benchmarking only.
- `seed`         makes the sketch (and the choice of verification slices)
  reproducible.  Without it the global RNG is used.

Element type follows `eltype(Γ)`; `tol` is relative and floored at
`sqrt(eps(T))` by `_qd_tolerance`, as in `FastDer3ValentMethod`.

Like `FastDer3ValentMethod`, the kernel solves over ALL matrices and the answer
is intersected with `Ω` afterwards (`_fastder_restrict_to_ops`), so the result
means the same thing `SylverLining` on that `Ω` means: coordinates in `Ω`, of
the derivations that lie in `Ω`.
"""
struct QuickDerMethod <: DerivationMethod
    restriction::Symbol
    sizes::Union{Nothing, Vector{Int}}
    solver::Symbol
    verify::Symbol
    nslices::Int
    seed::Union{Nothing, Int}
end

function QuickDerMethod(; restriction::Symbol = :random, sizes = nothing,
                        solver::Symbol = :AutoSolver, verify::Symbol = :random,
                        nslices::Integer = 4, seed = nothing)
    restriction in (:random, :corner) ||
        error("QuickDerMethod: restriction must be :random or :corner, got :$restriction.")
    verify in (:random, :full, :none) ||
        error("QuickDerMethod: verify must be :random, :full or :none, got :$verify.")
    nslices >= 1 || error("QuickDerMethod: nslices must be at least 1, got $nslices.")
    return QuickDerMethod(restriction,
                          sizes === nothing ? nothing : Int[Int(s) for s in sizes],
                          solver, verify, Int(nslices),
                          seed === nothing ? nothing : Int(seed))
end

# ---------------------------------------------------------------------------
# The contraction kernel -- the ONE place a backend swap has to touch
# ---------------------------------------------------------------------------

"""
    _qdn_front(N, a) -> NTuple{N,Int}

The permutation of `1:N` that brings axis `a` to the front and leaves the
others in their original order.  Its inverse puts it back.
"""
_qdn_front(N::Integer, a::Integer) =
    ntuple(i -> i == 1 ? Int(a) : (i - 1 < a ? i - 1 : i), N)

"""
    _qdn_ttm(G, M, a) -> Array

The mode-`a` product `G ×_a M` with `M` of size `(size(G,a), k)`:

    (G ×_a M)[.., i, ..] = Σ_p G[.., p, ..] M[p, i]

M's FIRST index meets the tensor.  That is the convention `applyDerivation` and
`embedITensors` use (an operator ITensor carries the frame index first), so
nothing in this file ever transposes an operator at the ITensor boundary.

Implemented as permutedims + reshape + `mul!`, i.e. one GEMM on the mode-`a`
unfolding, and used for EVERY contraction in this file -- the sketches, the
restricted solve, the lift right-hand sides and the verification -- so a later
move to TensorOperations/Strided/Finch or a GPU is a change to this function
alone.
"""
function _qdn_ttm(G::AbstractArray{T,N}, M::AbstractMatrix{T}, a::Integer) where {T,N}
    d = size(G, a)
    size(M, 1) == d || throw(DimensionMismatch(
        "mode-$a product: matrix is $(size(M)) but axis $a has length $d"))
    k = size(M, 2)
    GA = G isa Array{T,N} ? G : Array(G)
    if a == 1
        Gm = reshape(GA, d, :)
        out = Matrix{T}(undef, k, size(Gm, 2))
        mul!(out, transpose(M), Gm)
        return reshape(out, ntuple(i -> i == 1 ? k : size(GA, i), N))
    end
    perm = _qdn_front(N, a)
    Gm = reshape(permutedims(GA, perm), d, :)
    out = Matrix{T}(undef, k, size(Gm, 2))
    mul!(out, transpose(M), Gm)
    Op = reshape(out, ntuple(i -> i == 1 ? k : size(GA, perm[i]), N))
    return permutedims(Op, invperm(collect(perm)))
end

"""
    _qdn_unfold(G, a) -> Matrix

The mode-`a` unfolding of `G`: a `size(G,a) x prod(other dims)` matrix whose
columns run over the other axes in their natural order, column-major.  That
column order is the one every row permutation and every lift right-hand side in
this file assumes.
"""
function _qdn_unfold(G::AbstractArray{T,N}, a::Integer) where {T,N}
    GA = G isa Array{T,N} ? G : Array(G)
    d = size(GA, a)
    a == 1 && return reshape(GA, d, :)
    return reshape(permutedims(GA, _qdn_front(N, a)), d, :)
end

"""
    _qdn_row_perm(r, a) -> Vector{Int}

Where the rows of the axis-`a` block land.  Unfolding the restricted equation
with axis `a` LAST orders its rows by `(i_{-a}, i_a)`; the assembled system
orders them column-major over `(i_1..i_n)`.  `perm[t]` is the full-system row
of the `t`-th row in the axis-last order, so `Block[perm, :] = kron(I, Sᵗ)`.
"""
function _qdn_row_perm(r::AbstractVector{<:Integer}, a::Integer)
    n = length(r)
    lin = reshape(collect(1:prod(r)), r...)
    others = [b for b in 1:n if b != a]
    return vec(permutedims(lin, (others..., a)))
end

# ---------------------------------------------------------------------------
# Choosing the restriction sizes
# ---------------------------------------------------------------------------

"""
    _qdn_restriction_sizes(dims, engaged, n = length(dims)) -> Vector{Int}

Per-axis restriction sizes `r_1..r_n`, from the two conditions of
docs/design/QuickDer-valence-n.md section 1:

  (i)  `∏ r_a >= Σ_{a engaged} d_a r_a + slack`
       -- the restricted system has enough equations to be generically full
       column rank, so its null space is the restriction of the true
       derivation space and nothing more;
  (ii) for every engaged axis that will need lifting (`r_a < d_a`),
       `∏_{b≠a} r_b >= d_a + slack`
       -- the lift operator for that axis has full column rank.

The start is the balanced value `r_a = min(d_a, ceil((n·max d)^(1/(n-1))) + 1)`,
which is the smallest `r` that makes `∏r` outgrow `Σ d_a r_a` for equal
dimensions; from there the axis with the smallest `r_a/d_a` that is not already
saturated is bumped until both conditions hold.  Tiny axes (a colour axis of
length 3) saturate immediately, get `W_a = I`, and are never lifted.

The slack is 5% (with a small absolute floor).  Sitting exactly at the
crossover makes the restricted matrix square, and a square system whose true
null space is `k`-dimensional is only `k`-dimensional if it has full rank to
the last row -- which is the assumption most likely to fail on a not-quite-
generic tensor.  Examples: `(100,100,100)` -> 19; `(1000,1000,1000)` ->
`(57,56,56)`; `(100,100,100,3)` -> `(11,10,10,3)`; `(5,4,6,3)` -> `(4,4,4,3)`.

When every axis saturates the "restriction" is the identity, `r == dims`, the
restricted system IS the full system, and the conditions are not required (they
cannot always be met at small `d`, e.g. `(3,3,3)`); the loop stops because
there is nothing left to bump.
"""
function _qdn_restriction_sizes(dims::AbstractVector{<:Integer},
                                engaged::AbstractVector{Bool},
                                n::Integer = length(dims))
    n >= 2 || error("QuickDer needs valence at least 2, got $n.")
    d = Int[Int(x) for x in dims]
    base = ceil(Int, (n * maximum(d))^(1 / (n - 1))) + 1
    r = [min(d[a], base) for a in 1:n]

    unknowns() = sum(Int[d[a] * r[a] for a in 1:n if engaged[a]]; init = 0)
    function conditions_hold()
        u = unknowns()
        prod(r) >= u + max(4, ceil(Int, 0.05 * u)) || return false
        for a in 1:n
            (engaged[a] && r[a] < d[a]) || continue
            (prod(r) ÷ r[a]) >= d[a] + max(2, ceil(Int, 0.05 * d[a])) || return false
        end
        return true
    end

    while !conditions_hold()
        cands = [a for a in 1:n if r[a] < d[a]]
        isempty(cands) && break
        r[cands[argmin([r[c] / d[c] for c in cands])]] += 1
    end
    return r
end

"""Validate a user-supplied `sizes` override against the tensor's dimensions."""
function _qdn_check_sizes(sizes::AbstractVector{<:Integer}, dims::AbstractVector{<:Integer})
    length(sizes) == length(dims) ||
        error("QuickDer: sizes has $(length(sizes)) entries but the tensor has " *
              "$(length(dims)) axes.")
    all(a -> 1 <= sizes[a] <= dims[a], eachindex(sizes)) ||
        error("QuickDer: every restriction size must satisfy 1 <= r_a <= d_a; " *
              "got $(collect(sizes)) for dims $(collect(dims)).")
    return Int[Int(s) for s in sizes]
end

# ---------------------------------------------------------------------------
# The sketch
# ---------------------------------------------------------------------------

"""
Per-axis restriction data: `W` (d x r) and its orthogonal complement `Wp`
(d x (d-r)).  `ident` says `W = I[:, 1:r]` and `Wp = I[:, r+1:d]`, in which
case both are left empty and every contraction with them is a slice.  A
saturated axis (`r == d`) is always `ident`, so `Q = I` and its operator needs
no rotation back.
"""
struct _QDNAxis{T}
    d::Int
    r::Int
    ident::Bool
    W::Matrix{T}
    Wp::Matrix{T}
end

function _qdn_axis(::Type{T}, d::Integer, r::Integer, restriction::Symbol, rng) where {T}
    if restriction === :corner || r == d
        return _QDNAxis{T}(d, r, true, Matrix{T}(undef, d, 0), Matrix{T}(undef, d, 0))
    end
    # A full square QR gives W and W⊥ from one factorisation, and they are
    # exactly orthogonal to each other -- which the lift relies on when it
    # reassembles M_a = [Y_a Z_a] Qᵗ.
    Q = Matrix(qr(randn(rng, T, d, d)).Q)
    return _QDNAxis{T}(d, r, false, Q[:, 1:r], Q[:, (r + 1):d])
end

_qdn_modeW(G::AbstractArray{T,N}, ax::_QDNAxis{T}, a::Integer) where {T,N} =
    ax.ident ? (ax.r == size(G, a) ? G : Array(selectdim(G, a, 1:ax.r))) :
               _qdn_ttm(G, ax.W, a)

_qdn_modeWp(G::AbstractArray{T,N}, ax::_QDNAxis{T}, a::Integer) where {T,N} =
    ax.ident ? Array(selectdim(G, a, (ax.r + 1):ax.d)) : _qdn_ttm(G, ax.Wp, a)

"""
    _qdn_assemble(ax, Y, Z) -> Matrix

`M_a` from its two halves: `M_a = [Y_a Z_a] Q_aᵗ = Y_a W_aᵗ + Z_a W_a⊥ᵗ`.
For the corner restriction `Q_a = I` and the two halves simply sit side by
side, which is why `:corner` never has to rotate anything back.
"""
function _qdn_assemble(ax::_QDNAxis{T}, Y::AbstractMatrix{T}, Z::AbstractMatrix{T}) where {T}
    ax.ident && return Matrix(hcat(Y, Z))
    return Y * transpose(ax.W) + Z * transpose(ax.Wp)
end

# ---------------------------------------------------------------------------
# Restricted solve + lift
# ---------------------------------------------------------------------------

"""
    _qdn_pair_tensor(G, axs, a, b) -> Array

`H_{ab} = Γ ×_a W_a⊥ ×_{c ∉ {a,b}} W_c`, the coefficient of `Y_b` in the lift
equation of axis `a`.  The small `W_c` go on FIRST and `W_a⊥` last: contracting
`Γ ×_a W_a⊥` up front would cost `O(d^{n+1})` on the full tensor, while this
order pays `O(r·d^n)` for the first pass and then works on a tensor that is
already `r` in every axis but two.
"""
function _qdn_pair_tensor(G::AbstractArray{T,N}, axs::Vector{_QDNAxis{T}},
                          a::Integer, b::Integer) where {T,N}
    X = G
    for c in 1:N
        (c == a || c == b) && continue
        X = _qdn_modeW(X, axs[c], c)
    end
    return _qdn_modeWp(X, axs[a], a)
end

"""
    _qdn_cross_sketches(G, axs, engaged) -> Dict{Int,Array}

`S_a = Γ ×_{b≠a} W_b` for every engaged axis `a`, built through a shared prefix
chain (`Γ ×_1 W_1 ⋯ ×_{a-1} W_{a-1}`) so the total cost stays about `2·r·d^n`
rather than the `n²/2` passes the definition suggests.  Disengaged axes carry
no unknown and so need no cross sketch -- but they are still sketched, i.e.
their `W_b` is applied inside every other axis's `S_a`.
"""
function _qdn_cross_sketches(G::Array{T,N}, axs::Vector{_QDNAxis{T}},
                             engaged::AbstractVector{Bool}) where {T,N}
    pre = Vector{Array{T,N}}(undef, N)
    pre[1] = G
    for a in 2:N
        pre[a] = _qdn_modeW(pre[a - 1], axs[a - 1], a - 1)
    end
    S = Dict{Int, Array{T,N}}()
    for a in 1:N
        engaged[a] || continue
        X = pre[a]
        for b in (a + 1):N
            X = _qdn_modeW(X, axs[b], b)
        end
        S[a] = X
    end
    return S
end

"""
    _qdn_system_rows(rows, ncols) -> Int

How many rows the restricted system is presented with: `max(rows, ncols)`.

A null space is only reachable through `SVDSolver` if the thin SVD's `V` spans
the whole column space, and for a WIDE matrix (`rows < ncols`) it does not --
`svd` returns `min(rows, ncols)` right singular vectors, so a system with more
unknowns than equations silently reports at most `rows` null directions
instead of the `ncols - rank` it has.  Padding with zero rows costs nothing,
changes no null space, and makes the wide case correct.

The sizes `_qdn_restriction_sizes` picks are never wide -- that is exactly its
condition (i).  The padding is for the two cases that escape it: an explicit
`sizes` override, and a tensor small enough that every axis saturates and the
conditions cannot be met at all (a 3x3 matrix has 9 equations and 18
unknowns).
"""
_qdn_system_rows(rows::Integer, ncols::Integer) = max(Int(rows), Int(ncols))

"""
    _qdn_restricted_matrix(Uf, P, eaxes, r, dims, coff, ncols) -> Matrix

The restricted system as a dense matrix, built directly.

For axis `a` the block is `kron(I_{r_a}, S_a(a)ᵗ)` with its rows permuted from
the axis-last order `(i_{-a}, i_a)` into the column-major order `(i_1..i_n)`
the assembled system uses (`_qdn_row_perm`).  `kron(I, ·)` is block diagonal,
so the blocks are written straight into place rather than materialised.

Never via `Matrix(::FunctionMap)`: that would apply the map once per column,
i.e. `∏r·m` tensor contractions to build something this fills with `n·m·r_a`
copies of an already-computed unfolding.

The matrix is padded with zero rows to at least `ncols` (see
`_qdn_system_rows`), which never happens at the sizes `_qdn_restriction_sizes`
picks and matters when an explicit `sizes` override, or a tensor small enough
that every axis saturates, leaves the system WIDE.
"""
function _qdn_restricted_matrix(Uf::Dict{Int, Matrix{T}}, P::Matrix{T},
                                eaxes::Vector{Int}, r::Vector{Int},
                                dims::Vector{Int}, coff::Dict{Int, Int},
                                ncols::Integer) where {T}
    m = size(P, 1)
    R = prod(r)
    M = zeros(T, _qdn_system_rows(m * R, ncols), ncols)
    for a in eaxes
        Ua = Matrix(transpose(Uf[a]))          # R_a x d_a
        Ra = size(Ua, 1)
        perm = _qdn_row_perm(r, a)
        for rho in 1:m
            c = P[rho, a]
            iszero(c) && continue
            base = (rho - 1) * R
            for j in 1:r[a]
                rows = view(perm, ((j - 1) * Ra + 1):(j * Ra))
                cols = (coff[a] + (j - 1) * dims[a] + 1):(coff[a] + j * dims[a])
                @views M[base .+ rows, cols] .= c .* Ua
            end
        end
    end
    return M
end

"""
    _qdn_restricted_map(S, Uf, P, eaxes, r, dims, coff, ncols) -> LinearMap

The same system, matrix free.

Forward: `y ↦ vec(Σ_a P[ρ,a] (S_a ×_a Y_a))`, `n` mode products per apply.
Adjoint: `E ↦ (Σ_ρ P[ρ,a] · S_a(a) · E_ρ(a)ᵗ)_a`, which is the genuine adjoint,
not a stand-in -- `⟨L y, e⟩ = Σ_ρ P[ρ,a] ⟨Y_aᵗ S_a(a), E_ρ(a)⟩ = ⟨y, Lᵗ e⟩`.
Every solver in `NullSolvers.jl` that never densifies relies on that identity.
"""
function _qdn_restricted_map(S::Dict{Int, Array{T,N}}, Uf::Dict{Int, Matrix{T}},
                             P::Matrix{T}, eaxes::Vector{Int}, r::Vector{Int},
                             dims::Vector{Int}, coff::Dict{Int, Int},
                             ncols::Integer) where {T,N}
    m = size(P, 1)
    R = prod(r)
    rt = ntuple(a -> r[a], N)
    nrows = _qdn_system_rows(m * R, ncols)      # zero-padded when wide

    function fwd(y::AbstractVector)
        out = zeros(T, nrows)
        Out = reshape(view(out, 1:(R * m)), rt..., m)
        for a in eaxes
            Ya = reshape(T.(y[(coff[a] + 1):(coff[a] + dims[a] * r[a])]), dims[a], r[a])
            Ta = _qdn_ttm(S[a], Ya, a)
            for rho in 1:m
                c = P[rho, a]
                iszero(c) && continue
                sl = selectdim(Out, N + 1, rho)
                sl .+= c .* Ta
            end
        end
        return out
    end

    function adj(e::AbstractVector)
        E = reshape(T.(e[1:(R * m)]), rt..., m)
        y = zeros(T, ncols)
        for a in eaxes
            acc = zeros(T, dims[a], r[a])
            for rho in 1:m
                c = P[rho, a]
                iszero(c) && continue
                Eu = _qdn_unfold(Array(selectdim(E, N + 1, rho)), a)   # r_a x R_a
                acc .+= c .* (Uf[a] * transpose(Eu))
            end
            y[(coff[a] + 1):(coff[a] + dims[a] * r[a])] = vec(acc)
        end
        return y
    end

    return LinearMaps.LinearMap{T}(fwd, adj, nrows, ncols; ismutating = false)
end

"""
    _qdn_solve_and_lift(G, P, engaged, r, method, rng, atol, progress)
        -> Vector{Vector{Matrix}} or nothing

One attempt at the whole kernel: sketch, restricted solve, lift, consistency
filter.  Returns the surviving derivations as one matrix per axis (zero on the
disengaged axes, which carry no unknown), an EMPTY vector when the restricted
system already has no null space -- a legitimate answer, "Γ conforms to no
pattern for this chisel" -- or `nothing` when there were restricted solutions
but the consistency filter rejected every one of them, which is the caller's
cue to retry with a larger `r`.
"""
function _qdn_solve_and_lift(G::Array{T,N}, P::Matrix{T}, engaged::Vector{Bool},
                             r::Vector{Int}, method::QuickDerMethod, rng,
                             atol::Real, progress) where {T,N}
    dims = collect(size(G))
    m = size(P, 1)
    eaxes = [a for a in 1:N if engaged[a]]
    axs = [_qdn_axis(T, dims[a], r[a], method.restriction, rng) for a in 1:N]

    S = _qdn_cross_sketches(G, axs, engaged)
    Uf = Dict{Int, Matrix{T}}(a => _qdn_unfold(S[a], a) for a in eaxes)  # d_a x R_a

    coff = Dict{Int, Int}()
    ncols = 0
    for a in eaxes
        coff[a] = ncols
        ncols += dims[a] * r[a]
    end
    R = prod(r)

    # Dense while the matrix is a quarter of the null-solver's own budget --
    # the SVD needs the matrix plus its factors, so filling the whole budget
    # with the matrix alone would leave nothing for the solve.  The row count is
    # `m·∏r` except in the padded wide case (`_qdn_system_rows`), and it is the
    # count actually allocated that has to fit.
    dense_bytes = float(_qdn_system_rows(m * R, ncols)) * ncols * sizeof(T)
    if dense_bytes <= DENSE_BUDGET_BYTES / 2
        Mres = _qdn_restricted_matrix(Uf, P, eaxes, r, dims, coff, ncols)
        # The SVD is O(m n²) and at d = 100 (6859 x 5700) already 52 s; the
        # Gram route (`:GramSolver`, n x n, Cholesky-shifted subspace
        # iteration) is 3.8 s there at half the precision, which the lift's
        # consistency filter and the Z-law check can afford.  Small systems
        # keep the SVD's full precision for free.
        dsolver = ncols >= QDN_GRAM_MIN_COLS[] ? :GramSolver : :SVDSolver
        (vals, vecs, verdict) = solve_nullspace(LinearMaps.LinearMap(Mres), dsolver;
                                                tol = atol, nd = -1, progress = progress,
                                                label = "quickder restricted")
    else
        L = _qdn_restricted_map(S, Uf, P, eaxes, r, dims, coff, ncols)
        (vals, vecs, verdict) = solve_nullspace(L, method.solver;
                                                tol = atol, nd = -1, progress = progress,
                                                label = "quickder restricted")
    end

    k = size(vecs, 2)
    @debug "QuickDer restricted solve" rows = m * R cols = ncols nullity = k certified = verdict.certified rule = verdict.rule below = string(verdict.below) above = string(verdict.above) rss_GB = Sys.maxrss() / 2^30
    k == 0 && return Vector{Vector{Matrix{T}}}()

    Yv = Matrix{Matrix{T}}(undef, N, k)
    for a in eaxes, i in 1:k
        Yv[a, i] = reshape(T.(vecs[(coff[a] + 1):(coff[a] + dims[a] * r[a]), i]),
                           dims[a], r[a])
    end

    # ---- the lift, one thin QR per axis, shared by every basis vector
    lift = [a for a in eaxes if r[a] < dims[a]]
    Zv = Matrix{Matrix{T}}(undef, N, k)
    Rblocks = Matrix{T}[]
    scale = zero(real(T))
    for a in lift
        Ua = Matrix(transpose(Uf[a]))                       # R_a x d_a
        Ra = size(Ua, 1)
        da = dims[a]
        ha = da - r[a]
        A = vcat([P[rho, a] .* Ua for rho in 1:m]...)       # (m R_a) x d_a
        Hs = Dict{Int, Array{T,N}}(b => _qdn_pair_tensor(G, axs, a, b)
                                   for b in eaxes if b != a)

        B = zeros(T, m * Ra, k * ha)
        for i in 1:k, rho in 1:m
            acc = zeros(T, Ra, ha)
            for b in eaxes
                (b == a || iszero(P[rho, b])) && continue
                Wt = _qdn_ttm(Hs[b], Yv[b, i], b)
                acc .-= P[rho, b] .* transpose(_qdn_unfold(Wt, a))
            end
            B[((rho - 1) * Ra + 1):(rho * Ra), ((i - 1) * ha + 1):(i * ha)] = acc
        end

        # One thin QR per axis, shared by every basis vector.  Condition (ii)
        # of the design note says A has full column rank -- generically.  On a
        # tensor where it does not (the unscrambled sphere under the CORNER
        # restriction from d = 16 up: the corner block of a support hyperplane
        # is too thin to determine the lift) the triangular solve raises
        # `SingularException` instead of saying anything useful.  Fall back to
        # the minimum-norm solution there: it is still a genuine derivation
        # whenever the system is CONSISTENT, and the residual filter below is
        # exactly the test of that -- so a recoverable case is recovered and an
        # unrecoverable one is filtered out and reported by `derTrOpsReduced`.
        Z = if size(A, 1) < size(A, 2)
            # Condition (ii) fails outright -- only reachable through an
            # explicit `sizes` override that is too small.  Least squares still
            # returns the best available Z; the filter reports the damage.
            pinv(A) * B
        else
            try
                qr(A) \ B
            catch err
                (err isa LinearAlgebra.SingularException ||
                 err isa LinearAlgebra.LAPACKException) || rethrow()
                pinv(A) * B
            end
        end
        all(isfinite, Z) || (Z = pinv(A) * B)
        AZ = A * Z
        for i in 1:k
            Zv[a, i] = Z[:, ((i - 1) * ha + 1):(i * ha)]
        end
        scale = max(scale, norm(B), norm(AZ))

        # The lift residual is LINEAR in y, so its columns (one per restricted
        # basis vector) are what the consistency filter takes a null space of.
        # A thin QR compresses each axis block to k x k first, which is exact
        # -- an orthonormal Q does not change singular values -- and keeps the
        # peak memory at one axis block instead of all of them.
        Rm = reshape(AZ .- B, (m * Ra) * ha, k)
        push!(Rblocks, size(Rm, 1) >= k ? Matrix(qr(Rm).R) : Matrix(Rm))
    end

    # ---- consistency filter
    #
    # A restricted solution that is not the restriction of a true derivation
    # leaves a residual in the lift equations.  Filtering those combinations
    # out (rather than erroring, as the valence-3 reference does) is what makes
    # the method correct and not merely generic: on a tensor whose restricted
    # null space is legitimately too large, the spurious directions are removed
    # and the genuine ones survive.
    local C::Matrix{T}
    if isempty(Rblocks)
        C = Matrix{T}(LinearAlgebra.I, k, k)                # nothing to lift
    else
        sc = max(scale, eps(real(T)))
        Rall = vcat(Rblocks...) ./ sc
        @debug "QuickDer lift residual spectrum" k svals = string(round.(svdvals(Rall); sigdigits = 3)) atol
        C = nullspace(Rall; atol = real(T)(atol), rtol = zero(real(T)))
        size(C, 2) == 0 && return nothing
    end

    kc = size(C, 2)
    out = Vector{Vector{Matrix{T}}}(undef, kc)
    for j in 1:kc
        Ms = Vector{Matrix{T}}(undef, N)
        for a in 1:N
            if !engaged[a]
                # No unknown on a disengaged axis: its chisel column is zero, so
                # every matrix satisfies the equation there and zero is the
                # representative `SylverLining` returns after its engagement
                # reduction expands.
                Ms[a] = zeros(T, dims[a], dims[a])
                continue
            end
            Yc = zeros(T, dims[a], r[a])
            for i in 1:k
                Yc .+= C[i, j] .* Yv[a, i]
            end
            if r[a] == dims[a]
                Ms[a] = _qdn_assemble(axs[a], Yc, zeros(T, dims[a], 0))
            else
                Zc = zeros(T, dims[a], dims[a] - r[a])
                for i in 1:k
                    Zc .+= C[i, j] .* Zv[a, i]
                end
                Ms[a] = _qdn_assemble(axs[a], Yc, Zc)
            end
        end
        out[j] = Ms
    end
    return out
end

# ---------------------------------------------------------------------------
# Verification -- the Z-law, always
# ---------------------------------------------------------------------------

"""
    _qdn_verify(G, P, engaged, mats, method, atol, r, rng)

Check the DEFINING equation, because solve-and-lift is only generically correct
at a given `r` and the restricted system cannot tell you when its genericity
assumption failed.

`:full` evaluates `Σ_a P[ρ,a] (Γ ×_a M_a)` outright and is used when
`prod(dims) <= 2e7`; otherwise (and by default) `:random` checks `nslices`
output slices along the largest engaged axis `â`.  For the `a ≠ â` terms the
slice is taken from Γ FIRST, so those contractions run on a tensor that is
`nslices` deep rather than `d_â`; only the `a = â` term needs a pass over the
whole tensor, and it needs only the selected columns of `M_â`.

The residual is measured against `‖Γ‖·Σ_a‖M_a‖`, the size of the data the
equation is built from, so the test is invariant to how Γ and the basis happen
to be scaled.
"""
function _qdn_verify(G::Array{T,N}, P::Matrix{T}, engaged::Vector{Bool},
                     mats::Vector{Vector{Matrix{T}}}, method::QuickDerMethod,
                     atol::Real, r::Vector{Int}, rng) where {T,N}
    (method.verify === :none || isempty(mats)) && return nothing
    dims = collect(size(G))
    m = size(P, 1)
    gnorm = norm(G)
    RT = real(T)

    fail(res, bound) = error(
        "QuickDer: the lifted solution does not satisfy the derivation equation " *
        "(relative residual $(res) against the bound $(bound)). The tensor is not " *
        "generic enough for the restriction sizes r = $(r) on dims $(dims); retry " *
        "with a larger `sizes = $(min.(dims, ceil.(Int, 1.5 .* r)))`, with " *
        "restriction = :random if it was :corner, or fall back to :SylverLining.")

    if method.verify === :full && prod(dims) <= 2e7
        for Ms in mats
            bound = atol * gnorm * max(sum(norm, Ms), eps(RT))
            for rho in 1:m
                E = zeros(T, size(G))
                for a in 1:N
                    (!engaged[a] || iszero(P[rho, a])) && continue
                    E .+= P[rho, a] .* _qdn_ttm(G, Ms[a], a)
                end
                norm(E) <= bound || fail(norm(E), bound)
            end
        end
        return nothing
    end

    ahat = argmax([engaged[a] ? dims[a] : -1 for a in 1:N])
    ns = min(method.nslices, dims[ahat])
    sel = randperm(rng, dims[ahat])[1:ns]
    Gslice = Array(selectdim(G, ahat, sel))
    for Ms in mats
        bound = atol * gnorm * max(sum(norm, Ms), eps(RT))
        for rho in 1:m
            E = zeros(T, size(Gslice))
            if !iszero(P[rho, ahat])
                E .+= P[rho, ahat] .* _qdn_ttm(G, Ms[ahat][:, sel], ahat)
            end
            for a in 1:N
                (a == ahat || !engaged[a] || iszero(P[rho, a])) && continue
                E .+= P[rho, a] .* _qdn_ttm(Gslice, Ms[a], a)
            end
            norm(E) <= bound || fail(norm(E), bound)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# ITensor boundary
# ---------------------------------------------------------------------------

function _qdn_validate(Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor)
    Ω isa IndTransverseOps ||
        error("QuickDerMethod requires IndTransverseOps, got $(typeof(Ω)).")
    ndims(Γ) >= 2 || error("QuickDerMethod requires a tensor of valence at least 2.")
    valency(Ω) == ndims(Γ) ||
        error("QuickDerMethod: Ω has valency $(valency(Ω)) but Γ has $(ndims(Γ)) axes.")
    size(P, 2) == ndims(Γ) ||
        error("QuickDerMethod: the chisel has $(size(P,2)) columns but Γ has " *
              "$(ndims(Γ)) axes.")
    size(P, 1) >= 1 || error("QuickDerMethod: the chisel has no rows.")
    all(i -> hasind(Γ, i), frames(Ω)) ||
        error("QuickDerMethod: Γ does not carry the frame of Ω.")
    any(engaged(Matrix{Float64}(P))) ||
        error("QuickDerMethod requires at least one engaged axis.")
    return nothing
end

"""
    derTrOpsReduced(::QuickDerMethod, Ω, P, Γ; tol, nd, progress)
        -> (Ω, id_map, ders)

Solve-and-lift derivations of `Γ` for the chisel `P`, in the coordinates of
`Ω`.  The reduced operator space IS `Ω` and the expand map is the identity, as
in `FastDer3ValentMethod`: the kernel already solves over every axis, including
the disengaged ones (whose operator is zero, the representative `SylverLining`
expands to), so there is nothing to expand.

`nd > 0` caps the number of returned derivations; `tol` is relative and floored
at `sqrt(eps(eltype(Γ)))`.
"""
function derTrOpsReduced(
    method::QuickDerMethod,
    Ω::TransverseOps,
    P::AbstractMatrix,
    Γ::ITensor;
    tol::Real = 1e-6,
    nd = -1,
    progress = false,
    kwargs...,
)::Tuple{TransverseOps, LinearMaps.LinearMap, AbstractMatrix{<:Number}}
    _qdn_validate(Ω, P, Γ)

    # Read Γ in Ω's frame order -- the order the coordinates use, and the order
    # `array` can hand back without a d^n copy when it already matches.
    fr = frames(Ω)
    G0 = ITensors.array(Γ, fr...)
    T = eltype(G0)
    G = G0 isa Array{T} ? G0 : Array(G0)

    atol = _qd_tolerance(T, tol)
    dims = collect(size(G))
    n = length(dims)
    eng = engaged(Matrix{Float64}(P))
    Pm = Matrix{T}(P)
    rng = method.seed === nothing ? Random.default_rng() : MersenneTwister(method.seed)

    r = method.sizes === nothing ? _qdn_restriction_sizes(dims, eng, n) :
                                   _qdn_check_sizes(method.sizes, dims)

    # One retry with r bumped 50%.  The filter rejecting everything means the
    # restricted system saw solutions that no true derivation restricts to,
    # i.e. r was too small for THIS tensor; a bigger r is the only cure that
    # does not change the method.
    tried = Vector{Int}[]
    mats = nothing
    while true
        push!(tried, copy(r))
        mats = _qdn_solve_and_lift(G, Pm, eng, r, method, rng, atol, progress)
        mats === nothing || break
        length(tried) >= 2 && break
        bumped = [min(dims[a], max(r[a] + 1, ceil(Int, 1.5 * r[a]))) for a in 1:n]
        bumped == r && break                      # already unrestricted
        r = bumped
    end
    mats === nothing && error(
        "QuickDer: every restricted solution failed the lift consistency check at " *
        "restriction sizes $(tried) on dims $(dims). Γ is not generic enough for " *
        "this restriction; try `sizes = $(dims)`, `restriction = :random` if it " *
        "was :corner, or fall back to :SylverLining.")

    @debug "QuickDer after lift" nbasis = length(mats) rss_GB = Sys.maxrss() / 2^30
    _qdn_verify(G, Pm, eng, mats, method, atol, r, rng)
    @debug "QuickDer after verify" rss_GB = Sys.maxrss() / 2^30

    isempty(mats) && return (Ω,
        LinearMaps.LinearMap(identity, identity, globalDim(Ω), globalDim(Ω);
                             ismutating = false),
        zeros(T, globalDim(Ω), 0))

    # Universal derivations, cut down to the ones that live in Ω.
    ders = _fastder_restrict_to_ops(Ω, mats, atol)
    @debug "QuickDer after restrict_to_ops" nders = size(ders, 2) rss_GB = Sys.maxrss() / 2^30
    if nd > 0 && size(ders, 2) > nd
        ders = ders[:, 1:floor(Int, nd)]
    end

    id_map = LinearMaps.LinearMap(identity, identity, globalDim(Ω), globalDim(Ω);
                                  ismutating = false)
    return (Ω, id_map, ders)
end
