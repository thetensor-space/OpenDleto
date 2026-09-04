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
# THE KERNEL IS PLAIN ARRAYS -- OF WHATEVER KIND.  ITensors appear only at the
# boundary, and every contraction goes through `_qdn_ttm`, so swapping in
# TensorOperations, Strided or Finch is one function (see the contraction study
# in bench/reports/night-2026-09-03/contraction-options.md).  `device = :gpu`
# uses that same property from the other side: `_qdn_ttm`, `_qdn_unfold` and
# `_qdn_slice` are written against `AbstractArray` and allocate with
# `similar(G, ...)`, so handing them an `MtlArray` keeps every pass over the
# `d^n` tensor on the GPU with no second implementation.  Apple GPUs have no
# Float64, so that path is Float32 and exploratory; the certified answer is the
# Float64 CPU one.
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
Byte budget for the DENSE restricted matrix on the host route.  Mirrors
`DENSE_BUDGET_BYTES / 2` (`NullSolvers.jl`): the matrix plus the Gram plus the
Cholesky factor all have to fit, so filling the whole null-solver budget with
the matrix alone would leave nothing for the solve.  A `Ref` because it is a
policy number, not a fact about the machine -- benchmarks that want the dense
route at a size the default declines (a Float32 CPU run being compared against
a Float32 GPU run, say) raise it explicitly.

Not a `const` expression of `DENSE_BUDGET_BYTES`: `QuickDerN.jl` is included
before `NullSolvers.jl`, so that name does not exist yet at load time.
"""
const QDN_DENSE_BUDGET_BYTES = Ref(2.5 * 2^30)
# 2.5 GB, not 0.5: the matrix-free branch is the weak spot.  On a STRUCTURED
# tensor its spectrum is clustered and Arpack hits its iteration cap
# (XYAUPD_Exception(1)) on the scrambled sphere at d = 150 and 200, where the
# dense Gram route answers in seconds; random tensors converge there
# (bench/reports/night-2026-09-03/quickder-restricted-solvers.csv).  At 2.5 GB
# the Gram route reaches d = 200 at valence 3 (17576 x 15600, 2.2 GB) with a
# peak of ~3x the matrix, under the machine's 10 GB default budget.

"""
Byte budget for the dense restricted matrix on the DEVICE route
(`device = :gpu`).  Larger than the host budget because the device this was
written for has a 52 GB recommended working set while the process that feeds
it has a 16 GB cap: at valence 3 Float32 the matrix is 0.16 GB at d = 100,
1.1 GB at d = 200 and 3.3 GB at d = 300, and only the last is close to the
host-side ceiling (host copy + device copy + device Gram).
"""
const QDN_GPU_DENSE_BUDGET_BYTES = Ref(6.0 * 2^30)

"""
    _qdn_default_free_solver() -> Symbol

Null solver for the MATRIX-FREE restricted branch when the caller left
`solver = :AutoSolver`.

`:AutoSolver` puts LSMR first for a rectangular map, which is right for
`den`'s ill-conditioned densor map and wrong for this one.  Measured on the
matrix-free restricted branch (random `d^3`, 2 threads,
`bench/reports/night-2026-09-03/quickder-restricted-solvers.csv`):

| d | Arpack | Krylov | CG | LSMR |
|---|---|---|---|---|
| 150 | 10.5 s | 76 s | 210 s | did not finish |
| 200 | 49 s | - | - | - |
| 250 | 33 s | - | - | - |

So ARPACK when its package is loaded, `:KrylovSolver` otherwise (the KrylovKit
extension registers itself unconditionally).  An explicit `solver =` still
wins -- this only replaces the *default*.
"""
_qdn_default_free_solver() =
    haskey(SOLVER_REGISTRY, :ArpackSolver) ? :ArpackSolver : :KrylovSolver

"""
    QDN_STAGE_TIMES :: Ref{Union{Nothing, Dict{Symbol,Float64}}}

Opt-in per-stage wall clock, for benchmarking only.  Set it to an empty `Dict`
and the next `derTrOpsReduced` accumulates seconds under `:upload`, `:sketch`,
`:restricted` (dense matrix assembly), `:solve`, `:lift`, `:filter`, `:verify`
and `:restrict_ops`; leave it `nothing` (the default) and the cost is one `Ref`
load per stage.  `GramSolver` adds `:gram`, `:cholesky`, `:subspace` and
`:ritz` through the same dictionary.

Measuring a GPU stage needs a synchronisation point, and every stage here
already ends at one -- a `to_cpu`, or a scalar read like `norm(E)` or
`maximum(abs, diag(G))` -- because the next stage is host work that needs the
device's answer.  That is why the boundaries below sit *after* the transfer
rather than after the last kernel launch: an asynchronous launch would
otherwise charge its time to whichever stage happened to block first.
"""
const QDN_STAGE_TIMES = Ref{Union{Nothing, Dict{Symbol,Float64}}}(nothing)

"""
    _qdn_stage!(name, t0) -> Float64

Charge `time() - t0` to stage `name` when stage timing is on, and return the
current time so consecutive stages can chain (`t0 = _qdn_stage!(:sketch, t0)`).
"""
function _qdn_stage!(name::Symbol, t0::Float64)
    t = time()
    d = QDN_STAGE_TIMES[]
    d === nothing || (d[name] = get(d, name, 0.0) + (t - t0))
    return t
end

"""
    _qdn_check_device(device, T)

`device` is `:cpu` or `:gpu`; `:gpu` additionally needs a functional GPU
backend extension (`using Metal` alongside `using Dleto`) and a Float32
tensor, because Apple GPUs have no Float64 at all.  A GPU run is therefore an
exploratory Float32 run; Float64 certification stays on the CPU.
"""
function _qdn_check_device(device::Symbol, ::Type{T}) where {T}
    device === :cpu && return :cpu
    device === :gpu || error(
        "QuickDerMethod: device must be :cpu or :gpu, got :$device.")
    gpu_available() || error(
        "QuickDerMethod(device = :gpu): no GPU backend is loaded or functional. " *
        "Load one alongside Dleto (`using Metal` on Apple silicon) and check " *
        "`Dleto.gpu_available()`.")
    T === Float32 || error(
        "QuickDerMethod(device = :gpu): the GPU path is Float32 only (Apple GPUs " *
        "have no Float64), but eltype(Γ) is $T. Convert Γ to Float32 for an " *
        "exploratory run, or keep device = :cpu for the certified one.")
    return :gpu
end

"""
    QuickDerMethod(; restriction = :random, sizes = nothing, solver = :AutoSolver,
                     verify = :random, nslices = 4, seed = nothing,
                     device = :cpu)

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
- `solver`       null solver for the MATRIX-FREE restricted solve.  Left at
  `:AutoSolver` it means "pick the one that is fastest on THIS map", which is
  `_qdn_default_free_solver()`, not `AutoSolver`'s own LSMR-first rule.  The
  dense branch picks between `SVDSolver` and `GramSolver` by size and ignores
  this (`QDN_GRAM_MIN_COLS`).
- `verify`       `:random` (default) checks the defining equation on `nslices`
  output slices of the largest engaged axis; `:full` checks all of it when
  `prod(dims) <= 2e7`; `:none` skips the check and is for benchmarking only.
- `seed`         makes the sketch (and the choice of verification slices)
  reproducible.  Without it the global RNG is used.
- `device`       `:cpu` (default) or `:gpu`.  `:gpu` needs a functional GPU
  backend extension and a **Float32** tensor (Apple GPUs have no Float64), and
  errors clearly when either is missing.  What moves to the device is the work
  that scales with `d^n` or with the restricted matrix: the cross sketches, the
  pair tensors of the lift, the Gram/Cholesky/subspace stage of the restricted
  solve (`GramSolver(device = :gpu)`) and the verification.  What stays on the
  host is the small linear algebra the device has no kernels for -- the
  restricted matrix assembly (`kron` + row permutation, a memory-bound scatter
  of a `d_a x R_a` block), the per-axis lift QR, the consistency filter, and the
  `qr`/`svd` inside `GramSolver` (Metal.jl has neither).  A `:gpu` run is
  exploratory: Float32 is its ceiling, and the certified answer is the Float64
  CPU one.

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
    device::Symbol
end

function QuickDerMethod(; restriction::Symbol = :random, sizes = nothing,
                        solver::Symbol = :AutoSolver, verify::Symbol = :random,
                        nslices::Integer = 4, seed = nothing,
                        device::Symbol = :cpu)
    restriction in (:random, :corner) ||
        error("QuickDerMethod: restriction must be :random or :corner, got :$restriction.")
    verify in (:random, :full, :none) ||
        error("QuickDerMethod: verify must be :random, :full or :none, got :$verify.")
    nslices >= 1 || error("QuickDerMethod: nslices must be at least 1, got $nslices.")
    device in (:cpu, :gpu) ||
        error("QuickDerMethod: device must be :cpu or :gpu, got :$device.")
    return QuickDerMethod(restriction,
                          sizes === nothing ? nothing : Int[Int(s) for s in sizes],
                          solver, verify, Int(nslices),
                          seed === nothing ? nothing : Int(seed), device)
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
move to TensorOperations/Strided/Finch is a change to this function alone.

IT IS ALSO THE WHOLE OF THE GPU PORT.  `permutedims`, `reshape`, `similar` and
`mul!` are all implemented for `MtlArray`, so the function is written against
`AbstractArray` and the output is allocated with `similar(G, ...)` rather than
`Matrix{T}(undef, ...)`: hand it a device array and every intermediate stays on
the device.  Nothing here indexes an element, which is what a GPU array
forbids.  The one thing to keep in mind is the `Array(G)` fallback below: it is
guarded on `DenseArray`, and `MtlArray <: AbstractGPUArray <: DenseArray`, so a
device array never takes it (taking it would silently move `d^n` bytes to the
host).
"""
function _qdn_ttm(G::AbstractArray{T,N}, M::AbstractMatrix{T}, a::Integer) where {T,N}
    d = size(G, a)
    size(M, 1) == d || throw(DimensionMismatch(
        "mode-$a product: matrix is $(size(M)) but axis $a has length $d"))
    k = size(M, 2)
    GA = G isa DenseArray ? G : Array(G)
    if a == 1
        Gm = reshape(GA, d, :)
        out = similar(GA, T, (k, size(Gm, 2)))
        mul!(out, transpose(M), Gm)
        return reshape(out, ntuple(i -> i == 1 ? k : size(GA, i), N))
    end
    perm = _qdn_front(N, a)
    Gm = reshape(permutedims(GA, perm), d, :)
    out = similar(GA, T, (k, size(Gm, 2)))
    mul!(out, transpose(M), Gm)
    Op = reshape(out, ntuple(i -> i == 1 ? k : size(GA, perm[i]), N))
    return permutedims(Op, invperm(collect(perm)))
end

"""
    _qdn_slice(G, a, idx) -> array

`selectdim(G, a, idx)` materialised on the device `G` lives on.  `copy` of a
view does that for both `Array` and `MtlArray` (GPUArrays lowers it to a
kernel), where `Array(view)` would have moved a device slice to the host.
"""
_qdn_slice(G::AbstractArray, a::Integer, idx) = copy(selectdim(G, a, idx))

"""
    _qdn_host(x) -> Array

`x` as a host `Array`, without a copy when it is one already.  Every result
this file hands back to `_fastder_restrict_to_ops`, to a null solver or to a
host least-squares solve goes through here.
"""
_qdn_host(x::AbstractArray{T,N}) where {T,N} = x isa Array{T,N} ? x : Array(to_cpu(x))

"""
    _qdn_zeros_like(G, dims) -> array

A zero array of shape `dims` on the same device as `G`.
"""
_qdn_zeros_like(G::AbstractArray{T}, dims::Tuple) where {T} =
    fill!(similar(G, T, dims), zero(T))

"""
    _qdn_unfold(G, a) -> Matrix

The mode-`a` unfolding of `G`: a `size(G,a) x prod(other dims)` matrix whose
columns run over the other axes in their natural order, column-major.  That
column order is the one every row permutation and every lift right-hand side in
this file assumes.

Device-preserving, like `_qdn_ttm`: an `MtlArray` in gives an `MtlMatrix` out
(`reshape` and `permutedims` both keep the array type), and only a
non-`DenseArray` takes the host fallback.
"""
function _qdn_unfold(G::AbstractArray{T,N}, a::Integer) where {T,N}
    GA = G isa DenseArray ? G : Array(G)
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

`W` and `Wp` are typed `AbstractMatrix` rather than `Matrix` so the same struct
can carry device copies for the `d^n` contractions (`_qdn_axes_device`) while
the host copies stay behind for `_qdn_assemble`, which builds the answer the
caller gets.  Both are `d x r`-ish and tiny; the type looseness costs a dynamic
dispatch per mode product, against a GEMM.
"""
struct _QDNAxis{T}
    d::Int
    r::Int
    ident::Bool
    W::AbstractMatrix{T}
    Wp::AbstractMatrix{T}
end

function _qdn_axis(::Type{T}, d::Integer, r::Integer, restriction::Symbol, rng) where {T}
    if restriction === :corner || r == d
        return _QDNAxis{T}(d, r, true, Matrix{T}(undef, d, 0), Matrix{T}(undef, d, 0))
    end
    # A full square QR gives W and W⊥ from one factorisation, and they are
    # exactly orthogonal to each other -- which the lift relies on when it
    # reassembles M_a = [Y_a Z_a] Qᵗ.  Formed on the HOST even for a device
    # run: Metal.jl has no `qr`, and this is `d x d` once per axis.
    Q = Matrix(qr(randn(rng, T, d, d)).Q)
    return _QDNAxis{T}(d, r, false, Q[:, 1:r], Q[:, (r + 1):d])
end

"""
    _qdn_axes_device(axs) -> Vector{_QDNAxis}

The same axes with `W`/`Wp` resident on the GPU, so that every mode product
against the full tensor is a device GEMM.  An `ident` axis is left alone: its
matrices are empty (`d x 0`) and never contracted -- the slice path is used
instead -- and a zero-length device buffer is a needless special case.
"""
_qdn_axes_device(axs::Vector{_QDNAxis{T}}) where {T} =
    _QDNAxis{T}[ax.ident ? ax : _QDNAxis{T}(ax.d, ax.r, false, to_gpu(ax.W), to_gpu(ax.Wp))
                for ax in axs]

_qdn_modeW(G::AbstractArray{T,N}, ax::_QDNAxis{T}, a::Integer) where {T,N} =
    ax.ident ? (ax.r == size(G, a) ? G : _qdn_slice(G, a, 1:ax.r)) :
               _qdn_ttm(G, ax.W, a)

_qdn_modeWp(G::AbstractArray{T,N}, ax::_QDNAxis{T}, a::Integer) where {T,N} =
    ax.ident ? _qdn_slice(G, a, (ax.r + 1):ax.d) : _qdn_ttm(G, ax.Wp, a)

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
                          a::Integer, b::Integer)::AbstractArray{T,N} where {T,N}
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
function _qdn_cross_sketches(G::AbstractArray{T,N}, axs::Vector{_QDNAxis{T}},
                             engaged::AbstractVector{Bool}) where {T,N}
    pre = Vector{AbstractArray{T,N}}(undef, N)
    pre[1] = G
    for a in 2:N
        pre[a] = _qdn_modeW(pre[a - 1], axs[a - 1], a - 1)
    end
    S = Dict{Int, AbstractArray{T,N}}()
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

WHERE THE WORK RUNS.  `G` arrives on whichever device the caller put it on
(`method.device`).  Everything that touches it -- the cross sketches and the
pair tensors of the lift -- runs there, against the device copies of `W`/`W⊥`.
Everything downstream of the sketches is small (`S_a` is `r^{n-1} x d_a`, a few
MB even at d = 300) and is pulled back to the host, because that is where the
rest of the linear algebra has to happen anyway: the restricted matrix is a
memory-bound scatter of `kron(I, S_a(a)ᵗ)`, the lift is a QR, the consistency
filter is a `nullspace`, and the answer has to be host matrices for
`_fastder_restrict_to_ops`.  The one big piece that does go to the device is
the restricted solve, through `GramSolver(device = :gpu)`.
"""
function _qdn_solve_and_lift(G::AbstractArray{T,N}, P::Matrix{T}, engaged::Vector{Bool},
                             r::Vector{Int}, method::QuickDerMethod, rng,
                             atol::Real, progress) where {T,N}
    dims = collect(size(G))
    m = size(P, 1)
    eaxes = [a for a in 1:N if engaged[a]]
    on_gpu = method.device === :gpu
    haxs = [_qdn_axis(T, dims[a], r[a], method.restriction, rng) for a in 1:N]
    # Host axes build the answer (`_qdn_assemble`); device axes do the d^n work.
    axs = on_gpu ? _qdn_axes_device(haxs) : haxs

    tstage = time()
    S = _qdn_cross_sketches(G, axs, engaged)
    # d_a x R_a, and small: unfold on the device, then keep the host copy that
    # the restricted matrix, the lift operator and the adjoint all need.  This
    # is also the synchronisation point of the sketch stage on a device run.
    Uf = Dict{Int, Matrix{T}}(a => _qdn_host(_qdn_unfold(S[a], a)) for a in eaxes)
    tstage = _qdn_stage!(:sketch, tstage)

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
    # count actually allocated that has to fit.  The GPU route gets its own,
    # larger budget: the Gram and the Cholesky live on a device with a 52 GB
    # working set, so the binding constraint there is the HOST copy, not the
    # solve (`QDN_GPU_DENSE_BUDGET_BYTES`).
    dense_bytes = float(_qdn_system_rows(m * R, ncols)) * ncols * sizeof(T)
    dense_budget = on_gpu ? QDN_GPU_DENSE_BUDGET_BYTES[] : QDN_DENSE_BUDGET_BYTES[]
    if dense_bytes <= dense_budget
        Mres = _qdn_restricted_matrix(Uf, P, eaxes, r, dims, coff, ncols)
        tstage = _qdn_stage!(:restricted, tstage)
        # The SVD is O(m n²) and at d = 100 (6859 x 5700) already 52 s; the
        # Gram route (`GramSolver`, n x n, Cholesky-shifted subspace
        # iteration) is 3.8 s there at half the precision, which the lift's
        # consistency filter and the Z-law check can afford.  Small systems
        # keep the SVD's full precision for free.  `GramSolver` is the one
        # solver that carries the device through: it forms the Gram, the
        # Cholesky and the subspace solves on the GPU when asked.
        dsolver = ncols >= QDN_GRAM_MIN_COLS[] ?
                  GramSolver(device = on_gpu ? :gpu : :cpu) : SVDSolver()
        (vals, vecs, verdict) = solve_nullspace(LinearMaps.LinearMap(Mres), dsolver;
                                                tol = atol, nd = -1, progress = progress,
                                                label = "quickder restricted")
        tstage = _qdn_stage!(:solve, tstage)
    else
        # Matrix free stays on the HOST whatever `device` says: the null
        # solvers here iterate with host vectors, and one apply is
        # `∏r · Σd` flops on sketches of a few MB -- the measured cost at
        # d = 100 is 0.12 ms per apply, i.e. the iteration count is the whole
        # story (see the native-core-plan entry on the night board), and a
        # device round trip per apply would only add latency.
        Sh = Dict{Int, Array{T,N}}(a => _qdn_host(S[a]) for a in eaxes)
        L = _qdn_restricted_map(Sh, Uf, P, eaxes, r, dims, coff, ncols)
        fsolver = method.solver === :AutoSolver ? _qdn_default_free_solver() :
                                                  method.solver
        tstage = _qdn_stage!(:restricted, tstage)
        (vals, vecs, verdict) = solve_nullspace(L, fsolver;
                                                tol = atol, nd = -1, progress = progress,
                                                label = "quickder restricted")
        tstage = _qdn_stage!(:solve, tstage)
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
        # The pair tensors are the last pass over the full tensor, so they are
        # built on the device; the results are `r^{n-2} x d_b x (d_a - r_a)`
        # (a few MB even at d = 300) and everything downstream of them -- the
        # right-hand sides, the QR, the residual filter -- is host work, so
        # they come back here.
        Hs = Dict{Int, Array{T,N}}(b => _qdn_host(_qdn_pair_tensor(G, axs, a, b))
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

    tstage = _qdn_stage!(:lift, tstage)

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

    tstage = _qdn_stage!(:filter, tstage)

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
                Ms[a] = _qdn_assemble(haxs[a], Yc, zeros(T, dims[a], 0))
            else
                Zc = zeros(T, dims[a], dims[a] - r[a])
                for i in 1:k
                    Zc .+= C[i, j] .* Zv[a, i]
                end
                Ms[a] = _qdn_assemble(haxs[a], Yc, Zc)
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

Runs wherever `G` lives.  On a device run the answer's matrices are host
arrays (that is what the caller gets), so they are uploaded once here; the
accumulator, the mode products and the `norm` are all device operations, and
only the two scalars per check come back.
"""
function _qdn_verify(G::AbstractArray{T,N}, P::Matrix{T}, engaged::Vector{Bool},
                     mats::Vector{Vector{Matrix{T}}}, method::QuickDerMethod,
                     atol::Real, r::Vector{Int}, rng) where {T,N}
    (method.verify === :none || isempty(mats)) && return nothing
    dims = collect(size(G))
    m = size(P, 1)
    gnorm = norm(G)
    RT = real(T)
    up = method.device === :gpu ? to_gpu : identity

    fail(res, bound) = error(
        "QuickDer: the lifted solution does not satisfy the derivation equation " *
        "(relative residual $(res) against the bound $(bound)). The tensor is not " *
        "generic enough for the restriction sizes r = $(r) on dims $(dims); retry " *
        "with a larger `sizes = $(min.(dims, ceil.(Int, 1.5 .* r)))`, with " *
        "restriction = :random if it was :corner, or fall back to :SylverLining.")

    if method.verify === :full && prod(dims) <= 2e7
        for Ms in mats
            bound = atol * gnorm * max(sum(norm, Ms), eps(RT))
            Md = [engaged[a] ? up(Ms[a]) : Ms[a] for a in 1:N]
            for rho in 1:m
                E = _qdn_zeros_like(G, size(G))
                for a in 1:N
                    (!engaged[a] || iszero(P[rho, a])) && continue
                    E .+= P[rho, a] .* _qdn_ttm(G, Md[a], a)
                end
                res = norm(E)
                res <= bound || fail(res, bound)
            end
        end
        return nothing
    end

    ahat = argmax([engaged[a] ? dims[a] : -1 for a in 1:N])
    ns = min(method.nslices, dims[ahat])
    sel = randperm(rng, dims[ahat])[1:ns]
    Gslice = _qdn_slice(G, ahat, sel)
    for Ms in mats
        bound = atol * gnorm * max(sum(norm, Ms), eps(RT))
        # `Ms[ahat][:, sel]` is selected on the HOST and uploaded: a device
        # `getindex` with an index vector is a kernel launch for what is a
        # `d x nslices` copy.
        Md = [engaged[a] ? up(a == ahat ? Ms[a][:, sel] : Ms[a]) : Ms[a] for a in 1:N]
        for rho in 1:m
            E = _qdn_zeros_like(Gslice, size(Gslice))
            if !iszero(P[rho, ahat])
                E .+= P[rho, ahat] .* _qdn_ttm(G, Md[ahat], ahat)
            end
            for a in 1:N
                (a == ahat || !engaged[a] || iszero(P[rho, a])) && continue
                E .+= P[rho, a] .* _qdn_ttm(Gslice, Md[a], a)
            end
            res = norm(E)
            res <= bound || fail(res, bound)
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
at `sqrt(eps(eltype(Γ)))` by `qd_tolerance` (src/solvers/Precision.jl).

PRECISION.  The arithmetic runs in `compute_eltype(eltype(Γ))`, which is
`eltype(Γ)` except for Float16 (no CPU BLAS or LAPACK has a half-precision
path); the returned coordinates are rounded back to `eltype(Γ)`, so a Float16
tensor gives Float16 derivations at Float32 reliability.  The tolerance is
floored on the STORED type, not the computed one: promoting the arithmetic buys
stability, not information about the data.
"""
function derTrOpsReduced(
    method::QuickDerMethod,
    Ω::TransverseOps,
    P::AbstractMatrix,
    Γ::ITensor;
    tol::Real = TOL_DEFAULT,
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

    # STORAGE type vs COMPUTE type.  `Tc` is the arithmetic; it differs from `T`
    # only for Float16, which has no BLAS or LAPACK anywhere (see
    # `Dleto.compute_eltype`).  The whole kernel below is written against one
    # element type, so the promotion happens here, once, in the same pass that
    # already materialises `G` -- and it is also what lets a Float16 tensor use
    # `device = :gpu`, since Apple GPUs want exactly Float32.
    Tc = compute_eltype(T)
    G = (T === Tc && G0 isa Array{T}) ? G0 : Array{Tc}(G0)

    # One upload for the whole run: every pass over the full tensor (the cross
    # sketches, the pair tensors of the lift, the verification) then happens on
    # the device, and nothing sends `d^n` bytes back.
    _qdn_check_device(method.device, Tc)
    tstage = time()
    Gk = method.device === :gpu ? to_gpu(G) : G
    tstage = _qdn_stage!(:upload, tstage)

    # On the STORED type: every check the kernel makes compares a product of two
    # data-sized quantities against zero, and promoting the arithmetic does not
    # make the data finer.  `qd_tolerance` (src/solvers/Precision.jl) has the
    # measured table.
    atol = _qd_tolerance(T, tol)
    dims = collect(size(G))
    n = length(dims)
    eng = engaged(Matrix{Float64}(P))
    Pm = Matrix{Tc}(P)
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
        mats = _qdn_solve_and_lift(Gk, Pm, eng, r, method, rng, atol, progress)
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
    tstage = time()
    _qdn_verify(Gk, Pm, eng, mats, method, atol, r, rng)
    tstage = _qdn_stage!(:verify, tstage)
    @debug "QuickDer after verify" rss_GB = Sys.maxrss() / 2^30

    isempty(mats) && return (Ω,
        LinearMaps.LinearMap(identity, identity, globalDim(Ω), globalDim(Ω);
                             ismutating = false),
        zeros(T, globalDim(Ω), 0))

    # Universal derivations, cut down to the ones that live in Ω.  Rounded back
    # to the stored type: a Float16 tensor gets Float16 derivations, carrying
    # exactly the precision its data justifies and no more.
    ders = _fastder_restrict_to_ops(Ω, mats, atol)
    ders = eltype(ders) === T ? ders : Matrix{T}(ders)
    tstage = _qdn_stage!(:restrict_ops, tstage)
    @debug "QuickDer after restrict_to_ops" nders = size(ders, 2) rss_GB = Sys.maxrss() / 2^30
    if nd > 0 && size(ders, 2) > nd
        ders = ders[:, 1:floor(Int, nd)]
    end

    id_map = LinearMaps.LinearMap(identity, identity, globalDim(Ω), globalDim(Ω);
                                  ismutating = false)
    return (Ω, id_map, ders)
end
