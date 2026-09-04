
#
# Strata Dleto: AbstractDerivationMethods
#   Algorithms for solving Sylvester equations arising in chiseling.
# -----------------------------------------------------------------------------
# Copyright 2022-2026 Peter A. Brooksbank, Martin D. Kassabov, James B. Wilson
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the “Software”), 
# to deal in the Software without restriction, including without limitation the 
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.
# -----------------------------------------------------------------------------

# """
#     Derivations Methods

#     Interface to methods for solving Sylvester equations arising in chiseling.

# """
"""
    DerivationMethod

    An interface for computing derivations.  Derivation methods 
    should inherit DerivationMethod and implement der and den.
"""
abstract type DerivationMethod end;

"""
    get_derivation_method(method::Symbol; kwargs...)

Factory for selecting a derivation strategy by name.

Supported symbols:

- `:Auto` -- `:QuickDer` when the setting allows it (`IndTransverseOps`, an
  engaged axis, a tensor above `AUTODER_MIN_ENTRIES` entries), `:SylverLining`
  otherwise or whenever QuickDer's own verification rejects its answer
  (`src/solvers/AutoDer.jl`).  The default for `stratify`.
- `:SylverLining` -- the general method: build the derivation--densor operator
  as a `LinearMap` and hand it to a null solver.  Any chisel, any valency, any
  operator space.
- `:QuickDer` -- the *derivation* solve-and-lift at ANY valence
  (`src/solvers/QuickDerN.jl`, docs/design/QuickDer-valence-n.md): sketch every
  axis down to a block that is generically full column rank, solve that small
  system, lift each restricted basis vector back to the full axes by one
  least-squares solve per axis, filter the combinations that do not lift
  consistently, and verify the answer against the defining equation.  Any
  valence `n >= 2`, any dimensions, any chisel with at least one engaged axis,
  any `IndTransverseOps`.
- `:QuickDer3` -- the valence-3 transcription of Liu's `quick-der-lib.jl`, kept
  as the reference oracle for `:QuickDer`.  Valency 3, one-row fully engaged
  chisel, corner restriction, dense solve.  Alias: `:FastDer3Valent`.
- `:QuickSylver` -- Liu's *Sylvester* solve-and-lift (`quicksylver-lib.jl`):
  the same idea for `XR + SY = T`, restricting two axes and lifting an affine
  frame.  Chisels with exactly two engaged axes, e.g. `AdjointChisel`.

All three lift solvers are "solve-and-lift"; they differ in how many axes get
restricted and therefore in which chisels and valencies they can handle.
`:QuickDer` used to be an alias for the valence-3 transcription; it now names
the general method, and the transcription answers to `:QuickDer3` (and still to
`:FastDer3Valent`) so that the two can be compared on the valence-3 cases where
both apply.
"""
function get_derivation_method(method::Symbol; kwargs...)::DerivationMethod
    if method === :Auto
        return AutoDerMethod(; kwargs...)
    elseif method === :SylverLining
        return SylverLiningMethod(; kwargs...)
    elseif method === :QuickDer
        return QuickDerMethod(; kwargs...)
    elseif method === :QuickDer3 || method === :FastDer3Valent
        return FastDer3ValentMethod(; kwargs...)
    elseif method === :QuickSylver
        return QuickSylverMethod(; kwargs...)
    end
    error("Unknown derivation method symbol: $method. " *
          "Known: :Auto, :SylverLining, :QuickDer, :QuickDer3 (alias :FastDer3Valent), " *
          ":QuickSylver.")
end

"""
    der(method::DerivationMethod,
            Ω::TransverseOps,
            P::AbstractMatrix,
            Γ::ITensor;
            nd=-1,
            tol::Real=TOL_DEFAULT,
        ) :: Vector{Vector{ITensor}}

    The **Z-set**: a basis of the `P`-derivations of `Γ`.  Every returned
    derivation `D` satisfies the defining equation approximately, i.e.

        applyDerivation(Γ, D, Chisel(P, frame)) ≈ 0

    which is the law `test/TestDerivationLaws.jl` checks for every solver.

    Computes up to `nd` many `P`-derivations of `Γ` for the to the given chisel `P` and transverse operators `Ω`.
    If `nd` is negative or exceeds the dimension of the derivation space
    then the a basis for the derivation space is returned.
    - `method`: An instance of a subtype of `DerivationMethod` defining the solving method.
    - `Ω`: TransverseOps.
    - `P`: a linear chisel
    - `Γ`: The input tensor
    - `nd`: (optional) Maximum number of singular vectors to compute (default: 10). 
    If infinite `Inf` then return all, 
    - `tol`: (optional) Tolerance for the solver (default: 1e-6).
    attempt to compute a basis for the derivation space.

    Returns a vector of derivations as `ITensor`s with `a` axis labelled by `(a,a')`.
"""
function der(method::DerivationMethod,
    Ω::TransverseOps,
    P::AbstractMatrix,
    Γ::ITensor;
    tol::Float64=TOL_DEFAULT,
    nd=-1,
    kwargs...
    ) :: Vector{Vector{ITensor}}
    (rΩ, expand_map, reduced_der_cood) = derTrOpsReduced(method, Ω, P, Γ; tol=tol, nd=nd, kwargs...)
    return [ embedITensors(Ω, expand_map(reduced_der_cood[:,i]))
             for i in 1:size(reduced_der_cood,2) ]
end;

"""
    derReduced(method, Ω, P, Γ; tol, nd) :: Vector{Vector{ITensor}}

    As `der`, but the operators are embedded in the *reduced* transverse
    operator space rather than expanded back into `Ω`.  Careful with
    symmetries: axes that were fused or disengaged are not present.
"""
function derReduced(method::DerivationMethod,
    Ω::TransverseOps,
    P::AbstractMatrix,
    Γ::ITensor;
    tol::Float64=TOL_DEFAULT,
    nd=-1,
    kwargs...
    ) :: Vector{Vector{ITensor}}
    (rΩ, expand_map, reduced_der_cood) = derTrOpsReduced(method, Ω, P, Γ; tol=tol, nd=nd, kwargs...)
    return [ embedITensors(rΩ, reduced_der_cood[:,i])
             for i in 1:size(reduced_der_cood,2) ]
end;

"""
    derTrOps(method, Ω, P, Γ; tol, nd) :: AbstractMatrix

    A basis of the Z-set as the columns of a matrix, in the coordinates of
    the full `Ω`.  Each column can be turned into operators with
    `embedITensors(Ω, column)`.
"""
function derTrOps(method::DerivationMethod,
    Ω::TransverseOps,
    P::AbstractMatrix,
    Γ::ITensor;
    tol::Float64=TOL_DEFAULT,
    nd=-1
    ) :: AbstractMatrix{<: Number}
    (rΩ, expand_map, reduced_der_cood) = derTrOpsReduced(method, Ω, P, Γ; tol=tol, nd=nd)
    return hcat([expand_map(reduced_der_cood[:,i])
                 for i in 1:size(reduced_der_cood,2)]...)
end;


"""
    Retrun tuple of rΩ, expand_map and matrix whose columns can be expanded into ITensor via Ω
"""
function derTrOpsReduced(method::DerivationMethod,
    Ω::TransverseOps, 
    P::AbstractMatrix, 
    Γ::ITensor; 
    tol::Float64=TOL_DEFAULT,
    nd=10,
    kwargs...
    ) :: Tuple{TransverseOps, LinearMaps.LinearMap, AbstractMatrix{<: Number} }
    @assert false "Calling Placeholder Abstract Function"
end


# this should be  moved to DerivationMethodSylverLininig,jl  
derTrOpsReduced( Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor; tol::Float64=TOL_DEFAULT, nd=10) :: Tuple{TransverseOps, LinearMaps.LinearMap, AbstractMatrix{<: Number}} = 
    derTrOpsReduced(SylverLiningMethod(), Ω, P, Γ; tol=tol, nd=nd); 

# `den` -- the T-set / densor -- lives in Densors.jl, next to `stratify`.


#---- Convenience Functions -------------------------------------------------------

"""
    universalSetup(Γ::ITensor)

    The unrestricted defaults: the universal chisel on every axis, and full
    square matrices on every axis.  This is the hand-assembly that was
    repeated at every entry point; see docs/review/Refactor-Plan.md section 1
    for the `Chisel` type that replaces it.
"""
function universalSetup(Γ::ITensor)
    fr = collect(inds(Γ))
    return (UniversalChisel(length(fr)), fr, IndTransverseOps(fr, UniversalOp()))
end

__asITensor(Γ::AbstractArray) =
    ITensor(Γ, [Index(size(Γ, i), "a_$i") for i in 1:ndims(Γ)]...)

"""
    der(Γ; nd=-1, tol::Real=TOL_DEFAULT)

    Convenience method to compute derivations of a tensor using defaults:
    - Universal chisel
    - Universal transverse operators
    - SylverLiningMethod
"""
function der(Γ::ITensor; nd=-1, tol::Real=TOL_DEFAULT) :: Vector{Vector{ITensor}}
    ch, _, ops = universalSetup(Γ)
    return der(SylverLiningMethod(), ops, ch, Γ; nd=nd, tol=tol)
end

# NOTE on the two kinds of keyword: `kwargs...` here is forwarded to the
# *method constructor* (`solver=` and friends), while `progress` is a
# *per-call* option consumed by the solve.  They were conflated -- everything
# went to the constructor -- so `der(:SylverLining, Γ; progress=:solve)` died
# with "SylverLiningMethod does not support keyword progress".  Per-call
# options therefore have to be named explicitly in every symbol overload.
function der(method::Symbol, Γ::ITensor; nd=-1, tol::Real=TOL_DEFAULT,
             progress=false, kwargs...) :: Vector{Vector{ITensor}}
    ch, _, ops = universalSetup(Γ)
    return der(get_derivation_method(method; kwargs...), ops, ch, Γ;
               nd=nd, tol=tol, progress=progress)
end

der(Γ::AbstractArray; nd=-1, tol::Real=TOL_DEFAULT) =
    der(__asITensor(Γ); nd=nd, tol=tol)

der(method::Symbol, Γ::AbstractArray; nd=-1, tol::Real=TOL_DEFAULT, progress=false, kwargs...) =
    der(method, __asITensor(Γ); nd=nd, tol=tol, progress=progress, kwargs...)

function der(ch::AbstractMatrix, Γ::ITensor; nd=-1, tol::Real=TOL_DEFAULT)
    fr = collect(inds(Γ))
    ops = IndTransverseOps(fr, UniversalOp())
    return der(SylverLiningMethod(), ops, ch, Γ; nd=nd, tol=tol)
end

function der(method::Symbol, ch::AbstractMatrix, Γ::ITensor; nd=-1, tol::Real=TOL_DEFAULT,
             progress=false, kwargs...)
    fr = collect(inds(Γ))
    ops = IndTransverseOps(fr, UniversalOp())
    return der(get_derivation_method(method; kwargs...), ops, ch, Γ;
               nd=nd, tol=tol, progress=progress)
end

der(ch::AbstractMatrix, Γ::AbstractArray; nd=-1, tol::Real=TOL_DEFAULT) =
    der(ch, __asITensor(Γ); nd=nd, tol=tol)

function der(Ω::TransverseOps, ch::AbstractMatrix, Γ::ITensor;
             nd=-1, tol::Real=TOL_DEFAULT, progress=false,
             method::Union{DerivationMethod,Symbol}=:SylverLining, kwargs...)
    m = method isa Symbol ? get_derivation_method(method; kwargs...) : method
    return der(m, Ω, ch, Γ; nd=nd, tol=tol, progress=progress)
end

"""
    der(method::Symbol, Ω::TransverseOps, ch::AbstractMatrix, Γ::ITensor; ...)

Name the method by symbol while giving the full setting (Ω, ch, Γ) explicitly.

This overload was missing.  Every *partial* setting had a symbol form --
`der(:QuickDer, Γ)`, `der(:QuickDer, ch, Γ)` -- and the full setting had only
the instance form `der(SylverLiningMethod(), Ω, ch, Γ)` plus a `method=`
keyword on the argument-order-swapped `der(Ω, ch, Γ)`.  So the one call a
caller comparing methods on a fixed operator space would naturally write,
`der(:QuickDer, Ω, ch, Γ)`, was a `MethodError`.
"""
der(method::Symbol, Ω::TransverseOps, ch::AbstractMatrix, Γ::ITensor;
    nd=-1, tol::Real=TOL_DEFAULT, progress=false, kwargs...) =
    der(get_derivation_method(method; kwargs...), Ω, ch, Γ;
        nd=nd, tol=tol, progress=progress)


# ---------------------------------------------------------------------------
# The Z-law check, memory-lean
# ---------------------------------------------------------------------------

"""
    der_residual(Γ, D, chisel; block_bytes = 2^28) -> Real
    der_residual(G::AbstractArray, Ms::Vector{<:AbstractMatrix}, P; block_bytes) -> Real

`‖Σ_a P[ρ,a]·(Γ ×_a D_a)‖ / (‖Γ‖·max_a‖D_a‖)` -- the defining equation of a
derivation, measured relative to the size of the data it is built from so the
test is invariant to how `Γ` and the basis happen to be scaled.  This is the
check that says whether an answer from ANY solver is a derivation, so it is
the first thing a consumer of this package runs on a result, and it belongs
here rather than in a benchmark script.

`chisel` is the chisel matrix `P` (a `Chisel` is accepted and unwrapped);
`D` is one two-index `ITensor` per axis carrying that axis's frame index, the
`embedITensors` convention, or -- in the array form -- one matrix per axis with
the frame index FIRST, the index that meets the tensor.

WHY THIS IS NOT `applyDerivation`, which is what the benchmarks used to call.
That route goes through ITensor contraction, so it materialises a full `d^n`
intermediate per axis and accumulates them, AND it promotes: the chisel is
typically Float64, so a Float32 tensor is contracted in Float64 and every
intermediate is twice its size.  The bill is about `2(n+1)` tensor copies, and
it lands on whoever is CHECKING an answer rather than on the solver that
produced it -- which is why the benchmark harness's peak RSS was 15 to 20 times
the tensor on shapes where the stratification itself is a fraction of it.
Measured: a valence-4 sphere at d = 150 in Float64 peaks at 6.2 GB through a
profiler that does not check and was killed at 18.7 GB through the harness that
did.  On the video target shape, a 640x480x1800x3 Float32 movie is 6.6 GB and
the old route wanted ~66 GB of Float64 intermediates to check an answer that
fits in far less.

So: read the tensor and the operators as plain arrays in `Γ`'s OWN element
type, and accumulate `Σ_a P[ρ,a]·(Γ ×_a D_a)` with `_qdn_ttm!`'s `α`/`β` -- in
BLOCKS along the longest axis, so the working set is `block_bytes` and not the
tensor.  The block trick is that for a block `q` of axis `c` the output block
`E[..., q, ...]` needs only `G[..., q, ...]` for every term `a ≠ c`, and for
`a = c` it is exactly `G ×_c D_c[:, q]` -- so no term ever needs a full extra
copy.  Two blocks of `block_bytes` replace `2(n+1)` tensors, whatever the
tensor's size, and nothing is promoted: a Float32 tensor is checked entirely in
Float32 and the answer comes back Float32.

HOST ARRAYS ONLY.  The blocked accumulation is `mul!` with `β = 1`, which the
device path of `_qdn_ttm!` cannot do; a device tensor is checked by
`_qdn_verify`'s own slice route instead.

If the operators cannot be matched to axes by index (anything other than one
two-index operator carrying each frame index) the ITensor form falls back to
`applyDerivation` rather than guess.
"""
function der_residual(Γ::ITensor, D::Vector{ITensor}, chisel;
                      block_bytes::Integer = 2^28)
    P = chisel isa Chisel ? chisel.ch : chisel
    fr = collect(inds(Γ))
    n = length(fr)
    G = ITensors.array(Γ, fr...)
    T = eltype(G)
    Ms = Vector{Matrix{T}}(undef, n)
    ok = length(D) == n
    if ok
        for a in 1:n
            hits = filter(t -> hasind(t, fr[a]), D)
            others = length(hits) == 1 ?
                     filter(i -> i != fr[a], collect(inds(only(hits)))) : Index[]
            if length(hits) != 1 || length(others) != 1
                ok = false
                break
            end
            # The frame index FIRST, which is the index `_qdn_ttm` contracts and
            # the convention `applyDerivation` and `embedITensors` share.
            Ms[a] = Matrix{T}(Array(only(hits), fr[a], only(others)))
        end
    end
    if !ok
        C = chisel isa Chisel ? chisel : Chisel(P, fr)
        R = applyDerivation(Γ, D, C)
        return norm(R) / max(norm(Γ) * maximum(norm.(D)), eps())
    end
    return der_residual(G, Ms, P; block_bytes = block_bytes)
end

function der_residual(G::AbstractArray{T,N}, Ms::AbstractVector{<:AbstractMatrix},
                      P::AbstractMatrix; block_bytes::Integer = 2^28) where {T,N}
    RT = real(T)
    sq = der_residual_squares(G, Ms, P; block_bytes = block_bytes)
    scale = norm(G) * maximum(a -> norm(Ms[a]), 1:N)
    return sqrt(sum(sq)) / max(scale, eps(RT))
end

"""
    der_residual_squares(G, Ms, P; block_bytes = 2^28) -> Vector

`‖Σ_a P[ρ,a]·(G ×_a M_a)‖²`, one entry per ROW of the chisel, unnormalised --
the raw material of `der_residual` and of `_qdn_verify`'s `:full` branch, which
wants the rows separately because it tests each against a bound rather than
folding them into one number.

The blocking is what `der_residual` documents; the working set is two blocks of
`block_bytes` in `G`'s own element type, whatever the size of `G`.  With a
one-row chisel -- every case in `bench/` -- the accumulation order is exactly
the single running sum the benchmark used, so the reported digits do not move;
with several rows the per-row sums are the same quantity summed in a different
order.
"""
function der_residual_squares(G::AbstractArray{T,N}, Ms::AbstractVector{<:AbstractMatrix},
                              P::AbstractMatrix;
                              block_bytes::Integer = 2^28) where {T,N}
    length(Ms) == N || error("der_residual: $(length(Ms)) operators for a valence-$N " *
                             "tensor; one per axis is needed.")
    size(P, 2) == N || error("der_residual: the chisel has $(size(P,2)) columns but " *
                             "the tensor has $N axes.")
    Pm = Matrix{T}(P)
    RT = real(T)
    dims = size(G)
    ca = argmax(collect(dims))                 # block along the LONGEST axis
    per = max(length(G) ÷ dims[ca], 1)         # entries per unit of that axis
    step = clamp(fld(Int(block_bytes), sizeof(T) * per), 1, dims[ca])
    acc = zeros(RT, size(Pm, 1))
    # The two buffers are allocated ONCE, not once per block, so the bytes this
    # function asks for are bounded by `block_bytes` and do not grow with the
    # tensor -- which is the whole claim.  Both are overwritten in full at the
    # top of every block (`fill!` and `copyto!`), so reuse changes no arithmetic.
    # The last block is short and gets its own pair.
    Efull = Array{T}(undef, ntuple(i -> i == ca ? step : dims[i], N))
    Gfull = Array{T}(undef, size(Efull))
    for lo in 1:step:dims[ca]
        hi = min(dims[ca], lo + step - 1)
        w = hi - lo + 1
        blk = ntuple(i -> i == ca ? (lo:hi) : Colon(), N)
        Eb = w == step ? Efull : Array{T}(undef, ntuple(i -> i == ca ? w : dims[i], N))
        Gb = w == step ? Gfull : Array{T}(undef, size(Eb))
        copyto!(Gb, view(G, blk...))
        for rho in axes(Pm, 1)
            fill!(Eb, zero(T))
            for a in 1:N
                c = Pm[rho, a]
                iszero(c) && continue
                if a == ca
                    # This block of the output is the whole tensor against the
                    # selected COLUMNS of the operator on the blocked axis.
                    _qdn_ttm!(Eb, G, view(Ms[a], :, lo:hi), a, c, one(T))
                else
                    _qdn_ttm!(Eb, Gb, Ms[a], a, c, one(T))
                end
            end
            acc[rho] += sum(abs2, Eb)
        end
    end
    return acc
end
