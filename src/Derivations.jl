
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
    if method === :SylverLining
        return SylverLiningMethod(; kwargs...)
    elseif method === :QuickDer
        return QuickDerMethod(; kwargs...)
    elseif method === :QuickDer3 || method === :FastDer3Valent
        return FastDer3ValentMethod(; kwargs...)
    elseif method === :QuickSylver
        return QuickSylverMethod(; kwargs...)
    end
    error("Unknown derivation method symbol: $method. " *
          "Known: :SylverLining, :QuickDer, :QuickDer3 (alias :FastDer3Valent), " *
          ":QuickSylver.")
end

"""
    der(method::DerivationMethod,
            Ω::TransverseOps,
            P::AbstractMatrix,
            Γ::ITensor;
            nd=-1,
            tol::Real=1e-6,
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
    tol::Float64=1e-6,
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
    tol::Float64=1e-6,
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
    tol::Float64=1e-6,
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
    tol::Float64=1e-6,
    nd=10,
    kwargs...
    ) :: Tuple{TransverseOps, LinearMaps.LinearMap, AbstractMatrix{<: Number} }
    @assert false "Calling Placeholder Abstract Function"
end


# this should be  moved to DerivationMethodSylverLininig,jl  
derTrOpsReduced( Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor; tol::Float64=1e-6, nd=10) :: Tuple{TransverseOps, LinearMaps.LinearMap, AbstractMatrix{<: Number}} = 
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
    der(Γ; nd=-1, tol::Real=1e-6)

    Convenience method to compute derivations of a tensor using defaults:
    - Universal chisel
    - Universal transverse operators
    - SylverLiningMethod
"""
function der(Γ::ITensor; nd=-1, tol::Real=1e-6) :: Vector{Vector{ITensor}}
    ch, _, ops = universalSetup(Γ)
    return der(SylverLiningMethod(), ops, ch, Γ; nd=nd, tol=tol)
end

# NOTE on the two kinds of keyword: `kwargs...` here is forwarded to the
# *method constructor* (`solver=` and friends), while `progress` is a
# *per-call* option consumed by the solve.  They were conflated -- everything
# went to the constructor -- so `der(:SylverLining, Γ; progress=:solve)` died
# with "SylverLiningMethod does not support keyword progress".  Per-call
# options therefore have to be named explicitly in every symbol overload.
function der(method::Symbol, Γ::ITensor; nd=-1, tol::Real=1e-6,
             progress=false, kwargs...) :: Vector{Vector{ITensor}}
    ch, _, ops = universalSetup(Γ)
    return der(get_derivation_method(method; kwargs...), ops, ch, Γ;
               nd=nd, tol=tol, progress=progress)
end

der(Γ::AbstractArray; nd=-1, tol::Real=1e-6) =
    der(__asITensor(Γ); nd=nd, tol=tol)

der(method::Symbol, Γ::AbstractArray; nd=-1, tol::Real=1e-6, progress=false, kwargs...) =
    der(method, __asITensor(Γ); nd=nd, tol=tol, progress=progress, kwargs...)

function der(ch::AbstractMatrix, Γ::ITensor; nd=-1, tol::Real=1e-6)
    fr = collect(inds(Γ))
    ops = IndTransverseOps(fr, UniversalOp())
    return der(SylverLiningMethod(), ops, ch, Γ; nd=nd, tol=tol)
end

function der(method::Symbol, ch::AbstractMatrix, Γ::ITensor; nd=-1, tol::Real=1e-6,
             progress=false, kwargs...)
    fr = collect(inds(Γ))
    ops = IndTransverseOps(fr, UniversalOp())
    return der(get_derivation_method(method; kwargs...), ops, ch, Γ;
               nd=nd, tol=tol, progress=progress)
end

der(ch::AbstractMatrix, Γ::AbstractArray; nd=-1, tol::Real=1e-6) =
    der(ch, __asITensor(Γ); nd=nd, tol=tol)

function der(Ω::TransverseOps, ch::AbstractMatrix, Γ::ITensor;
             nd=-1, tol::Real=1e-6, progress=false,
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
    nd=-1, tol::Real=1e-6, progress=false, kwargs...) =
    der(get_derivation_method(method; kwargs...), Ω, ch, Γ;
        nd=nd, tol=tol, progress=progress)

