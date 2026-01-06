
#
# Strata Dleto: AbstractDerivationMethods
#   Algorithms for solving Sylvester equations arising in chiseling.
# -----------------------------------------------------------------------------
# Copyright 2022-2025 Peter A. Brooksbank, Martin D. Kassabov, James B. Wilson
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
    der(method::DerivationMethod, 
            Ω::TransverseOps, 
            P::LinearChisel, 
            Γ::ITensor; 
            nd::Integer=10,
            tol::Real=1e-6,
        ) :: Vector{Vector{ITensor}}

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
    nd=10,
    ) :: Vector{Vector{ITensor}} 
    @assert false "Calling Placeholder Abstract Function"
end;
# I think this should be renamed to compute derivations

function der(
    Ω::TransverseOps, 
    P::AbstractMatrix, 
    Γ::ITensor; 
    tol::Float64=1e-6,
    nd=10,
    ) :: Vector{Vector{ITensor}} 
    method = SylverLiningMethod()
    return der(method, Ω, P, Γ; tol=tol, nd=nd)
end;

"""
    den(method::DerivationMethod, 
            Ω::TransverseOps, 
            P::LinearChisel, 
            Δ ::Vector{Vector{ITensor}};
            nd::Integer=10,
            tol::Real=1e-6
        ) :: Vector{ITensor}

    Computes up to `nd` many tensors with `P`-derivations of `Γ` for the to the given chisel `P` and transverse operators `Ω`.
    If `nd` is negative or exceeds the dimension of the derivation space
    then the a basis for the derivation space is returned.
    - `method`: An instance of a subtype of `DerivationMethod` defining the solving method.
    - `Ω`: The transverse operators.
    - `P`: a linear chisel
    - `D`: derivations The input tensor
    - `nd`: (optional) Maximum number of singular vectors to compute (default: 10)
    - `tol`: (optional) Tolerance for the solver (default: 1e-6).
    
    Returns a vector `nd` many tensors that admit `D` as derivations.
"""
function den(method::DerivationMethod,
    Ω::TransverseOps, 
    P::AbstractMatrix, 
    der::Vector{ITensor}; 
    nd=10,
    tol=1e-6
    ) :: Vector{Vector{ITensor}} 
    @assert false "Calling Placeholder Abstract Function"
end;


#---- Convenience Functions -------------------------------------------------------

"""
    der(Γ; nd=10, tol::Real=1e-6)

    Convenience method to compute derivations of a tensor using defaults:
    - Universal chisel
    - Universal transverse operators
    - SylverLiningMethod
"""
function der(Γ::ITensor; nd=10, tol::Real=1e-6)
    ch = UniversalChisel(length(inds(Γ)))
    fr = collect(inds(Γ))
    ops = IndTransverseOps(fr, UniversalOp())
    return der(SylverLiningMethod(), ops, ch, Γ; nd=nd, tol=tol)
end

function der(Γ::AbstractArray; nd=10, tol::Real=1e-6)
    fr = [Index(size(Γ, i), "a_$i") for i in 1:ndims(Γ)]
    Σ = ITensor(Γ, fr...)
    return der(Σ; nd=nd, tol=tol)
end

function der(ch::AbstractMatrix, Γ::ITensor; nd=10, tol::Real=1e-6)
    fr = collect(inds(Γ))
    ops = IndTransverseOps(fr, UniversalOp())
    return der(SylverLiningMethod(), ops, ch, Γ; nd=nd, tol=tol)
end

function der(ch::AbstractMatrix, Γ::AbstractArray; nd=10, tol::Real=1e-6)
    fr = [Index(size(Γ, i), "a_$i") for i in 1:ndims(Γ)]
    Σ = ITensor(Γ, fr...)
    return der(ch, Σ; nd=nd, tol=tol)
end

