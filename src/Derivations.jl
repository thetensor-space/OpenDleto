
#
# Strata Dleto: Sylvester Solvers
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

"""
    Derivations & Densors.

    Interface to methods for solving Sylvester equations arising in chiseling.

"""
module Derivations

export der, den, stratify
# using SylverLining as SL
using ITensors
# using LinearMaps
# using ..Chisels: LinearChisel
# using ..TransverseOperators: TransverseOps


# struct Derivation 
#     C :: LinearChisel
#     Ω :: TransverseOps
#     op :: AbstractVector{AbstractMatrix}
# end

"""
    `DerivationMethod`

    An interface for computing derivations.  Derivation methods 
    should inherit `DerivationMethod` and implement `der`.
"""
abstract type DerivationMethod end

#---- Convenience Functions -------------------------------------------------------
function der(ops::TransverseOps, ch::AbstractMatrix, Γ::ITensor)
    return der(SylverLining(), ops, chisel(ch, inds(Γ)), Γ)
end
function der(ops::TransverseOps, ch::AbstractMatrix, Γ::AbstractArray)
    frame = [Index(size(Γ, i), "a$i") for i in 1:ndims(Γ)]
    Σ = ITensor(Γ, frame...)
    return der( ops, ch, Σ)
end
function der(ch::AbstractMatrix, Γ)
    ops = UniversalOps(frame)
    return der(ops, ch, Γ)
end
function der(Γ)
    ch = UniversalChisel(ndims(Γ))
    return der(ch, Γ)
end



"""
    der(method::DerivationMethod, 
            Ω::TransverseOps, 
            P::LinearChisel, 
            Γ::ITensor; 
            nd::Integer=10,
            tol::Real=1e-6,
        ) :: Vector{ITensor}

    Computes up to `nd` many `P`-derivations of `Γ` for the to the given chisel `P` and transverse operators `Ω`.
    If `nd` is negative or exceeds the dimension of the derivation space
    then the a basis for the derivation space is returned.
    - `method`: An instance of a subtype of `DerivationMethod` defining the solving method.
    - `Ω`: The transverse operators.
    - `P`: a linear chisel
    - `Γ`: The input tensor
    - `nd`: (optional) Maximum number of singular vectors to compute (default: 10). If negative, 
    - `tol`: (optional) Tolerance for the solver (default: 1e-6).
    attempt to compute a basis for the derivation space.

    Returns a vector of derivations as `ITensor`s with `a` axis labelled by `(a,a')`.
"""
function der(method::DerivationMethod,
    Ω::TransverseOps, 
    P::AbstractMatrix, 
    Γ::AbstractArray; 
    nd::Integer=10,
    tol::Real=1e-6,
    ) :: Vector{ITensor} end

"""
    den(method::DerivationMethod, 
            Ω::TransverseOps, 
            P::LinearChisel, 
            Δ ::Vector{ITensor};
            nd::Integer=10,
            tol::Real=1e-6,
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
    nd::Integer=10,
    tol::Real=1e-6,
    ) :: Vector{ITensor} end


#---------------- Generic Derivation Densor Functions -------------------------

struct SylvesterDerivationMethod <: DerivationMethod end


"""
    stratify(Γ::AbstractArray, der::Vector{ITensor}) 
    :: NamedTuple{(:tensor, :transform), Tuple{AbstractArray, TransverseOps}}

    Stratify the tensor Γ using the spall and the specified positions.
    Returns a named tuple with the sculpted tensor and the transforms used.

    - `Γ`: The input tensor
    - `der` a derivation to specify strata

    Returns a named tuple with fields:
    - `Σ` The sculpted tensor
    - `T` a transverse operator 
"""
function stratify(Γ::ITensor, der::Vector{ITensor}) 
    :: NamedTuple{(:tensor, :transform), Tuple{ITensor, Vector{ITensor}}}
    mats = map( M -> blockdiag(M).T, der )
    dims = [size(Γ, i) for i in 1:ndims(Γ)]
    T = contains(InvertibleOps(cat, dims), mats)
    Σ = ITensor(inds(Γ)...)
    for i in 1:ndims(Γ)
        setprime!(Σ, 1; plev=inds(Γ)[i])
        Σ = Σ * T[i]
        noprime!(Σ; plev=inds(Γ)[i])
    end
    return (;Σ, T)
end

#---------------- Internal Functions -------------------------------------

"""
    Group conjugate pairs of complex eigenvalues and eigenvectors into real blocks.

    Returns a named tuple with:
    - `D`: A block diagonal matrix with real blocks.
    - `T`: A matrix whose columns are the real eigenvectors or the real 
    and imaginary parts of complex conjugate pairs.

    Law
    ```julia
    res = blockdiag(M); isapprox(M * res.T, res.T * res.D)
    ```
"""
function blockdiag(M::Matrix)
    res = LinearAlgebra.eigen(M)
    eigenvals = res.values
    eigenvecs = res.vectors
    n = length(eigenvals)
    real_diag = zeros(real(eltype(eigenvals)), n,n)
    real_blocks = zeros(real(eltype(eigenvecs)), n, n)
    
    processed = falses(n)
    for i in 1:n
        if processed[i]
            continue
        end
        
        λ = eigenvals[i]
        if isreal(λ)
            # Real eigenvalue
            real_diag[i,i] = real(λ)
            real_blocks[:, i] = real(eigenvecs[:, i])
            processed[i] = true
        else
            j = i+findfirst(j -> !processed[j] && isapprox(eigenvals[j], conj(λ)), (i+1):n)
            if j !== nothing
                # Create real 2D subspace from conjugate pair
                v = eigenvecs[:, i]
                # Fill in the real block diagonal entries
                real_diag[i,i] = real(λ);  real_diag[i,j] = imag(λ);
                real_diag[j,i] = -imag(λ);  real_diag[j,j] = real(λ);
                # Fill in the real block eigenvectors
                real_blocks[:, i] = real(v)
                real_blocks[:, j] = imag(v)
                processed[i] = processed[j] = true
            end
        end
    end
    return (;D = real_diag, T = real_blocks)
end


end # module SylvesterSolvers