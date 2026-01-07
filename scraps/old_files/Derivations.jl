
#
# Strata Dleto: Sylvester Solvers
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

"""
    Derivations & Densors.

    Interface to methods for solving Sylvester equations arising in chiseling.

"""

# TBD: after labeling chisel columns as dictionaries, implement isder
# function isder(X::Vector{ITensor}, 
#     Ω::TransverseOps, 
#     P::AbstractMatrix, 
#     Γ::ITensor; 
#     tol::Real=1e-6,
#     ) :: Bool
#     Δ = ITensor(inds(Γ))
#     for X in Xs
#         Δ = PΓ*X
#     end
#     sylvester, ester = sylvesterLM(Ω, P, Γ)
#     vecX = vectorize(Ω, X)
#     res = sylvester * vecX
#     return norm(res) < tol
# end

"""
    `DerivationMethod`

    An interface for computing derivations.  Derivation methods 
    should inherit `DerivationMethod` and implement `der`.
"""
abstract type DerivationMethod end

#---- Convenience Functions -------------------------------------------------------
function der(Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor; nd::Integer=10, tol::Real=1e-6)
    return der(SylverLiningMethod(), Ω, P, Γ, nd, tol)
end
function der(Ω::TransverseOps, P::AbstractMatrix, Γ::AbstractArray; nd::Integer=10, tol::Real=1e-6)
    fr = [Index(size(Γ, i), "a_$i") for i in 1:ndims(Γ)]
    Σ = ITensor(Γ, fr...)
    return der(Ω, P, Σ; nd=nd, tol=tol)
end
function der(P::AbstractMatrix, Γ::AbstractArray; nd::Integer=10, tol::Real=1e-6)
    fr = [Index(size(Γ, i), "a_$i") for i in 1:ndims(Γ)]
    Σ = ITensor(Γ, fr...)
    Ω = UniversalOps(fr)
    return der(Ω, P, Σ; nd=nd, tol=tol)
end
function der(P::AbstractMatrix, Γ::ITensor; nd::Integer=10, tol::Real=1e-6)
    fr = collect(inds(Γ))
    Ω = UniversalOps(fr)
    return der(Ω, P, Γ; nd=nd, tol=tol)
end
function der(Γ; nd::Integer=-1, tol::Real=1e-6)
    P = UniversalChisel(ndims(Γ))
    return der(P, Γ; nd=nd, tol=tol)
end



"""
    der(method::DerivationMethod, 
            Ω::TransverseOps, 
            P::AbstractMatrix,
            Γ::ITensor; 
            nd::Integer=-1,
            tol::Real=1e-6,
        ) :: Vector{Vector{ITensor}}

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
    Γ::ITensor,
    nd::Integer=-1, 
    kwargs...,
    ) :: Vector{ITensor} end

struct SylverLiningMethod <: DerivationMethod end

function der(method::SylverLiningMethod,
    Ω::TransverseOps, 
    P::AbstractMatrix, 
    Γ::ITensor,
    nd::Integer=-1, 
    tol=1e-6,
    kwargs...,
    ) :: Vector{Vector{ITensor}}
    sylvester, ester = sylvesterLM(Ω, P, Γ)
    # if size(sylvester, 1) < 10000
        
    # M = Matrix(sylvester)
    # vals, vecs = eigen(M)
    vals, vecs = solve(sylvester, method.solver; nd=nd, tol=tol)

    vecs = vecs[:, findall(abs.(vals) .< tol)]
    if nd > 0 && size(vecs, 2) > nd
        vecs = vecs[:, 1:nd]
    end
    Xs = [ transverse(Ω, vecs[:,i]) for i in 1:size(vecs,2) ]
    return Xs
    # end
    # return nothing
end

"""
    den(method::DerivationMethod, 
            Ω::TransverseOps, 
            P::LinearChisel, 
            Δ ::Vector{ITensor};
            nd::Integer=-1,
            tol::Real=1e-6,
        ) :: Vector{Vector{ITensor}}

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
    nd::Integer=-1,
    tol::Real=1e-6,
    ) :: Vector{Vector{ITensor}} end


"""
    stratify(Γ::AbstractArray, der::Vector{ITensor}) 
    :: NamedTuple{(:Σ, :Xs), Tuple{AbstractArray, Vector{ITensor}}}

    Stratify the tensor Γ using the spall and the specified positions.
    Returns a named tuple with the sculpted tensor and the transforms used.

    - `Γ`: The input tensor
    - `der` a derivation to specify strata

    Returns a named tuple with fields:
    - `Σ` The sculpted tensor
    - `Xs` a transverse operator 
"""
function stratify(
        Γ::ITensor, 
        der::Vector{ITensor}
    )
    # Convert ITensors to Matrices before calling blockdiag
    Xs = Vector{ITensor}(undef, length(der))
    for i in 1:length(der)
        X = der[i]
        D, T = blockdiag(Array(X, inds(X)...))
        Xs[i] = ITensor(T, inds(X)...)
    end
    return (;Σ=Γ*Xs, Xs=Xs)
end

function stratify(
        Γ::ITensor
    )
    ders = der(Γ)
    if isempty(ders)
        # should never happen as there are always trivial derivations
        # so this indicates an error
        error("No derivations found for the given tensor, this indicates failure to converge in solvers, consider adjusting parameters.")
    end
    println("Found $(length(ders)) derivations for stratification.")
    # Select a random linear combination of derivations
    # Each derivation is a Vector{ITensor} with one ITensor per axis
    n_ders = length(ders)
    n_axes = ndims(Γ)
    coefs = [ randn(10) for _ in 1:n_ders ]
    
    # Initialize δ as zeros with same structure as ders[1]
    δ = Vector{ITensor}(undef, n_axes)
    for a in 1:n_axes
        # Sum: coefs[1]*ders[1][a] + coefs[2]*ders[2][a] + ...
        δ[a] = coefs[1] * ders[1][a]
        for d in 2:n_ders
            δ[a] += coefs[d] * ders[d][a]
        end
    end
    return stratify(Γ, δ)
end

function stratify(
        Γ::AbstractArray,
        der::Vector{ITensor}
    )
    # Convert array to ITensor and delegate
    return stratify(__ITensor(Γ), der)
end

function stratify(
        Γ::AbstractArray
    )
    return stratify(__ITensor(Γ))
end

function stratify(
        Γ::AbstractArray,
        der::Vector{ITensor}
    )
    # Convert array to ITensor and delegate
    return stratify(__ITensor(Γ), der)
end

function stratify(
        Γ::AbstractArray
    )
    ders = der(Γ)
    return stratify(Γ, ders[1])
end

#---------------- Generic Derivation Densor Functions -------------------------



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
function blockdiag(M::Matrix; tol::Float64=1e-10)
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
        # Check if eigenvalue is effectively real (imaginary part below tolerance)
        if abs(imag(λ)) < tol
            # Real eigenvalue
            real_diag[i,i] = real(λ)
            real_blocks[:, i] = real(eigenvecs[:, i])
            processed[i] = true
        else
            # Find the conjugate pair in remaining eigenvalues
            j = nothing
            for k in (i+1):n
                if !processed[k] && isapprox(eigenvals[k], conj(λ); atol=tol)
                    j = k
                    break
                end
            end
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
            else
                # No conjugate found - treat as real (shouldn't happen for real matrices)
                @warn "No conjugate pair found for eigenvalue $λ at index $i, treating as real"
                real_diag[i,i] = real(λ)
                real_blocks[:, i] = real(eigenvecs[:, i])
                processed[i] = true
            end
        end
    end
    return (;D = real_diag, T = real_blocks)
end


# end # module SylvesterSolvers