#
# Strata Dleto: Utils
#   ?????.
#
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
#-----------------------------------------------------------------------------

"""
    nondeg(Γ::ITensor, A::Vector{Index}; tol::Float64=1e-10)

    For each axis `a` in `A`, computes projection onto a subspace with kernel 
    radical of the tensor on that axis.

    - `Γ`: The input tensor.
    - `A`: A vector of indices of `Γ` to consider.
    - mode: Symbol, either `:trunc` (default) or `:Tucker`.
        - `:trunc`: Projects onto nondegenerate subspace.
        - `:full`: Performs Tucker decomposition without truncation.
    - `tol`: Tolerance for determining nondegeneracy (default: 1e-10).

    Returns a named tuple with fields:
    - `Δ`: The tensor obtained by contracting `Γ` with the nondegenerate bases.
    - `E`: A vector of matrices, where each matrix corresponds to the basis
           for the nondegenerate subspace of the respective index in `A`.

    Law: `Δ = Γ * E` 
"""
function nondeg(Γ::ITensor, 
                A::Vector{Index{T}}, 
                mode::Symbol=:trunc,
                tol::Float64=1e-10) where T
    fr = inds(Γ)
    # Sort fr from largest size to smallest
    fr = sort(fr; by=ITensors.dim, rev=true)
    Es = Vector{ITensor}(undef, length(A))
    Δ = Γ
    for (i, a) in enumerate(filter(x -> x in A, fr))
        a_comp = filter(e -> e != a, fr)
        U, S, V = ITensors.svd(Γ, a_comp)
        if mode == :full
            Es[i] = V
            Δ = Δ * Es[i]
            continue
        end
        # Extract diagonal singular values from S as a regular array
        S_vals = diag(Array(S, inds(S)...))
        nondeg_idx = findall(s -> s >= tol, S_vals)
# println("Nondegenerate indices for axis $(i): ", nondeg_idx)
        in_dim = ITensors.dim(a)
        if isempty(nondeg_idx)
            # Fully degenerate - return identity (edge case)
            e = addtags(a, "nondeg")
            Es[i] = ITensor(Matrix{Float64}(I, in_dim, in_dim), a, e)
        else
            # V is an ITensor with indices for the input space
            # Convert V to array and extract the nondegenerate columns
            r = inds(V, tags="Link,v" )
            V_arr = Array(V, r,a)  # a is original index, second is SVD index
            nondeg_basis = V_arr[:, nondeg_idx]  # Shape: in_dim × length(nondeg_idx)
            # Create index for the nondegenerate subspace
            a_nondeg = Index(length(nondeg_idx), "nondeg")
            a_nondeg = addtags(a_nondeg, tags(a))
            Es[i] = ITensor(nondeg_basis, a, a_nondeg)
            Δ = Δ * Es[i]
        end
    end
    return (;Δ = Δ, Es=Es)


        # # Get the other indices (all except a)
        # out_dim = Int(prod(ITensors.dim.(a_comp)))
        # in_dim = ITensors.dim(a)
        
        # # The linear function x_a ↦ < Γ | x_a > returning a flat vector
        # function __apply(x)
        #     X = ITensor(x, a)
        #     result = Γ * X  # Contracts over index a, result has other_inds
        #     # Convert to array and flatten
        #     return vec(Array(result, a_comp...))
        # end
        # function __coapply(y)
        #     X = ITensor(y, a_comp...)
        #     result = Γ * X  # Contracts over index a, result has other_inds
        #     # Convert to array and flatten
        #     return vec(Array(result, a_comp...))
        # end
        # L_a = LinearMaps.LinearMap(__apply, __coapply, out_dim, in_dim)

        # # Solve for image by taking largest singular values.
        # M = Matrix(L_a)  # Shape: out_dim × in_dim
        
        # # Use SVD to find the row space (nondegenerate subspace in input/domain)
        # # M = U * S * Vt, where V's columns are right singular vectors (in input space)
        # F = LinearAlgebra.svd(M)
        
        # Find singular values above tolerance - these correspond to nondegenerate directions
        # nondeg_idx = findall(s -> s >= tol, F.S)
        
    #     if isempty(nondeg_idx)
    #         # Fully degenerate - return identity (edge case)
    #         e = addtags(a, "nondeg")
    #         E[i] = ITensor(Matrix{Float64}(I, in_dim, in_dim), a, e)
    #     else
    #         # F.V is in_dim × min(in_dim, out_dim), columns are right singular vectors
    #         # F.Vt is min(in_dim, out_dim) × in_dim
    #         # We want V[:, nondeg_idx] which is F.Vt[nondeg_idx, :]'
    #         nondeg_basis = F.V[:, nondeg_idx]  # Shape: in_dim × length(nondeg_idx)
    #         # Create index for the nondegenerate subspace
    #         a_nondeg = Index(length(nondeg_idx), ",nondeg")
    #         a_nondeg = addtags(a_nondeg, tags(a))
    #         E[i] = ITensor(nondeg_basis, a, a_nondeg)
    #     end
    # end
    # return (;Δ = Γ*E, E=E)
end;
function nondeg(Γ::ITensor; mode::Symbol=:trunc, tol::Float64=1e-10)
    return nondeg(Γ, collect(inds(Γ)), mode, tol)
end;

function nondeg(Γ::AbstractArray, 
                A::Vector{Integer}, 
                tol::Float64=1e-10)
    iΓ = __ITensor(Γ)
    return nondeg(iΓ, [ ITensors.inds(iΓ)[a] for a in A ], tol)
end;

function nondeg(Γ::AbstractArray)
    return nondeg(Γ, collect(1:ndims(Γ)))
end;

"""
    Make ITensor out of AbstractArray without indices, 
    don't export this function is dangerous and makes 
    no promise to keep working in future versions of Dleto.jl.
    It is mainly here to conveniently fix a few places where 
    ITensors but 1691 is blocking proper use of AbstractArrays.
"""
function __ITensor(Γ::AbstractArray)::ITensor 
    frame = [ Index(size(Γ,a), "a$a") for a in 1:ndims(Γ) ]
    # a temporary fix for AbstractArray inputs like ReshapedArray which have a bug
    # in ITensors (Bug #1691)
    iΓ = typeof(Γ) <: Array ? ITensor(Γ, frame...) : ITensor( Array(Γ), frame...)
    return iΓ
end;


# -- Direct action by a vector of ITensors -----

function Base.:*(Γ::ITensor, X::Vector{ITensor})  
    Σ = Γ  
    for x in X
        Σ = Σ * x
    end
    return Σ
end

# --- Wrappers to promote AbstractArray to ITensor -----
# More specific method for matrices to avoid ambiguity with LinearAlgebra
function Base.:*(Γ::AbstractMatrix, X::Vector{ITensor}) 
    @assert length(X) == ndims(Γ) "Ambiguous frame matching: length of list of matrices must match array axes"
    fr = [ ITensors.inds(x)[1] for x in X ]
    iΓ = typeof(Γ) <: Array ? ITensor(Γ, fr...) : ITensor( Array(Γ), fr...)
    return iΓ * X
end

function Base.:*(Γ ::AbstractArray, X::Vector{ITensor}) 
    # detect as scalar multiplication
    # if length(X) == 1 && size(X[1]) == (1,1) 
    #     return store(X[1]) * Γ
    # end
    @assert length(X) == ndims(Γ) "Ambiguous frame matching: length of list of matrices must match array axes"
    # we need assert that dims are the same not only valances
    fr = [ ITensors.inds(x)[1] for x in X ]
    iΓ = typeof(Γ) <: Array ? ITensor(Γ, fr...) : ITensor( Array(Γ), fr...)
    return iΓ * X
end

function Base.:*(Γ::ITensor, X::Vector{<:AbstractMatrix}) 
    @assert length(X) == ndims(Γ) "Ambiguous frame matching: length of list of matrices must match ITensor axes"
    fr = inds(Γ)
    iX = [ ITensor( Array(X[i]), fr[i], __new_index_for_change_of_basis(fr[i]) ) for i in 1:length(X) ] 
    return Γ * iX
    # MDK This does not work if Γ = random_itensor(i,i',i'')!!!!!
end
function Base.:*(Γ::ITensor, X::Vector{T}) where T<:Number
    Δ = Γ
    for x in X
        Δ = Δ * x
    end
    return Δ
end

# Make actions Ambidextrous
function Base.:*(X::Vector{ITensor}, Γ::AbstractArray ) 
    return Γ * X
end

function Base.:*(X::Vector{ITensor}, Γ::ITensor ) 
    return Γ * X
end

function Base.:*(X::Vector{<:AbstractMatrix}, Γ::ITensor ) 
    return Γ * X
end

function Base.:*(X::Vector{T},Γ::ITensor ) where T<:Number
    return Γ * X
end




# --- Utiliity functions ---

__isapproxzero(x::Number)::Bool = isapprox(x,0.0);

