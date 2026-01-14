#
# Strata Dleto: Utils
#   Extending ITensors multiplication to AbstractArrays for convenience
#
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
#-----------------------------------------------------------------------------
using LinearAlgebra: I
using ITensors: ITensor, Index, inds, replaceinds, addtags, store, norm

"""
    Make ITensor out of AbstractArray without indices, 
    don't export this function is dangerous and makes 
    no promise to keep working in future versions of Dleto.jl.
    It is mainly here to conveniently fix a few places where 
    ITensors but 1691 is blocking proper use of AbstractArrays.
"""
function __ITensor(Γ::AbstractArray)::ITensor 
    frame = [ Index(size(Γ,a), "a$a") for a in 1:ndims(Γ) ]
    iΓ = try 
        ITensor(Γ, frame...) 
    catch e
        # a temporary fix for AbstractArray inputs like ReshapedArray 
        # which has a low-grade bug(?) in ITensors (Bug #1691)
        ITensor( Array(Γ), frame...)
    end
    return iΓ
end


"""
    match_idx[!](A::ITensor, a::Index,
               E::ITensor, e::Index, 
               tag::String
              )::NamedTuple{(:ae, :A, :E),Tuple{Index, ITensor, ITensor}}

    Given an ITensor `A` with index `a` and another ITensor `E` with index `e`,
    create a compound label `ae` and a shallow copy of both tensors with the index relabeled to `ae`.
    The function with `!` modifies the tensors in place.
""" 
function match_idx(
        A::ITensor, a::Index,
        E::ITensor, e::Index,
        tag::String="matched_idx"
    )::NamedTuple{(:ae, :A, :E),Tuple{Index, ITensor, ITensor}}
    @assert dim(a)==dim(e) "Indices must have the same dimension to be relabeled"

    ae = Index( dim(a), tag, tags(a))
    At = replaceind(A, a, ae )
    Et = replaceind(E, e, ae )
    return (;ae=ae,A= At, E= Et)
end
function match_idx!(
        A::ITensor, a::Index,
        E::ITensor, e::Index,
        tag::String="matched_idx"
    )::NamedTuple{(:ae, :A, :E),Tuple{Index, ITensor, ITensor}}
    ae = Index( dim(a), tag, tags(a))
    At = replaceind!(A, a, ae )
    Et = replaceind!(E, e, ae )
    return (;ae=ae,A= At, E= Et)
end


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


# Define ⊕ as a new operator (not extending Base since it doesn't exist there)
⊕(Γ::ITensor, Δ::ITensor) = begin
    first_frame = ITensors.inds(Γ)
    second_frame = ITensors.inds(Δ)
    Σ, fr = ITensors.directsum(Γ=>first_frame, Δ=>second_frame)
    return Σ
end

⊕(Γ::AbstractArray, Δ::AbstractArray) = begin
    iΓ = __ITensor(Γ)
    iΔ = __ITensor(Δ)
    return iΓ ⊕ iΔ
end


⊕(Γ::ITensor, Δ::AbstractArray) = begin
    iΔ = __ITensor(Δ)
    return Γ ⊕ iΔ
end
⊕(Γ::AbstractArray, Δ::ITensor) = begin
    iΓ = __ITensor(Γ)
    return iΓ ⊕ Δ
end

function Base.:+(Γ::ITensor, Δ::AbstractArray)
    iΔ = ITensor(Δ, inds(Γ)...)
    return Γ + iΔ
end

function Base.:+(Γ::AbstractArray, Δ::ITensor)
    iΓ = ITensor(Γ, inds(Δ)...)
    return iΓ + Δ
end

# --- Utiliity functions ---

__isapproxzero(x::Number)::Bool = isapprox(x,0.0);


# to be moved into Utils.jl
"""
    Real canonical form of a matrix
    Group conjugate pairs of complex eigenvalues and eigenvectors into real blocks.

    Returns a named tuple with:
    - `D`: A block diagonal matrix with real blocks.
    - `T`: A matrix whose columns are the real eigenvectors or the real 
    and imaginary parts of complex conjugate pairs.

    LawA
    ```julia
    res = realCanonicalForm(M); isapprox(M * res.T, res.T * res.D)
    ```
"""
function realCanonicalForm( M ::AbstractMatrix; tol::Float64=1e-10):: NamedTuple{(:D, :T), Tuple{AbstractMatrix,AbstractMatrix}}
    @assert size(M,1)==size(M,2) "Matrix must be square"
    if all( M .|> (x -> abs(x) < tol) )
        # M is zero matrix, return identityh transformation
        return (;D=zeros(eltype(M),size(M)), T = LinearAlgebra.Diagonal([ 1.0 for i = 1:size(M,1)]) )
    end
    eig = LinearAlgebra.eigen(M)
    evalues = eig.values
    evec = eig.vectors
    if isa(M, LinearAlgebra.Symmetric)              # no need to do anything if the matrix is symmetric
        return (; D= LinearAlgebra.Diagonal(evalues), T=evec) 
    end 
    found_complex=false
    n  = real.(evec)
    nn = real.(evec)
    D = zeros(eltype(n),size(M))
    D[1,1] = real(evalues[1])
    for i = 2: size(M,2)
        if ((((n[:,i] - n[:,i-1]) .|> x -> x*x) |> sum) > tol)
            # nn[:,i] =n[:,i]
            D[i,i] = real(evalues[i])
        else 
            nn[:,i] = imag.(evec[:,i])
            D[i,i] = real(evalues[i])
            D[i,i-1] = -imag(evalues[i])
            D[i-1,i] = imag(evalues[i])
            found_complex=true
        end
    end
    return (;D = found_complex ? D : LinearAlgebra.Diagonal(real.(evalues)) , T = nn)
end
