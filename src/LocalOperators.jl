#
# Strata Dleto: Local Operators
#   Creation and application of local operators for tensors.
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
    LocalOperators

    TO BE WRITTEN
"""

"""
    LocalOperators

    A subspace of operators between tensors in a tensor category,
    what is sometimes denoted hom(Γ,Υ), but with builtin functions 
    for compact encoding.

    The alignment of the axes is determined by the matching `Index` terms 
    of the individual tensors stored as `ITensor` terms.

"""
abstract type LocalOps  end


"""
    Return the native encoding of an operator in the transverse set,
    or `Nothing` if it is not a member.
"""
function coordinates(LΩ::LocalOps, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing} 
    @assert false "Calling Placeholder Abstract Function"
end;
function unsafe_coordinates(LΩ::LocalOps, M::AbstractMatrix) :: Vector{<: Number} 
    @assert false "Calling Placeholder Abstract Function"
end;
# this is the inverse of the embedding map

function transposeEmbedding(LΩ::LocalOps, M::AbstractMatrix) :: Vector{<:Number}
    @assert size(M)[1]==size(M)[2] "Incompatable Data"
    return unsafe_transposeEmbedding(LΩ, M)  
end;

function unsafe_transposeEmbedding(LΩ::LocalOps, M::AbstractMatrix) :: Vector{<:Number} 
    @assert false "Calling Placeholder Abstract Function"
end;
# this is the transpose of the embedding map

"""
    Convert the native encoding of an operator into matrix.
"""
function embedding(LΩ::LocalOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  
    @assert length(data)== localDim(LΩ,dim) "Incompatable Data"
    unsafe_embedding(LΩ,dim,data)
end;
function unsafe_embedding(LΩ::LocalOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  
    @assert false "Calling Placeholder Abstract Function"
end;



function localDim(LΩ::LocalOps, dim::Integer)::Integer 
    @assert false "Calling Placeholder Abstract Function"
end;

function containScalars(LΩ::LocalOps)::Bool  
    @assert false "Calling Placeholder Abstract Function"
end;

export LocalOps, coordinates, unsafe_coordinates, transposeEmbedding, unsafe_transposeEmbedding, embedding, unsafe_embedding, localDim, containScalars

struct LocalUniversalOps <: LocalOps end 
struct LocalDiagonalOps <: LocalOps end 
struct LocalSymmetricOps <: LocalOps end 
struct LocalAntiSymmetricOps <: LocalOps end 

#TODO
struct LocalScalarOps <: LocalOps end 
struct LocalEmptyOps <: LocalOps end 

export LocalUniversalOps, LocalDiagonalOps, LocalSymmetricOps, LocalAntiSymmetricOps, LocalScalarOps, LocalEmptyOps
#coordinates
coordinates(LΩ::LocalUniversalOps, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing} =
    size(M)[1]==size(M)[2] ? reshape(M,length(M)) : nothing;

function coordinates(LΩ::LocalDiagonalOps, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    (sizes[1]==sizes[2]) || return nothing
    if all(__isapproxzero, vcat([M[1:i-1,i] for i=1:sizes[1]]...))
        return [M[i,i] for i=1:size(M)[1]]
    else
        return nothing
    end
end;


function coordinates(LΩ::LocalSymmetricOps, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    dim=sizes[1]
    (sizes[1]==sizes[2]) || return nothing
    if all(__isapproxzero ,vcat([M[1:i-1,i] - M[i,1:i-1] for i=1:dim]...))
        return vcat([M[1:i,i] for i=1:dim]...)
    else 
        return nothing
    end
end;

function coordinates(LΩ::LocalAntiSymmetricOps, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    dim=sizes[1]
    (sizes[1]==sizes[2]) || return nothing
    if all(__isapproxzero ,vcat([M[1:i-1,i] + M[i,1:i-1] for i=1:dim]...))
        return vcat([M[1:i-1,i] for i=1:dim]...)
    else 
        return nothing
    end
end;


#unsafe_coordinates
unsafe_coordinates(LΩ::LocalUniversalOps, M::AbstractMatrix) :: Vector{<:Number} = reshape(M,length(M));

unsafe_coordinates(LΩ::LocalDiagonalOps, M::AbstractMatrix) :: Vector{<:Number} = [M[i,i] for i=1:size(M)[1]];

unsafe_coordinates(LΩ::LocalSymmetricOps, M::AbstractMatrix) :: Vector{<:Number} = vcat([M[1:i,i] for i=1:size(M)[1]]...);


unsafe_coordinates(LΩ::LocalAntiSymmetricOps, M::AbstractMatrix) :: Vector{<:Number} = vcat([M[1:(i-1),i] for i=1:size(M)[1]]...);


#unsafe_transposeEmbedding
unsafe_transposeEmbedding(LΩ::LocalUniversalOps, M::AbstractMatrix) :: Vector{<:Number} = reshape(M,length(M))

unsafe_transposeEmbedding(LΩ::LocalDiagonalOps, M::AbstractMatrix) :: Vector{<:Number} = [M[i,i] for i=1:size(M)[1]]

unsafe_transposeEmbedding(LΩ::LocalSymmetricOps, M::AbstractMatrix) :: Vector{<:Number} = vcat( [vcat( M[1:(i-1),i] + M[i,1:(i-1)],M[i,i]) for i=1:size(M)[1]]...);

unsafe_transposeEmbedding(LΩ::LocalAntiSymmetricOps, M::AbstractMatrix) :: Vector{<:Number} = vcat( [M[1:(i-1),i] - M[i,1:(i-1)] for i=1:size(M)[1]]...);

#unsafe_embedding
unsafe_embedding(LΩ::LocalUniversalOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix = reshape(data,dim,dim);

unsafe_embedding(LΩ::LocalDiagonalOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix =  LinearAlgebra.Diagonal(data);

function unsafe_embedding(LΩ::LocalSymmetricOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  
    A=zeros(eltype(data),dim,dim)
    k = 1
    for i = 1:dim
        A[1:i,i] = data[k:k+i-1]
        A[i,1:i] = data[k:k+i-1]
        k = k+i
    end 
    return LinearAlgebra.Symmetric(A)
end;

function unsafe_embedding(LΩ::LocalAntiSymmetricOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  
    A=zeros(eltype(data),dim,dim)
    k = 1
    for i = 1:dim
        A[1:(i-1),i] = data[k:k+i-2]
        A[i,1:(i-1)] = -data[k:k+i-2]
        k = k+i-1
    end 
    return A
end;

#local Dimension
localDim(::LocalUniversalOps, dim::Integer) = dim*dim; 
localDim(::LocalDiagonalOps, dim::Integer) = dim; 
localDim(::LocalSymmetricOps, dim::Integer) = dim*(dim+1) ÷ 2; 
localDim(::LocalAntiSymmetricOps, dim::Integer) = dim*(dim-1) ÷ 2; 

#contains Scalars
containScalars(::LocalUniversalOps)= true;
containScalars(::LocalDiagonalOps) = true; 
containScalars(::LocalSymmetricOps) = true; 
containScalars(::LocalAntiSymmetricOps) = false; 


