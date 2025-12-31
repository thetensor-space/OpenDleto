#
# Strata Dleto: LocalTransverse Operators
#   Creation and application of transverse operators for tensors.
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
abstract type LocalTransverseOps  end


"""
    Return the native encoding of an operator in the transverse set,
    or `Nothing` if it is not a member.
"""
function coordinates(LΩ::LocalTransverseOps, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing} end;
function unsafe_coordinates(LΩ::LocalTransverseOps, M::AbstractMatrix) :: Vector{<: Number} end;
# this is the inverse of the embedding map

function transposeEmbedding(LΩ::LocalTransverseOps, M::AbstractMatrix) :: Vector{<:Number}
    @assert size(M)[1]==size(M)[2]
    return unsafe_transposeEmbedding(LΩ, M)  
end;

function unsafe_transposeEmbedding(LΩ::LocalTransverseOps, M::AbstractMatrix) :: Vector{<:Number} end;
# this is the transpose of the embedding map

"""
    Convert the native encoding of an operator into matrix.
"""
function embedding(LΩ::TransverseOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  
    @assert length(data)== localDim(LΩ,dim) "Incompatable Data"
    unsafe_embedding(LΩ,dim,data)
end;
function unsafe_embedding(LΩ::TransverseOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  end



function localDim(LΩ::LocalTransverseOps, dim::Integer)::Integer end
function containScalars(LΩ::LocalTransverseOps)::Bool  end



struct LocalUniversalOps <: LocalTransverseOps end 
struct LocalDiagonalOps <: LocalTransverseOps end 
struct LocalSymmetricOps <: LocalTransverseOps end 
struct LocalAnitSymmetricOps <: LocalTransverseOps end 

#coordinates
function coordinates(LΩ::LocalUniversalOps, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    if size(M)[1]==size(M)[2]
        return reshape(M,length(M))
    else
        return Nothing
    end 
end;

function coordinates(LΩ::LocalDiagonalOps, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    if sizes[1]!=sizes[2]
        return Nothing
    end
    for i = 1: sizes[1]
        for j =(i+1): sizes[1]
            if M[i,j] != 0
                return Nothing
            end
        end
    end 
    return [M[i,i] for i=1:size(M)[1]]
end;


function coordinates(LΩ::LocalSymmetricOps, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    dim=sizes[1]
    k = 1
    if sizes[1]!=sizes[2]
        return Nothing
    end
    k = 1
    A=zeros(eltype(M),localDim(LΩ,dim))
    for i = 1:dim
        A[k] = M[i,i]
        k = k+1
        for j = (i+1):dim
            if M[i,j] !=  M[j,i]
                return Nothing
            end
            A[k] = M[i,j]
            k = k + 1
        end
    end 
    return A
end;

function coordinates(LΩ::LocalAnitSymmetricOps, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    dim=sizes[1]
    k = 1
    if sizes[1]!=sizes[2]
        return Nothing
    end
    k = 1
    A=zeros(eltype(M),localDim(LΩ,dim))
    for i = 1:dim
        if M[i,i]!=0
            return Nothing
        end
        for j = (i+1):dim
            if (M[i,j] +  M[j,i])!=
                return Nothing
            end
            A[k] = M[i,j]
            k = k + 1
        end
    end 
    return A
end;


#unsafe_coordinates
unsafe_coordinates(LΩ::LocalUniversalOps, M::AbstractMatrix) :: Vector{<:Number} = reshape(M,length(M));
unsafe_coordinates(LΩ::LocalDiagonalOps, M::AbstractMatrix) :: Vector{<:Number} = [M[i,i] for i=1:size(M)[1]]

function unsafe_coordinates(LΩ::LocalSymmetricOps, M::AbstractMatrix) :: Vector{<:Number}
    dim=size(M)[1]
    A=zeros(eltype(M),localDim(LΩ,dim))
    k = 1
    for i = 1:dim
        A[k] = M[i,i]
        k = k+1
        for j = (i+1):dim
            A[k] = (M[i,j] + M[j,i])/2
            k = k + 1
        end
    end 
    return A
end;

function unsafe_coordinates(LΩ::LocalAnitSymmetricOps, M::AbstractMatrix) :: Vector{<:Number}
    dim=size(M)[1]
    A=zeros(eltype(M),localDim(LΩ,dim))
    k = 1
    for i = 1:dim
        for j = (i+1):dim
            A[k] = M[i,j] - M[j,i]
            k = k + 1
        end
    end 
    return A
end;


#unsafe_transposeEmbedding
unsafe_transposeEmbedding(LΩ::LocalUniversalOps, M::AbstractMatrix) :: Vector{<:Number} = reshape(M,length(M))

unsafe_transposeEmbedding(LΩ::LocalDiagonalOps, M::AbstractMatrix) :: Vector{<:Number} = [M[i,i] for i=1:size(M)[1]]

function unsafe_transposeEmbedding(LΩ::LocalSymmetricOps, M::AbstractMatrix) :: Vector{<:Number}
    dim=size(M)[1]
    A=zeros(eltype(M),localDim(LΩ,dim))
    k = 1
    for i = 1:dim
        A[k] = M[i,i]
        k = k+1
        for j = (i+1):dim
            A[k] = M[i,j] + M[j,i]
            k = k + 1
        end
    end 
    return A
end;

function unsafe_transposeEmbedding(LΩ::LocalAnitSymmetricOps, M::AbstractMatrix) :: Vector{<:Number}
    dim=size(M)[1]
    A=zeros(eltype(M),localDim(LΩ,dim))
    k = 1
    for i = 1:dim
        for j = (i+1):dim
            A[k] = M[i,j] - M[j,i]
            k = k + 1
        end
    end 
    return A
end;

#unsafe_embedding
unsafe_embedding(LΩ::LocalUniversalOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrixMatrix = reshape(data,dim,dim);

unsafe_embedding(LΩ::LocalDiagonalOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix =  LinearAlgebra.Diagonal(data);

function unsafe_embedding(LΩ::LocalSymmetricOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  
    A=zeros(eltype(data),dim,dim)
    k = 1
    for i = 1:dim
        for j = i:dim
            A[i,j] = data[k]
            A[j,i] = data[k]
            k = k + 1
        end
    end 
    return A
end;

function unsafe_embedding(LΩ::LocalAnitSymmetricOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  
    A=zeros(eltype(data),dim,dim)
    k = 1
    for i = 1:dim
        for j = (i+1):dim
            A[i,j] = data[k]
            A[j,i] = - data[k]
            k = k + 1
        end
    end 
    return A
end;

#local Dimension
localDim(::LocalUniversalOps, dim::Integer) = dim*dim; 
localDim(::LocalDiagonalOps, dim::Integer) = dim; 
localDim(::LocalSymmetricOps, dim::Integer) = dim*(dim+1) ÷ 2; 
localDim(::LocalAnitSymmetricOps, dim::Integer) = dim*(dim-1) ÷ 2; 

#contains Scalars
containScalars(::LocalUniversalOps)= true;
containScalars(::LocalDiagonalOps) = true; 
containScalars(::LocalSymmetricOps) = true; 
containScalars(::LocalAnitSymmetricOps) = false; 


