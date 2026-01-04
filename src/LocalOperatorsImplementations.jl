#
# Strata Dleto: Local Operators Implementations
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
    LocalOperators Implementations

    Provides standard local operators like -- all_matrices, symmetric_matrices and diagonal_matrices
"""

# working with matrices of size zero does not work. Usually the result of the function is [] but this is of type Vector{Any} which is not Vector{<:Number}

"""
    All Matrices
"""
struct LocalUniversalOps <: LocalOps end; 

"""
    Diagonal Matrices
"""
struct LocalDiagonalOps <: LocalOps end;

"""
    Symmetric Matrices
"""
struct LocalSymmetricOps <: LocalOps end; 

"""
    Anti Symmetric Matrices
"""
struct LocalAntiSymmetricOps <: LocalOps end; 

"""
    Scalar Matrices
"""
struct LocalScalarOps <: LocalOps end; 
# this gives errors when combined with symmetries, no idea why

"""
    Empty Matrices
"""
struct LocalEmptyOps <: LocalOps end; 
# this gives errors when combined with symmetries, no idea why


export LocalUniversalOps, LocalDiagonalOps, LocalSymmetricOps, LocalAntiSymmetricOps, LocalScalarOps, LocalEmptyOps

#coordinates

coordinates(LΩ::LocalUniversalOps, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing} =
    size(M)[1]==size(M)[2] ? reshape(M,length(M)) : nothing;

function coordinates(LΩ::LocalDiagonalOps, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    (sizes[1]==sizes[2]) || return nothing
    if (all(__isapproxzero, vcat([M[1:(i-1),i] for i=1:sizes[1]]...)) && 
        all(__isapproxzero, vcat([M[(i+1):sizes[1],i] for i=1:(sizes[1]-1)]...) ))
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

function coordinates(LΩ::LocalScalarOps, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    dim=sizes[1]
    (sizes[1]==sizes[2]) || return nothing
    # dim==0 && return zeros(0)
    (all(__isapproxzero, vcat([M[1:i-1,i] for i=1:sizes[1]]...)) && all(__isapproxzero, vcat([M[i+1:dim,i] for i=1:sizes[1]]...))) || return nothing
    all(__isapproxzero,[ M[i,i] - M[1,1] for i = 2:dim]) || return nothing
    return [M[1,1]]
end;

function coordinates(LΩ::LocalEmptyOps, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    dim=sizes[1]
    (sizes[1]==sizes[2]) || return nothing
    all(__isapproxzero, vcat([M[1:dim] for i=1:sizes[1]]...)) || return nothing
    return zeros(0)
end;


#unsafe_coordinates

unsafe_coordinates(LΩ::LocalUniversalOps, M::AbstractMatrix) :: Vector{<:Number} = reshape(M,length(M));

unsafe_coordinates(LΩ::LocalDiagonalOps, M::AbstractMatrix) :: Vector{<:Number} = [M[i,i] for i=1:size(M)[1]];

unsafe_coordinates(LΩ::LocalSymmetricOps, M::AbstractMatrix) :: Vector{<:Number} = vcat([M[1:i,i] for i=1:size(M)[1]]...);

unsafe_coordinates(LΩ::LocalAntiSymmetricOps, M::AbstractMatrix) :: Vector{<:Number} = vcat([M[1:(i-1),i] for i=1:size(M)[1]]...);

# unsafe_coordinates(LΩ::LocalScalarOps, M::AbstractMatrix) :: Vector{<:Number} =  size(M)[1] ==0 ? zeros(0) : [M[1,1]];
unsafe_coordinates(LΩ::LocalScalarOps, M::AbstractMatrix) :: Vector{<:Number} =  [M[1,1]];

unsafe_coordinates(LΩ::LocalEmptyOps, M::AbstractMatrix) :: Vector{<:Number} =  zeros(0);


#unsafe_transposeEmbedding

unsafe_transposeEmbedding(LΩ::LocalUniversalOps, M::AbstractMatrix) :: Vector{<:Number} = reshape(M,length(M));

unsafe_transposeEmbedding(LΩ::LocalDiagonalOps, M::AbstractMatrix) :: Vector{<:Number} = [M[i,i] for i=1:size(M)[1]];

unsafe_transposeEmbedding(LΩ::LocalSymmetricOps, M::AbstractMatrix) :: Vector{<:Number} = vcat( [vcat( M[1:(i-1),i] + M[i,1:(i-1)],M[i,i]) for i=1:size(M)[1]]...);

unsafe_transposeEmbedding(LΩ::LocalAntiSymmetricOps, M::AbstractMatrix) :: Vector{<:Number} = vcat( [M[1:(i-1),i] - M[i,1:(i-1)] for i=1:size(M)[1]]...);

# unsafe_transposeEmbedding(LΩ::LocalScalarOps, M::AbstractMatrix) :: Vector{<:Number} = size(M)[1] ==0 ? zeros(0) : [sum([M[i,i] for i=1:size(M)[1]]) ];
unsafe_transposeEmbedding(LΩ::LocalScalarOps, M::AbstractMatrix) :: Vector{<:Number} = [sum([M[i,i] for i=1:size(M)[1]]) ];

unsafe_transposeEmbedding(LΩ::LocalEmptyOps, M::AbstractMatrix) :: Vector{<:Number} = zeros(0);

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

# unsafe_embedding(LΩ::LocalScalarOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix = dim==0 ? Matrix(LinearAlgebra.I,0,0) : Matrix(data[1]*LinearAlgebra.I,dim,dim); 
unsafe_embedding(LΩ::LocalScalarOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix = Matrix(data[1]*LinearAlgebra.I,dim,dim); 

unsafe_embedding(LΩ::LocalEmptyOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix = Matrix(0.0*LinearAlgebra.I,dim,dim); 


#local Dimension
localDim(::LocalUniversalOps, dim::Integer) = dim*dim; 
localDim(::LocalDiagonalOps, dim::Integer) = dim; 
localDim(::LocalSymmetricOps, dim::Integer) = dim*(dim+1) ÷ 2; 
localDim(::LocalAntiSymmetricOps, dim::Integer) = dim*(dim-1) ÷ 2; 
# localDim(::LocalScalarOps, dim::Integer) = dim==0 ? 0 : 1; 
localDim(::LocalScalarOps, dim::Integer) = 1; 
localDim(::LocalEmptyOps, dim::Integer) = 0; 

#contains Scalars
containScalars(::LocalUniversalOps)= true;
containScalars(::LocalDiagonalOps) = true; 
containScalars(::LocalSymmetricOps) = true; 
containScalars(::LocalAntiSymmetricOps) = false; 
containScalars(::LocalScalarOps) = true; 
containScalars(::LocalEmptyOps) = false; 
