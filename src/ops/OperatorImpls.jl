#
# Strata Dleto: Local Operators Implementations
#   Creation and application of local operators for tensors.
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

"""
    LocalOperators Implementations

    Provides standard local operators like -- all_matrices, symmetric_matrices and diagonal_matrices
"""

# working with matrices of size zero does not work. Usually the result of the function is [] but this is of type Vector{Any} which is not Vector{<:Number}

"""
    All Matrices
"""
struct UniversalOp <: Operator end; 

"""
    Diagonal Matrices
"""
struct DiagonalOp <: Operator end;

"""
    Symmetric Matrices
"""
struct SymmetricOp <: Operator end; 

"""
    Anti Symmetric Matrices
"""
struct AntiSymmetricOp <: Operator end; 

"""
    Scalar Matrices
"""
struct ScalarOp <: Operator end; 
# this gives errors when combined with symmetries, no idea why

"""
    Empty Matrices
"""
struct EmptyOp <: Operator end; 
# this gives errors when combined with symmetries, no idea why

#registe operators
OperatorsDict[:UniversalOp] = UniversalOp()
OperatorsDict[:DiagonalOp] = DiagonalOp()
OperatorsDict[:SymmetricOp] = SymmetricOp()
OperatorsDict[:AntiSymmetricOp] = AntiSymmetricOp()
OperatorsDict[:ScalarOp] = ScalarOp()
OperatorsDict[:EmptyOp] = EmptyOp()



#coordinates

coordinates(LΩ::UniversalOp, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing} =
    size(M)[1]==size(M)[2] ? reshape(M,length(M)) : nothing;

function coordinates(LΩ::DiagonalOp, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    (sizes[1]==sizes[2]) || return nothing
    if (all(__isapproxzero, vcat([M[1:(i-1),i] for i=1:sizes[1]]...)) && 
        all(__isapproxzero, vcat([M[(i+1):sizes[1],i] for i=1:(sizes[1]-1)]...) ))
        return [M[i,i] for i=1:size(M)[1]]
    else
        return nothing
    end
end;


function coordinates(LΩ::SymmetricOp, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    dim=sizes[1]
    (sizes[1]==sizes[2]) || return nothing
    if all(__isapproxzero ,vcat([M[1:i-1,i] - M[i,1:i-1] for i=1:dim]...))
        return vcat([M[1:i,i] for i=1:dim]...)
    else 
        return nothing
    end
end;

function coordinates(LΩ::AntiSymmetricOp, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    dim=sizes[1]
    (sizes[1]==sizes[2]) || return nothing
    if all(__isapproxzero ,vcat([M[1:i-1,i] + M[i,1:i-1] for i=1:dim]...))
        return vcat([M[1:i-1,i] for i=1:dim]...)
    else 
        return nothing
    end
end;

function coordinates(LΩ::ScalarOp, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    dim=sizes[1]
    (sizes[1]==sizes[2]) || return nothing
    # dim==0 && return zeros(0)
    (all(__isapproxzero, vcat([M[1:i-1,i] for i=1:sizes[1]]...)) && all(__isapproxzero, vcat([M[i+1:dim,i] for i=1:sizes[1]]...))) || return nothing
    all(__isapproxzero,[ M[i,i] - M[1,1] for i = 2:dim]) || return nothing
    return [M[1,1]]
end;

function coordinates(LΩ::EmptyOp, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    dim=sizes[1]
    (sizes[1]==sizes[2]) || return nothing
    all(__isapproxzero, vcat([M[1:dim] for i=1:sizes[1]]...)) || return nothing
    return zeros(0)
end;


#unsafe_coordinates

unsafe_coordinates(LΩ::UniversalOp, M::AbstractMatrix) :: Vector{<:Number} = reshape(Matrix(M),length(M));

unsafe_coordinates(LΩ::DiagonalOp, M::AbstractMatrix) :: Vector{<:Number} = [M[i,i] for i=1:size(M)[1]];

unsafe_coordinates(LΩ::SymmetricOp, M::AbstractMatrix) :: Vector{<:Number} = vcat([M[1:i,i] for i=1:size(M)[1]]...);

unsafe_coordinates(LΩ::AntiSymmetricOp, M::AbstractMatrix) :: Vector{<:Number} = vcat([M[1:(i-1),i] for i=1:size(M)[1]]...);

# unsafe_coordinates(LΩ::ScalarOp, M::AbstractMatrix) :: Vector{<:Number} =  size(M)[1] ==0 ? zeros(0) : [M[1,1]];
unsafe_coordinates(LΩ::ScalarOp, M::AbstractMatrix) :: Vector{<:Number} =  [M[1,1]];

unsafe_coordinates(LΩ::EmptyOp, M::AbstractMatrix) :: Vector{<:Number} =  zeros(0);


#unsafe_transposeEmbed

unsafe_transposeEmbed(LΩ::UniversalOp, M::AbstractMatrix) :: Vector{<:Number} = reshape(Matrix(M),length(M));

unsafe_transposeEmbed(LΩ::DiagonalOp, M::AbstractMatrix) :: Vector{<:Number} = [M[i,i] for i=1:size(M)[1]];

unsafe_transposeEmbed(LΩ::SymmetricOp, M::AbstractMatrix) :: Vector{<:Number} = vcat( [vcat( M[1:(i-1),i] + M[i,1:(i-1)],M[i,i]) for i=1:size(M)[1]]...);

unsafe_transposeEmbed(LΩ::AntiSymmetricOp, M::AbstractMatrix) :: Vector{<:Number} = vcat( [M[1:(i-1),i] - M[i,1:(i-1)] for i=1:size(M)[1]]...);

# unsafe_transposeEmbed(LΩ::ScalarOp, M::AbstractMatrix) :: Vector{<:Number} = size(M)[1] ==0 ? zeros(0) : [sum([M[i,i] for i=1:size(M)[1]]) ];
unsafe_transposeEmbed(LΩ::ScalarOp, M::AbstractMatrix) :: Vector{<:Number} = [sum([M[i,i] for i=1:size(M)[1]]) ];

unsafe_transposeEmbed(LΩ::EmptyOp, M::AbstractMatrix) :: Vector{<:Number} = zeros(0);

#unsafe_embed
unsafe_embed(LΩ::UniversalOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix = reshape(data,dim,dim);

unsafe_embed(LΩ::DiagonalOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix =  LinearAlgebra.Diagonal(data);

function unsafe_embed(LΩ::SymmetricOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  
    A=zeros(eltype(data),dim,dim)
    k = 1
    for i = 1:dim
        A[1:i,i] = data[k:k+i-1]
        A[i,1:i] = data[k:k+i-1]
        k = k+i
    end 
    return LinearAlgebra.Symmetric(A)
end;

function unsafe_embed(LΩ::AntiSymmetricOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  
    A=zeros(eltype(data),dim,dim)
    k = 1
    for i = 1:dim
        A[1:(i-1),i] = data[k:k+i-2]
        A[i,1:(i-1)] = -data[k:k+i-2]
        k = k+i-1
    end 
    return A
end;

# unsafe_embed(LΩ::ScalarOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix = dim==0 ? Matrix(LinearAlgebra.I,0,0) : Matrix(data[1]*LinearAlgebra.I,dim,dim); 
unsafe_embed(LΩ::ScalarOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix = Matrix(data[1]*LinearAlgebra.I,dim,dim); 

unsafe_embed(LΩ::EmptyOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix = Matrix(0.0*LinearAlgebra.I,dim,dim); 

unsafe_dualize(::DiagonalOp, dim::Integer, data::Vector{<:Number} ) = data 
unsafe_dualize(::SymmetricOp, dim::Integer, data::Vector{<:Number} ) = data 
unsafe_dualize(::AntiSymmetricOp, dim::Integer, data::Vector{<:Number} ) = -data 
unsafe_dualize(::ScalarOp, dim::Integer, data::Vector{<:Number} ) = data 
unsafe_dualize(::EmptyOp, dim::Integer, data::Vector{<:Number} ) = data 



#local Dimension
localDim(::UniversalOp, dim::Integer) = dim*dim; 
localDim(::DiagonalOp, dim::Integer) = dim; 
localDim(::SymmetricOp, dim::Integer) = dim*(dim+1) ÷ 2; 
localDim(::AntiSymmetricOp, dim::Integer) = dim*(dim-1) ÷ 2; 
# localDim(::ScalarOp, dim::Integer) = dim==0 ? 0 : 1; 
localDim(::ScalarOp, dim::Integer) = 1; 
localDim(::EmptyOp, dim::Integer) = 0; 

#contains Scalars
containScalars(::UniversalOp)= true;
containScalars(::DiagonalOp) = true; 
containScalars(::SymmetricOp) = true; 
containScalars(::AntiSymmetricOp) = false; 
containScalars(::ScalarOp) = true; 
containScalars(::EmptyOp) = false; 

#closedUnderDual
closedUnderDual(::UniversalOp)= true;
closedUnderDual(::DiagonalOp) = true; 
closedUnderDual(::SymmetricOp) = true; 
closedUnderDual(::AntiSymmetricOp) = true; 
closedUnderDual(::ScalarOp) = true; 
closedUnderDual(::EmptyOp) = true; 
