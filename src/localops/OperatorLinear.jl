#
# Strata Dleto: Local Linear Operators Implementations
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
    Local Linear Operators Implementations

    Provides standard local operators like -- all_matrices, symmetric_matrices and diagonal_matrices
"""

# working with matrices of size zero does not work. Usually the result of the function is [] but this is of type Vector{Any} which is not Vector{<:Number}

"""
    All Matrices
"""
struct UniversalOp <: LinearOperator end; 

"""
    Diagonal Matrices
"""
struct DiagonalOp <: LinearOperator end;

"""
    TriDiagonal Matrices
"""
struct TriDiagonalOp <: LinearOperator end; 

"""
    Symmetric Matrices
"""
struct SymmetricOp <: LinearOperator end; 

"""
    Anti Symmetric Matrices
"""
struct AntiSymmetricOp <: LinearOperator end; 

"""
    Circulant Matrices
"""

struct CirculantOp <: LinearOperator end; 
"""
    Scalar Matrices
"""
struct ScalarOp <: LinearOperator end; 


"""
    Empty Matrices
"""
struct EmptyOp <: LinearOperator end; 



#registe operators
LinearOperatorsDict[:default] = UniversalOp()
LinearOperatorsDict[:UniversalOp] = UniversalOp()
LinearOperatorsDict[:DiagonalOp] = DiagonalOp()
LinearOperatorsDict[:TriDiagonalOp] = TriDiagonalOp()
LinearOperatorsDict[:SymmetricOp] = SymmetricOp()
LinearOperatorsDict[:AntiSymmetricOp] = AntiSymmetricOp()
LinearOperatorsDict[:CirculantOp] = CirculantOp()
LinearOperatorsDict[:ScalarOp] = ScalarOp()
LinearOperatorsDict[:EmptyOp] = EmptyOp()









#coordinates

coordinates(::UniversalOp, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing} =
    size(M)[1]==size(M)[2] ? reshape(M,length(M)) : nothing;

function coordinates(::DiagonalOp, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    (sizes[1]==sizes[2]) || return nothing
    if (all(__isapproxzero, vcat([M[1:(i-1),i] for i=1:sizes[1]]...)) && 
        all(__isapproxzero, vcat([M[i,1:(i-1)] for i=1:sizes[1]]...) ))
        return [M[i,i] for i=1:size(M)[1]]
    else
        return nothing
    end
end;

function coordinates(::TriDiagonalOp, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    (sizes[1]==sizes[2]) || return nothing
    if (all(__isapproxzero, vcat([M[1:(i-2),i] for i=1:sizes[1]]...)) && 
        all(__isapproxzero, vcat([M[i,1:(i-2)] for i=1:sizes[1]]...) ))
        return vcat(
                [M[i,i] for i=1:size(M)[1]],
                [M[i,i-1] for i=2:size(M)[1]],
                [M[i-1,i] for i=2:size(M)[1]]
        )
    else
        return nothing
    end
end;


function coordinates(::SymmetricOp, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    dim=sizes[1]
    (sizes[1]==sizes[2]) || return nothing
    if all(__isapproxzero ,vcat([M[1:i-1,i] - M[i,1:i-1] for i=1:dim]...))
        return vcat([M[1:i,i] for i=1:dim]...)
    else 
        return nothing
    end
end;

function coordinates(::AntiSymmetricOp, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    dim=sizes[1]
    (sizes[1]==sizes[2]) || return nothing
    if all(__isapproxzero ,vcat([M[1:i-1,i] + M[i,1:i-1] for i=1:dim]...))
        return vcat([M[1:i-1,i] for i=1:dim]...)
    else 
        return nothing
    end
end;

function coordinates(::CirculantOp, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    dim=sizes[1]
    (sizes[1]==sizes[2]) || return nothing
    for j = 2:dim
        isapprox( M[1:dim-j+1,1],  M[j:dim,j]) || return nothing
        isapprox( M[dim-j+2:dim,1],  M[1:(j-1),j] ) || return nothing
    end
    return M[:,1]
end;


function coordinates(::ScalarOp, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    dim=sizes[1]
    (sizes[1]==sizes[2]) || return nothing
    # dim==0 && return zeros(0)
    (all(__isapproxzero, vcat([M[1:i-1,i] for i=1:sizes[1]]...)) && all(__isapproxzero, vcat([M[i+1:dim,i] for i=1:sizes[1]]...))) || return nothing
    all(__isapproxzero,[ M[i,i] - M[1,1] for i = 2:dim]) || return nothing
    return [M[1,1]]
end;

function coordinates(::EmptyOp, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing}
    sizes=size(M)
    dim=sizes[1]
    (sizes[1]==sizes[2]) || return nothing
    all(__isapproxzero, reshape(M,length(M)) ) || return nothing
    return zeros(0)
end;


#unsafe_coordinates

unsafe_coordinates(::UniversalOp, M::AbstractMatrix) :: Vector{<:Number} = reshape(Matrix(M),length(M));

# unsafe_coordinates(::DiagonalOp, M::AbstractMatrix) :: Vector{<:Number} = [M[i,i] for i=1:size(M)[1]];
unsafe_coordinates(::DiagonalOp, M::AbstractMatrix) :: Vector{<:Number} = diag(M);

unsafe_coordinates(::TriDiagonalOp, M::AbstractMatrix) :: Vector{<:Number} = vcat(
                [M[i,i] for i=1:size(M)[1]],
                [M[i,i-1] for i=2:size(M)[1]],
                [M[i-1,i] for i=2:size(M)[1]]
        );

unsafe_coordinates(::SymmetricOp, M::AbstractMatrix) :: Vector{<:Number} = vcat([M[1:i,i] for i=1:size(M)[1]]...);

unsafe_coordinates(::AntiSymmetricOp, M::AbstractMatrix) :: Vector{<:Number} = vcat([M[1:(i-1),i] for i=1:size(M)[1]]...);

unsafe_coordinates(::CirculantOp, M::AbstractMatrix) :: Vector{<:Number} = M[:,1];

unsafe_coordinates(::ScalarOp, M::AbstractMatrix) :: Vector{<:Number} =  [M[1,1]];

unsafe_coordinates(::EmptyOp, M::AbstractMatrix) :: Vector{<:Number} =  zeros(0);


#unsafe_transposeEmbed

unsafe_transposeEmbed(::UniversalOp, M::AbstractMatrix) :: Vector{<:Number} = reshape(Matrix(M),length(M));

# unsafe_transposeEmbed(::DiagonalOp, M::AbstractMatrix) :: Vector{<:Number} = [M[i,i] for i=1:size(M)[1]];
unsafe_transposeEmbed(::DiagonalOp, M::AbstractMatrix) :: Vector{<:Number} = diag(M);

unsafe_transposeEmbed(::TriDiagonalOp, M::AbstractMatrix) :: Vector{<:Number} = vcat(
                [M[i,i] for i=1:size(M)[1]],
                [M[i,i-1] for i=2:size(M)[1]],
                [M[i-1,i] for i=2:size(M)[1]]
        );

unsafe_transposeEmbed(::SymmetricOp, M::AbstractMatrix) :: Vector{<:Number} = vcat( [vcat( M[1:(i-1),i] + M[i,1:(i-1)],M[i,i]) for i=1:size(M)[1]]...);

unsafe_transposeEmbed(::AntiSymmetricOp, M::AbstractMatrix) :: Vector{<:Number} = vcat( [M[1:(i-1),i] - M[i,1:(i-1)] for i=1:size(M)[1]]...);

function unsafe_transposeEmbed(::CirculantOp, M::AbstractMatrix) :: Vector{<:Number}
    n = size(M,1)
    v = M[:,1]
    for j = 2:n
        v[1: n-j+1] += M[j:n,j]
        v[n-j+2:n] += M[1:(j-1),j]
    end
    return v
end;

# unsafe_transposeEmbed(Op::ScalarOp, M::AbstractMatrix) :: Vector{<:Number} = size(M)[1] ==0 ? zeros(0) : [sum([M[i,i] for i=1:size(M)[1]]) ];
# unsafe_transposeEmbed(::ScalarOp, M::AbstractMatrix) :: Vector{<:Number} = [sum([M[i,i] for i=1:size(M)[1]]) ];
unsafe_transposeEmbed(::ScalarOp, M::AbstractMatrix) :: Vector{<:Number} = [tr(M) ];

unsafe_transposeEmbed(::EmptyOp, M::AbstractMatrix) :: Vector{<:Number} = zeros(0);

#unsafe_embed
unsafe_embed(::UniversalOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix = reshape(data,dim,dim);

unsafe_embed(::DiagonalOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix =  LinearAlgebra.Diagonal(data);

# function unsafe_embed(::TriDiagonalOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix 
#     A=zeros(eltype(data),dim,dim)
#     A[1,1] = data[1]
#     for i = 2:dim
#         A[i,i] = data[i]
#         A[i,i-1] = data[i+dim-1]
#         A[i-1,i] = data[i+2*dim-2]
#     end 
#     return A
# end;    
### Tridiagonal breaks ITensors....
unsafe_embed(::TriDiagonalOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix =
    Matrix(LinearAlgebra.Tridiagonal(data[dim+1:2*dim-1],data[1:dim], data[2*dim:3*dim-2]));

function unsafe_embed(::SymmetricOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  
    A=zeros(eltype(data),dim,dim)
    k = 1
    for i = 1:dim
        A[1:i,i] = data[k:k+i-1]
        A[i,1:i] = data[k:k+i-1]
        k = k+i
    end 
    return LinearAlgebra.Symmetric(A)
end;

function unsafe_embed(::AntiSymmetricOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  
    A=zeros(eltype(data),dim,dim)
    k = 1
    for i = 1:dim
        A[1:(i-1),i] = data[k:k+i-2]
        A[i,1:(i-1)] = -data[k:k+i-2]
        k = k+i-1
    end 
    return A
end;

function unsafe_embed(::CirculantOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  
    A=zeros(eltype(data),dim,dim)
    for i = 1:dim
        A[i:dim,i] = data[1:dim-i+1]
        A[1:(i-1),i] = data[dim-i+2:dim]
    end 
    return A
end;

# unsafe_embed(Op::ScalarOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix = dim==0 ? Matrix(LinearAlgebra.I,0,0) : Matrix(data[1]*LinearAlgebra.I,dim,dim); 
unsafe_embed(::ScalarOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix = LinearAlgebra.Diagonal([data[1] for i=1:dim]);

unsafe_embed(::EmptyOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix = zeros(dim,dim);

unsafe_star(::DiagonalOp, dim::Integer, data::Vector{<:Number} ) = data;

unsafe_star(::TriDiagonalOp, dim::Integer, data::Vector{<:Number} ) = vcat(data[1:dim],data[(2*dim):(3*dim-2)],data[(dim+1):(2*dim-1)]); 
unsafe_star(::SymmetricOp, dim::Integer, data::Vector{<:Number} ) = data; 
unsafe_star(::AntiSymmetricOp, dim::Integer, data::Vector{<:Number} ) = -data; 
unsafe_star(::ScalarOp, dim::Integer, data::Vector{<:Number} ) = data; 
unsafe_star(::EmptyOp, dim::Integer, data::Vector{<:Number} ) = data; 



#local Dimension
localDim(::UniversalOp, dim::Integer) = dim*dim; 
localDim(::DiagonalOp, dim::Integer) = dim; 
localDim(::TriDiagonalOp, dim::Integer) = 3*dim -2; 
localDim(::SymmetricOp, dim::Integer) = dim*(dim+1) ÷ 2; 
localDim(::AntiSymmetricOp, dim::Integer) = dim*(dim-1) ÷ 2; 
localDim(::CirculantOp, dim::Integer) = dim; 
localDim(::ScalarOp, dim::Integer) = 1; 
localDim(::EmptyOp, dim::Integer) = 0; 

#contains Scalars
containScalars(::UniversalOp)= true;
containScalars(::DiagonalOp) = true; 
containScalars(::TriDiagonalOp) = true; 
containScalars(::SymmetricOp) = true; 
containScalars(::AntiSymmetricOp) = false; 
containScalars(::CirculantOp) = true; 
containScalars(::ScalarOp) = true; 
containScalars(::EmptyOp) = false; 

#closedUnderDual
closedUnderStar(::UniversalOp)= true;
closedUnderStar(::DiagonalOp) = true; 
closedUnderStar(::TriDiagonalOp) = true; 
closedUnderStar(::SymmetricOp) = true; 
closedUnderStar(::AntiSymmetricOp) = true; 
closedUnderStar(::CirculantOp) = true; 
closedUnderStar(::ScalarOp) = true; 
closedUnderStar(::EmptyOp) = true; 
