#
# Strata Dleto: Local Inverible Operators Implementations
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
    Local Inverible Operators Implementations

    Provides standard local operators like -- all_matrices, symmetric_matrices and diagonal_matrices
"""

# working with matrices of size zero does not work. Usually the result of the function is [] but this is of type Vector{Any} which is not Vector{<:Number}

"""
    All Invertible Matrices
"""
struct InvertableOp <: InvertableOperator end; 

"""
    Orthogonal Matrices
"""
struct OrthogonalOp <: InvertableOperator end;

# """
#     Permutation Matrices
# """
# struct PermutationOp <: InvertableOperator end; 

# """
#     SignedPermutation Matrices
# """
# struct SignedPermutationOp <: InvertableOperator end; 

"""
    OnlyIdenity Matrices
"""
struct OnlyIdOp <: InvertableOperator end; 

#registe operators
InvertableOperatorsDict[:default] = InvertableOp(); 
InvertableOperatorsDict[:InvertableOp] = InvertableOp(); 
InvertableOperatorsDict[:OrthogonalOp] = OrthogonalOp(); 
InvertableOperatorsDict[:OnlyIdOp] = OnlyIdOp(); 




#coordinates

function coordinates(::InvertableOp, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing} 
    size(M)[1]==size(M)[2] || return nothing
    isapprox(LinearAlgebra.det(M), 0.0) && return nothing
    return reshape(Matrix(M),length(M))
end;

function coordinates(::OrthogonalOp, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing} 
    size(M)[1]==size(M)[2] || return nothing
    isapprox(LinearAlgebra.det(M), 0.0) && return nothing
    isapprox(inv(M), LinearAlgebra.transpose(M)) || return nothing
    return reshape(Matrix(M),length(M))
end;

#needs fixing
function coordinates(::OnlyIdOp, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing} 
    size(M)[1]==size(M)[2] || return nothing
    isapprox(LinearAlgebra.det(M), 0.0) && return nothing
    isapprox(inv(M), LinearAlgebra.transpose(M)) || return nothing
    return zeros(0)
end;

#unsafe_coordinates

unsafe_coordinates(::InvertableOp, M::AbstractMatrix) :: Vector{<:Number} = reshape(Matrix(M),length(M));

unsafe_coordinates(::OrthogonalOp, M::AbstractMatrix) :: Vector{<:Number} = reshape(Matrix(M),length(M));

unsafe_coordinates(::OnlyIdOp, M::AbstractMatrix) :: Vector{<:Number} = zeros(0);

#unsafe_embed
unsafe_embed(::InvertableOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix = reshape(data,dim,dim);

unsafe_embed(::OrthogonalOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix = reshape(data,dim,dim);

unsafe_embed(::OnlyIdOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix = LinearAlgebra.Diagonal([1.0 for i=1:dim]);

function embed(Op::OrthogonalOp, dim::Integer, data::Vector{<:Number} ) ::Union{AbstractMatrix,Nothing}  
    @assert dim > 0 "Dimension needs to be positive"
    @assert length(data)== localDim(Op,dim) "Incompatable Data"
    M = unsafe_embed(Op,dim,data)
    if !isapprox(LinearAlgebra.det(M),0)
        return isapprox(inv(M), LinearAlgebra.transpose(M)) ? M : nothing
    else
        return nothing
    end
end;

embed(::OnlyIdOp, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix = LinearAlgebra.Diagonal([1.0 for i=1:dim]);

unsafe_star(::OrthogonalOp, dim::Integer, data::Vector{<:Number} ) = data 
unsafe_star(::OnlyIdOp, dim::Integer, data::Vector{<:Number} ) = data 



#local Dimension
localDim(::InvertableOp, dim::Integer) = dim*dim; 
localDim(::OrthogonalOp, dim::Integer) = dim*dim; 
localDim(::OnlyIdOp, dim::Integer) = 0; 


#closedUnderDual
closedUnderStar(::InvertableOp)= true;
closedUnderStar(::OrthogonalOp) = true; 
closedUnderStar(::OnlyIdOp) = true; 

function generate_random(::InvertableOp, dim::Integer)::Vector{<:Number}
    M = randn(dim,dim)
    while __isapproxzero(LinearAlgebra.det(M))
        M = randn(dim,dim)
    end
    return reshape(M, dim*dim) 
end;

function generate_random(::OrthogonalOp, dim::Integer)::Vector{<:Number}
    S= randn(dim,dim)
    Q = LinearAlgebra.eigen(LinearAlgebra.Symmetric(S*S') + LinearAlgebra.I).vectors
    return reshape(Q, dim*dim)
end;

generate_random(::OnlyIdOp, dim::Integer) = zeros(0); 
trivial(::OnlyIdOp, dim::Integer) = zeros(0); 
