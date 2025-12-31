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
    LocalTransverseOperators

    TO BE WRITTEN
"""

"""
    LocalTransverseOperators

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
function member(LΩ::LocalTransverseOps, mats::Matrix) :: Union{Vector{Number}, Nothing} end
function unsafe_member(LΩ::LocalTransverseOps, mats::Vector{Matrix}) :: Vector{Number} end

"""
    Convert the native encoding of an operator in the transverse set
    into a vector of matrices representing the operator on each engaged axis.
"""
function transverse(LΩ::TransverseOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  end

function localDim(LΩ::LocalTransverseOps, dim::Integer)::Integer end
function containScalars(LΩ::LocalTransverseOps)::Bool  end



struct LocalUniversalOps <: LocalTransverseOps end 
struct LocalDiagonalOps <: LocalTransverseOps end 
struct LocalSymmetricOps <: LocalTransverseOps end 
struct LocalAnitSymmetricOps <: LocalTransverseOps end 




function transverse(LΩ::LocalUniversalOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrixMatrix  
    @assert length(data)== localDim(LΩ,dim) "Incompatable Data"
    return reshape(data,dim,dim)
end;

function transverse(LΩ::LocalDiagonalOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  
    @assert length(data)== localDim(LΩ,dim) "Incompatable Data"
    return LinearAlgebra.Diagonal(data)
end;

function transverse(LΩ::LocalSymmetricOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  
    @assert length(data)== localDim(LΩ,dim) "Incompatable Data"
    # ???? return LinearAlgebra.Diagonal(data)
end;

function transverse(LΩ::LocalAnitSymmetricOps, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  
    @assert length(data)== localDim(LΩ,dim) "Incompatable Data"
    # ??? return LinearAlgebra.Diagonal(data)
end;

localDim(::LocalUniversalOps, dim::Integer) = dim*dim; 
localDim(::LocalDiagonalOps, dim::Integer) = dim; 
localDim(::LocalSymmetricOps, dim::Integer) = dim*(dim+1) ÷ 2; 
localDim(::LocalAnitSymmetricOps, dim::Integer) = dim*(dim-1) ÷ 2; 

containScalars(::LocalUniversalOps)= true;
containScalars(::LocalDiagonalOps) = true; 
containScalars(::LocalSymmetricOps) = true; 
containScalars(::LocalAnitSymmetricOps) = false; 


