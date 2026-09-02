#
# Strata Dleto: Local Operators Abstract
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
    LocalOperators

    TO BE WRITTEN
"""

"""
    LocalOperators

    A subspace of operators acting on each axis

    needs to prodive functions like coordinates, embed and theansposeEmbedding

"""
abstract type Operator  end


"""
    Return the native encoding of an operator in the transverse set,
    or `Nothing` if it is not a member.
"""
function coordinates(LΩ::Operator, M::AbstractMatrix) :: Union{AbstractVector{<:Number}, Nothing} 
    @assert false "Calling Placeholder Abstract Function"
end;
function unsafe_coordinates(LΩ::Operator, M::AbstractMatrix) :: AbstractVector{<: Number} 
    @assert false "Calling Placeholder Abstract Function"
end;
# this is the inverse of the embed map

function transposeEmbed(LΩ::Operator, M::AbstractMatrix) :: AbstractVector{<:Number}
    @assert size(M)[1]==size(M)[2] "Incompatable Data"
    return unsafe_transposeEmbed(LΩ, M)  
end;

function unsafe_transposeEmbed(LΩ::Operator, M::AbstractMatrix) :: AbstractVector{<:Number} 
    @assert false "Calling Placeholder Abstract Function"
end;
# this is the transpose of the embed map

"""
    Convert the native encoding of an operator into matrix.
"""
function embed(LΩ::Operator, dim::Integer, data::AbstractVector{<:Number} ) ::AbstractMatrix  
    @assert length(data)== localDim(LΩ,dim) "Incompatable Data"
    unsafe_embed(LΩ,dim,data)
end;

function unsafe_embed(LΩ::Operator, dim::Integer, data::AbstractVector{<:Number} ) ::AbstractMatrix  
    @assert false "Calling Placeholder Abstract Function"
end;

function dualize(LΩ::Operator, dim::Integer, data::AbstractVector{<:Number} ) ::AbstractVector{<:Number}
    @assert length(data)== localDim(LΩ,dim) "Incompatable Data"
    unsafe_dualize(LΩ,dim,data)
end;

unsafe_dualize(LΩ::Operator, dim::Integer, data::AbstractVector{<:Number} ) ::AbstractVector{<:Number} = 
    unsafe_coordinates(LΩ,Matrix(LinearAlgebra.transpose(unsafe_embed(LΩ,dim,data))));


function localDim(LΩ::Operator, dim::Integer)::Integer 
    @assert false "Calling Placeholder Abstract Function"
end;

function containScalars(LΩ::Operator)::Bool  
    @assert false "Calling Placeholder Abstract Function"
end;


