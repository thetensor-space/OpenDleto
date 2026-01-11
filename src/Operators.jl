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
    Dictionary of implemented local operators
"""
OperatorsDict = Dict{Symbol, Operator}()


"""
    Return the native encoding of an operator in the transverse set,
    or `Nothing` if it is not a member.
"""
function coordinates(Op::Operator, M::AbstractMatrix) :: Union{Vector{<:Number}, Nothing} 
    @assert false "Calling Placeholder Abstract Function"
end;
function unsafe_coordinates(Op::Operator, M::AbstractMatrix) :: Vector{<: Number} 
    @assert false "Calling Placeholder Abstract Function"
end;
# this is the inverse of the embed map

function transposeEmbed(Op::Operator, M::AbstractMatrix) :: Vector{<:Number}
    @assert size(M)[1]==size(M)[2] "Incompatable Data"
    return unsafe_transposeEmbed(Op, M)  
end;

function unsafe_transposeEmbed(Op::Operator, M::AbstractMatrix) :: Vector{<:Number} 
    @assert false "Calling Placeholder Abstract Function"
end;
# this is the transpose of the embed map

"""
    Convert the native encoding of an operator into matrix.
"""
function embed(Op::Operator, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  
    @assert length(data)== localDim(Op,dim) "Incompatable Data"
    unsafe_embed(Op,dim,data)
end;

function unsafe_embed(Op::Operator, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  
    @assert false "Calling Placeholder Abstract Function"
end;

function dualize(Op::Operator, dim::Integer, data::Vector{<:Number} ) ::Vector{<:Number}
    @assert length(data)== localDim(Op,dim) "Incompatable Data"
    unsafe_dualize(Op,dim,data)
end;

function unsafe_dualize(Op::Operator, dim::Integer, data::Vector{<:Number} ) ::Vector{<:Number}
    if closedUnderDual(Op)  
        unsafe_coordinates(Op,Matrix(LinearAlgebra.transpose(unsafe_embed(Op,dim,data))))
    else 
        @assert false "Calling Placeholder Abstract Function"
    end
end;

function localDim(Op::Operator, dim::Integer)::Integer 
    @assert false "Calling Placeholder Abstract Function"
end;

function containScalars(Op::Operator)::Bool  
    @assert false "Calling Placeholder Abstract Function"
end;

function closedUnderDual(Op::Operator)::Bool  
    @assert false "Calling Placeholder Abstract Function"
end;
