#
# Strata Dleto: Abstract Global Operators
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
    Abstract Global Operators

    TO BE WRITTEN
"""

"""
    Abstract Global Operators

    TO BE WRITTEN

"""
abstract type AbstractGlobalOps end

"""
    Return the native encoding of an operator in the transverse set,
    or `Nothing` if it is not a member.
"""
function coordinates(Ω::AbstractGlobalOps, M::Vector{<: AbstractMatrix} ) :: Union{Vector{<:Number}, Nothing} 
    @assert false "Calling Placeholder Abstract Function"
end;
function coordinates(Ω::AbstractGlobalOps, M::Vector{ITensor} ) :: Union{Vector{<:Number}, Nothing} 
    @assert false "Calling Placeholder Abstract Function"
end;
function unsafe_coordinates(Ω::AbstractGlobalOps, M::Vector{<: AbstractMatrix} ) :: Vector{<: Number} 
    @assert false "Calling Placeholder Abstract Function"
end;
function unsafe_coordinates(Ω::AbstractGlobalOps, M::Vector{ITensor} ) :: Vector{<: Number} 
    @assert false "Calling Placeholder Abstract Function"
end;
# this is the inverse of the embedding map

function transposeEmbedding(GΩ::AbstractGlobalOps, M::Vector{<:AbstractMatrix}) :: Vector{<:Number}
    @assert size(M)[1]==size(M)[2] "Incompatable Data"
    return unsafe_transposeEmbedding(LΩ, M)  
end;
function transposeEmbedding(GΩ::AbstractGlobalOps, M::Vector{ITensor}) :: Vector{<:Number}
    @assert size(M)[1]==size(M)[2] "Incompatable Data"
    return unsafe_transposeEmbedding(LΩ, M)  
end;

function unsafe_transposeEmbedding(GΩ::AbstractGlobalOps, M::Vector{<:AbstractMatrix}) :: Vector{<:Number} 
    @assert false "Calling Placeholder Abstract Function"
end;

function unsafe_transposeEmbedding(GΩ::AbstractGlobalOps, M::Vector{<:ITensor}) :: Vector{<:Number} 
    @assert false "Calling Placeholder Abstract Function"
end;
# this is the transpose of the embedding map

"""
    Convert the native encoding of an operator into matrix.
"""
function embeddingMatrices(GΩ::AbstractGlobalOps, data::Vector{<:Number} ) ::Vectror{<:AbstractMatrix}  
    @assert length(data)== dim(GΩ) "Incompatable Data"
    unsafe_embeddingMatrices(GΩ,data)
end;

function unsafe_embedding(GΩ::AbstractGlobalOps, data::Vector{<:Number} ) ::Vector{<:AbstractMatrix}
    @assert false "Calling Placeholder Abstract Function"
end;


"""
    dimension of the vectors in the encoding
"""
function dim(GΩ::AbstractGlobalOps)::Integer 
    @assert false "Calling Placeholder Abstract Function"
end;

function valancy(GΩ::AbstractGlobalOps)::Integer 
    @assert false "Calling Placeholder Abstract Function"
end;

function frames(GΩ::AbstractGlobalOps)::Vector{Index{<:Any}} 
    @assert false "Calling Placeholder Abstract Function"
end;

function framesTemporary(GΩ::AbstractGlobalOps)::Vector{Index{<:Any}} 
    @assert false "Calling Placeholder Abstract Function"
end;

function engaged(GΩ::AbstractGlobalOps)::Vector{Bool} 
    @assert false "Calling Placeholder Abstract Function"
end;
