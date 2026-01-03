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

    Abstract Class for providing encoding and decoving data from list of matrices/ITensors to in internal vectors.
    Need to provide functions
    -- coodinates: turn list of matrices into vector or retruns nothing.
    -- unsafe_coordinates: same but assumes that the vector is in the image and does not do any checking
    -- embeddingMatrices/embeddingITensors: turns a vector into list of matrices 
        (this needs to be a linear map, which is left(right?) of coordinatres) 
    -- unsafe_embeddingMatrices/unsafe_embeddingITensors: as above but does not do any checking
    -- transposeEmbedding: linear dual(transpose) of the emnedding, turns list of matrices into a vector

    several function like globalDim, axisDims, valancy: which provide data for the sizes of matrices
        
    most safe functions are auto-generated from the unsafe one by adding trivial checks
    
    -- reduceByEngaged(::AbstractGlobalOps, ::Vector{Bool})::AbstractGlobalOps produce new GlobalOps,
        where some of the axis are not engaged

"""
abstract type AbstractGlobalOps end


export AbstractGlobalOps
export coordinates, unsafe_coordinates, transposeEmbedding, unsafe_transposeEmbedding
export embeddingMatrices, unsafe_embeddingMatrices, embeddingITensors, unsafe_embeddingITensors

"""
    Return the native encoding of an operator in the transverse set,
    or `Nothing` if it is not a member.
"""
function coordinates(GΩ::AbstractGlobalOps, Mats::Vector{<: AbstractMatrix} ) :: Union{Vector{<:Number}, Nothing} 
    @assert false "Calling Placeholder Abstract Function"
end;
function coordinates(GΩ::AbstractGlobalOps, ITs::Vector{ITensor} ) :: Union{Vector{<:Number}, Nothing} 
    val = valency(GΩ)
    @assert length(ITs) == val "Incompatable Data"
    fr = frames(GΩ)
    frT = framesTemporary(GΩ) 
    @assert all([ length(inds(ITs[i])) == 2  for i =1:val ])  "Incompatable Data"
    @assert all([ inds(ITs[i])[1] == fr[i]  for i =1:val ])  "Incompatable Data"
    @assert all([ inds(ITs[i])[2] == frT[i]  for i =1:val ])  "Incompatable Data"
    coordinates(GΩ, ITs.|> __asMatrix);
end;

function unsafe_coordinates(GΩ::AbstractGlobalOps, Mats::Vector{<: AbstractMatrix} ) :: Vector{<: Number} 
    @assert false "Calling Placeholder Abstract Function"
end;
unsafe_coordinates(GΩ::AbstractGlobalOps, ITs::Vector{ITensor} ) :: Vector{<: Number} = 
    unsafe_coordinates(GΩ, ITs.|> __asMatrix);
# this is the inverse of the embedding map

function transposeEmbedding(GΩ::AbstractGlobalOps, Mats::Vector{<:AbstractMatrix}) :: Vector{<:Number}
    val = valency(GΩ)
    @assert length(Mats) == val "Incompatable Data"
    localdims= axisDims(GΩ) 
    @assert all([ size(Mats[i])[1] == localdims[i]  for i =1:val ])  "Incompatable Data"
    @assert all([ size(Mats[i])[2] == localdims[i]  for i =1:val ])  "Incompatable Data"
    return unsafe_transposeEmbedding(GΩ, Mats)  
end;
function transposeEmbedding(GΩ::AbstractGlobalOps, ITs::Vector{ITensor}) :: Vector{<:Number}
    val = valency(GΩ)
    @assert length(ITs) == val "Incompatable Data"
    fr = frames(GΩ)
    frT = framesTemporary(GΩ) 
    @assert all([ length(inds(ITs[i])) == 2  for i =1:val ])  "Incompatable Data"
    @assert all([ inds(ITs[i])[1] == fr[i]  for i =1:val ])  "Incompatable Data"
    @assert all([ inds(ITs[i])[2] == frT[i]  for i =1:val ])  "Incompatable Data"
    return unsafe_transposeEmbedding(LΩ, ITs .|> __asMatrix)  
end;

function unsafe_transposeEmbedding(GΩ::AbstractGlobalOps, Mats::Vector{<:AbstractMatrix}) :: Vector{<:Number} 
    @assert false "Calling Placeholder Abstract Function"
end;

unsafe_transposeEmbedding(GΩ::AbstractGlobalOps, ITs::Vector{<:ITensor}) :: Vector{<:Number} =
    unsafe_transposeEmbedding(GΩ, ITs.|> __asMatrix);
# this is the transpose of the embedding map

"""
    Convert the native encoding of an operator into matrices.
"""
function embeddingMatrices(GΩ::AbstractGlobalOps, data::Vector{<:Number} ) ::Vector{<:AbstractMatrix}  
    @assert length(data) == globalDim(GΩ) "Incompatable Data"
    unsafe_embeddingMatrices(GΩ,data)
end;

function unsafe_embeddingMatrices(GΩ::AbstractGlobalOps, data::Vector{<:Number} ) ::Vector{<:AbstractMatrix}
    @assert false "Calling Placeholder Abstract Function"
end;

function embeddingITensors(GΩ::AbstractGlobalOps, data::Vector{<:Number} ) ::Vector{ITensor}  
    @assert length(data) == globalDim(GΩ) "Incompatable Data"
    unsafe_embeddingITensors(GΩ,data)
end;

function unsafe_embeddingITensors(GΩ::AbstractGlobalOps, data::Vector{<:Number} ) ::Vector{<:ITensor}
    @assert false "Calling Placeholder Abstract Function"
end;


export globalDim, axisDims, valency, frames, framesTemporary, reduceByEngaged
"""
    dimension of the vectors in the encoding
"""
function globalDim(GΩ::AbstractGlobalOps)::Integer 
    @assert false "Calling Placeholder Abstract Function"
end;

function axisDims(GΩ::AbstractGlobalOps)::Vector{<:Integer} 
    @assert false "Calling Placeholder Abstract Function"
end;


function valency(GΩ::AbstractGlobalOps)::Integer 
    @assert false "Calling Placeholder Abstract Function"
end;

function frames(GΩ::AbstractGlobalOps)::Vector   #should be Vector{Index}
    @assert false "Calling Placeholder Abstract Function"
end;

function framesTemporary(GΩ::AbstractGlobalOps)::Vector  #should be Vector{Index}
    @assert false "Calling Placeholder Abstract Function"
end;

function reduceByEngaged(GΩ::AbstractGlobalOps, engaged::Vector{Bool})::AbstractGlobalOps 
    @assert false "Calling Placeholder Abstract Function"
end;


# no needed
# function engaged(GΩ::AbstractGlobalOps)::Vector{Bool} 
#     @assert false "Calling Placeholder Abstract Function"
# end;

#probably needs to be moved somewhere else
function __asMatrix(T::ITensor)::AbstractMatrix
    fr = inds(T)
    n = ITensors.dim(fr[1])
    m = ITensors.dim(fr[2])
    A = zeros(eltype(T),n,m)
    for ci in CartesianIndices(A)
        A[ci] = T[ci]
    end
    return A
end;