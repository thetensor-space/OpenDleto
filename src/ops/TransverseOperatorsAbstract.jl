#
# Strata Dleto: Abstract Global Operators
#   Creation and application of transverse operators for tensors.
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

import LinearMaps

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
    -- embedMatrices/embedITensors: turns a vector into list of matrices 
        (this needs to be a linear map, which is left(right?) of coordinatres) 
    -- unsafe_embedMatrices/unsafe_embedITensors: as above but does not do any checking
    -- transposeEmbed: linear dual(transpose) of the emnedding, turns list of matrices into a vector

    several function like globalDim, axisDims, valancy: which provide data for the sizes of matrices
        
    most safe functions are auto-generated from the unsafe one by adding trivial checks
    
    -- reduceByEngaged(::TransverseOps, ::Vector{Bool})::TransverseOps produce new GlobalOps,
        where some of the axis are not engaged

"""
abstract type TransverseOps end

"""
    Return the native encoding of an operator in the transverse set,
    or `nothing` if it is not a member.
"""
function coordinates(TOp::TransverseOps, Mats::Vector{<: AbstractMatrix} ) :: Union{Vector{<:Number}, Nothing} 
    @assert false "Calling Placeholder Abstract Function"
end;
function coordinates(TOp::TransverseOps, ITs::Vector{ITensor} ) :: Union{Vector{<:Number}, Nothing} 
    val = valency(TOp)
    @assert length(ITs) == val "Incompatable Data"
    fr = frames(TOp)
    frT = framesTemporary(TOp) 
    @assert all([ length(inds(ITs[i])) == 2  for i =1:val ])  "Incompatable Data"
    @assert all([ 
            ((inds(ITs[i])[1] == fr[i]) || (inds(ITs[i])[1] == frT[i])) && 
            ((inds(ITs[i])[2] == fr[i]) || (inds(ITs[i])[2] == frT[i])) && 
            (inds(ITs[i])[1] != inds(ITs[i])[2]) 
        for i =1:val ])  "Incompatable Data"
    return coordinates(TOp, [ inds(ITs[i])[1] == fr[i] ? __asMatrix(ITs[i]) : __asMatrixTranspose(ITs[i]) for i =1:val] );
end;

function unsafe_coordinates(TOp::TransverseOps, Mats::Vector{<: AbstractMatrix} ) :: Vector{<: Number} 
    @assert false "Calling Placeholder Abstract Function"
end;
unsafe_coordinates(TOp::TransverseOps, ITs::Vector{ITensor} ) :: Vector{<: Number} = 
    unsafe_coordinates(TOp, ITs.|> __asMatrix);
# this is the inverse of the embed map

function transposeEmbed(TOp::TransverseOps, Mats::Vector{<:AbstractMatrix}) :: Vector{<:Number}
    val = valency(TOp)
    @assert length(Mats) == val "Incompatable Data"
    localdims= axisDims(TOp) 
    @assert all([ size(Mats[i])[1] == localdims[i]  for i =1:val ])  "Incompatable Data"
    @assert all([ size(Mats[i])[2] == localdims[i]  for i =1:val ])  "Incompatable Data"
    return unsafe_transposeEmbed(TOp, Mats)  
end;

function transposeEmbed(TOp::TransverseOps, ITs::Vector{ITensor}) :: Vector{<:Number}
    val = valency(TOp)
    @assert length(ITs) == val "Incompatable Data"
    fr = frames(TOp)
    frT = framesTemporary(TOp) 
    @assert all([ length(inds(ITs[i])) == 2  for i =1:val ])  "Incompatable Data"
    @assert all([ 
            ((inds(ITs[i])[1] == fr[i]) || (inds(ITs[i])[1] == frT[i])) && 
            ((inds(ITs[i])[2] == fr[i]) || (inds(ITs[i])[2] == frT[i])) && 
            (inds(ITs[i])[1] != inds(ITs[i])[2]) 
        for i =1:val ])  "Incompatable Data"
    return unsafe_transposeEmbed(TOp, [ inds(ITs[i])[1] == fr[i] ? __asMatrix(ITs[i]) : __asMatrixTranspose(ITs[i]) for i =1:val] );
end;

function unsafe_transposeEmbed(TOp::TransverseOps, Mats::Vector{<:AbstractMatrix}) :: Vector{<:Number} 
    @assert false "Calling Placeholder Abstract Function"
end;

unsafe_transposeEmbed(TOp::TransverseOps, ITs::Vector{<:ITensor}) :: Vector{<:Number} =
    unsafe_transposeEmbed(TOp, ITs.|> __asMatrix);
# this is the transpose of the embed map

"""
    Convert the native encoding of an operator into matrices.
"""
function embedMatrices(TOp::TransverseOps, data::Vector{<:Number} ) ::Vector{<:AbstractMatrix}  
    @assert length(data) == globalDim(TOp) "Incompatable Data"
    unsafe_embedMatrices(TOp,data)
end;

function unsafe_embedMatrices(TOp::TransverseOps, data::Vector{<:Number} ) ::Vector{<:AbstractMatrix}
    @assert false "Calling Placeholder Abstract Function"
end;

function embedITensors(TOp::TransverseOps, data::Vector{<:Number} ) ::Vector{ITensor}  
    @assert length(data) == globalDim(TOp) "Incompatable Data"
    unsafe_embedITensors(TOp,data)
end;

function unsafe_embedITensors(TOp::TransverseOps, data::Vector{<:Number} ) ::Vector{<:ITensor}
    @assert false "Calling Placeholder Abstract Function"
end;

function embedITensorsSwaped(TOp::TransverseOps, data::Vector{<:Number} ) ::Vector{ITensor}  
    @assert length(data) == globalDim(TOp) "Incompatable Data"
    unsafe_embedITensorsSwapped(TOp,data)
end;

function unsafe_embedITensorsSwapped(TOp::TransverseOps, data::Vector{<:Number} ) ::Vector{<:ITensor}
    @assert false "Calling Placeholder Abstract Function"
end;

"""
    dimension of the vectors in the encoding
"""
function globalDim(TOp::TransverseOps)::Integer 
    @assert false "Calling Placeholder Abstract Function"
end;

function axisDims(TOp::TransverseOps)::Vector{<:Integer} 
    @assert false "Calling Placeholder Abstract Function"
end;


function valency(TOp::TransverseOps)::Integer 
    @assert false "Calling Placeholder Abstract Function"
end;

function frames(TOp::TransverseOps)::Vector   #should be Vector{Index}
    @assert false "Calling Placeholder Abstract Function"
end;

function framesTemporary(TOp::TransverseOps)::Vector  #should be Vector{Index}
    @assert false "Calling Placeholder Abstract Function"
end;

function reduceByEngaged(TOp::TransverseOps, engaged::Vector{Bool})::Tuple{TransverseOps, LinearMaps.LinearMap} 
    @assert false "Calling Placeholder Abstract Function"
end;


# no needed
# function engaged(TOp::TransverseOps)::Vector{Bool} 
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

function __asMatrixTranspose(T::ITensor)::AbstractMatrix
    fr = inds(T)
    n = ITensors.dim(fr[1])
    m = ITensors.dim(fr[2])
    A = zeros(eltype(T),m,n)
    for ci in CartesianIndices(A)
        A[ci] = T[ci[2],ci[1]]
    end
    return A
end;

# function to create temp index
function __globalOpsMakeTempIndex(I::Index)::Index
    return Index(ITensors.dim(I),"Site:Der,$I")
end;
