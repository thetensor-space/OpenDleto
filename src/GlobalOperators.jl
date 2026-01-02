#
# Strata Dleto: Global Operators
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
    Global Operators

    TO BE WRITTEN
"""

"""
    Global Operators

    TO BE WRITTEN

"""
struct GlobalOps <: AbstractGlobalOps
    val ::Integer
    frames ::Vector{Index{K}} where K
    framesTemp ::Vector{Index{KK}} where KK
    axisDims ::Vector{<:Integer}
    globalDim ::Integer
    offsets ::Vector{<:Integer}
    localOps ::Vector{<:LocalOps} 
    #inner constructor
    # we need ; after each line???
    GlobalOps(fr ::Vector{Index{K}} where K, localOps ::Vector{<:LocalOps}) = (
        val =  length(fr);
        @assert val == length(localOps);
        frTemp = fr .|> __makeTempIndex;
        axisDims = fr .|> ITensors.dim;
        localDims =[ localDim(localOps[i], axisDims[i]) for i=1:val];
        globalDim = sum(localDims);
        offsets = [ sum(localDims[1:(i-1)]) for i=1:(val+1)]; 
        new(val,fr,frTemp,axisDims,globalDim,offsets,localOps)
    )
end; 

#Todo function reduceByEngaged TO BE IMPLEMENTED

export GlobalOps

globalDim(GΩ::GlobalOps)::Integer  = GΩ.globalDim;

axisDims(GΩ::GlobalOps)::Vector{<:Integer} = GΩ.axisDims;

valancy(GΩ::GlobalOps)::Integer = GΩ.val

frames(GΩ::GlobalOps) = GΩ.frames

framesTemporary(GΩ::GlobalOps) = GΩ.framesTemp

unsafe_embeddingMatrices(GΩ::GlobalOps, data::Vector{<:Number} ) ::Vector{<:AbstractMatrix} = 
    [ unsafe_embedding(GΩ.localOps[i],GΩ.axisDims[i],data[(GΩ.offsets[i]+1):GΩ.offsets[i+1]]) for i=1:GΩ.val];

unsafe_embeddingITensors(GΩ::GlobalOps, data::Vector{<:Number} ) ::Vector{<:ITensor} = 
    [ ITensor(
        unsafe_embedding(GΩ.localOps[i],GΩ.axisDims[i],data[(GΩ.offsets[i]+1):GΩ.offsets[i+1]]),
        GΩ.frames[i],GΩ.framesTemp[i] ) 
        for i=1:GΩ.val
    ];

unsafe_transposeEmbedding(GΩ::GlobalOps, Mats::Vector{<:AbstractMatrix}) :: Vector{<:Number} =
    vcat([ unsafe_transposeEmbedding(GΩ.localOps[i], Mats[i]) for i=1:GΩ.val ]...);

unsafe_coordinates(GΩ::GlobalOps, Mats::Vector{<: AbstractMatrix} ) :: Vector{<: Number} =
    vcat([ unsafe_coordinates(GΩ.localOps[i], Mats[i]) for i=1:GΩ.val ]...);


function coordinates(GΩ::GlobalOps, Mats::Vector{<: AbstractMatrix} ) :: Union{Vector{<:Number}, Nothing}
    all([size(Mats[i])[1] == GΩ.axisDims[i] for i=1:GΩ.val]) || return nothing
    res= [coordinates(GΩ.localOps[i],Mats[i]) for i=1:GΩ.val]
    any(res .|> isnothing) && return nothing
    return vcat(res...)
end;


function reduceByEngaged(GΩ::GlobalOps, engaged::Vector{Bool})::AbstractGlobalOps 
    @assert false "Calling Placeholder Abstract Function"
end;


function __makeTempIndex(I::Index)::Index
    return I'
end;