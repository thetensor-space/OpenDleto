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
    Global Operators Independant

    Global Operartors whcih are product of independant operators on each axis 
"""

"""
    Global Operators Independant

    Global Operartors whcih are product of independant operators on each axis 
    stores info for each axis 
        - as Index in frames 
        - as tempIndex in framesTemp
        - dimension in axisDim
        - local operator in localOps
    in addition precomutes the Global dimenstion and the offsets (in the inner constructor)

    the embedding function dispaches the corresponding emebedding from the local operator on the view of the input data
    the coordinates and transposeEmbedding dispaches the corresponding function 
        from the local operator to the correct matrix and then combines the results    
"""
struct GlobalOpsIndependant <: AbstractGlobalOps
    val ::Integer
    frames ::Vector{Index{K}} where K
    framesTemp ::Vector{Index{KK}} where KK
    axisDims ::Vector{<:Integer}
    globalDim ::Integer
    offsets ::Vector{<:Integer}
    localOps ::Vector{<:LocalOps} 
    #inner constructor
    # we need ; after each line???
    GlobalOpsIndependant(fr ::Vector{Index{K}} where K, frTemp ::Vector{Index{KK}} where KK, localOps ::Vector{<:LocalOps}) = (
        val =  length(fr);
        @assert val == length(localOps) "Incompatable data";
        @assert (fr .|> ITensors.dim) == (frTemp .|> ITensors.dim) "Incompatable dimensions";
        axisDims = fr .|> ITensors.dim;
        localDims =[ localDim(localOps[i], axisDims[i]) for i=1:val];
        globalDim = sum(localDims);
        offsets = [ sum(localDims[1:(i-1)]) for i=1:(val+1)]; 
        new(val,fr,frTemp,axisDims,globalDim,offsets,localOps)
    )
end; 
#extra constructors
# auto generate new indexes
GlobalOpsIndependant(fr ::Vector{Index{K}} where K, localOps ::Vector{<:LocalOps}) = 
    GlobalOpsIndependant(fr, fr .|> __globalOpsMakeTempIndex ,localOps);
#same localOp on each axis
GlobalOpsIndependant(fr ::Vector{Index{K}} where K, frTemp ::Vector{Index{KK}} where KK, localOp ::LocalOps) = 
    GlobalOpsIndependant(fr, frTemp, fr .|> (x -> localOp) );
GlobalOpsIndependant(fr ::Vector{Index{K}} where K, localOp ::LocalOps) = 
    GlobalOpsIndependant(fr, fr .|> (x -> localOp) );



#Todo function reduceByEngaged TO BE IMPLEMENTED

export GlobalOpsIndependant

globalDim(GΩ::GlobalOpsIndependant)::Integer  = GΩ.globalDim;

axisDims(GΩ::GlobalOpsIndependant)::Vector{<:Integer} = GΩ.axisDims;

valancy(GΩ::GlobalOpsIndependant)::Integer = GΩ.val

frames(GΩ::GlobalOpsIndependant) = GΩ.frames

framesTemporary(GΩ::GlobalOpsIndependant) = GΩ.framesTemp

unsafe_embeddingMatrices(GΩ::GlobalOpsIndependant, data::Vector{<:Number} ) ::Vector{<:AbstractMatrix} = 
    [ unsafe_embedding(GΩ.localOps[i],GΩ.axisDims[i],data[(GΩ.offsets[i]+1):GΩ.offsets[i+1]]) for i=1:GΩ.val];

unsafe_embeddingITensors(GΩ::GlobalOpsIndependant, data::Vector{<:Number} ) ::Vector{<:ITensor} = 
    [ ITensor(
        unsafe_embedding(GΩ.localOps[i],GΩ.axisDims[i],data[(GΩ.offsets[i]+1):GΩ.offsets[i+1]]),
        GΩ.frames[i],GΩ.framesTemp[i] ) 
        for i=1:GΩ.val
    ];

unsafe_transposeEmbedding(GΩ::GlobalOpsIndependant, Mats::Vector{<:AbstractMatrix}) :: Vector{<:Number} =
    vcat([ unsafe_transposeEmbedding(GΩ.localOps[i], Mats[i]) for i=1:GΩ.val ]...);

unsafe_coordinates(GΩ::GlobalOpsIndependant, Mats::Vector{<: AbstractMatrix} ) :: Vector{<: Number} =
    vcat([ unsafe_coordinates(GΩ.localOps[i], Mats[i]) for i=1:GΩ.val ]...);


function coordinates(GΩ::GlobalOpsIndependant, Mats::Vector{<: AbstractMatrix} ) :: Union{Vector{<:Number}, Nothing}
    all([size(Mats[i])[1] == GΩ.axisDims[i] for i=1:GΩ.val]) || return nothing
    res= [coordinates(GΩ.localOps[i],Mats[i]) for i=1:GΩ.val]
    any(res .|> isnothing) && return nothing
    return vcat(res...)
end;


function reduceByEngaged(GΩ::GlobalOpsIndependant, engaged::Vector{Bool})::AbstractGlobalOps 
    @assert false "Calling Placeholder Abstract Function"
end;

# function to create temp index
function __globalOpsMakeTempIndex(I::Index)::Index
    return Index(ITensors.dim(I),"Temp index for $I")
end;