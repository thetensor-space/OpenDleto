#
# Strata Dleto: Transverse Operators
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


"""
    Transverse Operators Independent

    Transverse Operators which are product of independent operators on each axis 
    stores info for each axis 
        - as Index in frames 
        - as tempIndex in framesTemp
        - dimension in axisDim
        - local operator in localOps
    in addition pre-computes the Global dimension and the offsets (in the inner constructor)

    the embed function dispatches the corresponding embed from the local operator on the view of the input data
    the coordinates and transposeEmbed dispatches the corresponding function 
        from the local operator to the correct matrix and then combines the results    
"""
struct IndTransverseOps <: TransverseOps
    val ::Integer
    frames ::Vector{Index{K}} where K
    framesTemp ::Vector{Index{KK}} where KK
    axisDims ::Vector{<:Integer}
    globalDim ::Integer
    soffsets ::Vector{<:Integer}
    eoffsets ::Vector{<:Integer}
    localOps ::Vector{<:Operator} 
    #inner constructor
    # we need ; after each line???
    IndTransverseOps(fr ::Vector{Index{K}} where K, frTemp ::Vector{Index{KK}} where KK, localOps ::Vector{<:Operator}) = (
        val =  length(fr);
        @assert val == length(localOps) "Incompatable data";
        @assert (fr .|> ITensors.dim) == (frTemp .|> ITensors.dim) "Incompatable dimensions";
        @assert val > 0 "Do not accept Valency 0"; 
        axisDims = fr .|> ITensors.dim;
        localDims =[ localDim(localOps[i], axisDims[i]) for i=1:val];
        globalDim = sum(localDims);
        offsets = [ sum(localDims[1:(i-1)]) for i=1:(val+1)];
        start_offset = [ offsets[i] + 1 for i=1:val];
        end_offset = [ offsets[i + 1] for i=1:val];
        new(val,fr,frTemp,axisDims,globalDim,start_offset,end_offset,localOps)
    )
end; 
#extra constructors
# auto generate new indexes
IndTransverseOps(fr ::Vector{Index{K}} where K, localOps ::Vector{<:Operator}) = 
    IndTransverseOps(fr, fr .|> __globalOpsMakeTempIndex ,localOps);
#same localOp on each axis
IndTransverseOps(fr ::Vector{Index{K}} where K, frTemp ::Vector{Index{KK}} where KK, localOp ::Operator) = 
    IndTransverseOps(fr, frTemp, fr .|> (x -> localOp) );
IndTransverseOps(fr ::Vector{Index{K}} where K, localOp ::Operator) = 
    IndTransverseOps(fr, fr .|> (x -> localOp) );


globalDim(TOp::IndTransverseOps)::Integer  = TOp.globalDim;

axisDims(TOp::IndTransverseOps)::Vector{<:Integer} = TOp.axisDims;

valency(TOp::IndTransverseOps)::Integer = TOp.val

frames(TOp::IndTransverseOps) = TOp.frames
framesTemporary(TOp::IndTransverseOps) = TOp.framesTemp

unsafe_embedMatrices(TOp::IndTransverseOps, data::Vector{<:Number} ) ::Vector{<:AbstractMatrix} = 
    # [ unsafe_embed(TOp.localOps[i],TOp.axisDims[i],data[(TOp.offsets[i]+1):TOp.offsets[i+1]]) for i=1:TOp.val];
    [ unsafe_embed(TOp.localOps[i],TOp.axisDims[i],data[TOp.soffsets[i]:TOp.eoffsets[i]]) for i=1:TOp.val];

unsafe_embedITensors(TOp::IndTransverseOps, data::Vector{<:Number} ) ::Vector{<:ITensor} = 
    [ ITensor(
        Matrix(unsafe_embed(
            TOp.localOps[i],
            TOp.axisDims[i],
            data[TOp.soffsets[i]:TOp.eoffsets[i]])),
        TOp.frames[i],TOp.framesTemp[i] ) 
        for i=1:TOp.val
    ];

unsafe_embedITensorsSwapped(TOp::IndTransverseOps, data::Vector{<:Number} ) ::Vector{<:ITensor} = 
    [ ITensor(
        Matrix(unsafe_embed(
            TOp.localOps[i],
            TOp.axisDims[i],
            data[TOp.soffsets[i]:TOp.eoffsets[i]])),
        TOp.framesTemp[i],TOp.frames[i] ) 
        for i=1:TOp.val
    ];

unsafe_transposeEmbed(TOp::IndTransverseOps, Mats::Vector{<:AbstractMatrix}) :: Vector{<:Number} =
    vcat([ unsafe_transposeEmbed(TOp.localOps[i], Mats[i]) for i=1:TOp.val ]...);

unsafe_coordinates(TOp::IndTransverseOps, Mats::Vector{<: AbstractMatrix} ) :: Vector{<: Number} =
    vcat([ unsafe_coordinates(TOp.localOps[i], Mats[i]) for i=1:TOp.val ]...);


function coordinates(TOp::IndTransverseOps, Mats::Vector{<: AbstractMatrix} ) :: Union{Vector{<:Number}, Nothing}
    all([size(Mats[i])[1] == TOp.axisDims[i] for i=1:TOp.val]) || return nothing
    res= [coordinates(TOp.localOps[i],Mats[i]) for i=1:TOp.val]
    any(res .|> isnothing) && return nothing
    return vcat(res...)
end;


function reduceByEngaged(TOp::IndTransverseOps, engaged::Vector{Bool})::Tuple{TransverseOps, LinearMaps.LinearMap}
    @assert TOp.val == length(engaged) "Incompatible data"
    @assert any(engaged) "Can not reduce to Nothing"
    rOp = IndTransverseOps(TOp.frames[engaged],TOp.framesTemp[engaged],TOp.localOps[engaged])
    rindx = zeros(Int16, TOp.val);
    eindx = zeros(Int16, rOp.val);
    j=1 
    for i = 1:TOp.val
        if engaged[i]
            rindx[i] = j
            eindx[j] = i 
            j = j+1
        else
            rindx[i] = 0
        end
    end 

    function expand(rdata::Vector{<:Number})::Vector{<:Number}
        edata=zeros(eltype(rdata), TOp.globalDim)
        for i = 1: rOp.val
            edata[TOp.soffsets[eindx[i]]:TOp.eoffsets[eindx[i]]] = rdata[rOp.soffsets[i]:rOp.eoffsets[i]]
        end;
        return edata
    end;
    contract(edata::Vector{<:Number})::Vector{<:Number} = 
        vcat([ edata[TOp.soffsets[eindx[i]]:TOp.eoffsets[eindx[i]] ] for i= 1: rOp.val ]...); 
    # function contract(edata::Vector{<:Number})::Vector{<:Number}
    #     rdata=zeros(eltype(edata), rΩ.globalDim)
    #     for i = 1: rΩ.val
    #         rdata[rΩ.soffsets[i]:rΩ.eoffsets[i]] = edata[TOp.soffsets[eindx[i]]:TOp.eoffsets[eindx[i]] ]
    #     end;
    #     return rdata
    # end;
    return (rOp, LinearMaps.LinearMap(expand, contract, TOp.globalDim, rOp.globalDim; ismutating=false) )
end;
