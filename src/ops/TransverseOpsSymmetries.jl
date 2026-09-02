#
# Strata Dleto: Global Operators with Symetries/Duals
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
    Transverse Operators Symmetries

    Transverse Operators which are product of independent operators on each axis synbect to summtery restrictions.
    Input: array like [1 2 -2 4 4] saying that the third coordinated is dual to the second and the 4th and 5th are the same 
    stores info for each axis 
        - as Index in frames 
        - as tempIndex in framesTemp
        - dimension in axisDim
        - local operator in localOps
        - syms array like [1 2 -2 4 4] saying that the third coordinated is dual to the second and the 4th and 5th are the same
    in addition precomputes the transverse dimenstion and the offsets (in the inner constructor)

    the embed function dispatches the corresponding embedding from the local operator on the view of the input data
    the coordinates and transposeEmbed dispatches the corresponding function 
        from the local operator to the correct matrix and then combines the results    
"""
struct TransverseOpsSymmetries <: TransverseOps
    val ::Integer
    frames ::Vector{Index{K}} where K
    framesTemp ::Vector{Index{KK}} where KK
    axisDims ::Vector{<:Integer}
    globalDim ::Integer
    soffsets ::Vector{<:Integer}
    eoffsets ::Vector{<:Integer}
    localOps ::Vector{<:Operator}
    syms ::Vector{<:Integer}
    blocks ::Vector{Vector{Int8}}
    duals ::Vector{Bool} 
    #inner constructor
    # we need ; after each line???
    TransverseOpsSymmetries(
        fr ::Vector{Index{K}} where K, 
        frTemp ::Vector{Index{KK}} where KK, 
        localOps ::Vector{<:Operator},
        symmetries ::Vector{<:Integer}, 
        duals :: Union{Vector{Bool},BitVector}) = (
        val =  length(fr);
        @assert val > 0 "Do not accept Valency 0"; 
        @assert val == length(localOps) "Incompatable data";
        @assert (fr .|> ITensors.dim) == (frTemp .|> ITensors.dim) "Incompatable dimensions";
        axisDims = fr .|> ITensors.dim;
        @assert val == length(symmetries) "Incompatable data";
        @assert val == length(duals) "Incompatable data";
        @assert all(  symmetries .|> x -> x>0 ) "Incompatable data";
        @assert all( [symmetries[i] <= i for i=1:val]) "Incompatable symmetries";
        @assert all( [symmetries[symmetries[i]] == symmetries[i] for i=1:val]) "Incompatable symmetries";
        @assert all( [!duals[i] for i=1:val if i==symmetries[i]]) "Incompatable duals";
        blocks = [ Int8[j for j=1:val if  symmetries[j]==i] for i=1:val];
        @assert all([ axisDims[i] == axisDims[symmetries[i]]  for i=1:val if i==symmetries[i]]) "Incompatable local dimensions";
        @assert all([ localOps[i] == localOps[symmetries[i]]  for i=1:val if i!=symmetries[i]]) "Incompatable local operators";

        localDims =[ symmetries[i] == i ? localDim(localOps[i], axisDims[i]) : 0 for i=1:val];
        globalDim = sum(localDims);
        offsets = [ sum(localDims[1:(i-1)]) for i=1:(val+1)]; 
        start_offsets = [ offsets[symmetries[i]] + 1 for i=1:val]; 
        end_offsets = [ offsets[symmetries[i] + 1]  for i=1:val]; 
        new(val,fr,frTemp,axisDims,globalDim,start_offsets,end_offsets,localOps,symmetries,blocks,duals)
    )
end; 
#extra constructors need to add more
# signed symmetries
TransverseOpsSymmetries(
    fr ::Vector{Index{K}} where K, 
    frTemp ::Vector{Index{KK}} where KK, 
    localOps ::Vector{<:Operator},
    symmetries ::Vector{<:Integer}) = TransverseOpsSymmetries(fr, frTemp,localOps, symmetries .|> abs, symmetries .|> x -> x < 0); 
# auto generate new indexes
TransverseOpsSymmetries(fr ::Vector{Index{K}} where K, localOps ::Vector{<:Operator},symmetries ::Vector{<:Integer}) = 
    TransverseOpsSymmetries(fr, fr .|> __globalOpsMakeTempIndex ,localOps, symmetries);
#same localOp on each axis
TransverseOpsSymmetries(fr ::Vector{Index{K}} where K, frTemp ::Vector{Index{KK}} where KK, localOp ::Operator,symmetries ::Vector{<:Integer}) = 
    TransverseOpsSymmetries(fr, frTemp, fr .|> (x -> localOp), symmetries);
TransverseOpsSymmetries(fr ::Vector{Index{K}} where K, localOp ::Operator) = 
    TransverseOpsSymmetries(fr, fr .|> (x -> localOp), symmetries);


## #TO BE DONE

globalDim(GΩ::TransverseOpsSymmetries)::Integer  = GΩ.globalDim;

axisDims(GΩ::TransverseOpsSymmetries)::Vector{<:Integer} = GΩ.axisDims;

valency(GΩ::TransverseOpsSymmetries)::Integer = GΩ.val

frames(GΩ::TransverseOpsSymmetries) = GΩ.frames

framesTemporary(GΩ::TransverseOpsSymmetries) = GΩ.framesTemp


unsafe_embedMatrices(GΩ::TransverseOpsSymmetries, data::AbstractVector{<:Number} ) ::Vector{<:AbstractMatrix} = 
    [   GΩ.duals[i] ? 
            transpose(unsafe_embed(
                GΩ.localOps[i],
                GΩ.axisDims[i],
                data[GΩ.soffsets[i]:GΩ.eoffsets[i]]
            )) : 
            unsafe_embed(
                GΩ.localOps[i],
                GΩ.axisDims[i],
                data[GΩ.soffsets[i]:GΩ.eoffsets[i]]
            ) 
        for i=1:GΩ.val
    ];

unsafe_embedITensors(GΩ::TransverseOpsSymmetries, data::AbstractVector{<:Number} ) ::Vector{<:ITensor} = 
    [ ITensor(
        GΩ.duals[i] ? 
            # without matrix ITensor messes up the encoding
            Matrix(transpose(unsafe_embed(
                GΩ.localOps[i],
                GΩ.axisDims[i],
                data[GΩ.soffsets[i]:GΩ.eoffsets[i]]
            ))) : 
            unsafe_embed(
                GΩ.localOps[i],
                GΩ.axisDims[i],
                data[GΩ.soffsets[i]:GΩ.eoffsets[i]]),
        GΩ.frames[i],GΩ.framesTemp[i] )
    for i=1:GΩ.val
    ];

# The swapped orientation.  This was missing entirely: IndTransverseOps has it
# (TransverseOpsIndependant.jl:102) and the symmetries version fell through to
# the abstract placeholder, which asserts.  `sylvesterLM` applies this
# embedding inside `ester`, so no derivation could be solved with
# symmetry-restricted operators at all -- construction and plain embedding
# worked, solving did not.
unsafe_embedITensorsSwapped(GΩ::TransverseOpsSymmetries, data::AbstractVector{<:Number} ) ::Vector{<:ITensor} =
    [ ITensor(
        GΩ.duals[i] ?
            Matrix(transpose(unsafe_embed(
                GΩ.localOps[i],
                GΩ.axisDims[i],
                data[GΩ.soffsets[i]:GΩ.eoffsets[i]]
            ))) :
            unsafe_embed(
                GΩ.localOps[i],
                GΩ.axisDims[i],
                data[GΩ.soffsets[i]:GΩ.eoffsets[i]]),
        GΩ.framesTemp[i],GΩ.frames[i] )
    for i=1:GΩ.val
    ];

    #rewrote to avoid loops
unsafe_transposeEmbed(GΩ::TransverseOpsSymmetries, Mats::Vector{<:AbstractMatrix}) :: AbstractVector{<:Number} = 
    vcat( 
        [ 
            [ unsafe_transposeEmbed(
                    GΩ.localOps[j], 
                    GΩ.duals[j] ? Matrix(transpose(Mats[j])) : Matrix(Mats[j])
                ) 
            for j in GΩ.blocks[i] ] |> sum 
        for i=1:GΩ.val if GΩ.syms[i]==i ] ...
    ); 

unsafe_coordinates(GΩ::TransverseOpsSymmetries, Mats::Vector{<: AbstractMatrix} ) :: AbstractVector{<: Number} =
    vcat([ unsafe_coordinates(GΩ.localOps[i], Mats[i])  for i=1:GΩ.val if GΩ.syms[i]==i  ]...);


function coordinates(GΩ::TransverseOpsSymmetries, Mats::Vector{<: AbstractMatrix} ) :: Union{AbstractVector{<:Number}, Nothing}
    all([size(Mats[i])[1] == GΩ.axisDims[i] for i=1:GΩ.val]) || return nothing
    res= [coordinates(GΩ.localOps[i], GΩ.duals[i] ? Matrix(transpose(Mats[i])) : Matrix(Mats[i])) for i=1:GΩ.val]
    any(res .|> isnothing) && return nothing
    all([ isapprox(res[i], res[GΩ.syms[i]]) for i=1:GΩ.val if GΩ.syms[i] < i ]) || return nothing
    return vcat([res[i] for i=1:GΩ.val if GΩ.syms[i] == i ] ...)
end;

#Not Finished
function reduceByEngaged(GΩ::TransverseOpsSymmetries, engaged::Vector{Bool})::Tuple{TransverseOps, LinearMaps.LinearMap}
    val=GΩ.val
    syms=GΩ.syms
    duals=GΩ.duals
    blocks=GΩ.blocks
    @assert val == length(engaged) "Incompatible data"
    @assert any(engaged) "Can not reduce to Nothing"
    minblock = [ minimum(vcat(1000, [j for j in blocks[k] if engaged[j] ]...)) for k =1:val]
    renumber = [sum(engaged[1:k]) for k =1:val]
    rsyms = [ renumber[minblock[syms[k]]] for k =1:val if engaged[k] ]
    rduals = [ xor(duals[k], duals[minblock[syms[k]]] ) for k =1:val if engaged[k]]
    twisted = [ duals[minblock[syms[k]]]  for k =1:val if engaged[k]]
    c_idx = [ syms[k] for k =1:val if minblock[syms[k]]==k]
    c_twist = [ duals[k] for k =1:val if minblock[syms[k]]==k]
    c_renum = [ renumber[k] for k =1:val if minblock[syms[k]]==k]
    # @show syms,duals,engaged
    # @show rsyms, rduals, twisted
    # @show c_idx, c_twist, c_renum
    ## generate new GlobalOps
    # if all( [i==rsyms[i]  for i=1:length(rsyms)])
    #     rΩ = IndTransverseOps(GΩ.frames[engaged], GΩ.framesTemp[engaged], GΩ.localOps[engaged]) 
    # else
        rΩ = TransverseOpsSymmetries(
            GΩ.frames[engaged], GΩ.framesTemp[engaged], GΩ.localOps[engaged], 
            rsyms,rduals) 
    # end 
    function expand(rdata::AbstractVector{<:Number})::AbstractVector{<:Number}
        edata=zeros(eltype(rdata), GΩ.globalDim)
        for i= 1: length(c_idx)
            edata[GΩ.soffsets[c_idx[i]]:GΩ.eoffsets[c_idx[i]] ] = 
                c_twist[i] ?
                    unsafe_dualize(
                            rΩ.localOps[c_renum[i]],
                            rΩ.axisDims[c_renum[i]],
                            rdata[rΩ.soffsets[c_renum[i]]:rΩ.eoffsets[c_renum[i]] ]) :
                    rdata[rΩ.soffsets[c_renum[i]]:rΩ.eoffsets[c_renum[i]] ]
        end;
        return edata
    end;
    contract(edata::AbstractVector{<:Number})::AbstractVector{<:Number} = 
        vcat(
                [ 
                    c_twist[i] ? 
                        unsafe_dualize(
                            GΩ.localOps[c_idx[i]],
                            GΩ.axisDims[c_idx[i]],
                            edata[GΩ.soffsets[c_idx[i]]:GΩ.eoffsets[c_idx[i]] ]) : 
                        edata[GΩ.soffsets[c_idx[i]]:GΩ.eoffsets[c_idx[i]] ] 
                for i= 1: length(c_idx)]...
            ); 
    return (rΩ, LinearMaps.LinearMap(expand, contract, GΩ.globalDim, rΩ.globalDim; ismutating=false) )
end;