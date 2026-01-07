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

    Transverse Operators which are product of independent operators on each axis 
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
    offsets ::Vector{<:Integer}
    localOps ::Vector{<:Operator}
    syms ::Vector{<:Integer}
    duals ::Vector{Bool} 
    #inner constructor
    # we need ; after each line???
    TransverseOpsSymmetries(
        fr ::Vector{Index{K}} where K, 
        frTemp ::Vector{Index{KK}} where KK, 
        localOps ::Vector{<:Operator},
        symmetries ::Vector{<:Integer} ) = (
        val =  length(fr);
        @assert val == length(localOps) "Incompatable data";
        @assert (fr .|> ITensors.dim) == (frTemp .|> ITensors.dim) "Incompatable dimensions";
        axisDims = fr .|> ITensors.dim;
        @assert val == length(symmetries) "Incompatable data";
        syms = symmetries .|> abs;
        @assert all([ (syms[i] <= i) && (syms[i] >=1 ) && (symmetries[i] != -i)   for i=1:val]) "Incompatable dimensions";
        duals = symmetries .|> (x -> x < 0 );
        @assert all([ axisDims[i] == axisDims[syms[i]]    for i=1:val]) "Incompatable local dimensions";
        @assert all([ localOps[i] == localOps[syms[i]]    for i=1:val]) "Incompatable local operators";

        localDims =[ syms[i] == i ? localDim(localOps[i], axisDims[i]) : 0 for i=1:val];
        globalDim = sum(localDims);
        offsets = [ sum(localDims[1:(i-1)]) for i=1:(val+1)]; 
        new(val,fr,frTemp,axisDims,globalDim,offsets,localOps,syms,duals)
    )
end; 
#extra constructors
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


unsafe_embedMatrices(GΩ::TransverseOpsSymmetries, data::Vector{<:Number} ) ::Vector{<:AbstractMatrix} = 
    [   GΩ.duals[i] ? 
            transpose(unsafe_embed(
                GΩ.localOps[i],
                GΩ.axisDims[i],
                data[(GΩ.offsets[GΩ.syms[i]]+1):GΩ.offsets[GΩ.syms[i]+1]]
            )) : 
            unsafe_embed(
                GΩ.localOps[i],
                GΩ.axisDims[i],
                data[(GΩ.offsets[GΩ.syms[i]]+1):GΩ.offsets[GΩ.syms[i]+1]]
            ) 
        for i=1:GΩ.val
    ];

unsafe_embedITensors(GΩ::TransverseOpsSymmetries, data::Vector{<:Number} ) ::Vector{<:ITensor} = 
    [ ITensor(
        GΩ.duals[i] ? 
            # without matrix ITensor messes up the encoding
            Matrix(transpose(unsafe_embed(
                GΩ.localOps[i],
                GΩ.axisDims[i],
                data[(GΩ.offsets[GΩ.syms[i]]+1):GΩ.offsets[GΩ.syms[i]+1]]
            ))) : 
            unsafe_embed(
                GΩ.localOps[i],
                GΩ.axisDims[i],
                data[(GΩ.offsets[GΩ.syms[i]]+1):GΩ.offsets[GΩ.syms[i]+1]]),
        GΩ.frames[i],GΩ.framesTemp[i] ) 
    for i=1:GΩ.val
    ];

function unsafe_transposeEmbed(GΩ::TransverseOpsSymmetries, Mats::Vector{<:AbstractMatrix}) :: Vector{<:Number} 
    ltEmbedding = [ 
            unsafe_transposeEmbed(
                GΩ.localOps[i], 
                GΩ.duals[i] ? Matrix(transpose(Mats[i])) : Matrix(Mats[i])
            ) 
        for i=1:GΩ.val ]
    res = zeros(GΩ.globalDim)
    for i = 1:GΩ.val
        res[(GΩ.offsets[GΩ.syms[i]]+1):GΩ.offsets[GΩ.syms[i]+1]] += ltEmbedding[i]
    end 
    return res;
end;

unsafe_coordinates(GΩ::TransverseOpsSymmetries, Mats::Vector{<: AbstractMatrix} ) :: Vector{<: Number} =
    vcat([ GΩ.syms[i]==i ? unsafe_coordinates(GΩ.localOps[i], Mats[i]) : zeros(0) for i=1:GΩ.val ]...);


function coordinates(GΩ::TransverseOpsSymmetries, Mats::Vector{<: AbstractMatrix} ) :: Union{Vector{<:Number}, Nothing}
    all([size(Mats[i])[1] == GΩ.axisDims[i] for i=1:GΩ.val]) || return nothing
    res= [coordinates(GΩ.localOps[i], GΩ.duals[i] ? Matrix(transpose(Mats[i])) : Matrix(Mats[i])) for i=1:GΩ.val]
    any(res .|> isnothing) && return nothing
    # if any(res .|> isnothing)
    #     @show res .|> isnothing
    #     return nothing
    # end
    # for i=1:GΩ.val
    #     if GΩ.syms[i] != i && !isapprox(res[i], res[GΩ.syms[i]])
    #         @show i, GΩ.syms, GΩ.duals
    #         @show Mats[i], Mats[GΩ.syms[i]]
    #         @show Mats
    #     end
    # end
    filteredres= [GΩ.syms[i] == i ? res[i] : (isapprox(res[i], res[GΩ.syms[i]]) ? zeros(0) : nothing) for i=1:GΩ.val ]
    # filteredres= [GΩ.syms[i] == i ? res[i] : (isapprox(res[i], res[GΩ.syms[i]]) ? zeros(0) : zeros(0)) for i=1:GΩ.val ]
    any(filteredres .|> isnothing) && return nothing
    # if any(filteredres .|> isnothing)
    #     @show filteredres .|> isnothing
    #     return nothing
    # end
    return vcat(filteredres...)
end;

#Needs Testing
function reduceByEngaged(GΩ::TransverseOpsSymmetries, engaged::Vector{Bool})::TransverseOps 
    val=GΩ.val
    syms=GΩ.syms
    duals=GΩ.duals
    @assert val == length(engaged) "Incompatible data"

    renumsyms= [ minimum( [ syms[j]== k && engaged[j] ? j : 1000 for j = k:val ])  for k =1:val]
    # newsyms = [ engaged[k] ? renumsyms[syms[k]] : 1000 for k = 1:val ]
    newsyms = [ renumsyms[syms[k]]  for k = 1:val ]
    fixdual = [ engaged[k] ? xor(duals[k], duals[newsyms[k] ] ) :  true for k =1:val ] 
    renumber = [sum(engaged[1:k]) for k =1:val]
    newsymmetries_all = [engaged[k] ? (fixdual[k] ? -renumber[newsyms[k]] : renumber[newsyms[k]]) : 1000  for k =1:val]
    newsymmetries = newsymmetries_all[engaged] 
    ## generate new GlobalOps 
    return all( [i==newsymmetries[i]  for i=1:length(newsymmetries)]) ? 
        TransverseOpsIndependant(GΩ.frames[engaged], GΩ.framesTemp[engaged], GΩ.localOps[engaged]) : 
        TransverseOpsSymmetries(GΩ.frames[engaged], GΩ.framesTemp[engaged], GΩ.localOps[engaged], newsymmetries)
end;