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
    List Operators Independent

    List of Operators which are product of independent operators on each axis 
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
struct IndListOperators <: ListOperators
    val ::Integer
    axisDims ::Vector{<:Integer}
    globalDim ::Integer
    soffsets ::Vector{<:Integer}
    eoffsets ::Vector{<:Integer}
    localOps ::Vector{<:Operator} 
    isLinear ::Bool
    isInvertible::Bool
    #inner constructor
    # we need ; after each line???
    IndListOperators(axisDims::Vector{<:Integer}, localOps ::Vector{<:Operator}) = (
        val =  length(axisDims);
        @assert val == length(localOps) "Incompatable data";
        @assert val > 0 "Do not accept Valency 0"; 
        localDims =[ localDim(localOps[i], axisDims[i]) for i=1:val];
        globalDim = sum(localDims);
        offsets = [ sum(localDims[1:(i-1)]) for i=1:(val+1)];
        start_offset = [ offsets[i] + 1 for i=1:val];
        end_offset = [ offsets[i + 1] for i=1:val];
        isLinear = all( localOps.|> Op -> isa(Op, LinearOperator));
        isInvertible = all( localOps.|> Op -> isa(Op, InvertableOperator));
        new(val,axisDims,globalDim,start_offset,end_offset,localOps,isLinear,isInvertible)
    )
end; 


globalDim(LOp::IndListOperators)::Integer  = LOp.globalDim;

axisDims(LOp::IndListOperators)::Vector{<:Integer} = LOp.axisDims;

valency(LOp::IndListOperators)::Integer = LOp.val

unsafe_embedMatrices(LOp::IndListOperators, data::Vector{<:Number} ) ::Vector{<:AbstractMatrix} = 
    # [ unsafe_embed(LOp.localOps[i],LOp.axisDims[i],data[(LOp.offsets[i]+1):LOp.offsets[i+1]]) for i=1:LOp.val];
    [ 
        unsafe_embed(
            LOp.localOps[i],
            LOp.axisDims[i],
            @inbounds data[LOp.soffsets[i]:LOp.eoffsets[i]]
        ) 
        for i=1:LOp.val
    ];
function embedMatrices(LOp::IndListOperators, data::Vector{<:Number} ) :: Union{Vector{<:AbstractMatrix}, Nothing}
    length(data) == LOp.globalDim || return nothing
    res = [ 
        unsafe_embed(
            LOp.localOps[i],
            LOp.axisDims[i],
            @inbounds data[LOp.soffsets[i]:LOp.eoffsets[i]]
        ) 
        for i=1:LOp.val
    ];
    any(res .|> isnothing) && return nothing
    return res
end;


unsafe_transposeEmbed(LOp::IndListOperators, Mats::Vector{<:AbstractMatrix}) :: Vector{<:Number} =
    vcat([ unsafe_transposeEmbed(LOp.localOps[i], Mats[i]) for i=1:LOp.val ]...);

function transposeEmbed(LOp::IndListOperators, Mats::Vector{<:AbstractMatrix}) :: Union{Vector{<:Number}, Nothing}
    if LOp.isLinear 
        return vcat([ unsafe_transposeEmbed(LOp.localOps[i], Mats[i]) for i=1:LOp.val ]...)
    else
        return nothing
    end
end;

unsafe_coordinates(LOp::IndListOperators, Mats::Vector{<: AbstractMatrix} ) :: Vector{<: Number} =
    vcat([ unsafe_coordinates(LOp.localOps[i], Mats[i]) for i=1:LOp.val ]...);


function coordinates(LOp::IndListOperators, Mats::Vector{<: AbstractMatrix} ) :: Union{Vector{<:Number}, Nothing}
    all([size(Mats[i])[1] == LOp.axisDims[i] for i=1:LOp.val]) || return nothing
    res= [coordinates(LOp.localOps[i],Mats[i]) for i=1:LOp.val]
    any(res .|> isnothing) && return nothing
    return vcat(res...)
end;


function reduceBy(LOp::IndListOperators, engaged::Vector{Bool})::NamedTuple{(:rLOp, :expand_func, :reduce_map), Tuple{ListOperators, Function, LinearMaps.LinearMap}}
    @assert LOp.val == length(engaged) "Incompatible data"
    @assert any(engaged) "Can not reduce to Nothing"
    rOp = IndListOperators(LOp.axisDims[engaged],LOp.localOps[engaged])
    rindx = zeros(Int16, LOp.val);
    eindx = zeros(Int16, rOp.val);
    j=1 
    for i = 1:LOp.val
        if engaged[i]
            rindx[i] = j
            eindx[j] = i 
            j = j+1
        else
            rindx[i] = 0
        end
    end 

    function expand(rdata::Vector{<:Number})::Vector{<:Number}
        edata=zeros(eltype(rdata), LOp.globalDim)
        for i = 1: rOp.val
            # @inbounds edata[LOp.soffsets[eindx[i]]:LOp.eoffsets[eindx[i]]] = rdata[rOp.soffsets[i]:rOp.eoffsets[i]]
            edata[LOp.soffsets[eindx[i]]:LOp.eoffsets[eindx[i]]] = rdata[rOp.soffsets[i]:rOp.eoffsets[i]]
        end;
        return edata
    end;
    contract(edata::Vector{<:Number})::Vector{<:Number} = 
        vcat([ @inbounds edata[LOp.soffsets[eindx[i]]:LOp.eoffsets[eindx[i]] ] for i= 1: rOp.val ]...); 
    return (rLOp= rOp, expand_func= expand, 
        reduce_map=LinearMaps.LinearMap(contract, expand, rOp.globalDim, LOp.globalDim; ismutating=false) )
end;



isLinear(LOp::IndListOperators)::Bool = LOp.isLinear;

isInvertible(LOp::IndListOperators)::Bool = LOp.isInvertible;

generate_random(LOp::IndListOperators)::Vector{<:Number} = 
    vcat(
        [
            generate_random(LOp.localOps[i],LOp.axisDims[i])
            for i=1:LOp.val
        ]...
    );

function simplifyTo(LOp::IndListOperators)::NamedTuple{(:D, :T), Tuple{ListOperators,ListOperators}}  
    @assert LOp.isLinear "Only defined for Linear Operators"
    DlocalOps = LOp.localOps .|> simplifyTo .|> x -> x.D 
    TlocalOps = LOp.localOps .|> simplifyTo .|> x -> x.T
    DLOp = IndListOperators(LOp.axisDims, DlocalOps) 
    TLOp = IndListOperators(LOp.axisDims, TlocalOps)
    return (D=DLOp, T=TLOp) 
end;


function simplify(LOp::IndListOperators, data::Vector{<:Number} )::NamedTuple{(:d, :t), Tuple{Vector{<:Number},Vector{<:Number}} } 
    @assert LOp.isLinear "Only defined for Linear Operators"
    @assert length(data)==LOp.globalDim
    localres = [ 
                    simplfy(
                        LOp.localOps[i], LOp.axisDims[i],
                        @inbounds data[LOp.soffsets[i]:LOp.eoffsets[i]])
                for i=1:LOp.val];
    data_d = localres .|> x -> x.d
    data_t = localres .|> x -> x.t
    return (d= vcat(data_d...), t= vcat(data_t...))
end;

function scalarsMatrix(LOp::IndListOperators)::Matrix{<:Number}
    @assert isLinear(LOp) "Only defined for Linear Operators"
    sc = [ containScalars(LOp.localOps[i]) for i = 1:LOp.val]
    d = sum(sc)
    A = zeros(d,LOp.val)
    k=1
    for i=1:LOp.val
        if sc[i]
            A[k,i] = 1
            k = k+1
        end
    end
    # @show sc,d,LOp.val, size(A)
    # @show A
    return A
end