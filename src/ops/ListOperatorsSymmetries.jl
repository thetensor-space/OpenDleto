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
struct SymListOperators <: ListOperators
    val ::Integer
    axisDims ::Vector{<:Integer}
    globalDim ::Integer
    soffsets ::Vector{<:Integer}
    eoffsets ::Vector{<:Integer}
    localOps ::Vector{<:Operator}
    syms ::Vector{<:Integer}
    blocks ::Vector{Vector{Int8}}
    duals ::Vector{Bool}
    block_first::Vector{<:Integer} 
    isLinear ::Bool
    isInvertible::Bool
    #inner constructor
    # we need ; after each line???
    SymListOperators(
        axisDims::Vector{<:Integer},
        localOps ::Vector{<:Operator},
        symmetries ::Vector{<:Integer}, 
        duals :: Union{Vector{Bool},BitVector}) = (
        val =  length(axisDims);
        @assert val > 0 "Do not accept Valency 0"; 
        @assert val == length(localOps) "Incompatable data";
        @assert val == length(symmetries) "Incompatable data";
        @assert val == length(duals) "Incompatable data";
        @assert all(  symmetries .|> x -> x>0 ) "Incompatable symmetries";
        @assert all( [symmetries[i] <= i for i=1:val]) "Incompatable symmetries";
        @assert all( [symmetries[symmetries[i]] == symmetries[i] for i=1:val]) "Incompatable symmetries";
        # @assert all( [!duals[i] for i=1:val if i==symmetries[i]]) "Incompatable duals";
        @assert all( [closedUnderStar(localOps[i])   for i=1:val if duals[i] ]) "Incompatable duals/local operators";
        @assert all([ axisDims[i] == axisDims[symmetries[i]]  for i=1:val if i==symmetries[i]]) "Incompatable local dimensions";
        @assert all([ localOps[i] == localOps[symmetries[i]]  for i=1:val if i!=symmetries[i]]) "Incompatable local operators";

        blocks = [ Int8[j for j=1:val if  symmetries[j]==i] for i=1:val];
        block_first = [i for i=1:val if i==symmetries[i]];  

        localDims =[ symmetries[i] == i ? localDim(localOps[i], axisDims[i]) : 0 for i=1:val];
        globalDim = sum(localDims);
        offsets = [ sum(localDims[1:(i-1)]) for i=1:(val+1)]; 
        start_offsets = [ offsets[symmetries[i]] + 1 for i=1:val]; 
        end_offsets = [ offsets[symmetries[i] + 1]  for i=1:val]; 
        isLinear = all( localOps.|> Op -> isa(Op, LinearOperator));
        isInvertible = all( localOps.|> Op -> isa(Op, InvertableOperator));
        return new(
            val,axisDims,globalDim,
            start_offsets,end_offsets,
            localOps,
            symmetries,blocks,duals,block_first,
            isLinear,isInvertible)
    )
end; 
#extra constructors need to add more
# signed symmetries
SymListOperators(
    axisDims::Vector{<:Integer}, 
    localOps ::Vector{<:Operator},
    symmetries ::Vector{<:Integer}) = SymListOperators(axisDims, localOps, symmetries .|> abs, symmetries .|> x -> x < 0); 


## #TO BE DONE

globalDim(LOp::SymListOperators)::Integer  = LOp.globalDim;

axisDims(LOp::SymListOperators)::Vector{<:Integer} = LOp.axisDims;

valency(LOp::SymListOperators)::Integer = LOp.val



unsafe_embedMatrices(LOp::SymListOperators, data::Vector{<:Number} ) ::Vector{<:AbstractMatrix} = 
    [   LOp.duals[i] ? 
            unsafe_embed(
                LOp.localOps[i],
                LOp.axisDims[i],
                unsafe_star(LOp.localOps[i],LOp.axisDims[i],@inbounds data[LOp.soffsets[i]:LOp.eoffsets[i]])
            ) : 
            unsafe_embed(
                LOp.localOps[i],
                LOp.axisDims[i],
                @inbounds data[LOp.soffsets[i]:LOp.eoffsets[i]]
            ) 
        for i=1:LOp.val
    ];


    #rewrote to avoid loops
unsafe_transposeEmbed(LOp::SymListOperators, Mats::Vector{<:AbstractMatrix}) :: Vector{<:Number} = 
    vcat( 
        [ 
            [ LOp.duals[j] ? 
                unsafe_star(
                    LOp.localOps[i], LOp.axisDims[i], 
                    unsafe_transposeEmbed( LOp.localOps[j], Matrix(Mats[j]) )
                ) : 
                unsafe_transposeEmbed( LOp.localOps[j], Matrix(Mats[j]) )   
            for j in LOp.blocks[i] ] |> sum 
        for i in LOp.block_first ] ...
    ); 

function transposeEmbed(LOp::SymListOperators, Mats::Vector{<:AbstractMatrix}) :: Union{Vector{<:Number}, Nothing}
    if LOp.isLinear 
        return vcat( 
            [ 
                [ LOp.duals[j] ? 
                    unsafe_star(
                        LOp.localOps[i], LOp.axisDims[i], 
                        unsafe_transposeEmbed( LOp.localOps[j], Matrix(Mats[j]) )
                    ) : 
                    unsafe_transposeEmbed( LOp.localOps[j], Matrix(Mats[j]) )   
                for j in LOp.blocks[i] ] |> sum 
            for i in LOp.block_first ] ...
        )
    else
        return nothing
    end
end;

unsafe_coordinates(LOp::SymListOperators, Mats::Vector{<: AbstractMatrix} ) :: Vector{<: Number} =
    vcat(
            [                 
                LOp.duals[i] ?
                unsafe_star( 
                    LOp.localOps[i], LOp.axisDims[i], 
                    unsafe_coordinates(LOp.localOps[i], Mats[i]) 
                ) : 
                unsafe_coordinates(LOp.localOps[i], Mats[i])  
            for i in LOp.block_first   ]...
        );


function coordinates(LOp::SymListOperators, Mats::Vector{<: AbstractMatrix} ) :: Union{Vector{<:Number}, Nothing}
    all([size(Mats[i])[1] == LOp.axisDims[i] for i=1:LOp.val]) || return nothing
    res= [ LOp.duals[i] ?
            star(
                LOp.localOps[i], LOp.axisDims[i], 
                 coordinates(LOp.localOps[i], Matrix(Mats[i]))
            ) : 
            coordinates(LOp.localOps[i], Matrix(Mats[i])) 
        for i=1:LOp.val]
    any(res .|> isnothing) && return nothing
    all([ isapprox(res[i], res[LOp.syms[i]]) for i=1:LOp.val if LOp.syms[i] < i ]) || return nothing
    return vcat([res[i] for i in LOp.block_first  ] ...)
end;

function reduceBy(LOp::SymListOperators, engaged::Vector{Bool})::NamedTuple{(:rLOp, :expand_func, :reduce_map), Tuple{ListOperators, Function, LinearMaps.LinearMap}}
    val=LOp.val
    syms=LOp.syms
    duals=LOp.duals
    blocks=LOp.blocks
    @assert val == length(engaged) "Incompatible data"
    @assert any(engaged) "Can not reduce to Nothing"
    minblock = [ minimum(vcat(1000, [j for j in blocks[k] if engaged[j] ]...)) for k =1:val]
    renumber = [sum(engaged[1:k]) for k =1:val]
    rsyms = [ renumber[minblock[syms[k]]] for k =1:val if engaged[k] ]
    # rduals = [ xor(duals[k], duals[minblock[syms[k]]] ) for k =1:val if engaged[k]]
    rduals = duals[engaged]
    # twisted = [ duals[minblock[syms[k]]]  for k =1:val if engaged[k]]
    c_idx = [ syms[k] for k =1:val if minblock[syms[k]]==k]
    # c_twist = [ duals[k] for k =1:val if minblock[syms[k]]==k]
    c_renum = [ renumber[k] for k =1:val if minblock[syms[k]]==k]
    # @show syms,duals,engaged
    # @show rsyms, rduals, twisted
    # @show c_idx, c_twist, c_renum
    ## generate new GlobalOps
    if all( [i==rsyms[i]  for i=1:length(rsyms)]) && !any(rduals)
        rLOp = IndListOperators(LOp.axisDims[engaged], LOp.localOps[engaged]) 
    else
        rLOp = SymListOperators(
            LOp.axisDims[engaged], LOp.localOps[engaged], 
            rsyms,rduals) 
    end 
    function expand(rdata::Vector{<:Number})::Vector{<:Number}
        edata=zeros(eltype(rdata), LOp.globalDim)
        for i= 1: length(c_idx)
            @inbounds edata[LOp.soffsets[c_idx[i]]:LOp.eoffsets[c_idx[i]] ] = 
                @inbounds rdata[rLOp.soffsets[c_renum[i]]:rLOp.eoffsets[c_renum[i]] ]
                # c_twist[i] ?
                #     unsafe_star(
                #             rLOp.localOps[c_renum[i]],
                #             rLOp.axisDims[c_renum[i]],
                #             @inbounds rdata[rLOp.soffsets[c_renum[i]]:rLOp.eoffsets[c_renum[i]] ]) :
                #     @inbounds rdata[rLOp.soffsets[c_renum[i]]:rLOp.eoffsets[c_renum[i]] ]
        end;
        return edata
    end;
    function expand_func(rdata::Vector{<:Number})::Vector{<:Number}
        edata=zeros(eltype(rdata), LOp.globalDim)
        for i= 1: length(c_idx)
            @inbounds edata[LOp.soffsets[c_idx[i]]:LOp.eoffsets[c_idx[i]] ] = 
                @inbounds rdata[rLOp.soffsets[c_renum[i]]:rLOp.eoffsets[c_renum[i]] ]
                # c_twist[i] ?
                #     unsafe_star(
                #             rLOp.localOps[c_renum[i]],
                #             rLOp.axisDims[c_renum[i]],
                #             @inbounds rdata[rLOp.soffsets[c_renum[i]]:rLOp.eoffsets[c_renum[i]] ]) :
                #     @inbounds rdata[rLOp.soffsets[c_renum[i]]:rLOp.eoffsets[c_renum[i]] ]
        end;
        for i in LOp.block_first
            if minblock[i]==1000
                @inbounds edata[LOp.soffsets[i]:LOp.eoffsets[i]] = trivial(LOp.localOps[i],LOp.axisDims[i]) 
            end
        end
        return edata
    end;
    contract(edata::Vector{<:Number})::Vector{<:Number} = 
        vcat(
                [ 
                    @inbounds edata[LOp.soffsets[c_idx[i]]:LOp.eoffsets[c_idx[i]] ] 
                    # c_twist[i] ? 
                    #     unsafe_star(
                    #         LOp.localOps[c_idx[i]],
                    #         LOp.axisDims[c_idx[i]],
                    #         @inbounds edata[LOp.soffsets[c_idx[i]]:LOp.eoffsets[c_idx[i]] ]) : 
                    #     @inbounds edata[LOp.soffsets[c_idx[i]]:LOp.eoffsets[c_idx[i]] ] 
                for i= 1: length(c_idx)]...
            ); 
    return (rLOp= rLOp, expand_func= expand_func, 
        reduce_map=LinearMaps.LinearMap(contract, expand, rLOp.globalDim, LOp.globalDim; ismutating=false) )


end;

isLinear(LOp::SymListOperators)::Bool = LOp.isLinear;

isInvertible(LOp::SymListOperators)::Bool = LOp.isInvertible;

generate_random(LOp::SymListOperators)::Vector{<:Number} = 
    vcat(
        [
            generate_random(LOp.localOps[i],LOp.axisDims[i])
            for i in LOp.block_first
        ]...
    );

function simplifyTo(LOp::SymListOperators)::NamedTuple{(:D, :T), Tuple{ListOperators,ListOperators}}  
    @assert LOp.isLinear "Only defined for Linear Operators"
    DlocalOps = LOp.localOps .|> simplifyTo .|> x -> x.D 
    TlocalOps = LOp.localOps .|> simplifyTo .|> x -> x.T
    DLOp = SymListOperators(LOp.axisDims, DlocalOps, LOp.syms, LOp.duals) 
    TLOp = SymListOperators(LOp.axisDims, TlocalOps, LOp.syms, LOp.duals)
    return (D=DLOp, T=TLOp) 
end;

function simplify(LOp::SymListOperators, data::Vector{<:Number} )::NamedTuple{(:d, :t), Tuple{Vector{<:Number},Vector{<:Number}} } 
    @assert LOp.isLinear "Only defined for Linear Operators"
    @assert length(data)==LOp.globalDim
    localres = [ 
                    simplfy(
                        LOp.localOps[i], LOp.axisDims[i],
                        @inbounds data[LOp.soffsets[i]:LOp.eoffsets[i]])
                for i in LOp.block_first];
    data_d = localres .|> x -> x.d
    data_t = localres .|> x -> x.t
    return (d= vcat(data_d...), t= vcat(data_t...))
end;

function scalarsMatrix(LOp::SymListOperators)::Matrix{<:Number}
    @assert isLinear(LOp) "Only defined for Linear Operators"
    sc = [ containScalars(LOp.localOps[LOp.block_first[i]]) for i = 1:length(LOp.block_first)]
    d = sum(sc)
    A = zeros(d,LOp.val)
    k=1
    for i=1:length(LOp.block_first)
        if sc[i]
            for j in LOp.blocks[i]
                A[k,j] = 1
            end
            k = k+1
        end
    end
    # @show sc,d,LOp.val, size(A)
    # @show A
    return A
end