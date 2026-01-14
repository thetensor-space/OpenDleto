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
struct TransverseOps
    LOps        ::ListOperators
    frames      ::Framing{ITensors.Index}
    framesTemp  ::Framing{ITensors.Index}
    TransverseOps(LOps::ListOperators,frames::Framing{ITensors.Index},framesTemp::Framing{ITensors.Index}) = (
        #test consistency; add more verification
        @assert frames.len == framesTemp.len "Incompatible frames";
        return new(LOps,frames,framesTemp)
    )
end; 


swapFrames(TOp ::TransverseOps)::TransverseOps = TransverseOps(TOp.LOps, TOp.framesTemp, TOp.frames);

function embedITensors(TOp::TransverseOps, data::Vector{<:Number}) ::Vector{ITensors.ITensor}
    val = TOp.frames.len
    Mats = embedMatrices(TOp.LOps, data)
    return [ ITensors.ITensor(Mats[i], TOp.frames[i], TOp.framesTemp[i]) for i =1:val]
end;

function unsafe_embedITensors(TOp::TransverseOps, data::Vector{<:Number}) ::Vector{ITensors.ITensor}
    val = TOp.frames.len
    Mats = unsafe_embedMatrices(TOp.LOps, data)
    return [ ITensors.ITensor(Mats[i], TOp.frames[i], TOp.framesTemp[i]) for i =1:val]
end;

#ToDo
function reduceByEngaged(TOp::TransverseOps, engaged::Vector{Bool})::Tuple{TransverseOps, LinearMaps.LinearMap} 
    @assert false "Calling Placeholder Abstract Function"
end;


# #no need it is better to flip the frames inthe trnasverse op
# function embedITensorsSwapped(TOp::TransverseOps, data::Vector{<:Number}) ::Vector{ITensor}
#     val = TOp.frames.len
#     Mats = embedMatrices(TOp.LOps, data)
#     return [ ITensor(Mats[i], TOp.framesTemp.frame[i], TOp.frames.frame[i]) for i =1:val]
# end

# function unsafe_embedITensorsSwapped(TOp::TransverseOps, data::Vector{<:Number}) ::Vector{ITensor}
#     val = TOp.frames.len
#     Mats = unsafe_embedMatrices(TOp.LOps, data)
#     return [ ITensor(Mats[i], TOp.framesTemp.frame[i], TOp.frames.frame[i]) for i =1:val]
# end


function transposeEmbed(TOp::TransverseOps, ITs ::Vector{ITensors.ITensor})::Vector{<:Number}
    # reorder the tensors 
    @assert isLinear(TOp.LOps) "The transverse Op must be linear"
    D = Dict{ITensors.Index, Matrix}()
    for i in 1:length(ITs)   
        axises = ITensor.inds(ITs[i])
        @assert length(axises) == 2 "Incompatable Data"
        j=0
        if TOp.frames.haskey(axises[1])
            j =  TOp.frames[axises[1]]
            @assert TOp.frames[j] == axises[1] "Incompatable Data"
            @assert TOp.framesTemp[j] == axises[2] "Incompatable Data"
        else
            @assert TOp.frames.haskey(axises[2])
            j =  TOp.frames[axises[2]]
            @assert TOp.frames[j] == axises[2] "Incompatable Data"
            @assert TOp.framesTemp[j] == axises[1] "Incompatable Data"
        end
        D[TOp.frames[j]] = Array(ITs[i], TOp.frames[j], TOp.framesTemp[j])
    end
    Mats = toVector(TOp.frames, D)
    return transposeEmbed( TOp.LOps, Mats)
end

#public constructors
function TransverseOps(axises::Vector{<:ITensors.Index}, s::Vector{Symbol},tag::String="temp") ::TransverseOps
    @assert length(axises)==length(s) "not compatible"
    t_axises = axises .|> (x -> ITensors.addtags(x,tag)) 
    axisDims = ITensors.dim.(axises) 
    localOps = Operator.(s) 
    frames = Framing{ITensors.Index}(axises)
    t_frames = Framing{ITensors.Index}(t_axises)
    return TransverseOps( IndListOperators(axisDims, localOps) ,frames,t_frames)
end


function TransverseOps(axises::Vector{<:ITensors.Index}, s::Vector{Symbol}, symmetries ::Vector{<:Integer}, tag::String="temp")::TransverseOps
    @assert length(axises)==length(s) "not compatible"
    t_axises = axises .|> (x -> ITensors.addtags(x,tag)) 
    axisDims = ITensors.dim.(axises) 
    localOps = Operator.(s) 
    frames = Framing{ITensors.Index}(axises)
    t_frames = Framing{ITensors.Index}(t_axises)
    return TransverseOps( SymListOperators(axisDims, localOps, symmetries) ,frames,t_frames)
end
    
TransverseOps(axises::Vector{<:ITensors.Index}, s::Symbol,tag::String="temp")::TransverseOps = 
    TransverseOps(axises, [s for a in axises], tag);
    

function TransverseOps(Γ::ITensors.ITensor, s::Symbol,tag::String="temp")::TransverseOps 
    axises = vcat(ITensors.inds(Γ)...)
    TransverseOps(axises, [s for a in axises], tag);
end

# assumes that the ITensors are in order
unsafe_transposeEmbed(TOp::TransverseOps, ITs ::Vector{ITensors.ITensor})::Vector{<:Number} = 
    unsafe_transposeEmbed(
        TOp.LOps, 
        [ Array(ITs[i], TOp.frames[i], TOp.framesTemp[i])  for i =1:val] 
    );



# May be we need coordinates???

function changeBasis(Γ::ITensors.ITensor, TOp::TransverseOps, data::Vector{<:Number}, keep::Bool=false):: NamedTuple{(:Σ, :Xs), Tuple{ITensors.ITensor, Vector{ITensors.ITensor}}}
    val = TOp.frames.len
    inds = ITensors.inds(Γ)
    @assert isInvertible(TOp.LOps) "Only inverible Transverse Ops can change basis"
    @assert all([ TOp.frames[i] in inds  for i = 1:val ] ) "Incompatable itensors"   
    @assert all([ !(TOp.framesTemp[i] in inds)  for i = 1:val ] ) "Incompatable itensors"   
    Xs = embedITensors(TOp,data)
    Δ = Γ
    for i in 1:val
        Δ *= Xs[i]
        # Δ = Δ * Xs[i]
    end
    if keep 
        for i in 1:val
            ITensors.replaceindex!(Δ,TOp.framesTemp[i],TOp.frames[i])
        end
    end
    return (;Σ=Δ, Xs=Xs)
end

changeBasisRandom(Γ::ITensors.ITensor, TOp::TransverseOps, keep::Bool=false):: NamedTuple{(:Σ, :Xs), Tuple{ITensors.ITensor, Vector{ITensors.ITensor}}} = 
    changeBasis(Γ, TOp, generate_random(TOp.LOps), keep)





#simplfyTo
function simplifyTo(TOp::TransverseOps)::NamedTuple{(:D, :T), Tuple(TransverseOps,TransverseOps)} 
    res =  simplifyTo(TOp.LOps)
    return (D = TransverseOps(res.D, TOp.frames, TOp.framesTemp), T = TransverseOps(res.T, TOp.frames, TOp.framesTemp) ) 
end;
