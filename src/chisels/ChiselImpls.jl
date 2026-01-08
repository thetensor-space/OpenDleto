#
# Strata Dleto: Framed Chisels
#   Adding indexes to the matrix to make computation a bit more consisten 
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
    Framed Chisels

    Chisels with axis represented as ITensors.Index

"""
struct ChiselFramed{N}
    P ::AbstractMatrix
    frames ::Vector{Index{N}}
    idx::Dict{Index,Integer}
    axis ::Index{N}
    function ChiselFramed(
             P::AbstractMatrix, 
        frames ::Vector{Index{N}},
        a::Index{N}
        ) where N
        val=length(frames)
        @assert size(P,2)== val "Incompatible sizes"
        # test that all elements of frame are different 
        idx=Dict{Index,Int16}()
        for i=1:val
            idx[frames[i]] = i
        end
        return new{N}(P,frames, idx, a)
    end
end

ChiselFramed(P::AbstractMatrix{<:Number}, frames ::Vector{Index{K}} where K) = 
    ChiselFramed(P,frames, Index(size(P, 1), "chisel"));


function reduceByEngaged(FCh::ChiselFramed, engaged::Vector{Bool})::ChiselFramed
    @assert length(FCh.frames)==length(engaged) "Incompatable data"
    return ChiselFramed(FCh.ch[:,engaged],Fch.frames[engaged], FCh.ch_axis)
end

function reduceByEngaged(FCh::ChiselFramed, engaged::Dict{Index,Bool})::ChiselFramed
    @assert all(FCh.frames .|> (i -> haskey(enggaged,i)))== true  "Incompatable data"
    eng = [ engaged[FCh.frames[i]] for i=1:length(FCh.frames)]
    return reduceByEngaged(FCh, eng)
end

function applyDerivation(Γ::ITensor, Xes::Vector{ITensor}, FCh::ChiselFramed )::ITensor
    @assert all(Xes .|> (x ->ndims(x) == 2) ) == true "all Xes must have valancy 2"
    @assert all(Xes .|> (x -> xor( (inds(x) .|> i -> haskey(FCh.idx, i))...) ) ) == true "incompatible indexes"
    @assert all(Xes .|> (x -> xor( (inds(x) .|> i -> haskey(FCh.idx, i))...) ) ) == true "incompatible indexes"
    @assert all(FCh.frames .|> i -> i in inds(Γ )) == true "Γ misses some indexes"
    Γ_frame_ch = (FCh.ch_axis, inds(Γ)...)
    return [ ( 
                haskey(FCh.idx,  ind(Xes[i],1) ) ? 
                    (ci =ind(Xes[i],1) ; oi =ind(Xes[i],2) ) : (ci =ind(Xes[i],2) ; oi =ind(Xes[i],1) );
                replaceind(
                    ITensor(FCh.ch[:, FCh.idx[ci]],FCh.ch_axis) * Γ * Xes[i],
                    oi,ci) 
             ) 
             for i in 1:length(Xes) ] |> sum  
end
