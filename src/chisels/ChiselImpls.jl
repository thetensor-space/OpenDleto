#
# Strata Dleto: Framed Chisels
#   Adding indexes to the matrix to make computation a bit more consistent 
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
    Chisel

    A chisel carrying its frame: the coefficient matrix `ch` together with the
    `Index` terms its columns refer to, an `Index -> column` lookup, and the
    chisel axis.

    Renamed from `ChiselFramed`.  This is the type the refactor grows into the
    full `(𝕋, Ω, P)` setting of docs/review/Refactor-Plan.md section 1 -- the
    `Ω` (operator restriction) and `𝕋` (tensor space) slots are not here yet,
    so today it is still just `P` plus a frame.
"""
struct Chisel
    ch ::AbstractMatrix{<:Number}
    frames ::Vector{Index{K}} where K
    idx ::Dict{Index,Integer}
    ch_axis ::Index
    Chisel(P::AbstractMatrix{<:Number}, frames ::Vector{Index{K}} where K, ch_axis::Index) = (
        val=length(frames);
        @assert size(P,2)== val "Incompatible sizes";
        @assert size(P,1) == ITensors.dim(ch_axis) "Incompatable chisel axis";
        # test that all elements of frame are different 
        idx=Dict{Index,Int16}();
        for i=1:val
            idx[frames[i]] = i;
        end;
        return new(P, frames, idx, ch_axis)
    )
end;

Chisel(P::AbstractMatrix{<:Number}, frames ::Vector{Index{K}} where K) = 
    Chisel(P,frames, Index(size(P, 1), "chisel"));


# NOTE: both of these were dead on arrival -- the first referenced an
# undefined `Fch`, the second an undefined `enggaged`.  Neither has a caller,
# so fixing the typos changes nothing observable; it just makes the intended
# behaviour reachable.  They are repaired rather than deleted because this
# type is what Phase 1 grows into the full chisel.
#
# `FCh` in the signatures below is left over from the `ChiselFramed` name;
# not renamed here to keep this commit a pure rename.
function reduceByEngaged(FCh::Chisel, engaged::Vector{Bool})::Chisel
    @assert length(FCh.frames)==length(engaged) "Incompatable data"
    return Chisel(FCh.ch[:,engaged], FCh.frames[engaged], FCh.ch_axis)
end

function reduceByEngaged(FCh::Chisel, engaged::Dict{Index,Bool})::Chisel
    @assert all(FCh.frames .|> (i -> haskey(engaged,i)))== true  "Incompatable data"
    eng = [ engaged[FCh.frames[i]] for i=1:length(FCh.frames)]
    return reduceByEngaged(FCh, eng)
end

function applyDerivation(Γ::ITensor, Xes::Vector{ITensor}, FCh::Chisel )::ITensor
    @assert all(Xes .|> (x ->ndims(x) == 2) ) == true "all Xes must have valancy 2"
    @assert all(Xes .|> (x -> xor( (inds(x) .|> i -> haskey(FCh.idx, i))...) ) ) == true "incompatible indexes"
    @assert all(FCh.frames .|> i -> i in inds(Γ )) == true "Γ misses some indexes"
    Γ_frame_ch = (FCh.ch_axis, inds(Γ)...)
    return [ 
                let 
                    haskey(FCh.idx,  ind(Xes[i],1) ) ? 
                        (ci =ind(Xes[i],1) ; oi =ind(Xes[i],2) ) : (ci =ind(Xes[i],2) ; oi =ind(Xes[i],1) )
                    replaceind(
                        ITensor(FCh.ch[:, FCh.idx[ci]],FCh.ch_axis) * Γ * Xes[i],
                        oi,ci) 
                end 
            for i in 1:length(Xes) ] |> sum  
end
