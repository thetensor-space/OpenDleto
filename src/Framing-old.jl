#
# Strata Dleto: Framed Structs (usally by indexes)
#   Adding framing to objects 
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

abstract type ArrayLike end;

function A_sizes(::ArrayLike) ::Vector end;
function A_reduceBy(::ArrayLike, engaged::Vector{Bool})::ArrayLike end;

struct myVec <:ArrayLike
    d:: Vector
end;
A_sizes(x::myVec) ::Vector=x.d;
A_reduceBy(x::myVec, engaged::Vector{Bool})::myVec = myVec(x.d[engaged])

"""
    Framed Array-like Object

"""

struct Framing{I <:Any} 
    frame   :: Vector{<:I}
    len     :: Integer
    lookup  :: Dict{I, <:Integer}
    # compatible::Function
    Framing{I}(frame::Vector{<:I}) where  I = (
        len = length(frame);
        lookup = Dict{I, Int}(zip(frame, [i for i=1:len]));
        return new{I}(frame,len,lookup)
    )
end;


function toDict(F ::Framing{I}, V::Vector) where I
    @assert length(V) >= F.len "vector too short"
    return Dict{I,eltype(V)}(zip(F.frame, V[1:F.len]) )
end;

function toVector(F ::Framing{I}, D::Dict) where I
    @assert F.frame .|> k -> haskey(D,k) |> all "missing keys"
    return [ D[F.frame[i]] for i=1:F.len]
end;




abstract type Small end;
struct wrappedSmall
    x :: Small
end;

struct Ext <: Small
    z :: Int
end;

computefunc(a::Ext) = a.z*a.z;

struct Ext2 <: Small
    zz :: Int
end;

computefunc(a::Ext2) = a.zz+a.zz;

computefunc(a::Small) = 10;

e = Ext(1)
e2 = Ext2(2)
we = wrappedSmall(e)
we2 = wrappedSmall(e2)

computefunc(e)
computefunc(e2)

convert(::Type{Small}, w::wrappedSmall) = w.x

computefunc(we)
computefunc(we2)


struct FramedExtra{AL<:ArrayLike, F,IF,E}
    a ::AL
    # f ::Vector{<:F} 
    # fi ::Vector{<:IF}
    # idx ::Dict{IF,Integer}
    # extra :: E
    # compatible::Function
    FramedExtra(a::AL, f::Vector{<:F}, fi::Vector{<:FI}, compatible::Function, extra::E) = (
    #     val=length(frames);
    #     asizes = A_sizes(a);
    #     @assert length(asizes) == val "lenghts must be the same";
    #     @assert all( [ compatible(f[i], asizes[i]) for i =1:val ] )  "Incompatible data";
    #     # test that all elements of frame are different 
    #     idx=Dict{E,Int16}();
    #     fi = f .|> get_index; 
    #     for i=1:val
    #         idx[ fi[i]] = i;
    #     end;
        # return new(a, f, fi, Dict{E,Int16}(), extra, compatible)
        return new{AL, F,IF,E}(a)
    )
end;

function reduceByEngaged(F::FramedExtra{AL<:ArrayLike, F,IF,E}, engaged::Vector{Bool})::FramedExtra{AL<:ArrayLike, F,IF,E}
    @assert length(f.f)==length(engaged) "Incompatable data"
    FramedExtra{AL,F,IF,E}(A_reduceBy(F,engaged), F.f[engaged], F.fi[engaged], F.compatible, F.e)
end


Framed{AL,F,IF} = FramedExtra{AL,F,IF,Nothing}
Framed{AL,F,IF}(a::AL, f ::Vector{<:F}, compatible::Function, fi::Vector{<:FI}) = 
    Framed{AL,F,IF}(a,f,compatible,fi, nothing);




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
