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


"""
    Framing

    Used to switch between Arrays and Dicts. Allows object to be indexed both as Arrays and as Dicts
    Provides to fuctions 
        doDict      which makes a vector into a dict
        toVector    which makes a dict into a vector
        reduceBy    makes smaller framing using Vector{Bool}
        reduceBy_v  as above but returns vector

"""

struct Framing{I <:Any} 
    frame   :: Vector{<:I}
    len     :: Integer
    lookup  :: Dict{I, <:Integer}
    # compatible::Function
    Framing{I}(frame::Vector{<:I}) where  I = (
        len = length(frame);
        for i = 2:len 
            for j = 1:(i-1)
                @assert frame[i]!=frame[j] "indexes must be dsitinct";
            end;
        end;
        lookup = Dict{I, Int}(zip(frame, [i for i=1:len]));
        return new{I}(frame,len,lookup)
    )
end;


function toDict(F ::Framing{I}, V::Vector)::Dict where I
    @assert length(V) >= F.len "vector too short"
    return Dict{I,eltype(V)}(zip(F.frame, V[1:F.len]) )
end;

function toVector(F ::Framing{I}, D::Dict)::Vector where I 
    @assert all( F.frame .|> k -> haskey(D,k) ) "missing keys"
    return [ D[F.frame[i]] for i=1:F.len]
end;

function reduceBy_v(F ::Framing{I}, eng::Vector{Bool})::Vector{I} where I 
    return F.frame[eng]
end; 

function reduceBy_v(F ::Framing{I}, eng::BitVector)::Vector{I} where I 
    return F.frame[eng]
end;

function reduceBy(F ::Framing{I}, eng::Vector{Bool}) ::Framing{I} where I
    return  Framing{I}(F.frame[eng])
end;

function reduceBy(F ::Framing{I}, eng::BitVector) ::Framing{I} where I
    return Framing{I}(F.frame[eng])
end;

#can cause problems is I is an integer, but we will use it for I-ITensrs.Index
function Base.getindex(F ::Framing{I}, i::Integer)::I where I 
    return F.frame[i]
end;

function Base.getindex(F ::Framing{I}, i::I)::Integer where I 
    return F.lookup[i]
end;

# can be used instead of reduce
function Base.getindex(F ::Framing{I}, eng::BitVector)::Vector{<:I} where I 
    return F.frame[eng]
end;

function Base.getindex(F ::Framing{I}, eng::Vector{Bool})::Vector{<:I} where I 
    return F.frame[eng]
end;


function Base.getproperty(F ::Framing{I}, s ::Symbol) where I
    if s === :keys
        return keys(F.lookup)
    elseif s === :haskey
        return x -> haskey(F.lookup, x)
    else
        return getfield(F,s)
    end
end

