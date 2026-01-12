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

    Used to switch between Arrays and Dicts

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


function toDict(F ::Framing{I}, V::Vector) where I
    @assert length(V) >= F.len "vector too short"
    return Dict{I,eltype(V)}(zip(F.frame, V[1:F.len]) )
end;

function toVector(F ::Framing{I}, D::Dict) where I
    @assert all( F.frame .|> k -> haskey(D,k) ) "missing keys"
    return [ D[F.frame[i]] for i=1:F.len]
end;

