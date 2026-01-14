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
struct Chisel
    ch_m    ::Matrix{<:Number}
    # ch_v    ::Vector{<:Vector{<:Number}}
    # ch_v_it ::Vector{<:ITensors.ITensor}
    # ch_d    ::Dict{ITensors.Index,Vector{<:Number}}
    # ch_d_it ::Dict{ITensors.Index, <:ITensors.ITensor}
    ch_axis ::ITensors.Index
    frames  ::Framing{ITensors.Index}
    Chisel(ch::AbstractMatrix{<:Number}, frames ::Framing{ITensors.Index}, ch_axis::ITensors.Index) = (
        val=frames.len;
        @assert size(ch,2) == val "Incompatible number columns";
        @assert size(ch,1) == ITensors.ITensors.dim(ch_axis) "Incompatable chisel axis";
        ch_m = Matrix(ch);
        ch_v = [ ch_m[:,i] for i=1:val];
        ch_v_it = ch_v .|> v -> ITensors.ITensor(v,ch_axis);
        ch_d = toDict(frames,ch_v);
        ch_d_it = toDict(frames,ch_v_it);
        # return new(ch_m, ch_v,ch_v_it,ch_d,ch_d_it,ch_axis,frames)
        return new(ch_m,ch_axis,frames)
    )
end;

# extra constructors -- auto generate framing from an array and creat a new index of the corect dim
Chisel(ch::AbstractMatrix{<:Number}, frames ::Vector{<:ITensors.Index}, ch_axis::ITensors.Index)=
    Chisel(ch, Framing{ITensors.Index}(frames), ch_axis::ITensors.Index);

Chisel(ch::AbstractMatrix{<:Number}, frames ::Framing{ITensors.Index})::Chisel =
    Chisel(ch, frames, ITensors.Index( size(ch,1), "chisel"));

Chisel(ch::AbstractMatrix{<:Number}, frames ::Vector{<:ITensors.Index})::Chisel =
    Chisel(ch, Framing{ITensors.Index}(frames), ITensors.Index( size(ch,1), "chisel"));

"""
    evaluate chisel on vector of dictionary
"""
evaluate_chisel(ch::Chisel, v::Vector{<:Number})::Vector{<:Number} = ch.ch_m * v;

evaluate_chisel(ch::Chisel, D::Dict)::Vector{<:Number} = ch.ch_m * toVector(ch.frames, D);

"""
    evaluate chisel on vector of dictionary and turn it into ITensro
"""
evaluate_chisel_it(ch::Chisel, v::Vector{<:Number})::ITensors.ITensor = ITensors.ITensor(ch.ch_m *v, ch.ch_axis);

function evaluate_chisel_it(ch::Chisel,D::Dict)::ITensors.ITensor
    # return [D[ch.frames[i] * ch.ch_v_it[i] ] for i=1:ch.frames.len] |> sum
    return ITensors.ITensor(ch.ch_m * toVector(ch.frames,d),ch.ch_axis)
end

"""
    reduce chisel by removing some columns, eng is either Vector{Bool} or Dictionary
"""
function reduceBy(ch::Chisel, eng::Union{Vector{Bool},BitVector})::Chisel
    @assert length(eng) == ch.frames.len "incompatible reduction"
    return Chisel(ch.ch_m[:,eng], reduceBy(ch.frames,eng), ch.ch_axis) 
end
reduceBy(ch::Chisel, eng::Dict)::Chisel = reduce_chisel(ch,toVector(ch.frames, D))

"""
    extend chisel by adding zero columns
    this can also be used to reduce chisels...
    keeps the common columns and add zero column to the newly added one
"""
function extend_chisel(ch::Chisel, newframes::Framing{ITensors.Index})::Chisel 
    m = zeros( size(ch.ch_m,1), newframes.len)
    for i = 1:newframes.len
        if ch.frames.haskey(newframes[i])
            m[:,i] = ch.ch_m[:, ch.frames[newframes[i]] ]
        end
    end
    return Chisel(m, newframes, ch.ch_axis)
end

extend_chisel(ch::Chisel, newframes::Vector{<:ITensors.Index})::Chisel =extend_chisel(ch,Framing{ITensors.Index}(newframes)); 

"""
    find columns where the entires are nmostly zeros
"""
function engaged_axis(ch::Chisel; cutoff::Float64=1e-6)::Vector{Bool}
    if cutoff > 0.0
        return [ any(ch.ch_m[:,i] .|> (x -> abs(x) > cutoff))  for i = 1: ch.frames.len] 
    else
        return [ isapprox(ch.ch_m[:,i], zeros(size(ch.ch_m,1))) for i = 1: ch.frames.len]
    end
end

engaged_axis_dict(ch::Chisel; cutoff::Float64=1e-6)::Dict = toDict(ch.frames, engaged_axis(ch;cutoff=cutoff))


"""
    Replace chisel with equivalent one to improve stability.
"""
function normalize_chisel(ch::Chisel; cutoff::Float64=1e-6)::Chisel
    EA =engaged_axis(ch;cutoff=cutoff)
    rch = reduce_chisel(ch, EA)
    m = rch.ch_m
    svddecom = LinearAlgebra.svd(m)
    num = sum( svddecom.S .|> ( x -> abs(x) > cutoff))
    nch = Chisel(svddecom.Vt[1:num,:], rch.frames, ITensors.Index(num,"chisel,reduced"))
    return extend_chisel(nch, ch.frames)
end

replaceFrame(ch::Chisel, newframe::Framing{ITensors.Index})::Chisel = Chisel(ch.ch_m,newframe,ch.ch_axis);
replaceAxis(ch::Chisel, newaxis::ITensors.Index)::Chisel = Chisel(ch.ch_m,ch.frames, newaxis);