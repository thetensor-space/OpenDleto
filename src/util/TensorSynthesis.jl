
#
# Strata Dleto: Tensor Synthesis 
#   tools to create tensors for testing and demonstrations.
#
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
# 


#-------------------------------
# generate a tensor with support restricted by a distace function

"""
randTensorSupport(deltas::Vector{Vector}, frames::Vector{Index}, cutoff::Number, dist::Function)::ITensor

Generate a random tensor with supported on the constarint  dist(...) < cutoff
"""
function randomTensorSupport(
        # deltas::Vector{Vector{K}} where K <: Number,
        # frames::Vector{Index{L}} where L,
        deltas::Vector{<:Vector{<:Number}},
        frames::Vector{<:ITensors.Index},
        dist::Function;
        cutoff::Float64=1e-6
    )::ITensors.ITensor
    val = length(deltas)
    # @show val
    sizes = Tuple(length.(deltas))
    # @show sizes
    # @show frames
    @assert length(frames) == val "Ambiguous frame matching: length of list of frames must match array of deltas"
    @assert all([ITensors.dim(frames[i]) == length(deltas[i])  for i=1:val])  "Ambiguous frame matching: length of list of frames must match array of deltas"
    A = ITensors.ITensor(frames...)
    for ci in CartesianIndices(sizes)
        if dist([deltas[i][ci[i]] for i =1:val] ) < cutoff
            A[ci] = randn()
        end
    end
    return A
end

function randomTensorChisel(deltas::Vector{<:Vector{<:Number}}, frames::Vector{<:ITensors.Index}, ch::Chisel; cutoff::Float64=1e-6 )::ITensors.ITensor  
    newchisel = extend_chisel(ch, frames)
    return randomTensorSupport(deltas, frames, x -> LinearAlgebra.norm(evaluate_chisel(newchisel,x)) ;cutoff=cutoff)
end

randomTensorChisel(deltas::Vector{<:Vector{<:Number}}, f::Framed, ch::Chisel; kwargs...)::ITensors.ITensor = 
    randomTensorChisel(deltas, f.frames, ch; kwargs...);

#-------------------------------
# test if the support of a tensor is restricted by a distance function
"""
    measures how colse the suport of T is to the zero set of function dsit
    returns a named tuple is 
        norm    -- number between 0,1 where 0 means that the suport of T is inside the zero set of dist
        Tmass   -- L2 norm of the T squared
        dmass   -- L2 norm of tensor with entries dist function, squared
        dot     -- dot product of the above two vectors
                    norm is cosine of the angle between the vectors 
""" 
function normTensorSupport(T ::ITensors.ITensor, deltas::Vector{<:Vector{<:Number}}, dist::Function)
    frames=ITensors.inds(T)
    val = length(deltas)
    sizes = Tuple(deltas .|> length)
    @assert length(frames) == val "Ambiguous frame matching: length of list of frames must match array of deltas"
    @assert all([ITensors.dim(frames[i]) == length(deltas[i])  for i=1:val])  "Ambiguous frame matching: length of list of frames must match array of deltas" 
    res = [ 
        let 
            d= dist([deltas[i][ci[i]] for i =1:val] )
            [T[ci] * T[ci], abs( T[ci] * d ), d * d ] 
        end
        for ci in CartesianIndices(sizes)] |> sum
    return ( norm = res[2]*res[2]/(res[1] * res[3] + 1e-15), Tmass= res[1], dmass=res[3], dot=res[2] )
end;

function normTensorChisel(T ::ITensors.ITensor, deltas::Vector{<:Vector{<:Number}}, ch::Chisel)
    frames=ITensors.inds(T)
    newchisel = extend_chisel(ch, vcat(frames...) )
    return normTensorSupport( T, deltas, x -> LinearAlgebra.norm(evaluate_chisel(newchisel,x)))
end;
# TODO add type of the result as named tuple


