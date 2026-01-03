
#
# Strata Dleto: Tensor Synthesis 
#   tools to create tensors for testing and demonstrations.
#
# Copyright 2022-2025 Peter A. Brooksbank, Martin D. Kassabov, James B. Wilson
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

using ITensors


"""
__randomizeITensor(t::ITensor,f::Function, extratags)

Randomize ITensor by generating random transformations for each axis using the function f
add extra tags to the axis 
""" 
function __randomizeITensor(Γ::ITensor,f::Function,extratags="randomized")::NamedTuple{(:Γ, :X),Tuple{ITensor, Vector{ITensor}}}
    frame = inds(Γ)
    matrices = f(frame)
    X = [ ITensor(Matrix(matrices[a]), frame[a], addtags(frame[a],extratags)) for a in 1:ndims(Γ) ]
    return (;Γ=(Γ*X), X)
end;
# I am applying f to the list of axis, because I want to be able to make transformations to be the same if the axis are the same
# this is an internal function

"""
randomizeITensorSimilar(t::ITensor,f::Function)

Randomize ITensor by generating random transformations for each axis using the function f
""" 
randomizeITensorSimilar(Γ::ITensor,f::Function,extratags="randomized")::NamedTuple{(:Γ, :X),Tuple{ITensor, Vector{ITensor}}} = __randomizeITensor(Γ, x-> x.|>f ,extratags);
randomizeITensor(Γ, x-> x.|>f ,extratags);


randomizeITensorOthorgonal(Γ::ITensor,extratags="randomized")::NamedTuple{(:Γ, :X),Tuple{ITensor, Vector{ITensor}}} = 
randomizeITensorSimilar(Γ, randomOrthogonalMatrix);

randomizeITensorOthorgonal(t::AbstractArray,extratags="randomized")::NamedTuple{(:Γ, :X),Tuple{ITensor, Vector{ITensor}}} = 
randomizeITensorSimilar(ArrayToITensor(t), randomOrthogonalMatrix);

randomizeArrayOthorgonal(t::AbstractArray,extratags="randomized")::NamedTuple{(:Γ, :X),Tuple{ITensor, Vector{ITensor}}} = 
randomizeITensorSimilar(ArrayToITensor(t), randomOrthogonalMatrix);

randomizeITensorInvertible(Γ::ITensor,extratags="randomized")::NamedTuple{(:Γ, :X),Tuple{ITensor, Vector{ITensor}}} = 
randomizeITensorSimilar(Γ, randomInvertibleMatrix);

randomizeITensorInvertible(t::AbstractArray,extratags="randomized")::NamedTuple{(:Γ, :X),Tuple{ITensor, Vector{ITensor}}} = 
randomizeITensorSimilar(ArrayToITensor(t), randomInvertibleMatrix);

randomizeArrayInvertible(t::AbstractArray,extratags="randomized")::NamedTuple{(:Γ, :X),Tuple{ITensor, Vector{ITensor}}} = 
randomizeITensorSimilar(ArrayToITensor(t), randomInvertibleMatrix);


#-------------------------------
# generate a tensor with support restricted by a distace function

"""
randTensorSupport(deltas::Vector{Vector}, frames::Vector{Index}, cutoff::Number, dist::Function)::ITensor

Generate a random tensor supported on the constarint  dist(...) < cutoff
"""
function randTensorDistFunct(
        # deltas::Vector{Vector{K}} where K <: Number,
        # frames::Vector{Index{L}} where L,
        deltas::Vector{Vector{<: Number}},
        frames::Vector{Index{<: Any}},
        cutoff::Number,
        dist::Function
    )::ITensor
    val = length(deltas)
    sizes = deltas .|> length
    @assert length(frames) == val "Ambiguous frame matching: length of list of frames must match array of deltas"
    @assert all([ITensors.dim(frames[i]) == length(deltas[i])  for i=1:val])  "Ambiguous frame matching: length of list of frames must match array of deltas"

# MDK: add option to change the type of the entries
# it should be more efficient to generate it as zero ITensor and then add entries
    A = ITensor(frames...)
    for ci in CartesianIndices(sizes)
        if dist([deltas[i][ci[i]] for i =1:val] ) < cutoff
            A[ci] = randn()
        end
    end
    return A
    # res = zeros(Float64,sizes... )
    # for ci in CartesianIndices(res)
    #     if dist([deltas[i][ci[i]] for i =1:val] ) < cutoff
    #         res[ci] = randn()
    #     end
    # end
    # return ITensor(res, frames...)
end;



randTensorDistFunct(deltas::Vector{Vector{<:Number}},cutoff::Number,dist::Function)::ITensor =
randTensorDistFunct(deltas, [ Index(length(deltas[a]), "a$a") for a in 1:length(deltas)], cutoff,dist);


randTensorChisel(deltas::Vector{Vector{<: Number}}, frames::Vector{Index{<:Any}}, cutoff::Number, ch::Matrix)::ITensor = 
randTensorDistFunct(deltas,frames,cutoff, x -> EvaluateChisel(ch, x));

randTensorChisel(deltas::Vector{Vector{<:Number}}, cutoff::Number, ch::Matrix)::ITensor = 
randTensorDistFunct(deltas,[ Index(length(deltas[a]), "a$a") for a in 1:length(deltas)],cutoff, x -> EvaluateChisel(ch, x));


#-------------------------------
# test if the support of a tensor is restricted by a distance function
function ITensorNorm(Γ ::ITensor, deltas::Vector{Vector{<:Number}}, dist::Function)::Number
    frames=inds(Γ)
    val = length(deltas)
    sizes = deltas .|> length
    @assert length(frames) == val "Ambiguous frame matching: length of list of frames must match array of deltas"
    @assert all([frames[i].space == length(deltas[i])  for i=1:val])  "Ambiguous frame matching: length of list of frames must match array of deltas"

 
#    t = asarray(Γ) -- no need to make array we can use ccartesian indexes directly
    mass = 0.0
    distance = 0.0
    product = 0.0
    for ci in CartesianIndices(sizes)
        mass += Γ[ci] * Γ[ci]
        d= dist([deltas[i][ci[i]] for i =1:val] )
        distance += abs( Γ[ci] * d ) 
        product +=  d * d 
    end
    return distance*distance/(mass * product + 1e-15) 
end;

ITensorNormChisel(Γ ::ITensor, deltas::Vector{Vector{ <: Number}}, ch::Matrix)::Number=
ITensorNorm(Γ,deltas, x -> EvaluateChisel(ch, x) );



