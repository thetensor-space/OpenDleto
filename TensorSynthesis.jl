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


#-------------------------------
# generate random othorgonal matrix (likely uniform with respect to Haar maeasure, if not it should be close enough...)
"""
randomOthogonalMatrix(d::Integer)::Matrix

Generate a random orthogonal matrix of size d 
"""
function randomOthogonalMatrix(d::Integer)::AbstractMatrix
    if d < 1 
        throw(DimensionMismatch("matrix size needs to be non-negative"))
    end
    R = randn(d,d)
    M = LinearAlgebra.Symmetric(R*R')
    return LinearAlgebra.eigen(M).vectors
end;

#-------------------------------
# Measure distance to the surface
surfaceDistance(a,b,c) = Base.abs(a+b+c);

#-------------------------------
# Measure distance to the face curve
faceCurveDistance(a,b,c) = Base.abs(a-b);

#-------------------------------
# Measure distance to the face curve
curveDistance(a,b,c) = sqrt((a-b)*(a-b) + (a-c)*(a-c) + (c-b)*(c-b) );

#-------------------------------
# generate a tensor with support restricted by a distace function
function randomTensorSupport(xes::Vector, yes::Vector, zes::Vector, cutoff::Number, dist::Function)::Array
    res = zeros(Float64,(length(xes),length(yes), length(zes )))
    # loop over entries
    for ci in CartesianIndices(res)
        # make random numbers if the point satisfies the equation
        if dist(xes[ci[1]], yes[ci[2]], zes[ci[3]]) < cutoff
            res[ci] = randn()
        end
    end
    return res
end;

#-------------------------------
# generate a tensor supported on the a surface
"""
randomSurfaceTensor(xes::Vector, yes::Vector, zes::Vector, cutoff::Number)::Array 

Generate a random tensor supported on a surface with equation x_i + y_j + z_k \approx 0 
"""
randomSurfaceTensor(xes::Vector, yes::Vector, zes::Vector, cutoff::Number)= randomTensorSupport(xes,yes,zes,cutoff,surfaceDistance); 

#-------------------------------
# generate a tensor supported on the a face curve
"""
randomFaceCurveTensor(xes::Vector, yes::Vector, zes::Vector, cutoff::Number)::Array

Generate a random tensor supported on a surface with equation x_i \approx y_j, we need z_k to determine the size of the tensor 
"""
randomFaceCurveTensor(xes::Vector, yes::Vector, zes::Vector, cutoff::Number)= randomTensorSupport(xes,yes,zes,cutoff,faceCurveDistance); 

#-------------------------------
# generate a tensor supported on the a curve
"""
randomFaceCurveTensor(xes::Vector, yes::Vector, zes::Vector, cutoff::Number)::Array

Generate a random tensor supported on a surface with equation x_i \approx y_j \approx z_k 
"""
randomCurveTensor(xes::Vector, yes::Vector, zes::Vector, cutoff::Number)= randomTensorSupport(xes,yes,zes,cutoff,curveDistance); 


#-------------------------------
# test if the support of a tensor is restricted by a distace function
function testTensorSupport(t::AbstractArray, xes::Vector, yes::Vector, zes::Vector, dist::Function)::Number
    # test valancy and sizes
    if ndims(t) != 3
        throw(DimensionMismatch("wrong arity of tensor"))
    end
    sizes = size(t)
    if (sizes[1] != length(xes)) && (sizes[2] != length(yes)) && (sizes[3] != length(zes))
        throw(DimensionMismatch("incompatible equation"))
    end

    mass = 0.0
    # randmass =0.0
    distance = 0.0
    # randdist = 0.0
    product = 0.0
    for ci in CartesianIndices(t)
    #    r = randn()
        mass += t[ci] * t[ci]
    #    randmass += r*r
        distance += abs( t[ci] * dist(xes[ci[1]],yes[ci[2]],zes[ci[3]] ) ) 
    #    randdist += abs( r * dist(xes[ci[1]], yes[ci[2]], zes[ci[3]] ) )
        product +=  dist(xes[ci[1]], yes[ci[2]], zes[ci[3]]) * dist(xes[ci[1]], yes[ci[2]], zes[ci[3]]) 
    end
    return distance*distance/(mass * product + 1e-15) 
end;


#-------------------------------
# test if a tensor is supported on a surface
"""
testSurfaceTensor(t::AbstractArray, xes::Vector, yes::Vector, zes::Vector)::Number

Measure how far a tensor is from being supported on a surface with equation x_i + y_j + z_k =0 
The result is a number between 0 and 1 with (almost) 0 bing that the tensor is supported on the surface 
The normalization is not perfect -- it is a good idea to call this on random tensor for a comparision
"""
testSurfaceTensor(t::AbstractArray, xes::Vector, yes::Vector, zes::Vector) = testTensorSupport(t,xes,yes,zes,surfaceDistance); 


#-------------------------------
# test if a tensor is supported on a face curve
"""
testFaceCurveTensor(t::AbstractArray, xes::Vector, yes::Vector, zes::Vector)::Number

Measure how far a tensor is from being supported on a face curve with equation x_i = y_j 
The result is a number between 0 and 1 with (almost) 0 bing that the tensor is supported on the face curve 
The normalization is not perfect -- it is a good idea to call this on random tensor for a comparision
"""
testFaceCurveTensor(t::AbstractArray, xes::Vector, yes::Vector, zes::Vector) = testTensorSupport(t,xes,yes,zes,faceCurveDistance); 

#-------------------------------
# test if a tensor is supported on a curve
"""
testCurveTensor(t::AbstractArray, xes::Vector, yes::Vector, zes::Vector)::Number 

Measure how far a tensor is from being supported on a face curve with equation x_i = y_j = z_k 
The result is a number between 0 and 1 with (almost) 0 bing that the tensor is supported on the curve
The normalization is not perfect -- it is a good idea to call this on random tensor for a comparision
"""
testCurveTensor(t::AbstractArray, xes::Vector, yes::Vector, zes::Vector) = testTensorSupport(t,xes,yes,zes,curveDistance); 


