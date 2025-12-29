
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

module TensorSynthesis

using ITensors

# TBD: rename and rethink these functions and their names
export randomOrthogonalMatrix, spin, randomize
export randTensor, randSurfaceTensor, randFaceCurveTensor, randCurveTensor
export distSurfaceTensor, distFaceCurveTensor, distCurveTensor


function randomOrthogonalMatrix(n::Integer)
    mat = randn(n,n)
    Q,R = LinearAlgebra.qr(mat)
    D = Diagonal(sign.(diag(R)))
    return Q*D
end;

"""
spin(t::AbstractArray)

Applies random orthogonal changes of basis to all axes of the tensor t.

Retruns a record tuple with coordinates .tensor, .matrices
""" 
function spin(t::AbstractArray)
    mats = collect([ randomOrthogonalMatrix(size(t,a)) for a in 1:ndims(t) ])
    tensor = act(t,mats)
    return (;tensor, matrices)
end;


"""
randomize(t::ITensor)

Picks 3 random invertible matrices and use them to peform a random basis change of a 3 tensor.
Returuns named tupple with coordinates .tensor, .Xchange, .Ychange, .Zchange
""" 
function randomize(Γ::ITensor)::NamedTuple{(:Γ, :X),Tuple{ITensor, Vector{ITensor}}}
    function getRand(n) 
        mat = randn(n,n)
        count = 0
        while LinearAlgebra.rank(mat) < n && count < 100
            count += 1
            mat = randn(n,n)
        end
        if count > 99
            throw(ErrorException("Could not generate random invertible matrix"))
        end
        return mat
    end
    matrices = collect([ getRand(size(Γ,a)) for a in 1:ndims(Γ) ])
    frame = inds(Γ)
    X = [ ITensor(Matrix(matrices[a]), frame[a], frame[a]') for a in 1:ndims(Γ) ]
    # Σ = Γ
    # for x in X
    #     Σ = x * Σ
    #     noprime!(Σ)
    #     Σ = permute(Σ, inds(Γ)...)
    # end
    return (;Γ=(Γ*X), X)
end;

function randomize(t::AbstractArray)
    frame = [ Index(size(t,a), "a$a") for a in 1:ndims(t) ]
    if eltype(t) != Float64
        t = Float64.(t)
    end
    return randomize(ITensor(t, frame...))
end;
    


#-------------------------------
# generate random othorgonal matrix (likely uniform with respect to Haar maeasure, if not it should be close enough...)
# """
# randomOthogonalMatrix(d::Integer)::Matrix

# Generate a random orthogonal matrix of size d 
# """
# function randomOthogonalMatrix(d::Integer)::AbstractMatrix
#     if d < 1 
#         throw(DimensionMismatch("matrix size needs to be non-negative"))
#     end
#     R = randn(d,d)
#     M = LinearAlgebra.Symmetric(R*R')
#     return LinearAlgebra.eigen(M).vectors
# end;

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
"""
randTensorSupport(xes::Vector, yes::Vector, zes::Vector, cutoff::Number, dist::Function)::Array

Generate a random tensor supported on a surface defined by dist(x,y,z) < cutoff
"""
function randTensor(
        xes::Vector, 
        yes::Vector, 
        zes::Vector,
        cutoff::Number, 
        dist::Function
    )::ITensor

    res = zeros(Float64,(length(xes),length(yes), length(zes )))
    # loop over entries
    for ci in CartesianIndices(res)
        # make random numbers if the point satisfies the equation
        if dist(xes[ci[1]], yes[ci[2]], zes[ci[3]]) < cutoff
            res[ci] = randn()
        end
    end
    x = Index(length(xes), "x")
    y = Index(length(yes), "y")
    z = Index(length(zes), "z")
    return ITensor(res, x, y, z)
end;

#-------------------------------
# generate a tensor supported on the a surface
"""
randSurfaceTensor(xes::Vector, yes::Vector, zes::Vector, cutoff::Number)::Array 

Generate a random tensor supported on a surface with equation x_i + y_j + z_k \approx 0 
"""
function randSurfaceTensor(
        xes::Vector, 
        yes::Vector, 
        zes::Vector, 
        cutoff::Number
    ):: ITensor
    return randTensor(xes,yes,zes,cutoff,surfaceDistance)
end;

#-------------------------------
# generate a tensor supported on the a face curve
"""
randFaceCurveTensor(xes::Vector, yes::Vector, zes::Vector, cutoff::Number)::Array

Generate a random tensor supported on a surface with equation x_i \approx y_j, we need z_k to determine the size of the tensor 
"""
function randFaceCurveTensor(
        xes::Vector, 
        yes::Vector, 
        zes::Vector,
        cutoff::Number
    )::ITensor
    return randTensor(xes,yes,zes,cutoff,faceCurveDistance)
end;

#-------------------------------
# generate a tensor supported on the a curve
"""
randFaceCurveTensor(xes::Vector, yes::Vector, zes::Vector, cutoff::Number)::Array

Generate a random tensor supported on a surface with equation x_i \approx y_j \approx z_k 
"""
function randCurveTensor(
        xes::Vector, 
        yes::Vector, 
        zes::Vector, 
        cutoff::Number
    )::ITensor
    return randTensor(xes,yes,zes,cutoff,curveDistance)
end

#-------------------------------
# test if the support of a tensor is restricted by a distance function
function dist(Γ ::ITensor, xes::Vector, yes::Vector, zes::Vector, dist::Function)::Number
    t = store(Γ)
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
testSurfaceTensor(t::ITensor, xes::Vector, yes::Vector, zes::Vector)::Number

Measure how far a tensor is from being supported on a surface with equation x_i + y_j + z_k =0 
The result is a number between 0 and 1 with (almost) 0 bing that the tensor is supported on the surface 
The normalization is not perfect -- it is a good idea to call this on random tensor for a comparision
"""
function distSurfaceTensor(
        t::ITensor, 
        xes::Vector, 
        yes::Vector, 
        zes::Vector
    )::Number
    return dist(t,xes,yes,zes,surfaceDistance)
end


#-------------------------------
# test if a tensor is supported on a face curve
"""
distFaceCurveTensor(t::ITensor, xes::Vector, yes::Vector, zes::Vector)::Number

Measure how far a tensor is from being supported on a face curve with equation x_i = y_j 
The result is a number between 0 and 1 with (almost) 0 bing that the tensor is supported on the face curve 
The normalization is not perfect -- it is a good idea to call this on random tensor for a comparision
"""
function distFaceCurveTensor(
        t::ITensor, 
        xes::Vector, 
        yes::Vector, 
        zes::Vector
    )::Number
    return dist(t,xes,yes,zes,faceCurveDistance)
end


#-------------------------------
# test if a tensor is supported on a curve
"""
distCurveTensor(t::ITensor, xes::Vector, yes::Vector, zes::Vector)::Number 

Measure how far a tensor is from being supported on a face curve with equation x_i = y_j = z_k 
The result is a number between 0 and 1 with (almost) 0 bing that the tensor is supported on the curve
The normalization is not perfect -- it is a good idea to call this on random tensor for a comparision
"""
function distCurveTensor(
        t::ITensor, 
        xes::Vector, 
        yes::Vector, 
        zes::Vector
    )::Number
    return dist(t,xes,yes,zes,curveDistance)
end

end # module