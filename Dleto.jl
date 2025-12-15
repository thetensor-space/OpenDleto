#
# Strata Dleto: Dleto.jl
#   Master file for Dleto module.
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

# import Pkg
import LinearAlgebra
import Random
import SparseArrays
import Statistics
import ProgressMeter
import Arpack
ENV["WEBIO_JUPYTER_DETECTED"] = "false"
using PlotlyJS

include("TensorSpace.jl")
include("Chisels.jl")
include("Sylvester.jl")
include("TensorSynthesis.jl")




function faceBlocks(t::AbstractArray, svdfunc::Function=ArpackEigen)
    # test valancy
    if ndims(t) != 3
        throw(DimensionMismatch("wrong arity of tensor"))
    end
    sizes = [size(t)...]
    blocks = sizes  .|> (n -> n*n ) 

    # set up system of lin equation
    println("\r\n\tBuilding linear system...")
    @time M = buildFullLinearSystem(t, FaceCurveMatrix)

    # do SVD and pick the smallest vectors 
    println("\r\n\tComputing singular vectors for ", size(M), "...\n\t")
        # @time lastsvds = svd(M)
    @time lastsvds = svdfunc(M)

    println("\r\n\tExtracting matrices...")
    # exctract the correct vector
    maineigenvector = lastsvds[:,2]

    # expand to matrices
    @time XMatrix = expandToMatrix(maineigenvector, sizes[1], 0)
    @time YMatrix = expandToMatrix(maineigenvector, sizes[2], blocks[1])
    # @time ZMatrix = expandToMatrix(maineigenvector, sizes[3], blocks[1] + blocks[2])
    ZMatrix = LinearAlgebra.Diagonal([(1:sizes[3])...])

    return changeTensor(t, XMatrix, YMatrix, ZMatrix)
end;

function blocks(t::AbstractArray, svdfunc::Function=ArpackEigen)
    # test valancy
    if ndims(t) != 3
        throw(DimensionMismatch("wrong arity of tensor"))
    end
    sizes = [size(t)...]
    blocks = sizes  .|> (n -> n*n ) 

    # set up system of lin equation
    println("\r\n\tBuilding linear system...")
    @time M = buildFullLinearSystem(t, CurveMatrix)

    # do SVD and pick the smallest vectors 
    println("\r\n\tComputing singular vectors for ", size(M), "...\n\t")
        # @time lastsvds = svd(M)
    @time lastsvds = svdfunc(M)

    println("\r\n\tExtracting matrices...")
    # exctract the correct vector
    maineigenvector = lastsvds[:,2]

    # expand to matrices
    @time XMatrix = expandToMatrix(maineigenvector, sizes[1], 0)
    @time YMatrix = expandToMatrix(maineigenvector, sizes[2], blocks[1])
    # @time ZMatrix = expandToMatrix(maineigenvector, sizes[3], blocks[1] + blocks[2])
    ZMatrix = LinearAlgebra.Diagonal([(1:sizes[3])...])

    return changeTensor(t, XMatrix, YMatrix, ZMatrix)
end;

#-------------------------------
# find orhtogonal transformations to put support of a tensor on a surface
"""
function toFaceCurveTensor(t::AbstractArray, svdfunc::Function=ArpackEigen)

Change a basis of a tensor to make it supported on a face curve. 
The output is a named tuple with coordinates .tensor, .Xchange, .Ychange, .Zchange, .Xes, .Yes, .Zes
consiting of the transformed tensor, the 3 change of basis matrices, and the vectors defining the surface.

The second ardument is a function which performs the svd of some relatively large matrix and rerurns the smallesr singular vectors.
The defalut value (ArpackEigen) uses the Arpack library, the two other possible functions area LinearAlgebraSVD and LinearAlgebraEigen.
Sometimes Arpack crashes, so there is a build in fall back to LinearAlgebra function
"""
function toFaceCurveTensor(t::AbstractArray, svdfunc::Function=ArpackEigen)
    # test valancy
    if ndims(t) != 3
        throw(DimensionMismatch("wrong arity of tensor"))
    end
    sizes = [size(t)...]
    blocks = sizes  .|> (n -> n*(n+1)÷ 2) 

    # set up system of lin equation
    M = buildLinearSystem(t, FaceCurveMatrix)

    # do SVD and pick the smallest vectors 
    lastsvds= svdfunc(M)
    
    # exctract the correct vector
    maineigenvector = lastsvds[:,2]

    # expand to matrices
    XMatrix = expandToSymetricMatrix(maineigenvector, sizes[1], 0)
    YMatrix = expandToSymetricMatrix(maineigenvector, sizes[2], blocks[1])
    # make a fake Z matrix    
    # Julia tries to be very clever to makes the eigenvalues are range instead of a vector, I am using some trickery to avoid that
    ZMatrix = LinearAlgebra.Diagonal([(1:sizes[3])...])

    return changeTensor(t, XMatrix, YMatrix, ZMatrix)
end;

#-------------------------------
# find orhtogonal transformations to put support of a tensor on a curve
"""
function toCurveTensor(t::AbstractArray, svdfunc::Function=ArpackEigen)

Change a basis of a tensor to make it supported on a diagonal curve.
The output is a named tuple with coordinates .tensor, .Xchange, .Ychange, .Zchange, .Xes, .Yes, .Zes
consiting of the transformed tensor, the 3 change of basis matrices, and the vectors defining the surface.

The second ardument is a function which performs the svd of some relatively large matrix and rerurns the smallesr singular vectors.
The defalut value (ArpackEigen) uses the Arpack library, the two other possible functions area LinearAlgebraSVD and LinearAlgebraEigen.
Sometimes Arpack crashes, so there is a build in fall back to LinearAlgebra function
"""
function toCurveTensor(t::AbstractArray, svdfunc::Function=ArpackEigen)
    # test valancy
    if ndims(t) != 3
        throw(DimensionMismatch("wrong arity of tensor"))
    end
    sizes = [size(t)...]
    blocks = sizes  .|> (n -> n*(n+1)÷ 2) 

    # set up system of lin equation
    M = buildLinearSystem(t, CurveMatrix)

    # do SVD and pick the smallest vectors 
    lastsvds= svdfunc(M)
    
    # exctract the correct vector
    maineigenvector = lastsvds[:,2]

    # expand to matrices
    XMatrix = expandToSymetricMatrix(maineigenvector, sizes[1], 0)
    YMatrix = expandToSymetricMatrix(maineigenvector, sizes[2], blocks[1])
    ZMatrix = expandToSymetricMatrix(maineigenvector, sizes[3], blocks[1] + blocks[2])

    return changeTensor(t, XMatrix, YMatrix, ZMatrix)
end;

"""
function toSurfaceTensor(t::AbstractArray, svdfunc::Function=ArpackEigen)

Change a basis of a tensor to make it supported on a surface. 
The output is a named tuple with coordinates .tensor, .Xchange, .Ychange, .Zchange, .Xes, .Yes, .Zes
consiting of the transformed tensor, the 3 change of basis matrices, and the vectors defining the surface.

The second ardument is a function which performs the svd of some relatively large matrix and rerurns the smallesr singular vectors.
The defalut value (ArpackEigen) uses the Arpack library, the two other possible functions area LinearAlgebraSVD and LinearAlgebraEigen.
Sometimes Arpack crashes, so there is a build in fall back to LinearAlgebra function
"""
function TuckerDecomposition(t::AbstractArray, svdfunc::Function=ArpackEigen)
    # test valancy
    if ndims(t) != 3
        throw(DimensionMismatch("wrong arity of tensor"))
    end
    sizes = [size(t)...]
    blocks = sizes  .|> (n -> n*n) 

    # set up system of lin equation
    println("\tBuilding linear system...")
    @time M = buildFullLinearSystem(t, TuckerMatrix)

    # do SVD and pick the smallest vectors 
    println("\tCalculuating SVD of matrix of dimensions: ", size(M))
    @time lastsvds= svdfunc(M)
    
    # exctract the correct vector
    maineigenvector = lastsvds[:,1]

    # expand to matrices
    XMatrix = expandToSymetricMatrix(maineigenvector, sizes[1], 0)
    YMatrix = expandToSymetricMatrix(maineigenvector, sizes[2], blocks[1])
    ZMatrix = expandToSymetricMatrix(maineigenvector, sizes[3], blocks[1] + blocks[2])

    return changeTensor(t, XMatrix, YMatrix, ZMatrix)
end;






println("✓ Dleto.jl loaded successfully.")
