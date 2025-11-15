#
# Copyright (c) 2023 Peter A Brooksbank, Martin Kassabov,
# James B. Wilson.  All rights reserved.
#
import LinearAlgebra
import Random

# install Arpack if it is not installed -- comment the next two if it ialready installed
#import Pkg
#Pkg.add("Arpack")

import Arpack

#import Base

#-------------------------------
# Action of matrices on the tensor
# raise exception if the sizes are not compatible
"""
actionOnTensor(t ::AbstractArray, mat ::AbstractArray, dir ::Integer)

Act on a tensor by a matrix via a given axis 
"""
function actionOnTensor(t ::AbstractArray, mat ::AbstractArray, dir ::Integer) ::Array
    if (dir < 1) || (dir >  ndims(t))
        throw(DimensionMismatch("direction does not exist"))
    end
    if ndims(mat) != 2
        throw(DimensionMismatch("only matrices can act!"))
    end 
    sizes = [size(t)...]
    matsizes = size(mat)

    #matrices act on the right
    if matsizes[1] != sizes[dir]
        throw(DimensionMismatch("incompatible sizes of tensor and matrices"))
    end
    # compute sizes to reshape into 3 tensor
    p=prod(sizes[1 : (dir - 1)])
    q=prod(sizes[(dir + 1) : end])

    # reshape the tensor into 3 tensor
    ten = reshape(t, p, sizes[dir], q)
    # same for result
    res = similar(ten, p, matsizes[2], q)
    # use matrix multiplication to compute the action 
    for i = 1:q
        LinearAlgebra.mul!(@view(res[:, :, i]), @view(ten[:, :, i]), mat)
    end
    # find the new size
    sizes[dir] = matsizes[2]
    # reshape the result and return 
    return reshape(res, Tuple(sizes))
end;


#-------------------------------
# Act on all Axises
# m is a list of matrices

"""
actAllDirections(t ::AbstractArray, m::Vector{A} where A <: AbstractArray) ::Array

Act on a tensor by a many matrices on all axes 
"""
function actAllDirections(t ::AbstractArray, m::Vector{A} where A <: AbstractArray) ::Array
    if (ndims(t) != length(m))
        throw(DimensionMismatch("wrong size of vector of matrices"))
    end
    res= t;
    for i = 1:ndims(t)
        res = actionOnTensor( res, m[i], i)
    end
    return res
end;


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


#-------------------------------
# randomize a tensor by basis change with random orthogonal matrices
"""
randomizeTensor(t::AbstractArray)

Picks 3 random orthogonal matrices and use them to peform a random basis change of a 3 tensor.
Retruns named tupple with coordinates .tensor, .Xchange, .Ychange, .Zchange
""" 
function randomizeTensor(t::AbstractArray)
    # test valancy
    if ndims(t) != 3
        throw(DimensionMismatch("wrong arity of tensor"))
    end
    sizes = [size(t)...]
    Xchange=  randomOthogonalMatrix(sizes[1])
    Ychange=  randomOthogonalMatrix(sizes[2])
    Zchange=  randomOthogonalMatrix(sizes[3])
    tensor = actAllDirections(t,[Xchange,Ychange,Zchange])
    return (;tensor, Xchange, Ychange, Zchange)
end;

#-------------------------------
# combine some coordinatres of a vector into a symmetric matrix
# use with caution -- the function does not perform any checks!
function expandToSymetricMatrix(u::Vector, n:: Integer, offset::Integer )::AbstractMatrix
    M = zeros(Float64,n,n)
    k = 1 + offset
    for i = 1:n
        for j = i:n
            M[i,j] = u[k]
            M[j,i] = u[k]
            k = k + 1
        end
    end 
    return LinearAlgebra.Symmetric(M)
end;

#-------------------------------
# technical function, peroforms inpalce "multiplication" by symetric matrix 

function modifyRow!(row::Array, shift::Integer, n::Integer, index::Integer, vec::Vector, coef::Number)
    for i = 1:n
        if i <= index 
            row[shift+Int((2*n-i+2)*(i-1)/2)+index-i+1] += vec[i]*coef
        else
            row[shift+Int((2*n-index+2)*(index-1)/2)-index+i+1] += vec[i]*coef
        end
    end
    return nothing
end;


#-------------------------------
# technical functions for performing svd and returing the smallest singular vectors 
# these willl throw error if there are less then 10 singular vectors
function LinearAlgebraSVD(M::AbstractMatrix)
    svds = LinearAlgebra.svd(M)
    return svds.U[:,end:-1:end-9]
end;

function LinearAlgebraEigen(M::AbstractMatrix)
    eigens = LinearAlgebra.eigen( LinearAlgebra.Symmetric(M*M') )
    return eigens.vectors[:,1:10]
end;

#sometimes this crashes so we build fall back to LinearAlgebraEigen
function ArpackEigen(M::AbstractMatrix)
    try 
        eigens = Arpack.eigs( LinearAlgebra.Symmetric(M*M')  ; which =:SR, nev =20)
        @show eigens[1]
        return eigens[2]
    catch e
        return LinearAlgebraEigen(M)
    end
end;


#-------------------------------
# technical matrices which encode the system of equations
# Surface Equation Matrix
SurfaceMatrix = [1.0 1.0 1.0]
# Face Curve Equation Matrix
FaceCurveMatrix = [1.0 -1.0]
# Curve Equation Matrix
CurveMatrix = [1.0 -1.0 0.0; 1.0 0.0 -1.0; 0.0 1.0 -1.0]

#-------------------------------
# technical function for building the linear system 
# use with caution, there are no checks for consistency
function buildLinearSystem(t::AbstractArray, eqMatrix::AbstractMatrix)::Matrix
    sizes = [size(t)...]
    Msize = size(eqMatrix)
    blocks = sizes  .|> (n -> n*(n+1)÷ 2) 
    numvars =  sum(i -> blocks[i], 1: Msize[2])
    M = zeros( Float64, ( numvars, Msize[1] * length(t) )  )
    k=0
    R = CartesianIndices(t)
    for ci in R                            #  loop over entries of tensor
        li = LinearIndices(t)[ci]
        for i = 1:Msize[1] 
            s=0
            for j = 1:Msize[2]                
                # extract 1 dimensional slice of the tensor
                first = li - (ci[j] - 1)*stride(t,j)
                last = first + (sizes[j]- 1)*stride(t,j)
                slice = t[first:stride(t,j):last]
                # add it to the condition
                modifyRow!( M, numvars*k + s, sizes[j], ci[j], slice, eqMatrix[i,j] )
                s += blocks[j]
            end
            k += 1
        end
    end
    return M
end

#-------------------------------
# technical function for performing base change based on 3 symmetric matrices 
# use with caution, there are no checks for consistency
# the output is a named tuple
function changeTensor(t::AbstractArray, XMatrix::AbstractMatrix, YMatrix::AbstractMatrix, ZMatrix::AbstractMatrix)
    # compute change of basis and other data
    Xeig = LinearAlgebra.eigen(XMatrix)
    Yeig = LinearAlgebra.eigen(YMatrix)
    Zeig = LinearAlgebra.eigen(ZMatrix)
    Xchange = Xeig.vectors 
    Ychange = Yeig.vectors 
    Zchange = Zeig.vectors
    Xes = Xeig.values 
    Yes = Yeig.values 
    Zes = Zeig.values 
    # perform change of basis
    tensor=actAllDirections(t, [Xchange,Ychange,Zchange])

    return (;tensor, Xchange, Ychange, Zchange, Xes, Yes, Zes)
end;

#-------------------------------
# find orhtogonal transformations to put support of a tensor on a surface
"""
function toSurfaceTensor(t::AbstractArray, svdfunc::Function=ArpackEigen)

Change a basis of a tensor to make it supported on a surface. 
The output is a named tuple with coordinates .tensor, .Xchange, .Ychange, .Zchange, .Xes, .Yes, .Zes
consiting of the transformed tensor, the 3 change of basis matrices, and the vectors defining the surface.

The second ardument is a function which performs the svd of some relatively large matrix and rerurns the smallesr singular vectors.
The defalut value (ArpackEigen) uses the Arpack library, the two other possible functions area LinearAlgebraSVD and LinearAlgebraEigen.
Sometimes Arpack crashes, so there is a build in fall back to LinearAlgebra function
"""
function toSurfaceTensor(t::AbstractArray, svdfunc::Function=ArpackEigen)
    # test valancy
    if ndims(t) != 3
        throw(DimensionMismatch("wrong arity of tensor"))
    end
    sizes = [size(t)...]
    blocks = sizes  .|> (n -> n*(n+1)÷ 2) 

    # set up system of lin equation
    M = buildLinearSystem(t, SurfaceMatrix)

    # do SVD and pick the smallest vectors 
    lastsvds= svdfunc(M)
    
    # exctract the correct vector
    maineigenvector = lastsvds[:,3]

    # expand to matrices
    XMatrix = expandToSymetricMatrix(maineigenvector, sizes[1], 0)
    YMatrix = expandToSymetricMatrix(maineigenvector, sizes[2], blocks[1])
    ZMatrix = expandToSymetricMatrix(maineigenvector, sizes[3], blocks[1] + blocks[2])

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

function saveTensorToFile(tensor::AbstractArray, filename::String, threshold::Float64=1e-3)
    open(filename, "w") do file
        dims = size(tensor)
        println(file, "# i j k value")
        for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]
            val = tensor[i, j, k]
            if abs(val) > threshold
                println(file, "$i $j $k $val")
            end
        end
    end
end


using PlotlyJS
function plotTensor(tensor::AbstractArray, threshold::Float64=1e-2)

    # function for removing small entries
    dropSmall = x -> abs(x)< threshold ? 0 : x
    tensor = tensor .|> dropSmall
    
    # Get indices of non-zero values in the tensor
    indices = findall(x -> x != 0, tensor)
    dims = size(tensor)

    # Extract x, y, z coordinates
    x_coords = [idx[1] for idx in indices]
    y_coords = [idx[2] for idx in indices]
    z_coords = [idx[3] for idx in indices]

    # Create 3D scatter plot with bounding box based on tensor dimensions
    plot(scatter3d(
        x=x_coords, 
        y=y_coords, 
        z=z_coords,
        mode="markers",
        marker=attr(size=2, opacity=0.6)
    ), Layout(
        scene=attr(
            xaxis=attr(range=[1, dims[1]+1], title="X"),
            yaxis=attr(range=[1, dims[2]+1], title="Y"),
            zaxis=attr(range=[1, dims[3]+1], title="Z"),
            aspectmode="cube"
        ),
        title="3D Tensor Visualization"
    ))
end
