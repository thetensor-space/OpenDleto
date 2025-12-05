#
# Copyright (c) 2023 Peter A Brooksbank, Martin Kassabov,
# James B. Wilson.  All rights reserved.
#
import LinearAlgebra
import Random
# using LinearAlgebra

# install Arpack if it is not installed -- comment the next two if it ialready installed
#import Pkg
#Pkg.add("Arpack")

import Arpack


# import Pkg
# Pkg.add("PlotlyJS")
ENV["WEBIO_JUPYTER_DETECTED"] = "true"
using PlotlyJS

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
function expandToMatrix(u::Vector, n:: Integer, offset::Integer )::AbstractMatrix
    M = zeros(Float64,n,n)
    k = 1 + offset
    for i = 1:n
        for j = i:n
            M[i,j] = u[k]
            # M[j,i] = u[k]
            k = k + 1
        end
    end 
    return LinearAlgebra.Matrix(M)
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
# Tucker Equation Matrix
TuckerMatrix = [1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 1.0]

#-------------------------------
# technical function for building the linear system 
# use with caution, there are no checks for consistency
function buildFullLinearSystem(t::AbstractArray, eqMatrix::AbstractMatrix)::Matrix
    sizes = [size(t)...]
    Msize = size(eqMatrix)
    blocks = sizes .|> (n -> n*n)
    numvars =  sum(i -> blocks[i], 1: Msize[2])
    M = zeros( Float64, ( numvars, Msize[1] * length(t) )  )
    k=0
    println("\tSizes: ", size(M))
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
# technical function for building the linear system 
# use with caution, there are no checks for consistency
function buildFullLinearSystem(t::AbstractArray, eqMatrix::AbstractMatrix)::Matrix
    sizes = [size(t)...]
    Msize = size(eqMatrix)
    blocks = sizes .|> (n -> n*n)
    numvars =  sum(i -> blocks[i], 1: Msize[2])
    M = zeros( Float64, ( numvars, Msize[1] * length(t) )  )
    k=0
    println("\tSizes: ", size(M))
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
# technical function for building the linear system 
# use with caution, there are no checks for consistency
function buildLinearSystem(t::AbstractArray, eqMatrix::AbstractMatrix)::Matrix
    sizes = [size(t)...]
    Msize = size(eqMatrix)
    blocks = sizes  .|> (n -> n*(n+1)÷ 2) 
    numvars =  sum(i -> blocks[i], 1: Msize[2])
    M = zeros( Float64, ( numvars, Msize[1] * length(t) )  )
    println("\tNumber of blocks: ", Msize)
    println("\tNumber of variables: ", numvars)
    k=0
    println("\tSizes: ", size(M))
    R = CartesianIndices(t)
    println("Number of coordinates: ", length(R))
    println("loops to do: ", Msize[1] * Msize[2] * length(R))
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
# technical function for building the linear system 
# use with caution, there are no checks for consistency
# t[i,j,k]*(X[i,i] + Y[j,j] + Z[k,k]) =0
function buildDiagonalSystem(t::AbstractArray)::Matrix
    sizes = [size(t)...]
    println("Sizes: ", sizes)
    numvars = sum(sizes)  # Total number of diagonal entries across X, Y, Z
    
    # Count non-zero entries to determine number of equations
    # non_zero_indices = findall(x -> abs(x) > 1e-12, t)
    num_equations = length(t)
    
    M = zeros(Float64, (numvars, num_equations))
    println("Numvars: ", numvars)
    println("Matrix size: ", size(M))
    
    eq_idx = 1
    for ci in CartesianIndices(t)
        t_val = t[ci]
        
        # X[i,i] coefficient (first sizes[1] variables)
        M[ci[1], eq_idx] = t_val
        
        # Y[j,j] coefficient (next sizes[2] variables)
        M[sizes[1] + ci[2], eq_idx] = t_val
        
        # Z[k,k] coefficient (last sizes[3] variables)
        M[sizes[1] + sizes[2] + ci[3], eq_idx] = t_val
        
        eq_idx += 1
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
    println("\r\n\tBuilding linear system...")
    @time M = buildLinearSystem(t, SurfaceMatrix)

    # do SVD and pick the smallest vectors 
    println("\r\n\tComputing singular vectors for ", size(M), "...\n\t")
        # @time lastsvds = svd(M)
    @time lastsvds = svdfunc(M)

    println("\r\n\tExtracting matrices...")
    # exctract the correct vector
    maineigenvector = lastsvds[:,3]

    # expand to matrices
    @time XMatrix = expandToSymetricMatrix(maineigenvector, sizes[1], 0)
    @time YMatrix = expandToSymetricMatrix(maineigenvector, sizes[2], blocks[1])
    @time ZMatrix = expandToSymetricMatrix(maineigenvector, sizes[3], blocks[1] + blocks[2])

    return changeTensor(t, XMatrix, YMatrix, ZMatrix)
end;

function stratify(t::AbstractArray, svdfunc::Function=ArpackEigen)
    # test valancy
    if ndims(t) != 3
        throw(DimensionMismatch("wrong arity of tensor"))
    end
    sizes = [size(t)...]
    blocks = sizes

    # set up system of lin equation
    println("\r\n\tBuilding linear system...")
    @time M = buildFullLinearSystem(t, SurfaceMatrix)

    # do SVD and pick the smallest vectors 
    println("\r\n\tComputing singular vectors for ", size(M), "...\n\t")
        # @time lastsvds = svd(M)
    @time lastsvds = svdfunc(M)

    println("\r\n\tExtracting matrices...")
    # exctract the correct vector
    maineigenvector = lastsvds[:,3]

    # expand to matrices
    @time XMatrix = expandToMatrix(maineigenvector, sizes[1], 0)
    @time YMatrix = expandToMatrix(maineigenvector, sizes[2], blocks[1])
    @time ZMatrix = expandToMatrix(maineigenvector, sizes[3], blocks[1] + blocks[2])

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

function loadTensorFromFile(filename::String)
    # Read the file and parse entries
    entries = []
    max_i, max_j, max_k = 0, 0, 0
    
    open(filename, "r") do file
        for line in eachline(file)
            # Skip comments and empty lines
            if startswith(line, "#") || isempty(strip(line))
                continue
            end
            
            # Parse the line: i j k value
            parts = split(strip(line))
            if length(parts) == 4
                i = parse(Int, parts[1])
                j = parse(Int, parts[2])
                k = parse(Int, parts[3])
                val = parse(Float64, parts[4])
                
                push!(entries, (i, j, k, val))
                max_i = max(max_i, i)
                max_j = max(max_j, j)
                max_k = max(max_k, k)
            end
        end
    end
    
    # Create tensor array with appropriate dimensions
    tensor = zeros(Float64, max_i, max_j, max_k)
    
    # Populate the tensor with values
    for (i, j, k, val) in entries
        tensor[i, j, k] = val
    end
    
    return tensor
end




function createTensorFromIncidence3(M::AbstractMatrix{T}) where T
    # Check that M is a 01 matrix
    if !all(x -> x == 0 || x == 1, M)
        throw(ArgumentError("Matrix M must be a 0-1 matrix"))
    end
    
    d = size(M, 2)  # number of columns
    e = size(M, 1)  # number of rows
    
    # Check that each row has exactly 3 ones
    for row in 1:e
        if sum(M[row, :]) != 3
            throw(ArgumentError("Each row must have exactly 3 ones"))
        end
    end
    
    # Create d x d x d tensor initialized to zeros
    t = zeros(Float16, d, d, d)
    
    # For each row in M, find the three nonzero columns and set t[i,j,k] = 1
    for row in 1:e
        nonzero_cols = findall(x -> x == 1, M[row, :])
        if length(nonzero_cols) == 3
            i, j, k = nonzero_cols
            t[i, j, k] = 1
            t[i, k, j] = 1
            t[j, i, k] = 1
            t[j, k, i] = 1
            t[k, i, j] = 1
            t[k, j, i] = 1
        end
    end
    
    return t
end


function createTensorFromIncidence(M::AbstractMatrix{T}, m::Integer; field::Type=Float16) where T
    # Check that M is a 0-1 matrix
    if !all(x -> x == 0 || x == 1, M)
        throw(ArgumentError("Matrix M must be a 0-1 matrix"))
    end
    if m < 1
        throw(ArgumentError("m must be >= 1"))
    end

    d = size(M, 2)  # number of columns
    e = size(M, 1)  # number of rows
    if m > d
        throw(ArgumentError("m cannot exceed number of columns"))
    end

    # Create an m-way tensor of size d x d x ... x d with specified element type
    dims = ntuple(_ -> d, m)
    t = zeros(field, dims...)

    # recursive helper to set all permutations of indices in tensor to one(field)
    function set_permutations!(Tarr, inds::Vector{Int}, i::Int)
        if i == length(inds)
            Tarr[Tuple(inds)...] = one(field)
            return
        end
        for j in i:length(inds)
            inds[i], inds[j] = inds[j], inds[i]
            set_permutations!(Tarr, inds, i + 1)
            inds[i], inds[j] = inds[j], inds[i]
        end
    end

    # Process rows that have exactly m ones
    for row in 1:e
        cols = findall(x -> x == 1, M[row, :])
        if length(cols) == m
            set_permutations!(t, collect(cols), 1)
        end
    end

    return t
end

function concatIncidenceMatrices(mats::AbstractMatrix...) 
    if length(mats) == 0
        throw(ArgumentError("At least one incidence matrix is required"))
    end
    ncols = size(mats[1], 2)
    for M in mats
        if ndims(M) != 2
            throw(ArgumentError("All inputs must be 2D matrices"))
        end
        if size(M, 2) != ncols
            throw(ArgumentError("All incidence matrices must have the same number of columns (vertices)"))
        end
        if !all(x -> x == 0 || x == 1, M)
            throw(ArgumentError("Incidence matrices must be 0-1 matrices"))
        end
    end
    return vcat(mats...)
end

# Convenience method accepting a vector/collection of matrices
function concatIncidenceMatrices(coll::AbstractVector{<:AbstractMatrix})
    return concatIncidenceMatrices(coll...)
end





"""
generateHypergraph(kind::Symbol, d::Integer, k::Integer; kwargs...)

Generate standard hypergraph incidence matrices (rows = edges, columns = vertices).

Supported kinds:
- :complete  -> complete k-uniform hypergraph (all k-subsets)
- :random    -> random k-uniform hypergraph; provide either p (probability) or m (number of edges)
- :cycle     -> cyclic k-uniform hypergraph (edges are k consecutive vertices modulo d)
- :chain     -> chain of k-uniform hyperedges connected by single vertices (consecutive edges share exactly one vertex); use num_edges kwarg to specify exact count
- :star      -> star centered at `center` (default 1): all edges contain the center plus any k-1 others
- :fano      -> Fano plane (only valid for d==7, k==3)

Returns a 0-1 Matrix{Int} with one row per edge and d columns.
"""
function generateHypergraph(kind::Symbol, d::Integer, k::Integer; kwargs...)
    if d < 1 || k < 1 || k > d
        throw(ArgumentError("Invalid d,k: require 1 ≤ k ≤ d"))
    end

    # build incidence matrix from list of edges (each edge is a Vector{Int}). There are E edges and d vertices.
    function incidence_from_edges(edges::Vector{Vector{Int}}, d::Integer)
        E = length(edges)
        M = zeros(Int, E, d)
        for (r, edge) in enumerate(edges)
            for v in edge
                if v < 1 || v > d
                    throw(ArgumentError("vertex index out of range: $v"))
                end
                M[r, v] = 1
            end
        end
        return M
    end

    # generate all k-combinations of 1:d
    function combinations_vec(n::Integer, k::Integer)
        res = Vector{Vector{Int}}()
        if k == 0
            push!(res, Int[])
            return res
        end
        buf = Vector{Int}(undef, k)
        function rec(start::Int, depth::Int)
            if depth > k
                push!(res, copy(buf))
                return
            end
            remaining = k - depth + 1
            last = n - remaining + 1
            for i in start:last
                buf[depth] = i
                rec(i + 1, depth + 1)
            end
        end
        rec(1, 1)
        return res
    end

    kind = Symbol(kind)

    if kind === :complete
        edges = combinations_vec(d, k)
        return incidence_from_edges(edges, d)
    elseif kind === :random
        p = get(kwargs, :p, nothing)
        m = get(kwargs, :m, nothing)
        # generate all combinations then sample according to p or m
        all_edges = combinations_vec(d, k)
        if p !== nothing
            if !(0.0 <= p <= 1.0)
                throw(ArgumentError(":p must be in [0,1]"))
            end
            chosen = Vector{Vector{Int}}()
            for e in all_edges
                if rand() < p
                    push!(chosen, e)
                end
            end
            return incidence_from_edges(chosen, d)
        elseif m !== nothing
            if m < 0
                throw(ArgumentError(":m must be non-negative"))
            end
            E = length(all_edges)
            msel = min(Int(m), E)
            idx = randperm(E)[1:msel]
            chosen = all_edges[idx]
            return incidence_from_edges(chosen, d)
        else
            throw(ArgumentError("For :random provide either p=Probability or m=NumEdges"))
        end
    elseif kind === :cycle
        edges = Vector{Vector{Int}}()
        for i in 1:d
            edge = [( (i - 1 + j) % d ) + 1 for j in 0:k-1]
            push!(edges, edge)
        end
        return incidence_from_edges(edges, d)
    elseif kind === :star
        center = get(kwargs, :center, 1)
        if center < 1 || center > d
            throw(ArgumentError("center must be between 1 and d"))
        end
        others = [v for v in 1:d if v != center]
        # all (k-1)-subsets of others
        sub = combinations_vec(d - 1, k - 1)
        # map subsets of positions 1..(d-1) to actual vertex labels in `others`
        edges = Vector{Vector{Int}}()
        for s in sub
            edge = [center]
            for pos in s
                push!(edge, others[pos])
            end
            push!(edges, sort(edge))
        end
        return incidence_from_edges(edges, d)
    elseif kind === :chain
        # Create a chain of k-uniform hyperedges connected by single vertices
        # Each edge shares exactly one vertex with the next edge
        if k < 2
            throw(ArgumentError("chain requires k >= 2"))
        end
        edges = Vector{Vector{Int}}()
        
        # Determine the number of edges to create
        num_edges_kwarg = get(kwargs, :num_edges, nothing)
        if num_edges_kwarg !== nothing
            num_edges = Int(num_edges_kwarg)
            if num_edges < 1
                throw(ArgumentError(":num_edges must be >= 1"))
            end
            # Check if we have enough vertices for the requested number of edges
            # Total vertices needed: k + (num_edges - 1) * (k - 1)
            vertices_needed = k + (num_edges - 1) * (k - 1)
            if vertices_needed > d
                throw(ArgumentError("Not enough vertices for :num_edges=$num_edges with k=$k; need at least $vertices_needed vertices"))
            end
        else
            # Calculate how many edges we can create with d vertices
            # Each edge uses k vertices; each new edge shares 1 with previous, so adds k-1 new vertices
            # Total vertices needed: k + (num_edges - 1) * (k - 1)
            num_edges = div(d - k, k - 1) + 1
        end
        
        if num_edges < 1
            throw(ArgumentError("Not enough vertices to form even one edge; require d >= k"))
        end
        
        vertex_idx = 1
        for edge_num in 1:num_edges
            edge = collect(vertex_idx:(vertex_idx + k - 1))
            push!(edges, edge)
            # Next edge shares the last vertex, so advance by k-1
            vertex_idx += k - 1
        end
        
        return incidence_from_edges(edges, d)
    elseif kind === :fano
        if d != 7 || k != 3
            throw(ArgumentError(":fano only valid for d==7, k==3"))
        end
        blocks = [
            [1,2,3],
            [1,5,6],
            [1,4,7],
            [2,4,6],
            [2,5,7],
            [3,4,5],
            [3,6,7]
        ]
        return incidence_from_edges(blocks, d)
    else
        throw(ArgumentError("Unknown hypergraph kind: $kind"))
    end
end

using PlotlyJS

function plotTensor(tensor::AbstractArray, threshold::Float64=1e-2; 
                   xlabel::String="X", ylabel::String="Y", zlabel::String="Z",
                   title::String="3D Tensor Visualization", color::String="blue")

    # function for removing small entries
    dropSmall = x -> abs(x)< threshold ? 0 : x
    tensor = tensor .|> dropSmall
    
    # Get indices of non-zero values in the tensor
    indices = findall(x -> x != 0, tensor)
    dims = size(tensor)

    # Extract x, y, z coordinates and values
    x_coords = [idx[1] for idx in indices]
    y_coords = [idx[2] for idx in indices]
    z_coords = [idx[3] for idx in indices]
    values = [abs(tensor[idx]) for idx in indices]
    
    # Scale marker sizes proportional to values
    # Normalize values to a reasonable marker size range (2-20)
    if length(values) > 0
        min_val, max_val = extrema(values)
        if min_val ≈ max_val
            marker_sizes = fill(5.0, length(values))  # All same size if all values equal
        else
            # Scale values to range [2, 20] for marker sizes
            marker_sizes = 2.0 .+ 18.0 .* (values .- min_val) ./ (max_val - min_val)
        end
    else
        marker_sizes = Float64[]
    end

    println("Plotting $(length(indices)) points with value-proportional sizes...")
    
    # Create 3D scatter plot with bounding box based on tensor dimensions
    p = PlotlyJS.Plot(scatter3d(
        x=x_coords, 
        y=y_coords, 
        z=z_coords,
        mode="markers",
        marker=attr(size=marker_sizes, opacity=0.6, color=color)
    ), Layout(
        scene=attr(
            xaxis=attr(range=[1, dims[1]+1], title=xlabel),
            yaxis=attr(range=[1, dims[2]+1], title=ylabel),
            zaxis=attr(range=[1, dims[3]+1], title=zlabel),
            aspectmode="cube"
        ),
        title=title
    ))
    
    # Return the plot object to let notebook handle rendering
    return p
end

println("Dleto.jl loaded successfully.")
