#
# Strata Dleto: Tensor Space
#   Methods for changing tensors and other categorical operations.
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
# technical function, peroforms inpalce "multiplication" by symmetric matrix 

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
