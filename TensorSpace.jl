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




"""
    Engagement

An enum-like type representing the role of an axis in a chisel:
- `Primal`: axis is engaged in primal mode
- `Dual`: axis is engaged in dual mode, an approprriate transpose is applied 
- `Ambidextrous`: axis is engaged in both primal and dual modes
- `Disengaged`: axis is not engaged so constraints are dropped or applied as identity on these axes
"""
@enum Engagement begin
    Primal
    Dual
    Ambidextrous
    Disengaged
end

function Base.show(io::IO, ::MIME"text/plain", engagement::Array{Engagement})
    for e in engagement
        if e == Primal
            print(io, "→")
        elseif e == Dual
            print(io, "←")
        elseif e == Ambidextrous
            print(io, "↔")
        else
            print(io, "·" )
        end
    end
end
Base.show(io::IO, e::Engagement) = Base.show(io, MIME("text/plain"), [e])

#-------------------------------
# Action of matrices on the tensor
# raise exception if the sizes are not compatible


"""
act(t ::AbstractArray, mat ::AbstractArray, axis ::Integer)

Act on a tensor by a matrix via a given axis 
"""
function act(t ::AbstractArray, mat ::AbstractMatrix, axis ::Integer) ::Array
    if (axis < 1) || (axis >  ndims(t))
        throw(DimensionMismatch("Axis not in frame of tensor."))
    end
    dims = [size(t)...]
    matsizes = size(mat)

    #matrices act on the right
    if matsizes[1] != dims[axis]
        throw(DimensionMismatch("incompatible sizes of tensor and matrices"))
    end

    # Flatten the tensor
    p=prod(dims[1 : (axis - 1)])
    q=prod(dims[(axis + 1) : end])

    # reshape the tensor into 3 matrix
    ten = reshape(t, p, dims[axis], q)
    # make a place to store  result
    res = similar(ten, p, matsizes[2], q)
    # use matrix multiplication to compute the action 
    for i = 1:q
        LinearAlgebra.mul!(@view(res[:, :, i]), @view(ten[:, :, i]), mat)
    end
    # find the new size
    dims[axis] = matsizes[2]
    # reshape the result and return 
    return reshape(res, Tuple(dims))
end;


#-------------------------------
# Act on all Axises
# m is a list of matrices

"""
actAllDirections(t ::AbstractArray, m::Vector{A} where A <: AbstractArray) ::Array

Act on a tensor by a many matrices on all axes 
"""
function act(t ::AbstractArray, m::Vector{A} where A <: AbstractMatrix) ::Array
    if (ndims(t) != length(m))
        throw(DimensionMismatch("wrong size of vector of matrices"))
    end
    res= t;
    for a = 1:ndims(t)
        res = act( res, m[a], a)
    end
    return res
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
randomize(t::AbstractArray)

Picks 3 random orthogonal matrices and use them to peform a random basis change of a 3 tensor.
Retruns named tupple with coordinates .tensor, .Xchange, .Ychange, .Zchange
""" 
function randomize(t::AbstractArray)
    mats = collect([ randomMatrix(size(t,a)) for a in 1:ndims(t) ])
    tensor = act(t,mats)
    return (;tensor, matrices)
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


