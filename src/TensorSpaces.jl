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

module TensorSpaces

import LinearAlgebra

export act, spin, randomize, changeTensor, Engagement, Primal, Dual, Ambidextrous, Disengaged

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

struct TensorSpace
    frame :: Vector{Int}
    category :: Vector{Engagement}
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

    # Special high-speed cases for low valence, call into `TensorOperations.jl`
    if ndims(t) == 1. # t a vector, so Julia "column" 
        return mat*t
    elseif ndims(t) == 2 
        if axis == 1 # t a matrix
            return mat * t
        else    
            return t * mat
    elseif ndims(t) == 3 # t an a x b x c tensor
        if axis == 1
            @tensor res[i,j,k] := t[i',j,k] * mat[i,i']
            return res
        elseif axis == 2
            @tensor res[i,j,k] := t[i,j',k] * mat[j,j']
            return res
        else
            @tensor res[i,j,k] := t[i,j,k'] * mat[k,k']
            return res
    elseif ndims(t) == 4 # t an a x b x c x d tensor
        if axis == 1
            @tensor res[i,j,k,l] := t[i',j,k,l] * mat[i,i']
            return res
        elseif axis == 2
            @tensor res[i,j,k,l] := t[i,j',k,l] * mat[j,j']
            return res
        elseif axis == 3
            @tensor res[i,j,k,l] := t[i,j,k',l] * mat[k,k']
            return res
        else
            @tensor res[i,j,k,l] := t[i,j,k,l'] * mat[l,l']
            return res
    end

    ## Backup options is reshape and pay the price, but at least we use views so no copies.

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

"""
act(t ::AbstractArray, category::Array{Engagement}, m::Vector{A} where A <: AbstractMatrix) ::Array

Act on the engaged axes of the tensor category with the matrices given.
"""
function act(t ::AbstractArray, category::Array{Engagement}, m::Vector{A} where A <: AbstractMatrix) ::Array
    if (length(category) != length(m))
        throw(DimensionMismatch("Must act on all engaged axes."))
    end
    res= t;
    for a in 1:ndims(t)
        if category[a] == Disengaged
            continue
        elseif category[a] == Dual
            res = act( res, m[a]', a)
        else
            res = act( res, m[a], a)
        end
    end
    return res
end;

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
randomize(t::AbstractArray)

Picks 3 random orthogonal matrices and use them to peform a random basis change of a 3 tensor.
Retruns named tupple with coordinates .tensor, .Xchange, .Ychange, .Zchange
""" 
function randomize(t::AbstractArray)
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
    matrices = collect([ getRand(size(t,a)) for a in 1:ndims(t) ])
    tensor = act(t,matrices)
    return (;tensor, matrices)
end;

end # module TensorSpaces


