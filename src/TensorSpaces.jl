
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
    TensorSpaces

    Module providing methods for changing tensors and other categorical operations.

    Tensor spaces are specified by a frame with each axis of the frame being a named 
    vector space and an orientation.

    The orientation of each axis is given by an `Arrows` value:
    - `→`: axis is engaged in primal mode
    - `←`: axis is engaged in dual mode, an appropriate transpose is applied 
    - `↔`: axis is engaged in both primal and dual modes
    - `·`: axis is not engaged so constraints are skipped, applied as 0 or 1, depending on context.

    Axes are represented by the `Axis` type, which can be used with any other tensor.
    For a given tensor space `TS`, the frame can be obtained by `frame(TS)`, which returns
    a vector of tuples of the form `(Axis, Arrow)`.
"""
module TensorSpaces

import LinearAlgebra
# import ITensors
import TensorOperations

export AbstractTensor, ArrayTensor, frame, Axis, Arrows, right, left, both, none

"""
    Arrows

An enum-like type representing the role of an axis in a chisel:
- `→`: axis is engaged in primal mode
- `←`: axis is engaged in dual mode, an appropriate transpose is applied 
- `↔`: axis is engaged in both primal and dual modes
- `·`: axis is not engaged so constraints are skipped, applied as 0 or 1, depending on context.
"""
@enum Arrows begin
    right
    left
    both
    none
end

#---------------- Axis type ---------------------------------------------------
# Borrowing the philosophy of ITensor with named axes.

"""
    Axis

A structure representing an axis in a tensor space.
Fields:
- `dim`: An integer representing the dimension of the axis.
- `name`: (optional) A symbol representing the name of the axis, e.g. `:genre`, `:particle1` etc.

If two axis are given the same name (not `:unnamed`), they are considered equal.
Otherwise each instance is unique.  For example
```julia
a1 = Axis(3, name=:x)
a2 = Axis(4, name=:x)
a3 = Axis(3)
a4 = Axis(3)
a1 == a2  # true, same name
a3 == a4  # false, different instances
```
"""
struct Axis 
    dim::Integer
    name::Symbol
    function Axis(dim::Integer; name::Symbol=:unnamed)
        new(dim, name)
    end
end

import Base: ==, hash

function ==(a::Axis, b::Axis) 
    if (a.name == b.name) && (a.name != :unnamed)
        return true
    else
        return a === b
    end
end

# Override hash to ensure uniqueness per instance
function hash(a::Axis, h::UInt)
    return hash(objectid(a), h)
end

#---------------- TensorSpace abstract type -----------------------------------


"""
    AbstractTensor{t}

    An abstract type representing a tensor with named axes and orientations.
"""
abstract type AbstractTensor end

"""
    frame(ts::AbstractTensor) :: Vector{Tuple{Axis,Arrows}}

    Returns the axis and its orientation in the frame of `ts`
"""
function frame(ts::AbstractTensor)::Vector{Tuple{Axis,Arrows}} end


"""
    join(Γs ::Vector{AbstractTensor})::AbstractTensor

    Applies a sequence of tensors `Γs` by contracting over all shared axes.
    It is up to the individual `AbstractTensor` implementations to define 
    the contraction strategy.  For example, some may perform low-level 
    array eager operations while others may use lazy evaluation.

    Returns the resulting tensor after all contractions.
"""
function join(Γs ::Vector{AbstractTensor})::AbstractTensor end


#---------------- Abstract Array Tensors --------------------

"""
    ArrayTensor{Number} <: AbstractTensor

    A concrete implementation of `AbstractTensor` that wraps an `AbstractArray` 
    along with its frame (axes and orientations).
"""
struct ArrayTensor{Number} <: AbstractTensor
    data::AbstractArray{Number}
    frame::Vector{Tuple{Axis, Arrows}}
end
"""
    ArrayTensor(data::AbstractArray, frame::Vector{Tuple{Axis, Arrows}}) :: ArrayTensor

    Constructs an `ArrayTensor` from the given data and frame.
"""
function ArrayTensor(data::AbstractArray, frame::Vector{Tuple{Axis, Arrows}}) :: ArrayTensor
    if length(frame) != ndims(data)
        throw(DimensionMismatch("Tensor and frame must have the same length."))
    end
    return ArrayTensor{eltype(data)}(data, frame)
end
"""
    ArrayTensor(data::AbstractArray, orientation::Vector{Arrows}) :: ArrayTensor   

    Constructs an `ArrayTensor` from the given data and orientation.
    The axes are created as unnamed axes with dimensions matching the data.
"""
function ArrayTensor(data::AbstractArray, orientation::Vector{Arrows}) :: ArrayTensor   
    if length(orientation) != ndims(data)
        throw(DimensionMismatch("Tensor and category must have the same length."))
    end
    frame = [ (Axis(size(data, i)), orientation[i]) for i in 1:ndims(data) ]
    return ArrayTensor{eltype(data)}(data, frame)
end
# Tell Julia how to look up array-like methods, this way built-in functions work
# with `Γ[a]`` syntax, `size(Γ)`, `ndims(Γ)`, etc.
function Base.getindex(Γ::ArrayTensor, i::Integer)
    return Γ.data[i]
end
function Base.size(Γ::ArrayTensor)
    return size(Γ.data)
end
function Base.size(Γ::ArrayTensor, a::Integer)
    return size(Γ.data, a)
end
function Base.ndims(Γ::ArrayTensor)
    return ndims(Γ.data)
end

"""
    frame(Γ::ArrayTensor) :: Vector{Tuple{Axis,Arrows}}

    Returns the axis and its orientation in the frame of `Γ`
"""
function frame(Γ::ArrayTensor)::Vector{Tuple{Axis,Arrows}}
    return Γ.frame
end;

function join(Γs ::Vector{ArrayTensor})::ArrayTensor
    if length(Γs) == 0
        throw(ArgumentError("No tensors to join."))
    end
    res = Γs[1]
    for Γ in Γs[2:end]
        # find shared axes
        shared_axes = Dict{Int,Int}()
        for (i1, (axis1, _)) in enumerate(res.frame)
            for (i2, (axis2, _)) in enumerate(Γ.frame)
                if axis1 == axis2
                    shared_axes[i1] = i2
                end
            end
        end
        # perform contraction over shared axes
        new_data = res.data
        new_frame = res.frame
        for (i1, i2) in shared_axes
            @tensor new_data[:] := new_data[i1'] * Γ.data[i2']
            deleteat!(new_frame, i1)
        end
        # append remaining axes from Γ
        for (i, (axis, arrow)) in enumerate(Γ.frame)
            if !(i in values(shared_axes))
                push!(new_frame, (axis, arrow))
            end
        end
        res = ArrayTensor(new_data, new_frame)
    end
    return res
end;

"""
    apply(Γ::AbstractArray{T}, A::Vector{Number}, us::Vector{Vector{T}}) :: ArrayTensor{T} where {T}

    Applies a set of vectors `us` to the tensor `Γ` along the specified axes `A`.
    This effectively contracts the tensor with the vectors along those axes.

    Returns the resulting tensor after contraction.
"""


function apply(Γ::AbstractArray{T}, A::Vector{Number}, us::Vector{Vector{T}}) :: ArrayTensor{T} where {T}
    # check sizes
    if length(A) != length(us)
        throw(DimensionMismatch("Length of axes and vectors must be the same."))
    end
    for (i,a) in enumerate(A)
        if size(Γ, a) != length(us[i])
            throw(DimensionMismatch("Size of tensor and vector must be the same on axis $a."))
        end
    end
    # perform contraction
    res = Γ
    for (i,a) in enumerate(A)
        vec = us[i]
        @tensor res[:] := res[a'] * vec[a']
    end
    return res
end;



# ---------------- Helper functions --------------------

function Base.show(io::IO, ::MIME"text/plain", Arrows::Array{Arrows})
    for e in Arrows
        if e == right
            print(io, "→")
        elseif e == left
            print(io, "←")
        elseif e == both
            print(io, "↔")
        else
            print(io, "·" )
        end
    end
end
Base.show(io::IO, e::Arrows) = Base.show(io, MIME("text/plain"), [e])

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
        end
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
        end
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
act(t ::AbstractArray, category::Array{Arrows}, m::Vector{A} where A <: AbstractMatrix) ::Array

Act on the engaged axes of the tensor category with the matrices given.
"""
function act(t ::AbstractArray, category::Array{Arrows}, m::Vector{AbstractMatrix}) ::Array
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


