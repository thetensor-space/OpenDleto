#
# Strata Dleto: Abstract List of  Operators
#   Creation and application of transverse operators for tensors.
#
# -----------------------------------------------------------------------------
# Copyright 2022-2026 Peter A. Brooksbank, Martin D. Kassabov, James B. Wilson
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
#-----------------------------------------------------------------------------


"""
    Abstract List of Operators

    TO BE WRITTEN
"""

"""
    Abstract Global Operators

    Abstract Class for providing encoding and decoving data from list of matrices/ITensors to in internal vectors.
    Need to provide functions
    -- coodinates: turn list of matrices into vector or retruns nothing.
    -- unsafe_coordinates: same but assumes that the vector is in the image and does not do any checking
    -- embedMatrices/embedITensors: turns a vector into list of matrices 
        (this needs to be a linear map, which is left(right?) of coordinatres) 
    -- unsafe_embedMatrices/unsafe_embedITensors: as above but does not do any checking
    -- transposeEmbed: linear dual(transpose) of the emnedding, turns list of matrices into a vector

    several function like globalDim, axisDims, valancy: which provide data for the sizes of matrices
        
    most safe functions are auto-generated from the unsafe one by adding trivial checks
    
    -- reduceByEngaged(::TransverseOps, ::Vector{Bool})::TransverseOps produce new GlobalOps,
        where some of the axis are not engaged

"""
abstract type ListOperators end

"""
    Return the native encoding of an operator in the transverse set,
    or `nothing` if it is not a member.
"""
function coordinates(LOp::ListOperators, Mats::Vector{<: AbstractMatrix} ) :: Union{Vector{<:Number}, Nothing} 
    @assert false "Calling Placeholder Abstract Function"
end;

function unsafe_coordinates(LOp::ListOperators, Mats::Vector{<: AbstractMatrix} ) :: Vector{<: Number} 
    @assert false "Calling Placeholder Abstract Function"
end;

#only defined is all localOps are linear
function transposeEmbed(LOp::ListOperators, Mats::Vector{<:AbstractMatrix}) :: Union{Vector{<:Number},Nothing}
    @assert false "Calling Placeholder Abstract Function"
end;

#might return error
function unsafe_transposeEmbed(LOp::ListOperators, Mats::Vector{<:AbstractMatrix}) :: Vector{<:Number} 
    @assert false "Calling Placeholder Abstract Function"
end;


"""
    Convert the native encoding of an operator into matrices.
"""
function embedMatrices(LOp::ListOperators, data::Vector{<:Number} ) ::Union{Vector{<:AbstractMatrix},Nothing}  
    @assert length(data) == globalDim(LOp) "Incompatable Data"
    unsafe_embedMatrices(LOp,data)
end;

function unsafe_embedMatrices(TOp::ListOperators, data::Vector{<:Number} ) ::Vector{<:AbstractMatrix}
    @assert false "Calling Placeholder Abstract Function"
end;


"""
    dimension of the vectors in the encoding
"""
function globalDim(LOp::ListOperators)::Integer 
    @assert false "Calling Placeholder Abstract Function"
end;

function axisDims(LOp::ListOperators)::Vector{<:Integer} 
    @assert false "Calling Placeholder Abstract Function"
end;


function valency(LOp::ListOperators)::Integer 
    @assert false "Calling Placeholder Abstract Function"
end;

### the second output is an expand map which is linear
### as a result if the some of the local operaots are not 
### linear  then embed of the expand map might not be defined 
### using expand map when all oeprators are linear should be OK
### similarly using the reduce_map = expand_map'  should be OK always

function reduceByEngaged(LOp::ListOperators, engaged::Vector{Bool})::NamedTuple{(:rLOp, :expand_func, :reduce_map), Tuple{ListOperators, Function, LinearMaps.LinearMap}}
    @assert false "Calling Placeholder Abstract Function"
end;



function isLinear(LOp::ListOperators)::Bool
    @assert false "Calling Placeholder Abstract Function"
end;

function isInvertible(LOp::ListOperators)::Bool
    @assert false "Calling Placeholder Abstract Function"
end;

function generate_random(LOp::ListOperators)::Vector{<:Number} 
    @assert false "Calling Placeholder Abstract Function"
end;


###next two functions are only defined for List of operators which are linear
###otherwise will throw exception
function simplifyTo(Op::ListOperators)::NamedTuple{(:D, :T), Tuple(ListOperators,ListOperators)}  
    @assert false "Calling Placeholder Abstract Function"
end;

"""
Function which does the simplification (ie diagonalization of symetric matrices)
"""
function simplify(Op::ListOperators, data::Vector{<:Number} )::NamedTuple{(:d, :t), Tuple(Vector{<:Number},Vector{<:Number}) } 
    @assert false "Calling Placeholder Abstract Function"
end;


# #probably needs to be moved somewhere else
# function __asMatrix(T::ITensor)::AbstractMatrix
#     fr = inds(T)
#     n = ITensors.dim(fr[1])
#     m = ITensors.dim(fr[2])
#     A = zeros(eltype(T),n,m)
#     for ci in CartesianIndices(A)
#         A[ci] = T[ci]
#     end
#     return A
# end;

# function __asMatrixTranspose(T::ITensor)::AbstractMatrix
#     fr = inds(T)
#     n = ITensors.dim(fr[1])
#     m = ITensors.dim(fr[2])
#     A = zeros(eltype(T),m,n)
#     for ci in CartesianIndices(A)
#         A[ci] = T[ci[2],ci[1]]
#     end
#     return A
# end;

# # function to create temp index
# function __globalOpsMakeTempIndex(I::Index)::Index
#     return Index(ITensors.dim(I),"Site:Der,$I")
# end;
