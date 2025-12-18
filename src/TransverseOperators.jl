#
# Strata Dleto: Transverse Operators
#   Creation and application of transverse operators for tensors.
#
# -----------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------

"""
    TransverseOperators

    This is used primarily as an interface for compressing the 
    representation of transverse operators between tensor spaces allowing 
    for compact representations that exploit available symmetries
    or impose constraints while behaving compatibly with general transverse operators.

    A tranverse operator `Ops` should extend `TransverseOperators` implementing 
    the functions 
    - `contains(Ops, mats::Vector{Matrix})`
    - `transverse(Ops, data::Vector{Number})`
    
    and satisfy the "Membership Law": for all data v considered an operator of `Ops`
    ```julia 
    v = contains(Ops,transverse(Ops,v))
    ```
    If this rule is honored then Dleto can use `unsafe_contains` for efficiency.

    The built in transverse operator sets are:
    - `UniversalOps`: all operators on the engaged axes
    - `SymmetricOps`: symmetric operators on the engaged axes
    - `SkewSymmetricOps`: skew-symmetric operators on the engaged axes (TODO)
    - `DiagonalOps`: diagonal operators on the engaged axes 
    - `UserDefinedOps`: user defined insertion and matrix functions (TODO)
"""
module TransverseOperators

using LinearAlgebra
# using SkewLinearAlgebra
using ..TensorSpaces: Engagement, Primal, Dual, Ambidextrous, Disengaged

export TransverseOperators, UniversalOps, SymmetricOps, DiagonalOps

"""
    TransverseOperators

    A subspace of operators between tensors in a tensor category.

    This is used primarily as an interface for compressing the 
    representation of transformations to exploint available symmetries
    and constraints while remaining compatible with general tensor spaces.


"""
abstract type TransverseOps end

"""
    Return the native encoding of an operator in the transverse set,
    or `Nothing` if it is not a member.
"""
function contains(ops::TransverseOps, mats::Vector{AbstractMatrix}) :: Union{Vector{Number}, Nothing} end
function unsafe_contains(ops::TransverseOps, mats::Vector{Matrix}) :: Vector{Number} end

"""
    Convert the native encoding of an operator in the transverse set
    into a vector of matrices representing the operator on each engaged axis.
"""
function transverse(ops::TransverseOps, data::Vector{Number} ) :: Vector{AbstractMatrix} end


#------------------------------- Universal Operators -------------------------------------

struct UniversalOps <: TransverseOps
    dims :: Vector{Integer}
end

# Flatten the list of matrices, rejecting if wrong size.
function unsafe_contains(ops::UniversalOps, mats::Vector{Matrix}) :: Vector{Number}
    return [vec(M) for M in mats] |> vcat
end
function contains(ops::UniversalOps, mats::Vector{AbstractMatrix}) :: Union{Vector{Number}, Nothing}
    for (idx, M) in enumerate(mats)
        if size(M,1) != ops.dims[idx] || size(M,2) != ops.dims[idx]
            return nothing
        end
    end
    return unsafe_contains(ops, mats)
end
# Reshap vector into vector of matrices.
function transverse(ops::UniversalOps, data::Vector{Number} ) :: Vector{AbstractMatrix} 
    offset = 0
    mats = Vector{Matrix{eltype(data)}}(undef, length(ops.dims))
    for (idx, n) in enumerate(ops.dims)
        mats[idx] = reshape(data[offset+1:offset + n*n], (n,n))
        offset += n*n
    end
    return mats
end

#------------------------------- Symmetric Operators -------------------------------------
struct SymmetricOps <: TransverseOps
    dims :: Vector{Integer}
end

# Store only the lower triangle of each symmetric matrix
function unsafe_contains(ops::SymmetricOps, mats::Vector{AbstractMatrix}) :: Vector{Number}
    v = eltype(mats[1])[]
    for M in ops.mats
        flat_M = [view(M,i:size(M,1), :) for i in 1:size(M,1)] |> vcat
        push!(v, vec(flat_M)...)
    end
    return v |> vcat
end
function contains(ops::SymmetricOps, mats::Vector{AbstractMatrix}) :: Union{Vector{Number}, Nothing} 
    for (idx, M) in enumerate(mats)
        if size(M,1) != ops.dims[idx] || size(M,2) != ops.dims[idx] || !issymmetric(M)
            return nothing
        end
    end
    return unsafe_contains(ops, mats)
end
function transverse(ops::SymmetricOps, data::Vector{Number} ) :: Vector{AbstractMatrix}
    offset = 0
    mats = Vector{Symmetric}(undef, length(ops.dims))
    for (idx, n) in enumerate(ops.dims)
        M = zeros(eltype(data), n, n)
        for i in 1:n
            for j in i:n
                M[i,j] = data[offset + (i-1)*n - (i-2)*(i-1) ÷ 2 + (j - i +1)]
                # M[j,i] = M[i,j]  # Symmetric entry
            end
        end
        mats[idx] = LinearAlgebra.Symmetric(M)
        offset += n*(n+1) ÷ 2
    end
    return mats
end

#------------------------------- Diagonal Operators -------------------------------------
struct DiagonalOps <: TransverseOps 
    dims :: Vector{Integer}
end


# Store only the lower triangle of each symmetric matrix
function unsafe_contains(ops::DiagonalOps, mats::Vector{AbstractMatrix}) :: Vector{Number}
    v = Vector{eltype(mats[1])}(undef,sum(ops.dims))
    offset = 0
    for a in 1:length(ops.dims)
        for i in 1:ops.dims[a]
            v[offset + i] = mats[a][i,i]
        end
        offset += ops.dims[a]
    end
    return v
end
function contains(ops::DiagonalOps, mats::Vector{AbstractMatrix}) :: Union{Vector{Number}, Nothing} 
    for (idx, M) in enumerate(mats)
        if size(M,1) != ops.dims[idx] || size(M,2) != ops.dims[idx] || !isdiagonal(M)
            return nothing
        end
    end
    return unsafe_contains(ops, mats)
end
function transverse(ops::DiagonalOps, data::Vector{Number} ) :: Vector{AbstractMatrix}
    mats = Vector{Diagonal}(undef, length(ops.dims))
    for (idx, n) in enumerate(ops.dims)
        mats[idx] = Diagonal(data[offset+1:offset+n])
    end
    return mats
end


# struct TransverseOperators{A}
#     left :: TensorSpace
#     right :: TensorSpace
#     dimFormula :: Function
#     insert! :: Function
#     toMatrix :: Function
#     description :: String
# end;


# function toOp(ops::TransverseOps{A}, data::A ) :: Vector{Matrix{Float64}} where A
#     engaged = findall( e -> e != Disengaged, ops.left.category )
#     offset = 0
#     op_mats = Vector{Matrix{Float64}}(undef, length(engaged))
#     for (idx, a) in enumerate(engaged)
#         n = size(t,a)
#         d = ops.dimFormula(a,t)
#         mat_a = ops.toMatrix(mats, n, offset)
#         op_mats[idx] = mat_a
#         offset += d
#     end
#     return op_mats
# end


# function uni_insert!(M, row_offset::Integer, col_offset::Integer, coeffs::Vector{Number}, tube::AbstractMatrix, i_a)
#     d = size(tube,1)
#     col_start = col_offset + 1+(i_a-1) * d
#     insert_data = coeffs .* tube'           
#     col_end = col_start + d -1
#     # view accesses a submatrix without copying
#     view(M, (row_offset+1):(row_offset+size(coeffs,1)), col_start:col_end) .= insert_data
# end

# """
#     Create a universal operators with selected engaged axes.
# """
# function UniversalOps(category::Array{Engagement}, poly::Array{T,2}) :: TransverseOperators where T
#     engaged = findall(e -> e != Disengaged, category)
#     uni_dims(a,t) = a in engaged ? size(t,a)*size(t,a) : 0
#     uni_matrix(u::Vector, n:: Integer, offset::Integer )::AbstractMatrix = reshape( u[offset+1:offset+n*n],  (n,n) )
#     return TransverseOperators(uni_dims, uni_insert!, uni_matrix, "Universal Operators")
# end;

# # Position of (i,j) in lower triangular d×d matrix (i≥j)
# sym_pos(d,i,j) = i >= j ? (i-1)*i ÷ 2 + j : (j-1)*i ÷ 2 + i

# # Inverse: given position k in lower triangular storage, return (i,j)
# function sym_index(d, k)
#     # Row i starts at position (i-1)*i/2 + 1
#     # Find i such that (i-1)*i/2 < k ≤ i*(i+1)/2
#     i = ceil(Int, (-1 + sqrt(1 + 8*k)) / 2)
#     j = k - (i-1)*i ÷ 2
#     return (i, j)
# end

# # Position of (i,j) in upper triangular d×d matrix (i≤j)
# upper_tri_pos(d,i,j) = (i-1)*(2*d-i) ÷ 2 + (j-i+1)

# """
#     Create symmetric operators with selected engaged axes.
# """
# function SymmetricOps(category::Array{Engagement}, poly::Matrix) 
#     engaged = findall(e -> e != Disengaged, category)
#     sym_dims(a,t) = a in engaged ? size(t,a)*(size(t,a)+1) ÷ 2 : 0
#     function sym_insert!(M, row_offset, col_offset, coeff, tube, i_a)
#         d = size(tube,1)
#         insert_data = zeros(eltype(M), d*(d+1) ÷ 2)
#         for x in 1:d
#             insert_data[sym_pos(d, x, i_a)] = tube[x]
#         end
#         col_start = col_offset+1
#         col_end = col_start + d*(d+1) ÷ 2 -1
#         view(M, (row_offset+1):(row_offset+size(coeff,1)), col_start:col_end) .= coeff*insert_data'
#     end
#     function sym_matrix(u::Vector, n:: Integer, offset::Integer )::AbstractMatrix 
#         M = zeros(Float64,n,n)
#         for k in 1:(n*(n+1) ÷ 2)
#             (i,j) = sym_index(n, k)
#             M[i,j] = u[offset + k]
#             # M[j,i] = u[offset + k]  # Symmetric entry
#         end
#         # Julia type that stores only upper triangle, 
#         # it also triggers later functions like eigen 
#         # to obey symmetric rules, like have real eigenvalues. 
#         return LinearAlgebra.Symmetric(M) 
#     end;

#     return TransverseOperators(sym_dims, sym_insert!, sym_matrix, "Symmetric Operators")
# end;

end # module TransverseOperators