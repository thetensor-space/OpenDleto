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

# """
#     TransverseOperators

#     This is used primarily as an interface for compressing the 
#     representation of transverse operators between tensor spaces allowing 
#     for compact representations that exploit available symmetries
#     or impose constraints while behaving compatibly with general transverse operators.

#     The built in transverse operator sets are:
#     - `UniversalOps`: all operators on the engaged axes
#     - `SymmetricOps`: symmetric operators on the engaged axes
#     - `SkewSymmetricOps`: skew-symmetric operators on the engaged axes (TODO)
#     - `DiagonalOps`: diagonal operators on the engaged axes 
#     - `UserDefinedOps`: user defined insertion and matrix functions (TODO)

#     In mathematicals terms, assume we fix tensor spaces $T$ and $S$ in the same 
#     transverse tensor category with engaged axes $\mathcal{E}$.  Write 
#     $\hom_{\epsilon}(U,V)$ for the homomorphisms between space $U$ and $V$ 
#     in orientation:
#     - $U\to V$ for $\epsilon=$`Right`
#     - $U\from V$ for $\epsilon=$`Left`
#     - $U\leftrightarrow V$ for $\epsilon=$`Ambidextrous`
#     - $U\nleftrightarrow V$ for $\epsilon=$`Disengaged` (the zero map)

#     A transverse operator set $\Omega$ is type together with functions 
#     \[ 
#         \iota :\Omega \to \prod_{a\in \mathcal{E}} \hom_{\epsilon(a)}(T_a, S_a)
#         \qquad 
#         \in : \prod_{a\in \mathcal{E}} \hom_{\epsilon(a)}(T_a, S_a) \to \Omega^?
#     \]
#     where $\Omega^?=\Omega \sqcup Nothing$ and such that $\in\iota(\omega) = \omega$.
#     The function $\in $ is a constructive membership test that returns the given 
#     operators encoded in $\Omega$ if they belong to the set, and `Nothing` otherwise.
#     The function $\iota$ is an injection that decodes the operators in $\Omega$
#     as regular transverse operators.

#     To work with derivations and chisel operations, transverse operators must 
#     also provide a formula for the dimension of the spaces of operators in $\Omega$
#     in each engaged axis.
# """
module TransverseOperators

using LinearAlgebra
# using SkewLinearAlgebra
using ..TensorSpaces: Engagement, Primal, Dual, Ambidextrous, Disengaged

export TransverseOperators, UniversalOps, SymmetricOps

"""
    TransverseOperators

    A subspace of operators between tensors in a tensor category.

    This is used primarily as an interface for compressing the 
    representation of transformations to exploint available symmetries
    and constraints while remaining compatible with general tensor spaces.


"""
abstract type TransverseOperators{A} end

"""
    Return the native encoding of an operator in the transverse set,
    or `Nothing` if it is not a member.
"""
function member(ops::TransverseOperators{A}, mats::Vector{Matrix}) :: Union{A, Nothing} where A end

"""
    Convert the native encoding of an operator in the transverse set
    into a vector of matrices representing the operator on each engaged axis.
"""
function transverse(ops::TransverseOperators{A}, data::A ) :: Vector{Matrix} where A end

#------------------------------- Universal Operators -------------------------------------
struct UniversalOperators <: TransverseOperators{Vector{Matrix}} 
    mats :: Vector{Matrix}
end

# trivial encoding and decoding
member(ops::UniversalOperators, mats::Vector{Matrix}) :: Union{Vector{Matrix}, Nothing} = mats
tranasverse(ops::UniversalOperators, data::Vector{Matrix} ) :: Vector{Matrix} = data

#------------------------------- Symmetric Operators -------------------------------------
struct SymmetricOperators <: TransverseOperators{Vector{Symmetric}} 
    mats :: Vector{AbstractMatrix}
end

# Julia has a built in type for symmetric matrices which stores only the upper triangle
# and furthermore triggers later functions (like eigen methods) to obey symmetric rules,
function member(ops::SymmetricOperators, mats::Vector{AbstractMatrix}) :: Union{Vector{Symmetric}, Nothing} 
    for M in mats
        if !issymmetric(M)
            return nothing
        end
    end
    return LinearAlgebra.Symmetric.(mats)
end
function assymmetric(mats::Vector{AbstractMatrix}):: Vector{Symmetric} 
    return LinearAlgebra.Symmetric.(mats)
end
tranasverse(ops::SymmetricOperators, data::Vector{AbstractMatrix} ) :: Vector{Symmetric} = data

#------------------------------- Diagonal Operators -------------------------------------
struct DiagonalOperators <: TransverseOperators{Vector{Diagonal}} 
    mats :: Vector{Diagonal{Float64, Vector{Float64}}}
end

# Julia has a built in type for symmetric matrices which stores only the upper triangle
# and furthermore triggers later functions (like eigen methods) to obey symmetric rules,
function member(ops::DiagonalOperators, mats::Vector{AbstractMatrix}) :: Union{Vector{Diagonal}, Nothing} 
    for M in mats
        if !isdiagonal(M)
            return nothing
        end
    end
    return LinearAlgebra.Diagonal.(mats)
end
function asdiagonal(mats::Vector{AbstractMatrix}):: Vector{Diagonal} 
    return LinearAlgebra.Diagonal.(mats)
end
tranasverse(ops::DiagonalOperators, data::Vector{AbstractMatrix} ) :: Vector{Diagonal} = data



# struct TransverseOperators{A}
#     left :: TensorSpace
#     right :: TensorSpace
#     dimFormula :: Function
#     insert! :: Function
#     toMatrix :: Function
#     description :: String
# end;


# function toOp(ops::TransverseOperators{A}, data::A ) :: Vector{Matrix{Float64}} where A
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

"""
    Create a universal operators with selected engaged axes.

"""

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