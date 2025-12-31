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

    A transverse operator `Ops` should extend implement two functions
    - `member(Ops, mats::Vector{Matrix})`
    - `transverse(Ops, data::Vector{Number})`
    
    and satisfy the "Membership Law": for all data v considered an operator of `Ops`
    ```julia 
    v = member(Ops,transverse(Ops,v))
    ```
    If this rule is honored then Dleto can use `unsafe_member` for efficiency.

    The built in transverse operator sets are:
    - `UniversalOps`: all operators on the engaged axes
    - `GLOps`: invertible operators on the engaged axes
    - `SymmetricOps`: symmetric operators on the engaged axes
    - `OrthogonalOps`: orthogonal operators on the engaged axes 
    - `DiagonalOps`: diagonal operators on the engaged axes 
    - `TorusOps`: diagonal invertible operators on the engaged axes (TODO)
    - `HermitianOps`: skew-symmetric operators on the engaged axes (TODO)
    - `UnitaryOps`: unitary operators on the engaged axes (TODO)
    - `UserDefinedOps`: user defined insertion and matrix functions (TODO)
"""
# module TransverseOperators

# using ITensors
# using LinearAlgebra
# using SkewLinearAlgebra

# export TransverseOps, UniversalOps, engaged, frame, transverse, member, unsafe_member #, SymmetricOps, DiagonalOps, InvertibleOps, OrthogonalOps

"""
    TransverseOperators

    A subspace of operators between tensors in a tensor category,
    what is sometimes denoted hom(Γ,Υ), but with builtin functions 
    for compact encoding.

    The alignment of the axes is determined by the matching `Index` terms 
    of the individual tensors stored as `ITensor` terms.

"""
abstract type TransverseOps  end

abstract type LocalTransverseOps  end


"""
    Return the native encoding of an operator in the transverse set,
    or `Nothing` if it is not a member.
"""
function member(LΩ::LocalTransverseOps, mats::Matrix) :: Union{Vector{Number}, Nothing} end
function unsafe_member(LΩ::LocalTransverseOps, mats::Vector{Matrix}) :: Vector{Number} end

"""
    Convert the native encoding of an operator in the transverse set
    into a vector of matrices representing the operator on each engaged axis.
"""
function transverse(LΩ::TransverseOps, dim::Integer, data::Vector{<:Number} ) ::Matrix  end

function localDim(LΩ::LocalTransverseOps, dim::Integer)::Integer end
function containScalars(LΩ::LocalTransverseOps)::Bool  end



struct LocalUniversalOps <: LocalTransverseOps end 
struct LocalDiagonalOps <: LocalTransverseOps end 
struct LocalSymmetricOps <: LocalTransverseOps end 
struct LocalAnitSymmetricOps <: LocalTransverseOps end 


localDim(::LocalUniversalOps, dim::Integer) = dim*dim; 
localDim(::LocalDiagonalOps, dim::Integer) = dim; 
localDim(::LocalSymmetricOps, dim::Integer) = dim*(dim+1) ÷ 2; 
localDim(::LocalAnitSymmetricOps, dim::Integer) = dim*(dim-1) ÷ 2; 

containScalars(::LocalUniversalOps)= true;
containScalars(::LocalDiagonalOps) = true; 
containScalars(::LocalSymmetricOps) = true; 
containScalars(::LocalAnitSymmetricOps) = false; 



function frame(Ω::TransverseOps)::Vector{Index} end
function engaged(Ω::TransverseOps)::Vector{Bool} end


"""
    Return the native encoding of an operator in the transverse set,
    or `Nothing` if it is not a member.
"""
function member(Ω::TransverseOps, mats::Vector{ITensor}) :: Union{Vector{Number}, Nothing} end
function unsafe_member(Ω::TransverseOps, mats::Vector{Matrix}) :: Vector{Number} end

"""
    Convert the native encoding of an operator in the transverse set
    into a vector of matrices representing the operator on each engaged axis.
"""
function transverse(Ω::TransverseOps, data::Vector{<:Number} ) :: Vector{ITensor} end

function dim(Ω::TransverseOps)::Integer end


#------------------------------- Universal Operators -------------------------------------

# narrowed the type for method dispatch
struct UniversalOps <: TransverseOps 
    frame :: Vector{Index}
    engaged :: Vector{Bool}
end

function frame(Ω::UniversalOps)::Vector{Index{Int64}}
    return Ω.frame
end
function engaged(Ω::UniversalOps)::Vector{Bool}
    return Ω.engaged
end
function dim(Ω::UniversalOps)::Integer
    total = 0
    eng = engaged(Ω); fr = frame(Ω)
    println("engaged: ", eng)
    for a in 1:length(fr)
        if eng[a]
            total += ITensors.dim(fr[a])^2
        end
    end
    return total
end
function UniversalOps(frame::Vector{Index{Int64}})
    engaged = [ true for a in 1:length(frame) ]
    return UniversalOps(frame, engaged)
end

# Flatten the list of matrices, assumes ITensors are dense arrays
function unsafe_member(Ω::UniversalOps, mats::Vector{ITensor}) :: Vector{Number}
    # Extract arrays with consistent index ordering matching transverse
    # mats is indexed 1:n_engaged, corresponding to engaged axes in order
    eng = engaged(Ω); fr = frame(Ω)
    engaged_axes = findall(eng)
    return vcat([vec(Array(mats[i], fr[engaged_axes[i]], fr[engaged_axes[i]]')) for i in 1:length(mats)]...)
end
function member(Ω::UniversalOps, mats::Vector{ITensor}) :: Union{Vector{Number}, Nothing}
    eng = engaged(Ω); fr = frame(Ω)
    if length(mats) != length(eng)
        return nothing
    end
    total = 0
    for a in 1:length(fr)
        if eng[a] 
            if (inds(mats[a])[1], inds(mats[a])[2]) != (fr[a], fr[a]')
                return nothing
            else
                total += ITensors.dim(fr[a]) * ITensors.dim(fr[a])
            end
        end
    end
    data = Vector{eltype(mats[1])}(undef, total)
    offset = 0
    for a in 1:length(fr)
        if eng[a]
            M = mats[a]; l = fr[a]; r = fr[a]';
            # not great to copy here but ITensor doesn't accept views
            data[offset+1:offset + size(M,1)*size(M,2)] = collect(M[l=> i, r=> j] for i in 1:ITensors.dim(l), j in 1:ITensors.dim(r))
            offset += size(M,1)*size(M,2)
        end
    end
    return data
end

# Reshape vector into vector of matrices.
function transverse(Ω::UniversalOps, data::Vector{<:Number} ) :: Vector{ITensor} 
    eng = engaged(Ω); fr = frame(Ω)
    offset = 0
    n_engaged = count(eng)
    mats = Vector{ITensor}(undef, n_engaged)
    next = 1
    for a in findall(eng)
        l = ITensors.dim(fr[a]); r = ITensors.dim(fr[a]');
        # small technical issue is that Matrix copies data but ITensor does not 
        # accept output of reshape which is AbstractArray but not an Array.
        # so some copying going on here
        mat = Matrix(reshape(view(data, (offset+1):(offset + l*r)), (l, r)))
        mats[next] = ITensor(mat, fr[a], fr[a]')
        offset += l*r
        next += 1
    end
    return mats
end

# #-------------------------------- Invertible Operators -------------------------------------
# """
#     Invertible Transverse Operators

#     A subspace of operators between tensors in a tensor category
#     where each operator on an engaged axis is invertible.

# """
# struct InvertibleOps <: TransverseOps     
#     frame :: Vector{Index}
#     engaged :: Vector{Bool}
# end

# function frame(Ω::InvertibleOps)::Vector{Index}
#     return Ω.frame
# end
# function engaged(Ω::InvertibleOps)::Vector{Bool}
#     return Ω.engaged
# end

# # Flatten the list of matrices, rejecting if wrong size.
# function unsafe_member(Ω::InvertibleOps, mats::Vector{ITensor}) :: Vector{Number}
#     return [vec(asarray(M)) for M in mats] |> vcat
# end
# function member(Ω::InvertibleOps, mats::Vector{ITensor}) :: Union{Vector{Number}, Nothing}
#     if length(mats) != length(Ω.engaged)
#         return nothing
#     end
#     frame = Ω.frame
#     for a in 1:length(Ω.engaged)
#         e, f = Ω.engaged[a]
#         if inds(mats[a]) != (frame[e], frame[f]')
#             return nothing
#         end
#     end
#     total = sum(dim(Ω.frame[e]) * dim(Ω.frame[f]) for (e,f) in Ω.engaged)
#     data = Vector{eltype(mats[1])}(undef, total)
#     offset = 0
#     for (a, (e,f)) in enumerate(Ω.engaged)
#         M = mats[a]; 
#         if det(M) == 0
#             return nothing
#         end
#         l = frame[e]; r = frame[f]';
#         data[offset+1:offset + size(M,1)*size(M,2)] = vec(M[l=> i, r=> j] for i in 1:dim(l), j in 1:dim(r))
#         offset += size(M,1)*size(M,2)
#     end
#     return data
# end
# # Reshape vector into vector of matrices.
# function transverse(Ω::InvertibleOps, data::Vector{Number} ) :: Vector{AbstractMatrix} 
#     offset = 0; frame = Ω.frame
#     mats = Vector{Matrix{eltype(data)}}(undef, length(Ω.engaged))
#     for e in Ω.engaged
#         l = dim(frame[e]); r = dim(frame[e]')
#         # small technical issue is that Matrix copies data but ITensor does not 
#         # accept output of reshape which is AbstractArray but not an Array.
#         # so some copying going on here
#         mat = Matrix(reshape(view(data, (offset+1):(offset + l*r)), (l, r)))
#         mats[e] = ITensor(mat, frame[e], frame[e]')
#         offset += l*r
#     end
#     return mats
# end



# #------------------------------- Symmetric Operators -------------------------------------
# struct SymmetricOps <: TransverseOps
#     frame :: Vector{Index}
#     engaged :: Vector{Bool}
# end
# function frame(Ω::SymmetricOps)::Vector{Index}
#     return Ω.frame
# end
# function engaged(Ω::SymmetricOps)::Vector{Bool}
#     return Ω.engaged
# end
# function SymmetricOps(frame::Vector{Index})
#     engaged = [ true for a in 1:length(frame) ]
#     return SymmetricOps(frame, engaged)
# end

# # Store only the lower triangle of each symmetric matrix
# function unsafe_member(Ω::SymmetricOps, mats::Vector{ITensor}) :: Vector{Number}
#     v = eltype(mats[1])[]
#     for M in mats
#         flat_M = [view(store(M),i:size(M,1), :) for i in 1:size(M,1)] |> vcat
#         push!(v, vec(flat_M)...)
#     end
#     return v |> vcat
# end
# function member(ops::SymmetricOps, mats::Vector{ITensor}) :: Union{Vector{Number}, Nothing} 
#     for (idx, M) in enumerate(mats)
#         if size(M,1) != ops.ldims[idx] || size(M,2) != ops.rdims[idx] || !issymmetric(M)
#             return nothing
#         end
#     end
#     return unsafe_member(ops, mats)
# end
# function transverse(ops::SymmetricOps, data::Vector{Number} ) :: Vector{ITensor}
#     eng = engaged(ops); fr = frame(ops)
#     offset = 0
#     Xs = Vector{ITensor}(undef, length(ops.dims))
#     for a in 1:length(fr)
#         if !eng[a]
#             continue
#         end
#         d = dim(fr[a])
#         Xs[a] = ITensor(fr[a], fr[a]') # TBD: does ITensor have a symmetric type?
#         for i in 1:d
#             for j in i:d
#                 (Xs[a])[fr[a]=>i,fr[a]'=>j] = data[offset + (i-1)*d - (i-2)*(i-1) ÷ 2 + (j - i +1)]
#                 (Xs[a])[fr[a]=>j,fr[a]'=>i] = data[offset + (i-1)*d - (i-2)*(i-1) ÷ 2 + (j - i +1)]
#                 # M[j,i] = M[i,j]  # Symmetric entry
#             end
#         end
#         offset += d*(d+1) ÷ 2
#     end
#     return Xs
# end

# struct OrthogonalOps <: TransverseOps
#     frame :: Vector{Index}
#     engaged :: Vector{Bool}
# end
#     frame :: Vector{Index}
#     engaged :: Vector{Bool}
# end

# function frame(Ω::UniversalOps)::Vector{Index}
#     return Ω.frame
# end
# function engaged(Ω::UniversalOps)::Vector{Bool}
#     return Ω.engaged
# end
# function unsafe_member(ops::OrthogonalOps, mats::Vector{Matrix}) :: Vector{Number}
#     return [vec(M) for M in mats] |> vcat
# end
# """
#      checks orthogonality to within tolerance 1e-8, if that fails and you know it is 
#      orthogonal use unsafe_member for efficiency
# """
# function member(ops::OrthogonalOps, mats::Vector{AbstractMatrix}) :: Union{Vector{Number}, Nothing}
#     for (idx, M) in enumerate(mats)
#         if size(M,1) != ops.dims[idx] || size(M,2) != ops.dims[idx] || isapprox(M' * M, I, atol=1e-8) == false
#             return nothing
#         end
#     end
#     return unsafe_member(ops, mats)
# end
# """
#     Converts encoded data into orthogonal matrices.
# """
# function transverse(ops::OrthogonalOps, data::Vector{Number} ) :: Vector{AbstractMatrix} 
#     offset = 0
#     mats = Vector{Matrix{eltype(data)}}(undef, length(ops.dims))
#     for (idx, n) in enumerate(ops.dims)
#         mats[idx] = reshape(data[offset+1:offset + n*n], (n,n))
#         offset += n*n
#     end
#     return mats
# end

# #------------------------------- Diagonal Operators -------------------------------------
# struct DiagonalOps <: TransverseOps 
#     frame :: Vector{Index}
#     engaged :: Vector{Bool}
# end
# function frame(Ω::DiagonalOps)::Vector{Index}
#     return Ω.frame
# end
# function engaged(Ω::DiagonalOps)::Vector{Bool}
#     return Ω.engaged
# end

# # Store only the lower triangle of each symmetric matrix
# function unsafe_member(ops::DiagonalOps, mats::Vector{AbstractMatrix}) :: Vector{Number}
#     v = Vector{eltype(mats[1])}(undef,sum(ops.dims))
#     offset = 0
#     for a in 1:length(ops.dims)
#         for i in 1:ops.dims[a]
#             v[offset + i] = mats[a][i,i]
#         end
#         offset += ops.dims[a]
#     end
#     return v
# end
# function member(ops::DiagonalOps, mats::Vector{AbstractMatrix}) :: Union{Vector{Number}, Nothing} 
#     for (idx, M) in enumerate(mats)
#         if size(M,1) != ops.dims[idx] || size(M,2) != ops.dims[idx] || !isdiagonal(M)
#             return nothing
#         end
#     end
#     return unsafe_member(ops, mats)
# end
# function transverse(ops::DiagonalOps, data::Vector{Number} ) :: Vector{AbstractMatrix}
#     mats = Vector{Diagonal}(undef, length(ops.dims))
#     for (idx, n) in enumerate(ops.dims)
#         mats[idx] = Diagonal(data[offset+1:offset+n])
#     end
#     return mats
# end


#------------------------------- User Defined Operators -------------------------------------


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

# end # module TransverseOperators