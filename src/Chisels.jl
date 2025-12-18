#
# Strata Dleto: Chisels
#   Creation and adaptation of chisels for tensor decomposition.
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
    Chisels

    Data types and constructors for the constraint equations of derivations
    for specified chisels and operators.  The selection of operators is done
    through the `TransverseOperators.jl` module.
        
    This module exports:
    - `UniversalChisel`: fully engaged universal operators
    - `TuckerChisel`: tucker-type operators
    - `AdjointChisel`: adjoint-type operators
    - `CentroidChisel`: centroid-type operators

"""
module Chisels

using ..TensorSpaces: Engagement, Primal, Dual, Ambidextrous, Disengaged
# using ..TransverseOperators: TransverseOps, UniversalOps, SymmetricOps, 
#     contains, transverse

# TensorSpace.jl is included by the main Dleto module
export UniversalChisel, TuckerChisel, AdjointChisel, CentroidChisel #, SymmetricChisel #, constraint, constraints

#-------------------------------Chisel Type-------------------------------------

"""
    LinearChisel{T}  
    A linear chisel structure for tensor decomposition constraints.

    Access chisels through methods below.
"""
struct LinearChisel{T<:Number}
    category :: Array{Engagement}
    polynomials :: Array{T,2}
    # operators :: TransverseOps
end;

"""
    Print a linear chisel.
"""
function Base.show(io::IO, ::MIME"text/plain", chisel::LinearChisel)
    print(io, "Linear chisel category: ")
    Base.show(io, MIME("text/plain"), collect(chisel.category))
    println(io)
    println(io, chisel.polynomials)
    print(io, chisel.operators.description)
end;

#---------------Universal Chisel-----------------------------------------------
# """
#     Create a universal operators with selected engaged axes.
# """
# function UniversalOps(category::Array{Engagement}, poly::Array{T,2}) where T
#     engaged = findall(e -> e != Disengaged, category)
#     uni_dims(a,t) = a in engaged ? size(t,a)*size(t,a) : 0
#     function uni_insert!(M, row_offset, col_offset, coeff, tube, i_a)
#         d = size(tube,1)
#         col_start = col_offset + 1+(i_a-1) * d
#         insert_data = coeff .* tube'           
#         col_end = col_start + d -1
#         view(M, (row_offset+1):(row_offset+size(coeff,1)), col_start:col_end) .= insert_data
#     end
#     uni_matrix(u::Vector, n:: Integer, offset::Integer )::AbstractMatrix = reshape( u[offset+1:offset+n*n],  (n,n) )
#     return TransverseOperators(uni_dims, uni_insert!, uni_matrix, "Universal Operators")
# end;

"""
    Create a universal chisel with selected engaged axes.
"""
function UniversalChisel(category::Array{Engagement})
    mat = zeros(Number, 1, length(category))
    for (idx, a) in enumerate(category)
        if a != Disengaged
            mat[1,idx] = 1
        end
    end
    # Each engaged operator is a square matrix

    return LinearChisel(category, mat ) #, UniversalOps(category, mat))
end;


"""
    Create a universal chisel with all specified primal and dual axes.
    Note that if an axis is specified in both primal and dual lists,
    then it receives the primal designation. The primal and dual values 
    must be in the range 1 to valence.
    
    If called with only valence, all axes are set to Primal (fully engaged).
"""
function UniversalChisel(valence::Integer, primals::Vector{<:Integer}=Int[], 
    duals::Vector{<:Integer}=Int[])
    # If no primals or duals specified, engage all as primal
    if isempty(primals) && isempty(duals)
        cat = fill(Primal, valence)
    else
        cat = fill(Disengaged, valence)
        for a in duals
            cat[a] = Dual
        end
        for a in primals
            cat[a] = Primal
        end
    end
    return UniversalChisel(cat)
end;

# ch = UniversalChisel(3)
# ch = UniversalChisel(5, [1,3])


#-------------------------------Tucker Chisel----------------------------------
"""
    Create a Tucker chisel with selected engaged axes.
"""
function TuckerChisel(valence::Integer, engaged::Vector{<:Integer})
    cat = fill(Disengaged, valence)
    for a in engaged
        cat[a] = Primal
    end
    mat = zeros(Number, length(engaged), valence)
    for (idx, a) in enumerate(engaged)
        mat[idx, a] = 1
    end
    return LinearChisel(cat, mat ) #, UniversalOps(cat, mat))
end;

"""
    Create a Tucker chisel with all axes engaged.  
"""
function TuckerChisel(valence::Integer)
    return TuckerChisel(valence, collect(1:valence))
end;

#-------------------------------Centroid Chisel--------------------------------
"""
    Create a centroid chisel with specified engaged axes.
"""
function CentroidChisel(valence::Integer, engaged::Vector{<:Integer})
    cat = fill(Disengaged, valence)
    for a in engaged
        cat[a] = Ambidextrous
    end
    mat = zeros(Number, length(engaged)*(length(engaged)-1) ÷ 2, valence)
    row = 1
    for primal in 1:length(engaged)
        for dual in (primal+1):length(engaged)
            mat[row,engaged[primal]] += 1
            mat[row,engaged[dual]] += -1
            row += 1
        end
    end
    return LinearChisel(cat, mat, UniversalOps(cat, mat))
end;

function CentroidChisel(valence::Integer)
    return CentroidChisel(valence, collect(1:valence))
end;

#-------------------------------Adjoint Chisel---------------------------------
"""
    Create an adjoint chisel with specified primal and dual axes.
"""
function AdjointChisel(valence::Integer, primal::Integer, dual::Integer)
    cat = fill(Disengaged, valence)
    cat[dual] = Dual
    cat[primal] = Primal
    mat = zeros(Number, 1, valence)
    mat[1,primal] += 1
    mat[1,dual] += -1
    return LinearChisel(cat, mat ) #, UniversalOps(cat, mat))
end;


#-------------------------------Orthogonal Chisel------------------------------

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
# function SymmetricOps(category::Array{Engagement}, poly::Array{T,2}) where T
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
#             M[j,i] = u[offset + k]  # Symmetric entry
#         end
#         return LinearAlgebra.Matrix(M)
#     end;

#     return TransverseOperators(sym_dims, sym_insert!, sym_matrix, "Symmetric Operators")
# end;

# """
#     Create a symmetric chisel with selected engaged axes.
# """
# function SymmetricChisel(category::Array{Engagement})
#     mat = zeros(Number, 1, length(category))
#     for (idx, a) in enumerate(category)
#         if a != Disengaged
#             mat[1,idx] = 1
#         end
#     end
#     # Each engaged operator is a square matrix

#     return LinearChisel(category, mat) #, SymmetricOps(category, mat))
# end;

# """
#     Create a universal chisel with all specified primal and dual axes.
#     Note that if an axis is specified in both primal and dual lists,
#     then it recieves teh primal designation. The primal and dual values 
#     must be in the range 1 to valence.
# """
# function SymmetricChisel(valence::Integer, primals::Vector{<:Integer}, duals::Vector{<:Integer})
#     cat = fill(Disengaged, valence)
#     for a in duals
#         cat[a] = Dual
#     end
#     for a in primals
#         cat[a] = Primal
#     end
#     return SymmetricChisel(cat)
# end;

# """
#     Create a universal chisel with all axes engaged.
# """
# function SymmetricChisel(valence::Integer, engaged::Vector{<:Integer})
#     cat = fill(Disengaged, valence)
#     for a in engaged
#         cat[a] = Primal
#     end
#     return SymmetricChisel(cat)
# end;

# """
#     Create a universal chisel with all axes engaged.
# """
# function SymmetricChisel(valence::Integer)
#     cat = fill(Primal, valence)
#     return SymmetricChisel(cat)
# end;


end # module Chisels