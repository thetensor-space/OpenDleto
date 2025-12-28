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

using ITensors

export engaged, LinearChisel, Chisel, UniversalChisel, TuckerChisel, AdjointChisel, CentroidChisel 

function engaged(ch::Matrix)::Vector{Bool}
    valence = size(ch, 2)
    engaged = [ any(ch[:,a] .!= 0) for a in 1:valence ]
    return engaged
end;

#-------------------------------Chisel Type-------------------------------------

# """
#     LinearChisel{T}  
#     A linear chisel structure for tensor decomposition constraints.

# """
# struct LinearChisel
#     frame :: Vector{Index}
#     engaged :: Vector{Integer}
#     columns :: Vector{ITensor}
# end;

# """
#     Print a linear chisel.
# """
# function Base.show(io::IO, ::MIME"text/plain", chisel::LinearChisel)
#     println(io, "Linear chisel polynomials:")
#     mat = zeros(eltype(chisel.columns[1]), dim(chisel.columns[1]), length(chisel.frame))
#     for (idx, a) in enumerate(chisel.engaged)
#         mat[:, a] = store(chisel.columns[idx])
#     end
#     println(io, mat)
# end;

#-------------------------------Chisel Constructors-------------------------------------

# """
#     Create a linear chisel from a matrix of polynomials and a frame of indices.

#     - `polynomials`: A matrix whose columns define the chisel polynomials.
#     - `frame`: A vector of `Index` objects defining the tensor frame.

#     Returns a `LinearChisel` object.
# """
# function Chisel(polynomials::AbstractMatrix, frame::Vector{Index{Int64}})::Matrix
#     @assert size(polynomials, 2) == length(frame) "Chisel polynomial valence must match frame length."
#     ch_axis = Index(size(polynomials, 1), "chisel_axis")
#     engaged = [ a for a in 1:size(polynomials, 2) if any(polynomials[:,a] .!= 0) ]
#     columns = [ ITensor(polynomials[:, a], ch_axis ) for a in engaged ]
#     return LinearChisel(frame, engaged, columns)
# end;

"""
    Create a universal chisel with all axes engaged.
"""
function UniversalChisel(engaged::Vector{Bool})::Matrix
    polynomials = zeros(1, length(engaged) )
    for (i, is_engaged) in enumerate(engaged)
        if is_engaged
            polynomials[i] = 1
        end
    end
    norm = 1/sqrt(sum(polynomials .^ 2))
    polynomials .*= norm
    return polynomials
end;

"""
     Create a universal chisel with selected engaged axes.
"""
function UniversalChisel(valence::Integer)::Matrix  
    engaged = [ true for _ in 1:valence ]
    return UniversalChisel(engaged)
end;

function TuckerChisel(engaged::Vector{Bool})::Matrix  
    valence = length(engaged)
    e = sum(engaged)
    polynomials = zeros( e, valence )
    row = 1
    for a in 1:valence
        if engaged[a]
            polynomials[row,a] = 1.0
            row += 1
        end
    end
    return polynomials
end
function TuckerChisel(valence::Integer)::Matrix  
    engaged = [ true for _ in 1:valence ]
    return TuckerChisel(engaged)
end;

function AdjointChisel(valence::Integer, left::Integer, right::Integer)::Matrix  
    polynomials = zeros(1, valence )
    polynomials[1, left] = 1.0
    polynomials[1, right] = -1.0
    return polynomials
end;

function CentroidChisel(engaged::Vector{Bool})::Matrix  
    valence = length(engaged)
    is = [ i for i in 1:valence if engaged[i] ]
    e = length(is)
    polynomials = zeros( e*(e-1) ÷ 2, valence )
    for a in is 
        for b in is
            if a < b
                row = findfirst( x -> x == 0.0, polynomials[:,a] ) # next available row
                polynomials[row, a] = 1.0
                polynomials[row, b] = -1.0
            end
        end
    end
    
    return polynomials
end;



end # module Chisels