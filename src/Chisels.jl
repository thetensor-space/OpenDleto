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
    - `NormalizeChisel`: replace a chisel with equivalent one to improve numerical stability

"""

"""
     Create a universal chisel with selected engaged axes.
"""
function engaged(ch::Matrix, cutoff::Float64=1e-6)::Vector{Bool}
    valence = size(ch, 2)
    engaged = [ any(ch[:,a] .|> ( x -> abs(x) >cutoff )) for a in 1:valence ]
    return engaged
end;


"""
    Replace chisel with equivalent one to improve stability.
"""
function normalize_chisel(Ch::Matrix, cutoff::Float64=1e-6)::Matrix
    E=engaged(Ch)
    TE = TuckerChisel(E)
    svddecom = LinearAlgebra.svd(Ch*TE')
    num = sum( svddecom.S .|> ( x -> abs(x) > cutoff))
    RC =  svddecom.Vt[1:num,:]
    return RC * TE
end;
# MDK: I think it might be better to normalize using dimensions of the tensor, but I do not know what is the best way to do that 


"""
    Evaluate chisel on a vector, used to estimate the distance between the point and the surface.
"""
function __dist(Ch::Matrix, delta::Vector)::Number
    return Ch*delta .|> (x -> x*x) |> sum |> sqrt 
end;



#-------------------------------Chisel Constructors-------------------------------------

"""
    Create a universal chisel with selected engaged axes.
"""
function UniversalChisel(engaged::Vector{Bool})::Matrix
    polynomials = zeros(1, length(engaged) )
    for (i, is_engaged) in enumerate(engaged)
        if is_engaged
            polynomials[i] = 1
        end
    end
    # s = sum(polynomials .^ 2)
    # if s == 0
    #     return polynomials
    # end
    # norm = 1/sqrt(sum(polynomials .^ 2))
    # polynomials .*= norm
    return polynomials
end;

"""
    Create a universal chisel with all axes engaged.
"""
function UniversalChisel(valence::Integer)::Matrix  
    engaged = [ true for _ in 1:valence ]
    return UniversalChisel(engaged)
end;


"""
    Create a Tucker chisel with selected engaged axes.
"""
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

"""
    Create a Tucker chisel with all axes engaged.
"""
function TuckerChisel(valence::Integer)::Matrix  
    engaged = [ true for _ in 1:valence ]
    return TuckerChisel(engaged)
end;

"""
    Create a adjoint chisel with two axes engaged.
"""
function AdjointChisel(valence::Integer, left::Integer, right::Integer)::Matrix  
    polynomials = zeros(1, valence )
    polynomials[1, left] = 1.0
    polynomials[1, right] = -1.0
    return polynomials
end;

"""
    Create a Centroid chisel with selected engaged axes.
"""
function CentroidChisel(engaged::Vector{Bool})::Matrix  
    valence = length(engaged)
    is = [ i for i in 1:valence if engaged[i] ]
    e = length(is)
    polynomials = zeros( e*(e-1) ÷ 2, valence )
    row = 1    
    for a in is 
        for b in is
            if a < b
                polynomials[row, a] = 1.0
                polynomials[row, b] = -1.0
                row += 1
            end
        end
    end
    
    return polynomials
end;

"""
    Create a Centroid chisel with all axes engaged.
"""
function CentroidChisel(valence::Integer)::Matrix  
    engaged = [ true for _ in 1:valence ]
    return CentroidChisel(engaged)
end;


# end # module Chisels