
#
# Strata Dleto: Tensor Synthesis 3D 
#   tools to create 3D tensors for testing and demonstrations.
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

using ITensors


#-------------------------------
# generate a tensor supported on the a surface
"""
randSurfaceTensor(xes::Vector, yes::Vector, zes::Vector, cutoff::Number)::Array 

Generate a random tensor supported on a surface with equation x_i + y_j + z_k \approx 0 
"""
function randSurfaceTensor(
        xes::Vector, 
        yes::Vector, 
        zes::Vector, 
        cutoff::Number
    ):: ITensor
    return randTensorChisel([xes,yes,zes],cutoff,UniversalChisel(3))
end;

#-------------------------------
# generate a tensor supported on the a face curve
"""
randFaceCurveTensor(xes::Vector, yes::Vector, zes::Vector, cutoff::Number)::Array

Generate a random tensor supported on a surface with equation x_i \approx y_j, we need z_k to determine the size of the tensor 
"""
function randFaceCurveTensor(
        xes::Vector, 
        yes::Vector, 
        zes::Vector,
        cutoff::Number
    )::ITensor
    return randTensor([xes,yes,zes],cutoff,AdjointChisel(3,1,2))
end;

#-------------------------------
# generate a tensor supported on the a curve
"""
randFaceCurveTensor(xes::Vector, yes::Vector, zes::Vector, cutoff::Number)::Array

Generate a random tensor supported on a surface with equation x_i \approx y_j \approx z_k 
"""
function randCurveTensor(
        xes::Vector, 
        yes::Vector, 
        zes::Vector, 
        cutoff::Number
    )::ITensor
    return randTensor([xes,yes,zes],cutoff,CentroidChisel(3))
end

#-------------------------------
# test if a tensor is supported on a surface
"""
testSurfaceTensor(t::ITensor, xes::Vector, yes::Vector, zes::Vector)::Number

Measure how far a tensor is from being supported on a surface with equation x_i + y_j + z_k =0 
The result is a number between 0 and 1 with (almost) 0 bing that the tensor is supported on the surface 
The normalization is not perfect -- it is a good idea to call this on random tensor for a comparision
"""
normSurfaceTensor(
        t::ITensor, 
        xes::Vector, 
        yes::Vector, 
        zes::Vector
    )::Number = ITensorNormChisel(t,[xes,yes,zes],UniversalChisel(3));


#-------------------------------
# test if a tensor is supported on a face curve
"""
distFaceCurveTensor(t::ITensor, xes::Vector, yes::Vector, zes::Vector)::Number

Measure how far a tensor is from being supported on a face curve with equation x_i = y_j 
The result is a number between 0 and 1 with (almost) 0 bing that the tensor is supported on the face curve 
The normalization is not perfect -- it is a good idea to call this on random tensor for a comparision
"""
distFaceCurveTensor(
        t::ITensor, 
        xes::Vector, 
        yes::Vector, 
        zes::Vector
    )::Number= ITensorNormChisel(t,[xes,yes,zes],AdjointChisel(3,1,2));


#-------------------------------
# test if a tensor is supported on a curve
"""
distCurveTensor(t::ITensor, xes::Vector, yes::Vector, zes::Vector)::Number 

Measure how far a tensor is from being supported on a face curve with equation x_i = y_j = z_k 
The result is a number between 0 and 1 with (almost) 0 bing that the tensor is supported on the curve
The normalization is not perfect -- it is a good idea to call this on random tensor for a comparision
"""
distCurveTensor(
        t::ITensor, 
        xes::Vector, 
        yes::Vector, 
        zes::Vector
    )::Number= ITensorNormChisel(t,[xes,yes,zes],CentroidChisel(3));
