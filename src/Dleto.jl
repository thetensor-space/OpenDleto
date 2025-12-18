#
# Strata Dleto: Dleto.jl
#   Main module for Dleto package.
#
# Copyright 2022-2025 Peter A. Brooksbank, Martin D. Kassabov, James B. Wilson
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"), 
# to deal in the Software without restriction, including without limitation the 
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.
# 

module Dleto

import LinearAlgebra
import Random
import SparseArrays
import Statistics
import ProgressMeter
import TensorOperations
import CUDA
import cuTENSOR
import LinearMaps
import Arpack
import IterativeSolvers
import PlotlyJS

# Include/use component modules
include("TensorSpaces.jl")
using .TensorSpaces

include("TransverseOperators.jl")
using .TransverseOperators

include("Chisels.jl")
using .Chisels

include("Sylvester.jl")
using .SylvesterSolvers

include("TensorSynthesis.jl")
using .TensorSynthesis

include("SylvesterBB.jl")
using .SylvesterBB

include("TensorHypergraphs.jl")
using .TensorHypergraphs

include("TensorIO.jl")
using .TensorIO

# Export main functions

# Re-export from TensorSpaces
export act, spin, randomize, changeTensor, Engagement, Primal, Dual, Ambidextrous, Disengaged

export TransverseOperators, UniversalOps, SymmetricOps

# Re-export from Chisels
export UniversalChisel, TuckerChisel, AdjointChisel, CentroidChisel, SymmetricChisel,constraint, constraints

# Re-export from SylvesterSolvers
export der, sculpt, Derivation

# Re-export from TensorIO
export normalizeTensor, sidebyside, loadTensorFromFile, saveTensorToFile, plotTensor

export sylverlin3, sylverlin1, toMat, solveeig, solveit, cusylverlin3

# Re-export from TensorSynthesis
export randTensor, randSurfaceTensor, randFaceCurveTensor, randCurveTensor
export testSurfaceTensor, testFaceCurveTensor, testCurveTensor

function stratify(t::AbstractArray)
    ders = der(t)
    return sculpt(t, ders, collect(1:ndims(t)))
end

function orthoStratify(t::AbstractArray)
    ders = der(t)
    return sculpt(t, ders, collect(ndims(t):ndims(t)))
end

end # module Dleto
