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

import ITensors
using ITensors: ITensor, Index, inds, setprime!, noprime!, store, norm, addtags

include("ITensorExtension.jl")

include("Chisels.jl")   

include("TensorSynthesis.jl")
include("TensorSynthesis3D.jl")
# using .TensorSynthesis
# Re-export from TensorSynthesis
# TBD: Rethink these functions and their names for final users

using PlotlyJS

include("TensorIO.jl")
# using .TensorIO
# Re-export from TensorIO
export normalizeTensor, asarray, sidebyside, loadTensor, save, plotTensor

include("TransverseOperators.jl")
# using .TransverseOperators
# Re-export from TransverseOperators
export TransverseOps, UniversalOps, engaged, frame, dim, transverse, member, unsafe_member #, SymmetricOps, DiagonalOps, InvertibleOps, OrthogonalOps

include("SylverLining.jl")
# using .SylverLining
# Re-export from SylverLining
export sylvesterLM

include("Derivations.jl")
# using .Derivations
# Re-export from Derivations
export DerivationMethod, der, den, stratify


end # module Dleto
