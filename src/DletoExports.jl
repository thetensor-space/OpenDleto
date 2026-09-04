#
# Strata Dleto: Dleto.jl
#   Main module for Dleto package.
#
# Copyright 2022-2026 Peter A. Brooksbank, Martin D. Kassabov, James B. Wilson
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

# ============================================================================
# Exports
# ============================================================================

# Random.jl
export randomize_tensor 

# NonDegenerate.jl
export nondeg

# DletoUtil.jl
export ⊕
# what is this? is this tensor product of tensors? why not \otimes?

# Chisels.jl
export engaged, UniversalChisel, TuckerChisel, AdjointChisel, CentroidChisel
export normalize_chisel

# ChiselImpls.jl
export Chisel, applyDerivation

# Operators.jl
export Operator                             # abstract type
export embed, unsafe_embed                  # Embed
export coordinates, unsafe_coordinates      # Membership
export transposeEmbed, unsafe_transposeEmbed  
export dualize, unsafe_dualize
export localDim, containScalars

# OperatorsImpls.jl
export UniversalOp, DiagonalOp, SymmetricOp, AntiSymmetricOp, ScalarOp, EmptyOp

# Transverse.jl
export TransverseOps
export embedMatrices, unsafe_embedMatrices
export embedITensors, unsafe_embedITensors
export embedITensorsSwapped, unsafe_embedITensorsSwapped
export globalDim, axisDims, valency, frames, framesTemporary, reduceByEngaged

# GlobalOperatorsIndependant.jl
export IndTransverseOps

# GlobalOperatorsSymmetries.jl
export TransverseOpsSymmetries

export NullSolver
export solve_nullspace, available_solvers, register_solver!
export AutoSolver, SVDSolver, LUSolver, ShiftInvertSolver
export PROGRESS_TAGS, progress_spec

# Derivations.jl
# `der` is the Z-set: a basis of the P-derivations of a tensor.
# `den` is still an abstract placeholder (it asserts false for every
# method); the T-set work implements it against that name.
export DerivationMethod, get_derivation_method
export der, derReduced, derTrOps, derTrOpsReduced, den

# DerivationMethodSylverLininig.jl
export sylvesterLM, SylverLiningMethod
export FastDer3ValentMethod, QuickSylverMethod, QuickDerMethod, AutoDerMethod

# Densors.jl
export stratify, denLM

# TensorIO.jl
export normalize_tensor, compare, side_by_side, load_tensor, save, plot_tensor
export set_compare_layout, get_compare_layout

# TensorSynthesis.jl
export rand_den, randTensorChisel
export ITensorNorm, ITensorNormChisel

# TensorSynthesis3D.jl
export randSurfaceTensor, randFaceCurveTensor, randCurveTensor
export distSurfaceTensor, distFaceCurveTensor, distCurveTensor
