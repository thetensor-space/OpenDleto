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

# ============================================================================
# Exports
# ============================================================================

# Random.jl
export randomize_tensor, nondeg

# DletoUtil.jl
export ⊕

# Chisels.jl
export engaged, UniversalChisel, TuckerChisel, AdjointChisel, CentroidChisel
export normalize_chisel

# LocalOperatorsAbstract.jl
export Operator                             # abstract type
export embed, unsafe_embed                  # Embed
export coordinates, unsafe_coordinates      # Membership
export transposeEmbed, unsafe_transposeEmbed  
export localDim, containScalars

# LocalOperatorsImplementations.jl
export UniversalOp, DiagonalOp, SymmetricOp, AntiSymmetricOp, ScalarOp, EmptyOp

# GlobalOperatorsAbstract.jl
export TransverseOps
export embedMatrices, unsafe_embedMatrices, embedITensors, unsafe_embedITensors
export globalDim, axisDims, valency, frames, framesTemporary, reduceByEngaged

# GlobalOperatorsIndependant.jl
export IndTransverseOps

# GlobalOperatorsSymmetries.jl
export TransverseOpsSymmetries

# DerivationMethodAbstract.jl
export DerivationMethod, der, den

# DerivationMethodSylverLininig.jl
export sylvesterLM, SylverLiningMethod

# Stratify.jl
export stratify

# TensorIO.jl
export normalize_tensor, side_by_side, load_tensor, save, plot_tensor

export rand_den
export randTensor, randSurfaceTensor, randFaceCurveTensor, randCurveTensor
export distSurfaceTensor, distFaceCurveTensor, distCurveTensor


# ============================================================================
# Includes
# ============================================================================

include("DletoUtil.jl")

include("Random.jl")

include("Chisels.jl")   

include("TensorSynthesis.jl")
include("TensorSynthesis3D.jl")
# using .TensorSynthesis
# Re-export from TensorSynthesis
# TBD: Rethink these functions and their names for final users

include("Operators.jl")
include("OperatorsImpl.jl")

include("TransverseOperators.jl")
include("TransverseOpsIndependant.jl")
include("TransverseOpsSymmetries.jl")

include("DerivationMethodAbstract.jl")

include("DerivationMethodSylverLininig.jl")

include("Stratify.jl")

# using PlotlyJS
import Plots
import PlotlyBase
import PlotlyKaleido

# using Plots
# plotly()

include("TensorIO.jl")

# Initialize Plotly backend at runtime only
function __init__()
    # Suppress WebIO warnings
    ENV["WEBIO_WARN"] = "false"
    
    # Only set backend if we're not precompiling
    if ccall(:jl_generating_output, Cint, ()) != 1
        try
            Plots.plotlyjs()

        catch e
            @warn "Failed to set Plotly backend" exception=e
        end
    end
end
end # module Dleto
