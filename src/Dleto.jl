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

module Dleto

# ============================================================================
# Imports
# ============================================================================

# Linear Algebra libraries
import LinearAlgebra
import ITensors

# Plotting libraries
import Plots
import PlotlyBase
import PlotlyKaleido

# ============================================================================
# Exported Functions and Types
# ============================================================================

include("DletoExports.jl")

# ============================================================================
# Includes
# ============================================================================

include("DletoBase.jl")

# Fundamental Dleto Structures
    # Chisels
    include("Chisels.jl")   
    # Operator
    include("Operators.jl")
    # Transverse Operators
    include("TransverseOperators.jl")
    # Derivation Methods
    include("Derivations.jl")
    # Densors
    include("Densors.jl")

# Implementations
    # Chisel Implementations
    include("chisels/ChiselImpls.jl")
    # Operator Implementations 
    include("ops/OperatorImpls.jl")
    # Transverse Operator Implementations
    include("ops/TransverseOpsIndependant.jl")
    include("ops/TransverseOpsSymmetries.jl")
    # Sylver Lining Derivation Method
    include("SylverLining/SylverLininig.jl")
    # [COMING SOON] QuickSylver
    include("solvers/NullSolvers.jl")

# Supporting Utilities
    # Tensor IO
    include("util/TensorIO.jl")
    # Randomization and Distance Functions
    include("util/Random.jl")
    # Tucker & HoSVD
    include("util/Nondegenerate.jl")
    # Tensor Synthesis
    include("util/TensorSynthesis.jl")
    include("util/TensorSynthesis3D.jl")

# ============================================================================
# Module Initialization
# ============================================================================

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
