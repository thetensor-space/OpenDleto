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
# KrylovKit and IterativeSolvers are hard dependencies whose solver extensions
# (DletoKrylovKitExt, DletoIterativeSolversExt) only activate once the trigger
# package is loaded.  Loading them here means :KrylovSolver, :LanczosSolver,
# :CGSolver and :LSMRSolver are always registered, so `AutoSolver` sees the
# matrix-free solvers it was tuned with instead of falling back to a dense SVD.
# Arpack stays a weak dependency: `using Arpack` adds :ArpackSolver.
import KrylovKit
import IterativeSolvers

# # Plotting libraries
# import Plots
# import PlotlyBase
# import PlotlyKaleido

# ============================================================================
# Exported Functions and Types
# ============================================================================

include("DletoExports.jl")

# ============================================================================
# Includes
# ============================================================================

include("DletoBase.jl")

    # The floating-point policy: one place where the element type decides what
    # "zero" means.  First, because every solver and every `der`/`den` entry
    # point takes its default tolerance from it.
    include("solvers/Precision.jl")

# Fundamental Dleto Structures
    # Chisels
    include("Chisels.jl")
    # The Chisel type itself is fundamental -- Densors.jl annotates against it,
    # and annotations are evaluated at definition time -- so it is included
    # here rather than down in the implementations section.
    include("chisels/ChiselImpls.jl")
    # Operator
    include("Operators.jl")
    # Transverse Operators
    include("TransverseOperators.jl")
    # Derivation Methods
    include("Derivations.jl")
    # Densors
    include("Densors.jl")

# Implementations
    # Operator Implementations
    include("ops/OperatorImpls.jl")
    # Transverse Operator Implementations
    include("ops/TransverseOpsIndependant.jl")
    include("ops/TransverseOpsSymmetries.jl")
    # Sylver Lining Derivation Method
    include("SylverLining/SylverLining.jl")
    # Fast derivation strategy (3-valent, universal setup) -- the reference oracle
    include("solvers/FastDer3Valent.jl")
    # QuickDer: the same solve-and-lift generalised to any valence
    include("solvers/QuickDerN.jl")
    # AutoDer: QuickDer when the setting allows it, SylverLining otherwise
    include("solvers/AutoDer.jl")
    # QuickSylver: double-restriction solve-and-lift for adjoint-type chisels
    include("solvers/QuickSylver.jl")
    include("solvers/SolverProgress.jl")
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
    
    # # Only set backend if we're not precompiling
    # if ccall(:jl_generating_output, Cint, ()) != 1
    #     try
    #         Plots.plotlyjs()
    #     catch e
    #         @warn "Failed to set Plotly backend" exception=e
    #     end
    # end
end

end # module Dleto
