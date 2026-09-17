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
    # StreamingCore: QuickDer's sufficient statistics accumulated one frame at
    # a time, for a tensor whose stream axis is never complete.  After
    # QuickDerN, whose sketch and pair-tensor contractions it calls verbatim.
    include("solvers/StreamingCore.jl")
    # AutoDer: QuickDer when the setting allows it, SylverLining otherwise
    include("solvers/AutoDer.jl")
    # QuickSylver: double-restriction solve-and-lift for adjoint-type chisels
    include("solvers/QuickSylver.jl")
    include("solvers/SolverProgress.jl")
    include("solvers/NullSolvers.jl")
    # DerivationReport: what a derivation solve decided and on what evidence.
    # AFTER NullSolvers, because it carries a `NullVerdict` as a field and a
    # struct's field types are evaluated at definition time.  The derivation
    # methods above only NAME it inside function bodies, which is why they can
    # be included first.
    include("solvers/DerivationReport.jl")
    # Dense symmetric normal equations for the cubic all-symmetric setting.
    include("solvers/SymmetricGram.jl")

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

# Optional ergonomic wrapper API (`using Dleto.TensorSpace`)
include("TensorSpace.jl")
const ts = TensorSpace.ts

_ensure_tensor_element(Γ::TensorSpace.TensorElement) = Γ
_ensure_tensor_element(Γ::ITensors.ITensor) = TensorSpace.tensor(Γ)

# Notebook ergonomics: allow orthogonality checks like `X' * X ≈ I` when
# `X' * X` is represented as a matrix-shaped ITensor.
function Base.isapprox(A::ITensors.ITensor, J::LinearAlgebra.UniformScaling; kwargs...)
    as = inds(A)
    if length(as) != 2
        return false
    end
    i, j = as
    if ITensors.dim(i) != ITensors.dim(j)
        return false
    end
    return isapprox(A, J.λ * ITensors.delta(i, j); kwargs...)
end

Base.isapprox(J::LinearAlgebra.UniformScaling, A::ITensors.ITensor; kwargs...) =
    isapprox(A, J; kwargs...)

# Preserve wrapper semantics when tensor-space values are passed through the
# package-level API.  The underlying ITensor implementation still runs on the
# bare tensor, but the notebook keeps working with wrapped `TensorElement`s.
function randomize_tensor(Γ::TensorSpace.TensorElement; type::Symbol=:invertible)
    out = randomize_tensor(TensorSpace.unwrap(Γ); type=type)
    return (; Δ=_ensure_tensor_element(out.Δ), Xs=out.Xs)
end

function stratify(Γ::TensorSpace.TensorElement; kwargs...)
    out = stratify(TensorSpace.unwrap(Γ); kwargs...)
    return (; Σ=_ensure_tensor_element(out.Σ), Xs=out.Xs)
end

function stratify(Ω::TransverseOps, ch::AbstractMatrix, Γ::TensorSpace.TensorElement; kwargs...)
    out = stratify(Ω, ch, TensorSpace.unwrap(Γ); kwargs...)
    return (; Σ=_ensure_tensor_element(out.Σ), Xs=out.Xs)
end

function nondeg(Γ::TensorSpace.TensorElement; kwargs...)
    out = nondeg(TensorSpace.unwrap(Γ); kwargs...)
    return (; Δ=_ensure_tensor_element(out.Δ), Es=out.Es)
end

UniversalOps(Γ::TensorSpace.TensorElement) = UniversalOps(TensorSpace.unwrap(Γ))
DiagonalOps(Γ::TensorSpace.TensorElement) = DiagonalOps(TensorSpace.unwrap(Γ))
SymmetricOps(Γ::TensorSpace.TensorElement) = SymmetricOps(TensorSpace.unwrap(Γ))
AntiSymmetricOps(Γ::TensorSpace.TensorElement) = AntiSymmetricOps(TensorSpace.unwrap(Γ))
ScalarOps(Γ::TensorSpace.TensorElement) = ScalarOps(TensorSpace.unwrap(Γ))

der(Γ::TensorSpace.TensorElement; kwargs...) = der(TensorSpace.unwrap(Γ); kwargs...)
der(method::Symbol, Γ::TensorSpace.TensorElement; kwargs...) = der(method, TensorSpace.unwrap(Γ); kwargs...)
der(ch::AbstractMatrix, Γ::TensorSpace.TensorElement; kwargs...) = der(ch, TensorSpace.unwrap(Γ); kwargs...)
der(method::Symbol, ch::AbstractMatrix, Γ::TensorSpace.TensorElement; kwargs...) =
    der(method, ch, TensorSpace.unwrap(Γ); kwargs...)
der(method::Symbol, Ω::TransverseOps, ch::AbstractMatrix, Γ::TensorSpace.TensorElement; kwargs...) =
    der(method, Ω, ch, TensorSpace.unwrap(Γ); kwargs...)
der(Ω::TransverseOps, ch::AbstractMatrix, Γ::TensorSpace.TensorElement; kwargs...) =
    der(Ω, ch, TensorSpace.unwrap(Γ); kwargs...)
den(Γ::TensorSpace.TensorElement; kwargs...) = den(TensorSpace.unwrap(Γ); kwargs...)
der_residual(Γ::TensorSpace.TensorElement, D::Vector{ITensors.ITensor}, chisel; kwargs...) =
    der_residual(TensorSpace.unwrap(Γ), D, chisel; kwargs...)

# Wrapper-aware scalar utilities keep notebook workflows on TensorElement.
ITensorNorm(Γ::TensorSpace.TensorElement, deltas::Vector{<:AbstractVector{<:Number}}, dist::Function)::Number =
    ITensorNorm(TensorSpace.unwrap(Γ), deltas, dist)

ITensorNormChisel(Γ::TensorSpace.TensorElement, deltas::Vector{<:AbstractVector{<:Number}}, ch::Matrix)::Number =
    ITensorNormChisel(TensorSpace.unwrap(Γ), deltas, ch)

distSurfaceTensor(t::TensorSpace.TensorElement, xes::Vector, yes::Vector, zes::Vector)::Number =
    distSurfaceTensor(TensorSpace.unwrap(t), xes, yes, zes)

distFaceCurveTensor(t::TensorSpace.TensorElement, xes::Vector, yes::Vector, zes::Vector)::Number =
    distFaceCurveTensor(TensorSpace.unwrap(t), xes, yes, zes)

distCurveTensor(t::TensorSpace.TensorElement, xes::Vector, yes::Vector, zes::Vector)::Number =
    distCurveTensor(TensorSpace.unwrap(t), xes, yes, zes)

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
