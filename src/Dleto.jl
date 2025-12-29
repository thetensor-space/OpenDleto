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
using ITensors: ITensor, Index, inds, setprime!, noprime!, store, norm

include("Chisels.jl")   
# using .Chisels: engaged, UniversalChisel, TuckerChisel, AdjointChisel, CentroidChisel 
# Re-export from Chisels
export engaged, UniversalChisel, TuckerChisel, AdjointChisel, CentroidChisel 

include("TensorSynthesis.jl")
# using .TensorSynthesis
# Re-export from TensorSynthesis
# TBD: Rethink these functions and their names for final users
export randomOrthogonalMatrix, spin, randomize
export randTensor, randSurfaceTensor, randFaceCurveTensor, randCurveTensor
export distSurfaceTensor, distFaceCurveTensor, distCurveTensor

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

# -- Direct action by a vector of ITensors -----

function Base.:*(Γ::ITensor, X::Vector{ITensor})  
    Σ = Γ  
    for x in X
        # setprime!(Σ, 1; plev=inds(Γ)[i])
        Σ = Σ * x
        # noprime!(Σ; plev=inds(Γ)[i])
    end
    return Σ
end

# --- Wrappers to promote AbstractArray to ITensor -----
function Base.:*(Γ ::AbstractArray, X::Vector{ITensor}) 
    @assert length(X) == ndims(Γ) "Ambiguous frame matching: length of list of matrices must match array axes"
    frame = [ inds(x)[1] for x in X ]
    iΓ = typeof(Γ) <: Array ? ITensor(Γ, frame...) : ITensor( Array(Γ), frame...)
    return iΓ * X
end

function Base.:*(Γ::ITensor, X::Vector{AbstractMatrix}) 
    @assert length(X) == ndims(Γ) "Ambiguous frame matching: length of list of matrices must match ITensor axes"
    fr = inds(Γ)
    iX = [ ITensor( Array(X[i]), fr[i], prime(fr[i]) ) for i in 1:length(X) ]
    return Γ * iX
end

# Make actions Ambidextrous
function Base.:*(X::Vector{ITensor}, Γ::AbstractArray ) 
    return Γ * X
end
function Base.:*(X::Vector{ITensor}, Γ::ITensor ) 
    return Γ * X
end
function Base.:*(X::Vector{AbstractMatrix}, Γ::ITensor ) 
    return Γ * X
end

end # module Dleto
