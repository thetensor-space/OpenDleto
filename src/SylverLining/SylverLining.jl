
#
# Strata Dleto: Derivation Method SylverLining
#   Algorithms for solving Sylvester equations arising in chiseling.
# -----------------------------------------------------------------------------
# Copyright 2022-2026 Peter A. Brooksbank, Martin D. Kassabov, James B. Wilson
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
# -----------------------------------------------------------------------------

import LinearMaps
using ITensors



"""
    Sylver Lininig Derivation Method

    Derivation Method implemented via linear maps. 
"""
struct SylverLiningMethod <: DerivationMethod 
    solver::Symbol
end;
# The default is `:AutoSolver`, not `:SVDSolver`: it densifies when that is
# cheap (which it is for this map -- 9MB at n = 19) and stays matrix-free when
# it is not, instead of calling `Matrix` unconditionally.
SylverLiningMethod(; solver::Symbol=:AutoSolver)= SylverLiningMethod(solver);


function derTrOpsReduced(method::SylverLiningMethod,
    Ω::TransverseOps, 
    P::AbstractMatrix, 
    Γ::ITensor;
    tol::Real=1e-6,
    nd=-1,  # Don't type as integer to allow Inf 
    progress=false,
    kwargs...,
    ) ::Tuple{TransverseOps, LinearMaps.LinearMap, AbstractMatrix{<: Number} }
    Γ_frame = inds(Γ)
    val = ndims(Γ)
    @assert Γ_frame == frames(Ω) "Incompatable Indexes"
    @assert val == size(P, 2) "Incompatable Chisel"
    
    T = eltype(Γ)
    # Compute reduced operators (matching what sylvesterLM does internally)
    eng = engaged(P)
    (Ω_reduced,expand_map) = reduceByEngaged(Ω, eng, T)
    # Chisels default to Float64; do not let that promote a Float32 tensor.
    P_eng = Matrix{T}(P[:,eng])

    # if globalDim(reducedΩ) < 10000
    # MDK, we need to reduce the chisel and pass the reduced chisel to the helper function
    sylvester, ester_map = sylvesterLM(Ω_reduced, P_eng, Γ)
    
    # All of the null-solver policy -- when densifying is cheap enough, how
    # many vectors to ask an iterative method for, how to grow that request
    # until the null space is bracketed, which tolerance decides "null" --
    # lives in `solve_nullspace` (src/solvers/NullSolvers.jl) and is tuned
    # there once.  This function used to carry its own copy of all of it: a
    # hand-rolled `globalDim < 1000` dense gate, a dense `eigen` branch that
    # bypassed the solver interface entirely, `nv = globalDim(Ω)` asking an
    # iterative solver for the whole spectrum, and its own filter and
    # truncation.  `den` carried a second, slightly different copy.  Tuning
    # either one left the other stale.
    #
    # An empty result is a mathematical fact, not a solver failure: it says the
    # reduced derivation space is trivial, i.e. Γ conforms to no sparsity
    # pattern for this chisel.  A Tucker chisel on a generic tensor is the
    # standard example -- it forces each engaged D_a into the a-th radical,
    # which is zero, and its only scalar derivation lives on the disengaged
    # axis that the engagement reduction drops.  This used to assert
    # "Not enough eigenvalues computed; increase `tol` parameter", which
    # misreported the fact as a convergence problem.  Callers that need a
    # derivation (`stratify`) report it themselves.
    # Hand over the RECTANGULAR map, not the squared one.
    #
    # `sylvesterLM` returns both: `sylvester = sylve∘ester` is the Gram
    # operator `AᵗA`, and `ester_map` is `A` itself, with `sylve` as its
    # genuine adjoint.  A derivation is exactly an `X` with `ester(X) = 0`, so
    # `null(ester_map) = ` the derivation space -- the same null space, without
    # squaring anything.
    #
    # This was the derivation half of the deviation from Algorithm 2 of
    # null_patterns.pdf ("take the SVD of N"; the code eigendecomposed `NᵗN`).
    # Squaring costs half the available precision and, worse, silently
    # square-roots the tolerance: filtering `λ = σ²` at `tol·‖AᵗA‖` admits
    # every direction with `σ/σ_max ≤ √tol`.  Asking for 1e-6 gave 1e-3, which
    # on real video boxes meant ~130 reported "derivations" per box at a
    # residual of 2e-3.
    #
    # `solve_nullspace` squares it as a composition of maps for the solvers
    # that need eigenvalues, so nothing here has to know which those are.
    (λ, vecs) = solve_nullspace(ester_map, method.solver; tol=tol, nd=nd,
                                progress=progress, label="der")

    coords = size(vecs, 2) == 0 ? zeros(T, globalDim(Ω_reduced), 0) : Matrix{T}(vecs)
    return (Ω_reduced, expand_map, coords)
end



"""
    sylvesterLM(P::Matrix, T::ITensor)

    Constructs LinearMaps for the derivation and densor maps associated to the given chisel `ch` and tensor `T`.

    - `P`: A matrix whose columns define the chisel polynomials.
    - `T`: The input tensor.

    Returns a tuple `(derdensor_map, densormap)` where:
    - `derdensor_map`: the composed derivation-densor `LinearMap` (a real symmetric operator).
    - `densormap`: the densor operator with transpose---the derivation operator---included.
"""
function sylvesterLM(Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor) #::Tuple{LinearMaps.LinearMap, LinearMaps.LinearMap}
#temporary the function retunrs also the naked functions in addition to the linear maps
#this is done for only for testing but needs to be fixed during the merge.
#I am gettign stupid error is ch is and empty matrix!!!
#both Ω and Ch are assumed to be reduced!

    Γ_frame = inds(Γ)
    val = ndims(Γ)
    engsize = valency(Ω)
    @assert engsize == size(P, 2) "Incompatable Chisel"
    T = eltype(Γ)
    P_typed = Matrix{T}(P)
    ch_axis = Index(size(P, 1), "chisel")
    eng_axis = Index(engsize, "engaged")
    Cs = [ ITensor(P_typed[:,a], ch_axis) for a in 1:engsize ]

    Γ_frame_ch = (ch_axis, inds(Γ)...)

    Ωframe=frames(Ω)
    ΩframeTemp=framesTemporary(Ω)
    Γs_relabled =  [ replaceind(Γ, Ωframe[a], ΩframeTemp[a]) for a in 1:engsize]

    # Compute sizes for LinearMap
    densor_dim = prod([ITensors.dim(f) for f in Γ_frame_ch ])
    op_dim = globalDim(Ω)

    # Takes a vectorized representation of derivations
    # Returns a vectorized representation of the tensor
    function ester(Xvec)
        Xs = unsafe_embedITensorsSwapped(Ω, Xvec)
        Σ = [ Cs[a]*Xs[a]*Γs_relabled[a] for a in 1:engsize ] |> sum 
        return vec(Array(Σ, Γ_frame_ch...))
    end
    
    # sylv: takes a vectorized tensor, returns a vector of matrices (one per axis)
    function sylve(y)
        Σ = ITensor(y, Γ_frame_ch...)
        Ys = [ Γ * replaceind!(Cs[a]*Σ , Ωframe[a], ΩframeTemp[a]) for a in 1:engsize] 
        return unsafe_transposeEmbed(Ω,Ys)
    end

    # Compose sylv and ester as in sylvester4
    function sylvester(Xvec)
        Xs = unsafe_embedITensorsSwapped(Ω, Xvec)
        Σ = [ Cs[a]*Xs[a]*Γs_relabled[a] for a in 1:engsize ] |> sum 
        Ys = [ Γ * replaceind!(Cs[a]*Σ , Ωframe[a], ΩframeTemp[a]) for a in 1:engsize] 
        return unsafe_transposeEmbed(Ω,Ys)
    end

    # Wrap ester and sylve as LinearMaps
    densor_map = LinearMaps.LinearMap{T}(ester, sylve, densor_dim, op_dim; ismutating=false)
    derdensor_map = LinearMaps.LinearMap{T}(sylvester, sylvester, op_dim, op_dim; ismutating=false, issymmetric=true, isposdef=false)
    return derdensor_map, densor_map
end;
