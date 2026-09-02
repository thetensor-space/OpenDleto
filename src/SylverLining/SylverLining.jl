
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
# SylverLiningMethod(; solver::Symbol=:KrylovSolver)= SylverLiningMethod(solver);
SylverLiningMethod(; solver::Symbol=:SVDSolver)= SylverLiningMethod(solver);


function derTrOpsReduced(method::SylverLiningMethod,
    Ω::TransverseOps, 
    P::AbstractMatrix, 
    Γ::ITensor;
    tol::Float64=1e-6,
    nd=-1,  # Don't type as integer to allow Inf 
    kwargs...,
    ) ::Tuple{TransverseOps, LinearMaps.LinearMap, AbstractMatrix{<: Number} }
    Γ_frame = inds(Γ)
    val = ndims(Γ)
    # `nd <= 0` means "no cap -- return a basis of the whole derivation space",
    # which is what the docstring promises.  It used to be rewritten to `val`,
    # which silently truncated the basis to the valency: for the diagonal
    # 3x3x3 tensor the space is 6-dimensional (all diagonal triples with
    # a_i + b_i + c_i = 0) and callers got 3 of them.  `nv` is only how many
    # vectors to ask an *iterative* solver for; the truncation below now
    # applies only when the caller set a positive cap.
    nv = nd <= 0 ? val : nd
    @assert Γ_frame == frames(Ω) "Incompatable Indexes"
    @assert val == size(P, 2) "Incompatable Chisel"
    
    # Compute reduced operators (matching what sylvesterLM does internally)
    eng = engaged(P)
    (Ω_reduced,expand_map) = reduceByEngaged(Ω, eng)
    P_eng = P[:,eng]

    # if globalDim(reducedΩ) < 10000
    # MDK, we need to reduce the chisel and pass the reduced chisel to the helper function
    sylvester, ester = sylvesterLM(Ω_reduced, P_eng, Γ)
    
    # TBD: Call into a eigen solver library that handles LinearMaps directly
    vecs = Vector{Float64}[]
    λ = Float64[]
    if globalDim(Ω_reduced) < 1000
        M = Matrix(sylvester)
        λ, vecs_matrix = eigen(M)
        # Convert matrix columns to vector of vectors for consistency
        vecs = [vecs_matrix[:, i] for i in 1:size(vecs_matrix, 2)]
    else
        result = solve(sylvester, method.solver; nv=nv)
        λ = result.vals
        vecs = [result.vecs[:, i] for i in 1:size(result.vecs, 2)]
        # nev = min(nd, size(sylvester, 1))  # Number of eigenvalues to compute
        # x0 = randn(size(sylvester, 2))     # Initial guess vector

        # # Retry logic for convergence
        # max_attempts = 5
        # maxiter = 100
        # krylovdim = max(10, 2*nev)
        # converged = false
        
        # for attempt in 1:max_attempts
        #     λ, vecs, info = eigsolve(sylvester, x0, nev, :SR;
        #         maxiter=maxiter,
        #         krylovdim=krylovdim,
        #         tol=tol
        #     )
            
        #     converged = info.converged >= nev
            
        #     if converged
        #         # Success! Convert and continue
        #         λ = real.(λ)
        #         vecs = [real.(v) for v in vecs]
        #         break
        #     else
        #         # Not enough converged, increase parameters and retry
        #         @warn "Attempt $attempt: Only $(info.converged) of $nev eigenvalues converged. Retrying with increased parameters..."
        #         maxiter = Int(round(maxiter * 1.5))
        #         krylovdim = min(Int(round(krylovdim * 1.5)), size(sylvester, 1))
        #         x0 = randn(size(sylvester, 2))  # New random start
                
        #         if attempt == max_attempts
        #             # Last attempt failed, use what we have
        #             @warn "Final attempt: Using $(info.converged) converged eigenvalues out of $nev requested."
        #             λ = real.(λ)
        #             vecs = [real.(v) for v in vecs]
        #         end
        #     end
        # end
    end
    
    # Filter vectors by eigenvalue tolerance
    valid_indices = findall(abs.(λ) .< tol)
    @assert length(valid_indices) > 0 "Not enough eigenvalues computed; increase `tol` parameter."
    λ = λ[valid_indices]
    vecs = vecs[valid_indices]
    
    # Give only nd many vectors, unless nd < 0 or Inf
    if nd > 0 && length(vecs) > nd
        vecs = vecs[1:floor(Int, nd)]
    end
    
    return (Ω_reduced, expand_map, hcat(vecs...) )
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
    ch_axis = Index(size(P, 1), "chisel")
    eng_axis = Index(engsize, "engaged")
    Cs = [ ITensor(P[:,a],  ch_axis) for a in 1:engsize ]

    Γ_frame_ch = (ch_axis, inds(Γ)...)

    Ωframe=frames(Ω)
    ΩframeTemp=framesTemporary(Ω)
    Cs = [ ITensor(P[:,a],  ch_axis) for a in 1:engsize ]
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
    densor_map = LinearMaps.LinearMap(ester, sylve, densor_dim, op_dim; ismutating=false)
    derdensor_map = LinearMaps.LinearMap(sylvester, sylvester, op_dim, op_dim; ismutating=false, issymmetric=true, isposdef=false)
    return derdensor_map, densor_map
end;

