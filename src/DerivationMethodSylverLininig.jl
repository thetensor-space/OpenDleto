
#
# Strata Dleto: Derivation Method SylverLining
#   Algorithms for solving Sylvester equations arising in chiseling.
# -----------------------------------------------------------------------------
# Copyright 2022-2025 Peter A. Brooksbank, Martin D. Kassabov, James B. Wilson
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
struct SylverLiningMethod <: DerivationMethod end;

function der(method::SylverLiningMethod,
    Ω::TransverseOps, 
    Ch::AbstractMatrix, 
    Γ::ITensor;
    tol::Float64=1e-6,
    nd=10,  # Don't type as integer to allow Inf 
    kwargs...,
    ) :: Vector{Vector{ITensor}}
    Γ_frame = inds(Γ)
    val = ndims(Γ)
    @assert Γ_frame == frames(Ω) "Incompatable Indexes"
    @assert val == size(Ch, 2) "Incompatable Chisel"
    
    # Compute reduced operators (matching what sylvesterLM does internally)
    eng = engaged(Ch)
    reducedΩ = reduceByEngaged(Ω, eng)
    
    # if globalDim(reducedΩ) < 10000
        sylvester, ester = sylvesterLM(Ω, Ch, Γ)
        M = Matrix(sylvester)
        vals, vecs = eigen(M)
        vecs = vecs[:, findall(abs.(vals) .< tol)]
        # give only nd many vectors, unless nd < 0 or Inf
        if nd > 0 && size(vecs,2) > nd
            vecs = vecs[:, 1:floor(Int, nd)]
        end
        # Use reducedΩ to embed (eigenvectors have dimension globalDim(reducedΩ))
        return [ unsafe_embedITensors(reducedΩ, vecs[:,i]) for i in 1:size(vecs,2) ]
    # else
    #     @assert false "Dimenstion Too Large"
    #     return []
    # end
end



"""
    sylvesterLM(ch::Matrix, T::ITensor)

    Constructs LinearMaps for the derivation and densor maps associated to the given chisel `ch` and tensor `T`.

    - `ch`: A matrix whose columns define the chisel polynomials.
    - `T`: The input tensor.

    Returns a tuple `(derdensor_map, densormap)` where:
    - `derdensor_map`: the composed derivation-densor `LinearMap` (a real symmetric operator).
    - `densormap`: the densor operator with transpose---the derivation operator---included.
"""
function sylvesterLM(Ω::TransverseOps, ch::AbstractMatrix, Γ::ITensor) #::Tuple{LinearMaps.LinearMap, LinearMaps.LinearMap}
#temporary the function retunrs also the naked functions in addition to the linear maps
#this is done for only for testing but needs to be fixed during the merge.
#I am gettign stupid error is ch is and empty matrix!!!
    Γ_frame = inds(Γ)
    val = ndims(Γ)
    # @assert Γ_frame == frames(Ω) "Incompatable Indexes"
    # @assert val == size(ch, 2) "Incompatable Chisel"
    eng = engaged(ch)
    engsize = sum(eng)
    reducedch = ch[:,eng]
    ch_axis = Index(size(ch, 1), "chisel")
    eng_axis = Index(engsize, "engaged")
    # Id=Matrix(1.0*LinearAlgebra.I,engsize,engsize)
    # eng_tensors = [ ITensor( Id[:,i],eng_axis) for i in 1:engsize]
    Cs = [ ITensor(ch[:,a],  ch_axis) for a in 1:val ]
    reducedCs = Cs[eng]
 
    reducedCTensor=ITensor(reducedch,ch_axis, eng_axis) ## check order of indeces!
    reducedΩ = reduceByEngaged(Ω, eng)
    Γ_frame_ch = (ch_axis, inds(Γ)...)
    Γ_frame_eng = (eng_axis, inds(Γ)...)

    reducedΩframe=frames(reducedΩ)
    reducedΩframeTemp=framesTemporary(reducedΩ)

    # Compute sizes for LinearMap
    densor_dim = prod([ITensors.dim(f) for f in Γ_frame_ch ])
    op_dim = globalDim(reducedΩ)

    # Takes a vectorized representation of derivations
    # Returns a vectorized representation of the tensor
    function ester(Xvec)
        Xs = unsafe_embedITensors(reducedΩ, Xvec)
        Σ = ITensor(Γ_frame_ch)
        for a in 1:engsize
            Δ = reducedCs[a]*Xs[a]*Γ  # this swiches index to a tmep one
            replaceind!(Δ, reducedΩframeTemp[a], reducedΩframe[a])  # fix index
            # Σ += permute(Δ, ch_axis, Γ_frame...)
            Σ += Δ #Itensor is intelegent enough and there is no need to permute
        end
        return vec(Array(Σ, Γ_frame_ch...))
    end
    
    # sylv: takes a vectorized tensor, returns a vector of matrices (one per axis)
    function sylve(y)
        # y_array = Array(reshape(y, dims)) ## slow step?  Dense.  no need
        Σ = ITensor(y, Γ_frame_ch...)
        # Ys = [ 
        #         permute(
        #             replaceind!(reducedCs[a]*Σ , reducedΩframe[a], reducedΩframeTemp[a])* Γ,
        #             reducedΩframe[a], reducedΩframeTemp[a]; allow_alias = true
        #             ) 
        #         for a in 1:engsize] 
        Ys = [ Γ * replaceind!(reducedCs[a]*Σ , reducedΩframe[a], reducedΩframeTemp[a]) for a in 1:engsize] 
                # Permute can be avoided by swichting the order of the tensor multiplication
        return unsafe_transposeEmbed(reducedΩ,Ys)
    end

    # Compose sylv and ester as in sylvester4
    function sylvester(Xvec)
        Xs = unsafe_embedITensors(reducedΩ, Xvec)
        Σ = ITensor(Γ_frame_ch)
        for a in 1:engsize
            Δ = reducedCs[a]*Xs[a]*Γ  # this swiches index to a tmep one
            replaceind!(Δ, reducedΩframeTemp[a], reducedΩframe[a])  # fix index
            # Σ += permute(Δ, ch_axis, Γ_frame...) 
            Σ += Δ 
        end
        # Ys = [ 
        #         permute(
        #             replaceind!(reducedCs[a]*Σ , reducedΩframe[a], reducedΩframeTemp[a])* Γ,
        #             reducedΩframe[a], reducedΩframeTemp[a]; allow_alias = true
        #             ) 
        #         for a in 1:engsize] 
        Ys = [ Γ * replaceind!(reducedCs[a]*Σ , reducedΩframe[a], reducedΩframeTemp[a]) for a in 1:engsize] 
                # Permute can be avoided by swichting the order of the tensor multiplication
        return unsafe_transposeEmbed(reducedΩ,Ys)
    end

    # Wrap ester and sylve as LinearMaps
    densor_map = LinearMaps.LinearMap(ester, sylve, densor_dim, op_dim; ismutating=false)
    derdensor_map = LinearMaps.LinearMap(sylvester, sylvester, op_dim, op_dim; ismutating=false, issymmetric=true, isposdef=false)
    return derdensor_map, densor_map
    # return ester, sylve, sylvester, op_dim, densor_dim, derdensor_map, densor_map
end;

