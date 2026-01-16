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
struct SylverLinningMethod <: DerivationMethod 
    solver::NullSolver
end;


function derivations(
        method  ::SylverLinningMethod, 
        DP      ::DerivationProblem, 
        T       ::ITensors.ITensor; 
        nd      ::Integer=10,
        tol     ::Float64=1e-6,
        kwargs... 
        ) ::NamedTuple{(:ders, :vals), Tuple{Matrix{<:Number},Vector{<:Number}}}
    # test that the Tesnor is OK
    @assert testTensor(DP.TOp, T) "Tensor is not compatible"

    # reduce the problme to save space
    rDP = reduce(DP)

    # construct linear maps for reduces problem
    lmaps = sylvesterLM(rDP.DP, T)

    # solve reduced problem
    rders = solve(method.solver,lmaps.mapsquared; nd=nd, tol=tol, kwargs...)

    # expand solution 
    if size(rders.vecs,2) > 0
        ders = hcat([ (rDP.reduce_map)'*rders.vecs[:,i] for i =1:size(rders.vecs,2) ]... )
        return (;ders=ders,vals=rders.vals)
    else 
        # no derivations found
        return (;ders=zeros(globalDim(DP.TOp.LOps),0), vals=zeros(0))
    end
end;




"""
    sylvesterLM(P::Matrix, T::ITensor)

    Constructs LinearMaps for the derivation and densor maps associated to the given chisel `ch` and tensor `T`.

    - `P`: A matrix whose columns define the chisel polynomials.
    - `T`: The input tensor.

    Returns a tuple `(derdensor_map, densormap)` where:
    - `derdensor_map`: the composed derivation-densor `LinearMap` (a real symmetric operator).
    - `densormap`: the densor operator with transpose---the derivation operator---included.
"""
function sylvesterLM(        
    DP      ::DerivationProblem, 
    T       ::ITensors.ITensor) 

    T_frame = ITensors.inds(T)
    engsize = DP.ch.frames.len
    ch_axis = DP.ch.ch_axis

    Cs = [ ITensors.ITensor( DP.ch.ch_m[:,a],  ch_axis) for a in 1:engsize ]

    T_frame_ch = (ch_axis, inds(T)...)
    Ts_relabled =  [ ITensors.replaceind(T, DP.frames[a], DP.framesTemp[a]) for a in 1:engsize]

    # Compute sizes for LinearMap
    densor_dim = prod([ITensors.dim(f) for f in T_frame_ch ])
    op_dim = globalDim(DP.TOp.LOps)

    # Takes a vectorized representation of derivations
    # Returns a vectorized representation of the tensor
    function ester(Xvec)
        Xs = unsafe_embedITensorsSwapped(Dp.TOp, Xvec)
        Sigma = [ Cs[a]*Xs[a]*Ts_relabled[a] for a in 1:engsize ] |> sum 
        return vec(Array(Sigma, T_frame_ch...))
    end
    
    # sylv: takes a vectorized tensor, returns a vector of matrices (one per axis)
    function sylve(y)
        Sigma = ITensor(y, T_frame_ch...)
        Ys = [ T * ITensors.replaceind!(Cs[a]*Sigma , DP.frames[a], DP.framesTemp[a]) for a in 1:engsize] 
        return unsafe_transposeEmbed(DP.TOp,Ys)
    end

    # Compose sylv and ester as in sylvester4
    function sylvester(Xvec)
        Xs = unsafe_embedITensorsSwapped(DP.TOp, Xvec)
        Sigma = [ Cs[a]*Xs[a]*Ts_relabled[a] for a in 1:engsize ] |> sum 
        Ys = [ T * ITensors.replaceind!(Cs[a]*Sigma , DP.frames[a], DP.framesTemp[a]) for a in 1:engsize] 
        return unsafe_transposeEmbed(DP.TOp,Ys)
    end

    # Wrap ester and sylve as LinearMaps
    densor_map = LinearMaps.LinearMap(ester, sylve, densor_dim, op_dim; ismutating=false)
    derdensor_map = LinearMaps.LinearMap(sylvester, sylvester, op_dim, op_dim; ismutating=false, issymmetric=true, isposdef=false)
    return (mapsquared=derdensor_map, map=densor_map)
end;

