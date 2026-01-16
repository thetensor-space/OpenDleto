#
# Strata Dleto: Nondegenerate tensors
#   ?????.
#
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
#-----------------------------------------------------------------------------


"""
    nondeg(Γ::ITensor, A::Vector{Index}; tol::Float64=1e-10)

    For each axis `a` in `A`, computes projection onto a subspace with kernel 
    radical of the tensor on that axis.

    - `Γ`: The input tensor.
    - `A`: A vector of indices of `Γ` to consider.
    - mode: Symbol, either `:trunc` (default) or `:Tucker`.
        - `:trunc`: Projects onto nondegenerate subspace.
        - `:full`: Performs Tucker decomposition without truncation.
    - `tol`: Tolerance for determining nondegeneracy (default: 1e-10).

    Returns a named tuple with fields:
    - `Δ`: The tensor obtained by contracting `Γ` with the nondegenerate bases.
    - `E`: A vector of matrices, where each matrix corresponds to the basis
           for the nondegenerate subspace of the respective index in `A`.

    Law: `Δ = Γ * E` 
"""



function nondeg(T::ITensors.ITensor,
                A::Vector{<:ITensors.Index}; 
                mode::Symbol=:trunc,
                tol::Float64=1e-8):: NamedTuple{(:T, :Es), Tuple{ITensor, Vector{ITensor}}} 
#MDK, I think what you call :ful is actually HoSVD, and calling it Tucker might upset someone...
    fr = ITensors.inds(T)
    activeind = [A[a] for a = 1:length(A) if (A[a] in fr)]
    Es = activeind .|> (i -> __makeE(T,i,mode,tol))
    newT = T 
    for E in Es
        newT *= E
    end 
    return (;T = newT, Es=Es)
end;

nondeg(T::ITensors.ITensor, f::Framed; kwargs...) = nondeg(T, f.frame; kwargs...); 

function __makeE(T::ITensors.ITensor, i::ITensors.Index,mode::Symbol,tol::Float64)::ITensors.ITensor
    res = ITensors.svd(T,i)
    r = ITensors.inds(res.U, tags="Link,u" )[1]
    if mode ==:full
        newi = __new_index_for_nondegenerate(i, ITensors.dim(r))  
        return ITensors.replaceind!(res.U,r,newi)
    else
        S_vals = diag(Array(res.S, ITensors.inds(res.S)...))
        nondeg_idx = findall(s -> s >= tol, S_vals)
        U_arr = Array(res.U, i,r) 
        U_reduced = U_arr[:, nondeg_idx]  # Shape: in_dim × length(nondeg_idx)
        newi = __new_index_for_nondegenerate(i, length(nondeg_idx))
        return ITensors.ITensor(U_reduced,i,newi)
    end
end;


function __new_index_for_nondegenerate(i::ITensors.Index, size::Number)::ITensors.Index
    i_nondeg = ITensors.Index(size, tags(i))
    i_nondeg = addtags(i_nondeg, "nondeg")
    return i_nondeg
end;
