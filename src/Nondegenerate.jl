#
# Strata Dleto: Nondegenerate tensors
#   ?????.
#
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
function nondeg(Γ::ITensor, 
                A::Vector{Index{T}} where T; 
                mode::Symbol=:trunc,
                tol::Float64=1e-10):: NamedTuple{(:Δ, :Es), Tuple{ITensor, Vector{ITensor}}} 
#MDK, I think what you call :ful is actually HoSVD, and calling it Tucker might upset someone...

    fr = inds(Γ)
    # Sort fr from largest size to smallest 
    # MDK, what is the point of sorting?
    fr = sort(collect(fr); by=ITensors.dim, rev=true)
    Es = Vector{ITensor}(undef, length(A))
    Δ = Γ
    i=1;
    for a in filter(x -> x in A, fr)
        a_comp = filter(e -> e != a, fr)
        U, S, V = ITensors.svd(Γ, a_comp)
        if mode == :full
            Es[i] = V
            Δ = Δ * Es[i]
            # continue
        else
            # Extract diagonal singular values from S as a regular array
            S_vals = diag(Array(S, inds(S)...))
            nondeg_idx = findall(s -> s >= tol, S_vals)
            in_dim = ITensors.dim(a)
            if isempty(nondeg_idx)
                # Fully degenerate - return identity (edge case)
                e = __new_index_for_nondegenerate(a,in_dim)
                Es[i] = ITensor(Matrix{Float64}(I, in_dim, in_dim), a, e)
                # MDK we should trow an error since we are given practially the zero tensor.
            else
                # V is an ITensor with indices for the input space
                # Convert V to array and extract the nondegenerate columns
                r = inds(V, tags="Link,v" )
                V_arr = Array(V, r,a)  # a is original index, second is SVD index
                nondeg_basis = V_arr[:, nondeg_idx]  # Shape: in_dim × length(nondeg_idx)
                # Create index for the nondegenerate subspace
                a_nondeg = __new_index_for_nondegenerate(a,length(nondeg_idx))
                Es[i] = ITensor(nondeg_basis, a, a_nondeg)
                Δ = Δ * Es[i]
            end
        end
        i = i + 1
    end
    return (;Δ = Δ, Es=Es[1:(i-1)])
end;

nondeg(Γ::ITensor; mode::Symbol=:trunc, tol::Float64=1e-10) = nondeg(Γ, collect(inds(Γ)); mode=mode, tol=tol);

function nondeg(Γ::AbstractArray, 
                A::Vector{<:Integer}; 
                mode::Symbol=:trunc,
                tol::Float64=1e-10)
    iΓ = __ITensor(Γ)
    return nondeg(iΓ, [ ITensors.inds(iΓ)[a] for a in A ]; mode=mode, tol=tol)
end;

nondeg(Γ::AbstractArray; mode::Symbol=:trunc, tol::Float64=1e-10) = 
    nondeg(Γ, collect(1:ndims(Γ)); mode=mode, tol=tol);



function __new_index_for_nondegenerate(i::Index, size::Number)::Index
    i_nondeg = Index(size, tags(i))
    i_nondeg = addtags(i_nondeg, "nondeg")
    return i_nondeg
end;
