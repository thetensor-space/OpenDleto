#
# Strata Dleto: Utils
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

# ----- Randomization of tensors -----
"""
randomize_tensor(t::ITensor; type::Symbol=:invertible)

Picks random invertible matrices and use them to perform a random basis change of a tensor.

Returns a named tuple with fields:
- `Δ` the randomized tensor
- `Xs` the list of random matrices used for the basis change

type can be:
- `:invertible` (default) random invertible matrices
- `:orthogonal` random orthogonal matrices
""" 
function randomize_tensor(Γ::ITensor; type::Symbol=:invertible):: NamedTuple{(:Δ, :Xs), Tuple{ITensor, Vector{ITensor}}}
    fr = inds(Γ)
    mats = Vector{ITensor}(undef, length(fr))
    for a in 1:length(fr)        
        mat = __generate_random_matrix(type)(ITensors.dim(fr[a]))
        outer = __new_index_for_randomization(fr[a])
        mats[a] = ITensor( mat, fr[a], outer )
    end
    return (;Δ=Γ*mats, Xs=mats)
end;

function randomize_tensor(Γ::AbstractArray; type::Symbol=:invertible)
    iΓ = __ITensor(Γ)
    return randomize_tensor(iΓ; type=type)
end;

# --- Utiliity functions ---
function __random_invertible(n::Integer) 
    mat = randn(n,n)
    count = 0
    while LinearAlgebra.rank(mat) < n && count < 100
        count += 1
        mat = randn(n,n)
    end
    if count > 99
        throw(ErrorException("Could not generate random invertible matrix"))
    end
    return mat
end

__random_invertible(i::Index) = __random_invertible(ITensors.dim(i));

function __random_orthogonal(n::Integer)
    mat = randn(n,n)
    count = 0
    while LinearAlgebra.rank(mat) < n && count < 100
        count += 1
        mat = randn(n,n)
    end
    if count > 99
        throw(ErrorException("Could not generate random orthogonal matrix"))
    end
    Q,R = LinearAlgebra.qr(mat)
    D = Diagonal(sign.(diag(R)))
    return Q*D
end;
__random_orthogonal(i::Index) = __random_orthogonal(ITensors.dim(i));


function __generate_random_matrix(type::Symbol)::Function  
    if type == :invertible
        return __random_invertible
    elseif type == :orthogonal
        return __random_orthogonal
    else
        throw(ErrorException("Unknown randomization type: $type"))
    end
end;

### MDK potentially very unsafe! It is beter to add a suitable tag
function __new_index_for_randomization(i::Index)::Index
    return addtags(i, "Randomized," * string(rand(UInt16))) #i'
end;

function __new_index_for_change_of_basis(i::Index)::Index
    return addtags(i, "Basis," * string(rand(UInt16))) #i'
end;