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

"""
    Make ITensor out of AbstractArray without indices, 
    don't export this function is dangerous and makes 
    no promise to keep working in future versions of Dleto.jl.
    It is mainly here to conveniently fix a few places where 
    ITensors but 1691 is blocking proper use of AbstractArrays.
"""
function __ITensor(Γ::AbstractArray)::ITensor 
    frame = [ Index(size(Γ,a), "a$a") for a in 1:ndims(Γ) ]
    # a temporary fix for AbstractArray inputs like ReshapedArray which have a bug
    # in ITensors (Bug #1691)
    iΓ = typeof(Γ) <: Array ? ITensor(Γ, frame...) : ITensor( Array(Γ), frame...)
    return iΓ
end;


# -- Direct action by a vector of ITensors -----

function Base.:*(Γ::ITensor, X::Vector{ITensor})  
    Σ = Γ  
    for x in X
        Σ = Σ * x
    end
    return Σ
end;

# --- Wrappers to promote AbstractArray to ITensor -----
function Base.:*(Γ ::AbstractArray, X::Vector{ITensor}) 
    @assert length(X) == ndims(Γ) "Ambiguous frame matching: length of list of matrices must match array axes"
    # we need assert that dims are the same not only valances
    fr = [ ITensors.inds(x)[1] for x in X ]
    iΓ = typeof(Γ) <: Array ? ITensor(Γ, fr...) : ITensor( Array(Γ), fr...)
    return iΓ * X
end;

function Base.:*(Γ::ITensor, X::Vector{<:AbstractMatrix}) 
    @assert length(X) == ndims(Γ) "Ambiguous frame matching: length of list of matrices must match ITensor axes"
    fr = inds(Γ)
    iX = [ ITensor( Array(X[i]), fr[i], __new_index_for_change_of_basis(fr[i]) ) for i in 1:length(X) ] 
    return Γ * iX
    # MDK This does not work if Γ = random_itensor(i,i',i'')!!!!!
end;

# Make actions Ambidextrous
function Base.:*(X::Vector{ITensor}, Γ::AbstractArray ) 
    return Γ * X
end;

function Base.:*(X::Vector{ITensor}, Γ::ITensor ) 
    return Γ * X
end;

function Base.:*(X::Vector{<:AbstractMatrix}, Γ::ITensor ) 
    return Γ * X
end;

__isapproxzero(x::Number)::Bool = isapprox(x,0.0);

