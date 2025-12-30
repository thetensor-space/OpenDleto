#
# Strata Dleto: Chisels
#   Creation and adaptation of chisels for tensor decomposition.
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
    Make ITensor out of AbstractArray
"""
function ArrayToITensor(t::AbstractArray)::ITensor 
    frame = [ Index(size(t,a), "a$a") for a in 1:ndims(t) ]
    iΓ = typeof(Γ) <: Array ? ITensor(Γ, frame...) : ITensor( Array(Γ), frame...)
    return iΓ
    return ITensor(t, frame...)
end;


# -- Direct action by a vector of ITensors -----

function Base.:*(Γ::ITensor, X::Vector{ITensor})  
    Σ = Γ  
    for x in X
        Σ = Σ * x
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
    # need to assert square matrices and compatiblle dimensions
    fr = inds(Γ)
    iX = [ ITensor( Array(X[i]), fr[i], addtags(fr[i],"new") ) for i in 1:length(X) ]
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


