
#
# Strata Dleto: AbstractDerivationMethods
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

# """
#     Derivations Methods

#     Interface to methods for solving Sylvester equations arising in chiseling.

# """
"""
    DerivationMethod

    An interface for computing derivations.  Derivation methods 
    should inherit DerivationMethod and implement der and den.
"""
abstract type DerivationMethod end;

#generate DerivationMethod from symbol
function DerivationMethod(sym::Symbol=:default) 
    if sym in keys(DerivationMethodDict)
        return DerivationMethodDict[sym]
    else
        error("Unknown derivation method symbol: $sym")
    end
end;


"""
    derivations(
        method  ::DerivationMethod, 
        TOp     ::TransverseOps, 
        Ch      ::Chisel, 
        T       ::ITensor; 
        nd      ::Integer=10,
        tol     ::Float64=1e-6,
        ) :: Matrix{<:Number}

    Computes up to `nd` many derivations of `T` for the to the given chisel `Ch` and transverse operators `TOp`.
    If `nd` is negative or exceeds the dimension of the derivation space
    then the a basis for the derivation space is returned.
    - `method`: An instance of a subtype of `DerivationMethod` defining the solving method.
    - `Ω`: TransverseOps.
    - `P`: a linear chisel
    - `Γ`: The input tensor
    - `nd`: (optional) Maximum number of singular vectors to compute (default: 10). 
    If infinite `Inf` then return all, 
    - `tol`: (optional) Tolerance for the solver (default: 1e-6).
    attempt to compute a basis for the derivation space.

    Returns a matrix where columns represent basis of the space of derivations and 
    can be transformed into IThensors using methods of TOp.
"""
function derivations(
        method  ::DerivationMethod, 
        TOp     ::TransverseOps, 
        Ch      ::Chisel, 
        T       ::ITensors.ITensor; 
        nd      ::Integer=10,
        tol     ::Float64=1e-6,
        kwargs... 
        ) :: Matrix{<:Number}
    @assert false "Calling Placeholder Abstract Function"
end;


function derivationsIT(
        method  ::DerivationMethod, 
        TOp     ::TransverseOps, 
        Ch      ::Chisel, 
        T       ::ITensors.ITensor; 
        nd      ::Integer=10,
        tol     ::Float64=1e-6,
        kwargs... 
        ) :: Vector{<:Vector{<:ITensors.ITensor}} 
    der = derivations(method, TOp,Ch,T;nd=nd,tol=tol,kwargs...) 
    return [ embedITensors(TOp,der[:,i]) for i= 1:size(der,2) ]
end;


"""
    Dictionary of implemented/loaded derivation methods
"""
DerivationMethodDict = Dict{Symbol, DerivationMethod}()
