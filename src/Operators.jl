#
# Strata Dleto: Local Operators Abstract
#   Creation and application of local operators for tensors.
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
A Tensor Operator is any data type `Ω` that implement the functions 
```Julia
apply(ρ::Operator{Ω}, ω::Ω, Γ::ITensor)::ITensor where {Ω, N}
unsafe_apply(ω::Ω, Γ_in::ITensor, Γ_out::ITensor, a::Index{N})::ITensor where {Ω, N}
member(ω::Ω)::Bool where {Ω}   
```
We say that `ω` is an operator if `member(ω)==true`.
The behavior of all safe functions is limited to operators and may throw exceptions for non-operators.

The implementations should satisfy the following laws.

For each operator `ω`, scalar `s`, tensors `Γ, Δ` with equal indices, and index `a`:
```Julia
apply(ρ, ω, Γ + Δ) ≈ apply(ρ, ω, Γ) + apply(ρ, ω, Δ)
apply(ρ, ω, Γ * s) ≈ apply(ρ, ω, Γ) * s
apply(ρ, ω, Γ) == unsafe_apply(ρ, ω, Γ)  if ω in Ω == true
inds(apply(ρ, ω, Γ)) == inds(Γ)             if ω in Ω == true
```

---

A Linear Tensor Operator is a Tensor Operator `Ω` that 
is both a vector space and implements the following additional functions
```Julia
coapply(Λ::ITensor, Δ::ITensor, a::Index{N})::Ω where {Ω, N}
unsafe_apply(Λ::ITensor, Δ::ITensor, a::Index{N})::Ω where {Ω, N}
```
and such that these together with the operator functions satisfy the additional laws:
```Julia
apply(ρ, ω + λ, Γ) ≈ apply(ρ, ω, Γ) + apply(ρ, λ, Γ)
apply(ρ, s * ω, Γ) ≈ s * apply(ρ, ω, Γ)

coapply( ρ, Λ_1 + Λ_2, Δ) ≈ coapply(ρ, Λ_1, Δ) + coapply(ρ, Λ_2, Δ)
coapply(ρ, s * Λ, Δ) ≈ s * coapply(ρ, Λ, Δ)
coapply(ρ, Λ, Δ_1 + Δ_2) ≈ coapply(ρ, Λ, Δ_1) + coapply(ρ, Λ, Δ_2)
coapply(ρ, Λ, Δ * s) ≈ coapply(ρ, Λ, Δ) * s

coapply(ρ, Λ, s*Δ) ≈ s * coapply(ρ, Λ, Δ)
coapply(ρ, Λ, Δ) == unsafe_coapply(ρ, Λ, Δ)  if (Λ) == true
```
furthermore, `L ↦ coapply(L)` is the dual map to `ω ↦ apply(ω)`.
"""
module Operators

using ITensors
using DocStringExtensions

struct Operator{Ω} 
    a ::Index
end

"""
$(TYPEDSIGNATURES)
Apply the operator ω on the tensor Γ along the representation's axis.
"""
function apply(ρ::Operator{Ω}, ω::Ω, Γ::ITensor)::Union{ITensor, Nothing} where {Ω}
    @assert false "Calling Placeholder Abstract Function"
end
"""
$(TYPEDSIGNATURES)
[`Dleto.apply`](@ref) without safety checks.
"""
function unsafe_apply(ρ::Operator{Ω}, ω::Ω, Γ::ITensor)::ITensor where {Ω}
    @assert false "Calling Placeholder Abstract Function"
end

"""
$(TYPEDSIGNATURES)
Return `true` if ω is a member of the operator space.
"""
function Base.in(ω::Ω, ρ::Operator{Ω})::Bool where {Ω}
    @assert false "Calling Placeholder Abstract Function"
end


"""
$(TYPEDSIGNATURES)
Given tensors Γ and Δ with the same indices and an index a,
contract Γ with Δ on all axes except a and encode the corresponding 
operator `ω::Ω` on index `a` that results.

`unsafe_coapply` does the same without safety checks.
"""
function coapply(ρ::Operator{Ω}, Γ::ITensor, Δ::ITensor)::Ω where {Ω, N}
    @assert false "Calling Placeholder Abstract Function"
end
"""
$(TYPEDSIGNATURES)
[`Dleto.coapply`](@ref) without safety checks.
"""
function unsafe_coapply(ρ::Operator{Ω}, Γ::ITensor, Δ::ITensor)::Ω where {Ω, N}
    @assert false "Calling Placeholder Abstract Function"
end

end # module Operators