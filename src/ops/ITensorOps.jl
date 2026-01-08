#
# --- ITensor actions ---------------------------------------------------------
#
# An operator for the type ITensor.
# Here each ITensor must be labeled by `(a,a')`
# The apply command detect possibly conflicting use of `a'` and renames it if needed.


"""
$(TYPEDSIGNATURES)
Apply the operator `ω::ITensor` on the tensor Γ along the axis a.

- `ρ`: An `Operator{ITensor}` defining the action index `a`.
- `ω`: An `ITensor` with two indices `(a,a')` representing the operator.
- `Γ`: An `ITensor` on with index including `a`.

Returns an `ITensor` with the same indices as `Γ` resulting from acting by the operator `ω`.  
Note, if `Γ` includes `a'` as an index then the application implicitly 
reindexes both `ω` and `Γ` to avoid conflicts. 

`unsafe_apply` does the same without safety checks, in particular the user is 
expected to avoid using `a'` as index of `Γ` for properly behavior of `unsafe_apply`.

**Example.**
```Julia
using ITensors
using Dleto.Operators
x = Index(2, "x"); y = Index(3, "y"); z = Index(4, "z")
Γ = ITensor( collect(1:24), x,y,z)
ρ_x = Operator{ITensor}(x)

ω = asoperator(ρ_x, [2.0 0.0; 0.0 0.0])
result = apply(ρ_x, ω, Γ)

# result is ITensor([2.0 0.0 ; 0.0 -1.0]. x,y,z)
```
"""
function apply( ρ::Operator{ITensor}, ω::ITensor, Γ::ITensor)::ITensor
    @assert member(ω) "Operator ITensor must have exactly two indices that are dual to each other"
    @assert hasindex(ω, ρ.a) "Operator ITensor does not act on index $ρ.a"
    @assert hasindex(Γ, ρ.a) "Input ITensor does not have index $ρ.a"
    e = ρ.a 
    # Detect collision of index names
    if hasindex(Γ,ρ.a')
        e = addtags(ρ.a', "dot")  # Ensure correct contraction
        Γ = replaceind(Γ, ρ.a, e)
        ω = replaceind(ω, ρ.a', e)
    end
    return unsafe_apply(ρ, ω, Γ)
end

"""
$(TYPEDSIGNATURES)
[`Dleto.apply`](@ref) without safety checks.
"""
function unsafe_apply(ρ::Operator{ITensor}, ω::ITensor, Γ::ITensor)::ITensor
    return Γ * ω
end

"""
$(TYPEDSIGNATURES)
Return `true` if `ω::ITensor` with indices (a,a').
"""
function in(ω::ITensor, ρ::Operator{ITensor})::Bool
    return length(inds(ω)) == 2 && inds(ω)[1]' == ρ.a' &&  inds(ω)[2] == ρ.a
end

"""
$(TYPEDSIGNATURES)
Given a square matrix `M` and an index `a`,
construct the corresponding operator ITensor `ω::ITensor` with indices `(a,a')`.
"""
function asoperator(ρ::Operator{ITensor}, M::AbstractMatrix)::ITensor where {N}
    @assert size(M) == (dim(ρ.a),dim(ρ.a)) "Matrix should be square of same dimension as index."
    return ITensor(M, ρ.a, ρ.a')
end

"""
$(TYPEDSIGNATURES)
Given tensors Γ and Δ with the same indices and an index a,
contract Γ with Δ on all axes except a and encode the corresponding 
operator `ω::Ω` on index `a` that results.

`unsafe_coapply` does the same without safety checks.
"""
function coapply(ρ::Operator{ITensor}, Γ::ITensor, Δ::ITensor)::ITensor
    @assert inds(Γ) == inds(Δ) "Input ITensors must have the same indices"
    @assert hasindex(Γ, ρ.a) "Input ITensor Γ does not have index $ρ.a"
    @assert hasindex(Δ, ρ.a) "Input ITensor Δ does not have index $ρ.a"
    return unsafe_coapply(ρ, Γ, Δ)
end
function unsafe_coapply(ρ::Operator{ITensor}, Λ::ITensor, Δ::ITensor)::ITensor
    # assumes that Λ and Δ have the same indices
    Δ = replaceind(Δ, ρ.a, ρ.a') # Shallow copy just changes indices
    return Λ * Δ
end

