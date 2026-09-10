#
# Strata Dleto: TensorSpace soft wrappers
#   Optional operator ergonomics without type piracy.
#

module TensorSpace

using ITensors: ITensor, Index, dim
import ITensors: inds, store, array
import ..Dleto: act, ⊕, __ITensor
import ..Dleto: randSurfaceTensor as _randSurfaceTensor
import ..Dleto: randFaceCurveTensor as _randFaceCurveTensor
import ..Dleto: randCurveTensor as _randCurveTensor

import Base: *, +, getindex, isapprox, setindex!, size, ndims, Array

export Axis, TensorElement, SoftTensor, tensor, ts, unwrap, contract, ⊕
export randSurfaceTensor, randFaceCurveTensor, randCurveTensor

"""
    Axis

Preferred local name for `ITensors.Index`, so users can work in `Dleto.TensorSpace`
without bringing `Index` into their own namespace.
"""
const Axis = Index

"""
    TensorElement

ITensor-first wrapper for optional tensor-space operator ergonomics.
"""
struct TensorElement
    value::ITensor
end

"""
    SoftTensor{T}

Light wrapper that enables optional `*` and `+` ergonomics for tensor-space
workflows without adding pirated `Base` methods on foreign types.
"""
const SoftTensor = TensorElement

"""
    tensor(Γ)

Wrap a tensor-like object as `TensorElement`.
"""
tensor(Γ::ITensor) = TensorElement(Γ)
tensor(Γ::AbstractArray) = TensorElement(__ITensor(Γ))
tensor(Γ::AbstractArray, like::ITensor) = TensorElement(_to_tensor_element_like(Γ, like))
tensor(Γ::AbstractArray, like::TensorElement) = tensor(Γ, unwrap(like))

"""
    ts(Γ)

Wrap a tensor-like object so the opt-in `TensorSpace` operators are available:

- `ts(Γ) * Xs` dispatches to `act(Γ, Xs)`
- `ts(Γ) + Δ` adds tensors
- `ts(Γ) ⊕ Δ` computes direct sum
"""
ts(Γ::Union{ITensor, AbstractArray}) = tensor(Γ)

_ensure_tensor_element(x::TensorElement) = x
_ensure_tensor_element(x::ITensor) = tensor(x)

"""
    randSurfaceTensor(xes, yes, zes, cutoff)

Create a random surface-supported tensor and wrap it as `TensorElement`.
"""
randSurfaceTensor(xes::Vector, yes::Vector, zes::Vector, cutoff::Number) =
    _ensure_tensor_element(_randSurfaceTensor(xes, yes, zes, cutoff))

"""
    randFaceCurveTensor(xes, yes, zes, cutoff)

Create a random face-curve-supported tensor and wrap it as `TensorElement`.
"""
randFaceCurveTensor(xes::Vector, yes::Vector, zes::Vector, cutoff::Number) =
    _ensure_tensor_element(_randFaceCurveTensor(xes, yes, zes, cutoff))

"""
    randCurveTensor(xes, yes, zes, cutoff)

Create a random curve-supported tensor and wrap it as `TensorElement`.
"""
randCurveTensor(xes::Vector, yes::Vector, zes::Vector, cutoff::Number) =
    _ensure_tensor_element(_randCurveTensor(xes, yes, zes, cutoff))

"""
    unwrap(s::TensorElement)

Recover the wrapped tensor-like value.
"""
unwrap(s::TensorElement) = s.value

ndims(s::TensorElement) = ndims(unwrap(s))
size(s::TensorElement) = size(unwrap(s))
size(s::TensorElement, dims...) = size(unwrap(s), dims...)
getindex(s::TensorElement, inds...) = getindex(unwrap(s), inds...)
setindex!(s::TensorElement, v, inds...) = (unwrap(s)[inds...] = v; s)
inds(s::TensorElement, args...) = inds(unwrap(s), args...)
store(s::TensorElement) = store(unwrap(s))
array(s::TensorElement, args...) = array(unwrap(s), args...)
Array(s::TensorElement, args...) = Array(unwrap(s), args...)

# Convert supported values to TensorElement for sticky mixed operations.
_to_tensor_element(x::TensorElement) = x
_to_tensor_element(x::ITensor) = tensor(x)
_to_tensor_element(x::AbstractArray) = tensor(x)

function _to_tensor_element_like(x::AbstractArray, like::ITensor)
    length(size(x)) == ndims(like) ||
        throw(DimensionMismatch("TensorSpace promotion: array has $(ndims(x)) axes but tensor has $(ndims(like)) axes."))
    fr = inds(like)
    all(size(x, a) == dim(fr[a]) for a in 1:ndims(like)) ||
        throw(DimensionMismatch("TensorSpace promotion: array shape $(size(x)) does not match tensor axis dimensions $(Tuple(dim(i) for i in fr))."))
    iX = x isa Array ? ITensor(x, fr...) : ITensor(Array(x), fr...)
    return tensor(iX)
end

"""
    contract(a, b)

Tensor contraction helper returning a wrapped result.
"""
contract(a::Union{TensorElement, ITensor, AbstractArray}, b::Union{TensorElement, ITensor, AbstractArray}) =
    tensor(unwrap(_to_tensor_element(a)) * unwrap(_to_tensor_element(b)))

# --- Optional `*` ergonomics -------------------------------------------------

*(a::TensorElement, b::TensorElement) = contract(a, b)
*(a::TensorElement, b::ITensor) = contract(a, b)
*(a::ITensor, b::TensorElement) = contract(a, b)
*(a::TensorElement, b::AbstractArray) = contract(a, _to_tensor_element_like(b, unwrap(a)))
*(a::AbstractArray, b::TensorElement) = contract(_to_tensor_element_like(a, unwrap(b)), b)
*(a::AbstractArray, Xs::Vector{ITensor}) = ts(a) * Xs
*(a::AbstractArray, Xs::Vector{<:AbstractMatrix}) = ts(a) * Xs
*(s::TensorElement, Xs::Vector{ITensor}) = tensor(act(unwrap(s), Xs))
*(s::TensorElement, Ms::Vector{<:AbstractMatrix}) = tensor(act(unwrap(s), Ms))

# --- Optional `+` ergonomics -------------------------------------------------

+(a::TensorElement, b::TensorElement) = tensor(unwrap(a) + unwrap(b))
+(a::TensorElement, b::ITensor) = tensor(unwrap(a) + b)
+(a::ITensor, b::TensorElement) = tensor(a + unwrap(b))
+(a::TensorElement, b::AbstractArray) = tensor(unwrap(a) + unwrap(_to_tensor_element_like(b, unwrap(a))))
+(a::AbstractArray, b::TensorElement) = tensor(unwrap(_to_tensor_element_like(a, unwrap(b))) + unwrap(b))

# --- Optional direct-sum ergonomics ------------------------------------------

⊕(a::TensorElement, b::TensorElement) = tensor(⊕(unwrap(a), unwrap(b)))
⊕(a::TensorElement, b::ITensor) = tensor(⊕(unwrap(a), b))
⊕(a::ITensor, b::TensorElement) = tensor(⊕(a, unwrap(b)))
⊕(a::TensorElement, b::AbstractArray) = tensor(⊕(unwrap(a), unwrap(_to_tensor_element_like(b, unwrap(a)))))
⊕(a::AbstractArray, b::TensorElement) = tensor(⊕(unwrap(_to_tensor_element_like(a, unwrap(b))), unwrap(b)))

# --- Comparisons --------------------------------------------------------------

_tensor_array(x::ITensor) = Array(x, inds(x)...)

function _isapprox_ignoring_index_labels(a::ITensor, b::ITensor; kwargs...)
    try
        return isapprox(a, b; kwargs...)
    catch err
        return isapprox(_tensor_array(a), _tensor_array(b); kwargs...)
    end
end

isapprox(a::TensorElement, b::TensorElement; kwargs...) =
    _isapprox_ignoring_index_labels(unwrap(a), unwrap(b); kwargs...)
isapprox(a::TensorElement, b::ITensor; kwargs...) =
    _isapprox_ignoring_index_labels(unwrap(a), b; kwargs...)
isapprox(a::ITensor, b::TensorElement; kwargs...) =
    _isapprox_ignoring_index_labels(a, unwrap(b); kwargs...)
isapprox(a::TensorElement, b::AbstractArray; kwargs...) =
    _isapprox_ignoring_index_labels(unwrap(a), unwrap(_to_tensor_element_like(b, unwrap(a))); kwargs...)
isapprox(a::AbstractArray, b::TensorElement; kwargs...) =
    _isapprox_ignoring_index_labels(unwrap(_to_tensor_element_like(a, unwrap(b))), unwrap(b); kwargs...)

end # module TensorSpace
