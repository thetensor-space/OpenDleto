### convenience function useful for front end and exports 
###


# backward compatible

"""
randomize_tensor(t::ITensor; f::Function)

Picks random invertible matrices and use them to perform a random basis change of a tensor.

Returns a named tuple with fields:
- `Δ` the randomized tensor
- `Xs` the list of random matrices used for the basis change

f is a function with an argumen index and returns :invertible or :orthogonal
""" 
function randomize_tensor(
    Γ::ITensors.ITensor; 
    type::Symbol =:invertible,
    ):: NamedTuple{(:Δ, :Xs), Tuple{ITensors.ITensor, Vector{ITensors.ITensor}}}
    optype = (type== :invertible) ? :InvertableOp : :OrthogonalOp
    TOp = TransverseOps(T,optype)
    rT = changeBasisRandom(T,TOp)
    return (Δ=rT.Σ, Xs=rT.mats)
end

function randomize_tensor(Γ::AbstractArray; type::Symbol=:invertible)
    iΓ = __ITensor(Γ)
    return randomize_tensor(iΓ; type=type)
end;

"""
    Make ITensor out of AbstractArray without indices, 
    don't export this function is dangerous and makes 
    no promise to keep working in future versions of Dleto.jl.
    It is mainly here to conveniently fix a few places where 
    ITensors but 1691 is blocking proper use of AbstractArrays.
"""
function __ITensor(Γ::AbstractArray)::ITensors.ITensor 
    frame = [ ITensors.Index(size(Γ,a), "a$a") for a in 1:ndims(Γ) ]
    try 
        return ITensors.ITensor(Γ, frame...) 
    catch e
        return ITensors.ITensor( Array(Γ), frame...)
    end
end
