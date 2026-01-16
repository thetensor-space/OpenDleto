#public constructors
function TransverseOps(f::Framed, s::Vector{Symbol};tag::String="temp",sym::Vector{<:Integer}=Int64[])::TransverseOps
    axises = f.frame
    @assert length(axises)==length(s) "not compatible"
    t_axises = axises .|> (x -> ITensors.addtags(x,tag)) 
    axisDims = ITensors.dim.(axises) 
    localOps = Operator.(s) 
    t_f = Framed(t_axises)
    if sym==[]
        return TransverseOps( IndListOperators(axisDims, localOps), f, t_f)
    else 
        return TransverseOps( SymListOperators(axisDims, localOps, sym), f, t_f)
    end
end;

# expand a single symbol to List if applicalble
TransverseOps(f::Framed, s::Symbol, more...; kwargs...)::TransverseOps = 
    TransverseOps(f, [s for a in f.frame], more...; kwargs...);

#next two transform the first argument from ITensor/Array of Index to Framing
TransverseOps(axises::Vector{<:ITensors.Index}, more...; kwargs...)::TransverseOps = 
    TransverseOps( Framed(axises), more...; kwargs...);


TransverseOps(T::ITensors.ITensor, more...; kwargs...)::TransverseOps =
    TransverseOps(getFraming(T), more...; kwargs...);


### convenience function useful for front end and exports 
###


# backward compatible

"""

randomize_tensor(t::ITensor; type::Symbol)

Picks random invertible matrices and use them to perform a random basis change of a tensor.

Returns a named tuple with fields:
- `T` the randomized tensor
- `Xs` the list of random matrices used for the basis change

f is a function with an argumen index and returns :invertible or :orthogonal
""" 
function randomize_tensor(
    T::ITensors.ITensor; 
    type::Symbol =:invertible,
    ):: NamedTuple{(:T, :Xs), Tuple{ITensors.ITensor, Vector{ITensors.ITensor}}}
    TOp = TransverseOps(T, (type== :invertible) ? :InvertableOp : :OrthogonalOp )
    return  changeBasisRandom(T,TOp)
    # rT = changeBasisRandom(T,TOp)
    # return (T=rT.T, Xs=rT.Xs)
end

function randomize_tensor(A::AbstractArray; kwargs...)
    T = __ITensor(A)
    return randomize_tensor(T; kwargs...)
end;

"""
    Make ITensor out of AbstractArray without indices, 
    don't export this function is dangerous and makes 
    no promise to keep working in future versions of Dleto.jl.
    It is mainly here to conveniently fix a few places where 
    ITensors but 1691 is blocking proper use of AbstractArrays.
"""
function __ITensor(A::AbstractArray)::ITensors.ITensor 
    frame = [ ITensors.Index(size(A,a), "a$a") for a in 1:ndims(Γ) ]
    try 
        return ITensors.ITensor(A, frame...) 
    catch e
        return ITensors.ITensor( Array(A), frame...)
    end
end
