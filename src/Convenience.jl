#public constructors
function TransverseOps(axises::Vector{<:ITensors.Index}, s::Vector{Symbol};tag::String="temp") ::TransverseOps
    @assert length(axises)==length(s) "not compatible"
    t_axises = axises .|> (x -> ITensors.addtags(x,tag)) 
    axisDims = ITensors.dim.(axises) 
    localOps = Operator.(s) 
    frames = Framing{ITensors.Index}(axises)
    t_frames = Framing{ITensors.Index}(t_axises)
    return TransverseOps( IndListOperators(axisDims, localOps) ,frames,t_frames)
end


function TransverseOps(axises::Vector{<:ITensors.Index}, s::Vector{Symbol}, symmetries ::Vector{<:Integer}; tag::String="temp")::TransverseOps
    @assert length(axises)==length(s) "not compatible"
    t_axises = axises .|> (x -> ITensors.addtags(x,tag)) 
    axisDims = ITensors.dim.(axises) 
    localOps = Operator.(s) 
    frames = Framing{ITensors.Index}(axises)
    t_frames = Framing{ITensors.Index}(t_axises)
    return TransverseOps( SymListOperators(axisDims, localOps, symmetries) ,frames,t_frames)
end
    
function TransverseOps(axises::Vector{<:ITensors.Index}, s::Symbol; tag::String="temp",sym::Vector{<:Integer}=Int64[])::TransverseOps 
    if sym==[]
        return TransverseOps(axises, [s for a in axises]; tag=tag)
    else
        return TransverseOps(axises, [s for a in axises], sym; tag=tag)
    end
end;
    

function TransverseOps(T::ITensors.ITensor, s::Symbol; tag::String="temp",sym::Vector{<:Integer}=Int64[])::TransverseOps 
    axises = vcat(ITensors.inds(T)...)
    if sym==[]
        return TransverseOps(axises, [s for a in axises]; tag=tag)
    else
        return TransverseOps(axises, [s for a in axises],sym; tag=tag)
    end
end


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
    optype = (type== :invertible) ? :InvertableOp : :OrthogonalOp
    TOp = TransverseOps(T,optype)
    rT = changeBasisRandom(T,TOp)
    return (T=rT.T, Xs=rT.Xs)
end

function randomize_tensor(A::AbstractArray; type::Symbol=:invertible)
    T = __ITensor(A)
    return randomize_tensor(A; type=type)
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
