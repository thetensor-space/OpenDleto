module SylverLining

export sylvesterLM #, derdenSMO, invderdenSMO

using ITensors
import LinearMaps as LM
using .TransverseOperators: TransverseOps, engaged, transverse, contains

# import LinearSolve as LS
# import SciMLOperators as SMO

# # Reshape vector into vector of ITensor matrices.
# function transverse(axes, engaged, data)
#     transverse_mats = Vector{ITensor}(undef, length(engaged))
#     offset = 0; 
#     for e in engaged
#         l = dim(axes[e])
#         X = Array(reshape(view(data, (offset+1):(offset + l*l)), (l, l)))
#         transverse_mats[e] = ITensor( X, axes[e]', axes[e])
#         offset += l*l
#     end
#     return transverse_mats
# end

"""
    sylvesterLM(ch::Matrix, T::ITensor)

    Constructs LinearMaps for the derivation and densor maps associated to the given chisel `ch` and tensor `T`.

    - `ch`: A matrix whose columns define the chisel polynomials.
    - `T`: The input tensor.

    Returns a tuple `(derdensor_map, densormap)` where:
    - `derdensor_map`: the composed derivation-densor `LinearMap` (a real symmetric operator).
    - `densormap`: the densor operator with transpose---the derivation operator---included.
"""
function sylvesterLM(Ω::TransverseOps, ch::Matrix, T::ITensor)::Tuple{LM.LinearMap, LM.LinearMap}
    eng = engaged(Ω)
    # Get engaged columns of chisel and make tensor indices
    # engaged = [ a for a in 1:ndims(T) if ch[:,a] != 0 ] # all axes engaged
    Cs = [ ITensor(ch[:,a],  "Chisel$a") for a in 1:ndims(T) ]

    K = eltype(T)
    in_axes = collect(inds(T))
    chAxis = Index(size(ch, 1), "chisel")
    out_axes = vcat(chAxis, collect(inds(T)))
    n_axes = length(in_axes)

    # Compute sizes for LinearMap
    densor_dim = prod(size(T)) * size(ch,1)
    der_dim = sum(size(T, i)^2 for i in 1:n_axes)

    # Takes a vectorized representation of derivations
    # Returns a vectorized representation of the tensor
    function ester(Xvec)
        Xs = transverse(ops, in_axes, Xvec)
        Y = ITensor(out_axes...)
        for a in engaged
            X = Xs[a]
            C = Cs[a]
            setprime!(Y, 1; plev=in_axes[a]) # raise prime to isolate axis
            Z = C*X*T  # changes the index orders
            Y += permute(noprime!(Z), inds(Y))
        end
        return store(Y)
    end
    
    dims = size(T)
    # sylv: takes a vectorized tensor, returns a vector of matrices (one per axis)
    function sylve(y)
        y_array = Array(reshape(y, dims)) ## slow step?  Dense.
        Zs = Vector{K}(undef,der_dim)
        offset = 0
        for i in 1:n_axes
            Y = ITensor(y_array, is...)
            setprime!(Y, 1; plev=is[i]) # raise prime to isolate axis
            C = chisel_cols[i]
            Zi = C*Y*T
            noprime!(Y) # revert axis for next round.
            zi = store(Zi)
            # Copy zi into Zs at the correct offset
            l = dim(is[i])
            Zs[(offset+1): (offset+l*l)] .= vec(zi)
            offset += l*l
        end
        return Zs
    end

    # Compose sylv and ester as in sylvester4
    function sylvester(Xvec)
        # flattened vector reshaped as list of ITensors
        Xs = transverse(is, Xvec)
        Y = ITensor(zeros(dims), is...)
        for X in Xs
            Z = X*T  # changes the index orders with primes
            Y += permute(noprime!(Z), inds(Y))
        end
        Zs = Vector{K}(undef,der_dim)
        offset = 0
        for a in engaged
            # Y = ITensor(y_array, is...)
            setprime!(Y, 1; plev=is[i]) # raise prime to isolate axis
            # c = chisels[i]
            # C = ITensor(c, Index(length(c), "c"))
            Zi = Y*T
            noprime!(Y) # revert axis for next round.
            zi = store(Zi)
            # Copy zi into Zs at the correct offset
            l = dim(is[i])
            Zs[(offset+1):(offset+l*l)] .= vec(zi)
            offset += l*l
        end
        return Zs
    end

    # Wrap ester and sylve as LinearMaps
    densor_map = LinearMap(ester, sylve, densor_dim, der_dim; ismutating=false)
    derdensor_map = LinearMap(sylvester, sylvester, der_dim, der_dim; ismutating=false, issymmetric=true, isposdef=false)
    return derdensor_map, densor_map
end

function inv_sylvesterLM(ch::Matrix, T::ITensor)
    derdensor, densormap = sylvesterLM(ch, T)

    function solver(b)
        prob = LS.LinearProblem(derdensor, b)
        sol = LS.solve(prob, LS.KrylovJL_CG()) 
        return sol.u
    end

    inv_map = LinearMap(solver, size(densormap, 2), size(derdensor, 1); ismutating=false, issymmetric=true)
    return inv_map
end

function sylvesterSMO(ch::Matrix, T::ITensor)
    K = eltype(T)
    is = collect(inds(T))
    # println("Sylvester ITensor indices: ", is)
    n_axes = length(is)
    # Chisel columns as ITensor vectors
    # chisels = [ ITensor(ch[:,i] for i in 1:size(ch,2)]

    # Compute sizes for LinearMap
    den_dim = prod(size(T)) * size(ch,1)
    der_dim = sum(size(T, i)^2 for i in 1:n_axes)

    # ester: takes a vector of matrices (one per axis), returns a vectorized tensor
    function den_mul!(y, x, u, p, t)
        Xs = transverse(is, x)
        Y = ITensor(zeros(dims), is...)
        for X in Xs
            Z = X*T  # changes the index orders
            Y += permute(noprime!(Z), inds(Y))
        end
        return y .= store(Y)
    end
    
    dims = size(T)
    # sylv: takes a vectorized tensor, returns a vector of matrices (one per axis)
    function der_mul!(y, x, u, p, t)
        ## slow step?  Dense? TBD figure out.
        Y = ITensor(Array(reshape(x, dims)), is...)
        Zs = Vector{K}(undef,der_dim) # should use y directly as a view?
        offset = 0
        for i in 1:n_axes            
            setprime!(Y, 1; plev=is[i]) # raise prime to isolate axis
            # c = chisels[i]
            # C = ITensor(c, Index(length(c), "c"))
            Zi = Y*T
            noprime!(Y) # revert axis for next round.
            zi = store(Zi)
            # Copy zi into Zs at the correct offset
            l = dim(is[i])
            Zs[(offset+1): (offset+l*l)] .= vec(zi)
            offset += l*l
        end
        return y .= Zs # in future use directly 
    end

    # Compose sylv and ester as in sylvester4
    function derden_mul!(y, x, u, p, t)
        # flattened vector reshaped as list of ITensors
        Xs = transverse(is, x)
        Y = ITensor(zeros(dims), is...)
        for X in Xs
            Z = X*T  # changes the index orders with primes
            Y += permute(noprime!(Z), inds(Y))
        end
        Zs = Vector{K}(undef,der_dim)
        offset = 0
        for i in 1:n_axes
            # Y = ITensor(y_array, is...)
            setprime!(Y, 1; plev=is[i]) # raise prime to isolate axis
            # c = chisels[i]
            # C = ITensor(c, Index(length(c), "c"))
            Zi = Y*T
            noprime!(Y) # revert axis for next round.
            zi = store(Zi)
            # Copy zi into Zs at the correct offset
            l = dim(is[i])
            Zs[(offset+1):(offset+l*l)] .= vec(zi)
            offset += l*l
        end
        return y .= Zs # in future use directly 
    end

    # Wrap der, den, derden as SMO.FunctionOperators
    
    # input prototypeAop = FunctionOperator(A_mul!, yproto, xproto; u=nothing, p=nothing, t=0.0)
    densor_map = SMO.FunctionOperator(den_mul!, zeros(den_dim), zeros(der_dim) )
    der_map = SMO.FunctionOperator(der_mul!, zeros(der_dim), zeros(den_dim) )
    derdensor_map = SMO.FunctionOperator(derden_mul!, zeros(der_dim), zeros(der_dim))
    return derdensor_map, der_map, densor_map
end

function invderdenSMO(ch::Matrix, T::ITensor)
    derden_map, der_map, den_map = derdenSMO(ch, T)

    function inv_derden_mul!(y, b, u, p, t)
        prob = LS.LinearProblem(derden_map, b)
        y .= LS.solve(prob, LS.KrylovJL_CG()) 
        return y 
    end

    inv_map = SMO.FunctionOperator(inv_derden_mul!, zeros(size(derden_map, 2)), zeros(size(derden_map, 1)))
    return inv_map
end


end # module SylverLining