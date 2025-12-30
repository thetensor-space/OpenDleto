# module SylverLining

# export sylvesterLM #, derdenSMO, invderdenSMO

# using ITensors
import LinearMaps
using LinearMaps
using ITensors


# include("TransverseOperators.jl")
# using .TransverseOperators: TransverseOps, engaged, frame, transverse, contains



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
function sylvesterLM(Ω::TransverseOps, ch::Matrix, Γ::ITensor)::Tuple{LinearMap, LinearMap}
    # println("Constructing Sylvester LM for chisel of size ", size(ch), " and tensor of size ", size(Γ))
    eng = engaged(ch)
    # Get engaged columns of chisel and make tensor indices
    # engaged = [ a for a in 1:ndims(T) if ch[:,a] != 0 ] # all axes engaged
    ch_axis = Index(size(ch, 1), "chisel")
    Cs = [ ITensor(ch[:,a],  ch_axis) for a in 1:ndims(Γ) ]

    # K = eltype(Γ)
    Γ_frame = inds(Γ)
    Σ_frame = (ch_axis, inds(Γ)...)
    n_axes = length(Γ_frame)

    # Compute sizes for LinearMap
    # println("frame", Σ_frame)
    densor_dim = prod([ITensors.dim(f) for f in Σ_frame ])
    # println("densor dim: ", densor_dim, "Omega ", Ω)
    op_dim = Dleto.dim(Ω)
    # println("operator dim: ", op_dim)

    # Takes a vectorized representation of derivations
    # Returns a vectorized representation of the tensor
    function ester(Xvec)
        Xs = transverse(Ω, Xvec)
        Σ = ITensor(ch_axis, Γ_frame...)
        next = 1
        for a in 1:n_axes
            if !eng[a]
                continue
            end
            println(inds(Σ))
            p = plev(Γ_frame[a])
            X = Xs[next]
            X = addtags(inds(X)[2], "moved")  # X has (fr[a]', fr[a]), need (fr[a], fr[a]')
            C = Cs[a]
            Δ = C*X*Γ  # X has primed index, contracts with Γ, result has unprimed indices
            prime!(Δ, p, Γ_frame[a])  # Decrement prime level by 1 on axis a
            println(inds(Δ))
            Σ += permute(Δ, ch_axis, Γ_frame...)
            next += 1
        end
        return vec(Array(Σ, ch_axis, Γ_frame...))
    end
    
    dims = ITensors.dim.(Σ_frame)
    # sylv: takes a vectorized tensor, returns a vector of matrices (one per axis)
    function sylve(y)
        y_array = Array(reshape(y, dims)) ## slow step?  Dense.
        Σ = ITensor(y_array, ch_axis, Γ_frame...)
        n_engaged = count(eng)
        Z = Vector{ITensor}(undef, n_engaged)
        next = 1
        for a in 1:n_axes
            if !eng[a]
                continue
            end
            p = plev(Γ_frame[a])
            prime!(Σ, Γ_frame[a]) # raise prime to isolate axis
            C = Cs[a]
            Z[next] = C*Σ*Γ 
            next += 1
            setprime!(Σ,p, Γ_frame[a])  # Decrement prime level by 1 on axis a
        end
        return unsafe_member(Ω, Z)
    end

    # Compose sylv and ester as in sylvester4
    function sylvester(Xvec)
        # flattened vector reshaped as list of ITensors
        Xs = transverse(Ω, Xvec)
        # Create fresh Σ with unprimed indices
        Σ = ITensor(zeros(dims), ch_axis, Γ_frame...)
        next = 1
        for a in 1:n_axes
            if !eng[a]
                continue
            end
            p = plev(Γ_frame[a])
            X = Xs[next]
            C = Cs[a]
            Z = C*X*Γ  # X has (fr[a], fr[a]'), contracts with Γ on fr[a]
            setprime!(Z, p, Γ_frame[a])  # Decrement prime level by 1 on axis a
            Σ += permute(Z, ch_axis, Γ_frame...)
            next += 1
        end
        n_engaged = count(eng)
        Z = Vector{ITensor}(undef, n_engaged)
        next = 1
        for a in 1:n_axes
            if !eng[a]
                continue
            end
            p = plev(Γ_frame[a])
            prime!(Σ, Γ_frame[a]) # raise prime to isolate axis
            C = Cs[a]
            Z[next] = C*Σ*Γ
            setprime!(Σ,p, Γ_frame[a])  # Decrement prime level by 1 on axis a
            next += 1
        end
        return unsafe_member(Ω, Z)
    end

    # Wrap ester and sylve as LinearMaps
    densor_map = LinearMap(ester, sylve, densor_dim, op_dim; ismutating=false)
    derdensor_map = LinearMap(sylvester, sylvester, op_dim, op_dim; ismutating=false, issymmetric=true, isposdef=false)
    return derdensor_map, densor_map
end

# function inv_sylvesterLM(ch::Matrix, T::ITensor)
#     derdensor, densormap = sylvesterLM(ch, T)

#     function solver(b)
#         prob = LS.LinearProblem(derdensor, b)
#         sol = LS.solve(prob, LS.KrylovJL_CG()) 
#         return sol.u
#     end

#     inv_map = LinearMap(solver, size(densormap, 2), size(derdensor, 1); ismutating=false, issymmetric=true)
#     return inv_map
# end

# function sylvesterSMO(ch::Matrix, T::ITensor)
#     K = eltype(T)
#     is = collect(inds(T))
#     # println("Sylvester ITensor indices: ", is)
#     n_axes = length(is)
#     # Chisel columns as ITensor vectors
#     # chisels = [ ITensor(ch[:,i] for i in 1:size(ch,2)]

#     # Compute sizes for LinearMap
#     den_dim = prod(size(T)) * size(ch,1)
#     der_dim = sum(size(T, i)^2 for i in 1:n_axes)

#     # ester: takes a vector of matrices (one per axis), returns a vectorized tensor
#     function den_mul!(y, x, u, p, t)
#         Xs = transverse(is, x)
#         Y = ITensor(zeros(dims), is...)
#         for X in Xs
#             Z = X*T  # changes the index orders
#             Y += permute(noprime!(Z), inds(Y))
#         end
#         return y .= store(Y)
#     end
    
#     dims = size(T)
#     # sylv: takes a vectorized tensor, returns a vector of matrices (one per axis)
#     function der_mul!(y, x, u, p, t)
#         ## slow step?  Dense? TBD figure out.
#         Y = ITensor(Array(reshape(x, dims)), is...)
#         Zs = Vector{K}(undef,der_dim) # should use y directly as a view?
#         offset = 0
#         for i in 1:n_axes            
#             setprime!(Y, 1; plev=is[i]) # raise prime to isolate axis
#             # c = chisels[i]
#             # C = ITensor(c, Index(length(c), "c"))
#             Zi = Y*T
#             noprime!(Y) # revert axis for next round.
#             zi = store(Zi)
#             # Copy zi into Zs at the correct offset
#             l = dim(is[i])
#             Zs[(offset+1): (offset+l*l)] .= vec(zi)
#             offset += l*l
#         end
#         return y .= Zs # in future use directly 
#     end

#     # Compose sylv and ester as in sylvester4
#     function derden_mul!(y, x, u, p, t)
#         # flattened vector reshaped as list of ITensors
#         Xs = transverse(is, x)
#         Y = ITensor(zeros(dims), is...)
#         for X in Xs
#             Z = X*T  # changes the index orders with primes
#             Y += permute(noprime!(Z), inds(Y))
#         end
#         Zs = Vector{K}(undef,der_dim)
#         offset = 0
#         for i in 1:n_axes
#             # Y = ITensor(y_array, is...)
#             setprime!(Y, 1; plev=is[i]) # raise prime to isolate axis
#             # c = chisels[i]
#             # C = ITensor(c, Index(length(c), "c"))
#             Zi = Y*T
#             noprime!(Y) # revert axis for next round.
#             zi = store(Zi)
#             # Copy zi into Zs at the correct offset
#             l = dim(is[i])
#             Zs[(offset+1):(offset+l*l)] .= vec(zi)
#             offset += l*l
#         end
#         return y .= Zs # in future use directly 
#     end

#     # Wrap der, den, derden as SMO.FunctionOperators
    
#     # input prototypeAop = FunctionOperator(A_mul!, yproto, xproto; u=nothing, p=nothing, t=0.0)
#     densor_map = SMO.FunctionOperator(den_mul!, zeros(den_dim), zeros(der_dim) )
#     der_map = SMO.FunctionOperator(der_mul!, zeros(der_dim), zeros(den_dim) )
#     derdensor_map = SMO.FunctionOperator(derden_mul!, zeros(der_dim), zeros(der_dim))
#     return derdensor_map, der_map, densor_map
# end

# function invderdenSMO(ch::Matrix, T::ITensor)
#     derden_map, der_map, den_map = derdenSMO(ch, T)

#     function inv_derden_mul!(y, b, u, p, t)
#         prob = LS.LinearProblem(derden_map, b)
#         y .= LS.solve(prob, LS.KrylovJL_CG()) 
#         return y 
#     end

#     inv_map = SMO.FunctionOperator(inv_derden_mul!, zeros(size(derden_map, 2)), zeros(size(derden_map, 1)))
#     return inv_map
# end


# end # module SylverLining