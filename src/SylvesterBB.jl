module SylvesterBB


using LinearMaps
using TensorOperations
using Arpack
using IterativeSolvers
using CUDA
using cuTENSOR


export sylverlin3, sylverlin1, toMat, solveeig, solveit, cusylverlin3

#----------- Black-box Sylvester solvers --------------------------------------
function sylverlin3(s::AbstractArray, chisel::Matrix)
    # Make black-box linear map.
    d1 = size(s,1);
    d2 = size(s,2);
    d3 = size(s,3);
    dims = (size(chisel,1), size(s,1), size(s,2), size(s,3))
    function apply(x) 
        # without copying data, partition the vector and index it as square matrices
        A = reshape(view(x, 1:d1^2),          (d1, d1))
        B = reshape(view(x, (d1^2+1):(d1^2+d2^2)), (d2, d2))
        C = reshape(view(x, (d1^2+d2^2+1):(d1^2+d2^2+d3^2)), (d3, d3))
        # create an empty tensor of the required size
        # t = similar(s)
        t = zeros(eltype(s), dims)
        # call fast tensor contraction algorithms.
        @tensor t[c,i,j,k] += (chisel[:,1])[c] * A[i,i'] * s[i',j,k] + (chisel[:,2])[c] * B[j,j'] * s[i,j',k] + (chisel[:,3])[c] * C[k,k'] * s[i,j,k']
        return vec(t)
    end
    function apply_transpose(y)
        # Reshape vec into a tensor of the same shape as s
        t = reshape(y, dims)
        # Compute the contractions for the transpose
        # (You may need to flatten the result at the end)
        # Example using TensorOperations.jl:
        @tensor A′[ i,i′] := (chisel[:,1])[c]*t[c,i,j,k] * s[i′,j,k]
        @tensor B′[ j,j′] := (chisel[:,2])[c]*t[c,i,j,k] * s[i,j′,k]
        @tensor C′[ k,k′] := (chisel[:,3])[c]*t[c,i,j,k] * s[i,j,k′]
        # Flatten and concatenate A′, B′, C′ into a vector
        return vcat(vec(A′), vec(B′), vec(C′))
    end
    L = LinearMap( apply, apply_transpose, size(chisel,1)*size(s,1)*size(s,2)*size(s,3), size(s,1)^2 + size(s,2)^2 + size(s,3)^2; ismutating=false )
    return L
end


function cusylverlin3(s::CuArray, chisel::Matrix)
    # Make black-box linear map.
    d1 = size(s,1);
    d2 = size(s,2);
    d3 = size(s,3);
    dims = (size(chisel,1), size(s,1), size(s,2), size(s,3))
    function apply(x) 
        # without copying data, partition the vector and index it as square matrices
        A = reshape(view(x, 1:d1^2),          (d1, d1))
        B = reshape(view(x, (d1^2+1):(d1^2+d2^2)), (d2, d2))
        C = reshape(view(x, (d1^2+d2^2+1):(d1^2+d2^2+d3^2)), (d3, d3))
        # create an empty tensor of the required size
        # t = similar(s)
        t = zeros(eltype(s), dims)
        # call fast tensor contraction algorithms.
        @cutensor t[c,i,j,k] += (chisel[:,1])[c] * A[i,i'] * s[i',j,k] + (chisel[:,2])[c] * B[j,j'] * s[i,j',k] + (chisel[:,3])[c] * C[k,k'] * s[i,j,k']
        return vec(t)
    end
    function apply_transpose(y)
        # Reshape vec into a tensor of the same shape as s
        t = reshape(y, dims)
        # Compute the contractions for the transpose
        # (You may need to flatten the result at the end)
        # Example using TensorOperations.jl:
        @cutensor A′[ i,i′] := (chisel[:,1])[c]*t[c,i,j,k] * s[i′,j,k]
        @cutensor B′[ j,j′] := (chisel[:,2])[c]*t[c,i,j,k] * s[i,j′,k]
        @cutensor C′[ k,k′] := (chisel[:,3])[c]*t[c,i,j,k] * s[i,j,k′]
        # Flatten and concatenate A′, B′, C′ into a vector
        return vcat(vec(A′), vec(B′), vec(C′))
    end
    L = LinearMap( apply, apply_transpose, size(chisel,1)*size(s,1)*size(s,2)*size(s,3), size(s,1)^2 + size(s,2)^2 + size(s,3)^2; ismutating=false )
    return L
end

function sylverlin1(s::AbstractArray, chisel::Vector)
    d_a = size(s,1);
    dims = (length(chisel), size(s,1), size(s,2), size(s,3))
    function apply(x) 
        # without copying data, partition the vector and index it as square matrices
        A = reshape(view(x, 1:d_a^2), (d_a, d_a))
        # create an empty tensor of the required size
        # t = similar(s)        
        t = zeros(eltype(s), dims)
        # call fast tensor contraction algorithms.
        @tensor t[c,i,j,k] = chisel[c]*A[i,i'] * s[i',j,k]
        return vec(t)
    end
    function apply_transpose(y)
        # Reshape vec into a tensor of the same shape as s
        t = reshape(y, dims)
        # Compute the contractions for the transpose
        # (You may need to flatten the result at the end)
        # Example using TensorOperations.jl:
        @tensor A′[c,i,i′] := chisel[c]*t[i,j,k] * s[i′,j,k]
        # Flatten and concatenate A′, B′, C′ into a vector
        return vec(A′)
    end
    L = LinearMap( apply, apply_transpose, size(chisel,1)*size(s,1)*size(s,2)*size(s,3), size(s,1)^2; ismutating=false )
    return L
end

function toMat(L)
    nrows, ncols = size(L)
    M = zeros(Float64, nrows, ncols)
    for j in 1:ncols
        e_j = zeros(Float64, ncols)
        e_j[j] = 1.0
        M[:, j] = L * e_j
    end
    return M
end

function solveeig(t)
    L = sylverlin(t)
    vals, vecs, info = svdl(L) #; nsv=10)
    # vals, vecs, info = eigs(L*L'; nev=10, which=:SM)
    if info != 0
        println("Arpack did not converge, info = $info")
        println("Falling back to IterativeSolvers.svdl")
        vals, vecs = svdl(L; nsv=10, which=:SR)
    end
    offset1 = size(t,1)^2
    offset2 = offset1 + size(t,2)^2
    A = reshape(view(x,                     1:offset1),           (size(t,1), size(t,1)))
    B = reshape(view(x,       (offset1+1):offset2), (size(t,2), sizein(t,2)))
    C = reshape(view(x, (offset2+1):length(x)), (size(t,3), size(t,3)))
    return A, B, C
end

function solveit(t)
    L = sylverlin(t)
    u, s, v = svdl(L; nsv=10, which=:SM)
    offset1 = size(t,1)^2
    offset2 = offset1 + size(t,2)^2
    A = reshape(view(v[:,end],                     1:offset1),           (size(t,1), size(t,1)))
    B = reshape(view(v[:,end],       (offset1+1):offset2), (size(t,2), size(t,2)))      
    C = reshape(view(v[:,end], (offset2+1):length(v[:,end])), (size(t,3), size(t,3)))
    return A, B, C
end

end # module SylvesterBB