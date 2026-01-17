struct SVDSolver <: NullSolver end
    
function solve(::SVDSolver, L::LinearMaps.LinearMap; nd::Integer = 10, tol::Float64 = 1e-8, kwargs...)
    # println("Using SVDSolver...")
    # Use LinearAlgebra to compute the null space of L.
    # println("Converting LinearMap to Matrix for SVD...")
    M = Matrix(L)
    svds = LinearAlgebra.svd(M;full=true, kwargs...)
    # @show svds.S
    nvals = sum( svds.S .|> x -> (abs(x) < tol) ) 
    if nd >= 0
        nvals =min(nd,nvals)
    end 
    return (;vals=svds.S[end:-1:(end-nvals+1)], vecs=svds.V[:, end:-1:(end-nvals+1)])
end

NullSolversDict[:SVDSolver] = SVDSolver();
NullSolversDict[:default ] = SVDSolver();

struct EigenSolver <: NullSolver end;
#assumes that the matrix is symmetric    
function solve(::EigenSolver, L::LinearMaps.LinearMap; nd::Integer = 10, tol::Float64 = 1e-8, kwargs...)
    M = Matrix(L)
    eigens = LinearAlgebra.eigen( LinearAlgebra.Symmetric(M) )
    small = [ eigens.values[i] < tol for i = 1:size(M,1)]
    vals=eigens.values[small] 
    vecs=eigens.vectors[:,small]
    if (length(vals)) < nd || (nd < 0)
        return (; vals=vals, vecs=vecs )
    else
        return (; vals=vals[1:nd], vecs=vecs[:,1:nd] )
    end
end

NullSolversDict[:EigenSolver] = EigenSolver();


## LU solver might not be wokring!!!

struct LUSolver <: NullSolver end


function solve_extra(::LUSolver, L::LinearMaps.LinearMap; nd::Integer = 10, tol = 1e-8, kwargs...)
#function lusolve(L::AbstractMatrix; nd::Integer = 10, tol = 1e-8, kwargs...)
    # println("Using LUSolver on Matrix...")
    M = Matrix(L)
    # println("Performing LU Factorization...")
    F = LinearAlgebra.lu(M;kwargs...)
    U = F.U
    # L = F.L
    # p = F.p  # permutation

    # Determine rank and free variables
    rank = sum(abs.(diag(U)) .> tol)
    # dimkernel = sum(abs.(diag(U)) .< tol)
    n = size(M, 2)
    output = min(nd, n-rank)
    # free_vars = rank+1:n
    free_vars = (n-output+1):n
# println("Matrix rank: $rank, free variables: ", free_vars, " with tolerance $tol")
    # Build null space basis
    vecs = zeros(n,output)
    vals = zeros(output)
    for i = 1:output
        v = zeros(output)
        v[i] = 1.0
        # Solve U * x = -U[:, free_vars]*v_free for pivot variables
        rhs = -U[1:rank, free_vars] * v
        x = U[1:rank, 1:rank] \ rhs
        vecs[1:rank,i] = x
        vecs[n-output + i,i] = 1.0
        vals[i] = U[n-output+i,n-output+i]
    end
    return (;vals = vals, vecs=vecs) 
end;


NullSolversDict[:LUSolver] = LUSolver();
